from __future__ import annotations
import argparse, json, yaml
import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from .dataset import EssayDataset, detect_columns
from .modeling import DebertaRegressor
from .qwk import quadratic_weighted_kappa, optimize_thresholds, apply_thresholds
from transformers import AutoTokenizer


def collate_batch(batch):
    # Each item in `batch` is a dict from EssayDataset.__getitem__
    input_ids = [item["input_ids"] for item in batch]
    attention_mask = [item["attention_mask"] for item in batch]

    input_ids = torch.tensor(input_ids, dtype=torch.long)
    attention_mask = torch.tensor(attention_mask, dtype=torch.long)

    out = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }

    if "labels" in batch[0]:
        labels = [item["labels"] for item in batch]
        out["labels"] = torch.tensor(labels, dtype=torch.float32)  # shape [B]
    return out


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--k-folds", type=int, default=None)
    ap.add_argument("--debug", type=int, default=0)
    ap.add_argument("--text-col", type=str, default=None)
    ap.add_argument("--target-col", type=str, default=None)
    return ap.parse_args()


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def seed_everything(seed=42):
    import random, numpy as np, torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_fold(cfg, fold, df_train, df_valid, text_col, target_col, device):
    model_dir_or_name = cfg.get("model_dir") or cfg["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir_or_name,
        use_fast=False,  # 👈 important
        trust_remote_code=False,  # not needed for DeBERTa
    )
    train_ds = EssayDataset(
        df_train,
        tokenizer,
        text_col=text_col,
        target_col=target_col,
        max_length=cfg["max_length"],
    )
    valid_ds = EssayDataset(
        df_valid,
        tokenizer,
        text_col=text_col,
        target_col=target_col,
        max_length=cfg["max_length"],
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["train_batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        collate_fn=collate_batch,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=cfg["valid_batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        collate_fn=collate_batch,
    )

    model_name_or_dir = cfg.get("model_dir") or cfg["model_name"]
    model = DebertaRegressor(model_name_or_dir, num_labels=cfg["num_labels"]).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"])
    )
    num_steps = (
        cfg["epochs"] * len(train_loader) // max(1, int(cfg["gradient_accumulation"]))
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * num_steps), num_training_steps=num_steps
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=bool(cfg["fp16"]) and torch.cuda.is_available()
    )
    mse = nn.MSELoss()

    best_val = -1.0
    best_path = Path(cfg["output_dir"]) / f"fold{fold}" / "model.pt"
    best_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(cfg["epochs"]):
        model.train()
        running = 0.0
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch.get("labels")
            if labels is not None:
                labels = labels.to(device)

            with torch.amp.autocast(
                "cuda", enabled=bool(cfg["fp16"]) and torch.cuda.is_available()
            ):
                out = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = out["logits"].squeeze(-1)
                loss = mse(logits, labels)

            scaler.scale(loss).backward()
            if (step + 1) % cfg["gradient_accumulation"] == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
            running += loss.item()

        # Validation
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for batch in valid_loader:
                input_ids = torch.tensor(batch["input_ids"]).to(device)
                attention_mask = torch.tensor(batch["attention_mask"]).to(device)
                labels = torch.tensor(batch["labels"]).float().to(device)
                out = model(input_ids=input_ids, attention_mask=attention_mask)
                preds.extend(out["logits"].squeeze(-1).detach().cpu().numpy().tolist())
                trues.extend(labels.detach().cpu().numpy().tolist())
        # Map to 1..6
        if cfg.get("rounding", "nearest") == "optimized":
            th, val_qwk = optimize_thresholds(
                preds, trues, num_classes=cfg["score_max"]
            )
            pred_ints = apply_thresholds(preds, th)
        else:
            pred_ints = np.rint(
                np.clip(preds, cfg["score_min"], cfg["score_max"])
            ).astype(int)
            val_qwk = quadratic_weighted_kappa(
                np.array(trues).astype(int),
                pred_ints,
                cfg["score_min"],
                cfg["score_max"],
            )

        print(
            f"[fold {fold}] epoch {epoch + 1}/{cfg['epochs']} loss={running / len(train_loader):.4f} val_QWK={val_qwk:.4f}"
        )
        if val_qwk > best_val:
            best_val = val_qwk
            torch.save({"model_state": model.state_dict()}, best_path)

    return best_path, best_val


def main():
    args = parse_args()
    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.k_folds is not None:
        cfg["k_folds"] = args.k_folds

    seed_everything(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    data_dir = Path(cfg["data_dir"])
    train_csv = data_dir / "train.csv"
    if not train_csv.exists():
        raise FileNotFoundError(
            f"{train_csv} not found. Run: python scripts/download_data.py"
        )

    df = pd.read_csv(train_csv)

    # Column detection
    text_candidates = cfg.get("text_columns") or [
        "full_text",
        "text",
        "essay",
        "essay_text",
        "content",
    ]
    target_candidates = cfg.get("target_columns") or ["score", "target", "y"]
    id_candidates = cfg.get("id_columns") or ["essay_id", "id"]
    text_col, target_col, id_col = detect_columns(
        df, text_candidates, target_candidates, id_candidates
    )

    if args.text_col:
        text_col = args.text_col
    if args.target_col:
        target_col = args.target_col

    if text_col is None or target_col is None:
        raise ValueError(
            f"Could not detect text/target columns. Found: {df.columns.tolist()}"
        )

    # Debug subset
    if args.debug and args.debug > 0:
        df = df.sample(n=min(len(df), 1000), random_state=cfg["seed"]).reset_index(
            drop=True
        )

    # K-fold split stratified by label
    from sklearn.model_selection import StratifiedKFold

    k = cfg["k_folds"]
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=cfg["seed"])
    results = []
    for fold, (tr_idx, va_idx) in enumerate(skf.split(df, df[target_col].astype(int))):
        df_tr = df.iloc[tr_idx].reset_index(drop=True)
        df_va = df.iloc[va_idx].reset_index(drop=True)
        path, val = train_one_fold(
            cfg, fold, df_tr, df_va, text_col, target_col, device
        )
        results.append((fold, val, str(path)))

    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "cv_results.json", "w", encoding="utf-8") as f:
        json.dump(
            [{"fold": f, "val_qwk": v, "path": p} for f, v, p in results], f, indent=2
        )


if __name__ == "__main__":
    main()
