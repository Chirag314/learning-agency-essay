from __future__ import annotations
import os, argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from .dataset import EssayDataset, detect_columns
from .modeling import DebertaRegressor


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--text-col", type=str, default=None)
    ap.add_argument("--out", default="outputs/submission.csv")
    return ap.parse_args()


def load_cfg(path):
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_best_models(output_dir: str, k_folds: int):
    paths = []
    for fold in range(k_folds):
        p = Path(output_dir) / f"fold{fold}" / "model.pt"
        if not p.exists():
            raise FileNotFoundError(f"Missing model for fold {fold}: {p}")
        paths.append(str(p))
    return paths


def main():
    args = parse_args()
    cfg = load_cfg(args.config)
    data_dir = Path(cfg["data_dir"])
    test_csv = data_dir / "test.csv"
    if not test_csv.exists():
        raise FileNotFoundError(
            f"{test_csv} not found. Run: python scripts/download_data.py"
        )

    df_test = pd.read_csv(test_csv)
    text_candidates = cfg.get("text_columns") or [
        "full_text",
        "text",
        "essay",
        "essay_text",
        "content",
    ]
    id_candidates = cfg.get("id_columns") or ["essay_id", "id"]
    text_col, _, id_col = detect_columns(df_test, text_candidates, [], id_candidates)
    if args.text_col:
        text_col = args.text_col
    if text_col is None:
        raise ValueError(
            f"Could not detect text column in test.csv; columns: {df_test.columns.tolist()}"
        )
    if id_col is None:
        df_test["id"] = np.arange(len(df_test))
        id_col = "id"

    model_dir_or_name = cfg.get("model_dir") or cfg["model_name"]
    tok = AutoTokenizer.from_pretrained(
        model_dir_or_name, use_fast=False, trust_remote_code=False
    )
    ds = EssayDataset(
        df_test, tok, text_col=text_col, target_col=None, max_length=cfg["max_length"]
    )
    dl = DataLoader(
        ds,
        batch_size=cfg["valid_batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fold_paths = load_best_models(cfg["output_dir"], cfg["k_folds"])

    preds_folds = []
    for p in fold_paths:
        model = DebertaRegressor(model_dir_or_name, num_labels=cfg["num_labels"])
        state = torch.load(p, map_location="cpu")
        model.load_state_dict(state["model_state"])
        model.to(device).eval()

        preds = []
        with torch.no_grad():
            for batch in dl:
                input_ids = torch.tensor(batch["input_ids"]).to(device)
                attention_mask = torch.tensor(batch["attention_mask"]).to(device)
                out = model(input_ids=input_ids, attention_mask=attention_mask)
                preds.extend(out["logits"].squeeze(-1).detach().cpu().numpy().tolist())
        preds_folds.append(np.array(preds))

    preds = np.mean(preds_folds, axis=0)
    preds_int = np.rint(np.clip(preds, cfg["score_min"], cfg["score_max"])).astype(int)

    sub = pd.DataFrame({id_col: df_test[id_col], "score": preds_int})
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    sub.to_csv(args.out, index=False)
    print(f"[infer] wrote {args.out}")


if __name__ == "__main__":
    main()
