import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "outputs" / "cv_results.json"
IMG_DIR = ROOT / "reports" / "images"

results = json.load(open(RESULTS))

folds = [r["fold"] for r in results]
vals = [r["val_qwk"] for r in results]

plt.figure(figsize=(6, 4))
bars = plt.bar([f"fold {f}" for f in folds], vals, color="#3b6fa0")
plt.ylabel("Validation QWK")
plt.title("DeBERTa-v3-small fine-tune — per-fold validation QWK (real run)")
for b, v in zip(bars, vals):
    plt.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.4f}", ha="center", fontsize=9)
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig(IMG_DIR / "finetune_fold_qwk.png", dpi=130)
plt.close()

# Comparison vs the original LGBM ensemble's real logged result
plt.figure(figsize=(6, 4))
mean_finetune = sum(vals) / len(vals)
labels = ["This repo:\nDeBERTa-small fine-tune\n(mean of {} folds)".format(len(vals)),
          "Original submission:\nDeBERTa emb. + 15-fold LGBM\n(mean CV, from notebook output)"]
values = [mean_finetune, 0.8398]
plt.bar(labels, values, color=["#3b6fa0", "#5a9367"])
plt.ylabel("QWK")
plt.title("This repo's reproduction vs. the original medal submission")
for i, v in enumerate(values):
    plt.text(i, v + 0.01, f"{v:.4f}", ha="center", fontsize=9)
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig(IMG_DIR / "finetune_vs_original.png", dpi=130)
plt.close()

print("mean val QWK:", mean_finetune)
print("saved charts to", IMG_DIR)
