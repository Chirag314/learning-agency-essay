import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IMG_DIR = ROOT / "reports" / "images"

# Parsed directly from artifacts_train_log.txt (real run, 2026-08-13)
data = {
    0: {"loss": [0.8774, 0.3232], "qwk": [0.8002, 0.8138]},
    1: {"loss": [1.1018, 0.3388], "qwk": [0.7944, 0.8100]},
    2: {"loss": [0.9691, 0.3249], "qwk": [0.8072, 0.8118]},
}
colors = {0: "#3b6fa0", 1: "#5a9367", 2: "#c17a3d"}

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for fold, d in data.items():
    epochs = range(1, len(d["loss"]) + 1)
    ax[0].plot(epochs, d["loss"], marker="o", label=f"fold {fold}", color=colors[fold])
    ax[1].plot(epochs, d["qwk"], marker="o", label=f"fold {fold}", color=colors[fold])
ax[0].set_title("Train loss (MSE)")
ax[0].set_xlabel("Epoch")
ax[0].legend()
ax[1].set_title("Validation QWK")
ax[1].set_xlabel("Epoch")
ax[1].legend()
plt.tight_layout()
plt.savefig(IMG_DIR / "finetune_training_curves.png", dpi=130)
print("saved")
