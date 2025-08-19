
# Automated Essay Scoring 2.0 — DeBERTa v3 Baseline (GPU-ready)

This repository is a **GitHub-ready project** derived from a notebook for the Kaggle competition **Learning Agency Lab — Automated Essay Scoring 2.0**. It trains a DeBERTa v3 model to predict essay scores (1–6) and outputs a `submission.csv`.

- Competition: https://www.kaggle.com/competitions/learning-agency-lab-automated-essay-scoring-2
- Metric: **Quadratic Weighted Kappa (QWK)** (submissions are evaluated against private labels).
- Labels: integer scores **1–6**.
- GPU: Uses PyTorch with **automatic mixed precision** when a CUDA GPU is available.

## Project layout

```
.
├─ configs/
│  └─ default.yaml
├─ notebooks/
│  └─ deberta-v3-base.ipynb
├─ scripts/
│  ├─ download_data.py        # Pulls competition data via Kaggle API
│  └─ download_model.py       # Downloads Hugging Face model to ./models/
├─ src/
│  ├─ __init__.py
│  ├─ dataset.py              # Dataset + column auto-detection
│  ├─ modeling.py             # DeBERTa v3 (regression head)
│  ├─ qwk.py                  # Quadratic Weighted Kappa + threshold search
│  ├─ train.py                # K-fold training
│  └─ infer.py                # Inference + submission
├─ .github/workflows/ci.yml   # Basic CI smoke (optional)
├─ requirements.txt
├─ LICENSE
└─ README.md
```

## Prerequisites

- **Kaggle API credentials** available as environment variables:
  - `KAGGLE_USERNAME`
  - `KAGGLE_KEY`
- Python 3.10+ recommended, CUDA toolkit / NVIDIA driver for GPU.

## Quickstart (local)

```bash
# 1) Create venv and install deps
python -m venv .venv
# Windows:
.\.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate
pip install -r requirements.txt

# 2) Authenticate env for Kaggle (or rely on ~/.kaggle/kaggle.json)
# PowerShell (Windows):
$env:KAGGLE_USERNAME="your_username"
$env:KAGGLE_KEY="your_key"
# bash:
# export KAGGLE_USERNAME=your_username
# export KAGGLE_KEY=your_key

# 3) Download competition data and model
python scripts/download_data.py
python scripts/download_model.py

# 4) Train (GPU auto-detected)
python -m src.train --config configs/default.yaml

# 5) Inference + submission.csv
python -m src.infer --config configs/default.yaml
```

Artifacts will land under `outputs/`. Models are saved under `models/` (one folder per fold).

## Using GitHub Actions + Secrets

Add repository **Secrets** for `KAGGLE_USERNAME` and `KAGGLE_KEY`. The included CI workflow demonstrates a smoke run that downloads the dataset and imports packages. Uncomment training steps if you want to run a tiny debug epoch on CI.

## Data & Model

- Data are pulled **directly from Kaggle** using the API; nothing in `data/` is tracked by Git.
- The code downloads **`microsoft/deberta-v3-base`** locally to `./models/microsoft-deberta-v3-base` for repeatable runs.

## Notes

- The code **auto-detects** the text and target columns from `train.csv`. If detection fails, override via CLI:
  ```bash
  python -m src.train --config configs/default.yaml --text-col full_text --target-col score
  ```
- QWK-based **threshold optimization** is included for validation. Inference uses simple rounding by default.

## Push to GitHub

```bash
git init && git add .
git commit -m "AES2: DeBERTa v3 baseline project"
git branch -M main
git remote add origin <YOUR_GITHUB_REPO_URL>
git push -u origin main
```
