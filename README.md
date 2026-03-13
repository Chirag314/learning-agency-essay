<div align="center">

# 📝 Learning Agency Lab — Automated Essay Scoring 2.0

**A production-style NLP pipeline for automated essay scoring using DeBERTa v3, K-Fold training, and Quadratic Weighted Kappa optimization**

[![Kaggle Competition](https://img.shields.io/badge/Kaggle-AES2-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/competitions/learning-agency-lab-automated-essay-scoring-2)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](#tech-stack)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](#tech-stack)
[![Transformers](https://img.shields.io/badge/Hugging%20Face-Transformers-FFD21E?logo=huggingface&logoColor=black)](#tech-stack)
[![Model](https://img.shields.io/badge/Backbone-DeBERTa%20v3-6f42c1)](#modeling-approach)
[![License](https://img.shields.io/badge/License-MIT-green)](./LICENSE)

[Competition](https://www.kaggle.com/competitions/learning-agency-lab-automated-essay-scoring-2) •
[Data](https://www.kaggle.com/competitions/learning-agency-lab-automated-essay-scoring-2/data) •
[Evaluation](https://www.kaggle.com/competitions/learning-agency-lab-automated-essay-scoring-2/overview/evaluation) •
[Repository](https://github.com/Chirag314/learning-agency-essay)

</div>

---

## Overview

This repository contains a clean, reproducible Kaggle solution for **Learning Agency Lab — Automated Essay Scoring 2.0**, a competition focused on predicting essay quality directly from student writing.

The current implementation centers on a **DeBERTa v3-based regression pipeline** with:

- fold-based training
- validation-time **Quadratic Weighted Kappa (QWK)** optimization
- threshold tuning from continuous predictions to discrete score buckets
- Kaggle-ready inference that generates `submission.csv`

This project is structured more like a real ML codebase than a one-off notebook, making it useful for both competition iteration and portfolio presentation.

---

## Why this project is interesting

Automated essay scoring is a strong real-world ML problem because it combines:

- **long-form NLP understanding**
- **ordinal prediction**, not simple classification
- **metric-aware modeling**, where QWK matters more than generic loss alone
- **reproducible engineering**, from data download to inference

For an ML portfolio, this repo shows the ability to move from a Kaggle task to a maintainable project with scripts, configs, reusable modules, and a cleaner experimentation loop.

---

## Competition snapshot

| Item | Details |
|---|---|
| Competition | Learning Agency Lab — Automated Essay Scoring 2.0 |
| Problem type | Ordinal score prediction from essay text |
| Input | Student essay text |
| Target | Essay score |
| Score range | `1` to `6` |
| Evaluation metric | **Quadratic Weighted Kappa (QWK)** |
| Submission format | `essay_id,score` |

### What the metric means

Unlike plain accuracy, **QWK** gives partial credit when a prediction is close to the true score and penalizes larger errors more strongly. That makes it a natural fit for essay scoring, where predicting `5` instead of `6` is far less harmful than predicting `1` instead of `6`.

---

## Repository highlights

### Modeling
- **Backbone:** `microsoft/deberta-v3-base`
- Transformer encoder with a **regression-style head**
- Continuous score prediction followed by bucket mapping for final submission

### Training
- **K-Fold cross-validation**
- Auto-detection of essay and target columns from `train.csv`
- Validation-time threshold search for stronger QWK
- GPU-aware training with AMP support where available

### Inference
- Fold-aware checkpoint loading
- Competition-ready `submission.csv` generation
- Easy path for local machine, Codespaces, or cloud notebook execution

### Engineering
- Scripted Kaggle dataset download
- Scripted Hugging Face model snapshot download
- Config-driven workflow
- GitHub Actions smoke CI scaffold

---

## Project structure

```text
.
├─ .github/
│  └─ workflows/
│     └─ ci.yml                 # Smoke CI workflow
├─ configs/
│  └─ default.yaml              # Training / inference configuration
├─ notebooks/
│  └─ deberta-v3-base.ipynb     # Notebook experimentation
├─ scripts/
│  ├─ download_data.py          # Pull competition data via Kaggle API
│  └─ download_model.py         # Download HF model locally
├─ src/
│  ├─ __init__.py
│  ├─ dataset.py                # Dataset loading + column detection
│  ├─ modeling.py               # DeBERTa v3 model definition
│  ├─ qwk.py                    # QWK + threshold optimization
│  ├─ train.py                  # K-fold training entrypoint
│  └─ infer.py                  # Inference + submission creation
├─ requirements.txt
├─ LICENSE
└─ README.md
```

---

## End-to-end workflow

```text
Kaggle Data
    ↓
Pretrained DeBERTa v3
    ↓
Text Tokenization + Dataset Pipeline
    ↓
K-Fold Training
    ↓
Out-of-Fold Predictions
    ↓
QWK Threshold Optimization
    ↓
Fold Ensemble / Inference
    ↓
submission.csv
```

---

## Tech stack

- **Python**
- **PyTorch**
- **Transformers / Hugging Face**
- **scikit-learn**
- **NumPy / Pandas**
- **Kaggle API**

---

## Quickstart

### 1) Create environment

```bash
python -m venv .venv
source .venv/bin/activate   # macOS / Linux
# .venv\Scripts\activate   # Windows PowerShell

pip install -r requirements.txt
```

### 2) Configure Kaggle authentication

Use either environment variables:

```bash
export KAGGLE_USERNAME="your_username"
export KAGGLE_KEY="your_key"
```

or place `kaggle.json` under:

```bash
~/.kaggle/kaggle.json
```

### 3) Download competition data and model

```bash
python scripts/download_data.py
python scripts/download_model.py
```

### 4) Train

```bash
python -m src.train --config configs/default.yaml
```

### 5) Run inference

```bash
python -m src.infer --config configs/default.yaml
```

---

## Modeling approach

This repo treats essay scoring as a **continuous prediction problem** and then aligns predictions with the competition’s discrete score space.

That setup has a few practical advantages:

- smoother optimization during training
- flexibility for post-processing
- easier threshold calibration for QWK

The critical idea is that strong validation loss alone is not enough. Because the leaderboard metric is **QWK**, the pipeline includes threshold optimization to better convert raw model outputs into final score buckets.

---

## What this repo demonstrates

This project highlights several skills that map well to strong ML engineering work:

- turning a competition problem into a reusable codebase
- organizing experiments with scripts and configuration
- working with transformer models for long-form text tasks
- optimizing toward a business-style metric rather than only training loss
- building a repo that is easier to maintain, reproduce, and extend

---

## Results section template

You can update this section with your actual CV and leaderboard results.

```text
Backbone: DeBERTa v3 base
CV strategy: K-Fold
Metric: Quadratic Weighted Kappa (QWK)
Post-processing: Threshold optimization
Best CV: <add>
Best LB: <add>
```

A strong next step is to add a compact benchmark table here once you have multiple runs.

| Experiment | Backbone | CV | LB | Notes |
|---|---|---:|---:|---|
| Baseline | DeBERTa v3 base | TBD | TBD | Initial training pipeline |
| Exp 2 | TBD | TBD | TBD | Add notes |
| Exp 3 | TBD | TBD | TBD | Add notes |

---

## Reproducibility

To reproduce this project end to end, you need:

- Python environment setup
- Kaggle competition access
- Kaggle credentials
- pretrained model download
- config-driven training and inference

Expected local artifacts:

- `data/` for competition files
- `models/` for downloaded pretrained weights
- `outputs/` for checkpoints, predictions, and submission files

---

## Ideas for improvement

This repo already has a strong baseline structure. Good next upgrades would be:

- add **Weights & Biases** or **MLflow** experiment tracking
- publish **fold-wise CV results**
- add **error analysis** by adjacent score confusion
- compare **multiple transformer backbones**
- try **ordinal-aware objectives**
- build a lightweight **demo app or API** for inference
- document **inference latency and memory footprint**


---

## Acknowledgments

- **Kaggle** for hosting the competition
- **Learning Agency Lab** for the challenge
- **Hugging Face** for transformer tooling
- **Microsoft** for the DeBERTa model family

---

## License

This project is released under the **MIT License**. See [`LICENSE`](./LICENSE) for details.
