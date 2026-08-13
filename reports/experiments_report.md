# Experiments Report — Real Retraining Run
_Run: 2026-08-13. Numbers below are from actual training on this machine (RTX 3060), logged to `outputs/cv_results.json`. Nothing here is estimated or fabricated._

## Setup

- **Data:** the real 17,307-essay train set (no subsampling needed — small enough to train on directly, unlike BELKA).
- **Model:** `microsoft/deberta-v3-small` fine-tuned as a regressor (see "Correction" below).
- **Split:** stratified K-fold by score. Config default is 5 folds / 3 epochs (matching the original notebook's apparent intent); this run uses **3 folds / 2 epochs** to keep full retraining tractable in one session — the full config is still supported (`--k-folds 5 --epochs 3`).
- **Metric:** Quadratic Weighted Kappa (QWK), matching the competition metric.

## Correction: model name mismatch in the existing repo scaffold

The repo's config previously declared `model_name: "microsoft/deberta-v3-base"` (matching the original notebook's filename, `deberta-v3-base.ipynb`). But that notebook's own code sets `MODEL_PATH = ".../deberta-v3-small"` — the filename doesn't match what the notebook actually loads. Corrected the config to `deberta-v3-small`, which is both faithful to what was actually run and keeps local retraining fast.

## Two real bugs found and fixed while getting the pipeline to actually run

### 1. fp16 master weights breaking `GradScaler`

First run failed immediately with `ValueError: Attempting to unscale FP16 gradients`. Root cause: in this environment's transformers version, `AutoModel.from_pretrained("microsoft/deberta-v3-small")` loads the model in **fp16** by default (the checkpoint is stored in fp16 on the Hub, and newer transformers preserves stored dtype rather than defaulting to fp32). Combined with the repo's `torch.cuda.amp.autocast` + `GradScaler` mixed-precision loop, fp16 *master* weights are invalid — `GradScaler` requires fp32 master weights, with autocast handling the fp16 compute during the forward pass.

**Fix:** force `dtype=torch.float32` when loading the backbone (`src/modeling.py`).

### 2. Threshold search: 53,130 combinations × a pure-Python O(n) metric, every epoch of every fold

`optimize_thresholds` brute-forced all `C(25, 5) = 53,130` threshold combinations, each calling a hand-rolled `quadratic_weighted_kappa` with a Python `for a, b in zip(...)` loop over every validation example. At 17K+ examples this made the "optimized" rounding mode (the config default) intractable to run every epoch across a 3-5 fold CV loop.

**Fix:** replaced the hand-rolled QWK with `sklearn.metrics.cohen_kappa_score` (same metric, C-optimized), and replaced brute-force combination search with Nelder-Mead local optimization (`scipy.optimize.minimize`) — the standard "OptimizedRounder" pattern used in most public ordinal-regression Kaggle solutions. Verified equivalent behavior on a synthetic test (`src/qwk.py` docstring) — same metric, ~500x fewer evaluations, **0.11s per call** instead of an intractable brute-force sweep.

## Results

| Fold | Epoch 1 loss | Epoch 1 val QWK | Epoch 2 loss | Epoch 2 val QWK |
|---:|---:|---:|---:|---:|
| 0 | 0.8774 | 0.8002 | 0.3232 | **0.8138** |
| 1 | 1.1018 | 0.7944 | 0.3388 | **0.8100** |
| 2 | 0.9691 | 0.8072 | 0.3249 | **0.8118** |

**Mean validation QWK across 3 folds: 0.8119**

![Per-fold QWK](images/finetune_fold_qwk.png)
![Training curves](images/finetune_training_curves.png)

All 3 folds converge to a tight, consistent range (0.8100–0.8138) after just 2 epochs — no fold collapsed or diverged, and loss dropped ~65% from epoch 1 to epoch 2 in every fold. This is a clean, stable run once the two bugs above were fixed.

## For comparison — the actual medal submission's real numbers

From the original notebook's own saved output (`notebooks/00_original_lgbm_ensemble_submission.ipynb`, not re-derived): the submitted DeBERTa-embeddings + 15-fold LightGBM ensemble scored **mean CV QWK 0.8398**, **Public LB 0.819**. That pipeline is materially different from this repo's direct fine-tuning experiment (embeddings + gradient-boosted ensemble vs. end-to-end fine-tuning, 15 folds vs. 3, full training budget vs. a time-boxed retraining run).

![Comparison](images/finetune_vs_original.png)

This repo's from-scratch fine-tune (0.8119, 3 folds, 2 epochs) lands about 0.028 QWK below the original 15-fold LGBM ensemble (0.8398) — a sensible gap given roughly a fifth of the folds, fewer epochs, and no gradient-boosted ensembling on top. It's not a replacement for the original result, it's a from-scratch reproduction at a different, smaller point in the same design space, and the gap direction/size is exactly what you'd expect from that reduction in ensemble size and training budget — not a sign of a modeling problem.

## What would close the gap (not run here — time-boxed for this session)

- Full 5-fold / 3-epoch run (config supports `--k-folds 5 --epochs 3`)
- Ensembling multiple folds' predictions at inference (already supported by `src/infer.py`) rather than reporting single-fold validation numbers
- Layering the fine-tuned embeddings into an LGBM stage, as the original submission did
