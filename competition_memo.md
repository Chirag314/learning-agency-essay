# Competition Understanding — Learning Agency Lab: Automated Essay Scoring 2.0

**Competition:** [learning-agency-lab-automated-essay-scoring-2](https://www.kaggle.com/competitions/learning-agency-lab-automated-essay-scoring-2) (Kaggle)
**Task:** predict a holistic quality score (1–6) for student-written essays
**Metric:** Quadratic Weighted Kappa (QWK) — penalizes predictions further from the true score more heavily than near-misses
**My result:** Silver medal

---

## Dataset (verified against the real downloaded files, 2026-08-13)

| | |
|---|---:|
| Train essays | 17,307 |
| Test essays | 3 (public leaderboard placeholder set) |
| Columns | `essay_id`, `full_text`, `score` |

**Score distribution is imbalanced and ordinal** — most essays cluster around 2–4, with far fewer at the extremes:

| Score | Count |
|---:|---:|
| 1 | 1,252 |
| 2 | 4,723 |
| 3 | 6,280 |
| 4 | 3,926 |
| 5 | 970 |
| 6 | 156 |

Full EDA, including the length/vocabulary-vs-score relationship: `reports/eda_report.md`.

## My approach — two real notebooks, honestly provenanced

1. **`notebooks/00b_original_deberta_finetune.ipynb`** — fine-tunes `microsoft/deberta-v3-small` as a 5-fold regressor (despite the original filename saying "base," the code actually loads `deberta-v3-small`).
2. **`notebooks/00_original_lgbm_ensemble_submission.ipynb`** — the actual submitted notebook. Explicitly forked from 3 public kernels (credited in its own first cell: siddhvr's AES2 DeBERTa-LGBM baseline, olyatsimboy's 5-fold DeBERTa-LGBM, aikhmelnytskyy's quick-start LGBM), extended with CountVectorizer n-gram features. Uses DeBERTa embeddings as features into a 15-fold LightGBM ensemble with a custom QWK training objective.
   - **Real logged result (from the notebook's own saved output, not re-derived):** mean Cohen's kappa (QWK) across 15 folds = **0.8398**, Public LB = **0.819** (per the notebook's filename, `...lb-819`).

## What this repo adds beyond the original submission

Unlike BELKA, this competition's data is small enough (17K essays, 36MB) to actually retrain end-to-end locally rather than just reproduce inference. For this writeup (built retrospectively, 2026-08-13), I:

1. Ran real EDA on the full 17,307-essay set, including word-count/vocabulary correlation with score
2. **Found and fixed two real bugs** in the existing repo scaffold while getting it to actually run — see `reports/experiments_report.md`
3. Actually fine-tuned DeBERTa-v3-small end-to-end (not just inference against pretrained weights) and logged real per-fold QWK
