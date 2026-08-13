"""Real EDA on the AES2 train set (17,307 essays). Run: python3 eda/eda_essays.py"""
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "train.csv"
IMG_DIR = ROOT / "reports" / "images"
IMG_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA)
stats = {}

stats["n_essays"] = len(df)
stats["score_distribution"] = df["score"].value_counts().sort_index().to_dict()

df["char_len"] = df["full_text"].str.len()
df["word_count"] = df["full_text"].str.split().str.len()
df["sentence_count"] = df["full_text"].apply(lambda t: len(re.split(r"[.!?]+", t.strip())) if isinstance(t, str) else 0)
df["avg_word_len"] = df["full_text"].apply(
    lambda t: np.mean([len(w) for w in t.split()]) if isinstance(t, str) and t.split() else 0
)

stats["char_len_summary"] = df["char_len"].describe().to_dict()
stats["word_count_summary"] = df["word_count"].describe().to_dict()

# 1. Score distribution
plt.figure(figsize=(6, 4))
counts = df["score"].value_counts().sort_index()
bars = plt.bar(counts.index.astype(str), counts.values, color="#3b6fa0")
plt.title(f"Essay score distribution (n={len(df)})")
plt.xlabel("Score (1-6)")
plt.ylabel("Count")
for b, v in zip(bars, counts.values):
    plt.text(b.get_x() + b.get_width() / 2, v + 50, str(v), ha="center", fontsize=9)
plt.tight_layout()
plt.savefig(IMG_DIR / "score_distribution.png", dpi=130)
plt.close()

# 2. Word count distribution
plt.figure(figsize=(7, 4))
plt.hist(df["word_count"], bins=60, color="#5a9367")
plt.title("Essay word count distribution")
plt.xlabel("Word count")
plt.ylabel("Count")
plt.axvline(df["word_count"].median(), color="crimson", linestyle="--", label=f"median={df['word_count'].median():.0f}")
plt.legend()
plt.tight_layout()
plt.savefig(IMG_DIR / "word_count_dist.png", dpi=130)
plt.close()

# 3. Word count vs score (boxplot-ish via mean+std)
plt.figure(figsize=(7, 4))
grouped = df.groupby("score")["word_count"].agg(["mean", "std"])
plt.errorbar(grouped.index, grouped["mean"], yerr=grouped["std"], fmt="o-", capsize=4, color="#c17a3d")
plt.title("Word count vs. score (mean ± std)")
plt.xlabel("Score")
plt.ylabel("Word count")
plt.tight_layout()
plt.savefig(IMG_DIR / "wordcount_vs_score.png", dpi=130)
plt.close()

corr_wc = df["word_count"].corr(df["score"])
stats["word_count_score_correlation"] = float(corr_wc)

# 4. Avg word length vs score (proxy for vocabulary sophistication)
plt.figure(figsize=(7, 4))
grouped2 = df.groupby("score")["avg_word_len"].mean()
plt.plot(grouped2.index, grouped2.values, "o-", color="#7a6fa0")
plt.title("Mean word length vs. score")
plt.xlabel("Score")
plt.ylabel("Mean word length (chars)")
plt.tight_layout()
plt.savefig(IMG_DIR / "wordlen_vs_score.png", dpi=130)
plt.close()

corr_wl = df["avg_word_len"].corr(df["score"])
stats["avg_word_len_score_correlation"] = float(corr_wl)

stats["sentence_count_summary"] = df["sentence_count"].describe().to_dict()

with open(ROOT / "reports" / "eda_stats.json", "w") as f:
    json.dump(stats, f, indent=2, default=float)

print(json.dumps(stats, indent=2, default=float))
print("Saved charts to reports/images/")
