
import numpy as np

def quadratic_weighted_kappa(y_true, y_pred, min_rating=1, max_rating=6):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    assert y_true.shape == y_pred.shape

    num_ratings = int(max_rating - min_rating + 1)
    hist_true = np.zeros(num_ratings)
    hist_pred = np.zeros(num_ratings)
    O = np.zeros((num_ratings, num_ratings))

    for a, b in zip(y_true, y_pred):
        O[a - min_rating, b - min_rating] += 1
        hist_true[a - min_rating] += 1
        hist_pred[b - min_rating] += 1

    E = np.outer(hist_true, hist_pred) / y_true.size

    W = np.zeros((num_ratings, num_ratings))
    for i in range(num_ratings):
        for j in range(num_ratings):
            W[i, j] = ((i - j) ** 2) / ((num_ratings - 1) ** 2)

    denom = np.sum(W * E)
    return 1.0 - (np.sum(W * O) / denom if denom != 0 else 0.0)

def optimize_thresholds(scores_float, y_true_int, num_classes=6):
    scores = np.asarray(scores_float)
    y_true = np.asarray(y_true_int, dtype=int)
    best_kappa = -1.0
    best_th = None
    candidates = np.linspace(1.2, 5.8, 25)
    from itertools import combinations
    for idxs in combinations(range(len(candidates)), 5):
        th = np.array([candidates[i] for i in idxs])
        if not np.all(np.diff(th) > 0):
            continue
        preds = np.digitize(scores, th) + 1
        kappa = quadratic_weighted_kappa(y_true, preds, 1, num_classes)
        if kappa > best_kappa:
            best_kappa = kappa
            best_th = th
    return best_th, best_kappa

def apply_thresholds(scores_float, thresholds):
    scores = np.asarray(scores_float)
    return np.digitize(scores, thresholds) + 1
