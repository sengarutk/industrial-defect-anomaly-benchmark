import warnings
from typing import Dict

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


def compute_image_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        warnings.warn("compute_image_auroc: only one class present; returning NaN.")
        return float("nan")
    return float(roc_auc_score(labels, scores))


def compute_image_ap(labels: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        warnings.warn("compute_image_ap: only one class present; returning NaN.")
        return float("nan")
    return float(average_precision_score(labels, scores))


def compute_oracle_f1(labels: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    """Best F1 over all thresholds chosen *on the evaluated labels* (an in-sample upper bound, not deployable)."""
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if len(np.unique(labels)) < 2:
        return {"oracle_f1": float("nan"), "oracle_threshold": float("nan")}
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    p, r = precision[:-1], recall[:-1]
    f1 = np.divide(2 * p * r, p + r, out=np.zeros_like(p), where=(p + r) > 0)
    best = int(np.argmax(f1))
    return {"oracle_f1": float(f1[best]), "oracle_threshold": float(thresholds[best])}
