import warnings
from typing import Dict
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


def compute_image_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    """
    Computes sample/image-level Area Under the ROC curve.
    labels: binary ground truth labels [N] in {0, 1}
    scores: continuous anomaly scores [N]
    """
    y_true = np.asarray(labels).flatten().astype(int)
    y_score = np.asarray(scores).flatten().astype(float)

    if len(np.unique(y_true)) < 2:
        warnings.warn("compute_image_auroc: Ground truth contains only one class. Returning 0.5.")
        return 0.5

    try:
        score = float(roc_auc_score(y_true, y_score))
        return score
    except ValueError:
        return 0.5


def compute_image_ap(labels: np.ndarray, scores: np.ndarray) -> float:
    """
    Computes sample/image-level Average Precision (AP).
    labels: binary ground truth labels [N] in {0, 1}
    scores: continuous anomaly scores [N]
    """
    y_true = np.asarray(labels).flatten().astype(int)
    y_score = np.asarray(scores).flatten().astype(float)

    if len(np.unique(y_true)) < 2:
        warnings.warn("compute_image_ap: Ground truth contains only one class. Returning 0.0.")
        return 0.0

    try:
        score = float(average_precision_score(y_true, y_score))
        return score
    except ValueError:
        return 0.0


def auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Backward-compatible alias for compute_image_auroc.
    """
    return compute_image_auroc(y_true, y_score)


def compute_optimal_f1(
    labels: np.ndarray,
    scores: np.ndarray,
    num_thresholds: int = 1000
) -> Dict[str, float]:
    """
    Locates the optimal decision boundary that maximizes the F1-score:
      F1 = 2 * (Precision * Recall) / (Precision + Recall)
    Returns:
      dict with keys: max_f1, optimal_threshold, precision_at_optimal, recall_at_optimal
    """
    y_true = np.asarray(labels).flatten().astype(int)
    y_score = np.asarray(scores).flatten().astype(float)

    if len(y_true) == 0:
        return {
            "max_f1": 0.0,
            "optimal_threshold": 0.0,
            "precision_at_optimal": 0.0,
            "recall_at_optimal": 0.0,
        }

    unique_scores = np.unique(y_score)
    if len(unique_scores) <= num_thresholds:
        thresholds = unique_scores
    else:
        thresholds = np.linspace(float(np.min(y_score)), float(np.max(y_score)), num_thresholds)

    best_f1 = 0.0
    best_threshold = float(thresholds[0]) if len(thresholds) > 0 else 0.0
    best_prec = 0.0
    best_rec = 0.0

    for thresh in thresholds:
        preds = (y_score >= thresh).astype(int)
        tp = np.sum((preds == 1) & (y_true == 1))
        fp = np.sum((preds == 1) & (y_true == 0))
        fn = np.sum((preds == 0) & (y_true == 1))

        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(thresh)
            best_prec = prec
            best_rec = rec

    return {
        "max_f1": float(np.clip(best_f1, 0.0, 1.0)),
        "optimal_threshold": best_threshold,
        "precision_at_optimal": float(np.clip(best_prec, 0.0, 1.0)),
        "recall_at_optimal": float(np.clip(best_rec, 0.0, 1.0)),
    }
