from typing import Dict, Any, List
import numpy as np


def compute_ece(scores: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """
    Computes Expected Calibration Error (ECE).
    Continuous anomaly scores are normalized to [0, 1] via min-max scaling,
    partitioned into M equal-width confidence bins, and difference between
    mean confidence and empirical accuracy is weighted by bin counts.
    """
    y_score = np.asarray(scores, dtype=float).flatten()
    y_true = np.asarray(labels, dtype=int).flatten()

    N = len(y_score)
    if N == 0:
        return 0.0

    s_min = float(np.min(y_score))
    s_max = float(np.max(y_score))
    if s_max == s_min:
        norm_scores = np.zeros_like(y_score)
    else:
        norm_scores = (y_score - s_min) / (s_max - s_min + 1e-8)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for m in range(n_bins):
        bin_lower = bins[m]
        bin_upper = bins[m + 1]

        if m == n_bins - 1:
            in_bin = (norm_scores >= bin_lower) & (norm_scores <= bin_upper)
        else:
            in_bin = (norm_scores >= bin_lower) & (norm_scores < bin_upper)

        count = int(np.sum(in_bin))
        if count > 0:
            bin_conf = float(np.mean(norm_scores[in_bin]))
            bin_acc = float(np.mean(y_true[in_bin]))
            ece += (count / N) * abs(bin_acc - bin_conf)

    return float(np.clip(ece, 0.0, 1.0))


def get_reliability_diagram_data(
    scores: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15
) -> Dict[str, Any]:
    """
    Computes calibration curve components for reliability diagrams.
    Returns:
      dict with keys: bin_centers, bin_accuracies, bin_confidences, bin_counts
    """
    y_score = np.asarray(scores, dtype=float).flatten()
    y_true = np.asarray(labels, dtype=int).flatten()

    N = len(y_score)
    if N == 0:
        return {
            "bin_centers": [],
            "bin_accuracies": [],
            "bin_confidences": [],
            "bin_counts": []
        }

    s_min = float(np.min(y_score))
    s_max = float(np.max(y_score))
    if s_max == s_min:
        norm_scores = np.zeros_like(y_score)
    else:
        norm_scores = (y_score - s_min) / (s_max - s_min + 1e-8)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers: List[float] = []
    bin_accuracies: List[float] = []
    bin_confidences: List[float] = []
    bin_counts: List[int] = []

    for m in range(n_bins):
        bin_lower = bins[m]
        bin_upper = bins[m + 1]
        bin_centers.append(float((bin_lower + bin_upper) / 2.0))

        if m == n_bins - 1:
            in_bin = (norm_scores >= bin_lower) & (norm_scores <= bin_upper)
        else:
            in_bin = (norm_scores >= bin_lower) & (norm_scores < bin_upper)

        count = int(np.sum(in_bin))
        bin_counts.append(count)

        if count > 0:
            bin_confidences.append(float(np.mean(norm_scores[in_bin])))
            bin_accuracies.append(float(np.mean(y_true[in_bin])))
        else:
            bin_confidences.append(0.0)
            bin_accuracies.append(0.0)

    return {
        "bin_centers": bin_centers,
        "bin_accuracies": bin_accuracies,
        "bin_confidences": bin_confidences,
        "bin_counts": bin_counts,
    }
