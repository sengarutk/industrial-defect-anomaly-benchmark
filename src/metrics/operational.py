import warnings
from typing import Dict, Any, Union
import numpy as np


def compute_fa_at_1k(labels: np.ndarray, scores: np.ndarray, threshold: float) -> float:
    """
    Computes False Alarms per 1,000 normal items:
    FA@1k = (sum_{i in Normal} I(S_i >= tau) / N_normal) * 1000
    """
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)

    normal_mask = (labels == 0)
    n_normal = np.sum(normal_mask)
    if n_normal == 0:
        return 0.0

    false_alarms = np.sum((scores[normal_mask] >= threshold))
    return float((false_alarms / n_normal) * 1000.0)


def compute_md_at_1k(labels: np.ndarray, scores: np.ndarray, threshold: float) -> float:
    """
    Computes Missed Defects per 1,000 defective items (escaped defects):
    MD@1k = (sum_{j in Defect} I(S_j < tau) / N_defect) * 1000
    """
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)

    defect_mask = (labels == 1)
    n_defect = np.sum(defect_mask)
    if n_defect == 0:
        return 0.0

    missed_defects = np.sum((scores[defect_mask] < threshold))
    return float((missed_defects / n_defect) * 1000.0)


def compute_cost_weighted_error(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    cost_ratio: float = 10.0
) -> float:
    """
    Computes average cost per item where false alarms cost 1.0 and missed defects cost r:
    CWE(tau, r) = (1 / N) * [ sum_{i in Normal} I(S_i >= tau) * 1.0 + sum_{j in Defect} I(S_j < tau) * r ]
    """
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)

    n_total = len(labels)
    if n_total == 0:
        return 0.0

    normal_mask = (labels == 0)
    defect_mask = (labels == 1)

    fp = np.sum(scores[normal_mask] >= threshold)
    fn = np.sum(scores[defect_mask] < threshold)

    total_cost = (fp * 1.0) + (fn * float(cost_ratio))
    return float(total_cost / n_total)


def compute_tpr_at_alert_budget(
    labels: np.ndarray,
    scores: np.ndarray,
    max_alerts_per_1k: float = 5.0
) -> Dict[str, float]:
    """
    Identifies threshold tau_budget where the empirical alert rate on nominal images <= max_alerts_per_1k / 1000.
    Returns achieved TPR, achieved FPR, MD@1k, and the operating threshold tau_budget.
    """
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)

    normal_mask = (labels == 0)
    defect_mask = (labels == 1)

    n_normal = np.sum(normal_mask)
    n_defect = np.sum(defect_mask)

    if n_normal == 0:
        th = float(np.min(scores)) if len(scores) > 0 else 0.5
        tpr = 1.0 if n_defect > 0 else 0.0
        return {
            "tpr": float(tpr),
            "fpr": 0.0,
            "md_at_1k": float((1.0 - tpr) * 1000.0),
            "threshold": th,
            "fa_at_1k": 0.0
        }

    target_fpr = max_alerts_per_1k / 1000.0
    nominal_scores = scores[normal_mask]

    q = max(0.0, min(100.0, (1.0 - target_fpr) * 100.0))
    threshold = float(np.percentile(nominal_scores, q))

    fp = np.sum(nominal_scores >= threshold)
    fpr = float(fp / n_normal) if n_normal > 0 else 0.0

    if n_defect > 0:
        tp = np.sum(scores[defect_mask] >= threshold)
        tpr = float(tp / n_defect)
    else:
        tpr = 0.0

    md_at_1k = float((1.0 - tpr) * 1000.0)
    fa_at_1k = float(fpr * 1000.0)

    return {
        "tpr": tpr,
        "fpr": fpr,
        "md_at_1k": md_at_1k,
        "threshold": threshold,
        "fa_at_1k": fa_at_1k
    }


def compute_operator_overload(
    alert_stream: np.ndarray,
    operator_capacity_per_window: int = 60,
    window_size: int = 1000
) -> Dict[str, float]:
    """
    Chunks continuous inspection stream into operational windows (e.g. 1 hour = 1,000 parts).
    Computes mean review load per window, peak load, and Overload Probability:
    P(Overload) = (1 / W) * sum_{w=1}^W I(Alerts_w > C_operator)
    """
    alerts = np.asarray(alert_stream, dtype=int)
    n = len(alerts)

    if n == 0:
        return {
            "mean_load": 0.0,
            "peak_load": 0.0,
            "overload_probability": 0.0,
            "num_windows": 0
        }

    num_windows = max(1, n // window_size)
    window_loads = []

    for w in range(num_windows):
        start_idx = w * window_size
        end_idx = min((w + 1) * window_size, n) if w == num_windows - 1 else (w + 1) * window_size
        w_alerts = np.sum(alerts[start_idx:end_idx])
        window_loads.append(float(w_alerts))

    window_loads_arr = np.array(window_loads, dtype=float)
    mean_load = float(np.mean(window_loads_arr))
    peak_load = float(np.max(window_loads_arr))
    overload_prob = float(np.mean(window_loads_arr > operator_capacity_per_window))

    return {
        "mean_load": mean_load,
        "peak_load": peak_load,
        "overload_probability": overload_prob,
        "num_windows": int(num_windows)
    }
