"""Operating-point metrics.

Conventions used everywhere in the repository:
  * A part is flagged when score >= threshold.
  * FA@1k  = false alarms per 1,000 *nominal* parts  = 1000 * FPR.
  * MD@1k  = missed defects per 1,000 *defective* parts = 1000 * FNR.
  * Expected cost per inspected part at deployment prior pi and cost ratio r
    (cost of a false alarm = 1, cost of a missed defect = r):
        C(tau) = (1 - pi) * FPR(tau) + pi * r * FNR(tau)
    This single definition is used by every table and figure.
"""
from typing import Dict

import numpy as np
from scipy import stats


def _split(labels: np.ndarray, scores: np.ndarray):
    labels = np.asarray(labels, dtype=int).ravel()
    scores = np.asarray(scores, dtype=np.float64).ravel()
    if labels.shape != scores.shape:
        raise ValueError("labels and scores must have the same length")
    return scores[labels == 0], scores[labels == 1]


def false_positive_rate(nominal_scores: np.ndarray, threshold: float) -> float:
    nominal_scores = np.asarray(nominal_scores, dtype=np.float64)
    return float(np.mean(nominal_scores >= threshold)) if nominal_scores.size else float("nan")


def false_negative_rate(defect_scores: np.ndarray, threshold: float) -> float:
    defect_scores = np.asarray(defect_scores, dtype=np.float64)
    return float(np.mean(defect_scores < threshold)) if defect_scores.size else float("nan")


def compute_fa_at_1k(labels: np.ndarray, scores: np.ndarray, threshold: float) -> float:
    nom, _ = _split(labels, scores)
    return 1000.0 * false_positive_rate(nom, threshold)


def compute_md_at_1k(labels: np.ndarray, scores: np.ndarray, threshold: float) -> float:
    _, dfc = _split(labels, scores)
    return 1000.0 * false_negative_rate(dfc, threshold)


def expected_cost(fpr: float, fnr: float, prior: float, cost_ratio: float) -> float:
    return float((1.0 - prior) * fpr + prior * cost_ratio * fnr)


def compute_expected_cost(labels: np.ndarray, scores: np.ndarray, threshold: float,
                          prior: float, cost_ratio: float) -> float:
    nom, dfc = _split(labels, scores)
    return expected_cost(false_positive_rate(nom, threshold), false_negative_rate(dfc, threshold), prior, cost_ratio)


def alert_budget_threshold(nominal_scores: np.ndarray, max_alerts_per_1k: float) -> float:
    """Lowest threshold whose empirical FPR on `nominal_scores` is <= max_alerts_per_1k / 1000.

    With n nominal scores at most floor(n * budget / 1000) of them may be flagged.
    When that is zero (e.g. budget 5/1k with n < 200), the threshold lies just
    above the largest nominal score. The guarantee holds on the calibration
    scores only; the realised rate on new data must be measured separately.
    """
    s = np.sort(np.asarray(nominal_scores, dtype=np.float64).ravel())[::-1]
    if s.size == 0:
        raise ValueError("alert_budget_threshold needs at least one nominal score")
    k = int(np.floor(s.size * max_alerts_per_1k / 1000.0 + 1e-9))
    if k >= s.size:
        return float(s[-1])
    return float(np.nextafter(s[k], np.inf))


def min_resolvable_rate_per_1k(n: int) -> float:
    """Smallest non-zero alarm rate (per 1k) that n samples can express."""
    return 1000.0 / n if n > 0 else float("inf")


def clopper_pearson(k: int, n: int, ci: float = 0.95):
    """Exact binomial confidence interval for a proportion k/n."""
    if n == 0:
        return float("nan"), float("nan")
    alpha = 1.0 - ci
    lo = 0.0 if k == 0 else stats.beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else stats.beta.ppf(1 - alpha / 2, k + 1, n - k)
    return float(lo), float(hi)


def operating_point(labels: np.ndarray, scores: np.ndarray, threshold: float, ci: float = 0.95) -> Dict[str, float]:
    """Rates at a fixed threshold with exact Clopper-Pearson intervals (per 1k)."""
    nom, dfc = _split(labels, scores)
    fp, tp = int(np.sum(nom >= threshold)), int(np.sum(dfc >= threshold))
    fa_lo, fa_hi = clopper_pearson(fp, nom.size, ci)
    tpr_lo, tpr_hi = clopper_pearson(tp, dfc.size, ci)
    return {
        "threshold": float(threshold),
        "n_nominal": int(nom.size),
        "n_defect": int(dfc.size),
        "fa_at_1k": 1000.0 * fp / nom.size if nom.size else float("nan"),
        "fa_at_1k_ci_low": 1000.0 * fa_lo,
        "fa_at_1k_ci_high": 1000.0 * fa_hi,
        "tpr": tp / dfc.size if dfc.size else float("nan"),
        "tpr_ci_low": tpr_lo,
        "tpr_ci_high": tpr_hi,
        "md_at_1k": 1000.0 * (1.0 - tp / dfc.size) if dfc.size else float("nan"),
    }


def compute_operator_overload(alerts: np.ndarray, operator_capacity_per_window: int = 60,
                              window_size: int = 1000) -> Dict[str, float]:
    """Splits the alert stream into consecutive non-overlapping windows (a trailing partial window is dropped)
    and reports how often the number of alerts exceeds the operator capacity."""
    alerts = np.asarray(alerts, dtype=int).ravel()
    n_windows = alerts.size // window_size if window_size > 0 else 0
    if n_windows == 0:
        return {"p_overload": float("nan"), "mean_window_alerts": float("nan"),
                "max_window_alerts": float("nan"), "num_windows": 0}
    sums = alerts[:n_windows * window_size].reshape(n_windows, window_size).sum(axis=1)
    return {
        "p_overload": float(np.mean(sums > operator_capacity_per_window)),
        "mean_window_alerts": float(np.mean(sums)),
        "max_window_alerts": float(np.max(sums)),
        "num_windows": int(n_windows),
    }
