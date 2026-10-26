"""Cost-Calibrated Thresholding (CCT).

CCT picks the threshold that minimises expected cost on a labelled calibration
set, optionally subject to an alarm budget on the calibration nominals:

    tau* = argmin_tau  (1 - pi) FPR_cal(tau) + pi * r * FNR_cal(tau)
           s.t.        FPR_cal(tau) <= b / 1000         (if a budget b is given)

It is ordinary constrained empirical risk minimisation over a scalar threshold;
the repository evaluates it out-of-sample on disjoint data.
"""
from typing import Dict, Optional

import numpy as np

from .operational import expected_cost


def _candidate_thresholds(scores: np.ndarray) -> np.ndarray:
    u = np.unique(scores)
    # every distinct score, plus one value above the maximum ("flag nothing")
    return np.r_[u, np.nextafter(u[-1], np.inf)]


def optimize_cct_threshold(nominal_scores: np.ndarray, defect_scores: np.ndarray, prior: float,
                           cost_ratio: float, max_alerts_per_1k: Optional[float] = None) -> Dict[str, float]:
    nom = np.asarray(nominal_scores, dtype=np.float64).ravel()
    dfc = np.asarray(defect_scores, dtype=np.float64).ravel()
    if nom.size == 0 or dfc.size == 0:
        raise ValueError("CCT needs at least one nominal and one defective calibration score")
    cand = _candidate_thresholds(np.r_[nom, dfc])
    nom_sorted, dfc_sorted = np.sort(nom), np.sort(dfc)
    fpr = 1.0 - np.searchsorted(nom_sorted, cand, side="left") / nom.size   # P(nom >= tau)
    fnr = np.searchsorted(dfc_sorted, cand, side="left") / dfc.size          # P(def < tau)
    cost = (1.0 - prior) * fpr + prior * cost_ratio * fnr
    feasible = np.ones_like(cost, dtype=bool)
    if max_alerts_per_1k is not None:
        feasible = fpr <= max_alerts_per_1k / 1000.0 + 1e-12
    # "flag nothing" (last candidate) is always feasible, so the feasible set is never empty
    masked = np.where(feasible, cost, np.inf)
    best = int(np.argmin(masked))
    return {
        "threshold": float(cand[best]),
        "calib_cost": float(cost[best]),
        "calib_fpr": float(fpr[best]),
        "calib_fnr": float(fnr[best]),
        "budget_satisfied_on_calibration": bool(feasible[best]),
    }


def empirical_cost_curve(nominal_scores: np.ndarray, defect_scores: np.ndarray, thresholds: np.ndarray,
                         prior: float, cost_ratio: float) -> np.ndarray:
    nom = np.sort(np.asarray(nominal_scores, dtype=np.float64))
    dfc = np.sort(np.asarray(defect_scores, dtype=np.float64))
    fpr = 1.0 - np.searchsorted(nom, thresholds, side="left") / nom.size
    fnr = np.searchsorted(dfc, thresholds, side="left") / dfc.size
    return np.array([expected_cost(a, b, prior, cost_ratio) for a, b in zip(fpr, fnr)])
