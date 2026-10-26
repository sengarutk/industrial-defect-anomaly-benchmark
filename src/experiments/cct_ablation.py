"""Out-of-sample comparison of threshold rules, with repeated stratified test splits.

For every run archive and every split s = 1..n_splits:
  * the test set is split 50/50 (stratified by label) into a calibration half C and an evaluation half E;
  * calibration nominals = held-out train/good scores  U  nominal scores in C;  calibration defects = defects in C;
  * thresholds are chosen on calibration data only, then every metric is measured on E.

Rules compared (all evaluated with the same expected-cost definition):
  nominal_budget  - unsupervised: lowest threshold with calibration FPR <= budget (no defect labels used)
  cct_budget      - CCT with the same budget constraint (uses calibration defect labels)
  cct_unconstrained - CCT without a budget
  oracle          - cost-optimal threshold chosen on E itself (in-sample lower bound, not deployable)
"""
from typing import Dict, List, Sequence

import numpy as np

from src.metrics.cct import optimize_cct_threshold
from src.metrics.operational import alert_budget_threshold, expected_cost, false_negative_rate, false_positive_rate

STRATEGIES = ["nominal_budget", "cct_budget", "cct_unconstrained", "oracle"]


def stratified_half_split(labels: np.ndarray, rng: np.random.RandomState):
    """Returns (calibration_idx, evaluation_idx) with each class split as evenly as possible."""
    labels = np.asarray(labels, dtype=int)
    calib, evaluate = [], []
    for cls in (0, 1):
        idx = np.where(labels == cls)[0]
        rng.shuffle(idx)
        half = len(idx) // 2
        calib.append(idx[:half])
        evaluate.append(idx[half:])
    return np.sort(np.concatenate(calib)), np.sort(np.concatenate(evaluate))


def decision_changes(eval_scores: np.ndarray, eval_labels: np.ndarray, tau_from: float, tau_to: float) -> Dict[str, int]:
    """Counts decisions that flip when moving from tau_from to tau_to on the evaluation half."""
    a = eval_scores >= tau_from
    b = eval_scores >= tau_to
    nom, dfc = eval_labels == 0, eval_labels == 1
    return {
        "flips": int(np.sum(a != b)),
        "false_alarms_removed": int(np.sum(nom & a & ~b)),
        "false_alarms_added": int(np.sum(nom & ~a & b)),
        "defects_lost": int(np.sum(dfc & a & ~b)),
        "defects_gained": int(np.sum(dfc & ~a & b)),
    }


def evaluate_run(scores: np.ndarray, labels: np.ndarray, calib_nominal_scores: np.ndarray,
                 cost_ratios: Sequence[float], prior: float, budget_per_1k: float,
                 n_splits: int, seed: int):
    """Returns (metric_rows, decision_rows) for one run archive."""
    rng = np.random.RandomState(seed)
    rows: List[Dict] = []
    changes: List[Dict] = []
    for split in range(n_splits):
        c_idx, e_idx = stratified_half_split(labels, rng)
        cal_nom = np.r_[calib_nominal_scores, scores[c_idx][labels[c_idx] == 0]]
        cal_def = scores[c_idx][labels[c_idx] == 1]
        ev_s, ev_y = scores[e_idx], labels[e_idx]
        ev_nom, ev_def = ev_s[ev_y == 0], ev_s[ev_y == 1]
        tau_nominal = alert_budget_threshold(cal_nom, budget_per_1k)
        for r in cost_ratios:
            taus = {
                "nominal_budget": tau_nominal,
                "cct_budget": optimize_cct_threshold(cal_nom, cal_def, prior, r, budget_per_1k)["threshold"],
                "cct_unconstrained": optimize_cct_threshold(cal_nom, cal_def, prior, r, None)["threshold"],
                "oracle": optimize_cct_threshold(ev_nom, ev_def, prior, r, None)["threshold"],
            }
            for name, tau in taus.items():
                fpr, fnr = false_positive_rate(ev_nom, tau), false_negative_rate(ev_def, tau)
                rows.append({"split": split, "cost_ratio": r, "strategy": name, "threshold": tau,
                             "eval_fpr": fpr, "eval_fnr": fnr, "eval_fa_at_1k": 1000 * fpr,
                             "eval_cost": expected_cost(fpr, fnr, prior, r),
                             "budget_violated": bool(1000 * fpr > budget_per_1k + 1e-9),
                             "n_calib_nominal": int(cal_nom.size), "n_calib_defect": int(cal_def.size),
                             "n_eval_nominal": int(ev_nom.size), "n_eval_defect": int(ev_def.size)})
            changes.append({"split": split, "cost_ratio": r,
                            **decision_changes(ev_s, ev_y, taus["nominal_budget"], taus["cct_budget"]),
                            "n_eval_nominal": int(ev_nom.size), "n_eval_defect": int(ev_def.size)})
    return rows, changes
