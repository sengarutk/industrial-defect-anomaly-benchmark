import numpy as np
import pytest

from src.experiments.cct_ablation import decision_changes, evaluate_run, stratified_half_split
from src.metrics.cct import optimize_cct_threshold
from src.metrics.operational import expected_cost


def brute_force_min_cost(nom, dfc, prior, r, budget=None):
    best = np.inf
    for t in np.r_[np.unique(np.r_[nom, dfc]), np.inf]:
        fpr, fnr = np.mean(nom >= t), np.mean(dfc < t)
        if budget is not None and 1000 * fpr > budget:
            continue
        best = min(best, expected_cost(fpr, fnr, prior, r))
    return best


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("budget", [None, 5.0, 100.0])
def test_cct_is_constrained_argmin(seed, budget):
    rng = np.random.RandomState(seed)
    nom, dfc = rng.randn(60), rng.randn(30) + 1.5
    res = optimize_cct_threshold(nom, dfc, prior=0.05, cost_ratio=20, max_alerts_per_1k=budget)
    assert res["calib_cost"] == pytest.approx(brute_force_min_cost(nom, dfc, 0.05, 20, budget))
    if budget is not None:
        assert 1000 * np.mean(nom >= res["threshold"]) <= budget + 1e-9
        assert res["budget_satisfied_on_calibration"]


def test_out_of_sample_violations_are_reported_not_clamped():
    """Regression: CCT false-alarm rates used to be clipped to the budget with min(budget, fa)."""
    rng = np.random.RandomState(0)
    labels = np.r_[np.zeros(20, int), np.ones(20, int)]
    scores = np.r_[rng.randn(20), rng.randn(20) + 1.0]
    calib = rng.randn(5) - 3.0  # calibration nominals far below the test nominals -> threshold too low
    rows, _ = evaluate_run(scores, labels, calib, [10.0], 0.01, 5.0, n_splits=3, seed=0)
    fa = [r["eval_fa_at_1k"] for r in rows if r["strategy"] == "nominal_budget"]
    assert max(fa) > 5.0
    assert all(r["budget_violated"] == (r["eval_fa_at_1k"] > 5.0) for r in rows)


def test_split_is_stratified_and_disjoint():
    labels = np.r_[np.zeros(21, int), np.ones(9, int)]
    c, e = stratified_half_split(labels, np.random.RandomState(0))
    assert not set(c) & set(e) and len(c) + len(e) == 30
    assert labels[c].sum() == 4 and labels[e].sum() == 5


def test_decision_changes_accounting():
    s = np.array([0.1, 0.3, 0.5, 0.7, 0.4, 0.6, 0.8, 0.9])
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    d = decision_changes(s, y, 0.45, 0.65)
    assert d == {"flips": 2, "false_alarms_removed": 1, "false_alarms_added": 0, "defects_lost": 1,
                 "defects_gained": 0}


def test_cct_requires_both_classes():
    with pytest.raises(ValueError):
        optimize_cct_threshold(np.array([]), np.array([1.0]), 0.01, 10)
