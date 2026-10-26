import numpy as np
import pytest

from src.metrics.operational import (alert_budget_threshold, clopper_pearson, compute_expected_cost,
                                     compute_fa_at_1k, compute_md_at_1k, compute_operator_overload, expected_cost,
                                     min_resolvable_rate_per_1k, operating_point)


def test_rates_and_cost_definition():
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    scores = np.array([0.1, 0.6, 0.2, 0.3, 0.4, 0.8, 0.9, 0.7])
    assert compute_fa_at_1k(labels, scores, 0.5) == 250.0
    assert compute_md_at_1k(labels, scores, 0.5) == 250.0
    assert compute_expected_cost(labels, scores, 0.5, prior=0.01, cost_ratio=10) == pytest.approx(
        0.99 * 0.25 + 0.01 * 10 * 0.25)
    assert expected_cost(0.0, 0.0, 0.01, 50) == 0.0


@pytest.mark.parametrize("n,budget", [(20, 5.0), (42, 5.0), (300, 5.0), (1000, 5.0), (1000, 50.0), (57, 50.0)])
def test_alert_budget_threshold_guarantee_on_calibration(n, budget):
    rng = np.random.RandomState(n)
    s = rng.randn(n)
    tau = alert_budget_threshold(s, budget)
    assert 1000 * np.mean(s >= tau) <= budget + 1e-9
    # it is the lowest such threshold among the observed scores
    lower = np.sort(s)[np.sort(s) < tau]
    if lower.size:
        assert 1000 * np.mean(s >= lower.max()) > budget


def test_small_sample_budget_puts_threshold_above_max():
    s = np.linspace(0, 1, 42)
    assert alert_budget_threshold(s, 5.0) > s.max()
    assert min_resolvable_rate_per_1k(20) == 50.0


def test_clopper_pearson_known_values():
    lo, hi = clopper_pearson(0, 20)
    assert lo == 0.0 and hi == pytest.approx(0.1684, abs=1e-4)
    lo, hi = clopper_pearson(20, 20)
    assert hi == 1.0 and lo == pytest.approx(0.8316, abs=1e-4)


def test_operating_point_reports_intervals():
    labels = np.r_[np.zeros(20, int), np.ones(10, int)]
    scores = np.r_[np.linspace(0, 1, 20), np.linspace(2, 3, 10)]
    op = operating_point(labels, scores, threshold=1.5)
    assert op["fa_at_1k"] == 0 and op["tpr"] == 1.0
    assert op["fa_at_1k_ci_high"] == pytest.approx(168.4, abs=0.1)


def test_operator_overload_uses_non_overlapping_windows():
    alerts = np.r_[np.ones(20), np.zeros(980), np.ones(80), np.zeros(920), np.ones(10), np.zeros(990), np.ones(5)]
    res = compute_operator_overload(alerts, operator_capacity_per_window=60, window_size=1000)
    assert res["num_windows"] == 3
    assert res["p_overload"] == pytest.approx(1 / 3)
    assert res["max_window_alerts"] == 80
    assert np.isnan(compute_operator_overload(np.ones(10), 60, 1000)["p_overload"])
