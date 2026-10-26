import numpy as np
import pytest

from src.experiments.operational_eval import ProductionStreamSimulator


def sim(seed=0):
    return ProductionStreamSimulator(np.linspace(0, 1, 20), np.linspace(2, 3, 10), seed=seed)


def test_iid_exact_prior_and_pool_membership():
    labels, scores = sim().simulate_iid_stream(10000, 0.05)
    assert labels.sum() == 500
    assert np.all(scores[labels == 0] <= 1) and np.all(scores[labels == 1] >= 2)


def test_burst_has_target_rate_and_long_runs():
    labels, _ = sim(3).simulate_burst_stream(200000, 0.05, mean_block_length=20)
    assert labels.mean() == pytest.approx(0.05, abs=0.01)
    runs = np.diff(np.flatnonzero(np.diff(np.r_[0, labels, 0])))[::2]
    assert runs.mean() == pytest.approx(20, rel=0.2)


def test_drift_increases_scores_over_time():
    labels, scores = sim(1).simulate_drift_stream(10000, 0.01, drift_slope_sigma=2.0)
    early, late = scores[:1000][labels[:1000] == 0], scores[-1000:][labels[-1000:] == 0]
    assert late.mean() > early.mean()


def test_empty_pools_rejected():
    with pytest.raises(ValueError):
        ProductionStreamSimulator(np.array([]), np.array([1.0]))
