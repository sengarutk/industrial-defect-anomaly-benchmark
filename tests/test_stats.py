import numpy as np
import pytest

from src.metrics.stats import (bootstrap_ci, hierarchical_bootstrap_ci, holm_bonferroni, paired_wilcoxon,
                               validate_bootstrap_ci_coverage)


def test_wilcoxon_small_n_cannot_reach_significance():
    """With 5 independent units the smallest two-sided exact p-value is 2/2^5 = 0.0625."""
    a, b = np.array([0.1, 0.2, 0.3, 0.4, 0.5]), np.array([1, 2, 3, 4, 5.0])
    res = paired_wilcoxon(a, b)
    assert res["p_value"] == pytest.approx(0.0625)
    assert res["min_attainable_p"] == pytest.approx(0.0625)
    assert res["rank_biserial"] == -1.0 and res["hodges_lehmann"] < 0


def test_wilcoxon_all_ties():
    res = paired_wilcoxon(np.ones(4), np.ones(4))
    assert res["p_value"] == 1.0 and res["n_nonzero"] == 0


def test_holm_step_down():
    out = holm_bonferroni({"a": 0.001, "b": 0.015, "c": 0.04, "d": 0.25})
    assert out["a"]["reject"] and out["a"]["adjusted_p"] == pytest.approx(0.004)
    assert out["b"]["adjusted_p"] == pytest.approx(0.045) and out["b"]["reject"]
    assert not out["c"]["reject"] and not out["d"]["reject"]
    adj = [out[k]["adjusted_p"] for k in "abcd"]
    assert adj == sorted(adj)


def test_hierarchical_bootstrap_resamples_items_jointly():
    rng = np.random.RandomState(0)
    recs = [{"scores": rng.randn(50) + i, "labels": (rng.rand(50) < 0.5).astype(int)} for i in range(3)]
    res = hierarchical_bootstrap_ci(recs, lambda r: float(np.mean([x["scores"].mean() for x in r])),
                                    ["scores", "labels"], n_resamples=300, seed=1)
    assert res["ci_low"] < res["estimate"] < res["ci_high"]


def test_bootstrap_ci_and_coverage():
    m, lo, hi = bootstrap_ci(np.arange(10.0), n_bootstraps=500)
    assert lo < m < hi
    cov = validate_bootstrap_ci_coverage(n_simulations=150, sample_size=60, n_bootstraps=300, seed=3)
    assert 0.88 <= cov <= 0.99


def test_stratified_resampling_keeps_class_counts():
    rec = [{"scores": np.arange(10.0), "labels": np.r_[np.zeros(2, int), np.ones(8, int)]}]
    seen = []
    hierarchical_bootstrap_ci(rec, lambda r: seen.append(int((r[0]["labels"] == 0).sum())) or 0.0,
                              ["scores", "labels"], n_resamples=200, seed=0, strata_key="labels")
    assert set(seen) == {2}


def test_wilcoxon_with_zero_difference_uses_exact_distribution():
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    b = np.array([1.0, 2.5, 3.6, 4.7, 5.8])  # one zero, four negative differences
    res = paired_wilcoxon(a, b)
    assert res["n_nonzero"] == 4 and res["p_value"] == pytest.approx(0.125)
