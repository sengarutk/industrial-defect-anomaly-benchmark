import numpy as np
import torch

from src.experiments.coreset_benchmark import run_coreset_benchmark
from src.methods.coreset import (batched_topk_kcenter, coverage_radius, greedy_kcenter, greedy_kcenter_numpy,
                                 random_projection)


def clustered(n=600, d=16, seed=0):
    rng = np.random.RandomState(seed)
    centers = rng.randn(8, d) * 5
    return (centers[rng.randint(0, 8, n)] + rng.randn(n, d)).astype(np.float64)


def test_torch_greedy_matches_numpy_reference():
    x = clustered()
    ref = greedy_kcenter_numpy(x, 60, start_idx=3)
    got = greedy_kcenter(torch.from_numpy(x), 60, start_idx=3).numpy()
    np.testing.assert_array_equal(ref, got)
    assert len(set(got.tolist())) == 60


def test_greedy_beats_random_and_topk_is_only_approximate():
    x = torch.from_numpy(clustered(seed=1))
    g = greedy_kcenter(x, 40, 0)
    t = batched_topk_kcenter(x, 40, 0, batch_k=10)
    rnd = torch.from_numpy(np.random.RandomState(0).choice(len(x), 40, replace=False))
    assert len(torch.unique(t)) == 40
    assert coverage_radius(x, g) <= coverage_radius(x, rnd)
    assert coverage_radius(x, g) <= coverage_radius(x, t) + 1e-9


def test_random_projection_is_seeded():
    x = torch.randn(10, 64)
    assert torch.equal(random_projection(x, 8, 1), random_projection(x, 8, 1))
    assert not torch.equal(random_projection(x, 8, 1), random_projection(x, 8, 2))
    assert random_projection(x, 128, 1) is x


def test_benchmark_runs_on_cpu_and_reports_measured_cells():
    feats = torch.from_numpy(clustered(n=800, d=32).astype(np.float32))
    rows = run_coreset_benchmark(feats, [200, 400], [16, 32], ratio=0.1, seed=0, device="cpu")
    assert len(rows) == 4
    for r in rows:
        assert r["time_cpu_numpy_s"] > 0 and r["time_gpu_greedy_s"] > 0
        assert r["radius_gpu_greedy"] <= r["radius_random"] + 1e-6
