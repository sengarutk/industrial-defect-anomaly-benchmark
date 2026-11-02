"""Measured coreset-selection benchmark on real PatchCore patch features.

Every cell is timed; nothing is extrapolated. Compared implementations:
  cpu_numpy_greedy   - exact greedy k-center, NumPy (reference)
  gpu_torch_greedy   - exact greedy k-center, the implementation used by PatchCore
  gpu_topk_batched   - approximate variant that adds 50 points per step
  random             - uniform random subset
For each we report the median wall time over `repeats` runs, coverage radius (k-center objective) and, for
the exact GPU version, the fraction of selected indices identical to the NumPy
reference started from the same point.
"""
from typing import Dict, List, Sequence

import numpy as np
import torch

from src.methods.coreset import (batched_topk_kcenter, coverage_radius, greedy_kcenter, greedy_kcenter_numpy,
                                 random_projection, timed)


def _median_time(repeats: int, fn, *args, device=None):
    """Runs fn `repeats` times; returns (first result, median wall time). Selection is deterministic."""
    out, times = None, []
    for i in range(repeats):
        res, t = timed(fn, *args, device=device)
        out = res if i == 0 else out
        times.append(t)
    return out, float(np.median(times))


def run_coreset_benchmark(features: torch.Tensor, sample_sizes: Sequence[int], dims: Sequence[int],
                          ratio: float = 0.10, seed: int = 0, device: str = "cuda", repeats: int = 3) -> List[Dict]:
    dev = torch.device(device)
    pool = features.float().cpu()
    rows = []
    # warm up CUDA kernels so the first timed cell does not include context initialisation
    warm = torch.randn(2000, 64, device=dev)
    greedy_kcenter(warm, 50, 0)
    batched_topk_kcenter(warm, 50, 0)
    for d in dims:
        projected = random_projection(pool, d, seed=seed) if pool.shape[1] > d else pool
        for n in sample_sizes:
            if n > projected.shape[0]:
                raise ValueError(f"Requested N={n} but only {projected.shape[0]} features are available")
            rng = np.random.RandomState(seed + n + d)
            subset = projected[torch.from_numpy(rng.choice(projected.shape[0], n, replace=False))].contiguous()
            m = max(1, int(n * ratio))
            start = int(rng.randint(n))
            x_np = subset.numpy()
            x_gpu = subset.to(dev)

            idx_cpu, t_cpu = _median_time(repeats, greedy_kcenter_numpy, x_np, m, start)
            if dev.type == "cuda":
                torch.cuda.reset_peak_memory_stats(dev)
            idx_gpu, t_gpu = _median_time(repeats, greedy_kcenter, x_gpu, m, start, device=dev)
            vram = torch.cuda.max_memory_allocated(dev) / 2 ** 20 if dev.type == "cuda" else float("nan")
            idx_topk, t_topk = _median_time(repeats, batched_topk_kcenter, x_gpu, m, start, 50, device=dev)
            idx_rand = torch.from_numpy(rng.choice(n, m, replace=False))

            cpu_t = torch.from_numpy(idx_cpu)
            agreement = float(np.mean(idx_cpu == idx_gpu.cpu().numpy()))
            rows.append({
                "num_patches_N": n, "feature_dim_D": d, "coreset_M": m, "timing_repeats": repeats,
                "time_cpu_numpy_s": t_cpu, "time_gpu_greedy_s": t_gpu, "time_gpu_topk_s": t_topk,
                "speedup_gpu_greedy": t_cpu / t_gpu, "speedup_gpu_topk": t_cpu / t_topk,
                "radius_cpu_numpy": coverage_radius(x_gpu, cpu_t),
                "radius_gpu_greedy": coverage_radius(x_gpu, idx_gpu),
                "radius_gpu_topk": coverage_radius(x_gpu, idx_topk),
                "radius_random": coverage_radius(x_gpu, idx_rand),
                "index_agreement_gpu_vs_cpu": agreement,
                "peak_vram_mb_gpu_greedy": vram,
            })
            print(f"  N={n:>6} D={d:>3}: cpu={t_cpu:.2f}s gpu={t_gpu:.3f}s topk={t_topk:.3f}s "
                  f"agree={agreement:.3f}", flush=True)
    return rows
