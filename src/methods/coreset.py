"""Greedy k-center coreset selection (Sener & Savarese, 2018; used by PatchCore).

`greedy_kcenter` is the exact sequential greedy algorithm written with
device-agnostic torch ops: it runs unchanged on CPU or GPU and keeps the
argmax index on the device, so there is no host synchronisation per step.
`greedy_kcenter_numpy` is an independent NumPy implementation of the same
algorithm used as the CPU baseline and as a correctness reference.
`batched_topk_kcenter` is an *approximation* that adds the k farthest points
per step; it is only used in the systems ablation and is not the greedy rule.
"""
import time
from typing import Optional, Tuple

import numpy as np
import torch


def random_projection(features: torch.Tensor, out_dim: int, seed: int) -> torch.Tensor:
    """Gaussian Johnson-Lindenstrauss projection to out_dim (identity if already small enough)."""
    d = features.shape[1]
    if d <= out_dim:
        return features
    g = torch.Generator(device="cpu").manual_seed(seed)
    proj = torch.randn(d, out_dim, generator=g) / np.sqrt(out_dim)
    return features @ proj.to(features.device, features.dtype)


def greedy_kcenter(features: torch.Tensor, m: int, start_idx: int) -> torch.Tensor:
    """Exact greedy k-center. Returns LongTensor of m selected row indices (on features.device)."""
    n = features.shape[0]
    m = min(m, n)
    sq_norm = (features * features).sum(dim=1)
    idx = torch.empty(m, dtype=torch.long, device=features.device)
    current = torch.tensor([start_idx], dtype=torch.long, device=features.device)
    idx[0] = current[0]
    min_d = torch.full((n,), float("inf"), dtype=features.dtype, device=features.device)
    for i in range(1, m + 1):
        q = features.index_select(0, current)
        d = (sq_norm - 2.0 * (features @ q.T).squeeze(1) + (q * q).sum()).clamp_(min=0.0)
        min_d = torch.minimum(min_d, d)
        if i == m:
            break
        current = torch.argmax(min_d).view(1)
        idx[i] = current[0]
    return idx


def greedy_kcenter_numpy(features: np.ndarray, m: int, start_idx: int) -> np.ndarray:
    """Reference sequential greedy k-center in NumPy (same update rule as greedy_kcenter)."""
    n = features.shape[0]
    m = min(m, n)
    sq_norm = np.einsum("ij,ij->i", features, features)
    idx = np.empty(m, dtype=np.int64)
    idx[0] = start_idx
    current = start_idx
    min_d = np.full(n, np.inf, dtype=features.dtype)
    for i in range(1, m + 1):
        q = features[current]
        d = np.maximum(sq_norm - 2.0 * (features @ q) + q @ q, 0.0)
        np.minimum(min_d, d, out=min_d)
        if i == m:
            break
        current = int(np.argmax(min_d))
        idx[i] = current
    return idx


def batched_topk_kcenter(features: torch.Tensor, m: int, start_idx: int, batch_k: int = 50) -> torch.Tensor:
    """Approximate k-center: adds the batch_k farthest points per step (no 2-approximation guarantee)."""
    n = features.shape[0]
    m = min(m, n)
    sq_norm = (features * features).sum(dim=1)
    selected = [torch.tensor([start_idx], device=features.device)]
    q = features[start_idx:start_idx + 1]
    min_d = (sq_norm - 2.0 * (features @ q.T).squeeze(1) + (q * q).sum()).clamp_(min=0.0)
    count = 1
    while count < m:
        k = min(batch_k, m - count)
        top = torch.topk(min_d, k=k).indices
        selected.append(top)
        count += k
        q = features.index_select(0, top)
        d = (sq_norm.unsqueeze(1) - 2.0 * features @ q.T + (q * q).sum(dim=1).unsqueeze(0)).clamp_(min=0.0)
        min_d = torch.minimum(min_d, d.min(dim=1).values)
    return torch.cat(selected)[:m]


def coverage_radius(features: torch.Tensor, centers_idx: torch.Tensor, chunk: int = 8192) -> float:
    """max_x min_c ||x - c||: the k-center objective (lower is better)."""
    centers = features.index_select(0, centers_idx.to(features.device))
    worst = 0.0
    with torch.no_grad():
        for i in range(0, features.shape[0], chunk):
            d = torch.cdist(features[i:i + chunk], centers)
            worst = max(worst, float(d.min(dim=1).values.max()))
    return worst


def timed(fn, *args, device: Optional[torch.device] = None, **kwargs) -> Tuple[object, float]:
    """Runs fn and returns (result, wall seconds), synchronising CUDA around the call."""
    if device is not None and device.type == "cuda":
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    if device is not None and device.type == "cuda":
        torch.cuda.synchronize(device)
    return out, time.perf_counter() - t0
