"""Latency and memory profiling at batch size 1.

T_model: preprocessed tensor already on the device -> predict() returns scores
         and maps (includes the CPU-side Gaussian smoothing of each method).
T_e2e:   PNG file on disk -> decode -> resize -> normalise -> host-to-device
         copy -> predict(). Files are cycled through the real test images; the
         OS page cache is warm after the first pass, so disk I/O is not cold.
"""
import time
from typing import Dict, List, Sequence

import numpy as np
import torch

from src.mvtec import load_rgb, normalize_image


def _percentiles(latencies_ms: List[float]) -> Dict[str, float]:
    arr = np.asarray(latencies_ms, dtype=np.float64)
    return {"p50": float(np.percentile(arr, 50)), "p95": float(np.percentile(arr, 95)),
            "mean": float(arr.mean()), "std": float(arr.std())}


class LatencyProfiler:
    def __init__(self, warmup_runs: int = 50, active_runs: int = 300, e2e_runs: int = 100):
        self.warmup_runs = warmup_runs
        self.active_runs = active_runs
        self.e2e_runs = e2e_runs

    @staticmethod
    def _sync(device: torch.device) -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def profile_model(self, model, sample_input: torch.Tensor, device: str) -> Dict[str, float]:
        dev = torch.device(device)
        x = sample_input.to(dev)
        with torch.no_grad():
            for _ in range(self.warmup_runs):
                model.predict(x)
            self._sync(dev)
            if dev.type == "cuda":
                torch.cuda.reset_peak_memory_stats(dev)
            lat = []
            for _ in range(self.active_runs):
                self._sync(dev)
                t0 = time.perf_counter()
                model.predict(x)
                self._sync(dev)
                lat.append((time.perf_counter() - t0) * 1000.0)
        p = _percentiles(lat)
        peak = torch.cuda.max_memory_allocated(dev) / 2 ** 20 if dev.type == "cuda" else float("nan")
        return {"p50_model_ms": p["p50"], "p95_model_ms": p["p95"], "fps_model": 1000.0 / p["p50"],
                "peak_vram_mb": float(peak)}

    def profile_e2e(self, model, image_paths: Sequence[str], device: str, img_size: int = 256) -> Dict[str, float]:
        dev = torch.device(device)

        def run(path: str) -> None:
            x = normalize_image(load_rgb(path, img_size)).unsqueeze(0).to(dev)
            model.predict(x)
            self._sync(dev)

        with torch.no_grad():
            for i in range(min(10, self.e2e_runs)):
                run(image_paths[i % len(image_paths)])
            lat = []
            for i in range(self.e2e_runs):
                t0 = time.perf_counter()
                run(image_paths[i % len(image_paths)])
                lat.append((time.perf_counter() - t0) * 1000.0)
        p = _percentiles(lat)
        return {"p50_e2e_ms": p["p50"], "p95_e2e_ms": p["p95"], "fps_e2e": 1000.0 / p["p50"]}
