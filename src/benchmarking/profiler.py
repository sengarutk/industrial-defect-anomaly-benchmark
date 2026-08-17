import time
from typing import Callable, Any, Dict
import numpy as np
import torch


class CUDAPerformanceProfiler:
    def __init__(self, warmup_runs: int = 50, active_runs: int = 300):
        self.warmup_runs = warmup_runs
        self.active_runs = active_runs

    def profile(self, fn: Callable[[torch.Tensor], Any], sample_input: torch.Tensor) -> Dict[str, float]:
        """
        Profiles inference latency and peak VRAM allocation under strict GPU synchronization.
        """
        is_cuda = sample_input.is_cuda and torch.cuda.is_available()
        device = sample_input.device if is_cuda else None

        if is_cuda:
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)

        # Warm-up phase
        for _ in range(self.warmup_runs):
            _ = fn(sample_input)

        if is_cuda:
            torch.cuda.synchronize(device)

        # Active timing phase
        latencies_ms = []
        if is_cuda:
            start_events = [torch.cuda.Event(enable_timing=True) for _ in range(self.active_runs)]
            end_events = [torch.cuda.Event(enable_timing=True) for _ in range(self.active_runs)]

            for i in range(self.active_runs):
                start_events[i].record()
                _ = fn(sample_input)
                end_events[i].record()

            torch.cuda.synchronize(device)
            latencies_ms = [float(s.elapsed_time(e)) for s, e in zip(start_events, end_events)]
        else:
            for _ in range(self.active_runs):
                t0 = time.perf_counter()
                _ = fn(sample_input)
                t1 = time.perf_counter()
                latencies_ms.append(float((t1 - t0) * 1000.0))

        latencies = np.array(latencies_ms)
        p50 = float(np.percentile(latencies, 50))
        peak_vram_mb = float(torch.cuda.max_memory_allocated(device) / (1024 ** 2)) if is_cuda else 0.0

        return {
            "p50_ms": p50,
            "p90_ms": float(np.percentile(latencies, 90)),
            "p95_ms": float(np.percentile(latencies, 95)),
            "p99_ms": float(np.percentile(latencies, 99)),
            "mean_ms": float(np.mean(latencies)),
            "std_ms": float(np.std(latencies)),
            "fps": float(1000.0 / max(p50, 1e-6)),
            "peak_vram_mb": peak_vram_mb,
        }
