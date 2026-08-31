# PatchCore GPU Coreset Systems Benchmark (N=10,000, D=128, Ratio=0.10)

| method                 |   runtime_sec |   speedup_vs_cpu |   peak_vram_mb |   coverage_radius |
|:-----------------------|--------------:|-----------------:|---------------:|------------------:|
| cpu_sequential_greedy  |      1.55192  |         1        |         0      |           14.9339 |
| gpu_unbatched_greedy   |      1.61682  |         0.959857 |        14.7642 |           14.9248 |
| gpu_batched_vectorized |      6.92866  |         0.223985 |        17.5757 |           14.9247 |
| random_subsampling     |      0.155713 |         9.96651  |         0      |           15.3702 |