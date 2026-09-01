# PatchCore GPU Coreset Systems Benchmark (N=10,000, D=128, Ratio=0.10)

| method                 |   runtime_sec |   speedup_vs_cpu |   peak_vram_mb |   coverage_radius |
|:-----------------------|--------------:|-----------------:|---------------:|------------------:|
| cpu_sequential_greedy  |   1.28852     |           1      |         0      |           14.9339 |
| gpu_unbatched_greedy   |   0.124273    |          10.3684 |        14.7642 |           14.9248 |
| gpu_batched_vectorized |   0.00749657  |         171.881  |        14.5635 |           14.9247 |
| random_subsampling     |   0.000453343 |        2842.26   |         0      |           15.3702 |