# PatchCore GPU Coreset Systems Benchmark (N=10,000, D=128, Ratio=0.10)

| method                 |   runtime_sec |   speedup_vs_cpu |   peak_vram_mb |   coverage_radius |
|:-----------------------|--------------:|-----------------:|---------------:|------------------:|
| cpu_sequential_greedy  |   1.58214     |           1      |         0      |           14.9339 |
| gpu_unbatched_greedy   |   0.120063    |          13.1776 |        14.7642 |           14.9248 |
| gpu_batched_vectorized |   0.00806023  |         196.289  |        14.5635 |           14.9247 |
| random_subsampling     |   0.000525638 |        3009.94   |         0      |           15.3702 |