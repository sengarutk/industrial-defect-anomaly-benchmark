# Environment

Captured 2026-10-26 22:25 UTC by `scripts/collect_environment.py`.

* Platform: Linux-6.6.87.2-microsoft-standard-WSL2-x86_64-with-glibc2.17
* CPU: x86_64 (16 logical cores)
* Python: 3.8.20 (/home/sengar/miniconda3/envs/vmunet/bin/python)
* torch 1.13.0+cu117, torchvision 0.14.0+cu117, numpy 1.24.4, scipy 1.10.1, scikit-learn 1.3.2, pandas 2.0.3, opencv 5.0.0, matplotlib 3.7.5
* CUDA available: False
* Repository commit: ef83085

Note: during the October 2026 rerun, GPU access was blocked by the host OS inside WSL (NVML: "GPU access blocked by the operating system"), so the accuracy and robustness sweep ran on CPU. Each run record in results/mvtec_ad/runs/ stores the device it used. Latency (scripts/profile_latency.py) and the coreset systems benchmark (scripts/run_coreset_benchmark.py) require CUDA and are reported only when measured on a GPU.
