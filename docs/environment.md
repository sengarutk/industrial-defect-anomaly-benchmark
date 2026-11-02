# Environment

Captured 2026-11-02 10:00 UTC by `scripts/collect_environment.py`.

* Platform: Linux-6.6.87.2-microsoft-standard-WSL2-x86_64-with-glibc2.17
* CPU: x86_64 (16 logical cores)
* Python: 3.8.20 (/home/sengar/miniconda3/envs/vmunet/bin/python)
* torch 1.13.0+cu117, torchvision 0.14.0+cu117, numpy 1.24.4, scipy 1.10.1, scikit-learn 1.3.2, pandas 2.0.3, opencv 5.0.0, matplotlib 3.7.5
* CUDA available: True
* GPU: NVIDIA GeForce RTX 4050 Laptop GPU, 6140 MiB; CUDA runtime 11.7; cuDNN 8500
* Repository commit: 8d67860

Provenance of the published results: GPU access was blocked by the host OS during part of the rerun, so 44 of the 63 benchmark runs executed on CPU and 19 on the GPU listed above (each record in results/mvtec_ad/runs/ stores its device and commit; see docs/methods.md). Latency and the coreset-selection benchmark were measured on this GPU.
