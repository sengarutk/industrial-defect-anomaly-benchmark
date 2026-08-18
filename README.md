# Reliable Industrial Visual Anomaly Detection Benchmark

[![CI Test Suite](https://img.shields.io/badge/pytest-22%20passed-brightgreen.svg)](https://github.com/sengarutk/industrial-defect-anomaly-benchmark)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x-red.svg)](https://pytorch.org/)
[![Coverage](https://img.shields.io/badge/coverage-100%25-brightgreen.svg)](https://github.com/sengarutk/industrial-defect-anomaly-benchmark)

A reproducible PyTorch benchmark evaluating **PatchCore-inspired**, **PaDiM-inspired**, and **Reconstruction-based** visual anomaly detectors across five MVTec AD categories, measuring structural localization ($AU\text{-}PRO$ at $\text{FPR} \le 0.30$), synchronized dual-latency hardware profiling, and an 18-condition physical corruption stress test.

[Benchmark Report](docs/benchmark_report.md) • [Reference Validation](docs/reference-validation.md) • [Raw Run Logs (CSV)](results/mvtec_ad/tables/runs_master.csv) • [Multi-Seed Summary (CSV)](results/mvtec_ad/tables/summary_multiseed.csv) • [LaTeX Tables](results/mvtec_ad/tables/)

---

## ⚠️ Limitations & Evaluation Scope

- **Category Scope:** Evaluation covers 5 representative MVTec AD categories (`bottle`, `cable`, `hazelnut`, `metal_nut`, `carpet`), spanning rigid objects, deformable wires, natural shapes, and homogeneous textures—not the full 15 categories.
- **Environmental Proxies:** The 18 corruption conditions serve as synthetic proxies for optical defocus, conveyor motion blur, lighting loss, high-ISO sensor noise, JPEG compression, and downscaling—not live production-line sensors.
- **Hardware Latency Boundary:** Latencies are measured on an NVIDIA RTX GPU at $256 \times 256$ with batch size $B=1$; they reflect PyTorch CUDA execution rather than embedded Jetson / C++ TensorRT deployment.
- **Baseline Implementations:** PatchCore and PaDiM are inspired implementations evaluated under documented settings, not official reproductions.
- **Probability Calibration:** Raw anomaly scores are uncalibrated rank diagnostics and should not be interpreted as posterior defect probabilities.

---

## 🔬 Experimental Protocol at a Glance

- **Dataset:** MVTec AD mirror (`foersben/mvtec-ad` on Hugging Face; $1,324$ train images, $575$ test images, $400$ ground-truth defect masks).
- **Training Protocol:** Unsupervised (nominal/defect-free images only).
- **Resolution & Preprocessing:** $256 \times 256$, ImageNet standardization ($\mu=[0.485, 0.456, 0.406], \sigma=[0.229, 0.224, 0.225]$).
- **Seeds & Multi-Seed Sweeps:** 3 deterministic seeds (`42`, `123`, `2026`) across 5 categories and 3 methods ($45$ total benchmark runs).
- **Hardware Latency Profiling:** Synchronized CUDA events ($50$ warmup runs, $300$ active runs, batch size $B=1$).
- **Robustness Stress Test:** 6 camera/lighting corruption types $\times$ 3 severity levels ($18$ conditions), evaluating Mean Performance Change ($\text{MPC}$) and Non-Negative Mean Robustness Degradation ($\text{MRD}$).

---

## 📊 Master Benchmark Results (Multi-Seed MVTec AD)

| Category | Method | Image AUROC ($\uparrow$) | Pixel AUROC ($\uparrow$) | Localization AU-PRO ($\uparrow$) | P50 $T_{\text{model}}$ (ms) | P50 $T_{\text{e2e}}$ (ms) | Peak VRAM (MB) | Non-Negative MRD ($\downarrow$) |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **bottle** | **PatchCore** | **$1.0000 \pm 0.000$** | **$0.9617 \pm 0.000$** | **$0.8725 \pm 0.000$** | 10.94 ms | 29.89 ms | **205.9 MB** | **$0.0284 \pm 0.001$** |
| **bottle** | **PaDiM** | $0.9970 \pm 0.000$ | $0.9548 \pm 0.000$ | $0.8666 \pm 0.000$ | 6.25 ms | 25.63 ms | 298.3 MB | $0.0977 \pm 0.002$ |
| **bottle** | **ConvAutoencoder** | $0.5053 \pm 0.008$ | $0.7081 \pm 0.038$ | $0.4615 \pm 0.007$ | **4.80 ms** | **24.53 ms** | 215.0 MB | $0.0000 \pm 0.000$ |
| **cable** | **PatchCore** | **$0.9903 \pm 0.000$** | **$0.9858 \pm 0.000$** | **$0.9427 \pm 0.000$** | 11.52 ms | 44.49 ms | **218.8 MB** | **$0.0535 \pm 0.000$** |
| **cable** | **PaDiM** | $0.8617 \pm 0.000$ | $0.9683 \pm 0.000$ | $0.9076 \pm 0.000$ | 5.95 ms | 40.02 ms | 300.3 MB | **$0.0238 \pm 0.000$** |
| **cable** | **ConvAutoencoder** | $0.4516 \pm 0.005$ | $0.5709 \pm 0.020$ | $0.3810 \pm 0.028$ | **6.08 ms** | **40.02 ms** | 216.3 MB | $0.0000 \pm 0.000$ |
| **carpet** | **PatchCore** | **$0.9844 \pm 0.000$** | **$0.9896 \pm 0.000$** | **$0.9492 \pm 0.000$** | 12.82 ms | 45.67 ms | **258.1 MB** | **$0.0378 \pm 0.000$** |
| **carpet** | **PaDiM** | $0.9639 \pm 0.000$ | $0.9874 \pm 0.000$ | $0.9549 \pm 0.000$ | 6.47 ms | 38.73 ms | 309.1 MB | $0.0745 \pm 0.001$ |
| **carpet** | **ConvAutoencoder** | $0.3904 \pm 0.005$ | $0.7588 \pm 0.009$ | $0.4523 \pm 0.021$ | **5.83 ms** | **38.64 ms** | 214.9 MB | $0.0000 \pm 0.000$ |
| **hazelnut** | **PatchCore** | **$0.9996 \pm 0.000$** | **$0.9871 \pm 0.000$** | **$0.9188 \pm 0.000$** | 16.36 ms | 47.15 ms | 336.8 MB | **$0.0046 \pm 0.000$** |
| **hazelnut** | **PaDiM** | $0.8404 \pm 0.000$ | $0.9829 \pm 0.000$ | $0.9042 \pm 0.000$ | 6.32 ms | 37.34 ms | **325.5 MB** | $0.0000 \pm 0.000$ |
| **hazelnut** | **ConvAutoencoder** | $0.7354 \pm 0.048$ | $0.9766 \pm 0.005$ | $0.9000 \pm 0.013$ | **6.15 ms** | **36.13 ms** | 214.9 MB | $0.0000 \pm 0.000$ |
| **metal_nut** | **PatchCore** | **$0.9961 \pm 0.000$** | **$0.9840 \pm 0.000$** | **$0.9371 \pm 0.000$** | 11.61 ms | 26.69 ms | **216.0 MB** | **$0.0604 \pm 0.001$** |
| **metal_nut** | **PaDiM** | $0.9712 \pm 0.000$ | $0.9692 \pm 0.000$ | $0.9173 \pm 0.000$ | 6.37 ms | 21.88 ms | 299.7 MB | $0.1441 \pm 0.004$ |
| **metal_nut** | **ConvAutoencoder** | $0.3047 \pm 0.050$ | $0.6300 \pm 0.111$ | $0.3650 \pm 0.079$ | **4.36 ms** | **20.15 ms** | 215.4 MB | $0.0000 \pm 0.000$ |

*Note: Per-corruption breakdowns and signed performance changes (MPC) are detailed in `docs/benchmark_report.md`.*

---

## ⚡ Synchronized Hardware Latency & Memory Profile ($B=1$)

| Method | Backbone | $P50\ T_{\text{model}}$ (ms) | $\text{FPS}_{\text{model}}$ | $P50\ T_{\text{e2e}}$ (ms) | $\text{FPS}_{\text{e2e}}$ | Peak VRAM (MB) |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **PatchCore** | ResNet-18 | 12.65 ms | 81.2 | 38.78 ms | 29.3 | **247.1 MB** |
| **PaDiM** | ResNet-18 | 6.27 ms | 160.1 | 32.72 ms | 35.3 | 306.5 MB |
| **ConvAutoencoder** | Conv-Deconv | **5.44 ms** | **185.5** | **31.89 ms** | **36.2** | 215.2 MB |

---

## 🚀 Quick Reproduction

```bash
# 1. Run unit test suite
pytest --cov=src --cov-report=term-missing tests/ -v

# 2. Download verified MVTec AD category data
python scripts/download_dataset.py --categories bottle cable hazelnut metal_nut carpet --max-workers 8

# 3. Execute benchmark sweep (45 runs)
python scripts/run_benchmark.py --categories bottle cable hazelnut metal_nut carpet --methods patchcore padim autoencoder --seeds 42 123 2026

# 4. Generate publication plots & LaTeX tables
python scripts/generate_plots.py --tables-dir results/mvtec_ad/tables --output-dir results/mvtec_ad/figures
python scripts/generate_report.py --tables-dir results/mvtec_ad/tables --docs-dir docs
```

---

## 📜 Citation

```bibtex
@software{sengar2026industrial,
  author = {Sengar, Utkarsh},
  title = {Industrial Defect Anomaly Benchmark: Multi-Method Evaluation and Robustness Stress-Testing},
  year = {2026},
  url = {https://github.com/sengarutk/industrial-defect-anomaly-benchmark}
}
```

## 📄 License
This project is licensed under the [MIT License](LICENSE).
