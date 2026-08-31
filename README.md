# Reliable Industrial Visual Anomaly Detection Benchmark

[![CI Test Suite](https://img.shields.io/badge/pytest-37%20passed-brightgreen.svg)](https://github.com/sengarutk/industrial-defect-anomaly-benchmark)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x-red.svg)](https://pytorch.org/)
[![Coverage](https://img.shields.io/badge/coverage-83%25-brightgreen.svg)](https://github.com/sengarutk/industrial-defect-anomaly-benchmark)

A reproducible research benchmark evaluating **PatchCore-inspired**, **PaDiM-inspired**, and **Reconstruction-based** visual anomaly detectors across five MVTec AD categories, measuring structural localization ($AU\text{-}PRO$ at $\text{FPR} \le 0.30$), synchronized dual-latency hardware profiling, an 18-condition physical corruption stress test, and **operationally-constrained production stream simulations**.

[Benchmark Report](docs/benchmark_report.md) • [Reference Validation](docs/reference-validation.md) • [Operational Summary](results/mvtec_ad/tables/operational_results.md) • [Raw Run Logs (CSV)](results/mvtec_ad/tables/runs_master.csv) • [LaTeX Tables](results/mvtec_ad/tables/)

---

## 🎯 Research Thesis: Beyond AUROC
Standard anomaly detection benchmarks evaluate ranking quality across the entire ROC spectrum ($AUROC$, $AU\text{-}PRO$). In high-throughput industrial manufacturing, inspection viability is governed by **operator alert capacity** ($\le 5\text{ alarms/1,000 items}$) and **asymmetric escape costs** (where a missed defect costs $10\times\text{--}50\times$ more than a false alert). This benchmark implements a high-throughput production stream simulation framework measuring defect recall under strict alarm budgets and asymmetric cost-weighted error ($\text{CWE}$).

---

## 📊 Operational Benchmark Results (Constrained Alarm Budgets & Asymmetric Costs)

Evaluation under strict operator alarm budget ($\le 5\text{ alarms}/1,000$ parts) and $10\times$ missed-defect cost asymmetry ($r = 10$) across 45 multi-seed production streams (95% Bootstrap Confidence Intervals):

| Category | Method | TPR @ 5 Alarms/1k ($\uparrow$) | Missed Defects / 1k ($\downarrow$) | Cost-Weighted Error ($r=10$) ($\downarrow$) | Overload Prob. $P(\text{Overload})$ ($\downarrow$) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Bottle** | **PatchCore** | **$1.000\ [1.000, 1.000]$** | **$0.0\ [0.0, 0.0]$** | **$0.0116\ [0.0116, 0.0116]$** | **$0.367$** |
| | PaDiM | $0.955\ [0.955, 0.955]$ | $30.3\ [30.3, 30.3]$ | $0.2442\ [0.2442, 0.2442]$ | $0.533$ |
| | ConvAutoencoder | $0.066\ [0.061, 0.076]$ | $929.3\ [924.2, 939.4]$ | $7.1434\ [7.1047, 7.2209]$ | $0.033$ |
| **Cable** | **PatchCore** | **$0.859\ [0.859, 0.859]$** | **$130.4\ [130.4, 130.4]$** | **$0.8067\ [0.8067, 0.8067]$** | **$0.000$** |
| | PaDiM | $0.217\ [0.217, 0.217]$ | $750.0\ [750.0, 750.0]$ | $4.6067\ [4.6067, 4.6067]$ | $0.000$ |
| | ConvAutoencoder | $0.011\ [0.011, 0.011]$ | $989.1\ [989.1, 989.1]$ | $6.0733\ [6.0733, 6.0733]$ | $0.000$ |
| **Carpet** | **PatchCore** | **$0.955\ [0.955, 0.955]$** | **$44.9\ [44.9, 44.9]$** | **$0.3504\ [0.3504, 0.3504]$** | **$0.000$** |
| | PaDiM | $0.798\ [0.798, 0.798]$ | $191.0\ [191.0, 191.0]$ | $1.4615\ [1.4615, 1.4615]$ | $0.000$ |
| | ConvAutoencoder | $0.000\ [0.000, 0.000]$ | $1000.0\ [1000.0, 1000.0]$ | $7.6154\ [7.6154, 7.6154]$ | $0.000$ |
| **Hazelnut** | **PatchCore** | **$0.986\ [0.986, 0.986]$** | **$14.3\ [14.3, 14.3]$** | **$0.1000\ [0.1000, 0.1000]$** | **$0.000$** |
| | PaDiM | $0.071\ [0.071, 0.071]$ | $928.6\ [928.6, 928.6]$ | $5.9182\ [5.9182, 5.9182]$ | $0.000$ |
| | ConvAutoencoder | $0.352\ [0.343, 0.357]$ | $647.6\ [642.9, 657.1]$ | $4.1303\ [4.1000, 4.1909]$ | $0.000$ |
| **Metal Nut** | **PatchCore** | **$0.978\ [0.978, 0.978]$** | **$21.5\ [21.5, 21.5]$** | **$0.1826\ [0.1826, 0.1826]$** | **$0.100$** |
| | PaDiM | $0.602\ [0.602, 0.602]$ | $365.6\ [365.6, 365.6]$ | $2.9652\ [2.9652, 2.9652]$ | $0.133$ |
| | ConvAutoencoder | $0.000\ [0.000, 0.000]$ | $1000.0\ [1000.0, 1000.0]$ | $8.0957\ [8.0957, 8.0957]$ | $0.000$ |

*Wilcoxon Signed-Rank Test: PatchCore achieves statistically significant superiority over PaDiM ($W=0.0, p = 5.68\times 10^{-14}$) and Autoencoder ($W=0.0, p = 5.68\times 10^{-14}$).*

---

## ⚡ Synchronized Hardware Latency & Memory Profile ($B=1$)

| Method | Backbone | $P50\ T_{\text{model}}$ (ms) | $\text{FPS}_{\text{model}}$ | $P50\ T_{\text{e2e}}$ (ms) | $\text{FPS}_{\text{e2e}}$ | Peak VRAM (MB) |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **PatchCore** | ResNet-18 | 12.65 ms | 81.2 | 38.78 ms | 29.3 | **247.1 MB** |
| **PaDiM** | ResNet-18 | 6.27 ms | 160.1 | 32.72 ms | 35.3 | 306.5 MB |
| **ConvAutoencoder** | Conv-Deconv | **5.44 ms** | **185.5** | **31.89 ms** | **36.2** | 215.2 MB |

---

## ⚠️ Limitations & Evaluation Scope

- **Category Scope:** Evaluation covers 5 representative MVTec AD categories (`bottle`, `cable`, `hazelnut`, `metal_nut`, `carpet`) spanning rigid objects, deformable wires, natural shapes, and homogeneous textures.
- **Environmental Proxies:** The 18 corruption conditions serve as synthetic proxies for optical defocus, conveyor motion blur, lighting loss, high-ISO sensor noise, JPEG compression, and downscaling.
- **Hardware Latency Boundary:** Latencies are measured on an NVIDIA RTX GPU at $256 \times 256$ with batch size $B=1$; they reflect PyTorch CUDA execution rather than embedded Jetson / C++ TensorRT deployment.
- **Baseline Implementations:** PatchCore and PaDiM are inspired implementations evaluated under documented settings, not official reproductions.

---

## 🚀 Quick Reproduction

```bash
# 1. Run unit & integration test suite (37 tests)
pytest --cov=src --cov-report=term-missing tests/ -v

# 2. Download verified MVTec AD category data
python scripts/download_dataset.py --categories bottle cable hazelnut metal_nut carpet --max-workers 8

# 3. Execute master benchmark sweep (45 runs) with score caching
python scripts/run_benchmark.py --categories bottle cable hazelnut metal_nut carpet --methods patchcore padim autoencoder --seeds 42 123 2026 --save-scores

# 4. Run operational evaluation & generate publication assets
python scripts/run_operational_eval.py --scores-dir results/mvtec_ad/scores --output-dir results/mvtec_ad
python scripts/generate_operational_plots.py --scores-dir results/mvtec_ad/scores --output-dir results/mvtec_ad/figures/operational
python scripts/generate_plots.py --tables-dir results/mvtec_ad/tables --output-dir results/mvtec_ad/figures
python scripts/generate_report.py --tables-dir results/mvtec_ad/tables --docs-dir docs
```

---

## 📜 Citation

```bibtex
@software{sengar2026industrial,
  author = {Sengar, Utkarsh},
  title = {Industrial Defect Anomaly Benchmark: Multi-Method Evaluation and Operational Production Simulation},
  year = {2026},
  url = {https://github.com/sengarutk/industrial-defect-anomaly-benchmark}
}
```

## 📄 License
This project is licensed under the [MIT License](LICENSE).