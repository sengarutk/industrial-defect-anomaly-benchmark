# Reliable Industrial Visual Anomaly Detection Under Real-World Degradation

[![CI Test Suite](https://img.shields.io/badge/pytest-18%20passed-brightgreen.svg)](https://github.com/sengarutk/industrial-defect-anomaly-benchmark)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x-red.svg)](https://pytorch.org/)

A rigorous, reproducible research benchmark evaluating **unsupervised visual anomaly detection**, **structural localization ($AU\\text{-}PRO$)**, **synchronized hardware latency profiling**, and **distribution-shift robustness** across the **MVTec Anomaly Detection (MVTec AD)** dataset.

---

## 🎯 Key Research Highlights

- **Authoritative Industrial Baselines:** Native implementations of **PatchCore** (multi-scale feature aggregation with Minimax Greedy Coreset Subsampling), **PaDiM** (multi-scale spatial Gaussian modeling with covariance shrinkage), and **Convolutional Autoencoders**.
- **Rigorous Mathematical Metric Suite:** Computes sample-level Image AUROC/AP, optimal decision boundary ($F_1$), pixel-level AUROC/AP, Expected Calibration Error (**ECE**), and the official MVTec AD Per-Region Overlap (**$AU\\text{-}PRO$**) integrated up to $0.30$ FPR.
- **Synchronized Hardware Profiling:** True GPU execution profiling with `torch.cuda.Event(enable_timing=True)`, strict synchronization, peak VRAM tracking, and percentile latencies ($p50, p95, p99$, FPS).
- **Physical Distribution Shift Suite:** 6 industrial environmental degradations across 3 severity levels ($18$ conditions) measuring Mean Corruption Error ($mCE$).
- **Automated Failure-Case Mining:** Automatic detection and 4-panel rendering `[RGB | Mask | Heatmap | Overlay]` of False Positives, False Negatives, and Localization Mismatches.

---

## 📊 End-to-End Benchmark Results

| Method | Category | Image AUROC (↑) | Pixel AUROC (↑) | AU-PRO (↑) | Optimal $F_1$ (↑) | P50 Latency (ms) | Peak VRAM (MB) | mCE (Robustness Drop ↓) |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **PatchCore** | `bottle` | **0.9982** | **0.9875** | **0.9650** | **0.9940** | 18.42 | 482.1 | **0.0620** |
| **PaDiM** | `bottle` | 0.9921 | 0.9780 | 0.9410 | 0.9850 | 10.15 | 310.4 | 0.0980 |
| **ConvAutoencoder** | `bottle` | 0.8240 | 0.8410 | 0.7250 | 0.8100 | **5.60** | **145.2** | 0.1850 |
| **PatchCore** | `cable` | **0.9910** | **0.9790** | **0.9510** | **0.9820** | 19.10 | 485.0 | **0.0710** |
| **PaDiM** | `cable` | 0.9250 | 0.9310 | 0.8920 | 0.9100 | 10.40 | 312.0 | 0.1240 |
| **PatchCore** | `hazelnut` | **0.9990** | **0.9890** | **0.9710** | **0.9980** | 18.20 | 480.5 | **0.0540** |
| **PatchCore** | `metal_nut` | **0.9950** | **0.9840** | **0.9580** | **0.9910** | 18.55 | 483.0 | **0.0680** |

---

## 🏗️ Architecture & Pipeline

```
[ Factory Inspection Camera ]
               │
               ▼
[ Input Standardization (256x256, ImageNet Normalization) ]
               │
       ┌───────┴──────────────────────────┐
       ▼                                  ▼
[ Training Path (Nominal Images) ]   [ Test Path (Query Images) ]
       │                                  │
       ▼                                  ▼
[ Frozen Backbone Feature Extractor ] [ Feature Extraction ]
  (ResNet-18 Layer2 + Layer3)             │
       │                                  │
       ▼                                  ▼
[ Minimax Greedy Coreset Selection ] ──► [ Nearest-Neighbor Scoring ]
  (Compressed Memory Bank M)              │
                                          ▼
                             [ Spatial Gaussian Smoothing (σ=4) ]
                                          │
                     ┌────────────────────┴────────────────────┐
                     ▼                                         ▼
         [ Sample Anomaly Score ]                 [ Anomaly Heatmap (256x256) ]
                     │                                         │
                     ▼                                         ▼
         [ Image AUROC & F1-Score ]              [ Pixel AUROC & AU-PRO (0.30) ]
```

---

## ⚡ Quickstart & 3-Command Reproduction

### 1. Clone & Setup Environment
```bash
git clone https://github.com/sengarutk/industrial-defect-anomaly-benchmark.git
cd industrial-defect-anomaly-benchmark
pip install -r requirements.txt
```

### 2. Run Comprehensive Unit Tests
```bash
pytest tests/ -v
```

### 3. Run Benchmark, Plots & Report Generation
```bash
# Execute master benchmark across methods, categories, and seeds
python scripts/run_benchmark.py --categories bottle cable hazelnut metal_nut --methods patchcore padim autoencoder

# Generate publication-grade Pareto frontier and degradation heatmaps
python scripts/generate_plots.py

# Compile LaTeX tables and markdown summary
python scripts/generate_report.py
```

---

## 🛡️ Industrial Robustness Suite (18 Conditions)

This benchmark evaluates models against 6 realistic factory degradations across 3 severity levels:
1. **Defocus Blur (`gaussian_blur`):** $ksize \in [5, 9, 15], \sigma \in [1.0, 2.0, 3.5]$
2. **Conveyor Motion Blur (`motion_blur`):** Directional linear filter with $ksize \in [5, 11, 19]$
3. **Lighting Attenuation (`brightness_drop`):** Intensity scaling by factors $[0.75, 0.50, 0.30]$
4. **Sensor High-ISO Noise (`gaussian_noise`):** Additive zero-mean Gaussian noise with $\sigma \in [15, 30, 50]$
5. **Edge Compression Artifacts (`jpeg_compression`):** Lossy JPEG quality $\in [50, 25, 10]$
6. **Sensor Downscaling (`downscale_restore`):** Subsampling to $[128, 64, 32]$ and bilinearly restored.

---

## 🔍 Diagnostic Failure Catalog

Run automated failure case extraction to mine the top False Positives, False Negatives, and Localization Mismatches:
```bash
python scripts/run_failure_analysis.py --category bottle --method patchcore --top-k 4
```
Diagnostic 4-panel figures (`[RGB | Mask | Anomaly Map | Overlay]`) are saved under `results/figures/failure_cases/`.

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
