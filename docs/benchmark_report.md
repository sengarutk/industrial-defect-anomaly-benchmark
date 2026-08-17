# Flagship Benchmark Report: Industrial Visual Anomaly Detection

## Executive Summary
This document summarizes benchmark findings across production visual anomaly detectors evaluated on the MVTec Anomaly Detection dataset.

### Core Metrics & Key Findings
1. **PatchCore** achieves superior localization AU-PRO and Image AUROC across structured categories (`bottle`, `hazelnut`, `metal_nut`).
2. **PaDiM** provides optimal throughput-accuracy balance ($98.5$ FPS) with minimal memory footprint.
3. **Convolutional Autoencoders** achieve the highest inference speed ($178.6$ FPS) but suffer under high-frequency texture defects.

## Publication Figures
- **Pareto Tradeoff:** `results/figures/pareto_latency_vs_aupro.png`
- **Robustness Heatmap:** `results/figures/robustness_heatmap.png`
- **Uncertainty Calibration:** `results/figures/calibration_diagram.png`
- **Ablation Study:** `results/figures/robust_training_ablation.png`

## LaTeX Tables
Generated booktabs tables are available under `results/tables/`:
- `main_results.tex`
- `deployment_profiling.tex`
- `robustness_mCE.tex`
