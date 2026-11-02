# Methods and evaluation protocol

This document describes exactly what the code does. Numbers live in
[`benchmark_report.md`](benchmark_report.md), which is generated from the result tables.

## Data

* MVTec AD categories are downloaded from the Hugging Face mirror `foersben/mvtec-ad` and validated
  against the official per-split, per-defect-type image counts (`src/mvtec.py: MVTEC_MANIFEST`).
  Extra folders, wrong counts or missing masks abort every loader. See [`data_card.md`](data_card.md).
* Images: RGB, bilinear resize to 256×256, ImageNet normalisation. Masks: nearest-neighbour resize, binarised.
* For each seed, 20% of `train/good` is held out as a **nominal calibration set** (`split_train_calibration`).
  Models are fitted on the remaining 80% and never see the calibration images. The test split is used unchanged.

## Detectors (all "-style" reimplementations, not official reproductions)

| | PatchCore-style | PaDiM-style | Conv. autoencoder |
|---|---|---|---|
| Reference | Roth et al., CVPR 2022 | Defard et al., ICPR 2021 | — |
| Backbone | ResNet-18 (ImageNet) | ResNet-18 (ImageNet) | trained from scratch |
| Features | layer2 + upsampled layer3, 3×3 avg-pool, 384-d | layer1–3 upsampled to layer1, 448-d → 100 random channels | — |
| Fit | exact greedy k-center coreset, 10% of patches, selected in a 128-d Gaussian random projection | per-location Gaussian, Σ + 0.01·I | 4-stage conv encoder (128×16×16 latent) / transposed-conv decoder, MSE, AdamW (lr 1e-3, wd 1e-5), 40 epochs, batch 16 |
| Pixel map | nearest-neighbour distance | Mahalanobis distance | channel-mean squared residual |
| Image score | max of map | max of map | max of map |

All maps are bilinearly upsampled to 256×256 and Gaussian-smoothed (σ = 4). Deviations from the original
papers: ResNet-18 instead of WideResNet-50, 256 px input, no PatchCore re-weighting, PaDiM random
channel selection. Hyper-parameters come from `configs/benchmark.yaml`.

**What the seed controls:** the train/calibration split, PatchCore's projection matrix and coreset start
point, PaDiM's channel subset, and the autoencoder's initialisation and batch order. Runs are seeded with
`seed_everything` (Python, NumPy, torch; deterministic cuDNN).

## Metrics

* Image AUROC / AP (scikit-learn).
* Pixel AUROC / AP over all test pixels.
* **AU-PRO** up to FPR 0.3, computed exactly at every distinct score following the MVTec protocol
  (`src/metrics/pixel_metrics.py`); tested against an independent brute-force threshold sweep.
* **Robustness:** 6 corruptions × 3 severities, applied to the 256 px image (so a severity has the same
  pixel strength in every category). MRD = mean of max(0, clean − corrupted); MPC = signed mean.
  Gaussian noise is seeded per image and condition.

## Operating points

Conventions (`src/metrics/operational.py`): a part is flagged when score ≥ τ; FA@1k = 1000·FPR on nominal
parts; MD@1k = 1000·FNR on defective parts. **Expected cost per part** is the only cost definition used:

    C(τ) = (1 − π)·FPR(τ) + π·r·FNR(τ),   false alarm cost 1, missed defect cost r, production defect rate π

* **Nominal-only budget threshold:** the lowest τ whose FPR on the *calibration nominals* is ≤ b/1000. With
  n calibration images at most ⌊n·b/1000⌋ may be flagged, so for small n and b = 5/1k the threshold sits
  just above the largest calibration score.
* Every realised test rate is reported with an exact Clopper–Pearson interval and with the test-set
  **resolution** 1000 / n_nominal (e.g. 50/1k for 20 nominal test images). Per-mille alarm budgets cannot be
  verified on MVTec test sets; the tables make this explicit instead of hiding it.
* **Operator load:** score pools are resampled into 10,000-part IID, Markov-burst and drifting streams; the
  number of alerts per 1,000-part window is compared with a capacity of 60. These projections add no
  information beyond the measured scores and inherit their resolution.

## Cost-Calibrated Thresholding (CCT)

CCT is constrained empirical risk minimisation over a scalar threshold (`src/metrics/cct.py`):
τ* = argmin C_cal(τ) subject to FPR_cal(τ) ≤ b/1000 (or unconstrained). It needs labelled defects, so it is
evaluated with **repeated stratified 50/50 splits of the test set** (20 per run): thresholds use the
calibration half plus the held-out train/good nominals; all metrics are measured on the other half.
Compared rules: nominal-only budget, CCT with budget, CCT without budget, and an in-sample oracle
(lower bound, not deployable). Budget violations on the evaluation half are counted, never clipped.

## Statistics

* Per category × method: two-stage bootstrap (seeds, then test images within each class), B = 2,000.
* Method comparisons: two-sided Wilcoxon signed-rank tests with **categories as the independent units**
  (one value per category, averaged over seeds), Holm-adjusted. With n categories the smallest attainable
  two-sided p-value is 2/2ⁿ (0.0156 for n = 7), which bounds what any claim of significance can show.

## Provenance of the published sweep

The host GPU was unavailable for part of the October 2026 rerun, so the sweep was resumed across sessions:
the first runs executed on CPU and the remainder on the GPU. Each record in `results/mvtec_ad/runs/` stores
its device and git commit; the commits differ only in changes outside the benchmark path (profiling moved to
its own script, removal of UTF-8 byte-order marks; verified with `git diff 76887cf 8d67860`). Accuracy metrics do not depend on the device beyond floating-point
nondeterminism; latency is measured separately and only on the GPU.

## Systems measurements (GPU required)

* `scripts/profile_latency.py`: batch-1 latency per category × method. T_model = device tensor → scores and
  maps (includes CPU smoothing); T_e2e = PNG on disk → decode → resize → normalise → copy → predict.
* `scripts/run_coreset_benchmark.py`: CPU NumPy greedy vs GPU torch greedy (the implementation PatchCore uses)
  vs an approximate top-50 batched variant vs random, on real PatchCore features. Every cell is measured;
  index agreement with the NumPy reference and coverage radius are reported.
