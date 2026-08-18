# Baseline Architecture & Literature Reference Validation

This document details architectural configurations, design decisions, and literature baseline alignments for the Industrial Defect Anomaly Benchmark.

---

## 1. Baseline Implementation Scope & Attribution

> **Baseline Implementation Scope:** Our PatchCore-inspired and PaDiM-inspired implementations are sanity-checked against reported literature values. They are not intended as exact reproductions, as backbone selection (ResNet-18), feature taps, input resolution ($256 \times 256$), coreset subsampling ratio ($10\%$), 2D Gaussian post-processing ($\sigma=4$), and evaluation protocols may differ from the original publications (Roth et al., CVPR 2022; Defard et al., ICPR 2021).

### 1.1 PatchCore-Inspired Baseline
- **Literature Reference:** Roth et al., *"Towards Total Recall in Industrial Anomaly Detection"*, CVPR 2022.
- **Backbone Architecture:** ResNet-18 pretrained on ImageNet-1K (`ResNet18_Weights.IMAGENET1K_V1`).
- **Feature Layer Taps:** `layer2` ($128\text{-d}$, spatial downsampling $4\times$) and `layer3` ($256\text{-d}$, spatial downsampling $8\times$).
- **Patch Aggregation:** Locally aware patch pooling via `AvgPool2d(kernel_size=3, stride=1, padding=1)` followed by multi-scale bilinear interpolation to `layer2` spatial resolution and channel concatenation ($d = 384\text{-d}$).
- **Coreset Reduction:** Minimax Greedy $k$-Center Coreset Subsampling ($10\%$ subsampling ratio) with Johnson-Lindenstrauss random projection down to $128\text{-d}$ for memory-efficient distance computation.
- **Anomaly Scoring & Smoothing:** Nearest-neighbor Euclidean distance in feature memory bank $\mathcal{M}$ followed by 2D Gaussian spatial smoothing ($\sigma = 4$).

### 1.2 PaDiM-Inspired Baseline
- **Literature Reference:** Defard et al., *"PaDiM: a Patch Distribution Modeling Framework for Anomaly Detection and Localization"*, ICPR 2021.
- **Backbone Architecture:** ResNet-18 pretrained on ImageNet-1K.
- **Feature Layer Taps:** `layer1` ($64\text{-d}$), `layer2` ($128\text{-d}$), `layer3` ($256\text{-d}$) bilinearly interpolated and concatenated ($d_{\text{total}} = 448\text{-d}$).
- **Dimensionality Reduction:** Deterministic channel subsampling to $d_{\text{reduced}} = 100\text{-d}$ with fixed random seed.
- **Covariance Regularization:** Fixed Diagonal Regularization ($\Sigma + \lambda I_{100}$ with $\lambda = 0.01$) to guarantee positive-definiteness and numerical invertibility across all spatial locations.
- **Scoring:** Vectorized Mahalanobis distance against spatial Gaussian distribution parameters ($\mu_{(h,w)}, \Sigma_{(h,w)}^{-1}$).

### 1.3 Convolutional Autoencoder Baseline
- **Architecture:** 4-stage convolutional encoder-decoder with symmetric downsampling/upsampling:
  - Encoder: `[Conv2d(3->32), Conv2d(32->64), Conv2d(64->128), Conv2d(128->256)]`
  - Latent Bottleneck: $256 \times 16 \times 16$ spatial latent tensor.
  - Decoder: `[ConvTranspose2d(256->128), ConvTranspose2d(128->64), ConvTranspose2d(64->32), Conv2d(32->3)]`
- **Optimization:** AdamW optimizer ($\text{lr}=10^{-3}$, weight decay $=10^{-5}$), MSE reconstruction loss, trained for 40 epochs on nominal images.
- **Anomaly Scoring:** Pixel-level squared residual error $\|x - \hat{x}\|^2$ smoothed with $\sigma=4$.

---

## 2. Methodological & Empirical Notes

### 2.1 Per-Region Overlap (AU-PRO) Evaluation Protocol
Per-Region Overlap (AU-PRO) follows the MVTec-style evaluation protocol up to a maximum FPR of 0.30. False Positive Rates are strictly normalized against nominal background pixels:
$$N_{\text{normal}} = \sum (1 - \text{mask})$$

### 2.2 Calibration & Exploratory Rank Diagnostics
Raw anomaly scores are not intrinsically calibrated class probabilities. In unsupervised visual anomaly detection (trained on nominal $y=0$ data only), fitting posterior probability calibration without an independent labeled validation split introduces test-label leakage. Reliability diagrams are provided strictly as exploratory score-outcome rank diagnostics, not as deployment-calibrated risk estimates.

### 2.3 Dataset Provenance
Experiments utilize a community Hugging Face mirror (`foersben/mvtec-ad`) of the MVTec Anomaly Detection benchmark (Bergmann et al., CVPR 2019 / IJCV 2021). Downloaded directory structures and expected per-category image counts ($1,324$ train, $575$ test, $400$ masks) were validated on disk.
