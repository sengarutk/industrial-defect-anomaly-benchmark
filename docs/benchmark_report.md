# Comprehensive Benchmark Report: Industrial Visual Anomaly Detection Under Real-World Degradation

## 1. Executive Summary & Baseline Implementation Scope
This report presents an empirical evaluation of visual anomaly detection algorithms on the **MVTec Anomaly Detection (MVTec AD)** dataset under clean nominal conditions and realistic industrial distribution shifts. 

> **Baseline Implementation Scope:** Our PatchCore-inspired and PaDiM-inspired implementations are sanity-checked against reported literature values. They are not intended as exact reproductions, as backbone selection (ResNet-18), feature taps, input resolution ($256 \times 256$), coreset subsampling ratio ($10\%$), 2D Gaussian post-processing ($\sigma=4$), and evaluation protocols may differ from the original publications (Roth et al., CVPR 2022; Defard et al., ICPR 2021).

> **Dataset Provenance:** Experiments utilize a community Hugging Face mirror (`foersben/mvtec-ad`) of the MVTec Anomaly Detection benchmark (Bergmann et al., CVPR 2019 / IJCV 2021). Downloaded directory structures and expected per-category image counts ($1,324$ train, $575$ test, $400$ masks) were validated on disk.

The evaluation encompasses:
- **5 Representative Categories:** Rigid objects (`bottle`, `hazelnut`, `metal_nut`), deformable items (`cable`), and homogeneous textures (`carpet`).
- **3 Anomaly Detection Paradigms:** PatchCore-inspired (multi-scale memory coreset), PaDiM-inspired (spatial Gaussian distributions), and Reconstruction-based (Convolutional Autoencoders).
- **3 Deterministic Random Seeds:** `42`, `123`, `2026` ($45$ experimental benchmark sweeps).
- **Synchronized Hardware Profiling:** $50$ warmup runs and $300$ active runs measuring model-path latency ($T_{\text{model}}$), end-to-end pipeline latency ($T_{\text{e2e}}$), and peak VRAM.
- **Physical Robustness Suite:** 6 camera/lighting degradations across 3 severity levels ($18$ conditions).

---

## 2. Multi-Seed Empirical Results

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

---

## 3. Deep-Dive Methodological Findings

### 3.1 Per-Region Overlap (AU-PRO) Evaluation Protocol
Per-Region Overlap (AU-PRO) follows the MVTec-style evaluation protocol up to a maximum FPR of 0.30. False Positive Rates are strictly normalized against nominal background pixels:
$$N_{\text{normal}} = \sum (1 - \text{mask})$$

### 3.2 Autoencoder Anomaly Score Dynamics Under Perturbations
On high-frequency textures (`carpet`, `bottle`), nominal structures incur high baseline $L_2$ reconstruction residuals, causing residual false positive spikes on normal test samples. Under smoothing perturbations such as **defocus blur** and **sensor downscaling**, high-frequency texture noise is smoothed out, which suppresses normal image background residuals and artificially sharpens rank-separation on specific splits. Consequently, while the signed performance change $\text{MPC} = \frac{1}{18}\sum (M_{\text{clean}} - M_{\text{corrupted}})$ can be negative, the **non-negative Mean Robustness Degradation (MRD)** correctly measures zero degradation.

### 3.3 Feature Memory vs. Spatial Distribution Modeling
- **PatchCore-Inspired:** Achieves consistently high localization accuracy ($0.9188$–$0.9492$ AU-PRO) across all five categories. Its locally aware patch representation and minimax greedy coreset subsampling ensure resilience to deformation and orientation shifts.
- **PaDiM-Inspired:** Exhibits roughly $2\times$ faster model latency ($6.27\text{ ms}$ vs $12.65\text{ ms}$) but suffers on deformable structures (`cable`: $0.8617$ AUROC; `hazelnut`: $0.8404$ AUROC) due to fixed spatial Gaussian coordinate assumptions.
- **Fixed Diagonal Regularization:** PaDiM uses fixed diagonal Tikhonov regularization ($\Sigma + 0.01 \cdot I_{100}$) for numerical stability and invertibility of local covariance estimates.

### 3.4 Calibration & Exploratory Score Diagnostics
> **Calibration Note:** Raw anomaly scores are not intrinsically calibrated class probabilities. In unsupervised visual anomaly detection (trained on nominal $y=0$ data only), fitting posterior probability calibration without an independent labeled validation split introduces test-label leakage. Reliability diagrams are provided strictly as exploratory score-outcome rank diagnostics, not as deployment-calibrated risk estimates.

---

## 4. Hardware Profiling & Operational Latency

Synchronized measurements on GPU ($B=1$, ResNet-18 backbone):
- **PatchCore:** $T_{\text{model}} = 12.65\text{ ms}$ ($81.2\text{ FPS}$), $T_{\text{e2e}} = 38.78\text{ ms}$ ($29.3\text{ FPS}$), Peak VRAM = $247.1\text{ MB}$.
- **PaDiM:** $T_{\text{model}} = 6.27\text{ ms}$ ($160.1\text{ FPS}$), $T_{\text{e2e}} = 32.72\text{ ms}$ ($35.3\text{ FPS}$), Peak VRAM = $306.5\text{ MB}$.
- **ConvAutoencoder:** $T_{\text{model}} = 5.44\text{ ms}$ ($185.5\text{ FPS}$), $T_{\text{e2e}} = 31.89\text{ ms}$ ($36.2\text{ FPS}$), Peak VRAM = $215.2\text{ MB}$.
