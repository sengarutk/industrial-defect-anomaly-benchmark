# Dataset Data Card: MVTec Anomaly Detection (7 Categories)

- **Benchmark Dataset:** MVTec Anomaly Detection (Bergmann et al., CVPR 2019 / IJCV 2021)
- **Hugging Face Mirror:** `foersben/mvtec-ad`
- **Evaluated Categories (7 Categories, 63 Runs):**
  - **Rigid Objects:** `bottle` (209 train / 83 test), `metal_nut` (220 train / 115 test)
  - **Deformable Objects:** `cable` (224 train / 150 test)
  - **Natural/Organic Structures:** `hazelnut` (391 train / 110 test)
  - **Textures & Periodic Structures:** `carpet` (280 train / 117 test), `grid` (264 train / 78 test), `leather` (245 train / 124 test)
- **Calibration Protocol:** 50/50 Stratified Out-of-Sample Split for Cost-Calibrated Thresholding (CCT)
- **Physical Corruptions (18 Conditions):** Defocus Blur, Motion Blur, Brightness Decay, Sensor Noise, JPEG Compression, Interpolation Downscaling
- **Stream Dynamics:** IID, Block-Correlated Markov Burst ($L=20$), Gradual Illumination Drift ($0.10 \times \sigma_{\text{nom}}$), and Mixed Corruptions ($p_{\text{corr}} \in [0.10, 0.50]$)