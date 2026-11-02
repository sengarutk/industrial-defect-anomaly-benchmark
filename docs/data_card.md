# Data card

* **Dataset:** MVTec Anomaly Detection (Bergmann et al., CVPR 2019; IJCV 2021), CC BY-NC-SA 4.0.
* **Source used here:** Hugging Face mirror `foersben/mvtec-ad`, downloaded by `scripts/download_dataset.py`.
* **Integrity:** every category is checked against the official per-split, per-defect-type image counts in `src/mvtec.py` (`MVTEC_MANIFEST`) and every defect image must have a mask; loaders refuse anything else.
* **Preprocessing:** RGB, bilinear resize to 256×256, ImageNet normalisation; masks resized with nearest-neighbour.
* **Splits:** for each seed, 20% of train/good is held out as a nominal calibration set; the rest fits the model. The test split is used unchanged.

| category   |   train_good |   test_good |   test_defective |   defect_types |   fit_images_per_seed |   calibration_images_per_seed |
|:-----------|-------------:|------------:|-----------------:|---------------:|----------------------:|------------------------------:|
| bottle     |          209 |          20 |               63 |              3 |                   167 |                            42 |
| cable      |          224 |          58 |               92 |              8 |                   179 |                            45 |
| carpet     |          280 |          28 |               89 |              5 |                   224 |                            56 |
| grid       |          264 |          21 |               57 |              5 |                   211 |                            53 |
| hazelnut   |          391 |          40 |               70 |              4 |                   313 |                            78 |
| leather    |          245 |          32 |               92 |              5 |                   196 |                            49 |
| metal_nut  |          220 |          22 |               93 |              4 |                   176 |                            44 |

Totals: 1833 train, 777 test (556 defective, each with a ground-truth mask).
