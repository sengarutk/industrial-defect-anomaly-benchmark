# Benchmark report

Generated 2026-11-02 09:58 UTC by `scripts/generate_report.py` from `results/mvtec_ad/tables`. Do not edit by hand.

* Categories (7): bottle, cable, carpet, grid, hazelnut, leather, metal_nut — all verified against the official MVTec AD manifest.
* Methods: PatchCore, PaDiM, ConvAutoencoder; seeds: 42, 123, 2026; runs: 63.
* 20% of train/good is held out per seed for nominal-only threshold calibration; models never see it.
* Expected cost per part: (1-π)·FPR + π·r·FNR with π = 0.01; primary r = 10; primary budget 5 false alarms per 1,000 nominal parts.

## Clean accuracy and robustness (mean over seeds)

| category   | method      |   image_auroc |   image_auroc_std |   pixel_auroc |   aupro |   aupro_std |   mrd_auroc |   mrd_aupro |
|:-----------|:------------|--------------:|------------------:|--------------:|--------:|------------:|------------:|------------:|
| bottle     | autoencoder |        0.7844 |            0.0075 |        0.7352 |  0.5302 |      0.0030 |      0.1151 |      0.2080 |
| bottle     | padim       |        0.9974 |            0.0012 |        0.9841 |  0.9496 |      0.0010 |      0.1793 |      0.2341 |
| bottle     | patchcore   |        1.0000 |            0.0000 |        0.9830 |  0.9329 |      0.0013 |      0.0574 |      0.1075 |
| cable      | autoencoder |        0.3355 |            0.0082 |        0.6170 |  0.4061 |      0.0102 |      0.0009 |      0.0680 |
| cable      | padim       |        0.8549 |            0.0085 |        0.9706 |  0.9048 |      0.0054 |      0.1225 |      0.1950 |
| cable      | patchcore   |        0.9905 |            0.0021 |        0.9853 |  0.9398 |      0.0016 |      0.1709 |      0.1444 |
| carpet     | autoencoder |        0.3355 |            0.0254 |        0.7353 |  0.4173 |      0.0130 |      0.0052 |      0.0850 |
| carpet     | padim       |        0.9643 |            0.0046 |        0.9873 |  0.9431 |      0.0041 |      0.2483 |      0.2791 |
| carpet     | patchcore   |        0.9861 |            0.0002 |        0.9896 |  0.9346 |      0.0019 |      0.1060 |      0.1457 |
| grid       | autoencoder |        0.8502 |            0.0442 |        0.8184 |  0.6201 |      0.0404 |      0.2439 |      0.1331 |
| grid       | padim       |        0.8616 |            0.0503 |        0.9264 |  0.8321 |      0.0496 |      0.2948 |      0.3901 |
| grid       | patchcore   |        0.9680 |            0.0039 |        0.9774 |  0.9016 |      0.0046 |      0.3265 |      0.4847 |
| hazelnut   | autoencoder |        0.9811 |            0.0031 |        0.9781 |  0.8739 |      0.0130 |      0.0436 |      0.0949 |
| hazelnut   | padim       |        0.8581 |            0.0102 |        0.9808 |  0.8685 |      0.0028 |      0.1018 |      0.2169 |
| hazelnut   | patchcore   |        0.9999 |            0.0002 |        0.9868 |  0.8811 |      0.0053 |      0.0494 |      0.0733 |
| leather    | autoencoder |        0.8604 |            0.0053 |        0.9678 |  0.9330 |      0.0054 |      0.0602 |      0.1003 |
| leather    | padim       |        0.9994 |            0.0005 |        0.9935 |  0.9822 |      0.0043 |      0.1432 |      0.0991 |
| leather    | patchcore   |        1.0000 |            0.0000 |        0.9922 |  0.9729 |      0.0003 |      0.0559 |      0.0672 |
| metal_nut  | autoencoder |        0.3832 |            0.0120 |        0.6707 |  0.3489 |      0.0218 |      0.0035 |      0.0306 |
| metal_nut  | padim       |        0.9728 |            0.0037 |        0.9721 |  0.9159 |      0.0062 |      0.2241 |      0.3452 |
| metal_nut  | patchcore   |        0.9972 |            0.0006 |        0.9835 |  0.9348 |      0.0007 |      0.1679 |      0.1897 |

## Operating point at 5/1k (threshold from held-out nominals)

| category   | method      |   n_seeds |   budget_per_1k |    tpr |   tpr_ci_low |   tpr_ci_high |   test_fa_at_1k |   test_fa_ci_low |   test_fa_ci_high |   expected_cost_r10 |   cost_ci_low |   cost_ci_high |   budget_violation_rate |   test_resolution_per_1k |   calib_resolution_per_1k |   p_overload_iid |   p_overload_burst |
|:-----------|:------------|----------:|----------------:|-------:|-------------:|--------------:|----------------:|-----------------:|------------------:|--------------------:|--------------:|---------------:|------------------------:|-------------------------:|--------------------------:|-----------------:|-------------------:|
| bottle     | autoencoder |         3 |          5.0000 | 0.4444 |       0.2963 |        0.6032 |         16.6667 |           0.0000 |           66.6667 |              0.0721 |        0.0487 |         0.1141 |                  0.3333 |                  50.0000 |                   23.8095 |           0.1667 |             0.1000 |
| bottle     | padim       |         3 |          5.0000 | 0.9365 |       0.8889 |        0.9735 |         33.3333 |           0.0000 |          100.0000 |              0.0393 |        0.0042 |         0.1043 |                  0.6667 |                  50.0000 |                   23.8095 |           0.2000 |             0.2000 |
| bottle     | patchcore   |         3 |          5.0000 | 1.0000 |       1.0000 |        1.0000 |          0.0000 |           0.0000 |            0.0000 |              0.0000 |        0.0000 |         0.0000 |                  0.0000 |                  50.0000 |                   23.8095 |           0.0000 |             0.0000 |
| cable      | autoencoder |         3 |          5.0000 | 0.0072 |       0.0000 |        0.0217 |         34.4828 |           0.0000 |           86.2069 |              0.1334 |        0.1000 |         0.1793 |                  0.6667 |                  17.2414 |                   22.2222 |           0.3333 |             0.3000 |
| cable      | padim       |         3 |          5.0000 | 0.2428 |       0.1341 |        0.3877 |         22.9885 |           0.0000 |           63.2184 |              0.0985 |        0.0789 |         0.1285 |                  0.6667 |                  17.2414 |                   22.2222 |           0.0667 |             0.0667 |
| cable      | patchcore   |         3 |          5.0000 | 0.8659 |       0.7971 |        0.9239 |          0.0000 |           0.0000 |            0.0000 |              0.0134 |        0.0080 |         0.0203 |                  0.0000 |                  17.2414 |                   22.2222 |           0.0000 |             0.0000 |
| carpet     | autoencoder |         3 |          5.0000 | 0.0787 |       0.0300 |        0.1312 |        190.4762 |          71.4286 |          321.4286 |              0.2807 |        0.1574 |         0.4092 |                  1.0000 |                  35.7143 |                   17.8571 |           0.9667 |             1.0000 |
| carpet     | padim       |         3 |          5.0000 | 0.8352 |       0.7378 |        0.9251 |         59.5238 |           0.0000 |          142.8571 |              0.0754 |        0.0243 |         0.1508 |                  0.6667 |                  35.7143 |                   17.8571 |           0.6667 |             0.6333 |
| carpet     | patchcore   |         3 |          5.0000 | 0.9775 |       0.9551 |        0.9963 |        309.5238 |         154.7619 |          500.2976 |              0.3087 |        0.1555 |         0.4961 |                  1.0000 |                  35.7143 |                   17.8571 |           1.0000 |             1.0000 |
| grid       | autoencoder |         3 |          5.0000 | 0.0058 |       0.0000 |        0.0234 |          0.0000 |           0.0000 |            0.0000 |              0.0994 |        0.0977 |         0.1000 |                  0.0000 |                  47.6190 |                   18.8679 |           0.0000 |             0.0000 |
| grid       | padim       |         3 |          5.0000 | 0.3158 |       0.1930 |        0.4503 |         15.8730 |           0.0000 |           63.4921 |              0.0841 |        0.0620 |         0.1313 |                  0.3333 |                  47.6190 |                   18.8679 |           0.0000 |             0.1000 |
| grid       | patchcore   |         3 |          5.0000 | 0.8596 |       0.8012 |        0.9123 |         47.6190 |           0.0000 |          111.1111 |              0.0612 |        0.0152 |         0.1205 |                  1.0000 |                  47.6190 |                   18.8679 |           0.3667 |             0.2000 |
| hazelnut   | autoencoder |         3 |          5.0000 | 0.5857 |       0.3762 |        0.7382 |          0.0000 |           0.0000 |            0.0000 |              0.0414 |        0.0257 |         0.0605 |                  0.0000 |                  25.0000 |                   12.8205 |           0.0000 |             0.0000 |
| hazelnut   | padim       |         3 |          5.0000 | 0.5333 |       0.3523 |        0.7810 |        108.3333 |          33.3333 |          208.3333 |              0.1539 |        0.0911 |         0.2411 |                  1.0000 |                  25.0000 |                   12.8205 |           0.7000 |             0.7000 |
| hazelnut   | patchcore   |         3 |          5.0000 | 0.9667 |       0.9381 |        0.9905 |          0.0000 |           0.0000 |            0.0000 |              0.0033 |        0.0010 |         0.0067 |                  0.0000 |                  25.0000 |                   12.8205 |           0.0000 |             0.0000 |
| leather    | autoencoder |         3 |          5.0000 | 0.8587 |       0.8080 |        0.9058 |        500.0000 |         375.0000 |          614.5833 |              0.5091 |        0.3883 |         0.6240 |                  1.0000 |                  31.2500 |                   20.4082 |           1.0000 |             1.0000 |
| leather    | padim       |         3 |          5.0000 | 0.9928 |       0.9783 |        1.0000 |         31.2500 |           0.0000 |           83.3333 |              0.0317 |        0.0004 |         0.0832 |                  0.6667 |                  31.2500 |                   20.4082 |           0.3333 |             0.2000 |
| leather    | patchcore   |         3 |          5.0000 | 1.0000 |       1.0000 |        1.0000 |         72.9167 |          20.8333 |          135.4167 |              0.0722 |        0.0103 |         0.1341 |                  1.0000 |                  31.2500 |                   20.4082 |           0.6667 |             0.6667 |
| metal_nut  | autoencoder |         3 |          5.0000 | 0.0287 |       0.0108 |        0.0502 |        106.0606 |          30.3030 |          196.9697 |              0.2021 |        0.1282 |         0.2775 |                  1.0000 |                  45.4545 |                   22.7273 |           1.0000 |             1.0000 |
| metal_nut  | padim       |         3 |          5.0000 | 0.9104 |       0.8638 |        0.9570 |         90.9091 |          15.1515 |          196.9697 |              0.0990 |        0.0250 |         0.2150 |                  1.0000 |                  45.4545 |                   22.7273 |           0.4667 |             0.3333 |
| metal_nut  | patchcore   |         3 |          5.0000 | 0.9749 |       0.9534 |        0.9928 |         45.4545 |           0.0000 |          121.2121 |              0.0475 |        0.0018 |         0.1229 |                  0.6667 |                  45.4545 |                   22.7273 |           0.3667 |             0.3333 |

## Method comparison (categories as units)

| comparison               | metric            | unit     |   n_pairs |   n_nonzero |   mean_diff |   hodges_lehmann |   statistic |   p_value |   rank_biserial |   min_attainable_p |   holm_adjusted_p | reject_h0   |
|:-------------------------|:------------------|:---------|----------:|------------:|------------:|-----------------:|------------:|----------:|----------------:|-------------------:|------------------:|:------------|
| autoencoder vs padim     | expected_cost_r10 | category |         7 |           7 |      0.1081 |           0.0685 |      5.0000 |    0.1562 |          0.6429 |             0.0156 |            0.3125 | False       |
| autoencoder vs patchcore | expected_cost_r10 | category |         7 |           7 |      0.1189 |           0.0876 |      1.0000 |    0.0312 |          0.9286 |             0.0156 |            0.0938 | False       |
| padim vs patchcore       | expected_cost_r10 | category |         7 |           7 |      0.0108 |           0.0342 |     10.0000 |    0.5781 |          0.2857 |             0.0156 |            0.5781 | False       |

## Threshold rules, out of sample (r = 10)

| method      | strategy          |   expected_cost |   std_over_categories |   eval_fa_per_1k |   budget_violation_rate |
|:------------|:------------------|----------------:|----------------------:|-----------------:|------------------------:|
| autoencoder | cct_budget        |          0.0922 |                0.0286 |          15.7797 |                  0.1833 |
| autoencoder | cct_unconstrained |          0.0906 |                0.0317 |          16.3749 |                  0.1833 |
| autoencoder | nominal_budget    |          0.1031 |                0.0402 |          28.2318 |                  0.2595 |
| autoencoder | oracle            |          0.0624 |                0.0356 |           0.0000 |                  0.0000 |
| padim       | cct_budget        |          0.0660 |                0.0433 |          23.8735 |                  0.2881 |
| padim       | cct_unconstrained |          0.0761 |                0.0566 |          43.4500 |                  0.4048 |
| padim       | nominal_budget    |          0.0733 |                0.0335 |          33.8516 |                  0.3786 |
| padim       | oracle            |          0.0350 |                0.0334 |           1.3547 |                  0.0286 |
| patchcore   | cct_budget        |          0.0175 |                0.0231 |           8.6425 |                  0.0976 |
| patchcore   | cct_unconstrained |          0.0173 |                0.0227 |           8.6425 |                  0.0976 |
| patchcore   | nominal_budget    |          0.0359 |                0.0300 |          29.7175 |                  0.2833 |
| patchcore   | oracle            |          0.0043 |                0.0062 |           0.0000 |                  0.0000 |

| comparison                                       |   cost_ratio | unit     |   n_pairs |   n_nonzero |   mean_diff |   hodges_lehmann |   statistic |   p_value |   rank_biserial |   min_attainable_p |   holm_adjusted_p | reject_h0   |
|:-------------------------------------------------|-------------:|:---------|----------:|------------:|------------:|-----------------:|------------:|----------:|----------------:|-------------------:|------------------:|:------------|
| autoencoder: cct_budget vs nominal_budget        |      10.0000 | category |         7 |           6 |     -0.0108 |          -0.0093 |      3.0000 |    0.1562 |         -0.7143 |             0.0312 |            0.7812 | False       |
| autoencoder: cct_unconstrained vs nominal_budget |      10.0000 | category |         7 |           6 |     -0.0124 |          -0.0076 |      0.0000 |    0.0312 |         -1.0000 |             0.0312 |            0.1875 | False       |
| padim: cct_budget vs nominal_budget              |      10.0000 | category |         7 |           7 |     -0.0073 |          -0.0029 |      6.0000 |    0.2188 |         -0.5714 |             0.0156 |            0.8750 | False       |
| padim: cct_unconstrained vs nominal_budget       |      10.0000 | category |         7 |           7 |      0.0027 |           0.0016 |     13.0000 |    0.9375 |          0.0714 |             0.0156 |            1.0000 | False       |
| patchcore: cct_budget vs nominal_budget          |      10.0000 | category |         7 |           7 |     -0.0184 |          -0.0179 |     10.0000 |    0.5781 |         -0.2857 |             0.0156 |            1.0000 | False       |
| patchcore: cct_unconstrained vs nominal_budget   |      10.0000 | category |         7 |           7 |     -0.0186 |          -0.0182 |     10.0000 |    0.5781 |         -0.2857 |             0.0156 |            1.0000 | False       |

## Decision changes, nominal-only budget → CCT (budget), per evaluation half

| category   | method      |   cost_ratio |   flips |   false_alarms_removed |   false_alarms_added |   defects_lost |   defects_gained |   n_eval_nominal |   n_eval_defect |
|:-----------|:------------|-------------:|--------:|-----------------------:|---------------------:|---------------:|-----------------:|-----------------:|----------------:|
| bottle     | autoencoder |      10.0000 |  1.3333 |                 0.0333 |               0.0000 |         1.3000 |           0.0000 |          10.0000 |         32.0000 |
| bottle     | padim       |      10.0000 |  1.2667 |                 0.3333 |               0.0000 |         0.9333 |           0.0000 |          10.0000 |         32.0000 |
| bottle     | patchcore   |      10.0000 |  0.7333 |                 0.0000 |               0.0000 |         0.7333 |           0.0000 |          10.0000 |         32.0000 |
| cable      | autoencoder |      10.0000 |  0.1667 |                 0.1667 |               0.0000 |         0.0000 |           0.0000 |          29.0000 |         46.0000 |
| cable      | padim       |      10.0000 |  1.3667 |                 0.1667 |               0.0000 |         1.2000 |           0.0000 |          29.0000 |         46.0000 |
| cable      | patchcore   |      10.0000 |  0.9167 |                 0.0000 |               0.0000 |         0.9167 |           0.0000 |          29.0000 |         46.0000 |
| carpet     | autoencoder |      10.0000 |  0.8333 |                 0.3667 |               0.0000 |         0.4667 |           0.0000 |          14.0000 |         45.0000 |
| carpet     | padim       |      10.0000 |  1.0500 |                 0.0667 |               0.0000 |         0.9833 |           0.0000 |          14.0000 |         45.0000 |
| carpet     | patchcore   |      10.0000 |  1.7500 |                 0.9000 |               0.0000 |         0.8500 |           0.0000 |          14.0000 |         45.0000 |
| grid       | autoencoder |      10.0000 |  0.0000 |                 0.0000 |               0.0000 |         0.0000 |           0.0000 |          11.0000 |         29.0000 |
| grid       | padim       |      10.0000 |  0.9333 |                 0.0000 |               0.0000 |         0.9333 |           0.0000 |          11.0000 |         29.0000 |
| grid       | patchcore   |      10.0000 |  1.0500 |                 0.0167 |               0.0000 |         1.0333 |           0.0000 |          11.0000 |         29.0000 |
| hazelnut   | autoencoder |      10.0000 |  1.0333 |                 0.0000 |               0.0000 |         1.0333 |           0.0000 |          20.0000 |         35.0000 |
| hazelnut   | padim       |      10.0000 |  0.7333 |                 0.0333 |               0.0000 |         0.7000 |           0.0000 |          20.0000 |         35.0000 |
| hazelnut   | patchcore   |      10.0000 |  1.0333 |                 0.0000 |               0.0000 |         1.0333 |           0.0000 |          20.0000 |         35.0000 |
| leather    | autoencoder |      10.0000 |  1.0833 |                 0.2000 |               0.0000 |         0.8833 |           0.0000 |          16.0000 |         46.0000 |
| leather    | padim       |      10.0000 |  1.4167 |                 0.3167 |               0.0000 |         1.1000 |           0.0000 |          16.0000 |         46.0000 |
| leather    | patchcore   |      10.0000 |  1.5833 |                 0.7500 |               0.0000 |         0.8333 |           0.0000 |          16.0000 |         46.0000 |
| metal_nut  | autoencoder |      10.0000 |  0.7167 |                 0.4333 |               0.0000 |         0.2833 |           0.0000 |          11.0000 |         47.0000 |
| metal_nut  | padim       |      10.0000 |  1.2167 |                 0.0500 |               0.0000 |         1.1667 |           0.0000 |          11.0000 |         47.0000 |
| metal_nut  | patchcore   |      10.0000 |  1.5167 |                 0.3833 |               0.0000 |         1.1333 |           0.0000 |          11.0000 |         47.0000 |

## Image-score aggregation ablation (from stored pixel maps)

| method      | rule          |   image_auroc |   image_auroc_std |   image_ap |
|:------------|:--------------|--------------:|------------------:|-----------:|
| autoencoder | max           |        0.6472 |            0.2691 |     0.8252 |
| autoencoder | mean          |        0.5471 |            0.2360 |     0.7784 |
| autoencoder | percentile_95 |        0.5376 |            0.1977 |     0.7684 |
| autoencoder | percentile_99 |        0.6473 |            0.2408 |     0.8257 |
| autoencoder | top1pct_mean  |        0.6602 |            0.2625 |     0.8303 |
| padim       | max           |        0.9298 |            0.0667 |     0.9570 |
| padim       | mean          |        0.8050 |            0.1278 |     0.9049 |
| padim       | percentile_95 |        0.8959 |            0.0872 |     0.9497 |
| padim       | percentile_99 |        0.9552 |            0.0504 |     0.9743 |
| padim       | top1pct_mean  |        0.9551 |            0.0481 |     0.9743 |
| patchcore   | max           |        0.9917 |            0.0112 |     0.9969 |
| patchcore   | mean          |        0.8679 |            0.1496 |     0.9398 |
| patchcore   | percentile_95 |        0.9362 |            0.0765 |     0.9761 |
| patchcore   | percentile_99 |        0.9868 |            0.0202 |     0.9951 |
| patchcore   | top1pct_mean  |        0.9901 |            0.0153 |     0.9965 |

## Latency

| category   | method      |   seed | device                             |   p50_model_ms |   p95_model_ms |   fps_model |   peak_vram_mb |   p50_e2e_ms |   p95_e2e_ms |   fps_e2e |
|:-----------|:------------|-------:|:-----------------------------------|---------------:|---------------:|------------:|---------------:|-------------:|-------------:|----------:|
| bottle     | patchcore   |     42 | NVIDIA GeForce RTX 4050 Laptop GPU |        11.0099 |        13.0781 |     90.8270 |       167.1123 |      34.5209 |      44.7156 |   28.9680 |
| bottle     | padim       |     42 | NVIDIA GeForce RTX 4050 Laptop GPU |         7.5300 |        12.4793 |    132.8018 |       219.1914 |      25.5741 |      27.8648 |   39.1021 |
| bottle     | autoencoder |     42 | NVIDIA GeForce RTX 4050 Laptop GPU |         4.4706 |         5.1729 |    223.6843 |        11.4302 |      24.3237 |      30.5005 |   41.1121 |
| cable      | patchcore   |     42 | NVIDIA GeForce RTX 4050 Laptop GPU |        11.2640 |        13.0433 |     88.7784 |       175.5327 |      56.6846 |      64.7834 |   17.6415 |
| cable      | padim       |     42 | NVIDIA GeForce RTX 4050 Laptop GPU |         7.7017 |         9.5239 |    129.8420 |       219.1914 |      39.2694 |      43.3092 |   25.4651 |
| cable      | autoencoder |     42 | NVIDIA GeForce RTX 4050 Laptop GPU |         4.6807 |         6.0574 |    213.6418 |        11.4302 |      35.4941 |      41.4715 |   28.1737 |
| carpet     | patchcore   |     42 | NVIDIA GeForce RTX 4050 Laptop GPU |        12.5134 |        14.1783 |     79.9141 |       207.1030 |      50.0228 |      56.0383 |   19.9909 |
| carpet     | padim       |     42 | NVIDIA GeForce RTX 4050 Laptop GPU |         7.5280 |         9.1219 |    132.8375 |       219.1914 |      39.2855 |      41.6537 |   25.4547 |
| carpet     | autoencoder |     42 | NVIDIA GeForce RTX 4050 Laptop GPU |         4.5262 |         5.8411 |    220.9373 |        11.4302 |      36.7366 |      40.3191 |   27.2208 |
| grid       | patchcore   |     42 | NVIDIA GeForce RTX 4050 Laptop GPU |        11.5366 |        13.8326 |     86.6809 |       198.5195 |      28.9271 |      33.5916 |   34.5697 |
| grid       | padim       |     42 | NVIDIA GeForce RTX 4050 Laptop GPU |         7.4998 |         9.0164 |    133.3372 |       219.1914 |      23.0116 |      28.8934 |   43.4563 |
| grid       | autoencoder |     42 | NVIDIA GeForce RTX 4050 Laptop GPU |         4.2040 |         5.0323 |    237.8697 |        11.4302 |      19.1033 |      21.3666 |   52.3469 |
| hazelnut   | patchcore   |     42 | NVIDIA GeForce RTX 4050 Laptop GPU |        14.9298 |        16.4162 |     66.9802 |       269.5444 |      50.4856 |      57.7303 |   19.8076 |
| hazelnut   | padim       |     42 | NVIDIA GeForce RTX 4050 Laptop GPU |         7.3355 |         9.1162 |    136.3239 |       219.1914 |      37.6087 |      43.7671 |   26.5896 |
| hazelnut   | autoencoder |     42 | NVIDIA GeForce RTX 4050 Laptop GPU |         4.6823 |         6.2518 |    213.5714 |        11.4302 |      36.3775 |      44.5413 |   27.4896 |
| leather    | patchcore   |     42 | NVIDIA GeForce RTX 4050 Laptop GPU |        11.3477 |        13.1737 |     88.1240 |       187.4600 |      47.7274 |      59.0244 |   20.9523 |
| leather    | padim       |     42 | NVIDIA GeForce RTX 4050 Laptop GPU |         7.4523 |         9.0300 |    134.1877 |       219.1914 |      36.4688 |      41.9863 |   27.4207 |
| leather    | autoencoder |     42 | NVIDIA GeForce RTX 4050 Laptop GPU |         5.4684 |         6.7363 |    182.8694 |        11.4302 |      39.5661 |      44.8318 |   25.2742 |
| metal_nut  | patchcore   |     42 | NVIDIA GeForce RTX 4050 Laptop GPU |        11.9304 |        13.8756 |     83.8197 |       173.4287 |      33.8630 |      40.0131 |   29.5307 |
| metal_nut  | padim       |     42 | NVIDIA GeForce RTX 4050 Laptop GPU |         8.4901 |        10.7743 |    117.7841 |       219.1914 |      25.4017 |      27.5024 |   39.3674 |
| metal_nut  | autoencoder |     42 | NVIDIA GeForce RTX 4050 Laptop GPU |         6.0616 |         8.1954 |    164.9736 |        11.4302 |      24.7191 |      28.2671 |   40.4546 |

## Coreset selection benchmark

| device                             | feature_source                                                 |   num_patches_N |   feature_dim_D |   coreset_M |   timing_repeats |   time_cpu_numpy_s |   time_gpu_greedy_s |   time_gpu_topk_s |   speedup_gpu_greedy |   speedup_gpu_topk |   radius_cpu_numpy |   radius_gpu_greedy |   radius_gpu_topk |   radius_random |   index_agreement_gpu_vs_cpu |   peak_vram_mb_gpu_greedy |
|:-----------------------------------|:---------------------------------------------------------------|----------------:|----------------:|------------:|-----------------:|-------------------:|--------------------:|------------------:|---------------------:|-------------------:|-------------------:|--------------------:|------------------:|----------------:|-----------------------------:|--------------------------:|
| NVIDIA GeForce RTX 4050 Laptop GPU | bottle train/good (PatchCore layer2+3, random projection to D) |            1000 |              64 |         100 |                3 |             0.0022 |              0.0473 |            0.0015 |               0.0473 |             1.5279 |             1.9372 |              1.9372 |            4.0110 |          2.9094 |                       1.0000 |                   45.6729 |
| NVIDIA GeForce RTX 4050 Laptop GPU | bottle train/good (PatchCore layer2+3, random projection to D) |            5000 |              64 |         500 |                3 |             0.1935 |              0.1177 |            0.0043 |               1.6450 |            45.4145 |             1.3699 |              1.3699 |            2.0094 |          2.0692 |                       1.0000 |                   47.6494 |
| NVIDIA GeForce RTX 4050 Laptop GPU | bottle train/good (PatchCore layer2+3, random projection to D) |           10000 |              64 |        1000 |                3 |             0.1383 |              0.1679 |            0.0069 |               0.8233 |            20.1444 |             1.2188 |              1.2188 |            1.4319 |          2.0709 |                       0.9980 |                   50.1235 |
| NVIDIA GeForce RTX 4050 Laptop GPU | bottle train/good (PatchCore layer2+3, random projection to D) |           25000 |              64 |        2500 |                3 |             1.0835 |              0.4493 |            0.0158 |               2.4114 |            68.5948 |             1.0978 |              1.0978 |            1.1669 |          2.6605 |                       0.9888 |                   57.5361 |
| NVIDIA GeForce RTX 4050 Laptop GPU | bottle train/good (PatchCore layer2+3, random projection to D) |           50000 |              64 |        5000 |                3 |             6.2162 |              0.8275 |            0.0627 |               7.5117 |            99.1162 |             1.0085 |              1.0085 |            1.0481 |          2.3626 |                       0.9788 |                   69.8999 |
| NVIDIA GeForce RTX 4050 Laptop GPU | bottle train/good (PatchCore layer2+3, random projection to D) |          100000 |              64 |       10000 |                3 |            16.5325 |              1.7064 |            0.2825 |               9.6888 |            58.5176 |             0.9308 |              0.9308 |            0.9482 |          1.9672 |                       0.5249 |                   94.6191 |
| NVIDIA GeForce RTX 4050 Laptop GPU | bottle train/good (PatchCore layer2+3, random projection to D) |            1000 |             128 |         100 |                3 |             0.0058 |              0.0278 |            0.0011 |               0.2092 |             5.2945 |             1.8540 |              1.8540 |            3.7687 |          2.6128 |                       1.0000 |                   46.3145 |
| NVIDIA GeForce RTX 4050 Laptop GPU | bottle train/good (PatchCore layer2+3, random projection to D) |            5000 |             128 |         500 |                3 |             0.0492 |              0.0629 |            0.0016 |               0.7831 |            31.0874 |             1.3625 |              1.3625 |            1.8351 |          2.5927 |                       1.0000 |                   50.0908 |
| NVIDIA GeForce RTX 4050 Laptop GPU | bottle train/good (PatchCore layer2+3, random projection to D) |           10000 |             128 |        1000 |                3 |             0.2700 |              0.1587 |            0.0043 |               1.7014 |            62.5029 |             1.2304 |              1.2304 |            1.4445 |          2.4903 |                       1.0000 |                   55.3071 |
| NVIDIA GeForce RTX 4050 Laptop GPU | bottle train/good (PatchCore layer2+3, random projection to D) |           25000 |             128 |        2500 |                3 |             1.7126 |              0.4501 |            0.0245 |               3.8052 |            69.9870 |             1.1097 |              1.1097 |            1.1795 |          2.3621 |                       0.9928 |                   69.7432 |
| NVIDIA GeForce RTX 4050 Laptop GPU | bottle train/good (PatchCore layer2+3, random projection to D) |           50000 |             128 |        5000 |                3 |             6.7395 |              0.8662 |            0.0870 |               7.7806 |            77.5018 |             1.0230 |              1.0230 |            1.0571 |          1.8917 |                       0.9686 |                   94.3140 |
| NVIDIA GeForce RTX 4050 Laptop GPU | bottle train/good (PatchCore layer2+3, random projection to D) |          100000 |             128 |       10000 |                3 |            24.7081 |              2.9326 |            0.3611 |               8.4253 |            68.4154 |             0.9442 |              0.9442 |            0.9593 |          2.0491 |                       0.9282 |                  143.4473 |
| NVIDIA GeForce RTX 4050 Laptop GPU | bottle train/good (PatchCore layer2+3, random projection to D) |            1000 |             256 |         100 |                3 |             0.0037 |              0.0312 |            0.0006 |               0.1194 |             6.7460 |             1.9328 |              1.9328 |            3.6964 |          3.0758 |                       1.0000 |                   47.2910 |
| NVIDIA GeForce RTX 4050 Laptop GPU | bottle train/good (PatchCore layer2+3, random projection to D) |            5000 |             256 |         500 |                3 |             0.0823 |              0.0638 |            0.0023 |               1.2898 |            35.5501 |             1.3815 |              1.3815 |            1.9216 |          2.7347 |                       1.0000 |                   54.9736 |
| NVIDIA GeForce RTX 4050 Laptop GPU | bottle train/good (PatchCore layer2+3, random projection to D) |           10000 |             256 |        1000 |                3 |             0.5677 |              0.1897 |            0.0073 |               2.9930 |            77.5324 |             1.2610 |              1.2610 |            1.4664 |          2.4668 |                       0.9980 |                   64.7720 |
| NVIDIA GeForce RTX 4050 Laptop GPU | bottle train/good (PatchCore layer2+3, random projection to D) |           25000 |             256 |        2500 |                3 |             3.7029 |              0.3999 |            0.0342 |               9.2588 |           108.2378 |             1.1176 |              1.1176 |            1.1842 |          2.4579 |                       0.9788 |                   94.1572 |
| NVIDIA GeForce RTX 4050 Laptop GPU | bottle train/good (PatchCore layer2+3, random projection to D) |           50000 |             256 |        5000 |                3 |            13.0050 |              1.4552 |            0.1302 |               8.9370 |            99.8644 |             1.0344 |              1.0344 |            1.0722 |          2.1124 |                       0.9632 |                  143.1421 |
| NVIDIA GeForce RTX 4050 Laptop GPU | bottle train/good (PatchCore layer2+3, random projection to D) |          100000 |             256 |       10000 |                3 |            41.9359 |              5.2439 |            0.5060 |               7.9970 |            82.8738 |             0.9580 |              0.9582 |            0.9719 |          2.0022 |                       0.0955 |                  241.1035 |
