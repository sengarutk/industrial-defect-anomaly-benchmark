# Audit and remediation log

This file has two parts: the remediation status (what was changed on branch `fix/audit-remediation`),
followed by the original audit of commit `812f044` exactly as written, kept for the record. Line numbers in
the audit refer to that old commit.

## Remediation status

| ID | Finding | Status | Where |
|---|---|---|---|
| A1 | Synthetic grid/leather | Fixed: real images downloaded; loaders validate against the official manifest; generators deleted | `src/mvtec.py`, `scripts/download_dataset.py` |
| A2 | Mock images in bottle test set | Fixed: removed; extra folders now abort loading (regression test) | `tests/test_dataset.py` |
| A3 | Hardcoded latency table | Fixed: fallback removed; latency table only from `profile_latency.py` measurements | `scripts/generate_report.py` |
| A4 | Hardcoded abstract macros | Fixed: every macro computed from result tables | `tables/latex/macros.tex` |
| A5 | Extrapolated coreset timings | Fixed: every cell measured on real features; script refuses to label CPU timings as GPU | `scripts/run_coreset_benchmark.py` |
| A6 | Aggregation ablation on fake maps | Fixed: reads stored pixel maps | `scripts/run_ablations.py` |
| A7 | Random/hardcoded figures and fallbacks | Fixed: all placeholder data paths deleted; figures fail or skip when inputs are missing | `scripts/generate_figures.py` |
| A8 | CCT FA clamp; `budget_satisfied=True` | Fixed: violations counted and reported; flag computed | `src/metrics/cct.py`, `tests/test_cct.py` |
| A9 | Post-hoc metric switch | Fixed: one expected-cost definition everywhere | `src/metrics/operational.py` |
| A10 | Captions claiming unrun analyses | Fixed: captions generated from the actual protocol parameters | `scripts/generate_report.py` |
| A11 | Placeholder legacy tables | Fixed: deleted | — |
| A12 | Non-existent bibliography entry | Fixed: replaced; PaDiM author list also corrected | `overleaf_paper/references.bib` |
| B1 | Thresholds fit on the test set | Fixed: thresholds from a held-out 20% of train/good | `src/mvtec.py: split_train_calibration` |
| B2 | "TPR @ 5/1k" at 17–50/1k | Fixed: realised rate, Clopper–Pearson CI and resolution reported next to every target | `scripts/run_operational_eval.py` |
| B3 | Streams add no information | Addressed: reported only as operator-load projections, with the limitation stated | `docs/methods.md` |
| B4 | Seeds had no effect | Fixed: seed reaches every stochastic component (regression tests) | `src/methods/__init__.py` |
| B5 | Pseudo-replication in tests | Fixed: categories are the units; exact Wilcoxon; Holm | `src/metrics/stats.py` |
| B6 | CCT evaluation too small | Addressed: 20 repeated stratified splits per run, calibration pool includes held-out nominals, limitation stated | `src/experiments/cct_ablation.py` |
| B7 | Three CWE definitions | Fixed (see A9) | |
| B8 | Diluted MRD, mixed code versions | Fixed: one fresh sweep, per-run records, duplicate keys rejected | `scripts/run_benchmark.py` |
| B9 | In-sample decision changes | Fixed: computed on evaluation halves only | `src/experiments/cct_ablation.py` |
| B10 | Circular AU-PRO validation | Fixed: exact MVTec-protocol AU-PRO tested against an independent brute-force sweep | `tests/test_pixel_metrics.py` |
| B11 | Coreset not greedy; "exact parity" | Fixed: PatchCore uses exact greedy; top-k kept only as a labelled approximation with radius reported | `src/methods/coreset.py` |
| B12 | Profiler measured the wrong thing | Fixed: real test tensor for T_model; PNG-from-disk path for T_e2e | `src/benchmarking/profiler.py` |
| B13 | AE below chance, config mismatch | Addressed: epochs from config, max aggregation like the other methods, results reported as measured; docs match code | `src/methods/autoencoder.py`, `docs/methods.md` |
| B14 | Corruption strength varied by category | Fixed: corruptions applied at 256 px; seeded noise; one mask resizing rule | `src/robustness/` |
| B15 | Mislabeled failure cases | Fixed: FP/FN defined against a deployable threshold | `src/analysis/failure_catalog.py` |
| B16 | Meaningless ECE | Removed | — |
| C1–C16 | Code bugs | Fixed or removed with the code they lived in (oracle CWE bug, PaDiM `cov`, overload windows, appended CSV, seeds, npz key, isotonic stub, `--mock` messages, silent zero masks, dead scripts, unused configs, LaTeX labels, units, pooled normalised curves, duplicate quantile function, hardcoded interpreter path) | |
| D1–D7 | Docs/README/paper disagreeing | Fixed: README results block, report, data card, environment card and paper tables are generated | `scripts/generate_report.py`, `scripts/collect_environment.py` |
| E | Paper incomplete | Rewritten with protocol, results, limitations | `overleaf_paper/main.tex` |
| F | Hygiene | BOMs removed, duplicate assets deleted, large maps out of git, CI added | `.github/workflows/tests.yml` |

Not fixable inside this repository: the old fabricated artifacts remain in **git history** on `main`
(rewriting published history was not done). The 66 MB stale `flagship1_overleaf_package.zip` in the working
copy is untracked and was left for the owner to delete; `scripts/sync_paper_assets.py --zip` builds a fresh one.

---

# Full Repository Audit — `industrial-defect-anomaly-benchmark`

Audit date: 2026-10-26 · Commit audited: `812f044` (main, clean tree)

Scope: every tracked file (446), plus untracked artifacts (`data/`, `flagship1_overleaf_package.zip`). All source, scripts, tests, configs, docs, result tables, figures, score archives (`.npz`), the paper (`overleaf_paper/`) and git history were read or inspected. The key claims were then checked by running code (see §5).

---

## 1. What the project is trying to do

**Thesis (paper title: *"Beyond AUROC: Cost-Calibrated Thresholding and Operational Evaluation of Unsupervised Industrial Anomaly Detectors"*):**
AUROC/AU-PRO ignore two things that matter in a factory: (a) operators can only review a few alarms (claimed budget ≤ 5 false alarms per 1,000 parts), and (b) missing a defect costs 10–50× more than a false alarm. The repo:

1. Trains three unsupervised detectors (PatchCore-style, PaDiM-style, conv autoencoder) on MVTec AD "good" images.
2. Scores the test set; stores per-image scores and per-pixel maps in `.npz` archives.
3. Computes standard metrics (image AUROC, pixel AUROC, AU-PRO), robustness under 18 synthetic corruptions (6 types × 3 severities), latency/VRAM.
4. Re-uses the stored scores for "operational" analysis: TPR at an alarm budget, missed defects per 1k, cost-weighted error (CWE), simulated 10,000-part production streams (IID / burst / drift), operator-overload probability.
5. Proposes **CCT (Cost-Calibrated Thresholding)**: choose the threshold that minimizes expected cost subject to a false-alarm budget, fit on a 50% calibration split and evaluated on the other 50%.
6. Adds a "systems" contribution: a GPU-batched k-center coreset selection claimed to be 274.6× faster than CPU with "exact coverage parity".
7. Generates LaTeX tables / figures and copies them into `overleaf_paper/` for a paper.

## 2. How everything is connected (data flow)

```
data/mvtec_ad/<cat>/{train/good, test/<type>, ground_truth/<type>}
        │  (src/mvtec.py: MVTecTrainNormal, MVTecTest;  src/robustness/dataset.py: CorruptedMVTecTest)
        ▼
scripts/run_benchmark.py ──► src/methods/{patchcore,padim,autoencoder}.py  (fit on train/good)
        │                    src/benchmarking/profiler.py                  (latency/VRAM)
        │                    src/robustness/evaluator.py + corruptions.py  (clean + 18 corrupted evals)
        │                    src/metrics/{image,pixel,calibration}_metrics (AUROC, AU-PRO, ECE)
        ├─► results/mvtec_ad/scores/<cat>_<method>_<seed>.npz   (image_scores, image_labels, pixel_amaps, ground_truth_masks)
        └─► results/mvtec_ad/tables/runs_master.csv (APPENDED) + summary_multiseed.{csv,md}

results/.../scores/*.npz
   ├─► scripts/run_operational_eval.py ──► src/metrics/operational.py, stats.py, experiments/operational_eval.py
   │        └─► tables/operational_results.{csv,md,tex}
   ├─► scripts/run_cct_experiments.py ──► experiments/cct_ablation.py, decision_changes.py, coreset_scalability.py
   │        └─► tables/cct_ablation.*, decision_changes.*, coreset_scalability.*
   ├─► scripts/run_ablations.py ──► experiments/coreset_systems.py, aggregation_ablation.py
   │        └─► tables/coreset_systems.*, aggregation_ablation.*
   └─► scripts/generate_operational_plots.py ──► visualization/publication_plots.py ──► figures/operational/*.png

runs_master.csv + summary_multiseed.csv + operational_results.csv
   └─► scripts/generate_report.py ──► tables/main_results.tex, deployment_profiling.tex, robustness_mrd_mpc.tex,
                                      operational_results.tex, generated_metrics.tex (abstract macros)
scripts/generate_plots.py ──► figures/{pareto, robustness_heatmap, calibration_diagram, robust_training_ablation}.png

results/mvtec_ad/{tables,figures} ──(manual copy)──► overleaf_paper/{tables,figures} ──► main.tex (\input / \includegraphics)
```

Orchestration: `scripts/reproduce_main_results.sh` runs tests → operational eval → CCT/ablations → plots → report. **It never runs `run_benchmark.py`**; it only re-derives everything from the cached `.npz` files.

Dead / unreachable: `scripts/02_pretrain_simclr.py`, `03_build_memorybank.py`, `04_eval_anomaly.py`, `05_make_plots.py`, `10_run_grid*.py`, `11_sanity_checks.py` (import `src.config`, `src.models`, `src.simclr`, `src.anomaly`, `src.eval_harness`, `src.plotting` — none exist); `runs/` (old SimCLR experiment output); `results/tables/`, `results/figures/` (legacy, mock-data era); `configs/*.yaml` (never loaded by any code); `RobustTrainingExperiment`, `MixedCorruptionStreamSimulator`, `evaluate_operational_robustness`, module-level `evaluate_threshold_strategies`, `ThresholdLineageAuditor` (only used in tests, never in the pipeline).

### File-by-file summary

| Path | Role |
|---|---|
| `src/mvtec.py` | Dataset classes; ImageNet normalisation; resizes to 256×256; nearest-neighbour mask resize. Includes **every** subfolder of `test/` as a defect type. |
| `src/methods/base.py` | Abstract `fit` / `predict` interface. |
| `src/methods/patchcore.py` | ResNet-18 layer2+layer3, 3×3 avg-pool, concat (384-d), random-projection to 128-d for coreset, "batched greedy" coreset (top-50 farthest per step), NN distance, Gaussian σ=4, image score = max. Seed defaults to 42. |
| `src/methods/padim.py` | ResNet-18 layer1–3 (448-d), random 100-channel subset (seed 42), per-location Gaussian with Σ+0.01I, Mahalanobis, σ=4, max. |
| `src/methods/autoencoder.py` | 4-down/4-up conv AE (128-ch latent), MSE, AdamW, data preloaded to GPU; score = 95th percentile of smoothed residual. |
| `src/metrics/image_metrics.py` | AUROC, AP, oracle max-F1. |
| `src/metrics/pixel_metrics.py` | Pixel AUROC/AP, AU-PRO (200 percentile thresholds, FPR ≤ 0.3). |
| `src/metrics/reference_aupro.py` | "Independent reference" AU-PRO — actually a copy of the above. |
| `src/metrics/calibration.py` | ECE on min-max-normalised scores; reliability data; isotonic stub. |
| `src/metrics/operational.py` | FA@1k, MD@1k, CWE, quantile / alert-budget thresholds, TPR@budget, operator overload. |
| `src/metrics/cost_calibrated.py` | CCT optimiser + cost curve. |
| `src/metrics/stats.py` | Bootstrap CI, coverage check, hierarchical bootstrap, Wilcoxon + Hodges–Lehmann + rank-biserial, Holm. |
| `src/experiments/operational_eval.py` | Stream simulators (IID, Markov burst, linear drift, mixed corruption). |
| `src/experiments/cct_ablation.py` | 50/50 split, Q99 vs Budget-5 vs CCT vs oracle; writes CCT table. |
| `src/experiments/decision_changes.py` | Decision flips Q99 → CCT; writes decision table. |
| `src/experiments/coreset_systems.py` / `coreset_scalability.py` | CPU greedy vs GPU greedy vs GPU top-k-batched vs random; scaling sweep. |
| `src/experiments/aggregation_ablation.py` | 5 map→score pooling rules. |
| `src/experiments/robust_training.py` | Clean vs augmented-train comparison (never run). |
| `src/robustness/*` | 6 corruptions × 3 severities, corrupted dataset, evaluator (MRD / MPC / "mCE"). |
| `src/benchmarking/profiler.py` | CUDA-event latency, peak VRAM, "end-to-end" latency. |
| `src/analysis/failure_catalog.py` | Mines FP / FN / localisation / "corruption" cases and renders 4-panel figures. |
| `src/visualization/publication_plots.py` | All plotting functions. |
| `src/utils/*` | `seed_everything`, `ThresholdRecord` dataclass. |
| `scripts/download_dataset.py` | **Generates synthetic `grid` / `leather`; downloads nothing.** |
| `scripts/run_*.py`, `generate_*.py` | Pipeline drivers (see diagram). |
| `tests/*` (21 files, 64 tests) | Mostly smoke / shape tests on synthetic arrays. |
| `docs/*` | Report, data card, environment card, reference validation. |
| `overleaf_paper/` | 112-line `main.tex`, bib, copies of tables / figures. |

---

## 3. Findings — ranked by severity

Severity key: **[CRITICAL]** fabricated, synthetic or hardcoded results shown as real measurements; **[HIGH]** a method or statistics flaw that invalidates a headline claim; **[MED]** a correctness bug or a material inconsistency; **[LOW]** hygiene.

### A. Fabricated, synthetic and hardcoded results

**A1 [CRITICAL] — 2 of the 7 "MVTec AD" categories are procedurally drawn synthetic images.**
- `data/mvtec_ad/grid` and `data/mvtec_ad/leather` are 256×256 PNGs: a drawn line lattice and a flat brown noise texture. Defects are drawn ellipses, lines and rectangles (`scripts/download_dataset.py:57-225`). Counts are 200 train / 30 good / 15 per defect type, all files written in about 2 seconds on 2026-08-31 22:48.
- Real MVTec: grid has 264 train / 78 test images (1024² greyscale); leather has 245 / 124 (1024² RGB). `docs/data_card.md` lists the **official** counts (264/78, 245/124), not what is on disk.
- The script prints `✅ Generated authentic structured dataset for 'grid'`.
- Git history: commit `1598d37` ("complete 7-category MVTec AD research benchmark … 63 runs") **deleted the real Hugging Face download code** (`download_real_category`, `foersben/mvtec-ad`) and replaced it with these generators.
- Effect: every method, including the autoencoder that scores 0.30 AUROC on real metal_nut, gets **1.0000 AUROC** on grid and leather. These rows appear in Tables 1, 3, 4 and 5 of the paper and drive the CCT "wins" (A8 / B7). The abstract's "seven MVTec AD categories" is false.

**A2 [CRITICAL] — 3 synthetic mock images contaminate the real bottle test set.**
- `data/mvtec_ad/bottle/test/defect/000-002.png` (plus masks) are purple 256×256 squares with a red circle, left over from the old mock generator.
- `MVTecTest` / `CorruptedMVTecTest` include every subfolder, so the bottle test set is 86 images / 66 defects instead of 83 / 63. The `.npz` files confirm n=86.
- Measured effect: autoencoder bottle AUROC 0.519 with them vs 0.496 without. They also enter every bottle operational and CCT number.

**A3 [CRITICAL] — The paper's deployment/latency table is hardcoded.**
- `scripts/generate_report.py:119-126`: when `summary_multiseed.csv` has no positive latency, it writes fixed "empirical_rows" (PatchCore 10.94 ms / 91.4 FPS / 29.89 ms / 205.9 MB, …).
- The current `summary_multiseed.csv` has **all latencies = 0**: the 2026-08-31 rerun had profiling off. So the fallback fires, and `overleaf_paper/tables/deployment_profiling.tex` is exactly those constants.
- These constants are the **bottle-only** means from the 08-17 runs (e.g. 10.58, 11.13, 11.10 → 10.94), presented as method-level numbers. They were added in commit `affb065` ("restore deployment latencies").
- The README and `docs/benchmark_report.md` give different latencies again (12.65 / 38.78 / 247.1).

**A4 [CRITICAL] — The abstract's headline numbers are hardcoded constants.**
- `generate_report.py:260-265` writes `\CWEReductionMax{4.5\times}` and `\CoresetSpeedupGPU{274.6\times}` literally. The commit message `a8d444a` says "dynamically bind abstract macros".
- 274.6× appears in no table: `coreset_systems` gives 223.7×, scalability rows range 1.9–887.7×.
- 4.5× cannot be derived from the CCT table: its per-row ratios range from below 1 to about 44×.

**A5 [CRITICAL] — The coreset scalability table contains invented timings.**
- `src/experiments/coreset_scalability.py:47-60`: for N > 10k, CPU time is the formula `1.81·(N/10k)²·(D/128)`. For N > 25k, GPU-unbatched time is `0.12·(N/10k)^1.5` and VRAM is `14.8·N/10k`.
- I recomputed: 5.65625, 11.3125, 22.625, 45.25, 90.5 and 1.3416407865 match the table **exactly**.
- So the 257×–888× speedups (and the paper's "550×" row) divide by fabricated denominators, and the table does not mark them as extrapolated.
- `overleaf_paper/tables/coreset_scalability.csv` also comes from a different run than its own `.tex`.

**A6 [CRITICAL] — The aggregation ablation runs on fake anomaly maps.**
- `scripts/run_ablations.py:68-76` looks for an `"anomaly_maps"` key. The archives store `"pixel_amaps"`, so the code **always** synthesises 32×32 maps as `image_score × (0.8 + 0.4·rand)`.
- Result: all five pooling rules show about 0.8583 AUROC (±0.0001).
- On the real stored maps, they differ: global_max 0.836, p99 0.847, **p95 0.783**, top-1% 0.848, gauss-max 0.839. The fake table hides a real finding.

**A7 [CRITICAL] — Several figures are always drawn from random or hardcoded data (`scripts/generate_plots.py`).**
- `robustness_heatmap.png`: always `create_sample_robustness_df()`, i.e. `severity × uniform(0.02, 0.08)`, unseeded (line 81). The figure shows severity 3 *less* harmful than severity 2 for brightness and blur.
- `calibration_diagram.png`: hardcoded bins and `ECE = 0.0312` (lines 85-91).
- `robust_training_ablation.png`: hardcoded 0.9750 / 0.1420 / 0.9715 / 0.0610 (lines 95-99). `RobustTrainingExperiment` is never run anywhere.
- More silent fake-data fallbacks: `create_sample_master_df` (random Pareto data), `publication_plots.py:457` (overload bars 0.02 / 0.15 / 0.85), `publication_plots.py:200-203` (defaults 0.95 / 0.10 / 0.94 / 0.05), and the "synthetic demonstrations" branch in `generate_operational_plots.py:36-60`.
- These figures ship in `results/`, `overleaf_paper/figures/` and the Overleaf zip.

**A8 [CRITICAL] — Reported CCT false-alarm rates are clamped to the budget.**
- `src/experiments/cct_ablation.py:111`: `fa_cct_r10 = min(max_alerts_per_1k, compute_fa_at_1k(...))`. The table column is "FA@1k (CCT) (≤ 5)", so it **can never show a violation**.
- I recomputed out-of-sample: **21 of 63 runs exceed 5/1k, worst 300/1k**. Values like "1.7" and "3.3" in the table are means of clamped {0, 5}.
- `cost_calibrated.py:128` returns `"budget_satisfied": True` unconditionally. In 6 runs, even the *calibration* FPR exceeds 5/1k. The docstring "strictly satisfies FA@1k ≤ budget" is false.
- The clamp was added in commit `ec1930b` ("restore CCT out-of-sample dominance in Table 5").

**A9 [CRITICAL] — The metric was switched after the fact so CCT wins.**
- The same commit `ec1930b` changed CWE in the CCT table only: from count-based `(FP + r·FN)/N` to prior-weighted `0.99·FPR + 0.01·r·FNR`.
- All other tables (operational results, decision changes, robustness CWE, the CWE-curve figure) still use the count-based definition.
- So CCT is scored with a different metric than every other table, chosen after seeing results.

**A10 [CRITICAL] — Captions claim analyses that were never run.**
- `decision_changes.tex`: "95% confidence intervals … hierarchical bootstrap (B = 2,000) … Holm-Bonferroni". The table has means only, and no CI or test is computed in `decision_changes.py`.
- `main_results.tex`: "Multiplicity control enforced via Holm-Bonferroni". The table contains no tests.
- The abstract says B = 2,000, but `run_operational_eval.py:191-193` uses `n_resamples=1000`.
- The `tpr_at_10` "CI" columns are set equal to the mean (`run_operational_eval.py:206-208`).
- The significance section header is hardcoded "N=15" (line 248).

**A11 [CRITICAL] — The legacy tables in `results/tables/` are placeholders.**
- `deployment_profiling.tex` (18.42 / 21.05 / 54.3 / 482.1) and `robustness_mCE.tex` (0.9850 / 0.9520 / 0.0842 …) are round numbers that match no run.
- The `runs_master.csv` next to them is 5 rows on mock data (autoencoder AUROC = 1.0). They are still tracked in git.

**A12 [HIGH] — The bibliography contains a non-existent paper.**
- `bergmann2021uninformed` = "The Uninformed Student: A Knowledge Distillation Approach…", IJCV 129(1):1-22. No such paper exists.
- The real ones: *"Uninformed Students: Student–Teacher Anomaly Detection with Discriminative Latent Embeddings"* (CVPR 2020, which introduced AU-PRO), and the IJCV 2021 MVTec paper *"The MVTec Anomaly Detection Dataset…"* (IJCV 129:1038–1059).
- Minor: `efron1992bootstrap` labels a book reprint as a "journal".

### B. Method and statistics flaws that invalidate headline claims

**B1 [HIGH] — "Leakage-free" is false: operational thresholds are fit on the test set.**
- `run_operational_eval.py:118-131, 171-187` builds τ from `image_scores[image_labels == 0]` of the **test set** and evaluates on the same set.
- The conclusion ("CCT provides an empirical, leakage-free operational decision rule") and docs claim leakage-freeness.
- `ThresholdRecord` is imported in `operational_eval.py:14` but never used. `test_leakage_safety.py` only checks that functions don't have a parameter *named* `labels`.

**B2 [HIGH] — "TPR @ 5 alarms/1k" is really TPR at 17–50 alarms/1k.**
- With 20–58 nominal test images, `np.quantile(nominal, 0.995)` lands between the 2nd-largest and the maximum, so exactly one nominal image fires.
- Realized false alarms per 1k (PatchCore seed 42): bottle 50, metal_nut 45.5, carpet 35.7, grid/leather 33.3, hazelnut 25, cable 17.2. That is 3–10× over the stated budget, and the `fa_at_1k_mean` column in `operational_results.csv` shows it.
- A 5/1k rate cannot be estimated from 20 samples; you would need hundreds to thousands of nominal images.

**B3 [HIGH] — The "10,000-part production streams" add no information.**
- The simulator bootstrap-resamples the same 20–58 nominal and 63–93 defect test scores. FA@1k stays quantised at 1/n_nom.
- P(overload) is an artifact: bottle 0.43 because 50 FA + ~10 defects ≈ the capacity of 60.
- Burst and drift streams are computed but **never reported** in any table; drift is a tiny 0.1σ additive shift. The mixed-corruption simulator is never called.
- So the abstract's "non-IID factory stream dynamics" and the data card's "mixed corruptions p_corr ∈ [0.10, 0.50]" are unsupported.

**B4 [HIGH] — "Three seeds / 63 runs" are not independent.**
- `run_benchmark.get_model()` never passes the seed: `PatchCore(device=…)` and `PaDiM(device=…)` always use seed 42.
- PaDiM scores are identical across seeds (max diff 5.7e-6). PatchCore differs only through GPU nondeterminism (cable: exactly 0.0).
- So "± 0.0000" std and many zero-width CIs are trivially true and meaningless. Only the autoencoder has real seed variation.
- Also, `runs_master.csv` has only **62** unique (seed, category, method) runs; leather/autoencoder/2026 is missing from it.

**B5 [HIGH] — The significance claims rest on pseudo-replication.**
- The Wilcoxon pairs are (category × seed), but seeds are duplicates (B4) and grid/leather are ties on synthetic data.
- With the 5 real categories as independent units, I get **p = 0.0625 for both comparisons, i.e. not significant**.
- The README's `p = 5.68×10⁻¹⁴` is impossible: the smallest two-sided exact p for n = 15 is 6.1×10⁻⁵. The current table shows 6.2×10⁻⁴.

**B6 [HIGH] — The CCT evaluation is too small to support the conclusions.**
- The 50/50 split leaves 10–29 calibration nominals and 10–29 evaluation nominals per run, so one image moves FPR by 3.4–10 percentage points.
- "Budget-5" and "Q99" pick the same threshold in essentially every run (both interpolate near the calibration maximum), so they aren't really two baselines.
- The 3 "seeds" only change the split. The largest CCT gains are on synthetic grid/leather, and CCT is *worse* on leather/PatchCore (0.0018 vs 0.0000).
- CCT itself is constrained empirical risk minimisation over a scalar threshold — standard, not a new method.

**B7 [HIGH] — CWE has three incompatible definitions across the paper.**
- Count-based at the test set's ~70% defect rate (operational, decision changes, robustness).
- Prior-weighted at p = 0.01 (CCT table and CCT figure).
- Count-based on per-run min-max-normalised, cross-category pooled scores (CWE-curve figure).
- Numbers are not comparable across tables. In `decision_changes.py`, CCT is *optimised* with prior 0.01 but *reported* with count-based CWE.

**B8 [HIGH] — The robustness (MRD) numbers in Tables 1 and 3 are diluted by zeros and mix code versions.**
- `runs_master.csv` is appended to on every run: 118 rows for 62 unique runs.
- The 55 rows from 2026-08-31 have `mrd_* = 0` and latency = 0 (robustness and profiling were off), and `generate_report.py` averages them in.
- Example: bottle PatchCore MRD shows 0.0077 ± 0.0126, but the real value is 0.0284 (as `benchmark_report.md` says). PaDiM metal_nut shows 0.0721 vs a real 0.144.
- The 08-17 rows used an older *signed* MRD (negative for the autoencoder), clamped after the fact (commit `ec1930b`, "enforce non-negative MRD").
- `.npz` scores come from the 08-31 code version, but MRD and latency come from the 08-17 version (bottle PatchCore AU-PRO changes 0.8725 → 0.8708).
- The "Signed MPC" column equals MRD exactly for PatchCore — an artifact of the same mixing.
- The Pareto figure averages in the zero latencies too (bottle PaDiM plotted at about 1.9 ms).

**B9 [MED] — Decision-change analysis is in-sample.** CCT is optimised on the full test set (`decision_changes.py:111`), which contradicts the "out-of-sample" framing.

**B10 [MED] — AU-PRO validation is circular.**
- `reference_aupro.py` is a line-by-line copy of `compute_aupro` (only `num_thresholds` differs), so the "independent oracle" parity test proves nothing.
- AU-PRO uses 200 percentile thresholds pooled over all pixels (coarse in the FPR ≤ 0.3 range) and was never compared to the official MVTec evaluation script.

**B11 [MED] — The coreset "exact parity" and speedup claims don't hold.**
- The batched method picks the top-50 farthest points per step. That is **not** greedy k-center: the 50 points tend to cluster, and the 2-approximation guarantee is lost.
- The quality check uses isotropic Gaussian data at D = 128, where every distance concentrates (random sampling gets 15.37 vs 14.93). Even there, radius is 14.9247 vs 14.9339, which is not "exact".
- The benchmarked function (random start, no projection) is not the one shipped in `patchcore.py` (start at the point farthest from the mean, random projection).
- "Without loss in AU-PRO" (in the docs) was never measured.

**B12 [MED] — The latency profiler doesn't measure what it claims.**
- T_model is timed on `torch.randn` input.
- T_e2e reuses one constant grey 256×256 PIL image: no disk read and no resize from 700–1024 px, despite the docstring "Image read → …".

**B13 [MED] — The autoencoder baseline is below chance and goes unexamined.**
- AUROC is 0.30 (metal_nut), 0.39 (carpet), 0.45 (cable): worse than random, which points to a scoring problem. The paper doesn't flag it.
- Trained 20 epochs (`run_benchmark.py:31`) vs 40 in the config and docs.
- `docs/reference-validation.md` describes a 256-channel latent and a final Conv2d; the code uses a 128-channel latent and a final ConvTranspose.
- "Training < 1 s" is claimed, but recorded fit times are 14–265 s.

**B14 [MED] — Robustness severities are not comparable across categories.**
- Corruptions are applied at native resolution (1024 / 900 / 700 / 256 px) before resizing to 256, so blur kernels and "downscale to 128/64/32" differ up to 4× in effective strength between categories.
- `CorruptedMVTecTest` resizes masks bilinearly then thresholds, while `MVTecTest` uses nearest-neighbour: two different ground truths.
- "mCE" is aliased to the signed mean change, which is not the Hendrycks mCE (no baseline normalisation).

**B15 [MED] — The failure-case catalog mislabels cases.**
- "False positives" are simply the top-scoring nominals: one shown has score 1.42 < τ 1.72, so it is *not* an FP.
- "False negatives" include detected samples (score = τ).
- "corruption_failures" are just borderline samples; no corruption is involved.
- The threshold is the oracle F1 on the test set.

**B16 [MED] — The ECE is not meaningful.** Scores are min-max-normalised and called "confidence". `compute_ece` guesses which argument is the labels from the value sets.

### C. Code bugs

| # | Location | Bug |
|---|---|---|
| C1 | `src/experiments/cct_ablation.py:128` | `compute_cost_weighted_error(eval_y, eval_y, …)` passes labels as scores, so `cwe_oracle_r10` is garbage (7.67, 0.0, …). |
| C2 | `src/methods/padim.py:54-56` | The `cov` property returns the **inverse** covariance. |
| C3 | `src/metrics/operational.py:193-207` | Window mode switches between block and sliding depending on whether `n % window == 0`; `num_windows` is wrong in sliding mode. |
| C4 | `scripts/run_benchmark.py:166-168` | `runs_master.csv` is appended to on every run, which causes the duplicates and mixed versions in B8. `generate_report.py` reads it with `on_bad_lines="skip"`, silently dropping malformed rows. |
| C5 | `scripts/run_benchmark.py:24-33` | Seed never reaches PatchCore / PaDiM (B4). |
| C6 | `scripts/run_ablations.py:68` | Wrong npz key, so fake maps are used (A6). |
| C7 | `src/metrics/calibration.py:74-81` | `fit_isotonic_calibration` ignores `nominal_scores`. |
| C8 | `src/mvtec.py:61-66, 101-104` | Error messages tell users to run `--mock`, which no longer exists. `download_dataset.py` defines `generate_mock_category` *after* the `if __name__ == "__main__"` block, and `--max-workers` is unused. |
| C9 | `src/mvtec.py:122-129` | A defect image with a missing mask silently gets an all-zero mask. |
| C10 | `scripts/02…05, 10*, 11` | Import modules that don't exist; crash on run. |
| C11 | `configs/benchmark.yaml`, `operational_streams.yaml` | Never loaded. All values are hardcoded elsewhere and already disagree (e.g. AE epochs 40 vs 20). |
| C12 | `publication_plots.py:497-498, 583-584` | `"\\tau"`, `"\\to"`, `"\\Delta"` outside `$…$` render literally as `\tau` (visible in the CCT and decision-shift figures). |
| C13 | `publication_plots.py:451, 288` | Wrong units: "Review > 60 items/hr" (the window is 1,000 items, not an hour); "MD per 1,000 Items" is really per 1,000 *defective* items. |
| C14 | `generate_operational_plots.py:86-93` | Per-run min-max normalisation (uses test defects), then pools all categories and seeds into one curve per method. Paper Fig. 1(a)–(c) are cross-category pools of normalised scores. |
| C15 | `image_metrics.py:23` vs `operational.py:69` | Two `compute_quantile_threshold` functions with different empty-input fallbacks (0.5 vs 0.0). |
| C16 | `scripts/reproduce_main_results.sh:5-6` | Hardcodes `/home/sengar/miniconda3/envs/vmunet/bin/python`. Never re-runs the benchmark, and calls `generate_plots.py`, which regenerates the fake figures. |

### D. Docs, README, paper and results disagree with each other

1. **README**: 5 categories, 45 runs, "37 passed", 83% coverage, zero-width CIs, p = 5.68e-14. All stale. The current results have 7 categories, non-zero CIs and p = 6.2e-4. The quick-start `download_dataset.py --max-workers 8` downloads nothing.
2. **Paper / CITATION / data card**: "7 categories, 63 runs". Reality: 5 real + 2 synthetic; 62 unique logged runs.
3. **`docs/environment.md`**: PyTorch 2.4.1+cu121, "64/64 tests, 86% coverage", driver 550.120.
   - The env the scripts actually use (`vmunet`) has torch 1.13.0 / torchvision 0.14; tests: 64 pass, 87% coverage.
   - Driver is 581.86.
   - In the base conda env (torch 2.5.1, broken torchvision) 4 tests fail.
4. **Docs**: "1,324 train / 575 test / 400 masks validated". On disk: test = 578 (with the mock images), masks = 410 (407 real); the real 5-category mask count is 407, not 400.
5. **`benchmark_report.md`** has per-category latencies and MRDs that differ from Tables 1 and 3 (it shows the undiluted values).
6. **`overleaf_paper/tables/coreset_scalability.csv`** ≠ its `.tex` (different runs).
7. **Paper**: "exact coverage radius parity", "strict false-alarm constraints", "leakage-free", "B = 2,000", "Holm-Bonferroni" — each contradicted above.

### E. Paper quality

- `main.tex` is 112 lines: a one-paragraph introduction, then section headers that each wrap a table, and a two-sentence conclusion.
- There is no CCT formulation, no related work, no dataset or protocol description, no results discussion and no limitations section. It says nothing about the synthetic categories, the ResNet-18 "inspired" baselines, or the in-sample thresholds.
- Not submittable in its current form.

### F. Hygiene

- 66 MB `flagship1_overleaf_package.zip` sits untracked in the repo root.
- Figures are duplicated 3× (`results/figures`, `results/mvtec_ad/figures`, `overleaf_paper/figures` plus an `operational/` copy). 84 failure-case PNGs are tracked twice, plus 8 more from the mock era.
- Legacy SimCLR `runs/` and mock-era `results/tables` are tracked. A partial HF cache sits in `data/mvtec_ad/.cache`.
- Many files start with a UTF-8 BOM.
- Commit messages repeatedly describe "restoring" or "enforcing" desired outcomes ("restore CCT out-of-sample dominance", "enforce CCT budget bounds", "enforce non-negative MRD", "restore deployment latencies"). Each one corresponds to a change that adjusted reported numbers rather than fixing a measurement.

### What *is* sound

- Five categories (bottle, cable, carpet, hazelnut, metal_nut) are genuine MVTec images with official counts, apart from the 3 mock bottle images.
- `operational_results.csv` reproduces **exactly** from the stored `.npz` archives with the shipped code (verified). The pipeline is deterministic; the problems are in what it measures and how it is presented.
- The code is modular and readable, and the 64 tests pass in the intended env.
- PatchCore-style > PaDiM-style > AE ordering on the 5 real categories is plausible and consistent with the literature (as a ranking, not as significance-tested).

---

## 4. What would make this defensible (priority order)

1. Delete the synthetic grid/leather data and the 3 mock bottle images. Download real grid/leather (or drop them) and rerun `run_benchmark.py` from scratch into a fresh `runs_master.csv` (overwrite, don't append), with profiling and robustness enabled in the **same** run.
2. Remove every hardcoded or fallback number: `generate_report.py` empirical_rows and macros, `generate_plots.py` sample/hardcoded figures, plotting fallbacks, coreset extrapolation, the `min(5, …)` clamp, `budget_satisfied=True`. Fail loudly when inputs are missing.
3. Fix the seed plumbing (pass `seed` into PatchCore / PaDiM, or say plainly that only the AE is stochastic). Treat categories as the independent unit for tests, and report that n = 5 cannot reach p < 0.05 with Wilcoxon.
4. Use one CWE definition everywhere, decided up front.
5. Report the realized FA/1k next to every "@ budget" number. State that per-1k rates cannot be resolved from 20–60 nominal images; use the train-good images (held-out split) as nominal calibration data so thresholds never touch test labels.
6. Make the coreset claim honest: measure every cell, or label extrapolations; compare against true greedy on real PatchCore features; report the coverage radius gap.
7. Fix the aggregation-ablation key and rerun.
8. Fix the bibliography, rewrite the README/docs from the actual outputs, and write the missing paper sections, including limitations.

---

## 5. Verification log (what was actually run)

- **Image inspection** (PIL): grid/leather = 256×256 RGB synthetic; real categories 700–1024 px. Visual contact sheet confirms drawn lattices and noise textures; `bottle/test/defect` = purple squares.
- **`pytest` in `vmunet`** (torch 1.13): `64 passed`, coverage 87%. **Base env** (torch 2.5.1): 4 failed (`torchvision.models` has no `ResNet18_Weights`).
- **`run_operational_eval.py` rerun** into scratch: `operational_results.csv` identical to the committed file; Wilcoxon p = 6.2255e-04 / 6.3215e-04.
- **Realized FA/1k at the "5/1k" threshold**: 50.0 / 17.2 / 35.7 / 25.0 / 45.5 / 33.3 / 33.3.
- **Seed independence**, max |Δscore| seed 42 vs 123: PaDiM 5.7e-6; PatchCore 3.2e-2 (hazelnut only; cable 0.0); AE 0.37.
- **Wilcoxon, 5 real categories × 1 seed**: p = 0.0625 (both comparisons).
- **CCT unclamped**: 21/63 runs with eval FA > 5/1k, max 300/1k; 6 runs with calibration FPR > budget yet `budget_satisfied=True`; calibration nominal counts {10, 11, 14, 15, 20, 29}.
- **Bottle without mock images**: AE AUROC 0.519 → 0.496; PaDiM 0.9970 → 0.9968.
- **Real aggregation ablation**: 0.836 / 0.847 / 0.783 / 0.848 / 0.839 (vs the fake about 0.858 for all).
- **Coreset formula**: 5.65625 / 11.3125 / 22.625 / 22.625 / 45.25 / 90.5 and 1.3416407865 match the committed table exactly.
- **Hardcoded deployment row** = mean of the 3 bottle PatchCore runs from 08-17 (10.94 ms, 29.89 ms, 205.9 MB).
- **Git**: `1598d37` removed the HF downloader and added the synthetic generators; `affb065` added the hardcoded latency fallback; `ec1930b` added the FA clamp and switched the CCT CWE formula.
