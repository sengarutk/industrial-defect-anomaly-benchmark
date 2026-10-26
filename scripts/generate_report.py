"""Builds LaTeX tables, LaTeX macros and docs/benchmark_report.md from the measured tables.

Every number in the paper and report comes from results/<dataset>/tables/*.csv.
Missing required inputs raise an error; optional inputs (latency, coreset
benchmark) are reported as "not measured" instead of being filled in.
"""
import argparse
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.methods import METHOD_NAMES  # noqa: E402
from src.mvtec import MVTEC_MANIFEST  # noqa: E402
from src.utils import load_config  # noqa: E402

METHODS = ["patchcore", "padim", "autoencoder"]
CAT_LABEL = lambda c: c.replace("_", " ").title()  # noqa: E731


def need(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise SystemExit(f"Required input missing: {path}. Run the pipeline steps before generate_report.py.")
    return pd.read_csv(path)


def pm(mean: float, std: float, nd: int = 3) -> str:
    return f"${mean:.{nd}f} \\pm {std:.{nd}f}$"


def write(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def table(caption: str, label: str, colspec: str, header: str, body: list, wide: bool = True) -> str:
    env = "table*" if wide else "table"
    return "\n".join([f"\\begin{{{env}}}[t]", "\\centering", "\\small", f"\\caption{{{caption}}}",
                      f"\\label{{{label}}}", "\\resizebox{\\linewidth}{!}{%", f"\\begin{{tabular}}{{{colspec}}}",
                      "\\toprule", header + " \\\\", "\\midrule", *body, "\\bottomrule", "\\end{tabular}}",
                      f"\\end{{{env}}}", ""])


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate LaTeX tables, macros and the markdown report")
    parser.add_argument("--config", default="configs/benchmark.yaml")
    parser.add_argument("--operational-config", default="configs/operational.yaml")
    parser.add_argument("--results-dir", default="results/mvtec_ad")
    parser.add_argument("--docs-dir", default="docs")
    args = parser.parse_args()
    cfg = load_config(args.config)
    op = load_config(args.operational_config)
    T = os.path.join(args.results_dir, "tables")
    L = os.path.join(T, "latex")

    runs = need(os.path.join(T, "runs_master.csv"))
    cats = [c for c in cfg["dataset"]["categories"] if c in set(runs.category)]
    seeds = sorted(runs.seed.unique())
    expected = len(cfg["dataset"]["categories"]) * len(cfg["seeds"]) * len(cfg["methods"])
    if len(runs) != expected:
        raise SystemExit(f"runs_master.csv has {len(runs)} runs, expected {expected}; the sweep is incomplete.")
    op_sum = need(os.path.join(T, "operational_summary.csv"))
    op_runs = need(os.path.join(T, "operational_runs.csv"))
    sig_cost = need(os.path.join(T, "significance_cost.csv"))
    cct = need(os.path.join(T, "cct_summary.csv"))
    sig_cct = need(os.path.join(T, "significance_cct.csv"))
    dec = need(os.path.join(T, "cct_decisions.csv"))
    agg = need(os.path.join(T, "aggregation_summary.csv"))
    lat_path = os.path.join(T, "latency.csv")
    lat = pd.read_csv(lat_path) if os.path.exists(lat_path) else None
    core_path = os.path.join(T, "coreset_benchmark.csv")
    core = pd.read_csv(core_path) if os.path.exists(core_path) else None

    b = op["primary_alert_budget_per_1k"]
    r = op["primary_cost_ratio"]
    prior = op["deployment_prior"]
    B = op["statistics"]["bootstrap_resamples"]
    cost_col = f"expected_cost_r{int(r)}"
    n_seeds = len(seeds)

    # ---------------- Table: main results ----------------
    g = runs.groupby(["category", "method"])
    body = []
    for c in cats:
        for m in METHODS:
            sub = runs[(runs.category == c) & (runs.method == m)]
            body.append(f"{CAT_LABEL(c)} & {METHOD_NAMES[m]} & {pm(sub.image_auroc.mean(), sub.image_auroc.std())} & "
                        f"{pm(sub.pixel_auroc.mean(), sub.pixel_auroc.std())} & {pm(sub.aupro.mean(), sub.aupro.std())} \\\\")
        body.append("\\midrule")
    cat_means = g[["image_auroc", "pixel_auroc", "aupro"]].mean().reset_index()
    for m in METHODS:
        s = cat_means[cat_means.method == m]
        body.append(f"\\textit{{Mean}} & {METHOD_NAMES[m]} & {pm(s.image_auroc.mean(), s.image_auroc.std())} & "
                    f"{pm(s.pixel_auroc.mean(), s.pixel_auroc.std())} & {pm(s.aupro.mean(), s.aupro.std())} \\\\")
    write(os.path.join(L, "main_results.tex"), table(
        f"Clean test results on {len(cats)} MVTec AD categories. Per category: mean $\\pm$ std over {n_seeds} seeds "
        f"(seeds change the train/calibration split, PatchCore's projection and start point, PaDiM's channel "
        f"subset and the autoencoder's initialisation). \\textit{{Mean}} rows: mean $\\pm$ std over categories. "
        f"AU-PRO is integrated up to FPR $=0.3$ with the exact MVTec protocol.",
        "tab:main_results", "llccc",
        "\\textbf{Category} & \\textbf{Method} & \\textbf{Image AUROC} & \\textbf{Pixel AUROC} & \\textbf{AU-PRO}", body))

    # ---------------- Table: robustness ----------------
    n_cond = len(cfg["robustness"]["corruptions"]) * len(cfg["robustness"]["severities"])
    body = []
    for c in cats:
        for m in METHODS:
            sub = runs[(runs.category == c) & (runs.method == m)]
            body.append(f"{CAT_LABEL(c)} & {METHOD_NAMES[m]} & {sub.mrd_image_auroc.mean():.3f} & "
                        f"{sub.mrd_aupro.mean():.3f} & ${sub.mpc_image_auroc.mean():+.3f}$ \\\\")
        body.append("\\midrule")
    rob_means = g[["mrd_image_auroc", "mrd_aupro", "mpc_image_auroc"]].mean().reset_index()
    for m in METHODS:
        s = rob_means[rob_means.method == m]
        body.append(f"\\textit{{Mean}} & {METHOD_NAMES[m]} & {s.mrd_image_auroc.mean():.3f} & {s.mrd_aupro.mean():.3f} & "
                    f"${s.mpc_image_auroc.mean():+.3f}$ \\\\")
    write(os.path.join(L, "robustness.tex"), table(
        f"Robustness over {n_cond} corruption conditions (6 types $\\times$ 3 severities, applied after resizing to "
        f"{cfg['dataset']['img_size']} px). MRD $=\\frac{{1}}{{{n_cond}}}\\sum_i\\max(0,M_{{clean}}-M_i)$; "
        f"MPC is the signed mean. Values are means over {n_seeds} seeds.",
        "tab:robustness", "llccc",
        "\\textbf{Category} & \\textbf{Method} & \\textbf{MRD (AUROC)} $\\downarrow$ & \\textbf{MRD (AU-PRO)} $\\downarrow$ "
        "& \\textbf{MPC (AUROC)}", body))

    # ---------------- Table: operational ----------------
    body = []
    for c in cats:
        for m in METHODS:
            row = op_sum[(op_sum.category == c) & (op_sum.method == m)].iloc[0]
            body.append(f"{CAT_LABEL(c)} & {METHOD_NAMES[m]} & {row.tpr:.3f} [{row.tpr_ci_low:.3f}, {row.tpr_ci_high:.3f}] & "
                        f"{row.test_fa_at_1k:.1f} [{row.test_fa_ci_low:.1f}, {row.test_fa_ci_high:.1f}] & "
                        f"{row.test_resolution_per_1k:.1f} & {row[cost_col]:.4f} \\\\")
        body.append("\\midrule")
    body.pop()
    n_cal = runs.groupby("category").n_calib.first()
    write(os.path.join(L, "operational.tex"), table(
        f"Operating point at a target budget of {b:g} false alarms per 1,000 nominal parts. The threshold is set on "
        f"held-out train/good images only ({n_cal.min()}--{n_cal.max()} per category) and applied to the full test "
        f"set. Brackets: 95\\% two-stage bootstrap intervals over seeds and test images ($B={B}$). "
        f"``Resolution'' is the smallest non-zero rate the test set can express (1000 / nominal test images). "
        f"Expected cost uses defect prior $\\pi={prior:g}$ and cost ratio $r={r:g}$.",
        "tab:operational", "llcccc",
        "\\textbf{Category} & \\textbf{Method} & \\textbf{Test TPR} & \\textbf{Test FA/1k} & \\textbf{Resolution/1k} & "
        "\\textbf{Exp. cost}", body))

    # ---------------- Table: CCT ----------------
    sub = cct[cct.cost_ratio == r]
    macro = sub.groupby(["method", "strategy"]).agg(cost=("eval_cost", "mean"), cost_sd=("eval_cost", "std"),
                                                    fa=("eval_fa_at_1k", "mean"),
                                                    viol=("budget_violation_rate", "mean")).reset_index()
    strat_label = {"nominal_budget": "Nominal-only budget", "cct_budget": "CCT (budget)",
                   "cct_unconstrained": "CCT (no budget)", "oracle": "Oracle (in-sample)"}
    body = []
    for m in METHODS:
        for s in ["nominal_budget", "cct_budget", "cct_unconstrained", "oracle"]:
            row = macro[(macro.method == m) & (macro.strategy == s)].iloc[0]
            p = ""
            if s in ("cct_budget", "cct_unconstrained"):
                t = sig_cct[sig_cct.comparison == f"{m}: {s} vs nominal_budget"].iloc[0]
                p = f"{t.holm_adjusted_p:.3f}"
            body.append(f"{METHOD_NAMES[m]} & {strat_label[s]} & {pm(row.cost, row.cost_sd, 4)} & {row.fa:.1f} & "
                        f"{100 * row.viol:.0f}\\% & {p} \\\\")
        body.append("\\midrule")
    body.pop()
    n_splits = op["cct"]["n_splits"]
    write(os.path.join(L, "cct.tex"), table(
        f"Out-of-sample threshold selection, cost ratio $r={r:g}$, prior $\\pi={prior:g}$, budget {b:g}/1k. Each run's "
        f"test set is split 50/50 (stratified) {n_splits} times; thresholds use the calibration half plus held-out "
        f"train/good nominals; metrics are measured on the other half. Cost: mean $\\pm$ std over categories (each "
        f"category averaged over seeds and splits). ``Violations'': share of evaluations whose realised FA/1k exceeds "
        f"the budget. $p$: Holm-adjusted two-sided Wilcoxon signed-rank test against the nominal-only budget rule "
        f"with categories as units ($n={len(cats)}$, smallest attainable $p={2 / 2 ** len(cats):.3f}$ before adjustment).",
        "tab:cct", "llcccc",
        "\\textbf{Method} & \\textbf{Rule} & \\textbf{Expected cost} & \\textbf{FA/1k} & \\textbf{Violations} & \\textbf{$p$ (Holm)}",
        body, wide=False))

    # ---------------- Table: method significance ----------------
    body = [f"{t.comparison.replace('patchcore', 'PatchCore').replace('padim', 'PaDiM').replace('autoencoder', 'ConvAutoencoder')} "
            f"& {t.n_nonzero:.0f} & {t.hodges_lehmann:+.4f} & {t.p_value:.3f} & {t.holm_adjusted_p:.3f} \\\\"
            for t in sig_cost.itertuples()]
    write(os.path.join(L, "significance.tex"), table(
        f"Paired comparison of expected cost ($r={r:g}$, budget {b:g}/1k) between methods, one value per category "
        f"(mean over seeds). Two-sided Wilcoxon signed-rank test; Holm adjustment over the three comparisons.",
        "tab:significance", "lcccc",
        "\\textbf{Comparison} & \\textbf{$n$} & \\textbf{Hodges--Lehmann $\\Delta$} & \\textbf{$p$} & \\textbf{$p$ (Holm)}",
        body, wide=False))

    # ---------------- Optional tables ----------------
    if lat is not None:
        lm = lat.groupby("method")[["p50_model_ms", "p95_model_ms", "p50_e2e_ms", "peak_vram_mb"]].mean()
        body = [f"{METHOD_NAMES[m]} & {lm.loc[m, 'p50_model_ms']:.2f} & {lm.loc[m, 'p95_model_ms']:.2f} & "
                f"{lm.loc[m, 'p50_e2e_ms']:.2f} & {lm.loc[m, 'peak_vram_mb']:.0f} \\\\" for m in METHODS if m in lm.index]
        write(os.path.join(L, "latency.tex"), table(
            f"Latency at batch size 1 on {lat.device.iloc[0]}, mean over {lat.category.nunique()} categories. "
            f"$T_{{model}}$: device tensor to scores and maps; $T_{{e2e}}$: PNG on disk to scores and maps.",
            "tab:latency", "lcccc", "\\textbf{Method} & \\textbf{P50 $T_{model}$ (ms)} & \\textbf{P95 $T_{model}$ (ms)} & "
            "\\textbf{P50 $T_{e2e}$ (ms)} & \\textbf{Peak VRAM (MB)}", body, wide=False))
    if core is not None:
        body = [f"{int(x.num_patches_N):,} & {int(x.feature_dim_D)} & {x.time_cpu_numpy_s:.2f} & {x.time_gpu_greedy_s:.3f} & "
                f"{x.speedup_gpu_greedy:.1f}$\\times$ & {100 * x.index_agreement_gpu_vs_cpu:.1f}\\% & "
                f"{x.radius_gpu_topk / x.radius_gpu_greedy:.3f} \\\\" for x in core.itertuples()]
        write(os.path.join(L, "coreset.tex"), table(
            f"Greedy k-center coreset selection ($M=0.1N$) on real PatchCore features, all cells measured on "
            f"{core.device.iloc[0]}. Agreement: share of selected indices identical to the NumPy reference. "
            f"Last column: coverage radius of the approximate top-50 batched variant relative to exact greedy.",
            "tab:coreset", "rrccccc", "\\textbf{$N$} & \\textbf{$D$} & \\textbf{CPU (s)} & \\textbf{GPU (s)} & "
            "\\textbf{Speedup} & \\textbf{Agreement} & \\textbf{Top-50 radius ratio}", body))

    # ---------------- Macros ----------------
    pc_mean = cat_means.groupby("method").aupro.mean()
    auc_mean = cat_means.groupby("method").image_auroc.mean()
    nominal_rows = op_runs[op_runs.budget_per_1k == b]
    cct_macro = macro.set_index(["method", "strategy"])
    rel = {m: 100 * (cct_macro.loc[(m, "cct_budget"), "cost"] / cct_macro.loc[(m, "nominal_budget"), "cost"] - 1)
           for m in METHODS}
    macros = {
        "NumCategories": len(cats), "NumSeeds": n_seeds, "NumRuns": len(runs),
        "CalibFraction": f"{100 * cfg['dataset']['calib_fraction']:.0f}\\%",
        "PrimaryBudget": f"{b:g}", "PrimaryCostRatio": f"{r:g}", "DeploymentPrior": f"{prior:g}",
        "NumSplits": n_splits, "BootstrapB": B, "NumConditions": n_cond,
        "MinTestNominal": int(runs.n_test_nominal.min()), "MaxTestNominal": int(runs.n_test_nominal.max()),
        "CoarsestResolution": f"{1000 / runs.n_test_nominal.min():.0f}",
        "MinCalibNominal": int(runs.n_calib.min()), "MaxCalibNominal": int(runs.n_calib.max()),
        "BudgetViolationShare": f"{100 * nominal_rows.budget_violated_on_test.mean():.0f}\\%",
    }
    for m, name in [("patchcore", "PatchCore"), ("padim", "PaDiM"), ("autoencoder", "AE")]:
        macros[f"MeanAUROC{name}"] = f"{auc_mean[m]:.3f}"
        macros[f"MeanAUPRO{name}"] = f"{pc_mean[m]:.3f}"
        macros[f"CCTCostChange{name}"] = f"{rel[m]:+.0f}\\%"
        macros[f"CCTViolation{name}"] = f"{100 * cct_macro.loc[(m, 'cct_budget'), 'viol']:.0f}\\%"
        macros[f"NominalViolation{name}"] = f"{100 * cct_macro.loc[(m, 'nominal_budget'), 'viol']:.0f}\\%"
    if core is not None:
        big = core[core.feature_dim_D == 128].sort_values("num_patches_N").iloc[-1]
        macros.update(CoresetSpeedup=f"{big.speedup_gpu_greedy:.0f}\\times", CoresetN=f"{int(big.num_patches_N):,}",
                      CoresetAgreement=f"{100 * big.index_agreement_gpu_vs_cpu:.1f}\\%")
    lines = ["% Generated by scripts/generate_report.py from results tables. Do not edit by hand."]
    lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in macros.items()]
    lines.append(f"\\newcommand{{\\HasLatency}}{{{'1' if lat is not None else '0'}}}")
    lines.append(f"\\newcommand{{\\HasCoreset}}{{{'1' if core is not None else '0'}}}")
    write(os.path.join(L, "macros.tex"), "\n".join(lines) + "\n")

    # ---------------- Markdown report ----------------
    def md(df: pd.DataFrame) -> str:
        return df.to_markdown(index=False, floatfmt=".4f")

    main_md = runs.groupby(["category", "method"]).agg(
        image_auroc=("image_auroc", "mean"), image_auroc_std=("image_auroc", "std"),
        pixel_auroc=("pixel_auroc", "mean"), aupro=("aupro", "mean"), aupro_std=("aupro", "std"),
        mrd_auroc=("mrd_image_auroc", "mean"), mrd_aupro=("mrd_aupro", "mean")).reset_index()
    cct_md = macro.rename(columns={"cost": "expected_cost", "cost_sd": "std_over_categories",
                                   "fa": "eval_fa_per_1k", "viol": "budget_violation_rate"})
    parts = [
        "# Benchmark report", "",
        f"Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC} by `scripts/generate_report.py` from "
        f"`{T}`. Do not edit by hand.", "",
        f"* Categories ({len(cats)}): {', '.join(cats)} — all verified against the official MVTec AD manifest.",
        f"* Methods: {', '.join(METHOD_NAMES[m] for m in METHODS)}; seeds: {', '.join(map(str, seeds))}; "
        f"runs: {len(runs)}.",
        f"* {100 * cfg['dataset']['calib_fraction']:.0f}% of train/good is held out per seed for nominal-only threshold "
        "calibration; models never see it.",
        f"* Expected cost per part: (1-π)·FPR + π·r·FNR with π = {prior:g}; primary r = {r:g}; primary budget "
        f"{b:g} false alarms per 1,000 nominal parts.", "",
        "## Clean accuracy and robustness (mean over seeds)", "", md(main_md), "",
        f"## Operating point at {b:g}/1k (threshold from held-out nominals)", "", md(op_sum), "",
        "## Method comparison (categories as units)", "", md(sig_cost), "",
        f"## Threshold rules, out of sample (r = {r:g})", "", md(cct_md), "", md(sig_cct), "",
        "## Decision changes, nominal-only budget → CCT (budget), per evaluation half", "",
        md(dec[dec.cost_ratio == r]), "",
        "## Image-score aggregation ablation (from stored pixel maps)", "", md(agg), "",
        "## Latency", "", md(lat) if lat is not None else "Not measured in this build (requires a GPU; run "
        "`scripts/profile_latency.py`).", "",
        "## Coreset selection benchmark", "", md(core) if core is not None else "Not measured in this build "
        "(requires a GPU; run `scripts/run_coreset_benchmark.py`).", "",
    ]
    write(os.path.join(args.docs_dir, "benchmark_report.md"), "\n".join(parts))

    # ---------------- Data card ----------------
    rows = []
    for c in cats:
        man = MVTEC_MANIFEST[c]
        n_train = man["train"]["good"]
        n_test = sum(man["test"].values())
        rows.append({"category": c, "train_good": n_train, "test_good": man["test"]["good"],
                     "test_defective": n_test - man["test"]["good"], "defect_types": len(man["test"]) - 1,
                     "fit_images_per_seed": int(runs[runs.category == c].n_fit.iloc[0]),
                     "calibration_images_per_seed": int(runs[runs.category == c].n_calib.iloc[0])})
    dc = pd.DataFrame(rows)
    write(os.path.join(args.docs_dir, "data_card.md"), "\n".join([
        "# Data card", "",
        "* **Dataset:** MVTec Anomaly Detection (Bergmann et al., CVPR 2019; IJCV 2021), CC BY-NC-SA 4.0.",
        "* **Source used here:** Hugging Face mirror `foersben/mvtec-ad`, downloaded by `scripts/download_dataset.py`.",
        "* **Integrity:** every category is checked against the official per-split, per-defect-type image counts in "
        "`src/mvtec.py` (`MVTEC_MANIFEST`) and every defect image must have a mask; loaders refuse anything else.",
        "* **Preprocessing:** RGB, bilinear resize to 256×256, ImageNet normalisation; masks resized with "
        "nearest-neighbour.",
        f"* **Splits:** for each seed, {100 * cfg['dataset']['calib_fraction']:.0f}% of train/good is held out as a "
        "nominal calibration set; the rest fits the model. The test split is used unchanged.", "",
        md(dc), "",
        f"Totals: {dc.train_good.sum()} train, {dc.test_good.sum() + dc.test_defective.sum()} test "
        f"({dc.test_defective.sum()} defective, each with a ground-truth mask).", ""]))
    print(f"Wrote LaTeX to {L}, report and data card to {args.docs_dir}/")


if __name__ == "__main__":
    main()
