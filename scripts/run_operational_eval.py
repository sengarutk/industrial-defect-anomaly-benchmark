import argparse
import os
import sys
import glob
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd

# Ensure repository root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.metrics.operational import (
    compute_fa_at_1k,
    compute_md_at_1k,
    compute_cost_weighted_error,
    compute_tpr_at_alert_budget,
    compute_operator_overload
)
from src.metrics.stats import bootstrap_ci, compute_wilcoxon_significance
from src.metrics.image_metrics import compute_quantile_threshold
from src.experiments.operational_eval import ProductionStreamSimulator


def generate_latex_table(df_summary: pd.DataFrame, output_tex: str):
    os.makedirs(os.path.dirname(output_tex), exist_ok=True)
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\small",
        "\\caption{Operational Inspection Benchmark under Constrained Operator Alert Budgets and Asymmetric Escape Costs (95\\% Bootstrap Confidence Intervals).}",
        "\\label{tab:operational_results}",
        "\\begin{tabular}{llcccc}",
        "\\toprule",
        "\\textbf{Category} & \\textbf{Method} & \\textbf{TPR @ 5 Alarms/1k} $\\uparrow$ & \\textbf{MD @ 1k (Escapes)} $\\downarrow$ & \\textbf{CWE (=10$)} $\\downarrow$ & \\textbf{P(Overload)} $\\downarrow$ \\\\",
        "\\midrule"
    ]

    cats = sorted(df_summary["category"].unique())
    for cat in cats:
        cat_df = df_summary[df_summary["category"] == cat]
        for _, row in cat_df.iterrows():
            m_name = str(row["method"]).capitalize()
            tpr_str = f"{row['tpr_at_5_mean']:.3f} [{row['tpr_at_5_ci_low']:.3f}, {row['tpr_at_5_ci_high']:.3f}]"
            md_str = f"{row['md_at_1k_mean']:.1f} [{row['md_at_1k_ci_low']:.1f}, {row['md_at_1k_ci_high']:.1f}]"
            cwe_str = f"{row['cwe_r10_mean']:.4f} [{row['cwe_r10_ci_low']:.4f}, {row['cwe_r10_ci_high']:.4f}]"
            ovl_str = f"{row['overload_prob_mean']:.3f}"
            lines.append(f"{cat} & {m_name} & {tpr_str} & {md_str} & {cwe_str} & {ovl_str} \\\\")
        lines.append("\\midrule")

    if lines[-1] == "\\midrule":
        lines.pop()

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table*}"
    ])

    with open(output_tex, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"✅ Generated LaTeX Table: {output_tex}")


def main():
    parser = argparse.ArgumentParser(description="Master Operational Evaluation and Production Stream Runner")
    parser.add_argument("--scores-dir", type=str, default="results/mvtec_ad/scores")
    parser.add_argument("--output-dir", type=str, default="results/mvtec_ad")
    parser.add_argument("--priors", nargs="+", type=float, default=[0.01, 0.05, 0.15])
    parser.add_argument("--cost-ratios", nargs="+", type=float, default=[10.0, 20.0, 50.0])
    parser.add_argument("--alert-budgets", nargs="+", type=float, default=[5.0, 10.0])
    parser.add_argument("--n-stream", type=int, default=10000)
    args = parser.parse_args()

    tables_dir = os.path.join(args.output_dir, "tables")
    os.makedirs(tables_dir, exist_ok=True)

    npz_files = sorted(glob.glob(os.path.join(args.scores_dir, "*.npz")))
    if len(npz_files) == 0:
        print(f"❌ Error: No .npz files found in {args.scores_dir}. Please run 'python scripts/run_benchmark.py --save-scores' first.")
        sys.exit(1)

    print(f"=== Starting Operational Evaluation across {len(npz_files)} Score Archives ===")
    print(f"Defect Priors: {args.priors}")
    print(f"Cost Ratios: {args.cost_ratios}")
    print(f"Alert Budgets: {args.alert_budgets} per 1k\n")

    eval_rows = []

    for fpath in npz_files:
        fname = os.path.basename(fpath).replace(".npz", "")
        parts = fname.split("_")
        if len(parts) >= 3:
            category = "_".join(parts[:-2])
            method = parts[-2]
            seed = int(parts[-1])
        else:
            category, method, seed = parts[0], parts[1], 42

        data = np.load(fpath)
        image_labels = data["image_labels"]
        image_scores = data["image_scores"]

        norm_scores = image_scores[image_labels == 0]
        def_scores = image_scores[image_labels == 1]

        # 1. Alert-Budget Constrained TPR
        tpr_at_5_res = compute_tpr_at_alert_budget(image_labels, image_scores, max_alerts_per_1k=5.0)
        tpr_at_10_res = compute_tpr_at_alert_budget(image_labels, image_scores, max_alerts_per_1k=10.0)

        # 2. Threshold evaluation at 99% nominal quantile
        tau_99 = compute_quantile_threshold(norm_scores, quantile=0.99)
        fa_at_1k_99 = compute_fa_at_1k(image_labels, image_scores, threshold=tau_99)
        md_at_1k_99 = compute_md_at_1k(image_labels, image_scores, threshold=tau_99)
        cwe_r10 = compute_cost_weighted_error(image_labels, image_scores, threshold=tau_99, cost_ratio=10.0)
        cwe_r20 = compute_cost_weighted_error(image_labels, image_scores, threshold=tau_99, cost_ratio=20.0)
        cwe_r50 = compute_cost_weighted_error(image_labels, image_scores, threshold=tau_99, cost_ratio=50.0)

        # 3. Production Stream Simulation
        sim = ProductionStreamSimulator(norm_scores, def_scores, seed=seed)
        for prior in args.priors:
            stream_labels, stream_scores = sim.simulate_stream(n_total=args.n_stream, defect_prior=prior)
            strat_results = sim.evaluate_threshold_strategies(
                stream_labels=stream_labels,
                stream_scores=stream_scores,
                nominal_ref_scores=norm_scores,
                cost_ratio=10.0,
                defect_prior=prior
            )

            # Operator Overload Check
            alert_stream = (stream_scores >= tau_99).astype(int)
            overload_metrics = compute_operator_overload(
                alert_stream,
                operator_capacity_per_window=60,
                window_size=1000
            )

            row = {
                "category": category,
                "method": method,
                "seed": seed,
                "defect_prior": prior,
                "tau_99": tau_99,
                "tpr_at_5_budget": tpr_at_5_res["tpr"],
                "tpr_at_10_budget": tpr_at_10_res["tpr"],
                "fa_at_1k": fa_at_1k_99,
                "md_at_1k": md_at_1k_99,
                "cwe_r10": cwe_r10,
                "cwe_r20": cwe_r20,
                "cwe_r50": cwe_r50,
                "stream_cwe_oracle": strat_results["oracle_f1"]["cwe"],
                "stream_cwe_tau99": strat_results["nominal_quantile_99"]["cwe"],
                "stream_cwe_cost_opt": strat_results["cost_optimal"]["cwe"],
                "stream_tpr_cost_opt": strat_results["cost_optimal"]["tpr"],
                "mean_review_load": overload_metrics["mean_load"],
                "peak_review_load": overload_metrics["peak_load"],
                "overload_probability": overload_metrics["overload_probability"]
            }
            eval_rows.append(row)

    runs_df = pd.DataFrame(eval_rows)

    # Statistical Aggregation with Bootstrap 95% CIs
    summary_rows = []
    for (cat, m), group in runs_df[runs_df["defect_prior"] == 0.01].groupby(["category", "method"]):
        tpr5_m, tpr5_l, tpr5_h = bootstrap_ci(group["tpr_at_5_budget"].values)
        tpr10_m, tpr10_l, tpr10_h = bootstrap_ci(group["tpr_at_10_budget"].values)
        md_m, md_l, md_h = bootstrap_ci(group["md_at_1k"].values)
        fa_m, fa_l, fa_h = bootstrap_ci(group["fa_at_1k"].values)
        cwe_m, cwe_l, cwe_h = bootstrap_ci(group["cwe_r10"].values)
        ovl_m, ovl_l, ovl_h = bootstrap_ci(group["overload_probability"].values)

        summary_rows.append({
            "category": cat,
            "method": m,
            "tpr_at_5_mean": tpr5_m,
            "tpr_at_5_ci_low": tpr5_l,
            "tpr_at_5_ci_high": tpr5_h,
            "tpr_at_10_mean": tpr10_m,
            "tpr_at_10_ci_low": tpr10_l,
            "tpr_at_10_ci_high": tpr10_h,
            "md_at_1k_mean": md_m,
            "md_at_1k_ci_low": md_l,
            "md_at_1k_ci_high": md_h,
            "fa_at_1k_mean": fa_m,
            "cwe_r10_mean": cwe_m,
            "cwe_r10_ci_low": cwe_l,
            "cwe_r10_ci_high": cwe_h,
            "overload_prob_mean": ovl_m
        })

    summary_df = pd.DataFrame(summary_rows)

    # Save output tables
    out_csv = os.path.join(tables_dir, "operational_results.csv")
    out_md = os.path.join(tables_dir, "operational_results.md")
    out_tex = os.path.join(tables_dir, "operational_results.tex")

    summary_df.to_csv(out_csv, index=False)

    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# Operational Inspection Benchmark Summary (95% Bootstrap CIs)\n\n")
        f.write(summary_df.to_markdown(index=False))
        f.write("\n\n## Statistical Significance Tests (Wilcoxon Signed-Rank)\n\n")

        # Compute Wilcoxon Significance between methods
        m1 = runs_df[runs_df["method"] == "patchcore"]["cwe_r10"].values
        m2 = runs_df[runs_df["method"] == "padim"]["cwe_r10"].values
        m3 = runs_df[runs_df["method"] == "autoencoder"]["cwe_r10"].values

        min_len = min(len(m1), len(m2), len(m3))
        if min_len > 0:
            w_pc_padim = compute_wilcoxon_significance(m1[:min_len], m2[:min_len])
            w_pc_ae = compute_wilcoxon_significance(m1[:min_len], m3[:min_len])
            f.write(f"- **PatchCore vs. PaDiM (CWE r=10):** W = {w_pc_padim['statistic']:.1f}, p = {w_pc_padim['p_value']:.4e}\n")
            f.write(f"- **PatchCore vs. Autoencoder (CWE r=10):** W = {w_pc_ae['statistic']:.1f}, p = {w_pc_ae['p_value']:.4e}\n")

    generate_latex_table(summary_df, out_tex)

    print(f"\n✅ Operational Evaluation Complete! Results written to {tables_dir}")


if __name__ == "__main__":
    main()
