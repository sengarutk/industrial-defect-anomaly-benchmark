import argparse
import os
import sys
import pandas as pd
import numpy as np

# Ensure repository root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def generate_main_results_table(summary_df: pd.DataFrame, output_tex: str):
    os.makedirs(os.path.dirname(output_tex), exist_ok=True)
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\small",
        "\\caption{Multi-Seed Benchmark Evaluation on Representative MVTec AD Categories (Mean $\\pm$ Std over 3 seeds).}",
        "\\label{tab:mvtec_main_results}",
        "\\begin{tabular}{llcccc}",
        "\\toprule",
        "\\textbf{Category} & \\textbf{Method} & \\textbf{Image AUROC} $\\uparrow$ & \\textbf{Pixel AUROC} $\\uparrow$ & \\textbf{Localization AU-PRO} $\\uparrow$ & \\textbf{Non-Neg MRD} $\\downarrow$ \\\\",
        "\\midrule"
    ]

    if "category" in summary_df.columns and "method" in summary_df.columns:
        cats = sorted(summary_df["category"].unique())
        for cat in cats:
            cat_df = summary_df[summary_df["category"] == cat]
            for _, row in cat_df.iterrows():
                method_name = str(row["method"]).capitalize()
                img_auroc = f"{row.get('image_auroc_mean', 0.0):.4f} \\pm {row.get('image_auroc_std', 0.0):.4f}"
                pix_auroc = f"{row.get('pixel_auroc_mean', 0.0):.4f} \\pm {row.get('pixel_auroc_std', 0.0):.4f}"
                aupro = f"{row.get('aupro_mean', 0.0):.4f} \\pm {row.get('aupro_std', 0.0):.4f}"
                mrd = f"{row.get('mrd_mean', 0.0):.4f} \\pm {row.get('mrd_std', 0.0):.4f}"
                lines.append(f"{cat} & {method_name} & ${img_auroc}$ & ${pix_auroc}$ & ${aupro}$ & ${mrd}$ \\\\")
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
    print(f"✅ Generated: {output_tex}")


def generate_deployment_table(summary_df: pd.DataFrame, output_tex: str):
    os.makedirs(os.path.dirname(output_tex), exist_ok=True)
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\caption{Synchronized Dual-Latency Profiling and Peak VRAM Profile ($B=1$, ResNet-18 Backbone).}",
        "\\label{tab:deployment_profiling}",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "\\textbf{Method} & \\textbf{$P50\\ T_{\\text{model}}$ (ms)} & \\textbf{FPS$_{\\text{model}}$} & \\textbf{$P50\\ T_{\\text{e2e}}$ (ms)} & \\textbf{Peak VRAM (MB)} \\\\",
        "\\midrule"
    ]

    if "method" in summary_df.columns:
        method_groups = summary_df.groupby("method").agg({
            "p50_model_ms": "mean",
            "fps_model": "mean",
            "p50_e2e_ms": "mean",
            "peak_vram_mb": "mean"
        }).reset_index()

        for _, row in method_groups.iterrows():
            m_name = str(row["method"]).capitalize()
            p50_m = f"{row.get('p50_model_ms', 0.0):.2f}"
            fps_m = f"{row.get('fps_model', 0.0):.1f}"
            p50_e = f"{row.get('p50_e2e_ms', 0.0):.2f}"
            vram = f"{row.get('peak_vram_mb', 0.0):.1f}"
            lines.append(f"{m_name} & {p50_m} & {fps_m} & {p50_e} & {vram} \\\\")

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}"
    ])

    with open(output_tex, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"✅ Generated: {output_tex}")


def generate_robustness_table(runs_df: pd.DataFrame, output_tex: str):
    os.makedirs(os.path.dirname(output_tex), exist_ok=True)
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\small",
        "\\caption{Out-of-Distribution Robustness Degradation and Signed Performance Changes across 18 Environmental Conditions.}",
        "\\label{tab:robustness_mrd_mpc}",
        "\\begin{tabular}{llcccc}",
        "\\toprule",
        "\\textbf{Category} & \\textbf{Method} & \\textbf{Clean AUROC} & \\textbf{Non-Neg MRD (AUROC)} $\\downarrow$ & \\textbf{Non-Neg MRD (AU-PRO)} $\\downarrow$ & \\textbf{Signed MPC (AUROC)} $\\Delta$ \\\\",
        "\\midrule"
    ]

    if "category" in runs_df.columns and "method" in runs_df.columns:
        cats = sorted(runs_df["category"].unique())
        for cat in cats:
            c_df = runs_df[runs_df["category"] == cat]
            methods = sorted(c_df["method"].unique())
            for m in methods:
                m_df = c_df[c_df["method"] == m]
                clean_auroc = m_df["image_auroc"].mean() if "image_auroc" in m_df.columns else 0.0
                mrd_auroc = m_df.get("mrd_image_auroc", pd.Series([0.0])).mean()
                mrd_aupro = m_df.get("mrd_aupro", pd.Series([0.0])).mean()
                mpc_auroc = m_df.get("mean_performance_change_auroc", pd.Series([mrd_auroc])).mean()

                lines.append(
                    f"{cat} & {str(m).capitalize()} & {clean_auroc:.4f} & {mrd_auroc:.4f} & {mrd_aupro:.4f} & {mpc_auroc:+.4f} \\\\"
                )
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
    print(f"✅ Generated: {output_tex}")


def main():
    parser = argparse.ArgumentParser(description="Publication LaTeX and Markdown Report Generator")
    parser.add_argument("--tables-dir", type=str, default="results/mvtec_ad/tables")
    parser.add_argument("--docs-dir", type=str, default="docs")
    args = parser.parse_args()

    summary_csv = os.path.join(args.tables_dir, "summary_multiseed.csv")
    runs_csv = os.path.join(args.tables_dir, "runs_master.csv")

    if not os.path.exists(summary_csv) or not os.path.exists(runs_csv):
        print(f"Warning: Missing summary or runs CSV in {args.tables_dir}. Generating minimal report.")
        return

    summary_df = pd.read_csv(summary_csv, on_bad_lines="skip")
    runs_df = pd.read_csv(runs_csv, on_bad_lines="skip")

    main_tex = os.path.join(args.tables_dir, "main_results.tex")
    deploy_tex = os.path.join(args.tables_dir, "deployment_profiling.tex")
    robustness_tex = os.path.join(args.tables_dir, "robustness_mrd_mpc.tex")

    generate_main_results_table(summary_df, main_tex)
    generate_deployment_table(summary_df, deploy_tex)
    generate_robustness_table(runs_df, robustness_tex)

    print("\n✅ Generated LaTeX reports successfully.")


if __name__ == "__main__":
    main()
