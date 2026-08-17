import os
import pandas as pd
import numpy as np
import pytest

from src.visualization.publication_plots import (
    plot_pareto_frontier,
    plot_robustness_heatmap,
    plot_calibration_curve,
    plot_robust_training_ablation
)
from scripts.generate_report import (
    generate_latex_main_results,
    generate_latex_profiling,
    generate_latex_robustness,
    generate_markdown_report
)


def test_plotting_functions(tmp_path):
    """
    Verifies all 4 publication plotting utilities write valid non-empty PNG files.
    """
    # 1. Pareto Frontier
    df_master = pd.DataFrame([
        {"method": "patchcore", "category": "bottle", "aupro": 0.96, "p50_latency_ms": 18.0},
        {"method": "padim", "category": "bottle", "aupro": 0.94, "p50_latency_ms": 10.0},
        {"method": "autoencoder", "category": "bottle", "aupro": 0.72, "p50_latency_ms": 5.0},
    ])
    p1 = str(tmp_path / "pareto.png")
    out1 = plot_pareto_frontier(df_master, p1)
    assert os.path.exists(out1)
    assert os.path.getsize(out1) > 0

    # 2. Robustness Heatmap
    df_rob = pd.DataFrame([
        {"corruption_type": "gaussian_blur", "severity": 1, "delta_image_auroc": 0.02},
        {"corruption_type": "gaussian_blur", "severity": 2, "delta_image_auroc": 0.05},
        {"corruption_type": "gaussian_blur", "severity": 3, "delta_image_auroc": 0.09},
    ])
    p2 = str(tmp_path / "heatmap.png")
    out2 = plot_robustness_heatmap(df_rob, p2)
    assert os.path.exists(out2)
    assert os.path.getsize(out2) > 0

    # 3. Calibration Curve
    rel_data = {
        "bin_centers": [0.2, 0.5, 0.8],
        "bin_accuracies": [0.22, 0.48, 0.81],
        "bin_confidences": [0.20, 0.50, 0.80],
        "bin_counts": [10, 20, 15]
    }
    p3 = str(tmp_path / "calibration.png")
    out3 = plot_calibration_curve(rel_data, ece_score=0.025, output_path=p3)
    assert os.path.exists(out3)
    assert os.path.getsize(out3) > 0

    # 4. Robust Training Ablation
    ablation_data = {
        "clean_model": {"clean_auroc": 0.98, "mCE_auroc": 0.12},
        "robust_model": {"clean_auroc": 0.97, "mCE_auroc": 0.05}
    }
    p4 = str(tmp_path / "ablation.png")
    out4 = plot_robust_training_ablation(ablation_data, p4)
    assert os.path.exists(out4)
    assert os.path.getsize(out4) > 0


def test_report_generation(tmp_path):
    """
    Verifies report generator creates valid LaTeX table syntax and markdown document.
    """
    df = pd.DataFrame([
        {"method": "patchcore", "category": "bottle", "image_auroc_mean": 0.998, "pixel_auroc_mean": 0.985, "aupro_mean": 0.962, "max_f1": 0.99}
    ])

    tex1 = str(tmp_path / "main.tex")
    tex2 = str(tmp_path / "prof.tex")
    tex3 = str(tmp_path / "rob.tex")
    md = str(tmp_path / "report.md")

    generate_latex_main_results(df, tex1)
    generate_latex_profiling(df, tex2)
    generate_latex_robustness(tex3)
    generate_markdown_report(md)

    assert os.path.exists(tex1) and r"\begin{tabular}" in open(tex1).read()
    assert os.path.exists(tex2) and r"\toprule" in open(tex2).read()
    assert os.path.exists(tex3) and r"\bottomrule" in open(tex3).read()
    assert os.path.exists(md) and "# Flagship Benchmark Report" in open(md).read()
