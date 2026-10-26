"""Builds every figure from measured tables and score archives.

Figures whose inputs are missing (latency, coreset benchmark) are skipped with a
message; no figure is ever drawn from placeholder data.
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.metrics.operational import alert_budget_threshold  # noqa: E402
from src.utils import load_config, parse_run_key  # noqa: E402
from src.visualization import (plot_cct_cost, plot_coreset_scaling, plot_latency_vs_aupro,  # noqa: E402
                               plot_operator_load, plot_realized_vs_target, plot_robustness_heatmap,
                               plot_tpr_vs_budget)

BUDGET_GRID = np.unique(np.round(np.geomspace(1, 200, 25), 2))


def budget_curves(scores_dir: str):
    """Macro-average (over categories, after averaging seeds) TPR and realised FA/1k per budget."""
    rows = []
    for f in sorted(glob.glob(os.path.join(scores_dir, "*.npz"))):
        cat, method, seed = parse_run_key(f)
        d = np.load(f)
        s, y, cal = d["image_scores"], d["image_labels"], d["calib_scores"]
        for b in BUDGET_GRID:
            tau = alert_budget_threshold(cal, b)
            rows.append({"category": cat, "method": method, "seed": seed, "budget": b,
                         "tpr": np.mean(s[y == 1] >= tau), "fa": 1000 * np.mean(s[y == 0] >= tau)})
    df = pd.DataFrame(rows).groupby(["method", "category", "budget"])[["tpr", "fa"]].mean().reset_index()
    out = {}
    for m, g in df.groupby("method"):
        out[m] = g.groupby("budget").agg(tpr_mean=("tpr", "mean"), tpr_min=("tpr", "min"), tpr_max=("tpr", "max"),
                                         fa_mean=("fa", "mean")).reset_index()
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate all figures")
    parser.add_argument("--results-dir", default="results/mvtec_ad")
    parser.add_argument("--operational-config", default="configs/operational.yaml")
    args = parser.parse_args()
    op_cfg = load_config(args.operational_config)
    tables = os.path.join(args.results_dir, "tables")
    figs = os.path.join(args.results_dir, "figures")

    curves = budget_curves(os.path.join(args.results_dir, "scores"))
    print(plot_tpr_vs_budget(curves, os.path.join(figs, "tpr_vs_alert_budget.png")))
    print(plot_realized_vs_target(curves, os.path.join(figs, "realized_vs_target_alarms.png")))

    cct = pd.read_csv(os.path.join(tables, "cct_summary.csv"))
    macro = cct.groupby(["method", "strategy", "cost_ratio"])["eval_cost"].mean().reset_index()
    print(plot_cct_cost(macro, os.path.join(figs, "cct_cost_vs_ratio.png")))

    rob = pd.read_csv(os.path.join(tables, "robustness_conditions.csv"))
    print(plot_robustness_heatmap(rob, os.path.join(figs, "robustness_heatmap.png")))

    load = pd.read_csv(os.path.join(tables, "operator_load.csv"))
    print(plot_operator_load(load, os.path.join(figs, "operator_load.png"), op_cfg["priors"]))

    bench_csv = os.path.join(tables, "coreset_benchmark.csv")
    if os.path.exists(bench_csv):
        print(plot_coreset_scaling(pd.read_csv(bench_csv), os.path.join(figs, "coreset_scalability.png")))
    else:
        print(f"[skip] coreset figure: {bench_csv} not found (run scripts/run_coreset_benchmark.py on a GPU)")

    lat_csv = os.path.join(tables, "latency.csv")
    if os.path.exists(lat_csv):
        lat = pd.read_csv(lat_csv)
        runs = pd.read_csv(os.path.join(tables, "runs_master.csv"))
        acc = runs.groupby(["category", "method"])["aupro"].mean().reset_index()
        pts = lat.groupby(["category", "method"])["p50_model_ms"].mean().reset_index().merge(acc)
        print(plot_latency_vs_aupro(pts, os.path.join(figs, "latency_vs_aupro.png")))
    else:
        print(f"[skip] latency figure: {lat_csv} not found (run scripts/profile_latency.py on a GPU)")


if __name__ == "__main__":
    main()
