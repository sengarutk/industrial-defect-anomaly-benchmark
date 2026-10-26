"""Out-of-sample evaluation of Cost-Calibrated Thresholding (CCT) against a nominal-only budget threshold.

See src/experiments/cct_ablation.py for the protocol. Outputs in <results>/tables/:
  cct_runs.csv             every run x split x cost ratio x strategy (evaluation-half metrics)
  cct_summary.csv          per category x method x cost ratio x strategy (mean over seeds and splits)
  cct_decisions.csv        decision flips nominal_budget -> cct_budget on the evaluation halves
  significance_cct.csv     per method: paired Wilcoxon across categories (CCT vs nominal budget), Holm-adjusted
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.experiments.cct_ablation import evaluate_run  # noqa: E402
from src.metrics.stats import holm_bonferroni, paired_wilcoxon  # noqa: E402
from src.utils import load_config, parse_run_key  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="CCT out-of-sample evaluation")
    parser.add_argument("--config", default="configs/operational.yaml")
    parser.add_argument("--results-dir", default="results/mvtec_ad")
    args = parser.parse_args()
    cfg = load_config(args.config)
    tables = os.path.join(args.results_dir, "tables")
    os.makedirs(tables, exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.results_dir, "scores", "*.npz")))
    if not files:
        raise SystemExit("No score archives found; run scripts/run_benchmark.py first.")
    rows, decisions = [], []
    for f in files:
        cat, method, seed = parse_run_key(f)
        d = np.load(f)
        r_rows, r_dec = evaluate_run(d["image_scores"], d["image_labels"], d["calib_scores"],
                                     cfg["cost_ratios"], cfg["deployment_prior"],
                                     cfg["primary_alert_budget_per_1k"], cfg["cct"]["n_splits"],
                                     seed=cfg["cct"]["split_seed"] * 100003 + seed)
        key = {"category": cat, "method": method, "seed": seed}
        rows += [{**key, **r} for r in r_rows]
        decisions += [{**key, **r} for r in r_dec]

    runs = pd.DataFrame(rows)
    dec = pd.DataFrame(decisions)
    runs.to_csv(os.path.join(tables, "cct_runs.csv"), index=False)
    summary = runs.groupby(["category", "method", "cost_ratio", "strategy"]).agg(
        eval_cost=("eval_cost", "mean"), eval_cost_std=("eval_cost", "std"),
        eval_fa_at_1k=("eval_fa_at_1k", "mean"), eval_fnr=("eval_fnr", "mean"),
        budget_violation_rate=("budget_violated", "mean"),
        n_calib_nominal=("n_calib_nominal", "mean"), n_eval_nominal=("n_eval_nominal", "mean"),
        n_evaluations=("eval_cost", "size")).reset_index()
    summary.to_csv(os.path.join(tables, "cct_summary.csv"), index=False)
    dec_summary = dec.groupby(["category", "method", "cost_ratio"]).agg(
        flips=("flips", "mean"), false_alarms_removed=("false_alarms_removed", "mean"),
        false_alarms_added=("false_alarms_added", "mean"), defects_lost=("defects_lost", "mean"),
        defects_gained=("defects_gained", "mean"), n_eval_nominal=("n_eval_nominal", "mean"),
        n_eval_defect=("n_eval_defect", "mean")).reset_index()
    dec_summary.to_csv(os.path.join(tables, "cct_decisions.csv"), index=False)

    r_main = cfg["primary_cost_ratio"]
    unit = summary[summary.cost_ratio == r_main].pivot_table(index=["method", "category"], columns="strategy",
                                                             values="eval_cost")
    tests = {}
    for method in sorted(unit.index.get_level_values(0).unique()):
        u = unit.loc[method]
        for rule in ("cct_budget", "cct_unconstrained"):
            tests[f"{method}: {rule} vs nominal_budget"] = paired_wilcoxon(u[rule].values, u["nominal_budget"].values)
    holm = holm_bonferroni({k: v["p_value"] for k, v in tests.items()}, cfg["statistics"]["alpha"])
    sig = pd.DataFrame([{"comparison": k, "cost_ratio": r_main, "unit": "category", **v,
                         "holm_adjusted_p": holm[k]["adjusted_p"], "reject_h0": holm[k]["reject"]}
                        for k, v in tests.items()])
    sig.to_csv(os.path.join(tables, "significance_cct.csv"), index=False)
    macro = summary[summary.cost_ratio == r_main].groupby(["method", "strategy"])[
        ["eval_cost", "eval_fa_at_1k", "budget_violation_rate"]].mean()
    print(macro.round(4).to_string())
    print(sig.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
