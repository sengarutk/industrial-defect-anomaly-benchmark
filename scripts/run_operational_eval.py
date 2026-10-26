"""Operating-point evaluation with thresholds that never see the test set.

For each run archive, alarm-budget thresholds are computed from the scores of
the held-out train/good calibration images only, then applied to the full test
set. Every rate is reported with an exact Clopper-Pearson interval and with the
smallest non-zero rate the sample size can resolve.

Outputs in <results>/tables/:
  operational_runs.csv      per run x budget: realised test FA/1k, TPR, expected cost
  operator_load.csv         per run x prior x stream regime: projected alerts per window, P(overload)
  operational_summary.csv   per category x method at the primary budget (hierarchical bootstrap CI)
  significance_cost.csv     paired Wilcoxon across categories on expected cost, Holm-adjusted
"""
import argparse
import glob
import os
import sys
import zlib
from itertools import combinations

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.experiments.operational_eval import ProductionStreamSimulator  # noqa: E402
from src.metrics.operational import (alert_budget_threshold, compute_operator_overload, expected_cost,  # noqa: E402
                                     false_negative_rate, false_positive_rate, min_resolvable_rate_per_1k,
                                     operating_point)
from src.metrics.stats import hierarchical_bootstrap_ci, holm_bonferroni, paired_wilcoxon  # noqa: E402
from src.utils import load_config, parse_run_key  # noqa: E402


def load_runs(scores_dir: str):
    files = sorted(glob.glob(os.path.join(scores_dir, "*.npz")))
    if not files:
        raise SystemExit(f"No score archives in {scores_dir}; run scripts/run_benchmark.py first.")
    runs = []
    for f in files:
        cat, method, seed = parse_run_key(f)
        d = np.load(f, allow_pickle=False)
        runs.append({"category": cat, "method": method, "seed": seed, "scores": d["image_scores"],
                     "labels": d["image_labels"], "calib": d["calib_scores"]})
    return runs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/operational.yaml")
    parser.add_argument("--results-dir", default="results/mvtec_ad")
    args = parser.parse_args()
    cfg = load_config(args.config)
    prior = cfg["deployment_prior"]
    r_main = cfg["primary_cost_ratio"]
    b_main = cfg["primary_alert_budget_per_1k"]
    st = cfg["streams"]
    n_boot = cfg["statistics"]["bootstrap_resamples"]
    ci = cfg["statistics"]["ci"]
    tables = os.path.join(args.results_dir, "tables")
    os.makedirs(tables, exist_ok=True)

    runs = load_runs(os.path.join(args.results_dir, "scores"))
    op_rows, load_rows = [], []
    for run in runs:
        s, y, cal = run["scores"], run["labels"], run["calib"]
        nom, dfc = s[y == 0], s[y == 1]
        key = {k: run[k] for k in ("category", "method", "seed")}
        for b in cfg["alert_budgets_per_1k"]:
            tau = alert_budget_threshold(cal, b)
            op = operating_point(y, s, tau, ci)
            fpr, fnr = false_positive_rate(nom, tau), false_negative_rate(dfc, tau)
            row = {**key, "budget_per_1k": b, "n_calib_nominal": len(cal),
                   "calib_resolution_per_1k": min_resolvable_rate_per_1k(len(cal)),
                   "test_resolution_per_1k": min_resolvable_rate_per_1k(len(nom)),
                   "budget_violated_on_test": bool(op["fa_at_1k"] > b + 1e-9), **op}
            for r in cfg["cost_ratios"]:
                row[f"expected_cost_r{int(r)}"] = expected_cost(fpr, fnr, prior, r)
            op_rows.append(row)
            if b != b_main:
                continue
            sim = ProductionStreamSimulator(nom, dfc, seed=run["seed"])
            for p in cfg["priors"]:
                streams = {
                    "iid": sim.simulate_iid_stream(st["n_items"], p),
                    "burst": sim.simulate_burst_stream(st["n_items"], p, st["burst_mean_block_length"]),
                    "drift": sim.simulate_drift_stream(st["n_items"], p, st["drift_slope_sigma"]),
                }
                for regime, (_, stream_scores) in streams.items():
                    ov = compute_operator_overload(stream_scores >= tau, st["operator_capacity_per_window"],
                                                   st["window_size"])
                    load_rows.append({**key, "budget_per_1k": b, "prior": p, "regime": regime, **ov})

    op_df = pd.DataFrame(op_rows)
    load_df = pd.DataFrame(load_rows)
    op_df.to_csv(os.path.join(tables, "operational_runs.csv"), index=False)
    load_df.to_csv(os.path.join(tables, "operator_load.csv"), index=False)

    # Per category x method summary at the primary budget: two-stage bootstrap over seeds, then test images
    # resampled within each class (class counts fixed).
    summary = []
    for (cat, method), grp in pd.DataFrame(runs).groupby(["category", "method"]):
        records = []
        for _, run in grp.iterrows():
            records.append({"scores": run["scores"], "labels": run["labels"],
                            "tau": alert_budget_threshold(run["calib"], b_main)})

        def tpr(recs):
            return float(np.mean([np.mean(r["scores"][r["labels"] == 1] >= r["tau"]) for r in recs]))

        def fa(recs):
            return float(np.mean([1000 * np.mean(r["scores"][r["labels"] == 0] >= r["tau"]) for r in recs]))

        def cost(recs):
            return float(np.mean([expected_cost(np.mean(r["scores"][r["labels"] == 0] >= r["tau"]),
                                                np.mean(r["scores"][r["labels"] == 1] < r["tau"]), prior, r_main)
                                  for r in recs]))

        seed = zlib.crc32(f"{cat}|{method}".encode())  # stable across processes, unlike hash()
        h_tpr = hierarchical_bootstrap_ci(records, tpr, ["scores", "labels"], n_boot, ci, seed, strata_key="labels")
        h_fa = hierarchical_bootstrap_ci(records, fa, ["scores", "labels"], n_boot, ci, seed + 1, strata_key="labels")
        h_cost = hierarchical_bootstrap_ci(records, cost, ["scores", "labels"], n_boot, ci, seed + 2, strata_key="labels")
        sub = op_df[(op_df.category == cat) & (op_df.method == method) & (op_df.budget_per_1k == b_main)]
        ld = load_df[(load_df.category == cat) & (load_df.method == method) & (load_df.prior == prior)]
        summary.append({
            "category": cat, "method": method, "n_seeds": len(records), "budget_per_1k": b_main,
            "tpr": h_tpr["estimate"], "tpr_ci_low": h_tpr["ci_low"], "tpr_ci_high": h_tpr["ci_high"],
            "test_fa_at_1k": h_fa["estimate"], "test_fa_ci_low": h_fa["ci_low"], "test_fa_ci_high": h_fa["ci_high"],
            f"expected_cost_r{int(r_main)}": h_cost["estimate"], "cost_ci_low": h_cost["ci_low"],
            "cost_ci_high": h_cost["ci_high"],
            "budget_violation_rate": float(sub["budget_violated_on_test"].mean()),
            "test_resolution_per_1k": float(sub["test_resolution_per_1k"].iloc[0]),
            "calib_resolution_per_1k": float(sub["calib_resolution_per_1k"].iloc[0]),
            "p_overload_iid": float(ld[ld.regime == "iid"]["p_overload"].mean()),
            "p_overload_burst": float(ld[ld.regime == "burst"]["p_overload"].mean()),
        })
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(os.path.join(tables, "operational_summary.csv"), index=False)

    # Significance: one value per category (mean over seeds) -> paired Wilcoxon between methods.
    cost_col = f"expected_cost_r{int(r_main)}"
    unit = op_df[op_df.budget_per_1k == b_main].groupby(["category", "method"])[cost_col].mean().unstack()
    tests = {}
    for a, bm in combinations(sorted(unit.columns), 2):
        res = paired_wilcoxon(unit[a].values, unit[bm].values)
        tests[f"{a} vs {bm}"] = res
    holm = holm_bonferroni({k: v["p_value"] for k, v in tests.items()}, cfg["statistics"]["alpha"])
    sig = pd.DataFrame([{"comparison": k, "metric": cost_col, "unit": "category", **v,
                         "holm_adjusted_p": holm[k]["adjusted_p"], "reject_h0": holm[k]["reject"]}
                        for k, v in tests.items()])
    sig.to_csv(os.path.join(tables, "significance_cost.csv"), index=False)
    print(summary_df.round(4).to_string(index=False))
    print(sig.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
