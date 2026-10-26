"""End-to-end run of the real scripts on a toy category (small, fast, CPU)."""
import os
import sys

import pandas as pd
import pytest
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts")))

import generate_report  # noqa: E402
import run_ablations  # noqa: E402
import run_benchmark  # noqa: E402
import run_cct_experiments  # noqa: E402
import run_operational_eval  # noqa: E402


def write_configs(tmp_path, data_root, results):
    bench = {
        "dataset": {"root": data_root, "categories": ["toy"], "img_size": 64, "calib_fraction": 0.25},
        "seeds": [1, 2],
        "methods": {"patchcore": {"coreset_sampling_ratio": 0.1}, "padim": {"reduced_dim": 20},
                    "autoencoder": {"epochs": 1}},
        "training": {"batch_size": 4},
        "robustness": {"enabled": True, "corruptions": ["gaussian_blur"], "severities": [1, 3]},
        "profiling": {"warmup_runs": 1, "active_runs": 2, "e2e_runs": 2},
        "output": {"results_dir": results},
    }
    op = {"deployment_prior": 0.01, "priors": [0.01, 0.15], "cost_ratios": [10.0, 50.0], "primary_cost_ratio": 10.0,
          "alert_budgets_per_1k": [5.0, 100.0], "primary_alert_budget_per_1k": 5.0,
          "cct": {"n_splits": 3, "split_seed": 0},
          "streams": {"n_items": 2000, "operator_capacity_per_window": 60, "window_size": 1000,
                      "burst_mean_block_length": 20, "drift_slope_sigma": 0.1},
          "statistics": {"bootstrap_resamples": 50, "ci": 0.95, "alpha": 0.05}}
    b, o = tmp_path / "bench.yaml", tmp_path / "op.yaml"
    b.write_text(yaml.safe_dump(bench))
    o.write_text(yaml.safe_dump(op))
    return str(b), str(o)


def run(module, argv, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["x"] + argv)
    module.main()


def test_pipeline_end_to_end(tmp_path, toy_root, monkeypatch):
    results = str(tmp_path / "results")
    bench, op = write_configs(tmp_path, toy_root, results)
    run(run_benchmark, ["--config", bench], monkeypatch)
    master = pd.read_csv(os.path.join(results, "tables", "runs_master.csv"))
    assert len(master) == 6 and not master.duplicated(["category", "method", "seed"]).any()
    assert master.n_calib.iloc[0] == 3 and master.n_fit.iloc[0] == 9

    # a second invocation resumes instead of appending duplicate rows
    run(run_benchmark, ["--config", bench], monkeypatch)
    assert len(pd.read_csv(os.path.join(results, "tables", "runs_master.csv"))) == 6

    run(run_operational_eval, ["--config", op, "--results-dir", results], monkeypatch)
    run(run_cct_experiments, ["--config", op, "--results-dir", results], monkeypatch)
    run(run_ablations, ["--results-dir", results], monkeypatch)
    summ = pd.read_csv(os.path.join(results, "tables", "operational_summary.csv"))
    assert set(summ.method) == {"patchcore", "padim", "autoencoder"}
    cct = pd.read_csv(os.path.join(results, "tables", "cct_summary.csv"))
    assert set(cct.strategy) == {"nominal_budget", "cct_budget", "cct_unconstrained", "oracle"}

    docs = str(tmp_path / "docs")
    run(generate_report, ["--config", bench, "--operational-config", op, "--results-dir", results,
                          "--docs-dir", docs], monkeypatch)
    macros = open(os.path.join(results, "tables", "latex", "macros.tex")).read()
    assert "\\newcommand{\\NumRuns}{6}" in macros and "\\HasLatency}{0}" in macros
    assert os.path.exists(os.path.join(docs, "benchmark_report.md"))


def test_report_refuses_to_run_without_results(tmp_path, monkeypatch):
    """Regression: the report used to fall back to hardcoded latency rows and constant macros."""
    with pytest.raises(SystemExit):
        run(generate_report, ["--results-dir", str(tmp_path / "nothing"), "--docs-dir", str(tmp_path)], monkeypatch)
