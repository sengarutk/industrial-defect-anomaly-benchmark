"""Image-score aggregation ablation computed from the stored pixel maps of every run.

Output: <results>/tables/aggregation_ablation.csv (per run x rule) and aggregation_summary.csv
(per method x rule, mean and std over categories and seeds).
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.experiments.aggregation_ablation import run_aggregation_ablation  # noqa: E402
from src.utils import parse_run_key  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregation ablation from stored pixel maps")
    parser.add_argument("--results-dir", default="results/mvtec_ad")
    args = parser.parse_args()
    files = sorted(glob.glob(os.path.join(args.results_dir, "maps", "*.npz")))
    if not files:
        raise SystemExit("No pixel-map archives found under maps/; run scripts/run_benchmark.py first.")
    rows = []
    for f in files:
        cat, method, seed = parse_run_key(f)
        maps = np.load(f)["pixel_amaps"]
        labels = np.load(os.path.join(args.results_dir, "scores", os.path.basename(f)))["image_labels"]
        for rule, met in run_aggregation_ablation(maps, labels).items():
            rows.append({"category": cat, "method": method, "seed": seed, "rule": rule, **met})
    df = pd.DataFrame(rows)
    tables = os.path.join(args.results_dir, "tables")
    os.makedirs(tables, exist_ok=True)
    df.to_csv(os.path.join(tables, "aggregation_ablation.csv"), index=False)
    summary = df.groupby(["method", "rule"]).agg(image_auroc=("image_auroc", "mean"),
                                                 image_auroc_std=("image_auroc", "std"),
                                                 image_ap=("image_ap", "mean")).reset_index()
    summary.to_csv(os.path.join(tables, "aggregation_summary.csv"), index=False)
    print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
