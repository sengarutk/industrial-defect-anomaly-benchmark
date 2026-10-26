"""Measured CPU vs GPU coreset-selection benchmark on real PatchCore features (see src/experiments/coreset_benchmark.py).

Requires CUDA unless --allow-cpu is given (then the "GPU" columns are CPU torch timings and the
output is written to coreset_benchmark_cpu_only.csv so it cannot be mistaken for a GPU result).
"""
import argparse
import os
import sys

import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.experiments.coreset_benchmark import run_coreset_benchmark  # noqa: E402
from src.methods import PatchCore  # noqa: E402
from src.mvtec import ImageListDataset, list_train_images, split_train_calibration  # noqa: E402
from src.utils import load_config, seed_everything  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Coreset selection systems benchmark")
    parser.add_argument("--config", default="configs/benchmark.yaml")
    parser.add_argument("--category", default="bottle")
    parser.add_argument("--sizes", nargs="+", type=int, default=[1000, 5000, 10000, 25000, 50000, 100000])
    parser.add_argument("--dims", nargs="+", type=int, default=[64, 128, 256])
    parser.add_argument("--results-dir", default="results/mvtec_ad")
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise SystemExit("CUDA is not available; refusing to label CPU timings as GPU. Use --allow-cpu to test.")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = cfg["seeds"][0]
    seed_everything(seed)

    paths = list_train_images(cfg["dataset"]["root"], args.category)
    fit_paths, _ = split_train_calibration(paths, cfg["dataset"]["calib_fraction"], seed)
    extractor = PatchCore(device=device, seed=seed)
    loader = DataLoader(ImageListDataset(fit_paths, cfg["dataset"]["img_size"]), batch_size=8)
    feats = torch.cat([extractor.extract_patches(x).cpu() for x in loader])
    print(f"Extracted {feats.shape[0]} real patch features ({feats.shape[1]}-d) from {args.category}")

    rows = run_coreset_benchmark(feats, args.sizes, args.dims, cfg["methods"]["patchcore"]["coreset_sampling_ratio"],
                                 seed=seed, device=device)
    df = pd.DataFrame(rows)
    df.insert(0, "device", torch.cuda.get_device_name(0) if device == "cuda" else "cpu")
    df.insert(1, "feature_source", f"{args.category} train/good (PatchCore layer2+3, random projection to D)")
    tables = os.path.join(args.results_dir, "tables")
    os.makedirs(tables, exist_ok=True)
    name = "coreset_benchmark.csv" if device == "cuda" else "coreset_benchmark_cpu_only.csv"
    df.to_csv(os.path.join(tables, name), index=False)
    print(df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
