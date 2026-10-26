"""Latency / VRAM profiling at batch size 1 for every category x method (first configured seed).

Each model is fitted on the same fit split as the benchmark, then profiled with
src/benchmarking/profiler.py. Results go to <results>/tables/latency.csv with the device name;
CPU-only runs are written to latency_cpu.csv so they are never mistaken for GPU numbers.
"""
import argparse
import os
import sys

import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.benchmarking import LatencyProfiler  # noqa: E402
from src.methods import build_model  # noqa: E402
from src.mvtec import (ImageListDataset, list_test_samples, list_train_images, load_rgb, normalize_image,  # noqa: E402
                       split_train_calibration)
from src.utils import load_config, seed_everything  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Latency profiling")
    parser.add_argument("--config", default="configs/benchmark.yaml")
    parser.add_argument("--results-dir", default="results/mvtec_ad")
    parser.add_argument("--categories", nargs="+")
    parser.add_argument("--methods", nargs="+")
    args = parser.parse_args()
    cfg = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_name = torch.cuda.get_device_name(0) if device == "cuda" else "cpu"
    pc = cfg["profiling"]
    prof = LatencyProfiler(pc["warmup_runs"], pc["active_runs"], pc["e2e_runs"])
    seed = cfg["seeds"][0]
    img_size = cfg["dataset"]["img_size"]
    rows = []
    for cat in args.categories or cfg["dataset"]["categories"]:
        fit_paths, _ = split_train_calibration(list_train_images(cfg["dataset"]["root"], cat),
                                               cfg["dataset"]["calib_fraction"], seed)
        test_paths = [s["path"] for s in list_test_samples(cfg["dataset"]["root"], cat)]
        sample = normalize_image(load_rgb(test_paths[0], img_size)).unsqueeze(0)
        for method in args.methods or list(cfg["methods"].keys()):
            seed_everything(seed)
            model = build_model(method, dict(cfg["methods"][method]), seed=seed, device=device)
            model.fit(DataLoader(ImageListDataset(fit_paths, img_size), batch_size=cfg["training"]["batch_size"]))
            row = {"category": cat, "method": method, "seed": seed, "device": device_name}
            row.update(prof.profile_model(model, sample, device))
            row.update(prof.profile_e2e(model, test_paths, device, img_size))
            rows.append(row)
            print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
            del model
            if device == "cuda":
                torch.cuda.empty_cache()
    out = os.path.join(args.results_dir, "tables", "latency.csv" if device == "cuda" else "latency_cpu.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
