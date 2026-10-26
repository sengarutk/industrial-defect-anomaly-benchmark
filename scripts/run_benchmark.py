"""Fits every (category, method, seed), evaluates clean and corrupted test sets, profiles latency.

Outputs (under output.results_dir from configs/benchmark.yaml):
  scores/<cat>_<method>_<seed>.npz   image-level test scores, labels, defect types, paths, and the
                                     scores of the held-out nominal calibration images (tracked in git)
  maps/<cat>_<method>_<seed>.npz     float16 pixel maps + uint8 masks (large; not tracked in git)
  runs/<cat>_<method>_<seed>.json    one record per run (metrics, robustness summary, latency, provenance)
  robustness/<cat>_<method>_<seed>.csv   per-condition robustness metrics
  tables/runs_master.csv, tables/robustness_conditions.csv   rebuilt from the per-run files at the end

Runs whose JSON record already exists are skipped unless --force is given, so an
interrupted sweep can be resumed without duplicating rows.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.benchmarking import LatencyProfiler  # noqa: E402
from src.methods import build_model  # noqa: E402
from src.mvtec import ImageListDataset, list_train_images, split_train_calibration  # noqa: E402
from src.robustness import CachedTestSet, CorruptedTestDataset, evaluate_maps, predict_dataset, run_stress_test  # noqa: E402
from src.utils import ensure_dir, load_config, seed_everything  # noqa: E402


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def rebuild_tables(results_dir: str) -> None:
    runs_dir = os.path.join(results_dir, "runs")
    records = []
    for f in sorted(os.listdir(runs_dir)):
        if f.endswith(".json"):
            with open(os.path.join(runs_dir, f), encoding="utf-8") as fh:
                records.append(json.load(fh))
    tables = os.path.join(results_dir, "tables")
    ensure_dir(tables)
    df = pd.DataFrame(records).sort_values(["category", "method", "seed"]).reset_index(drop=True)
    if df.duplicated(["category", "method", "seed"]).any():
        raise RuntimeError("Duplicate run keys found in runs/; refusing to build runs_master.csv")
    df.to_csv(os.path.join(tables, "runs_master.csv"), index=False)

    rob_dir = os.path.join(results_dir, "robustness")
    if os.path.isdir(rob_dir):
        parts = [pd.read_csv(os.path.join(rob_dir, f)) for f in sorted(os.listdir(rob_dir)) if f.endswith(".csv")]
        if parts:
            pd.concat(parts).to_csv(os.path.join(tables, "robustness_conditions.csv"), index=False)
    print(f"Rebuilt tables from {len(df)} run records in {tables}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-seed MVTec AD benchmark")
    parser.add_argument("--config", default="configs/benchmark.yaml")
    parser.add_argument("--categories", nargs="+")
    parser.add_argument("--methods", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--results-dir")
    parser.add_argument("--force", action="store_true", help="Re-run keys that already have a record")
    parser.add_argument("--no-robustness", action="store_true")
    parser.add_argument("--no-profiling", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_root = cfg["dataset"]["root"]
    img_size = cfg["dataset"]["img_size"]
    categories = args.categories or cfg["dataset"]["categories"]
    methods = args.methods or list(cfg["methods"].keys())
    seeds = args.seeds or cfg["seeds"]
    results_dir = args.results_dir or cfg["output"]["results_dir"]
    do_rob = cfg["robustness"]["enabled"] and not args.no_robustness
    do_prof = cfg["profiling"]["enabled"] and not args.no_profiling
    batch_size = cfg["training"]["batch_size"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    commit = git_commit()

    for sub in ("scores", "maps", "runs", "robustness"):
        ensure_dir(os.path.join(results_dir, sub))
    prof = LatencyProfiler(cfg["profiling"]["warmup_runs"], cfg["profiling"]["active_runs"],
                           cfg["profiling"]["e2e_runs"])

    print(f"device={device} categories={categories} methods={methods} seeds={seeds} commit={commit}")
    for cat in categories:
        cache = None
        train_paths = list_train_images(data_root, cat)
        for seed in seeds:
            fit_paths, calib_paths = split_train_calibration(train_paths, cfg["dataset"]["calib_fraction"], seed)
            for method in methods:
                key = f"{cat}_{method}_{seed}"
                record_path = os.path.join(results_dir, "runs", f"{key}.json")
                if os.path.exists(record_path) and not args.force:
                    print(f"[skip] {key} (record exists)")
                    continue
                if cache is None:
                    cache = CachedTestSet(data_root, cat, img_size)
                print(f"[run] {key}: fit={len(fit_paths)} calib={len(calib_paths)} test={len(cache)}", flush=True)
                seed_everything(seed)
                model = build_model(method, dict(cfg["methods"][method]), seed=seed, device=device)
                fit_loader = DataLoader(ImageListDataset(fit_paths, img_size), batch_size=batch_size, shuffle=False)
                t0 = time.perf_counter()
                model.fit(fit_loader)
                if device == "cuda":
                    torch.cuda.synchronize()
                fit_time = time.perf_counter() - t0

                calib_scores, _ = predict_dataset(model, ImageListDataset(calib_paths, img_size), batch_size)
                clean_ds = CorruptedTestDataset(cache, None, 0, noise_seed=seed)
                scores, maps = predict_dataset(model, clean_ds, batch_size)
                clean = evaluate_maps(cache.labels, scores, cache.masks, maps)

                rel_paths = np.array([os.path.relpath(p, data_root) for p in cache.paths])
                np.savez_compressed(os.path.join(results_dir, "scores", f"{key}.npz"),
                                    image_scores=scores, image_labels=cache.labels, defect_types=cache.defect_types,
                                    image_paths=rel_paths, calib_scores=calib_scores,
                                    calib_paths=np.array([os.path.relpath(p, data_root) for p in calib_paths]))
                np.savez_compressed(os.path.join(results_dir, "maps", f"{key}.npz"),
                                    pixel_amaps=maps.astype(np.float16), ground_truth_masks=cache.masks)

                record = {"category": cat, "method": method, "seed": seed, "device": device,
                          "commit": commit, "timestamp": datetime.now(timezone.utc).isoformat(),
                          "n_fit": len(fit_paths), "n_calib": len(calib_paths), "n_test": len(cache),
                          "n_test_nominal": int((cache.labels == 0).sum()),
                          "n_test_defect": int((cache.labels == 1).sum()),
                          "fit_time_s": fit_time, **clean}

                if do_rob:
                    rob = run_stress_test(model, cache, clean, cfg["robustness"]["corruptions"],
                                          cfg["robustness"]["severities"], batch_size, noise_seed=seed)
                    record.update({k: rob[k] for k in ("mrd_image_auroc", "mrd_aupro", "mpc_image_auroc", "mpc_aupro")})
                    pd.DataFrame([{"category": cat, "method": method, "seed": seed, **r} for r in rob["conditions"]]) \
                        .to_csv(os.path.join(results_dir, "robustness", f"{key}.csv"), index=False)

                if do_prof:
                    sample = clean_ds[0][0].unsqueeze(0)
                    record.update(prof.profile_model(model, sample, device))
                    record.update(prof.profile_e2e(model, list(cache.paths), device, img_size))

                record = {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in record.items()}
                with open(record_path, "w", encoding="utf-8") as fh:
                    json.dump(record, fh, indent=2)
                print(f"      AUROC={clean['image_auroc']:.4f} AU-PRO={clean['aupro']:.4f} "
                      f"MRD={record.get('mrd_image_auroc', float('nan')):.4f} fit={fit_time:.1f}s", flush=True)
                del model
                if device == "cuda":
                    torch.cuda.empty_cache()

    rebuild_tables(results_dir)


if __name__ == "__main__":
    main()
