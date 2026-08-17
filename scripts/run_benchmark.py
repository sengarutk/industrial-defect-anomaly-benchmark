import argparse
import os
import sys
import time

# Ensure repository root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import List
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.utils import seed_everything
from src.mvtec import MVTecTrainNormal
from src.methods.patchcore import PatchCore
from src.methods.padim import PaDiM
from src.methods.autoencoder import ConvAutoencoder
from src.robustness.dataset import CorruptedMVTecTest
from src.robustness.evaluator import RobustnessEvaluator
from src.benchmarking.profiler import CUDAPerformanceProfiler


def get_model(method_name: str, device: str):
    method = method_name.lower()
    if method == "patchcore":
        return PatchCore(device=device)
    elif method == "padim":
        return PaDiM(device=device)
    elif method == "autoencoder":
        return ConvAutoencoder(device=device, epochs=20)
    else:
        raise ValueError(f"Unknown method: {method_name}")


def main():
    parser = argparse.ArgumentParser(description="Master Multi-Seed Benchmark Runner")
    parser.add_argument("--categories", nargs="+", default=["bottle", "cable", "hazelnut", "metal_nut"])
    parser.add_argument("--methods", nargs="+", default=["patchcore", "padim", "autoencoder"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--data-root", type=str, default="data/mvtec_ad")
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--run-robustness", action="store_true", default=True)
    parser.add_argument("--run-profiling", action="store_true", default=True)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tables_dir = os.path.join(args.output_dir, "tables")
    os.makedirs(tables_dir, exist_ok=True)
    master_csv_path = os.path.join(tables_dir, "runs_master.csv")

    profiler = CUDAPerformanceProfiler(warmup_runs=10, active_runs=50)

    rows = []
    print(f"=== Starting Benchmark on Device: {device} ===")
    print(f"Categories: {args.categories}")
    print(f"Methods: {args.methods}")
    print(f"Seeds: {args.seeds}\n")

    for seed in args.seeds:
        for cat in args.categories:
            for method_name in args.methods:
                print(f"--> Running [Seed: {seed}] [Cat: {cat}] [Method: {method_name}]")
                seed_everything(seed)

                # 1. Fit Model
                train_ds = MVTecTrainNormal(args.data_root, cat)
                train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

                t0 = time.time()
                model = get_model(method_name, device)
                model.fit(train_loader)
                fit_time_s = time.time() - t0

                # 2. Profile Hardware Latency & VRAM
                prof_results = {}
                if args.run_profiling:
                    sample_in = torch.randn(1, 3, 256, 256, device=device)
                    prof_results = profiler.profile(lambda inp: model.predict(inp), sample_in)

                # 3. Clean Evaluation
                evaluator = RobustnessEvaluator(model, args.data_root, cat, device=device)
                test_ds = CorruptedMVTecTest(args.data_root, cat, corruption_type=None)
                test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
                clean_metrics = evaluator.evaluate_split(test_loader)

                # 4. Robustness Stress-Test
                mCE_auroc, mCE_aupro = 0.0, 0.0
                if args.run_robustness:
                    stress_res = evaluator.run_full_stress_test(batch_size=args.batch_size)
                    mCE_auroc = stress_res["mCE_image_auroc"]
                    mCE_aupro = stress_res["mCE_aupro"]

                row = {
                    "timestamp": pd.Timestamp.utcnow().isoformat(),
                    "seed": seed,
                    "category": cat,
                    "method": method_name,
                    "device": device,
                    "fit_time_s": fit_time_s,
                    "image_auroc": clean_metrics["image_auroc"],
                    "image_ap": clean_metrics["image_ap"],
                    "pixel_auroc": clean_metrics["pixel_auroc"],
                    "pixel_ap": clean_metrics["pixel_ap"],
                    "aupro": clean_metrics["aupro"],
                    "max_f1": clean_metrics["max_f1"],
                    "ece": clean_metrics["ece"],
                    "mCE_image_auroc": mCE_auroc,
                    "mCE_aupro": mCE_aupro,
                    "p50_latency_ms": prof_results.get("p50_ms", 0.0),
                    "p95_latency_ms": prof_results.get("p95_ms", 0.0),
                    "fps": prof_results.get("fps", 0.0),
                    "peak_vram_mb": prof_results.get("peak_vram_mb", 0.0),
                }
                rows.append(row)
                df_row = pd.DataFrame([row])
                write_header = not os.path.exists(master_csv_path)
                df_row.to_csv(master_csv_path, mode="a", header=write_header, index=False)
                print(f"    Done: Image AUROC = {row['image_auroc']:.4f} | AU-PRO = {row['aupro']:.4f} | mCE = {row['mCE_image_auroc']:.4f}")

    # Generate multi-seed summary table
    df_all = pd.DataFrame(rows)
    group_cols = ["category", "method"]
    summary_df = df_all.groupby(group_cols).agg(
        image_auroc_mean=("image_auroc", "mean"),
        image_auroc_std=("image_auroc", "std"),
        aupro_mean=("aupro", "mean"),
        aupro_std=("aupro", "std"),
        pixel_auroc_mean=("pixel_auroc", "mean"),
        pixel_auroc_std=("pixel_auroc", "std"),
        mCE_mean=("mCE_image_auroc", "mean"),
        mCE_std=("mCE_image_auroc", "std"),
        p50_latency_ms=("p50_latency_ms", "mean"),
        fps=("fps", "mean"),
        peak_vram_mb=("peak_vram_mb", "mean"),
    ).reset_index()

    summary_csv = os.path.join(tables_dir, "summary_multiseed.csv")
    summary_df.to_csv(summary_csv, index=False)

    summary_md = os.path.join(tables_dir, "summary_multiseed.md")
    with open(summary_md, "w", encoding="utf-8") as f:
        f.write("# Master Multi-Seed Benchmark Summary (mean ± std)\n\n")
        f.write(summary_df.to_markdown(index=False))
        f.write("\n")

    print(f"\n=== Benchmark Complete! Saved summaries to {tables_dir} ===")


if __name__ == "__main__":
    main()
