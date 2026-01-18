import os
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .utils import ensure_dir


REQUIRED_COLS = ["category", "auroc_image", "avg_latency_s"]


def _safe_read_csv(path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        print(f"❌ runs file not found: {path}")
        return None

    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"❌ Failed to read CSV: {path}\n{e}")
        return None

    if df is None or len(df) == 0:
        print(f"⚠️ runs.csv is empty (0 rows): {path}")
        return None

    return df


def _ensure_method_mode_cols(df: pd.DataFrame) -> pd.DataFrame:
    # Backward compatibility: some older logs may not have method/mode.
    if "method" not in df.columns:
        df["method"] = "unknown_method"

    if "mode" not in df.columns:
        df["mode"] = "unknown_mode"

    if "run_name" not in df.columns:
        df["run_name"] = "unknown_run"

    return df


def plot_results(csv_path: str, out_dir: str):
    ensure_dir(out_dir)

    df = _safe_read_csv(csv_path)
    if df is None:
        print("⚠️ Nothing to plot.")
        return

    df = _ensure_method_mode_cols(df)

    # Validate required columns exist
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        print(f"❌ Missing required columns in runs.csv: {missing}")
        print("   Available columns:", df.columns.tolist())
        return

    # Create a combined label for legend clarity
    df["label"] = df["method"].astype(str) + "+" + df["mode"].astype(str)

    # -----------------------------
    # 1) AUROC by category (bar)
    # -----------------------------
    # mean across seeds if multiple
    bar_df = (
        df.groupby(["category", "label"], as_index=False)
        .agg(auroc_mean=("auroc_image", "mean"))
    )

    categories = sorted(bar_df["category"].unique().tolist())
    labels = sorted(bar_df["label"].unique().tolist())

    x = np.arange(len(categories))
    width = 0.8 / max(len(labels), 1)

    plt.figure(figsize=(10, 5))
    for i, lab in enumerate(labels):
        sub = bar_df[bar_df["label"] == lab].set_index("category")
        y = [sub.loc[c]["auroc_mean"] if c in sub.index else np.nan for c in categories]
        plt.bar(x + i * width, y, width=width, label=lab)

    plt.xticks(x + width * (len(labels) - 1) / 2, categories, rotation=0)
    plt.ylim(0.0, 1.02)
    plt.ylabel("Image AUROC (↑)")
    plt.title("Anomaly Detection AUROC by Category")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    out_path = os.path.join(out_dir, "auroc_by_category.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print("✅ Saved:", out_path)

    # ------------------------------------
    # 2) Pareto-like scatter: latency vs AUROC
    # ------------------------------------
    pareto_df = (
        df.groupby(["category", "label"], as_index=False)
        .agg(
            auroc=("auroc_image", "mean"),
            latency=("avg_latency_s", "mean"),
        )
    )

    plt.figure(figsize=(8, 6))
    for lab, g in pareto_df.groupby("label"):
        plt.scatter(g["latency"], g["auroc"], label=lab, s=80)

    plt.xlabel("Avg latency per image (s) (↓)")
    plt.ylabel("Image AUROC (↑)")
    plt.title("Latency vs AUROC (mean over seeds)")
    plt.grid(True, alpha=0.35)
    plt.legend()
    out_path = os.path.join(out_dir, "latency_vs_auroc_all.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print("✅ Saved:", out_path)

    # ------------------------------------
    # 3) Per-category scatter plots
    # ------------------------------------
    for cat, gcat in pareto_df.groupby("category"):
        plt.figure(figsize=(8, 6))
        for lab, g in gcat.groupby("label"):
            plt.scatter(g["latency"], g["auroc"], label=lab, s=120)

        plt.xlabel("Avg latency per image (s) (↓)")
        plt.ylabel("Image AUROC (↑)")
        plt.title(f"Latency vs AUROC — {cat}")
        plt.grid(True, alpha=0.35)
        plt.legend()
        out_path = os.path.join(out_dir, f"latency_vs_auroc_{cat}.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close()
        print("Saved:", out_path)

    print("Plotting done.")
