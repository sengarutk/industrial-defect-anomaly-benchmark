import os
import glob
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd

from src.metrics.cost_calibrated import optimize_cct_threshold
from src.metrics.operational import compute_quantile_threshold


def compute_decision_change_matrix(
    scores: np.ndarray,
    labels: np.ndarray,
    tau_baseline: float,
    tau_cct: float
) -> Dict[str, Any]:
    """
    Quantifies exact operational decision changes and attribution shifts when switching
    from baseline threshold (tau_baseline) to cost-calibrated threshold (tau_cct).
    """
    scores = np.asarray(scores, dtype=np.float64).ravel()
    labels = np.asarray(labels, dtype=int).ravel()
    n_total = len(scores)

    if n_total == 0:
        return {
            "total_flips": 0,
            "total_flip_rate": 0.0,
            "nominal_relief_count": 0,
            "nominal_relief_rate": 0.0,
            "defect_escape_count": 0,
            "defect_escape_rate": 0.0,
            "defect_catch_count": 0,
            "defect_catch_rate": 0.0,
            "tau_baseline": float(tau_baseline),
            "tau_cct": float(tau_cct)
        }

    pred_base = (scores >= tau_baseline).astype(int)
    pred_cct = (scores >= tau_cct).astype(int)

    nom_mask = (labels == 0)
    def_mask = (labels == 1)
    n_nom = int(np.sum(nom_mask))
    n_def = int(np.sum(def_mask))

    total_flips = int(np.sum(pred_base != pred_cct))
    total_flip_rate = float(total_flips / n_total)

    # False alarms eliminated (Nominal Relief)
    nom_relief = int(np.sum(nom_mask & (pred_base == 1) & (pred_cct == 0)))
    nom_relief_rate = float(nom_relief / n_nom) if n_nom > 0 else 0.0

    # Defects missed due to higher threshold (Defect Escape)
    def_escape = int(np.sum(def_mask & (pred_base == 1) & (pred_cct == 0)))
    def_escape_rate = float(def_escape / n_def) if n_def > 0 else 0.0

    # Defects caught if CCT lowered threshold
    def_catch = int(np.sum(def_mask & (pred_base == 0) & (pred_cct == 1)))
    def_catch_rate = float(def_catch / n_def) if n_def > 0 else 0.0

    return {
        "total_flips": total_flips,
        "total_flip_rate": total_flip_rate,
        "nominal_relief_count": nom_relief,
        "nominal_relief_rate": nom_relief_rate,
        "defect_escape_count": def_escape,
        "defect_escape_rate": def_escape_rate,
        "defect_catch_count": def_catch,
        "defect_catch_rate": def_catch_rate,
        "tau_baseline": float(tau_baseline),
        "tau_cct": float(tau_cct)
    }


def run_decision_change_analysis(
    scores_dir: str = "results/mvtec_ad/scores",
    output_dir: str = "results/mvtec_ad"
) -> pd.DataFrame:
    """
    Executes decision change attribution across all 45 benchmark evaluation runs.
    """
    tables_dir = os.path.join(output_dir, "tables")
    os.makedirs(tables_dir, exist_ok=True)

    npz_files = sorted(glob.glob(os.path.join(scores_dir, "*.npz")))
    if len(npz_files) == 0:
        return pd.DataFrame()

    records = []

    for fpath in npz_files:
        fname = os.path.basename(fpath).replace(".npz", "")
        parts = fname.split("_")
        if len(parts) >= 3:
            cat, meth, seed = "_".join(parts[:-2]), parts[-2], int(parts[-1])
        else:
            cat, meth, seed = parts[0], parts[1], 42

        data = np.load(fpath)
        labels = data["image_labels"]
        scores = data["image_scores"]

        nom_scores = scores[labels == 0]
        tau_99 = compute_quantile_threshold(nom_scores, quantile=0.99)
        cct_res = optimize_cct_threshold(scores, labels, cost_ratio=10.0, prior=0.01, max_alerts_per_1k=5.0)
        tau_cct = cct_res["threshold"]

        mat = compute_decision_change_matrix(scores, labels, tau_baseline=tau_99, tau_cct=tau_cct)
        records.append({
            "category": cat,
            "method": meth,
            "seed": seed,
            "total_flips": mat["total_flips"],
            "total_flip_rate": mat["total_flip_rate"],
            "nominal_relief_count": mat["nominal_relief_count"],
            "nominal_relief_rate": mat["nominal_relief_rate"],
            "defect_escape_count": mat["defect_escape_count"],
            "defect_escape_rate": mat["defect_escape_rate"],
            "defect_catch_count": mat["defect_catch_count"],
            "defect_catch_rate": mat["defect_catch_rate"],
            "tau_99": tau_99,
            "tau_cct": tau_cct
        })

    df = pd.DataFrame(records)
    out_csv = os.path.join(tables_dir, "decision_changes.csv")
    out_md = os.path.join(tables_dir, "decision_changes.md")

    df.to_csv(out_csv, index=False)

    summary_df = df.groupby(["category", "method"]).agg({
        "total_flips": "mean",
        "total_flip_rate": "mean",
        "nominal_relief_count": "mean",
        "defect_escape_count": "mean"
    }).reset_index()

    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# Decision-Change Attribution Summary (Transitioning from Quantile-99 to CCT)\n\n")
        f.write(summary_df.to_markdown(index=False))

    print(f"✅ Decision-Change Attribution Complete! Saved to {out_csv}")
    return df