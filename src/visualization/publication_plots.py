"""Paper figures. Every function takes measured data; none has a synthetic fallback.

Colour roles (validated with the dataviz palette checker, light mode, all pairs):
methods use categorical slots 1-3; threshold strategies use neutral + violet/magenta.
Each series also has its own marker so identity never depends on colour alone.
"""
import os
from typing import Dict, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

METHOD_ORDER = ["patchcore", "padim", "autoencoder"]
METHOD_LABEL = {"patchcore": "PatchCore", "padim": "PaDiM", "autoencoder": "ConvAutoencoder"}
METHOD_COLOR = {"patchcore": "#2a78d6", "padim": "#eb6834", "autoencoder": "#1baf7a"}
METHOD_MARKER = {"patchcore": "o", "padim": "s", "autoencoder": "^"}

STRATEGY_ORDER = ["nominal_budget", "cct_budget", "cct_unconstrained", "oracle"]
STRATEGY_LABEL = {"nominal_budget": "Nominal-only budget", "cct_budget": "CCT (budget)",
                  "cct_unconstrained": "CCT (no budget)", "oracle": "Oracle (in-sample)"}
STRATEGY_STYLE = {"nominal_budget": dict(color="#52514e", ls="--", marker="o"),
                  "cct_budget": dict(color="#6250d6", ls="-", marker="s"),
                  "cct_unconstrained": dict(color="#e87ba4", ls="-", marker="^"),
                  "oracle": dict(color="#9a9893", ls=":", marker="x")}

INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"


def set_style() -> None:
    plt.rcParams.update({
        "font.family": "serif", "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
        "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.edgecolor": INK_2, "axes.labelcolor": INK, "xtick.color": INK_2, "ytick.color": INK_2,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2.0, "lines.markersize": 5,
        "figure.dpi": 100, "savefig.bbox": "tight",
    })


def _save(fig, path: str) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    base = os.path.splitext(path)[0]
    fig.savefig(base + ".png", dpi=300)
    fig.savefig(base + ".pdf")
    plt.close(fig)
    return base + ".png"


def plot_tpr_vs_budget(curves: Dict[str, pd.DataFrame], path: str) -> str:
    """curves[method]: DataFrame with columns budget, tpr_mean, tpr_min, tpr_max (macro over categories)."""
    set_style()
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    for m in METHOD_ORDER:
        if m not in curves:
            continue
        c = curves[m]
        ax.plot(c["budget"], c["tpr_mean"], color=METHOD_COLOR[m], marker=METHOD_MARKER[m], markevery=3,
                label=METHOD_LABEL[m])
        ax.fill_between(c["budget"], c["tpr_min"], c["tpr_max"], color=METHOD_COLOR[m], alpha=0.12, lw=0)
    ax.set_xscale("log")
    ax.set_xlabel("Target budget (false alarms / 1k nominal)")
    ax.set_ylabel("Test defect recall (TPR)")
    ax.set_ylim(0, 1.02)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3, frameon=False, fontsize=7)
    return _save(fig, path)


def plot_realized_vs_target(curves: Dict[str, pd.DataFrame], path: str) -> str:
    """curves[method]: DataFrame with budget, fa_mean (realised test FA/1k, macro over categories)."""
    set_style()
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    lo, hi = None, None
    for m in METHOD_ORDER:
        if m not in curves:
            continue
        c = curves[m]
        ax.plot(c["budget"], c["fa_mean"], color=METHOD_COLOR[m], marker=METHOD_MARKER[m], markevery=3,
                label=METHOD_LABEL[m])
        c = c[c["fa_mean"] > 0]
        lo = c["budget"].min() if lo is None else min(lo, c["budget"].min())
        hi = c["budget"].max() if hi is None else max(hi, c["budget"].max())
    ax.plot([lo, hi], [lo, hi], color=INK_2, ls="--", lw=1.2, label="Realised = target")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Target budget (false alarms / 1k nominal)")
    ax.set_ylabel("Realised test FA / 1k")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, frameon=False, fontsize=7)
    return _save(fig, path)


def plot_cct_cost(summary: pd.DataFrame, path: str) -> str:
    """summary: cct_summary.csv macro-averaged over categories -> columns method, strategy, cost_ratio, eval_cost."""
    set_style()
    methods = [m for m in METHOD_ORDER if m in set(summary["method"])]
    fig, axes = plt.subplots(1, len(methods), figsize=(2.3 * len(methods), 2.5), sharey=True, squeeze=False)
    for ax, m in zip(axes[0], methods):
        sub = summary[summary.method == m]
        for s in STRATEGY_ORDER:
            g = sub[sub.strategy == s].sort_values("cost_ratio")
            ax.plot(g["cost_ratio"], g["eval_cost"], label=STRATEGY_LABEL[s], **STRATEGY_STYLE[s])
        ax.set_xscale("log")
        ax.set_xticks(sorted(sub["cost_ratio"].unique()))
        ax.set_xticklabels([f"{int(v)}" for v in sorted(sub["cost_ratio"].unique())])
        ax.minorticks_off()
        ax.set_title(METHOD_LABEL[m])
        ax.set_xlabel("Cost ratio $r$")
    axes[0, 0].set_ylabel("Expected cost per part\n(evaluation half)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=4, frameon=False, fontsize=7)
    return _save(fig, path)


def plot_robustness_heatmap(conditions: pd.DataFrame, path: str) -> str:
    """conditions: robustness_conditions.csv -> mean delta image AUROC per method x corruption x severity."""
    set_style()
    methods = [m for m in METHOD_ORDER if m in set(conditions["method"])]
    agg = conditions.groupby(["method", "corruption", "severity"])["delta_image_auroc"].mean().reset_index()
    vmax = float(np.nanmax(np.abs(agg["delta_image_auroc"])))
    fig, axes = plt.subplots(1, len(methods), figsize=(2.4 * len(methods), 2.7), sharey=True, squeeze=False)
    im = None
    for ax, m in zip(axes[0], methods):
        pv = agg[agg.method == m].pivot(index="corruption", columns="severity", values="delta_image_auroc")
        im = ax.imshow(pv.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        for (i, j), v in np.ndenumerate(pv.values):
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5,
                    color="white" if abs(v) > 0.6 * vmax else INK)
        ax.set_xticks(range(pv.shape[1]))
        ax.set_xticklabels([str(c) for c in pv.columns])
        ax.set_yticks(range(pv.shape[0]))
        ax.set_yticklabels([r.replace("_", " ") for r in pv.index])
        ax.set_title(METHOD_LABEL[m])
        ax.set_xlabel("Severity")
        ax.grid(False)
    cb = fig.colorbar(im, ax=axes[0].tolist(), fraction=0.04, pad=0.02)
    cb.set_label("Mean drop in image AUROC")
    return _save(fig, path)


def plot_coreset_scaling(bench: pd.DataFrame, path: str, dim: int = 128) -> str:
    set_style()
    sub = bench[bench.feature_dim_D == dim].sort_values("num_patches_N")
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.5))
    series = [("time_cpu_numpy_s", "CPU NumPy greedy (exact)", "#52514e", "o", "--"),
              ("time_gpu_greedy_s", "GPU torch greedy (exact)", "#2a78d6", "s", "-"),
              ("time_gpu_topk_s", "GPU top-50 batched (approx.)", "#eb6834", "^", "-")]
    for col, lab, colr, mk, ls in series:
        axes[0].plot(sub["num_patches_N"], sub[col], label=lab, color=colr, marker=mk, ls=ls)
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Patches $N$ (coreset $M = 0.1N$)")
    axes[0].set_ylabel("Selection time (s)")
    axes[0].legend(frameon=False, fontsize=7)
    for col, lab, colr, mk, ls in [("radius_gpu_greedy", "Exact greedy", "#2a78d6", "s", "-"),
                                   ("radius_gpu_topk", "Top-50 batched", "#eb6834", "^", "-"),
                                   ("radius_random", "Random subset", "#52514e", "o", "--")]:
        axes[1].plot(sub["num_patches_N"], sub[col] / sub["radius_gpu_greedy"], label=lab, color=colr,
                     marker=mk, ls=ls)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Patches $N$")
    axes[1].set_ylabel("Coverage radius / exact greedy")
    axes[1].legend(frameon=False, fontsize=7)
    return _save(fig, path)


def plot_latency_vs_aupro(points: pd.DataFrame, path: str) -> str:
    """points: per category x method with p50_model_ms and aupro (mean over seeds)."""
    set_style()
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    for m in METHOD_ORDER:
        g = points[points.method == m]
        if g.empty:
            continue
        ax.scatter(g["p50_model_ms"], g["aupro"], color=METHOD_COLOR[m], marker=METHOD_MARKER[m], s=28,
                   edgecolors="white", linewidths=0.8, label=METHOD_LABEL[m])
    ax.set_xlabel("P50 model latency (ms, batch 1)")
    ax.set_ylabel("AU-PRO (FPR $\\leq$ 0.3)")
    ax.legend(frameon=False, loc="lower right")
    return _save(fig, path)


def plot_operator_load(load: pd.DataFrame, path: str, priors: Sequence[float]) -> str:
    """load: operator_load.csv -> mean P(overload) per method x prior x regime."""
    set_style()
    agg = load.groupby(["method", "prior", "regime"])["p_overload"].mean().reset_index()
    regimes = ["iid", "burst", "drift"]
    fig, axes = plt.subplots(1, len(regimes), figsize=(6.8, 2.3), sharey=True)
    width = 0.26
    for ax, reg in zip(axes, regimes):
        for k, m in enumerate(METHOD_ORDER):
            g = agg[(agg.method == m) & (agg.regime == reg)].set_index("prior").reindex(priors)
            ax.bar(np.arange(len(priors)) + (k - 1) * width, g["p_overload"], width - 0.02,
                   color=METHOD_COLOR[m], label=METHOD_LABEL[m])
        ax.set_xticks(range(len(priors)))
        ax.set_xticklabels([f"{p:.0%}" for p in priors])
        ax.set_title(f"{reg} stream")
        ax.set_xlabel("Defect prior")
    axes[0].set_ylabel("P(window alerts > capacity)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=3, frameon=False, fontsize=7)
    return _save(fig, path)
