"""Renders failure cases from stored archives at the primary alarm-budget threshold.

For each category x method (first configured seed) one figure is written to
<results>/figures/failure_cases/<category>_<method>.png, plus failure_cases.csv
listing every mined case with its score and threshold.
"""
import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.analysis import mine_failures, top_region_iou  # noqa: E402
from src.methods import METHOD_NAMES  # noqa: E402
from src.metrics.operational import alert_budget_threshold  # noqa: E402
from src.mvtec import load_rgb  # noqa: E402
from src.utils import load_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Failure-case mining from score archives")
    parser.add_argument("--config", default="configs/benchmark.yaml")
    parser.add_argument("--operational-config", default="configs/operational.yaml")
    parser.add_argument("--results-dir", default="results/mvtec_ad")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()
    cfg = load_config(args.config)
    budget = load_config(args.operational_config)["primary_alert_budget_per_1k"]
    seed = cfg["seeds"][0]
    out_dir = os.path.join(args.results_dir, "figures", "failure_cases")
    os.makedirs(out_dir, exist_ok=True)
    records = []
    for cat in cfg["dataset"]["categories"]:
        for method in cfg["methods"]:
            key = f"{cat}_{method}_{seed}"
            s = np.load(os.path.join(args.results_dir, "scores", f"{key}.npz"))
            m = np.load(os.path.join(args.results_dir, "maps", f"{key}.npz"))
            tau = alert_budget_threshold(s["calib_scores"], budget)
            cases = mine_failures(s["image_scores"], s["image_labels"], m["pixel_amaps"], m["ground_truth_masks"],
                                  tau, args.top_k)
            items = [(kind, i) for kind, idx in cases.items() for i in idx]
            for kind, i in items:
                records.append({"category": cat, "method": method, "seed": seed, "type": kind,
                                "image_path": str(s["image_paths"][i]), "defect_type": str(s["defect_types"][i]),
                                "score": float(s["image_scores"][i]), "threshold": tau,
                                "top1pct_iou": top_region_iou(m["pixel_amaps"][i].astype(np.float32),
                                                              m["ground_truth_masks"][i])})
            if not items:
                continue
            fig, axes = plt.subplots(len(items), 3, figsize=(9, 3 * len(items)), squeeze=False)
            vmax = float(np.percentile(m["pixel_amaps"].astype(np.float32), 99.9))
            for row, (kind, i) in enumerate(items):
                img = load_rgb(os.path.join(cfg["dataset"]["root"], str(s["image_paths"][i])), cfg["dataset"]["img_size"])
                axes[row, 0].imshow(img)
                axes[row, 1].imshow(m["ground_truth_masks"][i], cmap="gray", vmin=0, vmax=1)
                axes[row, 2].imshow(m["pixel_amaps"][i].astype(np.float32), cmap="jet", vmin=0, vmax=vmax)
                axes[row, 0].set_ylabel(f"{kind.replace('_', ' ')}\n{s['defect_types'][i]}", fontsize=8)
                axes[row, 2].set_title(f"score {s['image_scores'][i]:.3f} vs $\\tau$ {tau:.3f}", fontsize=8)
                for ax in axes[row]:
                    ax.set_xticks([])
                    ax.set_yticks([])
            axes[0, 0].set_title("image", fontsize=9)
            axes[0, 1].set_title("ground truth", fontsize=9)
            fig.suptitle(f"{cat} / {METHOD_NAMES[method]} (seed {seed}, budget {budget:g}/1k)", fontsize=10)
            fig.tight_layout(rect=(0, 0, 1, 1 - 0.35 / (3 * len(items))))
            fig.savefig(os.path.join(out_dir, f"{cat}_{method}.png"), dpi=110)
            plt.close(fig)
    pd.DataFrame(records).to_csv(os.path.join(args.results_dir, "tables", "failure_cases.csv"), index=False)
    print(f"Wrote {len(records)} failure cases to {out_dir}")


if __name__ == "__main__":
    main()
