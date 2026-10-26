from typing import Dict

import numpy as np

from src.metrics.image_metrics import compute_image_ap, compute_image_auroc

AGGREGATION_RULES = ["max", "percentile_99", "percentile_95", "top1pct_mean", "mean"]


def aggregate_anomaly_maps(maps: np.ndarray, rule: str) -> np.ndarray:
    """Reduces maps [N, H, W] to image scores [N] with the given pooling rule."""
    flat = np.asarray(maps, dtype=np.float32).reshape(len(maps), -1)
    if rule == "max":
        return flat.max(axis=1)
    if rule == "percentile_99":
        return np.percentile(flat, 99.0, axis=1)
    if rule == "percentile_95":
        return np.percentile(flat, 95.0, axis=1)
    if rule == "top1pct_mean":
        k = max(1, int(0.01 * flat.shape[1]))
        return np.mean(np.partition(flat, -k, axis=1)[:, -k:], axis=1)
    if rule == "mean":
        return flat.mean(axis=1)
    raise ValueError(f"Unknown aggregation rule: {rule}")


def run_aggregation_ablation(maps: np.ndarray, labels: np.ndarray) -> Dict[str, Dict[str, float]]:
    out = {}
    for rule in AGGREGATION_RULES:
        s = aggregate_anomaly_maps(maps, rule)
        out[rule] = {"image_auroc": compute_image_auroc(labels, s), "image_ap": compute_image_ap(labels, s)}
    return out
