from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from torch.utils.data import DataLoader

from src.methods.base import BaseAnomalyDetector
from src.metrics.image_metrics import compute_image_ap, compute_image_auroc
from src.metrics.pixel_metrics import compute_aupro, compute_pixel_ap, compute_pixel_auroc

from .corruptions import CORRUPTION_TYPES
from .dataset import CachedTestSet, CorruptedTestDataset


def predict_dataset(model: BaseAnomalyDetector, dataset, batch_size: int = 8) -> Tuple[np.ndarray, np.ndarray]:
    """Runs the model over a dataset in order; returns (image_scores [N], anomaly_maps [N,H,W] float32)."""
    scores, maps = [], []
    for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False):
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        s, m = model.predict(x)
        scores.append(np.asarray(s, dtype=np.float64))
        maps.append(np.asarray(m, dtype=np.float32))
    return np.concatenate(scores), np.concatenate(maps)


def evaluate_maps(labels: np.ndarray, scores: np.ndarray, masks: np.ndarray, maps: np.ndarray) -> Dict[str, float]:
    return {
        "image_auroc": compute_image_auroc(labels, scores),
        "image_ap": compute_image_ap(labels, scores),
        "pixel_auroc": compute_pixel_auroc(masks, maps),
        "pixel_ap": compute_pixel_ap(masks, maps),
        "aupro": compute_aupro(masks, maps),
    }


def run_stress_test(model: BaseAnomalyDetector, cache: CachedTestSet, clean_metrics: Dict[str, float],
                    corruptions: Sequence[str] = CORRUPTION_TYPES, severities: Sequence[int] = (1, 2, 3),
                    batch_size: int = 8, noise_seed: int = 0) -> Dict[str, Any]:
    """Evaluates every (corruption, severity) condition and summarises the degradation.

    MRD (non-negative mean robustness degradation) = mean_i max(0, clean - corrupted_i)
    MPC (signed mean performance change)           = mean_i (clean - corrupted_i)
    """
    rows: List[Dict[str, Any]] = []
    for ctype in corruptions:
        for sev in severities:
            ds = CorruptedTestDataset(cache, ctype, sev, noise_seed=noise_seed)
            s, m = predict_dataset(model, ds, batch_size)
            met = evaluate_maps(cache.labels, s, cache.masks, m)
            rows.append({"corruption": ctype, "severity": sev, **met,
                         "delta_image_auroc": clean_metrics["image_auroc"] - met["image_auroc"],
                         "delta_aupro": clean_metrics["aupro"] - met["aupro"]})
    d_auc = np.array([r["delta_image_auroc"] for r in rows])
    d_pro = np.array([r["delta_aupro"] for r in rows])
    return {
        "conditions": rows,
        "mrd_image_auroc": float(np.mean(np.maximum(0.0, d_auc))),
        "mrd_aupro": float(np.mean(np.maximum(0.0, d_pro))),
        "mpc_image_auroc": float(np.mean(d_auc)),
        "mpc_aupro": float(np.mean(d_pro)),
    }
