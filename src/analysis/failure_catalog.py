"""Failure-case mining at a deployable threshold.

Definitions (threshold tau derived from held-out calibration nominals, never from test labels):
  false_positives          nominal test images with score >= tau, highest score first
  false_negatives          defective test images with score < tau, lowest score first
  localization_mismatches  detected defective images (score >= tau) whose top-1% map region
                           has the lowest IoU with the ground-truth mask
"""
from typing import Dict, List

import numpy as np


def top_region_iou(amap: np.ndarray, mask: np.ndarray, top_fraction: float = 0.01) -> float:
    k = max(1, int(top_fraction * amap.size))
    thr = np.partition(amap.ravel(), -k)[-k]
    pred = amap >= thr
    gt = mask > 0
    union = np.logical_or(pred, gt).sum()
    return float(np.logical_and(pred, gt).sum() / union) if union else 0.0


def mine_failures(scores: np.ndarray, labels: np.ndarray, maps: np.ndarray, masks: np.ndarray, tau: float,
                  top_k: int = 3) -> Dict[str, List[int]]:
    """Returns test-set indices for each failure type (lists may be empty)."""
    scores = np.asarray(scores)
    labels = np.asarray(labels)
    nom = np.where(labels == 0)[0]
    dfc = np.where(labels == 1)[0]
    fp = [int(i) for i in nom[np.argsort(-scores[nom])] if scores[i] >= tau][:top_k]
    fn = [int(i) for i in dfc[np.argsort(scores[dfc])] if scores[i] < tau][:top_k]
    detected = [int(i) for i in dfc if scores[i] >= tau]
    ious = {i: top_region_iou(maps[i].astype(np.float32), masks[i]) for i in detected}
    loc = sorted(ious, key=ious.get)[:top_k]
    return {"false_positives": fp, "false_negatives": fn, "localization_mismatches": loc}
