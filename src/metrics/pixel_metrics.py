import warnings

import numpy as np
from scipy.ndimage import label
from sklearn.metrics import average_precision_score, roc_auc_score


def compute_pixel_auroc(ground_truth_masks: np.ndarray, anomaly_maps: np.ndarray) -> float:
    gt = (np.asarray(ground_truth_masks) > 0.5).ravel()
    if gt.all() or not gt.any():
        warnings.warn("compute_pixel_auroc: only one class present; returning NaN.")
        return float("nan")
    return float(roc_auc_score(gt, np.asarray(anomaly_maps).ravel()))


def compute_pixel_ap(ground_truth_masks: np.ndarray, anomaly_maps: np.ndarray) -> float:
    gt = (np.asarray(ground_truth_masks) > 0.5).ravel()
    if gt.all() or not gt.any():
        warnings.warn("compute_pixel_ap: only one class present; returning NaN.")
        return float("nan")
    return float(average_precision_score(gt, np.asarray(anomaly_maps).ravel()))


def pro_curve(ground_truth_masks: np.ndarray, anomaly_maps: np.ndarray):
    """Exact per-region-overlap curve evaluated at every distinct score.

    Follows the MVTec AD evaluation protocol (Bergmann et al., 2020/2021):
    FPR is measured over all nominal pixels of all images; PRO is the mean,
    over connected ground-truth components, of the fraction of each component
    covered by the thresholded map. Returns (fpr, pro) arrays starting at (0, 0).
    """
    gt = np.asarray(ground_truth_masks) > 0.5
    scores = np.asarray(anomaly_maps, dtype=np.float64)
    if gt.shape != scores.shape:
        raise ValueError(f"mask shape {gt.shape} != map shape {scores.shape}")

    weights_pro = np.zeros(gt.shape, dtype=np.float64)
    sizes = []
    comp_masks = []
    for i in range(gt.shape[0]):
        if not gt[i].any():
            continue
        lab, n = label(gt[i])
        for k in range(1, n + 1):
            region = lab == k
            sizes.append(int(region.sum()))
            comp_masks.append((i, region))
    n_comp = len(sizes)
    n_normal = int((~gt).sum())
    if n_comp == 0 or n_normal == 0:
        return None
    for (i, region), size in zip(comp_masks, sizes):
        weights_pro[i][region] = 1.0 / (n_comp * size)
    weights_fpr = (~gt).astype(np.float64) / n_normal

    flat = scores.ravel()
    order = np.argsort(-flat, kind="stable")
    s_sorted = flat[order]
    cum_fpr = np.cumsum(weights_fpr.ravel()[order])
    cum_pro = np.cumsum(weights_pro.ravel()[order])
    last_of_tie = np.r_[s_sorted[1:] != s_sorted[:-1], True]
    fpr = np.r_[0.0, cum_fpr[last_of_tie]]
    pro = np.r_[0.0, cum_pro[last_of_tie]]
    return np.clip(fpr, 0.0, 1.0), np.clip(pro, 0.0, 1.0)


def integrate_up_to(fpr: np.ndarray, pro: np.ndarray, max_fpr: float) -> float:
    """Normalised area under (fpr, pro) for fpr in [0, max_fpr], linearly interpolating the crossing point."""
    cut = np.searchsorted(fpr, max_fpr, side="right")
    x = fpr[:cut]
    y = pro[:cut]
    if cut < len(fpr) and x[-1] < max_fpr:
        x0, x1, y0, y1 = fpr[cut - 1], fpr[cut], pro[cut - 1], pro[cut]
        y_end = y0 + (y1 - y0) * (max_fpr - x0) / (x1 - x0) if x1 > x0 else y0
        x = np.r_[x, max_fpr]
        y = np.r_[y, y_end]
    trapz = getattr(np, "trapezoid", None) or np.trapz
    return float(trapz(y, x) / max_fpr)


def compute_aupro(ground_truth_masks: np.ndarray, anomaly_maps: np.ndarray, max_fpr: float = 0.30) -> float:
    """Normalised area under the PRO curve up to max_fpr (MVTec AD protocol, default 0.30)."""
    curve = pro_curve(ground_truth_masks, anomaly_maps)
    if curve is None:
        warnings.warn("compute_aupro: masks contain no defect components or no nominal pixels; returning NaN.")
        return float("nan")
    return integrate_up_to(curve[0], curve[1], max_fpr)
