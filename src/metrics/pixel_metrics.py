import warnings
from typing import Union, List, Tuple
import numpy as np
import scipy.ndimage
from sklearn.metrics import roc_auc_score, average_precision_score


def compute_pixel_auroc(masks: np.ndarray, amaps: np.ndarray) -> float:
    """
    Computes pixel-level Area Under the ROC curve.
    masks: binary ground truth masks [N, H, W] or [H, W] in {0, 1}
    amaps: continuous anomaly maps [N, H, W] or [H, W]
    """
    y_true = np.asarray(masks).flatten().astype(int)
    y_score = np.asarray(amaps).flatten().astype(float)

    if len(np.unique(y_true)) < 2:
        warnings.warn("compute_pixel_auroc: Ground truth contains only one class. Returning 0.5.")
        return 0.5

    try:
        score = float(roc_auc_score(y_true, y_score))
        return score
    except ValueError:
        return 0.5


def compute_pixel_ap(masks: np.ndarray, amaps: np.ndarray) -> float:
    """
    Computes pixel-level Average Precision (AP / PR-AUC).
    masks: binary ground truth masks [N, H, W] or [H, W] in {0, 1}
    amaps: continuous anomaly maps [N, H, W] or [H, W]
    """
    y_true = np.asarray(masks).flatten().astype(int)
    y_score = np.asarray(amaps).flatten().astype(float)

    if len(np.unique(y_true)) < 2:
        warnings.warn("compute_pixel_ap: Ground truth contains only one class. Returning 0.0.")
        return 0.0

    try:
        score = float(average_precision_score(y_true, y_score))
        return score
    except ValueError:
        return 0.0


def compute_aupro(
    masks: np.ndarray,
    amaps: np.ndarray,
    max_fpr: float = 0.30,
    num_thresholds: int = 100
) -> float:
    """
    Computes the official MVTec AD Per-Region Overlap (AU-PRO) metric for structural defect localization.

    Protocol:
      1. Extract all individual connected defect components from the ground truth masks using scipy.ndimage.label.
      2. Scan threshold candidates from slightly above max(amap) to min(amap).
      3. For each threshold, binarize prediction, calculate global False Positive Rate over normal pixels,
         and calculate mean overlap across all connected defect components.
      4. Filter points to FPR <= max_fpr (default 0.30), interpolate at max_fpr, and integrate normalized area.
    """
    masks_arr = np.asarray(masks)
    amaps_arr = np.asarray(amaps)

    # Standardize dimensions to [N, H, W]
    if masks_arr.ndim == 2:
        masks_arr = masks_arr[np.newaxis, ...]
        amaps_arr = amaps_arr[np.newaxis, ...]
    elif masks_arr.ndim == 4 and masks_arr.shape[1] == 1:
        masks_arr = np.squeeze(masks_arr, axis=1)
        amaps_arr = np.squeeze(amaps_arr, axis=1)

    masks_bin = (masks_arr > 0.5).astype(np.uint8)
    N, H, W = masks_bin.shape

    # 1. Extract connected components for each image
    components: List[Tuple[int, np.ndarray, int]] = []
    for i in range(N):
        labeled_mask, num_features = scipy.ndimage.label(masks_bin[i])
        for f in range(1, num_features + 1):
            comp_mask = (labeled_mask == f)
            area = int(np.sum(comp_mask))
            if area > 0:
                components.append((i, comp_mask, area))

    if len(components) == 0:
        warnings.warn("compute_aupro: No defect components found in masks. Returning 0.0.")
        return 0.0

    total_normal_pixels = np.sum(1 - masks_bin)
    if total_normal_pixels == 0:
        warnings.warn("compute_aupro: No normal pixels found in masks. Returning 0.0.")
        return 0.0

    # 2. Threshold candidates (sample from max_val + eps down to min_val)
    min_val = float(np.min(amaps_arr))
    max_val = float(np.max(amaps_arr))
    eps = (max_val - min_val) * 1e-4 if max_val > min_val else 1e-6

    if min_val == max_val:
        thresholds = np.array([max_val + eps, min_val])
    else:
        thresholds = np.linspace(max_val + eps, min_val, num_thresholds)

    fpr_list: List[float] = []
    pro_list: List[float] = []

    # 3. Evaluate each threshold
    for thresh in thresholds:
        pred = (amaps_arr >= thresh)
        fp_count = np.sum(pred & (masks_bin == 0))
        fpr = float(fp_count / (total_normal_pixels + 1e-8))

        overlaps = [
            np.sum(pred[img_idx] & comp_mask) / area
            for img_idx, comp_mask, area in components
        ]
        pro = float(np.mean(overlaps))

        fpr_list.append(fpr)
        pro_list.append(pro)

    # 4. Sort points by FPR ascending, then PRO ascending
    sorted_pairs = sorted(zip(fpr_list, pro_list), key=lambda x: (x[0], x[1]))
    all_fprs = [p[0] for p in sorted_pairs]
    all_pros = [p[1] for p in sorted_pairs]

    # Prepend (0, 0) if necessary
    if all_fprs[0] > 0.0:
        all_fprs.insert(0, 0.0)
        all_pros.insert(0, 0.0)

    # Filter points where FPR <= max_fpr
    valid_indices = [i for i, f in enumerate(all_fprs) if f <= max_fpr]
    filtered_fprs = [all_fprs[i] for i in valid_indices]
    filtered_pros = [all_pros[i] for i in valid_indices]

    if len(filtered_fprs) == 0:
        return 0.0

    if filtered_fprs[-1] < max_fpr:
        interp_pro = float(np.interp(max_fpr, all_fprs, all_pros))
        filtered_fprs.append(max_fpr)
        filtered_pros.append(interp_pro)

    # 5. Integrate normalized area under curve
    area = np.trapz(filtered_pros, filtered_fprs)
    aupro = float(area / max_fpr)
    return float(np.clip(aupro, 0.0, 1.0))
