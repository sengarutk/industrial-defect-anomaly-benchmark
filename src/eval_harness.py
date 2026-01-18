import os
import time
from typing import Dict, Any

import numpy as np
import cv2

from .metrics import auroc
from .utils import ensure_dir


def save_heatmap_overlay(img_tensor, heatmap, out_path: str):
    """
    img_tensor: [3,H,W] in [0,1]
    heatmap: [H',W'] float
    """
    ensure_dir(os.path.dirname(out_path))

    img = (img_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    heat = heatmap.astype(np.float32)
    heat = cv2.resize(heat, (img.shape[1], img.shape[0]))
    heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)
    heat_color = cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_JET)

    overlay = (0.65 * img + 0.35 * heat_color).astype(np.uint8)
    cv2.imwrite(out_path, overlay[:, :, ::-1])  # RGB->BGR for cv2 writing


def evaluate_category(encoder, knn, memorybank, test_loader, device: str, save_dir: str = None, save_heatmaps: bool = True):
    """
    Evaluate one MVTec category, returning image-level AUROC and latency stats.
    Works with both:
      - patch scoring (heatmap available)
      - global scoring (heatmap None)
    """
    y_true, y_score = [], []
    latencies = []

    saved = 0
    max_save = 12

    for x, y, mask, meta in test_loader:
        t0 = time.time()
        score, heat = memorybank["score_fn"](encoder, x, knn, memorybank, device)
        dt = time.time() - t0

        y_true.append(int(y))
        y_score.append(float(score))
        latencies.append(dt)

        # only save overlays for patch mode
        if heat is not None and save_heatmaps and save_dir is not None and saved < max_save:
            out_path = os.path.join(save_dir, f"overlay_{saved}_y{int(y)}_{meta['defect_type'][0]}.png")
            save_heatmap_overlay(x[0], heat, out_path)
            saved += 1

    return {
        "auroc_image": auroc(y_true, y_score),
        "avg_latency_s": float(np.mean(latencies)) if len(latencies) else 0.0,
        "num_test": len(y_true),
    }
