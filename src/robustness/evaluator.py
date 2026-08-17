import os
import json
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.methods.base import BaseAnomalyDetector
from src.metrics.image_metrics import compute_image_auroc, compute_image_ap, compute_optimal_f1
from src.metrics.pixel_metrics import compute_pixel_auroc, compute_pixel_ap, compute_aupro
from src.metrics.calibration import compute_ece
from .corruptions import CORRUPTION_TYPES
from .dataset import CorruptedMVTecTest


class RobustnessEvaluator:
    """
    Comprehensive stress-testing engine evaluating anomaly detection baselines
    across clean benchmarks and 18 physical distribution-shift configurations.
    """
    def __init__(
        self,
        model: BaseAnomalyDetector,
        root: str,
        category: str,
        device: Optional[str] = None
    ):
        self.model = model
        self.root = root
        self.category = category
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def evaluate_split(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Runs inference over a dataloader and computes the full mathematical metric suite:
        - Image AUROC & Image AP
        - Optimal F1 score, precision, and recall
        - Pixel AUROC, Pixel AP, and AU-PRO (at max_fpr=0.30)
        - Expected Calibration Error (ECE)
        """
        all_image_scores = []
        all_labels = []
        all_anomaly_maps = []
        all_masks = []

        for batch in dataloader:
            x, y, mask, _ = batch
            scores, amaps = self.model.predict(x)

            all_image_scores.extend(scores.tolist())
            all_labels.extend(y.numpy().tolist())
            all_anomaly_maps.append(amaps)
            # mask is [B, 1, 256, 256] -> squeeze to [B, 256, 256]
            all_masks.append(mask.squeeze(1).numpy())

        scores_arr = np.array(all_image_scores)
        labels_arr = np.array(all_labels)
        amaps_arr = np.concatenate(all_anomaly_maps, axis=0)  # [N, 256, 256]
        masks_arr = np.concatenate(all_masks, axis=0)        # [N, 256, 256]

        # Compute full metric battery
        f1_dict = compute_optimal_f1(labels_arr, scores_arr)

        metrics = {
            "image_auroc": compute_image_auroc(labels_arr, scores_arr),
            "image_ap": compute_image_ap(labels_arr, scores_arr),
            "max_f1": f1_dict["max_f1"],
            "optimal_threshold": f1_dict["optimal_threshold"],
            "precision_at_optimal": f1_dict["precision_at_optimal"],
            "recall_at_optimal": f1_dict["recall_at_optimal"],
            "pixel_auroc": compute_pixel_auroc(masks_arr, amaps_arr),
            "pixel_ap": compute_pixel_ap(masks_arr, amaps_arr),
            "aupro": compute_aupro(masks_arr, amaps_arr, max_fpr=0.30),
            "ece": compute_ece(scores_arr, labels_arr, n_bins=15),
            "num_samples": len(labels_arr)
        }
        return metrics

    def run_full_stress_test(
        self,
        severities: List[int] = [1, 2, 3],
        batch_size: int = 4
    ) -> Dict[str, Any]:
        """
        Evaluates clean baseline M_clean followed by all 18 corruption combinations (6 types x 3 severities).
        Calculates metric drops (Delta M = M_clean - M_corrupted) and Mean Corruption Error (mCE).
        """
        # 1. Clean evaluation
        clean_ds = CorruptedMVTecTest(self.root, self.category, corruption_type=None)
        clean_loader = DataLoader(clean_ds, batch_size=batch_size, shuffle=False)
        clean_metrics = self.evaluate_split(clean_loader)

        # 2. Corrupted evaluations
        corrupted_results = []
        auroc_drops = []
        aupro_drops = []

        for c_type in CORRUPTION_TYPES:
            for sev in severities:
                corr_ds = CorruptedMVTecTest(self.root, self.category, corruption_type=c_type, severity=sev)
                corr_loader = DataLoader(corr_ds, batch_size=batch_size, shuffle=False)
                corr_metrics = self.evaluate_split(corr_loader)

                delta_image_auroc = clean_metrics["image_auroc"] - corr_metrics["image_auroc"]
                delta_aupro = clean_metrics["aupro"] - corr_metrics["aupro"]
                delta_pixel_auroc = clean_metrics["pixel_auroc"] - corr_metrics["pixel_auroc"]

                auroc_drops.append(delta_image_auroc)
                aupro_drops.append(delta_aupro)

                result_entry = {
                    "category": self.category,
                    "corruption_type": c_type,
                    "severity": sev,
                    "delta_image_auroc": delta_image_auroc,
                    "delta_aupro": delta_aupro,
                    "delta_pixel_auroc": delta_pixel_auroc,
                    **corr_metrics
                }
                corrupted_results.append(result_entry)

        # 3. Aggregate robustness summary metrics
        mCE_auroc = float(np.mean(auroc_drops)) if len(auroc_drops) > 0 else 0.0
        mCE_aupro = float(np.mean(aupro_drops)) if len(aupro_drops) > 0 else 0.0

        summary = {
            "category": self.category,
            "clean_metrics": clean_metrics,
            "corrupted_results": corrupted_results,
            "mCE_image_auroc": mCE_auroc,
            "mCE_aupro": mCE_aupro,
            "num_conditions": len(corrupted_results)
        }
        return summary

    def save_results(self, results: Dict[str, Any], output_path: str) -> None:
        """
        Serializes results to JSON and CSV formats under output_path directory.
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        # JSON dump
        json_path = output_path if output_path.endswith(".json") else f"{output_path}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        # CSV dump of corrupted results table
        csv_path = json_path.replace(".json", ".csv")
        df = pd.DataFrame(results["corrupted_results"])
        df.to_csv(csv_path, index=False)
