import numpy as np
from typing import Dict, Any, Tuple, Optional

from src.metrics.image_metrics import compute_optimal_f1, compute_quantile_threshold
from src.metrics.operational import (
    compute_fa_at_1k,
    compute_md_at_1k,
    compute_cost_weighted_error
)


class ProductionStreamSimulator:
    """
    Simulates high-throughput production inspection streams with realistic imbalanced defect priors (e.g. 1%, 5%, 15%)
    by resampling empirical score pools with replacement, and evaluates operational threshold selection strategies.
    """
    def __init__(
        self,
        nominal_scores: np.ndarray,
        defect_scores: np.ndarray,
        seed: int = 42
    ):
        self.nominal_scores = np.asarray(nominal_scores, dtype=float)
        self.defect_scores = np.asarray(defect_scores, dtype=float)
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def simulate_stream(
        self,
        n_total: int = 10000,
        defect_prior: float = 0.01
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Resamples with replacement from empirical score pools to construct an inspection stream
        of n_total items with exact defect prior p.
        Returns:
            stream_labels: np.ndarray of shape [n_total], dtype int (0 = normal, 1 = defect)
            stream_scores: np.ndarray of shape [n_total], dtype float
        """
        n_defect = int(round(n_total * defect_prior))
        n_nominal = n_total - n_defect

        # Handle empty nominal or defect pools gracefully
        if len(self.nominal_scores) == 0 and len(self.defect_scores) == 0:
            return np.zeros(n_total, dtype=int), np.zeros(n_total, dtype=float)
        elif len(self.nominal_scores) == 0:
            sample_nom = np.zeros(n_nominal, dtype=float)
            sample_def = self.rng.choice(self.defect_scores, size=n_defect, replace=True)
        elif len(self.defect_scores) == 0:
            sample_nom = self.rng.choice(self.nominal_scores, size=n_nominal, replace=True)
            sample_def = np.ones(n_defect, dtype=float)
        else:
            sample_nom = self.rng.choice(self.nominal_scores, size=n_nominal, replace=True)
            sample_def = self.rng.choice(self.defect_scores, size=n_defect, replace=True)

        labels = np.concatenate([np.zeros(n_nominal, dtype=int), np.ones(n_defect, dtype=int)])
        scores = np.concatenate([sample_nom, sample_def])

        # Permute stream ordering
        perm = self.rng.permutation(n_total)
        return labels[perm], scores[perm]

    def _find_cost_optimal_threshold(
        self,
        nominal_ref_scores: np.ndarray,
        defect_ref_scores: Optional[np.ndarray] = None,
        cost_ratio: float = 10.0,
        prior: float = 0.01,
        num_candidates: int = 200
    ) -> float:
        """
        Derives threshold that minimizes expected Cost-Weighted Error on reference score pools.
        """
        if len(nominal_ref_scores) == 0 and (defect_ref_scores is None or len(defect_ref_scores) == 0):
            return 0.5

        if defect_ref_scores is None or len(defect_ref_scores) == 0:
            defect_ref_scores = self.defect_scores

        if len(defect_ref_scores) == 0:
            # Fallback to nominal 99th percentile
            return float(np.percentile(nominal_ref_scores, 99.0))

        all_scores = np.concatenate([nominal_ref_scores, defect_ref_scores])
        if len(np.unique(all_scores)) <= num_candidates:
            candidates = np.sort(np.unique(all_scores))
        else:
            percentiles = np.linspace(0, 100, num_candidates)
            candidates = np.percentile(all_scores, percentiles)

        best_cost = float("inf")
        best_th = float(candidates[0])

        n_nom = len(nominal_ref_scores)
        n_def = len(defect_ref_scores)

        for th in candidates:
            fpr = np.mean(nominal_ref_scores >= th) if n_nom > 0 else 0.0
            fnr = np.mean(defect_ref_scores < th) if n_def > 0 else 0.0
            # Expected cost under defect prior p
            expected_cwe = (1.0 - prior) * fpr * 1.0 + prior * fnr * cost_ratio
            if expected_cwe < best_cost:
                best_cost = expected_cwe
                best_th = float(th)

        return best_th

    def evaluate_threshold_strategies(
        self,
        stream_labels: np.ndarray,
        stream_scores: np.ndarray,
        nominal_ref_scores: np.ndarray,
        cost_ratio: float = 10.0,
        defect_prior: float = 0.01
    ) -> Dict[str, Dict[str, float]]:
        """
        Compares three distinct operating threshold strategies:
          1. Oracle Maximum F1 (tau_oracle)
          2. Nominal Quantile 99% (tau_99)
          3. Cost-Optimal (tau_cost)
        """
        stream_labels = np.asarray(stream_labels, dtype=int)
        stream_scores = np.asarray(stream_scores, dtype=float)

        # 1. Oracle Max F1
        f1_res = compute_optimal_f1(stream_labels, stream_scores)
        tau_oracle = float(f1_res["optimal_threshold"])

        # 2. Nominal Quantile 99%
        if len(nominal_ref_scores) > 0:
            tau_99 = compute_quantile_threshold(nominal_ref_scores, quantile=0.99)
        else:
            norm_stream = stream_scores[stream_labels == 0]
            tau_99 = compute_quantile_threshold(norm_stream, quantile=0.99)

        # 3. Cost-Optimal
        tau_cost = self._find_cost_optimal_threshold(
            nominal_ref_scores=nominal_ref_scores,
            defect_ref_scores=self.defect_scores,
            cost_ratio=cost_ratio,
            prior=defect_prior
        )

        strategies = {
            "oracle_f1": tau_oracle,
            "nominal_quantile_99": tau_99,
            "cost_optimal": tau_cost
        }

        results: Dict[str, Dict[str, float]] = {}

        n_defect = np.sum(stream_labels == 1)
        for strat_name, th in strategies.items():
            fa_1k = compute_fa_at_1k(stream_labels, stream_scores, threshold=th)
            md_1k = compute_md_at_1k(stream_labels, stream_scores, threshold=th)
            cwe = compute_cost_weighted_error(stream_labels, stream_scores, threshold=th, cost_ratio=cost_ratio)
            if n_defect > 0:
                tpr = float(np.sum((stream_scores >= th) & (stream_labels == 1)) / n_defect)
            else:
                tpr = 0.0

            results[strat_name] = {
                "threshold": float(th),
                "fa_at_1k": float(fa_1k),
                "md_at_1k": float(md_1k),
                "cost_weighted_error": float(cwe),
                "cwe": float(cwe),
                "tpr": float(tpr)
            }

        return results
