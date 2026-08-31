from typing import Dict, Any, Tuple, Optional
import numpy as np


class CostCalibratedThresholdOptimizer:
    """
    Cost-Calibrated Thresholding (CCT) optimizer:
    Formulates decision threshold selection as an empirical risk minimization problem under
    asymmetric defect escape costs and operator false alert budget constraints.
    """

    @staticmethod
    def compute_empirical_cost_curve(
        scores: np.ndarray,
        labels: np.ndarray,
        cost_ratio: float = 10.0,
        prior: float = 0.01,
        num_thresholds: int = 200
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Sweeps candidate thresholds tau over empirical score distribution to compute
        FPR(tau), FNR(tau), and expected unit risk:
          C(tau) = (1 - p) * FPR(tau) * 1.0 + p * FNR(tau) * cost_ratio
          
        Returns: (thresholds, cost_curve, fpr_curve, fnr_curve)
        """
        scores = np.asarray(scores, dtype=np.float64).ravel()
        labels = np.asarray(labels, dtype=int).ravel()

        if len(scores) == 0:
            return np.array([]), np.array([]), np.array([]), np.array([])

        nom_mask = (labels == 0)
        def_mask = (labels == 1)
        n_nom = np.sum(nom_mask)
        n_def = np.sum(def_mask)

        # Generate candidate thresholds spanning empirical range
        min_s = float(np.min(scores))
        max_s = float(np.max(scores))
        if min_s == max_s:
            thresholds = np.array([min_s])
        else:
            thresholds = np.linspace(min_s - 1e-5, max_s + 1e-5, num_thresholds)

        cost_curve = []
        fpr_curve = []
        fnr_curve = []

        for tau in thresholds:
            fpr = float(np.sum(scores[nom_mask] >= tau) / n_nom) if n_nom > 0 else 0.0
            fnr = float(np.sum(scores[def_mask] < tau) / n_def) if n_def > 0 else 0.0
            unit_cost = (1.0 - prior) * fpr * 1.0 + prior * fnr * cost_ratio
            cost_curve.append(unit_cost)
            fpr_curve.append(fpr)
            fnr_curve.append(fnr)

        return (
            thresholds,
            np.array(cost_curve, dtype=np.float64),
            np.array(fpr_curve, dtype=np.float64),
            np.array(fnr_curve, dtype=np.float64)
        )

    @staticmethod
    def optimize_cct_threshold(
        val_scores: np.ndarray,
        val_labels: np.ndarray,
        cost_ratio: float = 10.0,
        prior: float = 0.01,
        max_alerts_per_1k: Optional[float] = 5.0
    ) -> Dict[str, Any]:
        """
        Solves the constrained empirical cost minimization problem:
          tau_CCT = argmin_tau C(tau)  s.t.  FA@1k(tau) <= max_alerts_per_1k
          
        If no threshold satisfies the budget, falls back to budget-constrained quantile threshold.
        """
        val_scores = np.asarray(val_scores, dtype=np.float64).ravel()
        val_labels = np.asarray(val_labels, dtype=int).ravel()

        if len(val_scores) == 0:
            return {
                "threshold": 0.0,
                "min_expected_cost": 0.0,
                "achieved_val_fpr": 0.0,
                "budget_satisfied": True
            }

        nom_scores = val_scores[val_labels == 0]
        max_allowed_fpr = (max_alerts_per_1k / 1000.0) if max_alerts_per_1k is not None else 1.0

        # High-resolution sweep over sorted unique validation scores
        thresholds = np.sort(np.unique(val_scores))
        # Add endpoints
        thresholds = np.concatenate([[thresholds[0] - 1e-5], thresholds, [thresholds[-1] + 1e-5]])

        n_nom = len(nom_scores)
        n_def = np.sum(val_labels == 1)

        best_tau = float(thresholds[0])
        min_cost = float("inf")
        achieved_fpr = 0.0
        budget_satisfied = False

        # First pass: find lowest cost satisfying FPR budget
        for tau in thresholds:
            fpr = float(np.sum(nom_scores >= tau) / n_nom) if n_nom > 0 else 0.0
            fnr = float(np.sum(val_scores[val_labels == 1] < tau) / n_def) if n_def > 0 else 0.0

            if fpr <= max_allowed_fpr + 1e-7:
                unit_cost = (1.0 - prior) * fpr * 1.0 + prior * fnr * cost_ratio
                if unit_cost < min_cost:
                    min_cost = unit_cost
                    best_tau = float(tau)
                    achieved_fpr = fpr
                    budget_satisfied = True

        # Fallback if unconstrained or no point satisfies budget
        if not budget_satisfied:
            if n_nom > 0:
                best_tau = float(np.quantile(nom_scores, max(0.0, min(1.0, 1.0 - max_allowed_fpr))))
                achieved_fpr = float(np.sum(nom_scores >= best_tau) / n_nom)
                fnr = float(np.sum(val_scores[val_labels == 1] < best_tau) / n_def) if n_def > 0 else 0.0
                min_cost = (1.0 - prior) * achieved_fpr * 1.0 + prior * fnr * cost_ratio
            else:
                best_tau = float(thresholds[-1])
                min_cost = 0.0
                achieved_fpr = 0.0

        return {
            "threshold": best_tau,
            "min_expected_cost": float(min_cost),
            "achieved_val_fpr": float(achieved_fpr),
            "budget_satisfied": budget_satisfied
        }


def compute_empirical_cost_curve(
    scores: np.ndarray,
    labels: np.ndarray,
    cost_ratio: float = 10.0,
    prior: float = 0.01,
    num_thresholds: int = 200
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return CostCalibratedThresholdOptimizer.compute_empirical_cost_curve(
        scores, labels, cost_ratio=cost_ratio, prior=prior, num_thresholds=num_thresholds
    )


def optimize_cct_threshold(
    val_scores: np.ndarray,
    val_labels: np.ndarray,
    cost_ratio: float = 10.0,
    prior: float = 0.01,
    max_alerts_per_1k: Optional[float] = 5.0
) -> Dict[str, Any]:
    return CostCalibratedThresholdOptimizer.optimize_cct_threshold(
        val_scores, val_labels, cost_ratio=cost_ratio, prior=prior, max_alerts_per_1k=max_alerts_per_1k
    )