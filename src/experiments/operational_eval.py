"""Production-stream projections built by resampling measured scores.

These streams contain no information beyond the measured score pools: each
part's score is drawn with replacement from the nominal or defect test scores.
They are used only to translate per-part error rates into operator workload
(alerts per window, overload probability) under different arrival patterns.
Their resolution is limited by the size of the pools they resample.
"""
from typing import Tuple

import numpy as np


class ProductionStreamSimulator:
    def __init__(self, nominal_scores: np.ndarray, defect_scores: np.ndarray, seed: int = 0):
        self.nominal_scores = np.asarray(nominal_scores, dtype=np.float64).ravel()
        self.defect_scores = np.asarray(defect_scores, dtype=np.float64).ravel()
        if self.nominal_scores.size == 0 or self.defect_scores.size == 0:
            raise ValueError("Both nominal and defect score pools must be non-empty")
        self.rng = np.random.RandomState(seed)

    def _fill(self, labels: np.ndarray) -> np.ndarray:
        scores = np.empty(labels.size, dtype=np.float64)
        n_def = int(labels.sum())
        scores[labels == 0] = self.rng.choice(self.nominal_scores, size=labels.size - n_def, replace=True)
        scores[labels == 1] = self.rng.choice(self.defect_scores, size=n_def, replace=True)
        return scores

    def simulate_iid_stream(self, n_total: int, defect_prior: float) -> Tuple[np.ndarray, np.ndarray]:
        """Exactly round(n_total * prior) defects at uniformly random positions."""
        n_def = int(round(n_total * defect_prior))
        labels = np.zeros(n_total, dtype=int)
        labels[self.rng.choice(n_total, size=n_def, replace=False)] = 1
        return labels, self._fill(labels)

    def simulate_burst_stream(self, n_total: int, defect_prior: float,
                              mean_block_length: float) -> Tuple[np.ndarray, np.ndarray]:
        """Two-state Markov chain with stationary defect rate `defect_prior` and mean burst length L.

        P(1 -> 0) = 1 / L and P(0 -> 1) = prior / (L * (1 - prior)).
        """
        L = max(1.0, float(mean_block_length))
        p10 = 1.0 / L
        p01 = min(1.0, defect_prior / (L * (1.0 - defect_prior)))
        u = self.rng.rand(n_total)
        labels = np.empty(n_total, dtype=int)
        state = int(self.rng.rand() < defect_prior)
        for t in range(n_total):
            labels[t] = state
            state = int(u[t] < p01) if state == 0 else int(u[t] >= p10)
        return labels, self._fill(labels)

    def simulate_drift_stream(self, n_total: int, defect_prior: float,
                              drift_slope_sigma: float) -> Tuple[np.ndarray, np.ndarray]:
        """IID stream plus a linear score offset rising to drift_slope_sigma * std(nominal) at the end."""
        labels, scores = self.simulate_iid_stream(n_total, defect_prior)
        sigma = float(np.std(self.nominal_scores)) or 1.0
        return labels, scores + np.linspace(0.0, 1.0, n_total) * drift_slope_sigma * sigma
