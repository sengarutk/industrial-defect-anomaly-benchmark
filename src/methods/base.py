import abc
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader


class BaseAnomalyDetector(abc.ABC):
    """Unified fit/predict interface for all detectors."""

    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    @abc.abstractmethod
    def fit(self, dataloader: DataLoader) -> None:
        """Fits on nominal images; batches are tensors [B, 3, H, W] (or tuples whose first item is)."""

    @abc.abstractmethod
    def predict(self, x: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (image_scores [B], anomaly_maps [B, H, W]) for a batch [B, 3, H, W]."""
