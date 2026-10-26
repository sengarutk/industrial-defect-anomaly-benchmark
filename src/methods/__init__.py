from typing import Any, Dict, Optional

from .autoencoder import ConvAutoencoder
from .base import BaseAnomalyDetector
from .padim import PaDiM
from .patchcore import PatchCore

__all__ = ["BaseAnomalyDetector", "PatchCore", "PaDiM", "ConvAutoencoder", "build_model", "METHOD_NAMES"]

METHOD_NAMES = {"patchcore": "PatchCore", "padim": "PaDiM", "autoencoder": "ConvAutoencoder"}


def build_model(method: str, params: Dict[str, Any], seed: int, device: Optional[str] = None) -> BaseAnomalyDetector:
    """Instantiates a detector from its config block; the run seed reaches every stochastic component."""
    method = method.lower()
    if method == "patchcore":
        return PatchCore(device=device, seed=seed, **params)
    if method == "padim":
        return PaDiM(device=device, seed=seed, **params)
    if method == "autoencoder":
        return ConvAutoencoder(device=device, **params)
    raise ValueError(f"Unknown method: {method}")
