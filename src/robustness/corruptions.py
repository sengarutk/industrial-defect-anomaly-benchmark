"""Synthetic image corruptions used as proxies for acquisition degradations.

All corruptions operate on RGB uint8 images *after* resizing to the model input
size, so a given severity has the same pixel-level strength for every category.
Gaussian noise draws from an explicit RNG to keep evaluations reproducible.
"""
from typing import Optional

import cv2
import numpy as np

CORRUPTION_TYPES = [
    "gaussian_blur",
    "motion_blur",
    "brightness_drop",
    "gaussian_noise",
    "jpeg_compression",
    "downscale_restore",
]

_PARAMS = {
    "gaussian_blur": {1: ((5, 5), 1.0), 2: ((9, 9), 2.0), 3: ((15, 15), 3.5)},
    "motion_blur": {1: 5, 2: 11, 3: 19},
    "brightness_drop": {1: 0.75, 2: 0.50, 3: 0.30},
    "gaussian_noise": {1: 15.0, 2: 30.0, 3: 50.0},
    "jpeg_compression": {1: 50, 2: 25, 3: 10},
    # target side length relative to the 256 px input
    "downscale_restore": {1: 128, 2: 64, 3: 32},
}


def _check(img: np.ndarray) -> np.ndarray:
    if not isinstance(img, np.ndarray) or img.dtype != np.uint8 or img.ndim != 3 or img.shape[2] != 3:
        raise TypeError("corruptions expect an RGB uint8 array of shape (H, W, 3)")
    return img


def apply_gaussian_blur(img: np.ndarray, severity: int) -> np.ndarray:
    ksize, sigma = _PARAMS["gaussian_blur"][severity]
    return cv2.GaussianBlur(_check(img), ksize, sigmaX=sigma, sigmaY=sigma)


def apply_motion_blur(img: np.ndarray, severity: int) -> np.ndarray:
    k = _PARAMS["motion_blur"][severity]
    kernel = np.eye(k, dtype=np.float32) / k
    return cv2.filter2D(_check(img), -1, kernel)


def apply_brightness_drop(img: np.ndarray, severity: int) -> np.ndarray:
    factor = _PARAMS["brightness_drop"][severity]
    return np.clip(_check(img).astype(np.float32) * factor, 0, 255).astype(np.uint8)


def apply_gaussian_noise(img: np.ndarray, severity: int, rng: Optional[np.random.RandomState] = None) -> np.ndarray:
    rng = rng if rng is not None else np.random.RandomState(0)
    sigma = _PARAMS["gaussian_noise"][severity]
    noisy = _check(img).astype(np.float32) + rng.normal(0.0, sigma, img.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def apply_jpeg_compression(img: np.ndarray, severity: int) -> np.ndarray:
    quality = _PARAMS["jpeg_compression"][severity]
    bgr = cv2.cvtColor(_check(img), cv2.COLOR_RGB2BGR)
    ok, enc = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return cv2.cvtColor(cv2.imdecode(enc, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def apply_downscale_restore(img: np.ndarray, severity: int) -> np.ndarray:
    h, w = _check(img).shape[:2]
    side = _PARAMS["downscale_restore"][severity]
    small = cv2.resize(img, (side, side), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


_DISPATCH = {
    "gaussian_blur": apply_gaussian_blur,
    "motion_blur": apply_motion_blur,
    "brightness_drop": apply_brightness_drop,
    "jpeg_compression": apply_jpeg_compression,
    "downscale_restore": apply_downscale_restore,
}


def apply_corruption(img: np.ndarray, corruption_type: str, severity: int,
                     rng: Optional[np.random.RandomState] = None) -> np.ndarray:
    if severity not in (1, 2, 3):
        raise ValueError(f"Severity must be 1, 2 or 3, got {severity}")
    if corruption_type == "gaussian_noise":
        return apply_gaussian_noise(img, severity, rng=rng)
    if corruption_type not in _DISPATCH:
        raise ValueError(f"Unknown corruption type '{corruption_type}'. Supported: {CORRUPTION_TYPES}")
    return _DISPATCH[corruption_type](img, severity)
