from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from src.mvtec import list_test_samples, load_mask, load_rgb, normalize_image

from .corruptions import apply_corruption


class CachedTestSet:
    """The validated test split of one category, decoded and resized once and kept in memory."""

    def __init__(self, root: str, category: str, img_size: int = 256):
        self.category = category
        self.samples: List[Dict[str, object]] = list_test_samples(root, category)
        self.images = np.stack([load_rgb(s["path"], img_size) for s in self.samples])
        self.masks = np.stack([load_mask(s["mask_path"], img_size) for s in self.samples])
        self.labels = np.array([s["label"] for s in self.samples], dtype=int)
        self.defect_types = np.array([s["defect_type"] for s in self.samples])
        self.paths = np.array([s["path"] for s in self.samples])

    def __len__(self) -> int:
        return len(self.samples)


class CorruptedTestDataset(Dataset):
    """Applies one corruption (or none) to a CachedTestSet; masks are never altered.

    Gaussian noise for item i uses RNG seed (noise_seed, i, severity), so every
    evaluation of the same condition sees identical noise.
    """

    def __init__(self, cache: CachedTestSet, corruption_type: Optional[str] = None, severity: int = 0,
                 noise_seed: int = 0):
        self.cache = cache
        self.corruption_type = corruption_type
        self.severity = severity
        self.noise_seed = noise_seed

    def __len__(self) -> int:
        return len(self.cache)

    def __getitem__(self, idx: int):
        img = self.cache.images[idx]
        if self.corruption_type is not None:
            rng = np.random.RandomState([self.noise_seed, idx, self.severity])
            img = apply_corruption(img, self.corruption_type, self.severity, rng=rng)
        x = normalize_image(np.ascontiguousarray(img))
        mask = torch.from_numpy(self.cache.masks[idx]).float().unsqueeze(0)
        return x, int(self.cache.labels[idx]), mask, idx
