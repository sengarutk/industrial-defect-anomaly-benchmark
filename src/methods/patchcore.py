from typing import Optional, Tuple

import numpy as np
import scipy.ndimage
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm
from torch.utils.data import DataLoader

from .base import BaseAnomalyDetector
from .coreset import greedy_kcenter, random_projection


class PatchCore(BaseAnomalyDetector):
    """PatchCore-style detector (Roth et al., CVPR 2022) with documented deviations.

    Deviations from the paper: ResNet-18 instead of WideResNet-50, 256x256 input,
    no score re-weighting. Features: layer2 + upsampled layer3 after 3x3 average
    pooling (384-d). Memory bank: exact greedy k-center coreset selected in a
    random-projected space (projection and start point drawn from `seed`).
    Image score: max of the Gaussian-smoothed nearest-neighbour distance map.
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        coreset_sampling_ratio: float = 0.10,
        projection_dim: int = 128,
        smoothing_sigma: float = 4.0,
        device: Optional[str] = None,
        seed: int = 0,
    ):
        super().__init__(device=device)
        if backbone != "resnet18":
            raise ValueError(f"Unsupported backbone: {backbone}")
        self.backbone_name = backbone
        self.coreset_sampling_ratio = coreset_sampling_ratio
        self.projection_dim = projection_dim
        self.smoothing_sigma = smoothing_sigma
        self.seed = seed
        self.net = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1).to(self.device).eval()
        for p in self.net.parameters():
            p.requires_grad = False
        self.avg_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.memory_bank: Optional[torch.Tensor] = None

    def _feature_map(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            x = x.to(self.device)
            h = self.net.maxpool(self.net.relu(self.net.bn1(self.net.conv1(x))))
            l2 = self.net.layer2(self.net.layer1(h))
            l3 = self.net.layer3(l2)
            p2 = self.avg_pool(l2)
            p3 = F.interpolate(self.avg_pool(l3), size=p2.shape[2:], mode="bilinear", align_corners=False)
            return torch.cat([p2, p3], dim=1)

    def extract_patches(self, x: torch.Tensor) -> torch.Tensor:
        f = self._feature_map(x)
        b, c, h, w = f.shape
        return f.permute(0, 2, 3, 1).reshape(b * h * w, c)

    def fit(self, dataloader: DataLoader) -> None:
        patches = torch.cat([self.extract_patches(b[0] if isinstance(b, (list, tuple)) else b) for b in dataloader])
        n = patches.shape[0]
        m = max(1, int(n * self.coreset_sampling_ratio))
        if m >= n:
            self.memory_bank = patches
            return
        projected = random_projection(patches, self.projection_dim, seed=self.seed)
        start = int(np.random.RandomState(self.seed).randint(n))
        idx = greedy_kcenter(projected, m, start_idx=start)
        self.memory_bank = patches.index_select(0, idx)

    def predict(self, x: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        if self.memory_bank is None:
            raise RuntimeError("PatchCore model is not fitted. Call .fit() first.")
        b, _, h_in, w_in = x.shape
        f = self._feature_map(x)
        _, c, h, w = f.shape
        q = f.permute(0, 2, 3, 1).reshape(-1, c)
        with torch.no_grad():
            dists = torch.cat([torch.cdist(q[i:i + 4096], self.memory_bank).min(dim=1).values
                               for i in range(0, q.shape[0], 4096)])
            amap = F.interpolate(dists.view(b, 1, h, w), size=(h_in, w_in), mode="bilinear", align_corners=False)
        amaps = amap.squeeze(1).cpu().numpy()
        smoothed = np.stack([scipy.ndimage.gaussian_filter(a, sigma=self.smoothing_sigma) for a in amaps])
        return smoothed.reshape(b, -1).max(axis=1).astype(np.float64), smoothed
