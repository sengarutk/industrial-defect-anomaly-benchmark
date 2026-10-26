from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.models as tvm
from scipy.ndimage import gaussian_filter
from torch.utils.data import DataLoader

from .base import BaseAnomalyDetector


class PaDiM(BaseAnomalyDetector):
    """PaDiM-style detector (Defard et al., ICPR 2021) with documented deviations.

    ResNet-18 layer1-3 features (448-d) upsampled to layer1 resolution; a random
    subset of `reduced_dim` channels drawn from `seed`; per-location Gaussian with
    covariance Sigma + reg_lambda * I; Mahalanobis distance map, Gaussian
    smoothing, image score = max of the map.
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        reduced_dim: int = 100,
        reg_lambda: float = 0.01,
        smoothing_sigma: float = 4.0,
        device: Optional[str] = None,
        seed: int = 0,
    ):
        super().__init__(device=device)
        if backbone != "resnet18":
            raise ValueError(f"Unsupported backbone: {backbone}")
        self.reduced_dim = reduced_dim
        self.reg_lambda = reg_lambda
        self.smoothing_sigma = smoothing_sigma
        self.seed = seed
        self.net = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1).to(self.device).eval()
        for p in self.net.parameters():
            p.requires_grad = False
        self.mean: Optional[torch.Tensor] = None      # (d, H, W)
        self.inv_cov: Optional[torch.Tensor] = None   # (H, W, d, d)
        self.idx_selected: Optional[torch.Tensor] = None

    def _embed(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            x = x.to(self.device)
            h = self.net.maxpool(self.net.relu(self.net.bn1(self.net.conv1(x))))
            f1 = self.net.layer1(h)
            f2 = self.net.layer2(f1)
            f3 = self.net.layer3(f2)
            size = f1.shape[-2:]
            return torch.cat([f1,
                              F.interpolate(f2, size=size, mode="bilinear", align_corners=False),
                              F.interpolate(f3, size=size, mode="bilinear", align_corners=False)], dim=1)

    def fit(self, dataloader: DataLoader) -> None:
        feats = []
        for batch in dataloader:
            e = self._embed(batch[0] if isinstance(batch, (list, tuple)) else batch)
            if self.idx_selected is None:
                c = e.shape[1]
                g = torch.Generator(device="cpu").manual_seed(self.seed)
                sel = torch.randperm(c, generator=g)[:self.reduced_dim] if self.reduced_dim < c else torch.arange(c)
                self.idx_selected = sel.sort().values.to(self.device)
            feats.append(e.index_select(1, self.idx_selected))
        emb = torch.cat(feats)                      # (B, d, H, W)
        b, d, h, w = emb.shape
        if b < 2:
            raise ValueError("PaDiM needs at least two training images to estimate covariances.")
        flat = emb.permute(2, 3, 0, 1).reshape(h * w, b, d)   # (L, B, d)
        mean = flat.mean(dim=1)                                 # (L, d)
        centered = flat - mean.unsqueeze(1)
        cov = centered.transpose(1, 2) @ centered / (b - 1)     # (L, d, d)
        cov = cov + self.reg_lambda * torch.eye(d, device=self.device).unsqueeze(0)
        self.inv_cov = torch.linalg.inv(cov).reshape(h, w, d, d)
        self.mean = mean.reshape(h, w, d).permute(2, 0, 1).contiguous()

    def predict(self, x: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        if self.mean is None or self.inv_cov is None:
            raise RuntimeError("PaDiM model is not fitted yet.")
        with torch.no_grad():
            e = self._embed(x).index_select(1, self.idx_selected)
            diff = (e - self.mean.unsqueeze(0)).permute(0, 2, 3, 1)
            dist = torch.einsum("bhwd,hwde,bhwe->bhw", diff, self.inv_cov, diff).clamp_(min=0.0).sqrt_()
            dist = F.interpolate(dist.unsqueeze(1), size=x.shape[-2:], mode="bilinear", align_corners=False)
        maps = dist.squeeze(1).cpu().numpy()
        smoothed = np.stack([gaussian_filter(m, sigma=self.smoothing_sigma) for m in maps])
        return smoothed.reshape(len(smoothed), -1).max(axis=1).astype(np.float64), smoothed
