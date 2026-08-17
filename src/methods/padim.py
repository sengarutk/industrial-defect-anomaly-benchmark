import os
from typing import Tuple, Optional, List
import numpy as np
import scipy.ndimage
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.models as tvm

from .base import BaseAnomalyDetector


class PaDiM(BaseAnomalyDetector):
    """
    PaDiM: A Patch Distribution Modeling Framework for Anomaly Detection and Localization.
    - Extracts multi-scale features from layer1, layer2, and layer3
    - Bilinearly aligns all features to layer1 spatial resolution (64x64 for 256x256 inputs)
    - Subsamples a fixed subset of d_dim channels (default 100)
    - Estimates multivariate Gaussian distribution (mean and inverse covariance with shrinkage) per patch position
    - Computes Mahalanobis distance at test time for precise pixel localization
    """
    def __init__(
        self,
        backbone: str = "resnet18",
        d_dim: int = 100,
        device: Optional[str] = None,
        seed: int = 42
    ):
        super().__init__(device=device)
        self.backbone_name = backbone
        self.d_dim = d_dim
        self.seed = seed

        if backbone == "resnet18":
            weights = tvm.ResNet18_Weights.IMAGENET1K_V1
            net = tvm.resnet18(weights=weights)
            total_channels = 64 + 128 + 256  # layer1: 64, layer2: 128, layer3: 256 = 448
        elif backbone == "resnet50":
            weights = tvm.ResNet50_Weights.IMAGENET1K_V2
            net = tvm.resnet50(weights=weights)
            total_channels = 256 + 512 + 1024  # 1792
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        self.net = net.to(self.device).eval()
        for param in self.net.parameters():
            param.requires_grad = False

        # Randomly select a fixed subset of channel indices
        g = torch.Generator()
        g.manual_seed(seed)
        actual_d = min(self.d_dim, total_channels)
        self.channel_indices = torch.randperm(total_channels, generator=g)[:actual_d]

        # Learned statistical parameters per spatial position (H_p=64, W_p=64)
        self.mean: Optional[torch.Tensor] = None          # [d, 64, 64]
        self.inv_cov: Optional[torch.Tensor] = None       # [64, 64, d, d]

    def _extract_embedding_map(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extracts layer1, layer2, layer3 features, upsamples to layer1 resolution (64x64),
        concatenates, and selects d_dim channels.
        Returns: [B, d_dim, 64, 64]
        """
        x = x.to(self.device)
        with torch.no_grad():
            x0 = self.net.conv1(x)
            x0 = self.net.bn1(x0)
            x0 = self.net.relu(x0)
            x0 = self.net.maxpool(x0)

            l1 = self.net.layer1(x0)  # [B, 64, 64, 64]
            l2 = self.net.layer2(l1)  # [B, 128, 32, 32]
            l3 = self.net.layer3(l2)  # [B, 256, 16, 16]

            # Upsample l2 and l3 to layer1 spatial resolution
            l2_up = F.interpolate(l2, size=l1.shape[2:], mode="bilinear", align_corners=False)
            l3_up = F.interpolate(l3, size=l1.shape[2:], mode="bilinear", align_corners=False)

            # Concat along channel dimension
            feat = torch.cat([l1, l2_up, l3_up], dim=1)  # [B, 448, 64, 64]

            # Select deterministic subset of channels
            feat_sub = feat[:, self.channel_indices.to(self.device), :, :]
            return feat_sub

    def fit(self, dataloader: DataLoader) -> None:
        """
        Estimates spatial Gaussian distributions (mean vector and inverse covariance matrix)
        for every position (i, j) in the [64, 64] grid across normal training images.
        """
        feats_list = []
        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                x = batch[0]
            else:
                x = batch
            feat = self._extract_embedding_map(x)
            feats_list.append(feat)

        # X: [N, d, H, W]
        X = torch.cat(feats_list, dim=0)
        N, d, H, W = X.shape

        # Compute sample mean: [d, H, W]
        mean = torch.mean(X, dim=0)

        # Compute covariance matrix per spatial location with shrinkage regularization
        # Reshape X centered: [N, d, H*W] -> permute to [H*W, N, d]
        X_centered = (X - mean.unsqueeze(0)).reshape(N, d, H * W).permute(2, 0, 1)  # [P, N, d] where P = H*W

        # Sample covariance: [P, d, d]
        cov = torch.bmm(X_centered.transpose(1, 2), X_centered) / max(N - 1, 1)

        # Shrinkage regularization: Sigma + 0.01 * I
        eye = torch.eye(d, device=self.device).unsqueeze(0)  # [1, d, d]
        cov_reg = cov + 0.01 * eye

        # Precompute pseudo-inverse of covariance: [P, d, d] -> [H, W, d, d]
        inv_cov = torch.linalg.pinv(cov_reg).reshape(H, W, d, d)

        self.mean = mean
        self.inv_cov = inv_cov

    def predict(self, x: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes Mahalanobis distance heatmap at each spatial position [64, 64],
        upsamples to [256, 256], applies Gaussian filter, and takes max as image score.
        """
        if self.mean is None or self.inv_cov is None:
            raise RuntimeError("PaDiM model is not fitted. Call .fit() first.")

        B, _, H_in, W_in = x.shape
        feat = self._extract_embedding_map(x)  # [B, d, H_p, W_p]
        _, d, H_p, W_p = feat.shape

        # delta: [B, d, H_p, W_p] -> permute to [B, H_p * W_p, d]
        delta = (feat - self.mean.unsqueeze(0)).reshape(B, d, H_p * W_p).permute(0, 2, 1)  # [B, P, d]
        inv_cov_flat = self.inv_cov.reshape(H_p * W_p, d, d)  # [P, d, d]

        # Mahalanobis distance squared: delta * inv_cov * delta^T
        # Einstein summation: b: batch, p: pixel pos, i,j: channels
        m_dist_sq = torch.einsum("bpi,pij,bpj->bp", delta, inv_cov_flat, delta)
        m_dist = torch.sqrt(torch.clamp(m_dist_sq, min=1e-12))  # [B, P]

        # Reshape to [B, 1, H_p, W_p]
        dist_map = m_dist.reshape(B, 1, H_p, W_p)

        # Upsample to [B, 256, 256]
        amaps_tensor = F.interpolate(dist_map, size=(H_in, W_in), mode="bilinear", align_corners=False)
        amaps_np = amaps_tensor.squeeze(1).detach().cpu().numpy()

        smoothed_amaps = np.zeros_like(amaps_np)
        image_scores = np.zeros(B, dtype=float)

        for b in range(B):
            smoothed_amaps[b] = scipy.ndimage.gaussian_filter(amaps_np[b], sigma=4)
            image_scores[b] = float(np.max(smoothed_amaps[b]))

        return image_scores, smoothed_amaps

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "mean": self.mean.cpu() if self.mean is not None else None,
            "inv_cov": self.inv_cov.cpu() if self.inv_cov is not None else None,
            "channel_indices": self.channel_indices.cpu(),
            "backbone": self.backbone_name,
            "d_dim": self.d_dim,
            "seed": self.seed
        }, path)

    def load(self, path: str) -> None:
        state = torch.load(path, map_location=self.device)
        m = state.get("mean")
        ic = state.get("inv_cov")
        self.mean = m.to(self.device) if m is not None else None
        self.inv_cov = ic.to(self.device) if ic is not None else None
        self.channel_indices = state.get("channel_indices")
        self.backbone_name = state.get("backbone", "resnet18")
        self.d_dim = state.get("d_dim", 100)
        self.seed = state.get("seed", 42)
