import os
from typing import Tuple, Optional, List
import numpy as np
import scipy.ndimage
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.models as tvm

from .base import BaseAnomalyDetector


class PatchCore(BaseAnomalyDetector):
    """
    PatchCore anomaly detector:
    - Multi-scale intermediate feature extraction (layer2 + layer3 of ResNet-18)
    - Locally aware spatial neighborhood aggregation via AvgPool2d
    - Memory-efficient Minimax Greedy Coreset Selection (k-Center Greedy with random projection)
    - Nearest-neighbor anomaly scoring & spatial Gaussian smoothed heatmap generation
    """
    def __init__(
        self,
        backbone: str = "resnet18",
        coreset_sampling_ratio: float = 0.10,
        projection_dim: int = 128,
        device: Optional[str] = None,
        seed: int = 42
    ):
        super().__init__(device=device)
        self.backbone_name = backbone
        self.coreset_sampling_ratio = coreset_sampling_ratio
        self.projection_dim = projection_dim
        self.seed = seed

        # Build feature extractor backbone
        if backbone == "resnet18":
            weights = tvm.ResNet18_Weights.IMAGENET1K_V1
            net = tvm.resnet18(weights=weights)
        elif backbone == "resnet50":
            weights = tvm.ResNet50_Weights.IMAGENET1K_V2
            net = tvm.resnet50(weights=weights)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        self.net = net.to(self.device).eval()
        for param in self.net.parameters():
            param.requires_grad = False

        self.avg_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.memory_bank: Optional[torch.Tensor] = None  # [M, C_total]

    def _extract_multiscale_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extracts multi-scale patch features from layer2 and layer3.
        For 256x256 input:
          layer2 -> [B, 128, 32, 32]
          layer3 -> [B, 256, 16, 16] -> interpolated to [B, 256, 32, 32]
        Concatenated -> [B, 384, 32, 32] -> permuted & reshaped to [B * 1024, 384].
        """
        x = x.to(self.device)
        with torch.no_grad():
            x0 = self.net.conv1(x)
            x0 = self.net.bn1(x0)
            x0 = self.net.relu(x0)
            x0 = self.net.maxpool(x0)

            l1 = self.net.layer1(x0)
            l2 = self.net.layer2(l1)
            l3 = self.net.layer3(l2)

            # Locally aware pooling
            p2 = self.avg_pool(l2)
            p3 = self.avg_pool(l3)

            # Align spatial dimensions to layer2 resolution (32x32 for 256x256 inputs)
            p3_up = F.interpolate(p3, size=p2.shape[2:], mode="bilinear", align_corners=False)

            # Concatenate along channel dimension: [B, C2+C3, H, W]
            features = torch.cat([p2, p3_up], dim=1)
            B, C, H, W = features.shape

            # Permute to [B, H, W, C] and reshape to [B * H * W, C]
            patches = features.permute(0, 2, 3, 1).reshape(B * H * W, C)
            return patches

    def _greedy_coreset_subsampling(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Minimax Greedy Coreset Selection (k-Center Greedy).
        Reduces N patch embeddings to M = max(1, int(N * ratio)) representative patch vectors.
        """
        N, D = patches.shape
        M = max(1, int(N * self.coreset_sampling_ratio))

        if M >= N:
            return patches

        # Random projection for fast distance computation during coreset selection
        if D > self.projection_dim:
            g = torch.Generator(device=self.device)
            g.manual_seed(self.seed)
            proj_mat = torch.randn(D, self.projection_dim, device=self.device, generator=g)
            proj_mat, _ = torch.linalg.qr(proj_mat)  # Orthonormal projection
            patches_proj = torch.matmul(patches, proj_mat)
        else:
            patches_proj = patches

        # Initialize coreset with the point furthest from the dataset mean
        center = torch.mean(patches_proj, dim=0, keepdim=True)
        init_dists = torch.norm(patches_proj - center, dim=1)
        start_idx = int(torch.argmax(init_dists).item())

        selected_indices = [start_idx]
        min_distances = torch.norm(patches_proj - patches_proj[start_idx:start_idx+1], dim=1)

        # Iterative greedy furthest-point selection
        batch_chunk = 5000  # Avoid VRAM spikes on large matrix diffs
        for _ in range(1, M):
            next_idx = int(torch.argmax(min_distances).item())
            selected_indices.append(next_idx)

            # Update min distances to current coreset
            query_pt = patches_proj[next_idx:next_idx+1]
            for c_start in range(0, N, batch_chunk):
                c_end = min(c_start + batch_chunk, N)
                chunk_dists = torch.norm(patches_proj[c_start:c_end] - query_pt, dim=1)
                min_distances[c_start:c_end] = torch.minimum(min_distances[c_start:c_end], chunk_dists)

        return patches[selected_indices]

    def fit(self, dataloader: DataLoader) -> None:
        """
        Extracts all multi-scale patch features across nominal training images
        and applies greedy coreset reduction to construct the memory bank.
        """
        all_patches_list = []
        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                x = batch[0]
            else:
                x = batch
            patches = self._extract_multiscale_features(x)
            all_patches_list.append(patches)

        all_patches = torch.cat(all_patches_list, dim=0)
        self.memory_bank = self._greedy_coreset_subsampling(all_patches)

    def predict(self, x: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """
        Queries test batch [B, 3, 256, 256] against the coreset memory bank.
        Returns:
            image_scores: np.ndarray of shape [B]
            anomaly_maps: np.ndarray of shape [B, 256, 256]
        """
        if self.memory_bank is None:
            raise RuntimeError("PatchCore model is not fitted. Call .fit() first.")

        B, _, H_in, W_in = x.shape
        x = x.to(self.device)

        with torch.no_grad():
            x0 = self.net.conv1(x)
            x0 = self.net.bn1(x0)
            x0 = self.net.relu(x0)
            x0 = self.net.maxpool(x0)

            l1 = self.net.layer1(x0)
            l2 = self.net.layer2(l1)
            l3 = self.net.layer3(l2)

            p2 = self.avg_pool(l2)
            p3 = self.avg_pool(l3)
            p3_up = F.interpolate(p3, size=p2.shape[2:], mode="bilinear", align_corners=False)

            features = torch.cat([p2, p3_up], dim=1)  # [B, C, 32, 32]
            B_feat, C_feat, H_feat, W_feat = features.shape

            # Query patches: [B * H_feat * W_feat, C_feat]
            query_patches = features.permute(0, 2, 3, 1).reshape(B_feat * H_feat * W_feat, C_feat)

            # Compute min Euclidean distance from each query patch to the memory bank
            # Chunking query patches to avoid OOM on large batches / test sets
            chunk_size = 2048
            min_dists_list = []
            for i in range(0, query_patches.shape[0], chunk_size):
                q_chunk = query_patches[i:i + chunk_size]
                # Pairwise distance matrix [chunk_size, M]
                dists = torch.cdist(q_chunk, self.memory_bank, p=2.0)
                min_dists, _ = torch.min(dists, dim=1)
                min_dists_list.append(min_dists)

            min_dists_all = torch.cat(min_dists_list, dim=0)

            # Reshape patch distances to [B, 1, 32, 32]
            patch_scores = min_dists_all.reshape(B, 1, H_feat, W_feat)

            # Bilinearly upsample to original resolution (256, 256)
            amaps_tensor = F.interpolate(patch_scores, size=(H_in, W_in), mode="bilinear", align_corners=False)
            amaps_np = amaps_tensor.squeeze(1).cpu().numpy()  # [B, 256, 256]

        # Apply 2D spatial Gaussian filter to smooth heatmap & compute image scores
        smoothed_amaps = np.zeros_like(amaps_np)
        image_scores = np.zeros(B, dtype=float)

        for b in range(B):
            smoothed_amaps[b] = scipy.ndimage.gaussian_filter(amaps_np[b], sigma=4)
            image_scores[b] = float(np.max(smoothed_amaps[b]))

        return image_scores, smoothed_amaps

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "memory_bank": self.memory_bank.cpu() if self.memory_bank is not None else None,
            "backbone": self.backbone_name,
            "coreset_sampling_ratio": self.coreset_sampling_ratio,
            "projection_dim": self.projection_dim,
            "seed": self.seed
        }, path)

    def load(self, path: str) -> None:
        state = torch.load(path, map_location=self.device)
        mb = state.get("memory_bank")
        self.memory_bank = mb.to(self.device) if mb is not None else None
        self.backbone_name = state.get("backbone", "resnet18")
        self.coreset_sampling_ratio = state.get("coreset_sampling_ratio", 0.10)
        self.projection_dim = state.get("projection_dim", 128)
        self.seed = state.get("seed", 42)
