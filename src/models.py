import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm


class Encoder(nn.Module):
    """
    ResNet backbone with feature output for SSL and anomaly scoring.
    Returns embedding vector + patch feature map.
    """
    def __init__(self, backbone: str = "resnet18", out_dim: int = 512):
        super().__init__()

        if backbone == "resnet18":
            net = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1)
            feat_dim = 512
        elif backbone == "resnet50":
            net = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2)
            feat_dim = 2048
        else:
            raise ValueError("backbone must be resnet18 or resnet50")

        # remove classifier
        self.backbone = nn.Sequential(*list(net.children())[:-2])  # until last conv feature map
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Linear(feat_dim, out_dim)
        self.out_dim = out_dim
        self.feat_dim = feat_dim

    def forward(self, x):
        fmap = self.backbone(x)                 # [B, C, H, W]
        pooled = self.pool(fmap).flatten(1)     # [B, C]
        emb = self.proj(pooled)                 # [B, out_dim]
        return emb, fmap


class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, proj_dim: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, proj_dim),
        )

    def forward(self, x):
        return self.mlp(x)
