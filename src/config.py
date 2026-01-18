from dataclasses import dataclass
from typing import List


@dataclass
class TrainConfig:
    seed: int = 42
    device: str = "cuda"
    num_workers: int = 4

    categories: List[str] = None
    mvtec_root: str = "data/mvtec_ad"

    # backbone
    backbone: str = "resnet18"  # resnet18 | resnet50

    # simclr
    batch_size: int = 128
    epochs: int = 20
    lr: float = 3e-4
    weight_decay: float = 1e-4
    temperature: float = 0.5
    proj_dim: int = 128

    # anomaly scoring
    knn_k: int = 5
    pca_dim: int = 128  # reduce memorybank for speed (systems angle)

    # runtime
    save_heatmaps: bool = True

    def __post_init__(self):
        if self.categories is None:
            # choose a small set = fast + meaningful
            self.categories = ["bottle", "cable", "hazelnut", "metal_nut"]
