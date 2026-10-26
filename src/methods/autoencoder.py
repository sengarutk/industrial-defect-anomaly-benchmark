from typing import Optional, Tuple

import numpy as np
import scipy.ndimage
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .base import BaseAnomalyDetector


class _ConvAutoencoderArch(nn.Module):
    """(3, 256, 256) -> (128, 16, 16) -> (3, 256, 256); four stride-2 stages each way."""

    def __init__(self):
        super().__init__()

        def down(cin, cout):
            return [nn.Conv2d(cin, cout, 4, 2, 1), nn.BatchNorm2d(cout), nn.LeakyReLU(0.2, inplace=True)]

        def up(cin, cout):
            return [nn.ConvTranspose2d(cin, cout, 4, 2, 1), nn.BatchNorm2d(cout), nn.LeakyReLU(0.2, inplace=True)]

        self.encoder = nn.Sequential(*down(3, 32), *down(32, 64), *down(64, 128), *down(128, 128))
        self.decoder = nn.Sequential(*up(128, 128), *up(128, 64), *up(64, 32), nn.ConvTranspose2d(32, 3, 4, 2, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


class ConvAutoencoder(BaseAnomalyDetector):
    """Convolutional autoencoder baseline trained with MSE on nominal images.

    Pixel map: channel-mean squared reconstruction error, Gaussian smoothed.
    Image score: max of the map (same aggregation as PatchCore and PaDiM).
    Weight initialisation and batch order come from the global torch RNG,
    which the caller seeds via seed_everything().
    """

    def __init__(
        self,
        epochs: int = 40,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        batch_size: int = 16,
        smoothing_sigma: float = 4.0,
        device: Optional[str] = None,
    ):
        super().__init__(device=device)
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.smoothing_sigma = smoothing_sigma
        self.model = _ConvAutoencoderArch().to(self.device)

    def fit(self, dataloader: DataLoader) -> None:
        data = torch.cat([b[0] if isinstance(b, (list, tuple)) else b for b in dataloader]).to(self.device)
        n = data.shape[0]
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = nn.MSELoss()
        self.model.train()
        for _ in range(self.epochs):
            perm = torch.randperm(n, device=self.device)
            for i in range(0, n, self.batch_size):
                xb = data[perm[i:i + self.batch_size]]
                if xb.shape[0] < 2:  # BatchNorm needs more than one sample per batch
                    continue
                opt.zero_grad()
                loss = loss_fn(self.model(xb), xb)
                loss.backward()
                opt.step()
        self.model.eval()

    def predict(self, x: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        self.model.eval()
        with torch.no_grad():
            x = x.to(self.device)
            residual = ((x - self.model(x)) ** 2).mean(dim=1).cpu().numpy()
        smoothed = np.stack([scipy.ndimage.gaussian_filter(r, sigma=self.smoothing_sigma) for r in residual])
        return smoothed.reshape(len(smoothed), -1).max(axis=1).astype(np.float64), smoothed
