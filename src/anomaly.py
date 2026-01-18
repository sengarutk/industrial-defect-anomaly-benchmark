import time
from typing import Dict, Any

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


@torch.no_grad()
def extract_patch_embeddings(fmap: torch.Tensor) -> torch.Tensor:
    """
    fmap: [B, C, H, W]
    -> patch embeddings: [B*H*W, C]
    """
    B, C, H, W = fmap.shape
    patches = fmap.permute(0, 2, 3, 1).contiguous().view(B * H * W, C)
    return patches


def _apply_pca_if_needed(X: np.ndarray, pca_dim: int):
    pca = None
    if pca_dim is not None and pca_dim > 0 and pca_dim < X.shape[1]:
        pca = PCA(n_components=pca_dim, random_state=0)
        X = pca.fit_transform(X)
    return X, pca


def build_memorybank_patch(encoder, loader, device: str, pca_dim: int = 128) -> Dict[str, Any]:
    """
    Build memorybank from patch embeddings of normal train images.
    """
    encoder.eval()
    all_patches = []

    t0 = time.time()
    for x in loader:
        x = x.to(device)
        _, fmap = encoder(x)
        patches = extract_patch_embeddings(fmap)  # [N, C]
        all_patches.append(patches.cpu().numpy())

    X = np.concatenate(all_patches, axis=0)
    build_time = time.time() - t0

    X, pca = _apply_pca_if_needed(X, pca_dim)

    return {
        "X": X.astype(np.float32),
        "pca": pca,
        "build_time_s": build_time,
        "mode": "patch",
    }


@torch.no_grad()
def build_memorybank_global(encoder, loader, device: str, pca_dim: int = 128) -> Dict[str, Any]:
    """
    Build memorybank from pooled/global embeddings of normal train images.
    This matches SimCLR training objective (global embedding).
    """
    encoder.eval()
    all_embs = []

    t0 = time.time()
    for x in loader:
        x = x.to(device)
        emb, _ = encoder(x)  # [B, D]
        all_embs.append(emb.cpu().numpy())

    X = np.concatenate(all_embs, axis=0)
    build_time = time.time() - t0

    X, pca = _apply_pca_if_needed(X, pca_dim)

    return {
        "X": X.astype(np.float32),
        "pca": pca,
        "build_time_s": build_time,
        "mode": "global",
    }


def fit_knn(memorybank: Dict[str, Any], knn_k: int = 5) -> NearestNeighbors:
    X = memorybank["X"]
    knn = NearestNeighbors(n_neighbors=knn_k, algorithm="auto", metric="euclidean")
    knn.fit(X)
    return knn


@torch.no_grad()
def score_image_patchwise(encoder, x: torch.Tensor, knn: NearestNeighbors, memorybank: Dict[str, Any], device: str):
    """
    Returns:
      image_score: float
      patch_scores: (H, W) heatmap scores
    """
    encoder.eval()
    x = x.to(device)

    _, fmap = encoder(x)  # fmap [B,C,H,W]
    patches = extract_patch_embeddings(fmap).cpu().numpy()

    if memorybank["pca"] is not None:
        patches = memorybank["pca"].transform(patches)

    dists, _ = knn.kneighbors(patches)  # [N, k]
    patch_score = dists.mean(axis=1)    # [N]

    B, C, H, W = fmap.shape
    heat = patch_score.reshape(H, W)
    image_score = float(patch_score.max())  # image anomaly = max patch anomaly

    return image_score, heat


@torch.no_grad()
def score_image_global(encoder, x: torch.Tensor, knn: NearestNeighbors, memorybank: Dict[str, Any], device: str):
    """
    Global embedding scoring (no heatmap).
    Returns:
      score: float
      heat: None
    """
    encoder.eval()
    x = x.to(device)

    emb, _ = encoder(x)  # [1, D]
    v = emb.cpu().numpy()

    if memorybank["pca"] is not None:
        v = memorybank["pca"].transform(v)

    dists, _ = knn.kneighbors(v)  # [1, k]
    score = float(dists.mean())

    return score, None
