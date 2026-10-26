import os
import sys

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import src.mvtec as mvtec  # noqa: E402

TOY_MANIFEST = {"train": {"good": 12}, "test": {"good": 6, "scratch": 4, "dent": 4}}


def _texture(rng, size=64):
    base = rng.randint(90, 110, (size, size, 3)).astype(np.uint8)
    base[::8, :, :] = 160
    return base


def build_toy_category(root: str, name: str = "toy", seed: int = 0) -> str:
    """Writes a tiny MVTec-style category (procedural images, for unit tests only)."""
    rng = np.random.RandomState(seed)
    cat = os.path.join(root, name)
    for split, types in TOY_MANIFEST.items():
        for dtype, n in types.items():
            d = os.path.join(cat, split, dtype)
            os.makedirs(d, exist_ok=True)
            for i in range(n):
                img = _texture(rng)
                if dtype != "good":
                    mask = np.zeros(img.shape[:2], dtype=np.uint8)
                    y, x = rng.randint(10, 40, 2)
                    mask[y:y + 12, x:x + 12] = 255
                    img[mask > 0] = [230, 30, 30] if dtype == "scratch" else [20, 20, 20]
                    gd = os.path.join(cat, "ground_truth", dtype)
                    os.makedirs(gd, exist_ok=True)
                    Image.fromarray(mask).save(os.path.join(gd, f"{i:03d}_mask.png"))
                Image.fromarray(img).save(os.path.join(d, f"{i:03d}.png"))
    return cat


@pytest.fixture
def toy_root(tmp_path, monkeypatch):
    """A validated toy category registered in the manifest for the duration of the test."""
    root = str(tmp_path / "data")
    build_toy_category(root)
    monkeypatch.setitem(mvtec.MVTEC_MANIFEST, "toy", TOY_MANIFEST)
    return root
