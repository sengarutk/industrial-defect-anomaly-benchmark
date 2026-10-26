import numpy as np
import pytest
import torch

from src.methods.base import BaseAnomalyDetector
from src.robustness import CORRUPTION_TYPES, CachedTestSet, CorruptedTestDataset, apply_corruption, run_stress_test
from src.robustness.evaluator import evaluate_maps, predict_dataset


def pattern(size=64):
    y, x = np.ogrid[:size, :size]
    g = ((np.sin(x / 3.0) + np.cos(y / 3.0) + 2.0) * 60).astype(np.uint8)
    return np.stack([g, g, g], axis=-1)


@pytest.mark.parametrize("ctype", CORRUPTION_TYPES)
def test_corruptions_shape_and_monotone_severity(ctype):
    img = pattern()
    diffs = []
    for sev in (1, 2, 3):
        out = apply_corruption(img, ctype, sev, rng=np.random.RandomState(0))
        assert out.shape == img.shape and out.dtype == np.uint8
        diffs.append(np.abs(out.astype(float) - img).mean())
    assert diffs[0] <= diffs[1] <= diffs[2]


def test_noise_is_reproducible_and_inputs_validated():
    img = pattern()
    a = apply_corruption(img, "gaussian_noise", 2, rng=np.random.RandomState(5))
    b = apply_corruption(img, "gaussian_noise", 2, rng=np.random.RandomState(5))
    assert np.array_equal(a, b)
    with pytest.raises(ValueError):
        apply_corruption(img, "gaussian_blur", 4)
    with pytest.raises(ValueError):
        apply_corruption(img, "fog", 1)
    with pytest.raises(TypeError):
        apply_corruption(img.astype(np.float32), "gaussian_blur", 1)


class MeanBrightness(BaseAnomalyDetector):
    def fit(self, dataloader):
        pass

    def predict(self, x):
        maps = x.mean(dim=1).numpy()
        return maps.reshape(len(maps), -1).max(axis=1), maps


def test_stress_test_on_toy_category(toy_root):
    cache = CachedTestSet(toy_root, "toy", img_size=64)
    assert len(cache) == 14 and cache.masks.sum() > 0
    clean_ds = CorruptedTestDataset(cache, None)
    x, y, mask, idx = clean_ds[0]
    assert x.shape == (3, 64, 64) and mask.shape == (1, 64, 64)
    model = MeanBrightness(device="cpu")
    s, m = predict_dataset(model, clean_ds, batch_size=4)
    clean = evaluate_maps(cache.labels, s, cache.masks, m)
    res = run_stress_test(model, cache, clean, ["gaussian_blur", "gaussian_noise"], [1, 3], batch_size=4)
    assert len(res["conditions"]) == 4
    assert res["mrd_image_auroc"] >= 0 and res["mrd_image_auroc"] >= res["mpc_image_auroc"]
    ds1 = CorruptedTestDataset(cache, "gaussian_noise", 2, noise_seed=1)
    assert torch.equal(ds1[3][0], CorruptedTestDataset(cache, "gaussian_noise", 2, noise_seed=1)[3][0])
