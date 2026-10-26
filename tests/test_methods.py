import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.methods import METHOD_NAMES, BaseAnomalyDetector, ConvAutoencoder, PaDiM, PatchCore, build_model
from src.utils import seed_everything


def loader(n=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    return DataLoader(TensorDataset(torch.randn(n, 3, 64, 64, generator=g)), batch_size=2)


def test_interface_and_factory():
    for cls in (PatchCore, PaDiM, ConvAutoencoder):
        assert issubclass(cls, BaseAnomalyDetector)
    assert set(METHOD_NAMES) == {"patchcore", "padim", "autoencoder"}
    with pytest.raises(ValueError):
        build_model("unknown", {}, seed=0, device="cpu")


@pytest.mark.parametrize("method,params", [("patchcore", {}), ("padim", {"reduced_dim": 20}),
                                           ("autoencoder", {"epochs": 1})])
def test_fit_predict_shapes(method, params):
    seed_everything(0)
    model = build_model(method, params, seed=0, device="cpu")
    model.fit(loader())
    scores, maps = model.predict(torch.randn(2, 3, 64, 64))
    assert scores.shape == (2,) and maps.shape == (2, 64, 64)
    assert np.isfinite(scores).all() and np.isfinite(maps).all()
    assert np.allclose(scores, maps.reshape(2, -1).max(axis=1))


def test_patchcore_seed_changes_coreset_and_is_reproducible():
    """Regression: the run seed used to never reach PatchCore, so all 'seeds' were identical."""
    banks = []
    for seed in (1, 1, 2):
        m = PatchCore(device="cpu", seed=seed, coreset_sampling_ratio=0.1)
        m.fit(loader())
        banks.append(m.memory_bank)
    assert torch.equal(banks[0], banks[1])
    assert not torch.equal(banks[0], banks[2])
    assert banks[0].shape == (int(4 * 8 * 8 * 0.1), 384)  # 64 px input -> 8x8 layer2 patches


def test_padim_seed_changes_channel_subset():
    a, b = PaDiM(device="cpu", seed=1, reduced_dim=20), PaDiM(device="cpu", seed=2, reduced_dim=20)
    a.fit(loader())
    b.fit(loader())
    assert not torch.equal(a.idx_selected, b.idx_selected)
    assert a.mean.shape == (20, 16, 16) and a.inv_cov.shape == (16, 16, 20, 20)


def test_unfitted_models_raise():
    with pytest.raises(RuntimeError):
        PatchCore(device="cpu").predict(torch.randn(1, 3, 64, 64))
    with pytest.raises(RuntimeError):
        PaDiM(device="cpu").predict(torch.randn(1, 3, 64, 64))
