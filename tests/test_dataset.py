import os
import shutil

import numpy as np
import pytest
from PIL import Image

from src.mvtec import (MVTEC_MANIFEST, DatasetIntegrityError, MVTecTest, MVTecTrainNormal, list_test_samples,
                       split_train_calibration, validate_category)


def test_manifest_matches_published_totals():
    totals = {c: (m["train"]["good"], sum(m["test"].values())) for c, m in MVTEC_MANIFEST.items()}
    assert totals["bottle"] == (209, 83)
    assert totals["grid"] == (264, 78)
    assert totals["leather"] == (245, 124)
    assert sum(t for t, _ in totals.values()) == 3629
    assert sum(t for _, t in totals.values()) == 1725


def test_valid_toy_category_loads(toy_root):
    counts = validate_category(toy_root, "toy")
    assert counts["train"]["good"] == 12
    ds = MVTecTest(toy_root, "toy", img_size=64)
    assert len(ds) == 14
    x, y, mask, meta = ds[ds.samples.index(next(s for s in ds.samples if s["label"] == 1))]
    assert x.shape == (3, 64, 64) and mask.shape == (1, 64, 64) and y == 1 and mask.sum() > 0
    assert len(MVTecTrainNormal(toy_root, "toy", img_size=64)) == 12


def test_extra_defect_folder_is_rejected(toy_root):
    """Regression: a stray mock 'defect' folder used to be silently evaluated as a defect type."""
    extra = os.path.join(toy_root, "toy", "test", "defect")
    os.makedirs(extra)
    Image.fromarray(np.zeros((8, 8, 3), np.uint8)).save(os.path.join(extra, "000.png"))
    with pytest.raises(DatasetIntegrityError, match="extra folders"):
        validate_category(toy_root, "toy")
    with pytest.raises(DatasetIntegrityError):
        list_test_samples(toy_root, "toy")


def test_wrong_counts_and_missing_masks_are_rejected(toy_root):
    os.remove(os.path.join(toy_root, "toy", "train", "good", "000.png"))
    with pytest.raises(DatasetIntegrityError, match="expected 12"):
        validate_category(toy_root, "toy")


def test_missing_mask_is_rejected(toy_root):
    os.remove(os.path.join(toy_root, "toy", "ground_truth", "dent", "000_mask.png"))
    with pytest.raises(DatasetIntegrityError, match="masks missing"):
        validate_category(toy_root, "toy")


def test_unknown_or_absent_category(tmp_path):
    with pytest.raises(DatasetIntegrityError):
        validate_category(str(tmp_path), "not_a_category")
    with pytest.raises(DatasetIntegrityError, match="download_dataset.py"):
        validate_category(str(tmp_path), "bottle")


def test_calibration_split_is_disjoint_and_seeded():
    paths = [f"img_{i:03d}.png" for i in range(50)]
    fit, cal = split_train_calibration(paths, 0.2, seed=1)
    assert len(cal) == 10 and len(fit) == 40 and not set(fit) & set(cal)
    assert split_train_calibration(paths, 0.2, seed=1) == (fit, cal)
    assert split_train_calibration(paths, 0.2, seed=2)[1] != cal
    shutil.rmtree  # keep import used for readability of helpers above
