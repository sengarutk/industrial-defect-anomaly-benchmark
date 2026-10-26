"""MVTec AD data access with strict integrity checks.

Every loader validates the category against the official MVTec AD manifest
(image counts per split and defect type) before reading anything, so synthetic,
partial or contaminated folders are rejected instead of silently evaluated.
"""
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Official MVTec AD image counts (Bergmann et al., CVPR 2019 / IJCV 2021).
# test counts are listed per defect type; "good" is the nominal test split.
MVTEC_MANIFEST: Dict[str, Dict[str, Dict[str, int]]] = {
    "bottle": {"train": {"good": 209}, "test": {"good": 20, "broken_large": 20, "broken_small": 22, "contamination": 21}},
    "cable": {"train": {"good": 224}, "test": {"good": 58, "bent_wire": 13, "cable_swap": 12, "combined": 11,
                                               "cut_inner_insulation": 14, "cut_outer_insulation": 10,
                                               "missing_cable": 12, "missing_wire": 10, "poke_insulation": 10}},
    "capsule": {"train": {"good": 219}, "test": {"good": 23, "crack": 23, "faulty_imprint": 22, "poke": 21,
                                                 "scratch": 23, "squeeze": 20}},
    "carpet": {"train": {"good": 280}, "test": {"good": 28, "color": 19, "cut": 17, "hole": 17,
                                                "metal_contamination": 17, "thread": 19}},
    "grid": {"train": {"good": 264}, "test": {"good": 21, "bent": 12, "broken": 12, "glue": 11,
                                              "metal_contamination": 11, "thread": 11}},
    "hazelnut": {"train": {"good": 391}, "test": {"good": 40, "crack": 18, "cut": 17, "hole": 18, "print": 17}},
    "leather": {"train": {"good": 245}, "test": {"good": 32, "color": 19, "cut": 19, "fold": 17, "glue": 19, "poke": 18}},
    "metal_nut": {"train": {"good": 220}, "test": {"good": 22, "bent": 25, "color": 22, "flip": 23, "scratch": 23}},
    "pill": {"train": {"good": 267}, "test": {"good": 26, "color": 25, "combined": 17, "contamination": 21, "crack": 26,
                                              "faulty_imprint": 19, "pill_type": 9, "scratch": 24}},
    "screw": {"train": {"good": 320}, "test": {"good": 41, "manipulated_front": 24, "scratch_head": 24,
                                               "scratch_neck": 25, "thread_side": 23, "thread_top": 23}},
    "tile": {"train": {"good": 230}, "test": {"good": 33, "crack": 17, "glue_strip": 18, "gray_stroke": 16, "oil": 18,
                                              "rough": 15}},
    "toothbrush": {"train": {"good": 60}, "test": {"good": 12, "defective": 30}},
    "transistor": {"train": {"good": 213}, "test": {"good": 60, "bent_lead": 10, "cut_lead": 10, "damaged_case": 10,
                                                    "misplaced": 10}},
    "wood": {"train": {"good": 247}, "test": {"good": 19, "color": 8, "combined": 11, "hole": 10, "liquid": 10,
                                              "scratch": 21}},
    "zipper": {"train": {"good": 240}, "test": {"good": 32, "broken_teeth": 19, "combined": 16, "fabric_border": 17,
                                                "fabric_interior": 16, "rough": 17, "split_teeth": 18,
                                                "squeezed_teeth": 16}},
}

IMAGE_EXTENSIONS = (".png",)


class DatasetIntegrityError(RuntimeError):
    """Raised when a category on disk does not match the official MVTec AD manifest."""


def _list_images(directory: str) -> List[str]:
    if not os.path.isdir(directory):
        return []
    return sorted(os.path.join(directory, f) for f in os.listdir(directory)
                  if f.lower().endswith(IMAGE_EXTENSIONS) and not f.startswith("."))


def _visible_subdirs(directory: str) -> List[str]:
    if not os.path.isdir(directory):
        return []
    return sorted(d for d in os.listdir(directory)
                  if not d.startswith(".") and os.path.isdir(os.path.join(directory, d)))


def validate_category(root: str, category: str) -> Dict[str, Dict[str, int]]:
    """Checks folders, image counts and mask coverage against the official manifest.

    Returns the observed counts. Raises DatasetIntegrityError on any mismatch
    (missing or extra defect folders, wrong counts, missing masks).
    """
    if category not in MVTEC_MANIFEST:
        raise DatasetIntegrityError(f"Unknown MVTec AD category '{category}'.")
    expected = MVTEC_MANIFEST[category]
    cat_dir = os.path.join(root, category)
    if not os.path.isdir(cat_dir):
        raise DatasetIntegrityError(
            f"Category directory not found: '{cat_dir}'. "
            f"Run: python scripts/download_dataset.py --categories {category}")

    problems: List[str] = []
    observed: Dict[str, Dict[str, int]] = {"train": {}, "test": {}, "ground_truth": {}}

    for split in ("train", "test"):
        split_dir = os.path.join(cat_dir, split)
        found = set(_visible_subdirs(split_dir))
        want = set(expected[split])
        if found != want:
            extra, missing = sorted(found - want), sorted(want - found)
            problems.append(f"{split}/: extra folders {extra}, missing folders {missing}")
        for dtype in sorted(found | want):
            n = len(_list_images(os.path.join(split_dir, dtype)))
            observed[split][dtype] = n
            if dtype in expected[split] and n != expected[split][dtype]:
                problems.append(f"{split}/{dtype}: {n} images, expected {expected[split][dtype]}")

    gt_dir = os.path.join(cat_dir, "ground_truth")
    want_gt = {d for d in expected["test"] if d != "good"}
    found_gt = set(_visible_subdirs(gt_dir))
    if found_gt != want_gt:
        problems.append(f"ground_truth/: extra folders {sorted(found_gt - want_gt)}, "
                        f"missing folders {sorted(want_gt - found_gt)}")
    for dtype in sorted(want_gt):
        imgs = _list_images(os.path.join(cat_dir, "test", dtype))
        missing_masks = [p for p in imgs if not os.path.exists(_mask_path(root, category, dtype, p))]
        observed["ground_truth"][dtype] = len(imgs) - len(missing_masks)
        if missing_masks:
            problems.append(f"ground_truth/{dtype}: {len(missing_masks)} masks missing")

    if problems:
        raise DatasetIntegrityError(
            f"Category '{category}' at '{cat_dir}' does not match the official MVTec AD manifest:\n  - "
            + "\n  - ".join(problems))
    return observed


def _mask_path(root: str, category: str, defect_type: str, image_path: str) -> str:
    base = os.path.splitext(os.path.basename(image_path))[0]
    return os.path.join(root, category, "ground_truth", defect_type, f"{base}_mask.png")


def load_rgb(path: str, img_size: int = 256) -> np.ndarray:
    """Reads an image as RGB uint8 resized to (img_size, img_size) with bilinear filtering."""
    with Image.open(path) as img:
        img = img.convert("RGB").resize((img_size, img_size), Image.BILINEAR)
        return np.asarray(img, dtype=np.uint8).copy()


def load_mask(path: Optional[str], img_size: int = 256) -> np.ndarray:
    """Reads a binary mask resized with nearest-neighbour; returns zeros when path is None."""
    if path is None:
        return np.zeros((img_size, img_size), dtype=np.uint8)
    with Image.open(path) as m:
        m = m.convert("L").resize((img_size, img_size), Image.NEAREST)
        return (np.asarray(m) > 127).astype(np.uint8)


def normalize_image(arr: np.ndarray) -> torch.Tensor:
    """uint8 HxWx3 -> ImageNet-normalised float tensor 3xHxW."""
    t = torch.from_numpy(arr).permute(2, 0, 1).float().div_(255.0)
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (t - mean) / std


def denormalize_image(tensor: torch.Tensor) -> np.ndarray:
    """ImageNet-normalised tensor 3xHxW -> RGB uint8 HxWx3."""
    mean = torch.tensor(IMAGENET_MEAN, device=tensor.device).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=tensor.device).view(3, 1, 1)
    arr = torch.clamp(tensor * std + mean, 0.0, 1.0).cpu().numpy().transpose(1, 2, 0)
    return (arr * 255.0).round().astype(np.uint8)


def list_train_images(root: str, category: str, validate: bool = True) -> List[str]:
    if validate:
        validate_category(root, category)
    return _list_images(os.path.join(root, category, "train", "good"))


def split_train_calibration(paths: Sequence[str], calib_fraction: float, seed: int) -> Tuple[List[str], List[str]]:
    """Splits train/good into a fitting set and a held-out nominal calibration set.

    The calibration images are never seen during fitting, so thresholds derived
    from their scores are free of test-set information.
    """
    paths = list(paths)
    if not 0.0 <= calib_fraction < 1.0:
        raise ValueError("calib_fraction must be in [0, 1).")
    n_calib = int(round(len(paths) * calib_fraction))
    order = np.random.RandomState(seed).permutation(len(paths))
    calib = sorted(paths[i] for i in order[:n_calib])
    fit = sorted(paths[i] for i in order[n_calib:])
    return fit, calib


class ImageListDataset(Dataset):
    """Nominal images from an explicit path list (used for fitting and calibration)."""

    def __init__(self, paths: Sequence[str], img_size: int = 256):
        if len(paths) == 0:
            raise ValueError("ImageListDataset received an empty path list.")
        self.paths = list(paths)
        self.img_size = img_size

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return normalize_image(load_rgb(self.paths[idx], self.img_size))


class MVTecTrainNormal(ImageListDataset):
    """All train/good images of a validated category."""

    def __init__(self, root: str, category: str, img_size: int = 256, validate: bool = True):
        super().__init__(list_train_images(root, category, validate=validate), img_size)
        self.root = root
        self.category = category


def list_test_samples(root: str, category: str, validate: bool = True) -> List[Dict[str, object]]:
    """Test samples (good + each official defect type) with label and mask path."""
    if validate:
        validate_category(root, category)
    defect_types = MVTEC_MANIFEST[category]["test"] if category in MVTEC_MANIFEST else {}
    test_dir = os.path.join(root, category, "test")
    types = sorted(defect_types) if validate else _visible_subdirs(test_dir)
    samples = []
    for dtype in types:
        for path in _list_images(os.path.join(test_dir, dtype)):
            is_defect = dtype != "good"
            mask = _mask_path(root, category, dtype, path) if is_defect else None
            if is_defect and not os.path.exists(mask):
                raise DatasetIntegrityError(f"Missing ground-truth mask for defect image '{path}'.")
            samples.append({"path": path, "label": int(is_defect), "defect_type": dtype, "mask_path": mask})
    if not samples:
        raise DatasetIntegrityError(f"No test samples found for '{category}' under '{test_dir}'.")
    return samples


class MVTecTest(Dataset):
    """Clean test split: returns (image tensor, label, mask tensor [1,H,W], meta)."""

    def __init__(self, root: str, category: str, img_size: int = 256, validate: bool = True):
        self.root = root
        self.category = category
        self.img_size = img_size
        self.samples = list_test_samples(root, category, validate=validate)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        x = normalize_image(load_rgb(s["path"], self.img_size))
        mask = torch.from_numpy(load_mask(s["mask_path"], self.img_size)).float().unsqueeze(0)
        meta = {"category": self.category, "defect_type": s["defect_type"], "img_path": s["path"]}
        return x, s["label"], mask, meta
