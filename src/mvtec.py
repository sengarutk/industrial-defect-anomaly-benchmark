import os
from typing import List, Tuple, Optional, Dict

from PIL import Image
from torch.utils.data import Dataset


def _img_to_tensor(img):
    import torchvision.transforms as T
    tf = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
    ])
    return tf(img)


class MVTecTrainNormal(Dataset):
    """
    Normal-only training set (unsupervised anomaly detection assumption).
    """
    def __init__(self, root: str, category: str):
        self.root = root
        self.category = category
        self.img_dir = os.path.join(root, category, "train", "good")
        self.paths = [os.path.join(self.img_dir, p) for p in sorted(os.listdir(self.img_dir)) if p.endswith(".png")]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")
        x = _img_to_tensor(img)
        return x


class MVTecTest(Dataset):
    """
    Test contains both good and defects + pixel masks.
    Returns:
      x: image tensor
      y: image label (0 normal, 1 anomaly)
      mask: optional mask tensor (1 = anomaly pixel), None for good
      meta: {path, defect_type}
    """
    def __init__(self, root: str, category: str):
        self.root = root
        self.category = category
        self.test_dir = os.path.join(root, category, "test")
        self.gt_dir = os.path.join(root, category, "ground_truth")

        self.samples = []
        defect_types = sorted(os.listdir(self.test_dir))

        for d in defect_types:
            ddir = os.path.join(self.test_dir, d)
            if not os.path.isdir(ddir):
                continue
            for p in sorted(os.listdir(ddir)):
                if not p.endswith(".png"):
                    continue
                img_path = os.path.join(ddir, p)
                y = 0 if d == "good" else 1

                mask_path = None
                if y == 1:
                    # mask name format: <name>_mask.png
                    base = p.replace(".png", "")
                    mask_path = os.path.join(self.gt_dir, d, base + "_mask.png")

                self.samples.append((img_path, y, mask_path, d))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        import torch
        import torchvision.transforms as T

        img_path, y, mask_path, defect_type = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        x = _img_to_tensor(img)

        # Always return a tensor mask (DataLoader cannot collate None)
        tfm = T.Compose([T.Resize((256, 256)), T.ToTensor()])

        if mask_path is not None and os.path.exists(mask_path):
            m = Image.open(mask_path).convert("L")
            mask = (tfm(m) > 0.5).float()
        else:
            # good sample → empty mask
            mask = torch.zeros((1, 256, 256), dtype=torch.float32)

        meta = {"path": img_path, "defect_type": defect_type}
        return x, int(y), mask, meta
