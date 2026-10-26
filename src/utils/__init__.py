import os
import random
from typing import Any, Dict

import numpy as np
import yaml


def ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seeds Python, NumPy and PyTorch RNGs; optionally forces deterministic cuDNN kernels."""
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config file '{path}' must contain a mapping.")
    return cfg


def parse_run_key(filename: str):
    """'<category>_<method>_<seed>.npz' -> (category, method, seed). Category may contain '_'."""
    stem = os.path.splitext(os.path.basename(filename))[0]
    parts = stem.split("_")
    if len(parts) < 3:
        raise ValueError(f"Cannot parse run key from '{filename}'; expected <category>_<method>_<seed>.")
    return "_".join(parts[:-2]), parts[-2], int(parts[-1])
