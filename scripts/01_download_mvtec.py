import os
from src.config import TrainConfig

def main():
    cfg = TrainConfig()
    root = cfg.mvtec_root

    assert os.path.exists(root), f"Dataset not found at: {root}"
    for cat in cfg.categories:
        p = os.path.join(root, cat)
        assert os.path.isdir(p), f"Missing category folder: {p}"
        assert os.path.isdir(os.path.join(p, "train", "good")), "Missing train/good"
        assert os.path.isdir(os.path.join(p, "test")), "Missing test/"
        assert os.path.isdir(os.path.join(p, "ground_truth")), "Missing ground_truth/"
    print("MVTec AD structure looks correct.")

if __name__ == "__main__":
    main()
