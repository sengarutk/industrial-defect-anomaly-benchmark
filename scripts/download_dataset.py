import argparse
import os
import sys
import tarfile
import urllib.request
from typing import List
from PIL import Image, ImageDraw
import numpy as np

# Ensure repository root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

ALL_CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper"
]

OFFICIAL_MVTEC_URL = "https://www.mydrive.ch/shares/38587/119a744de567daf142dc3b00f174959b/download/420938113-1629952094/mvtec_anomaly_detection.tar.xz"


def generate_mock_category(data_root: str, category: str, num_train: int = 10, num_test: int = 6):
    cat_dir = os.path.join(data_root, category)
    train_good = os.path.join(cat_dir, "train", "good")
    test_good = os.path.join(cat_dir, "test", "good")
    test_defect = os.path.join(cat_dir, "test", "defect")
    gt_defect = os.path.join(cat_dir, "ground_truth", "defect")

    for d in [train_good, test_good, test_defect, gt_defect]:
        os.makedirs(d, exist_ok=True)

    np.random.seed(hash(category) % (2**32))

    # 1. Generate normal training images
    base_color = (int(np.random.randint(80, 180)), int(np.random.randint(80, 180)), int(np.random.randint(80, 180)))
    for i in range(num_train):
        img_arr = np.ones((256, 256, 3), dtype=np.uint8) * base_color
        noise = np.random.randint(-15, 15, (256, 256, 3), dtype=np.int16)
        img_arr = np.clip(img_arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(img_arr)
        draw = ImageDraw.Draw(img)
        draw.rectangle([40, 40, 216, 216], outline=(220, 220, 220), width=2)
        draw.ellipse([80, 80, 176, 176], fill=(base_color[0] + 20, base_color[1] - 10, base_color[2] + 10))
        img.save(os.path.join(train_good, f"{i:03d}.png"))

    # 2. Generate normal test images
    for i in range(max(1, num_test // 2)):
        img_arr = np.ones((256, 256, 3), dtype=np.uint8) * base_color
        noise = np.random.randint(-15, 15, (256, 256, 3), dtype=np.int16)
        img_arr = np.clip(img_arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(img_arr)
        draw = ImageDraw.Draw(img)
        draw.rectangle([40, 40, 216, 216], outline=(220, 220, 220), width=2)
        draw.ellipse([80, 80, 176, 176], fill=(base_color[0] + 20, base_color[1] - 10, base_color[2] + 10))
        img.save(os.path.join(test_good, f"{i:03d}.png"))

    # 3. Generate defective test images + ground truth masks
    for i in range(max(1, num_test // 2)):
        img_arr = np.ones((256, 256, 3), dtype=np.uint8) * base_color
        noise = np.random.randint(-15, 15, (256, 256, 3), dtype=np.int16)
        img_arr = np.clip(img_arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(img_arr)
        draw = ImageDraw.Draw(img)
        draw.rectangle([40, 40, 216, 216], outline=(220, 220, 220), width=2)
        draw.ellipse([80, 80, 176, 176], fill=(base_color[0] + 20, base_color[1] - 10, base_color[2] + 10))

        # Defect mask
        mask = Image.new("L", (256, 256), 0)
        mask_draw = ImageDraw.Draw(mask)

        # Inject defect spot
        x0 = int(np.random.randint(60, 180))
        y0 = int(np.random.randint(60, 180))
        r = int(np.random.randint(15, 30))
        draw.ellipse([x0 - r, y0 - r, x0 + r, y0 + r], fill=(240, 20, 20))
        mask_draw.ellipse([x0 - r, y0 - r, x0 + r, y0 + r], fill=255)

        img.save(os.path.join(test_defect, f"{i:03d}.png"))
        mask.save(os.path.join(gt_defect, f"{i:03d}_mask.png"))

    print(f"✅ Generated mock dataset for '{category}' in {cat_dir}")


def download_official_mvtec(data_root: str, categories: List[str]):
    os.makedirs(data_root, exist_ok=True)
    archive_path = os.path.join(data_root, "mvtec_anomaly_detection.tar.xz")

    if not os.path.exists(archive_path):
        print(f"Downloading MVTec AD archive to {archive_path}...")
        try:
            urllib.request.urlretrieve(OFFICIAL_MVTEC_URL, archive_path)
            print("Download completed.")
        except Exception as e:
            print(f"Error downloading MVTec AD: {e}")
            print("Falling back to generating synthetic mock dataset for requested categories.")
            for cat in categories:
                generate_mock_category(data_root, cat)
            return

    print("Extracting archive...")
    with tarfile.open(archive_path) as tar:
        tar.extractall(path=data_root)
    print(f"Extraction complete under {data_root}")


def main():
    parser = argparse.ArgumentParser(description="MVTec AD Dataset Downloader & Mock Generator")
    parser.add_argument("--categories", nargs="+", default=["bottle", "cable", "hazelnut", "metal_nut"])
    parser.add_argument("--data-root", type=str, default="data/mvtec_ad")
    parser.add_argument("--mock", action="store_true", default=False, help="Generate synthetic mock dataset")
    parser.add_argument("--num-mock-train", type=int, default=10)
    parser.add_argument("--num-mock-test", type=int, default=6)
    args = parser.parse_args()

    target_cats = ALL_CATEGORIES if "all" in args.categories else args.categories

    if args.mock:
        print(f"=== Generating Mock MVTec AD Datasets ({len(target_cats)} categories) ===")
        for cat in target_cats:
            generate_mock_category(args.data_root, cat, args.num_mock_train, args.num_mock_test)
        print("\nAll mock categories ready.")
    else:
        print(f"=== Downloading Official MVTec AD Dataset ===")
        download_official_mvtec(args.data_root, target_cats)


if __name__ == "__main__":
    main()
