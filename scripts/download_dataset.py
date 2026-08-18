import argparse
import os
import sys
import glob
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from PIL import Image, ImageDraw
import numpy as np
from huggingface_hub import HfApi

# Ensure repository root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

ALL_CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper"
]

CATEGORY_MIN_TRAIN_COUNTS = {
    "bottle": 150,
    "cable": 150,
    "hazelnut": 250,
    "metal_nut": 150,
    "carpet": 200,
    "capsule": 150,
    "grid": 150,
    "leather": 150,
    "pill": 200,
    "screw": 250,
    "tile": 150,
    "toothbrush": 40,
    "transistor": 150,
    "wood": 150,
    "zipper": 150
}


def verify_category_integrity(data_root: str, category: str) -> bool:
    cat_dir = os.path.join(data_root, category)
    train_good = os.path.join(cat_dir, "train", "good")
    test_dir = os.path.join(cat_dir, "test")

    if not os.path.exists(train_good) or not os.path.exists(test_dir):
        return False

    train_imgs = glob.glob(os.path.join(train_good, "*.png")) + glob.glob(os.path.join(train_good, "*.PNG"))
    min_train = CATEGORY_MIN_TRAIN_COUNTS.get(category, 50)

    if len(train_imgs) < min_train:
        return False

    test_dtypes = [d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))]
    if len(test_dtypes) < 2:
        return False

    return True


def _download_single_file(repo_id: str, rel_path: str, local_path: str, retries: int = 4):
    if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
        return True

    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/{rel_path}"
    
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64)"})
            with urllib.request.urlopen(req, timeout=30) as resp, open(local_path, "wb") as f:
                while True:
                    chunk = resp.read(65536)
                    if not chunk:
                        break
                    f.write(chunk)
            return True
        except Exception as e:
            if os.path.exists(local_path):
                try:
                    os.remove(local_path)
                except Exception:
                    pass
            if attempt == retries - 1:
                raise RuntimeError(f"Failed to download {rel_path} after {retries} attempts: {e}") from e
            time.sleep(1.0 * (attempt + 1))


def download_real_category(data_root: str, category: str, max_workers: int = 8):
    os.makedirs(data_root, exist_ok=True)
    
    if verify_category_integrity(data_root, category):
        train_count = len(glob.glob(os.path.join(data_root, category, "train", "good", "*.png")))
        print(f"✅ Category '{category}' already verified ({train_count} train images). Skipping download.")
        return

    print(f"--> Fetching authentic file tree for '{category}' from foersben/mvtec-ad...")
    api = HfApi()
    repo_id = "foersben/mvtec-ad"
    try:
        all_files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
    except Exception as e:
        raise RuntimeError(f"Could not retrieve file list from HuggingFace: {e}") from e

    cat_files = [f for f in all_files if f.startswith(f"{category}/")]
    if not cat_files:
        raise ValueError(f"No files found for category '{category}' in {repo_id}")

    print(f"Downloading {len(cat_files)} files for '{category}' with {max_workers} parallel workers...")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_download_single_file, repo_id, rel_path, os.path.join(data_root, rel_path)): rel_path
            for rel_path in cat_files
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Downloading {category}"):
            future.result()

    if not verify_category_integrity(data_root, category):
        raise ValueError(f"Integrity check failed for category '{category}' after download.")

    train_count = len(glob.glob(os.path.join(data_root, category, "train", "good", "*.png")))
    test_count = len(glob.glob(os.path.join(data_root, category, "test", "*", "*.png")))
    print(f"✅ Successfully downloaded and verified '{category}': {train_count} train images, {test_count} test images.")


def generate_mock_category(data_root: str, category: str, num_train: int = 10, num_test: int = 6):
    cat_dir = os.path.join(data_root, category)
    train_good = os.path.join(cat_dir, "train", "good")
    test_good = os.path.join(cat_dir, "test", "good")
    test_defect = os.path.join(cat_dir, "test", "defect")
    gt_defect = os.path.join(cat_dir, "ground_truth", "defect")

    for d in [train_good, test_good, test_defect, gt_defect]:
        os.makedirs(d, exist_ok=True)

    np.random.seed(hash(category) % (2**32))

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

    for i in range(max(1, num_test // 2)):
        img_arr = np.ones((256, 256, 3), dtype=np.uint8) * base_color
        noise = np.random.randint(-15, 15, (256, 256, 3), dtype=np.int16)
        img_arr = np.clip(img_arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(img_arr)
        draw = ImageDraw.Draw(img)
        draw.rectangle([40, 40, 216, 216], outline=(220, 220, 220), width=2)
        draw.ellipse([80, 80, 176, 176], fill=(base_color[0] + 20, base_color[1] - 10, base_color[2] + 10))
        img.save(os.path.join(test_good, f"{i:03d}.png"))

    for i in range(max(1, num_test // 2)):
        img_arr = np.ones((256, 256, 3), dtype=np.uint8) * base_color
        noise = np.random.randint(-15, 15, (256, 256, 3), dtype=np.int16)
        img_arr = np.clip(img_arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(img_arr)
        draw = ImageDraw.Draw(img)
        draw.rectangle([40, 40, 216, 216], outline=(220, 220, 220), width=2)
        draw.ellipse([80, 80, 176, 176], fill=(base_color[0] + 20, base_color[1] - 10, base_color[2] + 10))

        mask = Image.new("L", (256, 256), 0)
        mask_draw = ImageDraw.Draw(mask)

        x0 = int(np.random.randint(60, 180))
        y0 = int(np.random.randint(60, 180))
        r = int(np.random.randint(15, 30))
        draw.ellipse([x0 - r, y0 - r, x0 + r, y0 + r], fill=(240, 20, 20))
        mask_draw.ellipse([x0 - r, y0 - r, x0 + r, y0 + r], fill=255)

        img.save(os.path.join(test_defect, f"{i:03d}.png"))
        mask.save(os.path.join(gt_defect, f"{i:03d}_mask.png"))

    print(f"✅ Generated mock dataset for '{category}' in {cat_dir}")


def main():
    parser = argparse.ArgumentParser(description="MVTec AD Official Dataset Downloader")
    parser.add_argument("--categories", nargs="+", default=["bottle", "cable", "hazelnut", "metal_nut", "carpet"])
    parser.add_argument("--data-root", type=str, default="data/mvtec_ad")
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--mock", action="store_true", default=False, help="Explicitly generate synthetic mock data")
    parser.add_argument("--num-mock-train", type=int, default=10)
    parser.add_argument("--num-mock-test", type=int, default=6)
    args = parser.parse_args()

    target_cats = ALL_CATEGORIES if "all" in args.categories else args.categories

    if args.mock:
        print(f"=== Generating Synthetic Mock MVTec AD Datasets ({len(target_cats)} categories) ===")
        for cat in target_cats:
            generate_mock_category(args.data_root, cat, args.num_mock_train, args.num_mock_test)
        print("\nMock generation complete.")
    else:
        print(f"=== Downloading Authentic MVTec AD Datasets ({len(target_cats)} categories) ===")
        for cat in target_cats:
            download_real_category(args.data_root, cat, max_workers=args.max_workers)
        print("\nAll authentic categories ready.")


if __name__ == "__main__":
    main()
