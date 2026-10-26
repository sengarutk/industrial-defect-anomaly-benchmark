"""Downloads MVTec AD categories from the Hugging Face mirror and verifies them.

Each category is checked against the official manifest in src/mvtec.py after
download; a category that does not match exactly is reported as an error.
Nothing is ever generated locally.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.mvtec import MVTEC_MANIFEST, DatasetIntegrityError, validate_category  # noqa: E402

HF_REPO_ID = "foersben/mvtec-ad"


def download_category(data_root: str, category: str, max_workers: int) -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        allow_patterns=[f"{category}/train/*", f"{category}/test/*", f"{category}/ground_truth/*"],
        local_dir=data_root,
        max_workers=max_workers,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Download and verify MVTec AD categories")
    parser.add_argument("--categories", nargs="+", default=["bottle", "cable", "carpet", "grid", "hazelnut",
                                                            "leather", "metal_nut"])
    parser.add_argument("--data-root", type=str, default="data/mvtec_ad")
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--verify-only", action="store_true", help="Only validate what is on disk")
    args = parser.parse_args()

    os.makedirs(args.data_root, exist_ok=True)
    failed = []
    for cat in args.categories:
        if cat not in MVTEC_MANIFEST:
            print(f"[error] unknown category '{cat}'")
            failed.append(cat)
            continue
        try:
            validate_category(args.data_root, cat)
            print(f"[ok] {cat}: already present and matches the official manifest")
            continue
        except DatasetIntegrityError as err:
            if args.verify_only:
                print(f"[error] {err}")
                failed.append(cat)
                continue
        print(f"[download] {cat} from {HF_REPO_ID}")
        download_category(args.data_root, cat, args.max_workers)
        try:
            counts = validate_category(args.data_root, cat)
            n_test = sum(counts["test"].values())
            print(f"[ok] {cat}: {counts['train']['good']} train / {n_test} test images verified")
        except DatasetIntegrityError as err:
            print(f"[error] {err}")
            failed.append(cat)

    if failed:
        print(f"Categories with problems: {failed}")
        return 1
    print("All requested categories match the official MVTec AD manifest.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
