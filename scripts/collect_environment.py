"""Writes docs/environment.md from the interpreter that runs it (no hand-written versions)."""
import argparse
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone


def version(mod: str) -> str:
    try:
        m = __import__(mod)
        return getattr(m, "__version__", "installed")
    except Exception as e:  # pragma: no cover - informational only
        return f"unavailable ({type(e).__name__})"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="docs/environment.md")
    parser.add_argument("--note", default="", help="Free-text note appended to the file")
    args = parser.parse_args()
    import torch

    cuda = torch.cuda.is_available()
    lines = [
        "# Environment", "",
        f"Captured {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC} by `scripts/collect_environment.py`.", "",
        f"* Platform: {platform.platform()}",
        f"* CPU: {platform.processor() or platform.machine()} ({os.cpu_count()} logical cores)",
        f"* Python: {sys.version.split()[0]} ({sys.executable})",
        f"* torch {torch.__version__}, torchvision {version('torchvision')}, numpy {version('numpy')}, "
        f"scipy {version('scipy')}, scikit-learn {version('sklearn')}, pandas {version('pandas')}, "
        f"opencv {version('cv2')}, matplotlib {version('matplotlib')}",
        f"* CUDA available: {cuda}",
    ]
    if cuda:
        p = torch.cuda.get_device_properties(0)
        lines += [f"* GPU: {p.name}, {p.total_memory / 2 ** 20:.0f} MiB; CUDA runtime {torch.version.cuda}; "
                  f"cuDNN {torch.backends.cudnn.version()}"]
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
        lines.append(f"* Repository commit: {commit}")
    except Exception:
        pass
    if args.note:
        lines += ["", args.note]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
