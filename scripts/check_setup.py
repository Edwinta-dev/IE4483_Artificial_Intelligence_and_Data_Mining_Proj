"""Report local Python, torch, weights, and Dogs vs Cats data readiness."""

from __future__ import annotations

import argparse
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ie4483.config import load_config
from ie4483.paths import resolve_data_root, weights_cache_dir

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tif", ".tiff"}
WEIGHT_FILES = ("resnet18-f37072fd.pth", "resnet50-11ad3fa6.pth")


def inspect_setup(data_root: str | Path | None = None, cache_dir: str | Path | None = None) -> str:
    """Return a readable setup report without raising for unavailable CUDA."""
    import torch

    root = Path(data_root) if data_root is not None else resolve_data_root()
    cache = Path(cache_dir) if cache_dir is not None else weights_cache_dir()
    lines = [f"Python: {platform.python_version()}", f"PyTorch: {torch.__version__}"]
    if torch.cuda.is_available():
        try:
            torch.empty(1, device="cuda")
            lines.append("CUDA: available (test tensor allocated)")
        except Exception as exc:  # noqa: BLE001 - report any CUDA runtime/driver failure instead of crashing.
            lines.append(f"CUDA: unavailable ({type(exc).__name__}: {exc})")
    else:
        lines.append("CUDA: unavailable (torch.cuda.is_available() is false)")

    lines.append(f"Weights cache: {cache}")
    for filename in WEIGHT_FILES:
        state = "cached" if (cache / filename).is_file() else "MISSING"
        lines.append(f"  {filename}: {state}")

    lines.append(f"Data root: {root}")
    for relative in ("train/cat", "train/dog", "val/cat", "val/dog", "test"):
        folder = root / relative
        count = sum(1 for item in folder.iterdir() if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES) if folder.is_dir() else 0
        status = "" if folder.is_dir() else " (MISSING)"
        lines.append(f"  {relative}: {count} images{status}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="YAML config; data.root overrides IE4483_DATA")
    parser.add_argument("--data-root", type=Path, help="explicit data root override")
    args = parser.parse_args()
    configured = load_config(args.config).data_root if args.config else None
    root = resolve_data_root(args.data_root or configured)
    print(inspect_setup(root))


if __name__ == "__main__":
    main()
