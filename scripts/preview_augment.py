"""Save a contact sheet of repeated views under the configured train transform."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from PIL import Image

from ie4483.config import load_config
from ie4483.transforms import IMAGENET_MEAN, IMAGENET_STD, build_train_transform


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument("--output", type=Path, default=Path("runs/augmentation_preview.png"))
    args = parser.parse_args()
    if args.n < 1:
        parser.error("--n must be at least 1")
    cfg = load_config(args.config)
    image = Image.open(args.image).convert("RGB")
    transform = build_train_transform(cfg)
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    columns = min(4, args.n)
    rows = (args.n + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(3 * columns, 3 * rows), squeeze=False)
    for index, axis in enumerate(axes.flat):
        axis.axis("off")
        if index >= args.n:
            continue
        tensor = transform(image).cpu()
        pixels = (tensor * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
        axis.imshow(pixels)
        axis.set_title(f"View {index + 1}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(args.output, dpi=150, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
