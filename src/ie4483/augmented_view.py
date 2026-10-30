"""Create a fixed, on-disk training augmentation view."""

from __future__ import annotations

import csv
import random
import shutil
from collections import Counter
from pathlib import Path

from PIL import Image, ImageEnhance

from ie4483.seed import seed_everything

ADJUSTMENTS = ("horizontal_flip", "rotation", "random_resized_crop", "colour_jitter")
OUTPUT_SIZE = (224, 224)


def _adjust(image: Image.Image, adjustment: str, rng: random.Random) -> tuple[Image.Image, str]:
    if adjustment == "horizontal_flip":
        return image.transpose(Image.Transpose.FLIP_LEFT_RIGHT), "horizontal"
    if adjustment == "rotation":
        angle = rng.uniform(-15.0, 15.0)
        return image.rotate(angle, resample=Image.Resampling.BILINEAR), f"angle={angle:.8f}"
    if adjustment == "random_resized_crop":
        width, height = image.size
        scale = rng.uniform(0.8, 1.0)
        ratio = rng.uniform(3 / 4, 4 / 3)
        crop_width = min(width, max(1, round((width * height * scale * ratio) ** 0.5)))
        crop_height = min(height, max(1, round((width * height * scale / ratio) ** 0.5)))
        left = rng.randint(0, width - crop_width)
        top = rng.randint(0, height - crop_height)
        result = image.crop((left, top, left + crop_width, top + crop_height)).resize(OUTPUT_SIZE, Image.Resampling.BILINEAR)
        return result, f"scale={scale:.8f};ratio={ratio:.8f};left={left};top={top};width={crop_width};height={crop_height}"
    if adjustment == "colour_jitter":
        factors = [rng.uniform(0.8, 1.2) for _ in range(3)]
        result = image
        for enhancer, factor in zip((ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color), factors):
            result = enhancer(result).enhance(factor)
        return result, f"brightness={factors[0]:.8f};contrast={factors[1]:.8f};saturation={factors[2]:.8f}"
    raise ValueError(f"unknown adjustment: {adjustment}")


def make_augmented_view(input_dir: str | Path, output_dir: str | Path, mode: str, seed: int = 0) -> dict[str, int]:
    """Materialize one seeded adjustment per training image and copy held-out data."""
    source, destination = Path(input_dir), Path(output_dir)
    if mode not in {"augmented", "mixed"}:
        raise ValueError("mode must be 'augmented' or 'mixed'")
    if not source.is_dir():
        raise FileNotFoundError(f"input directory does not exist: {source}")
    if source.resolve() == destination.resolve() or source.resolve() in destination.resolve().parents:
        raise ValueError("output directory must not be the input directory or inside it")
    seed_everything(seed)
    rng = random.Random(seed)
    counts: Counter[str] = Counter()
    rows: list[tuple[str, str, str]] = []
    for label in ("cat", "dog"):
        input_class = source / "train" / label
        if not input_class.is_dir():
            raise FileNotFoundError(f"missing training class directory: {input_class}")
        output_class = destination / "train" / label
        output_class.mkdir(parents=True, exist_ok=True)
        for path in sorted(p for p in input_class.iterdir() if p.is_file()):
            if mode == "mixed":
                shutil.copy2(path, output_class / path.name)
            with Image.open(path) as opened:
                image = opened.convert("RGB")
            operation = rng.choice(ADJUSTMENTS)
            adjusted, parameter = _adjust(image, operation, rng)
            filename = f"{path.stem}_aug.jpg" if mode == "mixed" else path.name
            adjusted.save(output_class / filename, format="JPEG", quality=95)
            counts[operation] += 1
            rows.append((f"train/{label}/{filename}", operation, parameter))
    for split in ("val", "test"):
        path = source / split
        if path.exists():
            shutil.copytree(path, destination / split, dirs_exist_ok=True)
    with (destination / "augmentation.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("file", "adjustment", "parameter"))
        writer.writerows(rows)
    return {operation: counts[operation] for operation in ADJUSTMENTS}
