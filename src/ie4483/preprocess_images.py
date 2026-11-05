"""Resize the raw Dogs vs Cats handoff into the 224x224 JPEG tree the pipeline reads."""

from __future__ import annotations

import argparse
import hashlib
import io
import logging
from collections import Counter
from pathlib import Path

from PIL import Image, ImageOps

from ie4483.paths import repo_root

LOGGER = logging.getLogger(__name__)

IMAGE_SIZE = (224, 224)
SUPPORTED_FORMATS = (".jpg", ".jpeg", ".png")
# Splits in priority order: when an image appears in several, the earliest split keeps it.
SPLIT_ORDER = {"train": 0, "val": 1, "test": 2}
JPEG_QUALITY = 95


def _clean_generated_splits(output_dir: Path) -> int:
    """Delete supported images left in the output's own split folders by an earlier run; return the count."""
    removed = 0
    for split in SPLIT_ORDER:
        for old_path in (output_dir / split).rglob("*"):
            if old_path.is_file() and old_path.suffix.lower() in SUPPORTED_FORMATS:
                old_path.unlink()
                removed += 1
    return removed


def _convert(image_path: Path, image_size: tuple[int, int]) -> bytes:
    """Return the image as EXIF-oriented, centre-fitted RGB JPEG bytes."""
    with Image.open(image_path) as original:
        image = ImageOps.exif_transpose(original).convert("RGB")
        image = ImageOps.fit(image, image_size, method=Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=JPEG_QUALITY)
    return buffer.getvalue()


def preprocess_images(input_dir: Path, output_dir: Path, image_size: tuple[int, int] = IMAGE_SIZE) -> None:
    """Convert every supported image under ``input_dir`` into ``output_dir`` as ``image_size`` JPEGs.

    Splits are processed in train, val, test order and an exact-duplicate converted image is kept only the first
    time, so no image crosses the training/evaluation boundary. Test images are never dropped because
    ``submission.csv`` needs one row per test ID. Images left in ``output_dir``'s train/val/test folders by an
    earlier run are removed first so stale files cannot survive.
    """
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory {input_dir} does not exist.")
    input_resolved, output_resolved = input_dir.resolve(), output_dir.resolve()
    if input_resolved == output_resolved or input_resolved in output_resolved.parents:
        raise ValueError("Output directory cannot be the same as or a subdirectory of the input directory.")

    image_files = sorted(
        (path for path in input_dir.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_FORMATS),
        key=lambda path: (
            SPLIT_ORDER.get(path.relative_to(input_dir).parts[0], len(SPLIT_ORDER)),
            path.relative_to(input_dir).as_posix(),
        ),
    )
    if not image_files:
        LOGGER.warning("No supported image files found in %s.", input_dir)
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    removed = _clean_generated_splits(output_dir)
    LOGGER.info(
        "Input %s -> output %s: %d images at %dx%d (removed %d stale output images)",
        input_dir,
        output_dir,
        len(image_files),
        *image_size,
        removed,
    )

    processed, failed = 0, []
    folder_count: Counter[Path] = Counter()
    seen_hashes: set[bytes] = set()
    for index, image_path in enumerate(image_files, start=1):
        relative_path = image_path.relative_to(input_dir)
        try:
            data = _convert(image_path, image_size)
        except (OSError, ValueError) as error:
            failed.append((str(image_path), str(error)))
            LOGGER.error("Failed to process %s -- %s", image_path, error)
            continue
        digest = hashlib.sha256(data).digest()
        if digest in seen_hashes and relative_path.parts[0] != "test":
            continue
        seen_hashes.add(digest)
        output_path = (output_dir / relative_path).with_suffix(".jpg")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(data)
        processed += 1
        folder_count[relative_path.parent] += 1
        if index % 100 == 0 or index == len(image_files):
            LOGGER.info("Processed %d/%d images...", index, len(image_files))

    LOGGER.info("Found %d, wrote %d, failed %d; output folder %s", len(image_files), processed, len(failed), output_dir)
    for path, error in failed:
        LOGGER.error("Failed image: %s -- %s", path, error)


def main() -> None:
    root = repo_root()
    parser = argparse.ArgumentParser(description="Resize the raw dataset to 224x224 JPEGs.")
    parser.add_argument(
        "--input", type=Path, default=root / "raw-data" / "datasets", help="folder holding train/, val/, test/"
    )
    parser.add_argument("--output", type=Path, default=root / "data", help="the pipeline's data root")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    preprocess_images(args.input, args.output, IMAGE_SIZE)


if __name__ == "__main__":
    main()
