"""Local image datasets and deterministic data loader helpers."""

from __future__ import annotations

import csv
import logging
import random
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import DataLoader, Dataset

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp"}
LOGGER = logging.getLogger(__name__)


def _default_transform(image: Image.Image) -> torch.Tensor:
    """Convert RGB pixels to a float tensor in channel-first order."""
    return torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1).float().div_(255)


class ImageFolderDataset(Dataset):
    """Images stored under ``root/split/class_name`` with ordered labels."""

    def __init__(
        self,
        root: str | Path,
        split: str,
        classes: Sequence[str],
        transform: Callable[[Image.Image], Any] | None = None,
        n_per_class: int | None = None,
        seed: int = 42,
        manifest_path: str | Path | None = None,
    ) -> None:
        if not classes or len(set(classes)) != len(classes):
            raise ValueError("classes must be a non-empty sequence of unique names")
        if n_per_class is not None and n_per_class < 0:
            raise ValueError("n_per_class must be non-negative or None")
        self.root = Path(root)
        self.split = split
        self.classes = tuple(classes)
        self.class_to_idx = {name: index for index, name in enumerate(classes)}
        self.transform = transform or _default_transform
        self.unreadable_count = 0
        self.samples: list[tuple[Path, int]] = []
        rng = random.Random(seed)

        for class_name, label in self.class_to_idx.items():
            folder = self.root / split / class_name
            files = sorted(
                (path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES),
                key=lambda path: path.name,
            ) if folder.is_dir() else []
            readable: list[Path] = []
            for path in files:
                try:
                    with Image.open(path) as image:
                        image.verify()
                    readable.append(path)
                except (OSError, ValueError, UnidentifiedImageError):
                    self.unreadable_count += 1
                    LOGGER.warning("Skipping unreadable image: %s", path)
            if n_per_class is not None and len(readable) > n_per_class:
                readable = sorted(rng.sample(readable, n_per_class))
            self.samples.extend((path, label) for path in readable)

        LOGGER.info("Skipped %d unreadable image(s) in %s", self.unreadable_count, split)
        if manifest_path is not None:
            self.write_manifest(manifest_path)

    def write_manifest(self, path: str | Path) -> None:
        manifest = Path(path)
        manifest.parent.mkdir(parents=True, exist_ok=True)
        with manifest.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(("split", "class", "path"))
            for image_path, label in self.samples:
                writer.writerow((self.split, self.classes[label], str(image_path)))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Any, int]:
        path, label = self.samples[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
            image = self.transform(image)
        return image, label


class UnlabelledTestDataset(Dataset):
    """Unlabelled images named by numeric IDs in ``root/test``."""

    def __init__(self, root: str | Path, transform: Callable[[Image.Image], Any] | None = None) -> None:
        self.transform = transform or _default_transform
        folder = Path(root) / "test"
        self.samples: list[tuple[Path, int]] = []
        self.unreadable_count = 0
        if folder.is_dir():
            for path in folder.iterdir():
                if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
                    continue
                try:
                    image_id = int(path.stem)
                except ValueError:
                    LOGGER.warning("Skipping test image with non-numeric ID: %s", path)
                    continue
                try:
                    with Image.open(path) as image:
                        image.verify()
                except (OSError, ValueError, UnidentifiedImageError):
                    self.unreadable_count += 1
                    LOGGER.warning("Skipping unreadable image: %s", path)
                    continue
                self.samples.append((path, image_id))
        self.samples.sort(key=lambda sample: sample[1])
        LOGGER.info("Skipped %d unreadable image(s) in test", self.unreadable_count)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Any, int]:
        path, image_id = self.samples[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
            image = self.transform(image)
        return image, image_id


def make_dataloader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int = 0,
    *,
    shuffle: bool = False,
    seed: int = 42,
) -> DataLoader:
    """Create a loader with deterministic shuffling and CUDA-aware pinning."""
    if batch_size < 1 or num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers non-negative")
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )
