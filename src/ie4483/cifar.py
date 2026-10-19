"""CIFAR-10 Python batch loading and deterministic train/validation split."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

CIFAR10_CLASSES = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def cifar_transform(image_size: int, training: bool = False):
    steps = [transforms.Resize((image_size, image_size))]
    if training:
        steps += [transforms.RandomCrop(image_size, padding=4), transforms.RandomHorizontalFlip()]
    steps += [transforms.ToTensor(), transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)]
    return transforms.Compose(steps)


def _batch(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("rb") as stream:
        data = pickle.load(stream, encoding="bytes")
    images = np.asarray(data[b"data"], dtype=np.uint8).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    labels = np.asarray(data.get(b"labels", data.get(b"fine_labels")), dtype=np.int64)
    if len(images) != len(labels):
        raise ValueError(f"CIFAR image/label count mismatch in {path}")
    return images, labels


class CIFAR10Dataset(Dataset):
    """Labeled images from standard CIFAR-10 Python batches."""

    def __init__(self, root: str | Path, split: str = "train", transform=None, seed: int = 42,
                 val_fraction: float = 0.1) -> None:
        self.root = Path(root) / "cifar-10-batches-py"
        self.transform = transform
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be train, val, or test")
        names = [f"data_batch_{i}" for i in range(1, 6)] if split != "test" else ["test_batch"]
        batches = [_batch(self.root / name) for name in names]
        images = np.concatenate([batch[0] for batch in batches])
        labels = np.concatenate([batch[1] for batch in batches])
        indices = np.arange(len(labels))
        if split != "test":
            if not 0 < val_fraction < 1:
                raise ValueError("val_fraction must be between 0 and 1")
            rng = np.random.default_rng(seed)
            val_parts, train_parts = [], []
            for class_id in range(len(CIFAR10_CLASSES)):
                class_indices = indices[labels == class_id].copy()
                rng.shuffle(class_indices)
                n_val = round(len(class_indices) * val_fraction)
                val_parts.extend(class_indices[:n_val])
                train_parts.extend(class_indices[n_val:])
            indices = np.array(train_parts if split == "train" else val_parts, dtype=np.int64)
        self.split = split
        self.source_indices = indices
        self.images = images[indices]
        self.labels = labels[indices]

    @property
    def samples(self) -> list[tuple[str, int]]:
        """Stable per-image identities for feature-cache keys."""
        return [(f"cifar10/{'test' if self.split == 'test' else 'train'}/{int(i)}", int(label))
                for i, label in zip(self.source_indices, self.labels)]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[Any, int]:
        image = Image.fromarray(self.images[index], mode="RGB")
        if self.transform:
            image = self.transform(image)
        else:
            image = torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1).float().div_(255)
        return image, int(self.labels[index])
