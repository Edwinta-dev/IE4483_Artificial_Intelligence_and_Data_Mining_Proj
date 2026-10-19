"""Helpers for identifying CIFAR-10 cat/dog stand-in datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PROXY_MARKER = "PROXY_DATASET.txt"


def is_proxy_dataset(root: str | Path) -> bool:
    return (Path(root) / PROXY_MARKER).is_file()


def write_proxy_marker(root: str | Path, *, seed: int, n_train: int, n_val: int) -> Path:
    path = Path(root) / PROXY_MARKER
    path.write_text(
        "Stand-in CIFAR-10 cat/dog dataset. Do not report its metrics as Dogs vs Cats results.\n"
        + json.dumps({"dataset": "CIFAR-10", "classes": {"cat": 3, "dog": 5}, "seed": seed,
                      "n_train_per_class": n_train, "n_val_per_class": n_val}, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return path


def proxy_metric(root: str | Path) -> dict[str, Any]:
    return {"proxy": True} if is_proxy_dataset(root) else {}
