"""Helpers for finding a CUDA training batch that fits with headroom."""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

EFFECTIVE_BATCH_SIZE = 32


def accumulation_steps(safe_batch_size: int, effective_batch_size: int = EFFECTIVE_BATCH_SIZE) -> int | None:
    """Return the whole number of microbatches needed to reach the target."""
    if safe_batch_size < 1:
        return None
    if effective_batch_size < 1:
        raise ValueError("effective_batch_size must be positive")
    return math.ceil(effective_batch_size / safe_batch_size)


def write_probe_json(
    path: str | Path,
    results: list[dict[str, Any]],
    image_size: int,
    safety_margin: float = 0.8,
    seed: int = 42,
) -> Path:
    """Write the machine-readable probe report and return its path."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "image_size": image_size,
        "effective_batch_size": EFFECTIVE_BATCH_SIZE,
        "safety_margin": safety_margin,
        "seed": seed,
        "results": results,
    }
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return destination


def find_max_batch(
    step_fn: Callable[[int], object],
    start: int,
    limit: int,
    safety_margin: float = 0.8,
) -> int:
    """Return the safety-adjusted largest successful batch in ``[start, limit]``.

    ``step_fn`` must perform one complete training step and raise
    ``torch.cuda.OutOfMemoryError`` when it cannot fit. It is called with
    increasing batch sizes; CUDA cache is emptied after every attempt.
    """
    if isinstance(start, bool) or not isinstance(start, int) or start < 1:
        raise ValueError("start must be a positive integer")
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < start:
        raise ValueError("limit must be an integer greater than or equal to start")
    if isinstance(safety_margin, bool) or not 0 < safety_margin <= 1:
        raise ValueError("safety_margin must be in (0, 1]")

    def attempt(batch_size: int) -> bool:
        try:
            step_fn(batch_size)
            return True
        except torch.cuda.OutOfMemoryError:
            return False
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if not attempt(start):
        return 0
    low, high = start, min(limit + 1, max(start + 1, start * 2))
    while high <= limit and attempt(high):
        low = high
        high = min(limit + 1, high * 2)
    high = min(high, limit + 1)
    while low + 1 < high:
        middle = (low + high) // 2
        if attempt(middle):
            low = middle
        else:
            high = middle
    return max(1, int(low * safety_margin))
