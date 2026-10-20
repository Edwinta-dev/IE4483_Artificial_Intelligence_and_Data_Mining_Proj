"""Deterministic CIFAR class-imbalance helpers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import Subset, WeightedRandomSampler


def imbalanced_subset(dataset, fractions: dict[int | str, float], seed: int) -> tuple[Subset, dict[int, int]]:
    """Keep configured per-class fractions, sampling without replacement."""
    labels = np.asarray(dataset.labels)
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    counts = {}
    for label in sorted(np.unique(labels).tolist()):
        fraction = float(fractions.get(label, fractions.get(str(label), 1.0)))
        if not 0 < fraction <= 1:
            raise ValueError(f"imbalance fraction for class {label} must be in (0, 1]")
        indices = np.flatnonzero(labels == label)
        rng.shuffle(indices)
        keep = min(len(indices), max(1, round(len(indices) * fraction)))
        selected.extend(indices[:keep].tolist())
        counts[int(label)] = keep
    rng.shuffle(selected)
    return Subset(dataset, selected), counts


def class_weights(labels, num_classes: int) -> torch.Tensor:
    counts = torch.bincount(torch.as_tensor(labels, dtype=torch.long), minlength=num_classes).float()
    if torch.any(counts == 0):
        raise ValueError("class weights require at least one training example per class")
    weights = counts.sum() / (num_classes * counts)
    return weights / weights.mean()


def balanced_sampler(labels, seed: int) -> WeightedRandomSampler:
    labels = torch.as_tensor(labels, dtype=torch.long)
    counts = torch.bincount(labels)
    if torch.any(counts == 0):
        raise ValueError("balanced sampler requires at least one training example per class")
    weights = 1.0 / counts[labels].float()
    return WeightedRandomSampler(weights, len(labels), replacement=True,
                                 generator=torch.Generator().manual_seed(seed))


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None, label_smoothing: float = 0.0):
        super().__init__()
        if gamma < 0:
            raise ValueError("focal gamma must be non-negative")
        self.gamma = gamma
        self.register_buffer("weight", weight)
        self.label_smoothing = label_smoothing

    def forward(self, logits, target):
        ce = nn.functional.cross_entropy(logits, target, weight=self.weight,
                                         label_smoothing=self.label_smoothing, reduction="none")
        if self.gamma == 0:
            return ce.mean()
        true_class_probability = logits.softmax(dim=1).gather(1, target.unsqueeze(1)).squeeze(1)
        return (((1 - true_class_probability).clamp_min(0) ** self.gamma) * ce).mean()


def write_class_counts(path: str | Path, counts: dict[int | str, int]) -> None:
    Path(path).write_text(json.dumps({str(k): v for k, v in sorted(counts.items(), key=lambda item: str(item[0]))}, indent=2) + "\n",
                          encoding="utf-8")
