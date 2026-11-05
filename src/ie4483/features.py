"""Deterministic frozen-backbone feature extraction and cache helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from ie4483.config import get_setting


def feature_cache_key(config: Any, manifest: list[tuple[str, int]]) -> str:
    """Hash all inputs that can change pooled backbone features."""
    payload = {
        "backbone": get_setting(config, "model", "architecture", default="resnet18"),
        "pretrained": get_setting(config, "model", "pretrained", default=False),
        "image_size": get_setting(config, "data", "image_size", default=224),
        "eval_transform": "Resize-CenterCrop-ImageNetNormalize",
        "manifest": manifest,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def extract_features(
    model: torch.nn.Module, loader: DataLoader, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run a frozen ResNet once and retain pooled features and labels on CPU."""
    model.eval()
    features, labels = [], []
    with torch.inference_mode():
        for images, target in loader:
            images = images.to(device)
            output = model.forward_features(images) if hasattr(model, "forward_features") else model(images)
            features.append(output.detach().cpu())
            labels.append(torch.as_tensor(target).cpu())
    width = getattr(model, "feature_dim", 0)
    return (
        torch.cat(features) if features else torch.empty((0, width)),
        torch.cat(labels) if labels else torch.empty((0,), dtype=torch.long),
    )


def load_or_create_cache(path: str | Path, builder):
    """Load cached tensors, or atomically persist the result from builder()."""
    destination = Path(path)
    if destination.exists():
        return torch.load(destination, map_location="cpu", weights_only=True), True
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = builder()
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)
    return payload, False
