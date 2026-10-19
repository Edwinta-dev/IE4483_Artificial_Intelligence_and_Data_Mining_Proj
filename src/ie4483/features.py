"""Deterministic frozen-backbone feature extraction and cache helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader


def feature_cache_key(config: Any, manifest: list[tuple[str, int]]) -> str:
    """Hash all inputs that can change pooled backbone features."""
    def get(section: str, key: str, default: Any = None) -> Any:
        part = config.get(section, {}) if isinstance(config, dict) else getattr(config, section, None)
        return part.get(key, default) if isinstance(part, dict) else getattr(part, key, default)

    payload = {
        "backbone": get("model", "architecture", "resnet18"),
        "pretrained": get("model", "pretrained", False),
        "image_size": get("data", "image_size", 224),
        "eval_transform": "Resize-CenterCrop-ImageNetNormalize",
        "manifest": manifest,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def extract_features(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Run a frozen ResNet once and retain pooled features and labels on CPU."""
    model.eval()
    features, labels = [], []
    with torch.inference_mode():
        for images, target in loader:
            images = images.to(device)
            backbone = getattr(model, "backbone", None)
            if backbone is not None:
                output = backbone.conv1(images)
                output = backbone.bn1(output)
                output = backbone.relu(output)
                output = backbone.maxpool(output)
                for layer in (backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4):
                    output = layer(output)
                output = torch.flatten(backbone.avgpool(output), 1)
            else:
                output = model(images)
            features.append(output.detach().cpu())
            labels.append(torch.as_tensor(target).cpu())
    width = getattr(model, "feature_dim", 0)
    return (torch.cat(features) if features else torch.empty((0, width)),
            torch.cat(labels) if labels else torch.empty((0,), dtype=torch.long))


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
