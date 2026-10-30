"""ResNet transfer-learning models and compact model summaries."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, ResNet50_Weights, resnet18, resnet50

_BACKBONES = {
    "resnet18": (resnet18, ResNet18_Weights.DEFAULT, 512),
    "resnet50": (resnet50, ResNet50_Weights.DEFAULT, 2048),
}


def _get(config: Any, key: str, default: Any = None) -> Any:
    if isinstance(config, Mapping):
        return config.get(key, default)
    return getattr(config, key, default)


def _model_config(config: Any) -> Any:
    return _get(config, "model", config)


class _ResNetWithHead(nn.Module):
    def __init__(self, backbone: nn.Module, feature_dim: int, head: nn.Module, freeze_mode: str):
        super().__init__()
        self.backbone = backbone
        self.feature_dim = feature_dim
        self.head = head
        self.freeze_mode = freeze_mode

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.backbone.conv1(inputs)
        features = self.backbone.bn1(features)
        features = self.backbone.relu(features)
        features = self.backbone.maxpool(features)
        features = self.backbone.layer1(features)
        features = self.backbone.layer2(features)
        features = self.backbone.layer3(features)
        features = self.backbone.layer4(features)
        features = self.backbone.avgpool(features)
        return self.head(torch.flatten(features, 1))

    def train(self, mode: bool = True) -> _ResNetWithHead:
        super().train(mode)
        if mode and self.freeze_mode in {"frozen", "last_block"}:
            for name, module in self.backbone.named_modules():
                frozen_part = self.freeze_mode == "frozen" or not name.startswith("layer4.")
                if frozen_part and isinstance(module, nn.modules.batchnorm._BatchNorm):
                    module.eval()
        return self


def build_model(cfg: Any) -> nn.Module:
    """Build a ResNet-18/50 and custom classifier head from model config values.

    ``cfg`` may be the project's Config, a model config object, or a mapping. In
    addition to the existing fields, optional values are ``num_classes``
    (default 2), ``dropout`` (default 0), ``hidden_dim`` (default None), and
    ``freeze_mode`` (``frozen``, ``last_block``, or ``full``).
    """
    config = _model_config(cfg)
    architecture = _get(config, "architecture", "resnet18")
    if architecture not in _BACKBONES:
        raise ValueError(f"unsupported architecture {architecture!r}; choose resnet18 or resnet50")
    constructor, weights_enum, feature_dim = _BACKBONES[architecture]
    pretrained = _get(config, "pretrained", False)
    weights = weights_enum.DEFAULT if pretrained else None
    if pretrained:
        filename = Path(weights_enum.DEFAULT.url).name
        checkpoint = Path(torch.hub.get_dir()) / "checkpoints" / filename
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"Pretrained weights for {architecture} are not cached at {checkpoint}; "
                "run scripts/fetch_weights.py to cache them. Downloads are disabled in build_model."
            )
    backbone = constructor(weights=weights)

    num_classes = _get(config, "num_classes", 2)
    dropout = _get(config, "dropout", 0.0)
    hidden_dim = _get(config, "hidden_dim", None)
    if isinstance(num_classes, bool) or not isinstance(num_classes, int) or num_classes < 1:
        raise ValueError("num_classes must be a positive integer")
    if isinstance(dropout, bool) or not isinstance(dropout, (int, float)) or not 0 <= dropout < 1:
        raise ValueError("dropout must be in [0, 1)")
    layers: list[nn.Module] = []
    if dropout:
        layers.append(nn.Dropout(float(dropout)))
    if hidden_dim is not None:
        if isinstance(hidden_dim, bool) or not isinstance(hidden_dim, int) or hidden_dim < 1:
            raise ValueError("hidden_dim must be a positive integer or None")
        layers.extend((nn.Linear(feature_dim, hidden_dim), nn.ReLU(inplace=True)))
        if dropout:
            layers.append(nn.Dropout(float(dropout)))
        input_dim = hidden_dim
    else:
        input_dim = feature_dim
    layers.append(nn.Linear(input_dim, num_classes))
    backbone.fc = nn.Identity()

    freeze_mode = _get(config, "freeze_mode", None)
    if freeze_mode is None:
        freeze_mode = "frozen" if _get(config, "freeze_backbone", True) else "full"
    if freeze_mode not in {"frozen", "last_block", "full"}:
        raise ValueError("freeze_mode must be 'frozen', 'last_block', or 'full'")
    model = _ResNetWithHead(backbone, feature_dim, nn.Sequential(*layers), freeze_mode)
    if freeze_mode == "frozen":
        for parameter in model.backbone.parameters():
            parameter.requires_grad = False
    elif freeze_mode == "last_block":
        for name, parameter in model.backbone.named_parameters():
            parameter.requires_grad = name.startswith("layer4.")
    return model


def model_summary(model: nn.Module, image_size: int | tuple[int, int]) -> dict[str, Any]:
    """Return input/output shapes, head feature width and parameter counts."""
    height, width = (image_size, image_size) if isinstance(image_size, int) else image_size
    if height < 1 or width < 1:
        raise ValueError("image_size dimensions must be positive")
    parameter = next(model.parameters(), None)
    device = parameter.device if parameter is not None else torch.device("cpu")
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            output = model(torch.zeros((1, 3, height, width), device=device))
    finally:
        model.train(was_training)
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {
        "input_shape": (1, 3, height, width),
        "output_shape": tuple(output.shape),
        "feature_dim": getattr(model, "feature_dim", None),
        "trainable_parameters": trainable,
        "total_parameters": total,
    }
