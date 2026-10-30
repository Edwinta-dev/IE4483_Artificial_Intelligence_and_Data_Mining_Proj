from __future__ import annotations

import pytest
import torch
from torch import nn

from ie4483.model import build_model, model_summary


@pytest.mark.parametrize("architecture", ["resnet18", "resnet50"])
@pytest.mark.parametrize("num_classes", [2, 10])
def test_output_shape(architecture: str, num_classes: int) -> None:
    model = build_model({"architecture": architecture, "pretrained": False, "num_classes": num_classes})
    model.eval()
    assert model(torch.zeros(1, 3, 32, 32)).shape == (1, num_classes)
    expected_dim = 512 if architecture == "resnet18" else 2048
    summary = model_summary(model, 32)
    assert summary["feature_dim"] == expected_dim
    assert summary["input_shape"] == (1, 3, 32, 32)
    assert summary["output_shape"] == (1, num_classes)


@pytest.mark.parametrize("architecture", ["resnet18", "resnet50"])
@pytest.mark.parametrize("freeze_mode", ["frozen", "last_block", "full"])
def test_trainable_parameters_and_frozen_batchnorm(architecture: str, freeze_mode: str) -> None:
    model = build_model(
        {"architecture": architecture, "pretrained": False, "freeze_mode": freeze_mode, "num_classes": 2}
    )
    head_count = sum(parameter.numel() for parameter in model.head.parameters())
    layer4_count = sum(parameter.numel() for parameter in model.backbone.layer4.parameters())
    backbone_count = sum(parameter.numel() for parameter in model.backbone.parameters())
    expected = {
        "frozen": head_count,
        "last_block": head_count + layer4_count,
        "full": head_count + backbone_count,
    }[freeze_mode]
    assert sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad) == expected
    model.train()
    frozen_batchnorms = [
        module
        for name, module in model.backbone.named_modules()
        if isinstance(module, nn.modules.batchnorm._BatchNorm)
        and (freeze_mode == "frozen" or (freeze_mode == "last_block" and not name.startswith("layer4.")))
    ]
    if freeze_mode == "frozen":
        assert frozen_batchnorms
        assert all(not module.training for module in frozen_batchnorms)
    elif freeze_mode == "last_block":
        assert frozen_batchnorms
        assert all(not module.training for module in frozen_batchnorms)
        assert model.backbone.layer4[0].bn1.training
    else:
        assert all(module.training for module in model.modules() if isinstance(module, nn.modules.batchnorm._BatchNorm))


def test_missing_pretrained_weights_names_fetch_script(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setattr(torch.hub, "get_dir", lambda: str(tmp_path))
    with pytest.raises(FileNotFoundError, match="scripts/fetch_weights.py"):
        build_model({"architecture": "resnet18", "pretrained": True})


def test_custom_head_options() -> None:
    model = build_model(
        {"architecture": "resnet18", "pretrained": False, "num_classes": 10, "dropout": 0.25, "hidden_dim": 16}
    )
    assert isinstance(model.head[0], nn.Dropout)
    assert isinstance(model.head[1], nn.Linear) and model.head[1].out_features == 16
    assert isinstance(model.head[-1], nn.Linear) and model.head[-1].out_features == 10
