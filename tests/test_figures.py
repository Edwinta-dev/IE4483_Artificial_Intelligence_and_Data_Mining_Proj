from __future__ import annotations

import pytest
import torch

from ie4483.config import Config, DataConfig, ModelConfig
from ie4483.figures import model_architecture_figure, write_model_card
from ie4483.model import build_model, model_summary


@pytest.mark.parametrize("architecture", ["resnet18", "resnet50"])
def test_report_model_artifacts_match_forward_shapes(tmp_path, architecture: str) -> None:
    config = Config(data=DataConfig(image_size=32), model=ModelConfig(architecture=architecture, pretrained=False))
    model = build_model(config).eval()
    output = model(torch.zeros(1, 3, 32, 32))
    summary = model_summary(model, 32)
    assert tuple(output.shape) == summary["output_shape"] == (1, 2)
    paths = model_architecture_figure(model, 32, tmp_path)
    card = write_model_card(model, config, {
        "optimizer": "adamw", "backbone_learning_rate": 0.0001, "scheduler": "cosine",
        "step_size": 5, "gamma": 0.1, "label_smoothing": 0.0,
    }, tmp_path)
    assert all(path.is_file() and path.stat().st_size for path in paths)
    text = card.read_text(encoding="utf-8")
    assert f"Input shape: `{summary['input_shape']}`" in text
    assert f"Output shape: `{summary['output_shape']}`" in text
    assert f"{summary['trainable_parameters']:,} trainable / {summary['total_parameters']:,} total" in text
