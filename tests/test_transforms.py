from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

from ie4483.config import load_config
from ie4483.transforms import (
    build_eval_transform,
    build_train_transform,
    describe_transforms,
)

BASE = Path(__file__).parents[1] / "configs" / "base.yaml"


def test_eval_and_disabled_train_shape_dtype_and_determinism():
    cfg = load_config(BASE, ["data.image_size=16"])
    image = Image.fromarray(np.full((22, 30, 3), 128, dtype=np.uint8))
    eval_transform = build_eval_transform(cfg)
    train_transform = build_train_transform(cfg)
    first, second = train_transform(image), train_transform(image)
    assert first.shape == (3, 16, 16)
    assert first.dtype == torch.float32
    assert torch.equal(first, second)
    assert torch.equal(first, eval_transform(image))
    expected = (128 / 255 - torch.tensor([0.485, 0.456, 0.406])) / torch.tensor([0.229, 0.224, 0.225])
    assert torch.allclose(first[:, 8, 8], expected, atol=1e-5)


def test_each_augmentation_switch_changes_pipeline_description():
    base = yaml.safe_load(BASE.read_text(encoding="utf-8"))
    baseline = describe_transforms(base)
    options = [
        {"random_resized_crop": True},
        {"horizontal_flip": True},
        {"rotation_degrees": 15.0},
        {"color_jitter": True},
    ]
    for option in options:
        config = {**base, "augment": {**base["augment"], **option}}
        description = describe_transforms(config)
        assert description != baseline
        assert build_train_transform(config) != build_eval_transform(config)


def test_preview_command_writes_png(tmp_path):
    image_path = tmp_path / "input.png"
    Image.fromarray(np.full((24, 32, 3), 150, dtype=np.uint8)).save(image_path)
    output = tmp_path / "preview.png"
    subprocess.run([
        sys.executable, str(Path(__file__).parents[1] / "scripts" / "preview_augment.py"),
        "--image", str(image_path), "--n", "3", "--config", str(BASE), "--output", str(output),
    ], check=True)
    with Image.open(output) as preview:
        assert preview.format == "PNG"
        assert preview.width > 0 and preview.height > 0
