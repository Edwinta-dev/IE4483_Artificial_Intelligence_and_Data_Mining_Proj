from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import yaml

from ie4483.vram import accumulation_steps, find_max_batch, write_probe_json
from scripts import probe_vram


def test_find_max_batch_uses_margin_and_searches_limit() -> None:
    attempts: list[int] = []

    def step(batch_size: int) -> None:
        attempts.append(batch_size)
        if batch_size > 7:
            raise torch.cuda.OutOfMemoryError("simulated capacity")

    assert find_max_batch(step, start=1, limit=20) == 5
    assert 7 in attempts
    assert all(size > 0 for size in attempts)


def test_find_max_batch_handles_start_failure_and_custom_margin() -> None:
    def always_fails(_batch_size: int) -> None:
        raise torch.cuda.OutOfMemoryError("simulated capacity")

    assert find_max_batch(always_fails, start=2, limit=8) == 0
    assert find_max_batch(lambda _size: None, start=3, limit=3, safety_margin=1.0) == 3


@pytest.mark.parametrize(("safe", "expected"), [(32, 1), (8, 4), (20, 2), (0, None)])
def test_accumulation_steps(safe: int, expected: int | None) -> None:
    assert accumulation_steps(safe) == expected


def test_write_probe_json_round_trips(tmp_path: Path) -> None:
    row = {
        "architecture": "resnet50",
        "freeze_mode": "full",
        "safe_batch_size": 4,
        "peak_memory_bytes": 1234,
        "gradient_accumulation_steps": 8,
    }
    output = write_probe_json(tmp_path / "nested" / "vram_probe.json", [row], 224)
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["image_size"] == 224
    assert payload["effective_batch_size"] == 32
    assert payload["safety_margin"] == 0.8
    assert payload["seed"] == 42
    assert payload["results"] == [row]


def test_write_configs_selects_configured_backbone_and_mode(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "model:\n  architecture: resnet50\n  freeze_backbone: false\n"
        "train:\n  batch_size: 32\n  gradient_accumulation_steps: 1\n",
        encoding="utf-8",
    )
    results = {("resnet50", "full"): {"safe_batch_size": 3, "gradient_accumulation_steps": 11}}
    probe_vram._write_configs([config_path], results)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert config["train"]["batch_size"] == 3
    assert config["train"]["gradient_accumulation_steps"] == 11


def test_cli_exits_without_cuda(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr("sys.argv", ["probe_vram.py"])
    probe_vram.main()
    assert "CUDA is unavailable" in capsys.readouterr().out
