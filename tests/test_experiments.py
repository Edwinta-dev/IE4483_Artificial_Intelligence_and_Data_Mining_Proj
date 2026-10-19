from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

import torch
import yaml
from PIL import Image

from ie4483 import experiments


def _dataset(root):
    for split in ("train", "val"):
        for label, name in enumerate(("cat", "dog")):
            folder = root / split / name
            folder.mkdir(parents=True)
            for index in range(2):
                Image.new("RGB", (32, 32), (label * 180, index * 45, 20)).save(folder / f"{index}.png")


def test_two_by_one_sweep_aggregates_runs_resumes_and_writes_chart(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert experiments._cell_name(("a/b",)) != experiments._cell_name(("a-b",))
    data = tmp_path / "images"
    _dataset(data)
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({
        "data": {"root": str(data), "image_size": 32},
        "model": {"architecture": "resnet18", "pretrained": False, "freeze_backbone": True},
        "train": {"batch_size": 2, "epochs": 1, "learning_rate": 0.001,
                  "mixed_precision": False, "gradient_accumulation_steps": 1},
        "eval": {"batch_size": 2},
    }), encoding="utf-8")
    sweep_file = tmp_path / "sweep.yaml"
    sweep_file.write_text(yaml.safe_dump({
        "config": str(config), "seeds": [0, 1],
        "overrides": ["train.batch_size=1"],
        "parameters": {"train.learning_rate": [0.001, 0.002]},
    }), encoding="utf-8")
    output = tmp_path / "results"
    calls = 0
    original = experiments._run_one

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(experiments, "_run_one", counted)
    rows = experiments.sweep(sweep_file, output)
    assert calls == 4
    assert len(rows) == 4
    table = list(csv.DictReader((output / "summary.csv").open(encoding="utf-8")))
    values_by_cell = {}
    for row in rows:
        metrics = json.loads((Path(row["run"]) / "val_metrics.json").read_text(encoding="utf-8"))
        assert float(row["val_accuracy"]) == metrics["accuracy"]
        run_config = yaml.safe_load((Path(row["run"]) / "config.yaml").read_text(encoding="utf-8"))
        assert run_config["train"]["batch_size"] == 1
        values_by_cell.setdefault(float(row["train.learning_rate"]), []).append(metrics["accuracy"])
    assert len(values_by_cell) == 2
    assert len(table) == 2
    for row in table:
        expected = values_by_cell[float(row["train.learning_rate"])]
        assert float(row["mean"]) == statistics.mean(expected)
        assert float(row["std"]) == statistics.stdev(expected)
    summary = (output / "summary.md").read_text(encoding="utf-8")
    for values in values_by_cell.values():
        assert f"{statistics.mean(values):.6f} ± {statistics.stdev(values):.6f}" in summary
    assert (output / "sweep.png").is_file()
    partial_run = Path(rows[0]["run"])
    (partial_run / "val_metrics.json").unlink()
    again = experiments.sweep(sweep_file, output)
    assert calls == 5
    assert any(Path(row["run"]) == partial_run for row in again)
    assert [(r["train.learning_rate"], r["seed"], r["val_accuracy"]) for r in again] == [
        (r["train.learning_rate"], r["seed"], r["val_accuracy"]) for r in rows
    ]


def test_seed_summary_includes_per_seed_mean_and_std(tmp_path, monkeypatch):
    def fake_run(_config, seed, root, _overrides, _resume=None):
        run_dir = root / f"run-{seed}"
        run_dir.mkdir(parents=True)
        accuracy = {3: 0.25, 4: 0.75}[seed]
        (run_dir / "val_metrics.json").write_text(json.dumps({"accuracy": accuracy}), encoding="utf-8")
        return run_dir, accuracy

    monkeypatch.setattr(experiments, "_run_one", fake_run)
    rows = experiments.run_seeds("unused.yaml", [3, 4], tmp_path / "seeds")
    assert [row["val_accuracy"] for row in rows] == [0.25, 0.75]
    with (tmp_path / "seeds" / "summary.csv").open(encoding="utf-8") as stream:
        table = list(csv.DictReader(stream))
    assert [float(row["mean"]) for row in table] == [0.5, 0.5]
    assert [float(row["std"]) for row in table] == [statistics.stdev([0.25, 0.75])] * 2
    assert "Mean: 0.500000; standard deviation: 0.353553" in (tmp_path / "seeds" / "summary.md").read_text(encoding="utf-8")
