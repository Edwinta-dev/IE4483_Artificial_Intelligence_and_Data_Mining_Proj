from __future__ import annotations

import csv
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from ie4483.config import load_config
from ie4483.features import extract_features, feature_cache_key, load_or_create_cache
from ie4483.model import build_model
from ie4483.train import train


def _images(root, split):
    for label, cls in enumerate(("cat", "dog")):
        folder = root / split / cls
        folder.mkdir(parents=True)
        for i in range(2):
            Image.new("RGB", (36, 36), (label * 200, i * 30, 40)).save(folder / f"{i}.png")


def test_two_epoch_training_writes_artifacts_and_is_seeded(tmp_path, monkeypatch):
    import ie4483.train as training

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    _images(tmp_path, "train")
    _images(tmp_path, "val")
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "data:\n  root: '" + str(tmp_path).replace("\\", "/") + "'\n  image_size: 32\n"
        "model:\n  architecture: resnet18\n  pretrained: false\n  freeze_backbone: true\n"
        "train:\n  batch_size: 2\n  seed: 9\n  epochs: 2\n  learning_rate: 0.001\n  mixed_precision: true\n  gradient_accumulation_steps: 1\n"
        "eval:\n  batch_size: 2\n",
        encoding="utf-8",
    )
    config = load_config(config_file)
    extra = {"patience": 0, "feature_cache": "auto"}
    extraction_calls = 0
    original_extract = training.extract_features

    def counted_extract(*args, **kwargs):
        nonlocal extraction_calls
        extraction_calls += 1
        return original_extract(*args, **kwargs)

    monkeypatch.setattr(training, "extract_features", counted_extract)
    run_a = train(config, extra, tmp_path / "runs")
    run_b = train(config, extra, tmp_path / "runs")
    assert extraction_calls == 2
    for run in (run_a, run_b):
        assert {"config.yaml", "transforms.txt", "manifest.csv", "metrics.csv", "best.pt"} <= {
            path.name for path in run.iterdir()
        }
        rows = list(csv.DictReader((run / "metrics.csv").open(encoding="utf-8")))
        assert rows
        assert all(torch.isfinite(torch.tensor(float(row["train_loss"]))) for row in rows)
    rows_a = list(csv.DictReader((run_a / "metrics.csv").open(encoding="utf-8")))
    rows_b = list(csv.DictReader((run_b / "metrics.csv").open(encoding="utf-8")))
    assert [(r["train_loss"], r["val_loss"], r["train_acc"], r["val_acc"]) for r in rows_a] == [
        (r["train_loss"], r["val_loss"], r["train_acc"], r["val_acc"]) for r in rows_b
    ]


def test_feature_cache_key_and_reuse(tmp_path):
    base = {"model": {"architecture": "resnet18", "pretrained": False}, "data": {"image_size": 32}}
    key = feature_cache_key(base, [("a.png", 0)])
    assert key != feature_cache_key(base, [("b.png", 0)])
    assert key != feature_cache_key({**base, "data": {"image_size": 64}}, [("a.png", 0)])
    count = 0

    def build():
        nonlocal count
        count += 1
        return {"x": torch.ones(2)}

    cache_path = tmp_path / "cache.pt"
    first, reused = load_or_create_cache(cache_path, build)
    second, reused_again = load_or_create_cache(cache_path, build)
    assert not reused and reused_again and count == 1
    torch.testing.assert_close(first["x"], second["x"])


def test_cached_features_preserve_head_outputs(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    model = build_model({"architecture": "resnet18", "pretrained": False, "freeze_backbone": True})
    model.eval()
    images = torch.rand(2, 3, 32, 32)
    from torch.utils.data import DataLoader, TensorDataset

    features, labels = extract_features(model, DataLoader(TensorDataset(images, torch.tensor([0, 1]))), torch.device("cpu"))
    torch.testing.assert_close(model(images), model.head(features), rtol=1e-5, atol=1e-6)
    assert labels.tolist() == [0, 1]


def test_patience_zero_stops_on_first_validation_loss_regression(tmp_path, monkeypatch):
    import ie4483.train as training

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.head = torch.nn.Linear(2, 2)

        def forward(self, x):
            return self.head(x)

    calls = 0

    def epoch(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls % 2:
            return 1.0, 0.5
        return (0.5 if calls == 2 else 0.6), 0.5

    monkeypatch.setattr(training, "_epoch", epoch)
    cfg = SimpleNamespace(
        train=SimpleNamespace(batch_size=2, learning_rate=0.01, gradient_accumulation_steps=1,
                              mixed_precision=False, epochs=4),
        eval=SimpleNamespace(batch_size=2),
    )
    data = (torch.ones(2, 2), torch.tensor([0, 1]))
    training._fit(Tiny(), torch.utils.data.DataLoader(torch.utils.data.TensorDataset(*data), batch_size=2),
                  torch.utils.data.DataLoader(torch.utils.data.TensorDataset(*data), batch_size=2), cfg,
                  {"optimizer": "adamw", "weight_decay": 0.0, "scheduler": "none", "patience": 0,
                   "class_weights": None, "label_smoothing": 0.0, "warmup_epochs": 0},
                  tmp_path, torch.device("cpu"))
    rows = list(csv.DictReader((tmp_path / "metrics.csv").open(encoding="utf-8")))
    assert len(rows) == 2


def test_cosine_schedule_reaches_minimum_after_final_warmup_adjusted_epoch(tmp_path, monkeypatch):
    import ie4483.train as training

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.head = torch.nn.Linear(2, 2)

        def forward(self, x):
            return self.head(x)

    monkeypatch.setattr(training, "_epoch", lambda *args, **kwargs: (1.0, 0.5))
    cfg = SimpleNamespace(
        train=SimpleNamespace(batch_size=2, learning_rate=0.1, gradient_accumulation_steps=1,
                              mixed_precision=False, epochs=4),
        eval=SimpleNamespace(batch_size=2),
    )
    data = (torch.ones(2, 2), torch.tensor([0, 1]))
    training._fit(Tiny(), torch.utils.data.DataLoader(torch.utils.data.TensorDataset(*data), batch_size=2),
                  torch.utils.data.DataLoader(torch.utils.data.TensorDataset(*data), batch_size=2), cfg,
                  {"optimizer": "adamw", "weight_decay": 0.0, "scheduler": "cosine", "patience": 10,
                   "class_weights": None, "label_smoothing": 0.0, "warmup_epochs": 1},
                  tmp_path, torch.device("cpu"))
    checkpoint = torch.load(tmp_path / "last.pt", map_location="cpu", weights_only=False)
    assert checkpoint["scheduler"]["T_max"] == 3
    assert checkpoint["scheduler"]["last_epoch"] == 3
    assert checkpoint["optimizer"]["param_groups"][0]["lr"] == pytest.approx(0.0)


def test_weighted_sampler_and_class_weights_work_on_image_folders(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    _images(tmp_path, "train")
    _images(tmp_path, "val")
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "data:\n  root: '" + str(tmp_path).replace("\\", "/") + "'\n  image_size: 32\n"
        "model:\n  architecture: resnet18\n  pretrained: false\n  freeze_backbone: true\n"
        "train:\n  batch_size: 2\n  seed: 9\n  epochs: 1\n  mixed_precision: false\n"
        "eval:\n  batch_size: 2\n",
        encoding="utf-8",
    )
    extra = {"feature_cache": "off", "weighted_sampler": True, "class_weighted": True}
    run = train(load_config(config_file), extra, tmp_path / "runs")
    assert (run / "best.pt").is_file()


def test_interrupted_training_resumes_with_matching_weights_and_metrics(tmp_path, monkeypatch):
    import ie4483.train as training

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    _images(tmp_path, "train")
    _images(tmp_path, "val")
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "data:\n  root: '" + str(tmp_path).replace("\\", "/") + "'\n  image_size: 32\n"
        "model:\n  architecture: resnet18\n  pretrained: false\n  freeze_backbone: true\n"
        "train:\n  batch_size: 2\n  seed: 17\n  epochs: 3\n  learning_rate: 0.001\n  mixed_precision: false\n  gradient_accumulation_steps: 1\n"
        "eval:\n  batch_size: 2\n",
        encoding="utf-8",
    )
    config = load_config(config_file)
    extra = {"feature_cache": "off", "patience": 10}
    straight = train(config, extra, tmp_path / "straight")

    original_save = training._atomic_save
    last_saves = 0

    def stop_after_epoch_one(payload, path):
        nonlocal last_saves
        original_save(payload, path)
        if path.name == "last.pt":
            last_saves += 1
            if last_saves == 1:
                raise RuntimeError("simulated interruption")

    monkeypatch.setattr(training, "_atomic_save", stop_after_epoch_one)
    try:
        train(config, extra, tmp_path / "interrupted")
    except RuntimeError as error:
        assert str(error) == "simulated interruption"
    else:
        raise AssertionError("training did not stop at the simulated interruption")
    partial = next((tmp_path / "interrupted").iterdir())
    monkeypatch.setattr(training, "_atomic_save", original_save)
    resumed = train(config, extra, tmp_path / "interrupted", resume=partial)

    straight_rows = list(csv.DictReader((straight / "metrics.csv").open(encoding="utf-8")))
    resumed_rows = list(csv.DictReader((resumed / "metrics.csv").open(encoding="utf-8")))
    assert [row["epoch"] for row in resumed_rows] == ["1", "2", "3"]
    for expected, actual in zip(straight_rows, resumed_rows):
        for key in ("train_loss", "train_acc", "val_loss", "val_acc", "lr"):
            assert float(actual[key]) == pytest.approx(float(expected[key]), abs=1e-7)
    torch.testing.assert_close(torch.load(straight / "best.pt", weights_only=False)["model"],
                               torch.load(resumed / "best.pt", weights_only=False)["model"], rtol=1e-6, atol=1e-7)

    changed = load_config(config_file)
    changed.train.learning_rate *= 2
    try:
        train(changed, extra, tmp_path / "interrupted", resume=partial)
    except ValueError as error:
        assert "configuration differs" in str(error)
    else:
        raise AssertionError("resume accepted a mismatched configuration")


def test_repeatability_check_compares_same_seed_and_records_switch(tmp_path, monkeypatch):
    import yaml

    from scripts.check_repeatable import check_repeatable

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    _images(tmp_path, "train")
    _images(tmp_path, "val")
    config_file = tmp_path / "repeat.yaml"
    config_file.write_text(
        "data:\n  root: '" + str(tmp_path).replace("\\", "/") + "'\n  image_size: 32\n"
        "model:\n  architecture: resnet18\n  pretrained: false\n  freeze_backbone: true\n"
        "train:\n  batch_size: 2\n  seed: 19\n  epochs: 1\n  mixed_precision: false\n  deterministic: true\n"
        "eval:\n  batch_size: 2\n",
        encoding="utf-8",
    )
    same, different = check_repeatable(config_file, 1, tmp_path / "repeat-runs", ['train.feature_cache="off"'])
    assert same == 0.0
    assert different > 0.0
    configs = [yaml.safe_load((tmp_path / "repeat-runs" / f"run-{i}" / next(
        (tmp_path / "repeat-runs" / f"run-{i}").iterdir()
    ) / "config.yaml").read_text(encoding="utf-8")) for i in range(3)]
    assert [item["train"]["seed"] for item in configs] == [19, 19, 20]
    assert all(item["train"]["deterministic"] is True for item in configs)


def test_weighted_sampler_applies_on_frozen_feature_cache_path(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    _images(tmp_path, "train")
    _images(tmp_path, "val")
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "data:\n  root: '" + str(tmp_path).replace("\\", "/") + "'\n  image_size: 32\n"
        "model:\n  architecture: resnet18\n  pretrained: false\n  freeze_backbone: true\n"
        "train:\n  batch_size: 2\n  seed: 9\n  epochs: 1\n  mixed_precision: false\n"
        "eval:\n  batch_size: 2\n",
        encoding="utf-8",
    )
    import ie4483.train as train_module
    seen = []
    original = train_module._fit
    monkeypatch.setattr(train_module, "_fit", lambda model, loader, *a, **k: seen.append(loader) or original(model, loader, *a, **k))
    train(load_config(config_file), {"feature_cache": "auto", "weighted_sampler": True}, tmp_path / "runs")
    assert isinstance(seen[0].sampler, torch.utils.data.WeightedRandomSampler)
