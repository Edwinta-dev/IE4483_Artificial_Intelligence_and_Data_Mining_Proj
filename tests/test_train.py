from __future__ import annotations

import csv
from types import SimpleNamespace

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
