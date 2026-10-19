import csv
import json

import torch
from PIL import Image

from ie4483.config import load_config
from ie4483.evaluate import evaluate
from ie4483.train import train


def _make_split(root, split):
    for label, class_name in enumerate(("cat", "dog")):
        folder = root / split / class_name
        folder.mkdir(parents=True)
        for index in range(2):
            Image.new("RGB", (36, 36), (label * 210, index * 50, 30)).save(folder / f"{index}.png")


def test_evaluate_writes_validation_artifacts_and_result_contract(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    _make_split(tmp_path, "train")
    _make_split(tmp_path, "val")
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        f"data:\n  root: '{str(tmp_path).replace(chr(92), '/')}'\n  image_size: 32\n"
        "model:\n  architecture: resnet18\n  pretrained: false\n  freeze_backbone: true\n"
        "train:\n  batch_size: 2\n  seed: 13\n  epochs: 1\n  learning_rate: 0.001\n"
        "eval:\n  batch_size: 2\n",
        encoding="utf-8",
    )
    config = load_config(config_file)
    run = train(config, {"feature_cache": "off", "patience": 2}, tmp_path / "runs")
    results = tmp_path / "results" / "latest.json"
    metrics = evaluate(run, results)

    expected = {"val_metrics.json", "val_predictions.csv", "confusion_matrix.png"}
    expected.update(f"gallery_{kind}_{name}.png" for kind in ("correct", "wrong", "uncertain")
                    for name in ("cat", "dog"))
    assert expected <= {path.name for path in run.iterdir()}
    with (run / "val_predictions.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    accuracy = sum(int(row["true"]) == int(row["predicted"]) for row in rows) / len(rows)
    assert metrics["accuracy"] == accuracy
    assert json.loads((run / "val_metrics.json").read_text(encoding="utf-8")) == metrics
    assert json.loads(results.read_text(encoding="utf-8")) == {
        "metric": "accuracy", "value": accuracy, "seed": 13, "split": "val", "notes": run.name,
    }
