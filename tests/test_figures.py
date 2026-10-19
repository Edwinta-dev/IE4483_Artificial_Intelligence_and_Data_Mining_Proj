from __future__ import annotations

import pytest
import torch

from ie4483.config import Config, DataConfig, ModelConfig
from ie4483.figures import model_architecture_figure, training_curves, write_model_card
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
    assert "Selection: best.pt by validation accuracy; early stopping by validation loss." in text


def test_training_curves_mark_epoch_with_highest_validation_accuracy(tmp_path, monkeypatch) -> None:
    from ie4483 import figures

    rows = [
        {"epoch": "1", "train_loss": "1", "train_acc": ".5", "val_loss": "1.1", "val_acc": ".4"},
        {"epoch": "3", "train_loss": ".5", "train_acc": ".8", "val_loss": ".6", "val_acc": ".9"},
        {"epoch": "4", "train_loss": ".4", "train_acc": ".9", "val_loss": ".7", "val_acc": ".8"},
    ]
    captured = {}

    def capture(fig, stem, output_dir):
        captured["fig"] = fig
        return (), ()

    monkeypatch.setattr(figures, "_save_report_figure", capture)
    training_curves(rows, "fake run", tmp_path)
    assert [list(line.get_xdata()) for line in captured["fig"].axes[0].lines[-1:]] == [[3, 3]]
    assert "Best epoch 3" in captured["fig"].axes[0].lines[-1].get_label()
    import matplotlib.pyplot as plt
    plt.close(captured["fig"])


def test_run_report_tables_and_artifacts_include_proxy_label(tmp_path) -> None:
    import csv
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from report_figures import report_figures

    data = tmp_path / "proxy-data"
    data.mkdir()
    (data / "PROXY_DATASET.txt").write_text("proxy", encoding="utf-8")
    run = tmp_path / "run-a"
    run.mkdir()
    (run / "config.yaml").write_text(
        f"data:\n  root: {data.as_posix()}\n  image_size: 32\nmodel:\n  architecture: resnet18\n  pretrained: false\n  freeze_backbone: true\ntrain:\n  learning_rate: 0.01\n",
        encoding="utf-8",
    )
    (run / "transforms.txt").write_text("flip", encoding="utf-8")
    (run / "metrics.csv").write_text(
        "epoch,train_loss,train_acc,val_loss,val_acc,lr,seconds\n"
        "1,1,0.5,1.2,0.4,0.01,2\n2,0.5,0.8,0.6,0.9,0.01,4\n", encoding="utf-8")
    output = tmp_path / "report"
    summaries = report_figures([run], output)
    with (output / "comparison.csv").open(newline="", encoding="utf-8") as stream:
        row = next(csv.DictReader(stream))
    assert row["backbone"] == "resnet18"
    assert row["freeze_mode"] == "frozen"
    assert float(row["learning_rate"]) == 0.01
    assert row["augmentation"] == "flip"
    assert int(row["epochs_run"]) == 2
    assert float(row["best_val_accuracy"]) == 0.9
    assert float(row["time_per_epoch"]) == 3
    assert summaries[0]["proxy"] is True
    assert "PROXY" in (output / "comparison.md").read_text(encoding="utf-8")
    assert all((run / f"training_curves.{ext}").is_file() for ext in ("png", "pdf"))
    assert all((output / f"comparison.{ext}").is_file() for ext in ("png", "pdf"))
    assert (output / "comparison.csv").is_file()
