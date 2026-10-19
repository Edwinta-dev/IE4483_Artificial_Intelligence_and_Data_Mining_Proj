"""Create validation metrics and report figures for a completed training run."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from ie4483.cifar import CIFAR10Dataset, cifar_transform
from ie4483.data import ImageFolderDataset, make_dataloader
from ie4483.model import build_model
from ie4483.seed import seed_everything
from ie4483.train import load_training_config
from ie4483.transforms import build_eval_transform

_GALLERY_LIMIT = 5


def _scores(y_true: list[int], y_pred: list[int], loss: float, classes: list[str]) -> dict[str, Any]:
    per_class = {}
    for index, name in enumerate(classes):
        tp = sum(t == index and p == index for t, p in zip(y_true, y_pred))
        fp = sum(t != index and p == index for t, p in zip(y_true, y_pred))
        fn = sum(t == index and p != index for t, p in zip(y_true, y_pred))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[name] = {"accuracy": recall, "precision": precision, "recall": recall, "f1": f1}
    return {"accuracy": sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true) if y_true else 0.0,
            "per_class": per_class, "loss": loss, "count": len(y_true)}


def _plot_confusion(matrix: np.ndarray, classes: list[str], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(matrix, cmap="Blues")
    ax.set(xticks=range(len(classes)), yticks=range(len(classes)), xticklabels=classes,
           yticklabels=classes, xlabel="Predicted", ylabel="True", title="Validation confusion matrix")
    for row in range(len(classes)):
        for col in range(len(classes)):
            ax.text(col, row, str(matrix[row, col]), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _gallery(records: list[dict[str, Any]], kind: str, class_name: str, output: Path) -> None:
    if kind == "correct":
        candidates = [r for r in records if r["true"] == r["predicted"] and r["true_name"] == class_name]
        chosen = sorted(candidates, key=lambda r: r["confidence"], reverse=True)[:_GALLERY_LIMIT]
    elif kind == "wrong":
        candidates = [r for r in records if r["true"] != r["predicted"] and r["true_name"] == class_name]
        chosen = sorted(candidates, key=lambda r: r["confidence"], reverse=True)[:_GALLERY_LIMIT]
    else:
        candidates = [r for r in records if r["true_name"] == class_name]
        chosen = sorted(candidates, key=lambda r: abs(r["p_dog"] - 0.5))[:_GALLERY_LIMIT]
    fig, axes = plt.subplots(1, max(1, len(chosen)), figsize=(3 * max(1, len(chosen)), 3.6), squeeze=False)
    if not chosen:
        axes[0, 0].text(0.5, 0.5, "No matching validation images", ha="center", va="center")
        axes[0, 0].axis("off")
    for ax, record in zip(axes[0], chosen):
        with Image.open(record["path"]) as image:
            ax.imshow(image.convert("RGB"))
        ax.set_title(f"true={record['true_name']}  pred={record['predicted_name']}\np(dog)={record['p_dog']:.3f}")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output, dpi=140)
    plt.close(fig)


def evaluate(run: str | Path, results_path: str | Path = "results/latest.json", split: str = "val") -> dict[str, Any]:
    """Evaluate best.pt against validation, or CIFAR-10's official test split."""
    if split not in {"val", "test"}:
        raise ValueError("split must be val or test")
    run_dir = Path(run)
    config, extra = load_training_config(run_dir / "config.yaml")
    seed_everything(config.train.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config).to(device)
    checkpoint = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    if len(config.data.classes) == 10:
        dataset = CIFAR10Dataset(config.data_root, split, cifar_transform(config.data.image_size), config.train.seed)
    elif split == "test":
        raise ValueError("--split test is only supported for CIFAR-10")
    else:
        dataset = ImageFolderDataset(config.data_root, "val", config.data.classes, build_eval_transform(config),
                                     seed=config.train.seed)
    loader = make_dataloader(dataset, config.eval.batch_size, extra["num_workers"])
    truths: list[int] = []
    predictions: list[int] = []
    probabilities: list[float] = []
    loss_sum = 0.0
    with torch.inference_mode():
        for images, labels in loader:
            logits = model(images.to(device)).float()
            targets = labels.to(device)
            loss_sum += float(torch.nn.functional.cross_entropy(logits, targets, reduction="sum").item())
            probs = logits.softmax(dim=1).cpu()
            truths.extend(labels.tolist())
            predictions.extend(probs.argmax(dim=1).tolist())
            probabilities.extend(probs[:, 1].tolist())
    if not truths:
        raise ValueError(f"{split} split contains no readable images")
    classes = list(config.data.classes)
    records = []
    samples = dataset.samples if hasattr(dataset, "samples") else [("", int(label)) for label in dataset.labels]
    for (path, truth), pred, p_dog in zip(samples, predictions, probabilities):
        records.append({"path": str(path), "true": truth, "predicted": pred, "p_dog": p_dog,
                        "true_name": classes[truth], "predicted_name": classes[pred],
                        "confidence": p_dog if pred == 1 else 1 - p_dog})
    metrics = _scores(truths, predictions, loss_sum / len(truths), classes)
    with (run_dir / f"{split}_predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("path", "true", "predicted", "p_dog"))
        writer.writerows((r["path"], r["true"], r["predicted"], f"{r['p_dog']:.10g}") for r in records)
    matrix = np.zeros((len(classes), len(classes)), dtype=int)
    for truth, pred in zip(truths, predictions):
        matrix[truth, pred] += 1
    metrics["confusion_matrix"] = matrix.tolist()
    (run_dir / f"{split}_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    confusion_path = run_dir / ("confusion_matrix.png" if split == "val" else "test_confusion_matrix.png")
    _plot_confusion(matrix, classes, confusion_path)
    if split == "val" and hasattr(dataset, "samples"):
        for name in classes:
            for kind in ("correct", "wrong", "uncertain"):
                _gallery(records, kind, name, run_dir / f"gallery_{kind}_{name}.png")
    result_path = Path(results_path)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps({"metric": "accuracy", "value": metrics["accuracy"],
                                       "seed": config.train.seed, "split": split, "notes": run_dir.name},
                                      indent=2) + "\n", encoding="utf-8")
    return metrics
