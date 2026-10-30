"""Validation and CSV output for unlabeled test predictions."""

from __future__ import annotations

import csv
from collections.abc import Sequence
from pathlib import Path


def validate_predictions(
    image_ids: Sequence[int], labels: Sequence[int], test_count: int, classes: Sequence[str]
) -> None:
    """Reject incomplete or semantically invalid predictions before writing."""
    if len(image_ids) != test_count or len(labels) != test_count:
        raise ValueError(f"prediction row count must equal test image count ({test_count})")
    if len(set(image_ids)) != len(image_ids):
        raise ValueError("test image IDs must be unique")
    if any(label not in (0, 1) for label in labels):
        raise ValueError("submission labels must be 0 or 1")
    if len(classes) != 2 or classes[1] != "dog":
        raise ValueError("run class order must place dog at index 1")


def write_prediction_csvs(
    output_dir: str | Path,
    image_ids: Sequence[int],
    probabilities: Sequence[Sequence[float]],
    classes: Sequence[str],
    test_count: int,
) -> None:
    """Write submission.csv and probability-bearing test_predictions.csv."""
    if len(probabilities) != test_count or any(len(row) != 2 for row in probabilities):
        raise ValueError(f"prediction row count must equal test image count ({test_count})")
    labels = [int(float(row[1]) >= float(row[0])) for row in probabilities]
    validate_predictions(image_ids, labels, test_count, classes)
    ordered = sorted(zip(image_ids, labels, probabilities), key=lambda row: row[0])
    output = Path(output_dir)
    with (output / "submission.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("id", "label"))
        writer.writerows((image_id, label) for image_id, label, _ in ordered)
    with (output / "test_predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("id", "p_cat", "p_dog"))
        writer.writerows((image_id, f"{float(row[0]):.10g}", f"{float(row[1]):.10g}")
                         for image_id, _, row in ordered)
