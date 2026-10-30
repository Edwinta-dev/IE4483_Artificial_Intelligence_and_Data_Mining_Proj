from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from export_report_summary import export_summary


def _write_run(path: Path, accuracy: float, *, submission: bool = False) -> None:
    path.mkdir(parents=True)
    (path / "val_metrics.json").write_text(json.dumps({"accuracy": accuracy, "loss": 0.2}), encoding="utf-8")
    (path / "metrics.csv").write_text(
        "epoch,train_loss,train_acc,val_loss,val_acc,lr,seconds\n1,0.4,0.8,0.3,0.75,0.01,2\n",
        encoding="utf-8",
    )
    (path / "val_predictions.csv").write_text(
        "path,true,predicted,p_dog\na,0,0,0.1\nb,1,0,0.4\n", encoding="utf-8"
    )
    if submission:
        (path / "submission.csv").write_text("id,label\n1,0\n2,1\n", encoding="utf-8")
        (path / "test_predictions.csv").write_text(
            "id,p_dog,p_cat\n1,0.2,0.8\n2,0.8,0.2\n", encoding="utf-8"
        )


def _fixture_repo(root: Path) -> Path:
    (root / "runs").mkdir(parents=True)
    (root / "runs/MATRIX_DONE").touch()
    for pipeline, sweep in (("baseline", "gpu_baseline"), ("augmented", "gpu_aug"), ("mixed", "gpu_mixed")):
        for backbone in ("resnet18", "resnet50"):
            for freeze in ("frozen", "last_block", "full"):
                run = root / "runs/sweeps" / sweep / f"{backbone}__{freeze}__cell/seed_42/run"
                _write_run(run, 0.8)
    for group in ("best", "aug", "mixed"):
        for seed in range(3):
            run = root / "runs/seeds" / group / f"seed_{seed}" / "run"
            _write_run(run, 0.8, submission=group == "best" and seed == 1)
    for folder in ("train/cat", "train/dog", "val/cat", "val/dog", "test"):
        path = root / "data" / folder
        path.mkdir(parents=True)
        (path / "1.jpg").touch()
    (root / "data_aug").mkdir()
    (root / "data_aug/augmentation.csv").write_text(
        "file,adjustment,parameter\ntrain/cat/a.jpg,flip,none\n", encoding="utf-8"
    )
    return root


def test_export_summary_writes_matrix_curves_seeds_and_error_counts(tmp_path: Path) -> None:
    repo = _fixture_repo(tmp_path)
    out = export_summary(repo)
    with (out / "matrix.csv").open(newline="", encoding="utf-8") as stream:
        matrix = list(csv.DictReader(stream))
    with (out / "seeds.csv").open(newline="", encoding="utf-8") as stream:
        seeds = list(csv.DictReader(stream))
    with (out / "confusion.csv").open(newline="", encoding="utf-8") as stream:
        confusion = list(csv.DictReader(stream))
    assert len(matrix) == 18
    assert len(seeds) == 9
    assert (out / "curves.csv").is_file()
    assert (out / "misclassified.csv").read_text(encoding="utf-8").count("b,1,0,0.4") == 1
    assert confusion[0]["fn"] == "1"
    assert json.loads((out / "submission_stats.json").read_text(encoding="utf-8"))["labels"] == {"cat": 1, "dog": 1}


def test_export_summary_rejects_run_without_completion_marker(tmp_path: Path) -> None:
    repo = _fixture_repo(tmp_path)
    run = next((repo / "runs/sweeps/gpu_baseline").glob("*/seed_42/run"))
    (run / "val_metrics.json").unlink()
    with pytest.raises(FileNotFoundError, match="missing val_metrics.json"):
        export_summary(repo)


def test_export_summary_rejects_missing_matrix_cell(tmp_path: Path) -> None:
    repo = _fixture_repo(tmp_path)
    cell = next((repo / "runs/sweeps/gpu_aug").glob("resnet18__frozen__*/"))
    import shutil

    shutil.rmtree(cell)
    with pytest.raises(FileNotFoundError, match="Incomplete augmented matrix"):
        export_summary(repo)
