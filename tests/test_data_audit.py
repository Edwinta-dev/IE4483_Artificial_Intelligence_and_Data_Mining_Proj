from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from PIL import Image

from ie4483.data_audit import audit_data


def save_image(path: Path, color: str, size: tuple[int, int] = (4, 7)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)


def clean_tree(root: Path) -> None:
    for rel in ("train/cat", "train/dog", "val/cat", "val/dog", "test"):
        (root / rel).mkdir(parents=True, exist_ok=True)
    save_image(root / "train/cat/a.png", "red")
    save_image(root / "train/dog/b.png", "blue")
    save_image(root / "val/cat/c.png", "green")
    save_image(root / "val/dog/d.png", "yellow")
    save_image(root / "test/1.png", "black")


def run_cli(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    project = Path(__file__).resolve().parents[1]
    return subprocess.run(
        [sys.executable, str(project / "scripts/audit_data.py"), "--root", str(root), *args],
        capture_output=True,
        text=True,
        check=False,
    )


def test_clean_tree_writes_reports_counts_and_size_stats(tmp_path):
    root = tmp_path / "clean"
    clean_tree(root)
    report = audit_data(root)
    assert report["layout_ok"]
    assert report["counts"]["train/cat"] == 1
    assert report["image_dimensions"]["width"] == {"min": 4, "median": 4, "max": 4}
    assert (root / "audit.json").is_file() and (root / "audit.md").is_file()
    assert json.loads((root / "audit.json").read_text(encoding="utf-8"))["layout_ok"]
    result = run_cli(root)
    assert result.returncode == 0


def test_missing_class_and_corrupt_file_are_reported(tmp_path):
    root = tmp_path / "data"
    clean_tree(root)
    (root / "val/dog/d.png").unlink()
    (root / "val/dog").rmdir()
    (root / "train/cat/broken.png").write_bytes(b"broken")
    result = run_cli(root)
    assert result.returncode == 1
    report = json.loads((root / "audit.json").read_text(encoding="utf-8"))
    assert "val/dog" in report["missing_directories"]
    assert "train/cat/broken.png" in report["unreadable_files"]


def test_exact_duplicate_within_train_is_reported(tmp_path):
    root = tmp_path / "data"
    clean_tree(root)
    (root / "train/dog/copy.png").write_bytes((root / "train/cat/a.png").read_bytes())
    report = audit_data(root)
    assert ["train/cat/a.png", "train/dog/copy.png"] in report["exact_duplicates_within_train"]


def test_train_validation_duplicate_causes_exit_one(tmp_path):
    root = tmp_path / "data"
    clean_tree(root)
    (root / "val/cat/leak.png").write_bytes((root / "train/cat/a.png").read_bytes())
    result = run_cli(root)
    assert result.returncode == 1
    assert "val/cat/leak.png" in json.loads((root / "audit.json").read_text(encoding="utf-8"))["train_validation_duplicates"]


def test_compare_lists_files_unique_to_each_root(tmp_path):
    root, original = tmp_path / "clean", tmp_path / "raw"
    clean_tree(root)
    clean_tree(original)
    (root / "train/cat/a.png").unlink()
    save_image(root / "train/cat/added.png", "purple")
    report = audit_data(root, original)
    assert report["compare"] == {"root_only": ["train/cat/added.png"], "compare_only": ["train/cat/a.png"]}
    assert "train/cat/a.png" in (root / "audit.md").read_text(encoding="utf-8")
