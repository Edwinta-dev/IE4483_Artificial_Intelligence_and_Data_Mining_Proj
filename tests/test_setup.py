from __future__ import annotations

import importlib.util
from pathlib import Path


def load_setup_module():
    script = Path(__file__).resolve().parents[1] / "scripts" / "check_setup.py"
    spec = importlib.util.spec_from_file_location("check_setup", script)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_check_setup_reports_complete_layout_and_missing_weights(tmp_path):
    module = load_setup_module()
    root = tmp_path / "data"
    for folder in ("train/cat", "train/dog", "val/cat", "val/dog", "test"):
        path = root / folder
        path.mkdir(parents=True)
        (path / "1.jpg").write_bytes(b"image")
        (path / "notes.txt").write_text("not an image")

    report = module.inspect_setup(root, tmp_path / "empty-cache")
    assert "train/cat: 1 images" in report
    assert "val/dog: 1 images" in report
    assert "test: 1 images" in report
    assert "resnet18-f37072fd.pth: MISSING" in report
    assert "resnet50-11ad3fa6.pth: MISSING" in report
    assert "Python:" in report and "PyTorch:" in report and "CUDA:" in report


def test_check_setup_reports_incomplete_layout_and_counts_cached_weights(tmp_path):
    module = load_setup_module()
    root = tmp_path / "partial"
    (root / "train/cat").mkdir(parents=True)
    (root / "train/cat" / "cat.png").write_bytes(b"image")
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / module.WEIGHT_FILES[0]).write_bytes(b"weights")

    report = module.inspect_setup(root, cache)
    assert "train/cat: 1 images" in report
    assert "train/dog: 0 images (MISSING)" in report
    assert "resnet18-f37072fd.pth: cached" in report
    assert "resnet50-11ad3fa6.pth: MISSING" in report
