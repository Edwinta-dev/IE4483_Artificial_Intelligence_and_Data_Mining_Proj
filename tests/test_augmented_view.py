import csv
import hashlib
from pathlib import Path

from PIL import Image

from ie4483.augmented_view import ADJUSTMENTS, make_augmented_view


def _tree(root: Path) -> None:
    for label in ("cat", "dog"):
        folder = root / "train" / label
        folder.mkdir(parents=True)
        for index in range(32):
            Image.new("RGB", (40, 32), (index * 5 % 256, 70, 130)).save(folder / f"{index}.jpg")
    for split in ("val/cat", "val/dog", "test"):
        folder = root / split
        folder.mkdir(parents=True)
        (folder / "keep.bin").write_bytes(b"held-out bytes\x00")


def _hashes(folder: Path) -> dict[str, str]:
    return {str(path.relative_to(folder)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in folder.rglob("*") if path.is_file()}


def test_seeded_adjustments_and_held_out_copies(tmp_path):
    source = tmp_path / "source"
    _tree(source)
    first, second = tmp_path / "first", tmp_path / "second"
    counts = make_augmented_view(source, first, "augmented", seed=0)
    make_augmented_view(source, second, "augmented", seed=0)
    assert counts == {name: counts[name] for name in ADJUSTMENTS}
    assert set(counts.values())
    assert _hashes(first / "train") == _hashes(second / "train")
    for split in ("val", "test"):
        assert _hashes(source / split) == _hashes(first / split)
    with (first / "augmentation.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 64
    assert {row["adjustment"] for row in rows} == set(ADJUSTMENTS)


def test_mixed_mode_doubles_training_files_and_csv_rows(tmp_path):
    source, output = tmp_path / "source", tmp_path / "mixed"
    _tree(source)
    counts = make_augmented_view(source, output, "mixed", seed=7)
    for label in ("cat", "dog"):
        files = list((output / "train" / label).iterdir())
        assert len(files) == 64
        assert sum(path.stem.endswith("_aug") for path in files) == 32
    with (output / "augmentation.csv").open(newline="", encoding="utf-8") as stream:
        assert len(list(csv.DictReader(stream))) == sum(counts.values()) == 64
