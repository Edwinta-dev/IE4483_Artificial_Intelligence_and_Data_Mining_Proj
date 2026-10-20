from __future__ import annotations

import csv

from PIL import Image

from ie4483.data import ImageFolderDataset, UnlabelledTestDataset, make_dataloader


def save_image(path, color):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 4), color).save(path)


def test_label_order_seeded_subsets_corrupt_skips_and_manifest(tmp_path):
    root = tmp_path / "data"
    for class_name, color in (("cat", "red"), ("dog", "blue")):
        for number in range(6):
            save_image(root / "train" / class_name / f"{number}.png", color)
    (root / "train" / "cat" / "broken.png").write_bytes(b"not an image")
    manifest = tmp_path / "run" / "manifest.csv"

    first = ImageFolderDataset(root, "train", ["cat", "dog"], n_per_class=3, seed=17, manifest_path=manifest)
    second = ImageFolderDataset(root, "train", ["cat", "dog"], n_per_class=3, seed=17)
    assert first.unreadable_count == 1
    assert first.samples == second.samples
    assert {label for _, label in first.samples if label == 0} == {0}
    assert [label for _, label in first.samples] == [0, 0, 0, 1, 1, 1]
    with manifest.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 6
    assert {row["split"] for row in rows} == {"train"}
    assert {row["class"] for row in rows} == {"cat", "dog"}


def test_test_dataset_numeric_id_sorting_and_loader(tmp_path):
    folder = tmp_path / "data" / "test"
    for image_id in (10, 2, 1):
        save_image(folder / f"{image_id}.jpg", "green")
    (folder / "notes.txt").write_text("ignore")
    dataset = UnlabelledTestDataset(tmp_path / "data")
    assert [image_id for _, image_id in dataset.samples] == [1, 2, 10]
    assert [batch[1].tolist() for batch in make_dataloader(dataset, 2)] == [[1, 2], [10]]

