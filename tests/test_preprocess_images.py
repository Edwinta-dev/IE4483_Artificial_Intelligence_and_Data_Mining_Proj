from pathlib import Path

from PIL import Image

from ie4483.preprocess_images import preprocess_images


def write_image(path: Path, color: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (12, 8), color).save(path)


def test_preprocessing_deduplicates_across_splits_and_replaces_old_images(tmp_path):
    source, output = tmp_path / "source", tmp_path / "output"
    write_image(source / "train/cat/first.png", "red")
    write_image(source / "train/cat/copy.png", "red")
    write_image(source / "val/cat/leak.png", "red")
    write_image(source / "val/dog/unique.png", "blue")
    write_image(source / "test/1.png", "green")
    write_image(output / "train/cat/stale.jpg", "black")

    preprocess_images(source, output, (16, 16))

    assert len(list((output / "train/cat").glob("*.jpg"))) == 1
    assert (output / "train/cat/copy.jpg").is_file()
    assert not (output / "train/cat/first.jpg").exists()
    assert not (output / "val/cat/leak.jpg").exists()
    assert (output / "val/dog/unique.jpg").is_file()
    assert (output / "test/1.jpg").is_file()
    assert not (output / "train/cat/stale.jpg").exists()
