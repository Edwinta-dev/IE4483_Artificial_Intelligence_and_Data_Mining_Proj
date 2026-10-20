"""Materialize CIFAR-10 cat and dog classes as a marked folder dataset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ie4483.cifar import CIFAR10Dataset
from ie4483.proxy import PROXY_MARKER, write_proxy_marker


def make_proxy(cifar_root: str | Path, out: str | Path, n_train: int = 4000,
               n_val: int = 400, seed: int = 42) -> Path:
    if n_train < 1 or n_val < 1:
        raise ValueError("n_train and n_val must be positive per-class counts")
    root = Path(out)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"output directory is not empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    datasets = {split: CIFAR10Dataset(cifar_root, split, seed=seed) for split in ("train", "val", "test")}
    rows: list[tuple[str, str, str, str, str]] = []
    for split in ("train", "val"):
        dataset = datasets[split]
        count = n_train if split == "train" else n_val
        for class_name, class_id in (("cat", 3), ("dog", 5)):
            chosen = [i for i, label in enumerate(dataset.labels) if int(label) == class_id]
            if len(chosen) < count:
                raise ValueError(f"requested {count} {class_name} images for {split}, found {len(chosen)}")
            folder = root / split / class_name
            folder.mkdir(parents=True, exist_ok=True)
            for output_index, index in enumerate(chosen[:count]):
                image = dataset.images[index]
                filename = f"{output_index}.png"
                Image.fromarray(image, mode="RGB").save(folder / filename)
                digest = hashlib.sha256(image.tobytes()).hexdigest()
                rows.append((split, class_name, filename, str(dataset.source_indices[index]), digest))
    test = datasets["test"]
    test_folder = root / "test"
    test_folder.mkdir(parents=True, exist_ok=True)
    test_ids = [i for i, label in enumerate(test.labels) if int(label) in (3, 5)]
    for numeric_id, index in enumerate(test_ids, start=1):
        image = test.images[index]
        Image.fromarray(image, mode="RGB").save(test_folder / f"{numeric_id}.png")
        rows.append(("test", "", f"{numeric_id}.png", str(test.source_indices[index]),
                     hashlib.sha256(image.tobytes()).hexdigest()))
    write_proxy_marker(root, seed=seed, n_train=n_train, n_val=n_val)
    with (root / "manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("split", "class", "filename", "cifar_index", "sha256"))
        writer.writerows(rows)
    return root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cifar-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--n-train", type=int, default=4000, help="images per class")
    parser.add_argument("--n-val", type=int, default=400, help="images per class")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    make_proxy(args.cifar_root, args.out, args.n_train, args.n_val, args.seed)
    print(f"Created marked proxy dataset at {args.out} ({PROXY_MARKER})")


if __name__ == "__main__":
    main()
