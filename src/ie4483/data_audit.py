"""Integrity and handoff reports for Dogs vs Cats image trees."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp"}
EXPECTED_DIRS = ("train/cat", "train/dog", "val/cat", "val/dog", "test")


def _image_files(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted((p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES), key=lambda p: p.name)


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _distribution(values: list[int]) -> dict[str, int | float | None]:
    if not values:
        return {"min": None, "median": None, "max": None}
    ordered = sorted(values)
    mid = len(ordered) // 2
    median: int | float = ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2
    return {"min": ordered[0], "median": median, "max": ordered[-1]}


def audit_data(root: str | Path, compare: str | Path | None = None) -> dict[str, Any]:
    """Inspect a dataset root and write audit.json and audit.md into that root."""
    root = Path(root)
    missing = [rel for rel in EXPECTED_DIRS if not (root / rel).is_dir()]
    counts: dict[str, Any] = {}
    unreadable: list[str] = []
    dimensions: list[tuple[int, int]] = []
    hashes: dict[str, list[str]] = defaultdict(list)
    paths: dict[str, list[Path]] = {}

    for rel in EXPECTED_DIRS:
        folder = root / rel
        files = _image_files(folder)
        paths[rel] = files
        counts[rel] = len(files)
        for path in files:
            try:
                with Image.open(path) as image:
                    image.verify()
                with Image.open(path) as image:
                    dimensions.append((image.width, image.height))
            except (OSError, ValueError, UnidentifiedImageError):
                unreadable.append(path.relative_to(root).as_posix())
            if rel.startswith("train/"):
                hashes[_hash(path)].append(path.relative_to(root).as_posix())

    train_hashes = {digest for digest, names in hashes.items() if names}
    split_hashes: dict[str, dict[str, list[str]]] = {"val": defaultdict(list), "test": defaultdict(list)}
    for rel in ("val/cat", "val/dog", "test"):
        for path in paths[rel]:
            try:
                split = "test" if rel == "test" else "val"
                split_hashes[split][_hash(path)].append(path.relative_to(root).as_posix())
            except OSError:
                pass
    validation_leakage = sorted(name for digest in train_hashes for name in split_hashes["val"].get(digest, []))
    test_leakage = sorted(name for digest in train_hashes for name in split_hashes["test"].get(digest, []))
    train_duplicates = [sorted(names) for names in hashes.values() if len(names) > 1]

    comparison = None
    if compare is not None:
        other = Path(compare)
        def rel_files(base: Path) -> set[str]:
            if not base.is_dir():
                return set()
            return {p.relative_to(base).as_posix() for p in base.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES}
        current, previous = rel_files(root), rel_files(other)
        comparison = {"root_only": sorted(current - previous), "compare_only": sorted(previous - current)}

    report: dict[str, Any] = {
        "root": str(root),
        "layout_ok": not missing,
        "missing_directories": missing,
        "counts": counts,
        "unreadable_files": sorted(unreadable),
        "image_dimensions": {"width": _distribution([w for w, _ in dimensions]), "height": _distribution([h for _, h in dimensions])},
        "exact_duplicates_within_train": train_duplicates,
        "train_validation_duplicates": validation_leakage,
        "train_test_duplicates": test_leakage,
        "compare": comparison,
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    lines = [f"# Data audit: {root}", "", f"Layout: {'OK' if not missing else 'MISSING ' + ', '.join(missing)}", "", "## Image counts"]
    lines.extend(f"- {rel}: {count}" for rel, count in counts.items())
    lines += ["", "## Image dimensions (width and height)"]
    for axis, stats in report["image_dimensions"].items():
        lines.append(f"- {axis}: min={stats['min']}, median={stats['median']}, max={stats['max']}")
    for title, key in (("Unreadable files", "unreadable_files"), ("Exact duplicates within train", "exact_duplicates_within_train"), ("Train to validation duplicates", "train_validation_duplicates"), ("Train to test duplicates", "train_test_duplicates")):
        lines += ["", f"## {title}"]
        values = report[key]
        lines.extend(("- " + (", ".join(v) if isinstance(v, list) else str(v))) for v in values) if values else lines.append("- None")
    if comparison is not None:
        lines += ["", "## Compare listing", "Files only in root: " + (", ".join(comparison["root_only"]) or "None"), "Files only in compare root: " + (", ".join(comparison["compare_only"]) or "None")]
    (root / "audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report
