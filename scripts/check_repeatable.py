"""Check repeated training metrics against a fixed-seed tolerance."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from ie4483.train import load_training_config, train

METRIC_COLUMNS = ("train_loss", "train_acc", "val_loss", "val_acc")


def _metrics(run_dir: Path) -> dict[int, dict[str, float]]:
    with (run_dir / "metrics.csv").open(newline="", encoding="utf-8") as stream:
        return {
            int(row["epoch"]): {name: float(row[name]) for name in METRIC_COLUMNS}
            for row in csv.DictReader(stream)
        }


def check_repeatable(config_path: str | Path, epochs: int, output: str | Path,
                     overrides: list[str] | None = None, tolerance: float = 0.0) -> tuple[float, float]:
    """Train twice at the configured seed and once at a different seed."""
    if epochs < 1:
        raise ValueError("epochs must be at least 1")
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    config, extra = load_training_config(config_path, overrides)
    config.train.epochs = epochs
    root = Path(output)
    seed = config.train.seed
    runs = []
    for index, run_seed in enumerate((seed, seed, seed + 1)):
        config.train.seed = run_seed
        runs.append(_metrics(train(config, extra, root / f"run-{index}")))
    common_epochs = set(runs[0]) & set(runs[1])
    if len(common_epochs) != epochs or set(runs[0]) != set(runs[1]):
        raise RuntimeError("same-seed runs produced different epoch counts")
    largest_same = max(
        abs(runs[0][epoch][metric] - runs[1][epoch][metric])
        for epoch in common_epochs for metric in METRIC_COLUMNS
    )
    different_seed = max(
        abs(runs[0][epoch][metric] - runs[2][epoch][metric])
        for epoch in set(runs[0]) & set(runs[2]) for metric in METRIC_COLUMNS
    )
    print(f"largest_same_seed_metric_difference={largest_same:.12g}")
    print(f"largest_different_seed_metric_difference={different_seed:.12g}")
    print(f"tolerance={tolerance:.12g} result={'PASS' if largest_same <= tolerance else 'FAIL'}")
    return largest_same, different_seed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--epochs", required=True, type=int)
    parser.add_argument("--tolerance", type=float, default=0.0)
    parser.add_argument("--output", default="runs/repeatability")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()
    largest, _ = check_repeatable(args.config, args.epochs, args.output, args.set, args.tolerance)
    if largest > args.tolerance:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
