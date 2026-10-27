"""Export text-only summaries of completed Dogs vs Cats runs for the report notebook."""
from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PIPELINES = {"baseline": "gpu_baseline", "augmented": "gpu_aug", "mixed": "gpu_mixed"}
FREEZE_ORDER = {"frozen": 0, "last_block": 1, "full": 2}
BACKBONES = ("resnet18", "resnet50")
FREEZES = tuple(FREEZE_ORDER)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _run_dir(cell: Path, seed: str) -> Path:
    seed_dir = cell / seed
    runs = [p for p in seed_dir.iterdir() if p.is_dir() and p.name != "feature_cache"]
    if len(runs) != 1:
        raise FileNotFoundError(f"Expected one run under {seed_dir}, found {len(runs)}")
    run = runs[0]
    if not (run / "val_metrics.json").is_file():
        raise FileNotFoundError(f"Unfinished run: {run} (missing val_metrics.json)")
    return run


def export_summary(repo_root: Path = ROOT, output: Path | None = None) -> Path:
    """Build report CSV/JSON files from completed runs; never evaluates or trains."""
    if not (repo_root / "runs/MATRIX_DONE").is_file():
        raise FileNotFoundError("Missing runs/MATRIX_DONE; finish the owner-run sweep first: python scripts/sweep.py --sweep configs/sweeps/gpu_<aug|mixed>.yaml --report docs/sweep_results_<aug|mixed>.md")
    out = output or repo_root / "runs/report/summary"
    runs: list[tuple[str, str, Path]] = []
    for pipeline, sweep in PIPELINES.items():
        sweep_root = repo_root / "runs/sweeps" / sweep
        if not sweep_root.is_dir():
            raise FileNotFoundError(f"Missing sweep folder: {sweep_root}")
        for cell in sweep_root.iterdir():
            if not cell.is_dir():
                continue
            parts = cell.name.split("__")
            if len(parts) < 2:
                continue
            backbone, freeze = parts[0], parts[1]
            runs.append((pipeline, f"{backbone}|{freeze}", _run_dir(cell, "seed_42")))
        found = {tuple(config.split("|")) for name, config, _ in runs if name == pipeline}
        expected = {(backbone, freeze) for backbone in BACKBONES for freeze in FREEZES}
        if found != expected:
            missing = sorted(expected - found)
            raise FileNotFoundError(f"Incomplete {pipeline} matrix under {sweep_root}; missing {missing}. Finish the owner-run sweep first: python scripts/sweep.py --sweep configs/sweeps/gpu_<aug|mixed>.yaml --report docs/sweep_results_<aug|mixed>.md")

    matrix: list[dict] = []
    curves: list[dict] = []
    confusion: list[dict] = []
    candidate_errors: list[tuple[float, Path]] = []
    for pipeline, config, run in runs:
        backbone, freeze = config.split("|")
        result = _read_json(run / "val_metrics.json")
        if result.get("proxy") or result.get("is_proxy"):
            raise ValueError(f"Proxy-marked result cannot be reported: {run}")
        with (run / "metrics.csv").open(newline="", encoding="utf-8") as stream:
            epochs = list(csv.DictReader(stream))
        config_path = run / "config.yaml"
        max_epochs = None
        if config_path.is_file():
            import yaml
            config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
            max_epochs = config.get("train", {}).get("epochs")
        stopped_early = max_epochs is not None and len(epochs) < int(max_epochs)
        matrix.append({
            "backbone": backbone, "freeze": freeze, "pipeline": pipeline,
            "val_acc": result["accuracy"], "val_loss": result["loss"],
            "epochs": len(epochs),
            "sec_per_epoch": sum(float(row["seconds"]) for row in epochs) / len(epochs),
            "run_path": run.relative_to(repo_root).as_posix(),
        })
        for row in epochs:
            curves.append({"backbone": backbone, "freeze": freeze, "pipeline": pipeline,
                           **row, "early_stop": stopped_early and row is epochs[-1]})
        pred_path = run / "val_predictions.csv"
        if not pred_path.is_file():
            raise FileNotFoundError(f"Missing validation predictions: {pred_path}")
        counts = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
        with pred_path.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                truth, pred = int(row["true"]), int(row["predicted"])
                key = "tp" if truth == 1 and pred == 1 else "fp" if truth == 0 and pred == 1 else "tn" if truth == 0 else "fn"
                counts[key] += 1
        confusion.append({"backbone": backbone, "freeze": freeze, "pipeline": pipeline, **counts})
        candidate_errors.append((float(result["accuracy"]), run))

    matrix.sort(key=lambda r: (r["backbone"], FREEZE_ORDER[r["freeze"]], list(PIPELINES).index(r["pipeline"])))
    _write_csv(out / "matrix.csv", ["backbone", "freeze", "pipeline", "val_acc", "val_loss", "epochs", "sec_per_epoch", "run_path"], matrix)
    _write_csv(out / "curves.csv", ["backbone", "freeze", "pipeline", "epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr", "seconds", "early_stop"], curves)
    _write_csv(out / "confusion.csv", ["backbone", "freeze", "pipeline", "tp", "fp", "tn", "fn"], confusion)

    seed_rows: list[dict] = []
    for group in ("best", "aug", "mixed"):
        for seed_dir in sorted((repo_root / "runs/seeds" / group).glob("seed_*")):
            run = _run_dir(seed_dir.parent, seed_dir.name)
            result = _read_json(run / "val_metrics.json")
            seed_rows.append({"pipeline": group, "seed": seed_dir.name.removeprefix("seed_"), "val_acc": result["accuracy"], "val_loss": result["loss"], "run_path": run.relative_to(repo_root).as_posix()})
    _write_csv(out / "seeds.csv", ["pipeline", "seed", "val_acc", "val_loss", "run_path"], seed_rows)

    best_run = max(candidate_errors, key=lambda item: item[0])[1]
    with (best_run / "val_predictions.csv").open(newline="", encoding="utf-8") as stream:
        errors = [{"path": r["path"], "true": r["true"], "pred": r["predicted"], "prob": r["p_dog"]}
                  for r in csv.DictReader(stream) if r["true"] != r["predicted"]]
    _write_csv(out / "misclassified.csv", ["path", "true", "pred", "prob"], errors)

    best_seed = repo_root / "runs/seeds/best/seed_1"
    submission_run = _run_dir(best_seed.parent, best_seed.name)
    with (submission_run / "submission.csv").open(newline="", encoding="utf-8") as stream:
        labels = [int(row["label"]) for row in csv.DictReader(stream)]
    with (submission_run / "test_predictions.csv").open(newline="", encoding="utf-8") as stream:
        confidences = [max(float(row["p_dog"]), 1 - float(row["p_dog"])) for row in csv.DictReader(stream)]
    bins = [0, .5, .6, .7, .8, .9, .95, .99, 1.0000001]
    hist = [{"range": f"{lo:.2f}-{min(hi,1):.2f}", "count": sum(lo <= value < hi for value in confidences)} for lo, hi in itertools.pairwise(bins)]
    (out / "submission_stats.json").write_text(json.dumps({"labels": {"cat": labels.count(0), "dog": labels.count(1)}, "confidence_histogram": hist, "source_run": submission_run.relative_to(repo_root).as_posix()}, indent=2) + "\n", encoding="utf-8")

    counts = {}
    for split in ("train/cat", "train/dog", "val/cat", "val/dog", "test"):
        counts[split] = sum(p.is_file() for p in (repo_root / "data" / split).iterdir())
    (out / "data_counts.json").write_text(json.dumps({"counts": counts, "deduplicated_images": 6}, indent=2) + "\n", encoding="utf-8")
    with (repo_root / "data_aug/augmentation.csv").open(newline="", encoding="utf-8") as stream:
        adjustments = list(csv.DictReader(stream))
    adjustment_counts: dict[str, int] = {}
    for row in adjustments:
        adjustment_counts[row["adjustment"]] = adjustment_counts.get(row["adjustment"], 0) + 1
    _write_csv(out / "adjustments.csv", ["adjustment", "images", "example_parameters"],
               [{"adjustment": key, "images": adjustment_counts[key],
                 "example_parameters": next(r["parameter"] for r in adjustments if r["adjustment"] == key)}
                for key in sorted(adjustment_counts)])
    examples = []
    for adjustment in sorted({r["adjustment"] for r in adjustments}):
        row = next(r for r in adjustments if r["adjustment"] == adjustment)
        examples.append({"adjustment": adjustment, "source_path": row["file"], "parameter": row["parameter"]})
    _write_csv(out / "adjustment_examples.csv", ["adjustment", "source_path", "parameter"], examples)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "runs/report/summary")
    args = parser.parse_args()
    print(export_summary(ROOT, args.output))


if __name__ == "__main__":
    main()
