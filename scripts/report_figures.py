"""Create training curves and a comparison table from completed run folders."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ie4483 import figures, proxy, train

FIELDS = ("run", "backbone", "freeze_mode", "learning_rate", "augmentation", "epochs_run",
          "best_val_accuracy", "time_per_epoch", "proxy")


def _summarize_run(run_dir: Path) -> dict[str, Any]:
    with (run_dir / "metrics.csv").open(newline="", encoding="utf-8") as stream:
        metrics = list(csv.DictReader(stream))
    if not metrics:
        raise ValueError(f"run has no epoch metrics: {run_dir}")
    config_values = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8")) or {}
    typed_config, _ = train.load_training_config(run_dir / "config.yaml")
    model = config_values.get("model", {})
    training_values = config_values.get("train", {})
    freeze_mode = model.get("freeze_mode")
    if freeze_mode is None:
        freeze_mode = "frozen" if model.get("freeze_backbone", True) else "full"
    best_index = max(range(len(metrics)), key=lambda index: float(metrics[index]["val_acc"]))
    best = metrics[best_index]
    is_proxy = proxy.is_proxy_dataset(typed_config.data_root)
    figures.training_curves(metrics, run_dir.name + (" (PROXY)" if is_proxy else ""), run_dir)
    return {
        "run": run_dir.name,
        "backbone": model.get("architecture", typed_config.model.architecture),
        "freeze_mode": freeze_mode,
        "learning_rate": training_values.get("learning_rate", typed_config.train.learning_rate),
        "augmentation": (run_dir / "transforms.txt").read_text(encoding="utf-8").strip()
        if (run_dir / "transforms.txt").is_file() else yaml.safe_dump(config_values.get("augment", {}),
                                                                       sort_keys=True).strip(),
        "epochs_run": len(metrics),
        "best_val_accuracy": float(best["val_acc"]),
        "time_per_epoch": sum(float(row["seconds"]) for row in metrics) / len(metrics),
        "proxy": is_proxy,
        "epochs": [int(row["epoch"]) for row in metrics],
        "val_accuracy_curve": [float(row["val_acc"]) for row in metrics],
    }


def report_figures(runs: list[Path], output: Path) -> list[dict[str, Any]]:
    """Write per-run charts, overlay, and CSV/Markdown comparison summaries."""
    if not runs:
        raise ValueError("at least one run directory is required")
    output.mkdir(parents=True, exist_ok=True)
    summaries = [_summarize_run(Path(run)) for run in runs]
    figures.comparison_figure(summaries, output)
    csv_path = output / "comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows({field: summary[field] for field in FIELDS} for summary in summaries)
    headers = ["Run", "Backbone", "Freeze mode", "LR", "Augmentation", "Epochs run",
               "Best val acc", "Time per epoch (s)"]
    lines = ["# Run comparison", "", "| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]
    for summary in summaries:
        run_name = summary["run"] + (" (PROXY)" if summary["proxy"] else "")
        values = [run_name, summary["backbone"], summary["freeze_mode"], f"{summary['learning_rate']:g}",
                  str(summary["augmentation"]).replace("|", "\\|"), summary["epochs_run"],
                  f"{summary['best_val_accuracy']:.6f}", f"{summary['time_per_epoch']:.3f}"]
        lines.append("| " + " | ".join(map(str, values)) + " |")
    (output / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True, type=Path, help="run directories to compare")
    parser.add_argument("--out", required=True, type=Path, help="directory for report figures and tables")
    args = parser.parse_args()
    summaries = report_figures(args.runs, args.out)
    print(json.dumps([{field: item[field] for field in FIELDS} for item in summaries], indent=2))


if __name__ == "__main__":
    main()
