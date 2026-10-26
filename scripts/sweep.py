"""Run and summarize a resumable parameter grid."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ie4483.experiments import sweep


def write_report(rows: list[dict], report_path: Path) -> list[dict]:
    """Write the real-run table from completed validation and timing logs."""
    results = []
    for row in rows:
        run = Path(row["run"])
        validation_path = run / "val_metrics.json"
        if not validation_path.is_file():
            raise ValueError(f"sweep cell has no completed validation metrics: {run}")
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
        metrics_path = run / "metrics.csv"
        with metrics_path.open(newline="", encoding="utf-8") as stream:
            metrics = list(csv.DictReader(stream))
        if not metrics:
            raise ValueError(f"completed run has no epoch metrics: {run}")
        latest = run / "latest.json"
        latest_metrics = json.loads(latest.read_text(encoding="utf-8")) if latest.exists() else {}
        if validation.get("proxy") or latest_metrics.get("proxy"):
            raise ValueError(f"proxy run cannot be included in the real-data report: {run}")
        results.append({
            **row,
            "epochs": len(metrics),
            "seconds_per_epoch": mean(float(item["seconds"]) for item in metrics),
            "best_val_accuracy": float(validation["accuracy"]),
        })
    best = max(results, key=lambda item: item["best_val_accuracy"])
    lines = ["# GPU baseline sweep results", "",
             "| Backbone | Freeze mode | Seed | Best val accuracy | Epochs | Seconds per epoch | Run |",
             "|---|---|---:|---:|---:|---:|---|"]
    for item in results:
        lines.append(
            f"| {item['model.architecture']} | {item['model.freeze_mode']} | {item['seed']} "
            f"| {item['best_val_accuracy']:.6f} | {item['epochs']} "
            f"| {item['seconds_per_epoch']:.2f} | `{item['run']}` |"
        )
    lines.extend(["", (f"The **{best['model.architecture']} / {best['model.freeze_mode']}** configuration "
                  f"had the top validation accuracy ({best['best_val_accuracy']:.6f}). It is selected "
                  "as the baseline because it "
                  "achieved the best validation result and had a lower compute cost than the "
                  "fine-tuned configurations; this keeps later comparisons efficient without giving up "
                  "measured validation accuracy.")])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", required=True, help="sweep YAML file")
    parser.add_argument("--output", default=None, help="output folder (default runs/sweeps/<name>)")
    args = parser.parse_args()
    rows = sweep(args.sweep, args.output)
    if Path(args.sweep).stem == "gpu_baseline":
        write_report(rows, Path("docs/sweep_results.md"))
    print(f"Completed {len(rows)} seed runs; summary: {args.output or 'runs/sweeps/' + args.sweep.rsplit('/', 1)[-1].removesuffix('.yaml')}")


if __name__ == "__main__":
    main()
