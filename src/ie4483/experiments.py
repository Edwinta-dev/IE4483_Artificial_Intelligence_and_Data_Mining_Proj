"""Seeded training summaries and resumable parameter sweeps."""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
import re
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

from ie4483.evaluate import evaluate
from ie4483.train import load_training_config, train


def _override(key: str, value: Any) -> str:
    return f"{key}={yaml.safe_dump(value, default_flow_style=True).strip()}"


def _run_one(config_path: str | Path, seed: int, run_root: Path, overrides: list[str],
             resume_dir: Path | None = None) -> tuple[Path, float]:
    all_overrides = [*overrides, f"train.seed={seed}"]
    config, extra = load_training_config(config_path, all_overrides)
    run_dir = train(config, extra, run_root, resume=resume_dir)
    metrics = evaluate(run_dir, run_dir / "latest.json")
    return run_dir, float(metrics["accuracy"])


def _group_value(value: Any) -> Any:
    """Return a hashable, readable grouping key for a sweep parameter value."""
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return value


def _write_summary(rows: list[dict[str, Any]], output: Path, dimensions: list[str] | None = None) -> None:
    output.mkdir(parents=True, exist_ok=True)
    fields = (dimensions or []) + ["seed", "run", "val_accuracy"]
    csv_rows = rows
    if dimensions:
        grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
        for row in rows:
            grouped.setdefault(tuple(_group_value(row[key]) for key in dimensions), []).append(row)
        seed_ids = sorted({row["seed"] for row in rows})
        fields = dimensions + ["seed_count", "mean", "std"] + [f"seed_{seed}_accuracy" for seed in seed_ids]
        csv_rows = []
        for key, entries in grouped.items():
            values = [float(entry["val_accuracy"]) for entry in entries]
            csv_rows.append({
                **dict(zip(dimensions, key)),
                "seed_count": len(values),
                "mean": mean(values),
                "std": stdev(values) if len(values) > 1 else 0.0,
                **{f"seed_{entry['seed']}_accuracy": entry["val_accuracy"] for entry in entries},
            })
    elif rows:
        values = [float(row["val_accuracy"]) for row in rows]
        spread = stdev(values) if len(values) > 1 else 0.0
        fields += ["mean", "std"]
        csv_rows = [{**row, "mean": mean(values), "std": spread} for row in rows]
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(csv_rows)
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(tuple(_group_value(row.get(key)) for key in (dimensions or [])), []).append(row)
    lines = ["# Experiment summary", ""]
    if dimensions:
        lines.append("| " + " | ".join([*dimensions, "Seeds", "Mean validation accuracy ± std"]) + " |")
        lines.append("|" + "|".join(["---"] * (len(dimensions) + 2)) + "|")
        for key, entries in grouped.items():
            values = [float(item["val_accuracy"]) for item in entries]
            spread = stdev(values) if len(values) > 1 else 0.0
            lines.append("| " + " | ".join(map(str, [*key, len(values), f"{mean(values):.6f} ± {spread:.6f}"])) + " |")
    else:
        lines.extend(["| Seed | Validation accuracy | Run |", "|---:|---:|---|"])
        for row in rows:
            lines.append(f"| {row['seed']} | {float(row['val_accuracy']):.6f} | {row['run']} |")
        values = [float(row["val_accuracy"]) for row in rows]
        spread = stdev(values) if len(values) > 1 else 0.0
        lines.extend(["", f"Mean: {mean(values):.6f}; standard deviation: {spread:.6f}"])
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_seeds(config_path: str | Path, seeds: list[int], output: str | Path,
              overrides: list[str] | None = None) -> list[dict[str, Any]]:
    """Train and evaluate one run per seed, saving per-seed and aggregate metrics."""
    if not seeds:
        raise ValueError("at least one seed is required")
    output = Path(output)
    rows = []
    for seed in seeds:
        seed_root = output / f"seed_{seed}"
        partial = _partial_run(seed_root)
        run_dir, accuracy = _run_one(config_path, seed, seed_root, overrides or [], partial)
        rows.append({"seed": seed, "run": str(run_dir), "val_accuracy": accuracy})
    _write_summary(rows, output)
    return rows


def _cell_name(values: tuple[Any, ...]) -> str:
    parts = [re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value)).strip("-") or "empty" for value in values]
    identity = hashlib.sha256(json.dumps(values, sort_keys=True).encode("utf-8")).hexdigest()[:10]
    return "__".join([*parts, identity])


def _partial_run(seed_dir: Path) -> Path | None:
    runs = sorted(path for path in seed_dir.glob("*") if path.is_dir() and (path / "last.pt").is_file())
    return runs[-1] if runs else None


def sweep(sweep_path: str | Path, output: str | Path | None = None) -> list[dict[str, Any]]:
    """Execute each parameter combination and resume completed seed runs."""
    spec_path = Path(sweep_path)
    spec = yaml.safe_load(spec_path.read_text(encoding="utf-8")) or {}
    config_path = (spec_path.parent.parent.parent / spec.get("config", "base.yaml")).resolve()
    # A config path can also be explicitly relative to the current directory.
    configured = Path(spec.get("config", "configs/base.yaml"))
    if configured.exists():
        config_path = configured
    parameters = spec.get("parameters")
    seeds = spec.get("seeds")
    if not isinstance(parameters, dict) or not parameters or not all(isinstance(v, list) and v for v in parameters.values()):
        raise ValueError("sweep parameters must map config keys to non-empty lists")
    if not isinstance(seeds, list) or not seeds:
        raise ValueError("sweep seeds must be a non-empty list")
    shared_overrides = spec.get("overrides", [])
    if not isinstance(shared_overrides, list) or not all(isinstance(item, str) for item in shared_overrides):
        raise ValueError("sweep overrides must be a list of KEY=VALUE strings")
    dimensions = list(parameters)
    combinations = list(itertools.product(*(parameters[key] for key in dimensions)))
    output = Path(output or Path("runs") / "sweeps" / spec_path.stem)
    rows: list[dict[str, Any]] = []
    for values in combinations:
        cell = dict(zip(dimensions, values))
        cell_dir = output / _cell_name(values)
        for seed in seeds:
            seed_dir = cell_dir / f"seed_{seed}"
            existing = sorted(seed_dir.glob("*/val_metrics.json")) if seed_dir.exists() else []
            if existing:
                run_dir = existing[-1].parent
                metrics = json.loads(existing[-1].read_text(encoding="utf-8"))
                accuracy = float(metrics["accuracy"])
            else:
                overrides = [*shared_overrides, *(_override(key, value) for key, value in cell.items())]
                run_dir, accuracy = _run_one(config_path, int(seed), seed_dir, overrides,
                                             _partial_run(seed_dir))
            rows.append({**cell, "seed": seed, "run": str(run_dir), "val_accuracy": accuracy})
        _write_summary(rows, output, dimensions)
    # Sort cells by average validation accuracy; ties retain their declared grid order.
    def cell_key(values):
        return tuple(_group_value(value) for value in values)

    def row_key(row):
        return cell_key(row[d] for d in dimensions)

    cell_scores = {}
    for values in combinations:
        selected = [float(row["val_accuracy"]) for row in rows if row_key(row) == cell_key(values)]
        cell_scores[cell_key(values)] = mean(selected)
    ordered = sorted(combinations, key=lambda values: cell_scores[cell_key(values)], reverse=True)
    ordered_rows = [row for values in ordered for row in rows if row_key(row) == cell_key(values)]
    _write_summary(ordered_rows, output, dimensions)
    means = [cell_scores[cell_key(values)] for values in ordered]
    errors = []
    for values in ordered:
        scores = [float(row["val_accuracy"]) for row in rows if row_key(row) == cell_key(values)]
        errors.append(stdev(scores) if len(scores) > 1 else 0.0)
    labels = [", ".join(f"{key}={value}" for key, value in zip(dimensions, values)) for values in ordered]
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.1), 4))
    ax.bar(range(len(labels)), means, yerr=errors, capsize=4)
    ax.set(xticks=range(len(labels)), xticklabels=labels, ylabel="Validation accuracy", xlabel="Parameters")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(output / "sweep.png", dpi=150)
    plt.close(fig)
    return ordered_rows
