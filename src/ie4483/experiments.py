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

from ie4483.config import load_config
from ie4483.evaluate import evaluate
from ie4483.train import train


def _override(key: str, value: Any) -> str:
    return f"{key}={yaml.safe_dump(value, default_flow_style=True).strip()}"


def _run_one(
    config_path: str | Path, seed: int, run_root: Path, overrides: list[str], resume_dir: Path | None = None
) -> tuple[Path, float]:
    all_overrides = [*overrides, f"train.seed={seed}"]
    config = load_config(config_path, all_overrides)
    run_dir = train(config, run_root, resume=resume_dir)
    metrics = evaluate(run_dir, run_dir / "latest.json")
    return run_dir, float(metrics["accuracy"])


def _group_value(value: Any) -> Any:
    """Return a hashable, readable grouping key for a sweep parameter value."""
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return value


def _mean_std(values: list[float]) -> tuple[float, float]:
    return mean(values), (stdev(values) if len(values) > 1 else 0.0)


def _group_rows(rows: list[dict[str, Any]], dimensions: list[str]) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    """Rows grouped by their (hashable) values for ``dimensions``, in first-seen order."""
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(tuple(_group_value(row.get(key)) for key in dimensions), []).append(row)
    return grouped


def _write_summary(rows: list[dict[str, Any]], output: Path, dimensions: list[str] | None = None) -> None:
    """Write ``summary.csv`` and ``summary.md``: one line per seed, or one per parameter cell when swept."""
    output.mkdir(parents=True, exist_ok=True)
    dimensions = dimensions or []
    grouped = _group_rows(rows, dimensions)
    if dimensions:
        seed_ids = sorted({row["seed"] for row in rows})
        fields = dimensions + ["seed_count", "mean", "std"] + [f"seed_{seed}_accuracy" for seed in seed_ids]
        csv_rows = []
        lines = [
            "# Experiment summary",
            "",
            "| " + " | ".join([*dimensions, "Seeds", "Mean validation accuracy ± std"]) + " |",
            "|" + "|".join(["---"] * (len(dimensions) + 2)) + "|",
        ]
        for key, entries in grouped.items():
            values = [float(entry["val_accuracy"]) for entry in entries]
            average, spread = _mean_std(values)
            csv_rows.append(
                {
                    **dict(zip(dimensions, key)),
                    "seed_count": len(values),
                    "mean": average,
                    "std": spread,
                    **{f"seed_{entry['seed']}_accuracy": entry["val_accuracy"] for entry in entries},
                }
            )
            lines.append("| " + " | ".join(map(str, [*key, len(values), f"{average:.6f} ± {spread:.6f}"])) + " |")
    else:
        fields = ["seed", "run", "val_accuracy"]
        values = [float(row["val_accuracy"]) for row in rows]
        csv_rows = rows
        lines = ["# Experiment summary", "", "| Seed | Validation accuracy | Run |", "|---:|---:|---|"]
        lines.extend(f"| {row['seed']} | {float(row['val_accuracy']):.6f} | {row['run']} |" for row in rows)
        if rows:
            average, spread = _mean_std(values)
            fields += ["mean", "std"]
            csv_rows = [{**row, "mean": average, "std": spread} for row in rows]
            lines.extend(["", f"Mean: {average:.6f}; standard deviation: {spread:.6f}"])
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(csv_rows)
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_seeds(
    config_path: str | Path, seeds: list[int], output: str | Path, overrides: list[str] | None = None
) -> list[dict[str, Any]]:
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


def default_sweep_output(sweep_path: str | Path) -> Path:
    """Where a sweep writes its runs when no output folder is given."""
    return Path("runs") / "sweeps" / Path(sweep_path).stem


def _resolve_config(spec_path: Path, configured: str | Path) -> Path:
    """Resolve a sweep's ``config`` value: as given (cwd-relative), else relative to the sweep's repo root."""
    candidate = Path(configured)
    if candidate.is_absolute() or candidate.exists():
        return candidate
    # Sweep specs live in <root>/configs/sweeps/, so the repository root is three levels up.
    from_root = spec_path.resolve().parents[2] / candidate
    if from_root.exists():
        return from_root
    raise FileNotFoundError(f"sweep config not found: {configured} (named in {spec_path})")


def sweep(sweep_path: str | Path, output: str | Path | None = None) -> list[dict[str, Any]]:
    """Execute each parameter combination and resume completed seed runs."""
    spec_path = Path(sweep_path)
    spec = yaml.safe_load(spec_path.read_text(encoding="utf-8")) or {}
    config_path = _resolve_config(spec_path, spec.get("config", "configs/base.yaml"))
    parameters = spec.get("parameters")
    seeds = spec.get("seeds")
    if (
        not isinstance(parameters, dict)
        or not parameters
        or not all(isinstance(v, list) and v for v in parameters.values())
    ):
        raise ValueError("sweep parameters must map config keys to non-empty lists")
    if not isinstance(seeds, list) or not seeds:
        raise ValueError("sweep seeds must be a non-empty list")
    shared_overrides = spec.get("overrides", [])
    if not isinstance(shared_overrides, list) or not all(isinstance(item, str) for item in shared_overrides):
        raise ValueError("sweep overrides must be a list of KEY=VALUE strings")
    dimensions = list(parameters)
    combinations = list(itertools.product(*(parameters[key] for key in dimensions)))
    output = Path(output) if output else default_sweep_output(spec_path)
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
                run_dir, accuracy = _run_one(config_path, int(seed), seed_dir, overrides, _partial_run(seed_dir))
            rows.append({**cell, "seed": seed, "run": str(run_dir), "val_accuracy": accuracy})
        _write_summary(rows, output, dimensions)
    # Order cells by mean validation accuracy; ties keep their declared grid order.
    by_cell = _group_rows(rows, dimensions)
    cells = sorted(
        by_cell, key=lambda cell: _mean_std([float(row["val_accuracy"]) for row in by_cell[cell]])[0], reverse=True
    )
    ordered_rows = [row for cell in cells for row in by_cell[cell]]
    _write_summary(ordered_rows, output, dimensions)
    stats = [_mean_std([float(row["val_accuracy"]) for row in by_cell[cell]]) for cell in cells]
    means, errors = [average for average, _ in stats], [spread for _, spread in stats]
    raw_values = {tuple(_group_value(value) for value in values): values for values in combinations}
    labels = [", ".join(f"{key}={value}" for key, value in zip(dimensions, raw_values[cell])) for cell in cells]
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.1), 4))
    ax.bar(range(len(labels)), means, yerr=errors, capsize=4)
    ax.set(xticks=range(len(labels)), xticklabels=labels, ylabel="Validation accuracy", xlabel="Parameters")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(output / "sweep.png", dpi=150)
    plt.close(fig)
    return ordered_rows
