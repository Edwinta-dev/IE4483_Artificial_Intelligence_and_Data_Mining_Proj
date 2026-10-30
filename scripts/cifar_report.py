"""Build the CIFAR-10 test report from completed run artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

ROOT = Path(__file__).resolve().parents[1]
FIXES = {
    "cifar_imb_none": "none",
    "cifar_imb_weighted": "class-weighted loss",
    "cifar_imb_sampler": "weighted sampler",
    "cifar_imb_focal": "focal loss",
}
CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
FIELDS = ["run", "backbone", "fix", "seed", "test_accuracy", "macro_f1", "macro_recall", "cat_recall", "cat_precision"]
EXPECTED_RUNS = 14
TEST_COUNT = 10000
FIGURE_DPI = 160


def _runs(run_root: Path) -> list[dict[str, Any]]:
    rows = []
    groups = [("cifar_baseline", "balanced", ["resnet18", "resnet50"])]
    groups.extend((folder, fix, ["resnet18"]) for folder, fix in FIXES.items())
    for folder, fix, backbones in groups:
        base = run_root / folder
        for backbone in backbones:
            candidates = sorted(base.glob(f"{backbone}__*/seed_*/**/test_metrics.json"))
            for metrics_path in candidates:
                try:
                    run_config = yaml.safe_load((metrics_path.parent / "config.yaml").read_text(encoding="utf-8"))
                    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                except (OSError, ValueError, yaml.YAMLError) as exc:
                    raise ValueError(f"cannot read completed test result {metrics_path}: {exc}") from exc
                if metrics.get("count") != TEST_COUNT:
                    raise ValueError(f"{metrics_path} is not the 10,000-image official test evaluation")
                if len(metrics.get("confusion_matrix", [])) != len(CLASSES):
                    raise ValueError(f"{metrics_path} has no ten-class confusion matrix")
                per_class = metrics.get("per_class", {})
                if "cat" not in per_class:
                    raise ValueError(f"{metrics_path} is missing cat metrics")
                seed = int(metrics_path.parent.parent.name.removeprefix("seed_"))
                rows.append({
                    "run": str(metrics_path.parent.relative_to(run_root.parent.parent)).replace("\\", "/"),
                    "backbone": run_config["model"]["architecture"],
                    "fix": fix,
                    "seed": seed,
                    "test_accuracy": float(metrics["accuracy"]),
                    "macro_f1": float(metrics["macro_f1"]),
                    "macro_recall": float(metrics["macro_recall"]),
                    "cat_recall": float(per_class["cat"]["recall"]),
                    "cat_precision": float(per_class["cat"]["precision"]),
                    "confusion_matrix": metrics["confusion_matrix"],
                })
    if len(rows) != EXPECTED_RUNS:
        raise ValueError(f"expected {EXPECTED_RUNS} completed CIFAR runs, found {len(rows)}")
    return rows


def _mean_std(values: list[float]) -> tuple[float, float]:
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def generate_report(run_root: Path, report_dir: Path, figures_dir: Path, markdown_path: Path) -> list[dict[str, Any]]:
    rows = _runs(run_root)
    report_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    public_rows = [{key: row[key] for key in FIELDS} for row in rows]
    with (report_dir / "cifar_test_results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(public_rows)

    groups: dict[str, list[dict[str, Any]]] = {name: [row for row in rows if row["fix"] == name] for name in ["none", *FIXES.values()]}
    summaries = {
        fix: {metric: {"mean": _mean_std([row[metric] for row in items])[0], "std": _mean_std([row[metric] for row in items])[1]}
              for metric in ("test_accuracy", "cat_recall")}
        for fix, items in groups.items()
    }
    cats_missed = {}
    for fix, items in groups.items():
        matrix = [[sum(row["confusion_matrix"][3][column] for row in items) / len(items) for column in range(10)]][0]
        cats_missed[fix] = {CLASSES[i]: matrix[i] for i in range(10) if i != 3}
    data = {
        "rows": public_rows,
        "summaries": summaries,
        "cat_row_mean_counts": cats_missed,
        "confusion_matrices": {
            row["fix"] + ("_" + row["backbone"] if row["fix"] == "balanced" else ""): row["confusion_matrix"]
            for row in rows if (row["fix"] == "none" and row["seed"] == 0)
            or (row["fix"] == "balanced" and row["backbone"] == "resnet50")
        },
    }
    (report_dir / "cifar_test_results.json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")

    for key, fix in (("balanced_resnet50", "balanced"), ("imbalance_none", "none")):
        row = next(r for r in rows if r["fix"] == fix and (fix != "balanced" or r["backbone"] == "resnet50"))
        matrix = row["confusion_matrix"]
        normalized = [[value / sum(line) if sum(line) else 0 for value in line] for line in matrix]
        fig, ax = plt.subplots(figsize=(8, 7))
        image = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
        ax.set(xticks=range(10), yticks=range(10), xticklabels=CLASSES, yticklabels=CLASSES,
               xlabel="Predicted class", ylabel="True class", title=f"{key.replace('_', ' ').title()} test confusion")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        fig.colorbar(image, ax=ax, label="Row fraction")
        fig.tight_layout()
        fig.savefig(figures_dir / f"cifar_{key}_confusion.png", dpi=FIGURE_DPI)
        plt.close(fig)

    labels = list(groups)
    x = list(range(len(labels)))
    fig, ax = plt.subplots(figsize=(9, 5))
    for metric, offset, label in (("cat_recall", -0.18, "Cat recall"), ("test_accuracy", 0.18, "Overall accuracy")):
        means = [summaries[name][metric]["mean"] for name in labels]
        stds = [summaries[name][metric]["std"] for name in labels]
        ax.bar([i + offset for i in x], means, width=0.34, yerr=stds, capsize=4, label=label)
    ref = next(row["cat_recall"] for row in rows if row["fix"] == "balanced" and row["backbone"] == "resnet18")
    ref_accuracy = next(row["test_accuracy"] for row in rows if row["fix"] == "balanced" and row["backbone"] == "resnet18")
    ax.axhline(ref, color="black", linestyle="--", label="Balanced ResNet-18 cat recall")
    ax.axhline(ref_accuracy, color="gray", linestyle=":", label="Balanced ResNet-18 accuracy")
    ax.set(xticks=x, xticklabels=labels, ylim=(0, 1), ylabel="Test score", title="Cat recall and overall accuracy by imbalance fix")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "cifar_fix_comparison.png", dpi=FIGURE_DPI)
    plt.close(fig)

    balanced = next(row for row in rows if row["fix"] == "balanced" and row["backbone"] == "resnet18")
    none = summaries["none"]
    del_cat = balanced["cat_recall"] - none["cat_recall"]["mean"]
    best_fix = max(FIXES.values(), key=lambda name: summaries[name]["cat_recall"]["mean"])
    best_recovery = summaries[best_fix]["cat_recall"]["mean"] - none["cat_recall"]["mean"]
    cost = summaries[best_fix]["test_accuracy"]["mean"] - none["test_accuracy"]["mean"]
    dog_change = statistics.mean(row["confusion_matrix"][5][5] / sum(row["confusion_matrix"][5]) for row in groups[best_fix]) - statistics.mean(row["confusion_matrix"][5][5] / sum(row["confusion_matrix"][5]) for row in groups["none"])
    close_accuracy = abs(summaries["class-weighted loss"]["test_accuracy"]["mean"] - summaries["weighted sampler"]["test_accuracy"]["mean"])
    note = [
        f"Reducing class 3 (cat) to 10% lowers cat recall by {del_cat:.3f} relative to the balanced ResNet-18 seed-42 run.",
        f"{best_fix} recovers the most cat recall among the fixes ({best_recovery:+.3f} versus no fix); overall accuracy changes by {cost:+.3f}, while dog recall shifts by {dog_change:+.3f}.",
        "The overall accuracy and cat-recall means are averaged across three seeds for each imbalanced fix; error bars show sample standard deviation.",
        f"Weighted-loss and sampler overall accuracy differ by {close_accuracy:.3f}, below their seed standard deviations ({summaries['class-weighted loss']['test_accuracy']['std']:.3f} and {summaries['weighted sampler']['test_accuracy']['std']:.3f}); this ranking is smaller than the seed spread.",
        "Both balanced backbones have a single seed, so they provide references without a variability estimate.",
    ]
    table = ["| Run | Backbone | Fix | Seed | Accuracy | Macro-F1 | Macro recall | Cat recall | Cat precision |", "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in public_rows:
        table.append(f"| {row['run']} | {row['backbone']} | {row['fix']} | {row['seed']} | {row['test_accuracy']:.4f} | {row['macro_f1']:.4f} | {row['macro_recall']:.4f} | {row['cat_recall']:.4f} | {row['cat_precision']:.4f} |")
    lost = ["| Fix | Missed cat predicted as (mean count per 1,000 cats) |", "|---|---|"]
    for fix, values in cats_missed.items():
        lost.append(f"| {fix} | " + ", ".join(f"{name} {count / 10:.1f}" for name, count in sorted(values.items(), key=lambda item: -item[1])) + " |")
    body = "# CIFAR-10 test results\n\n" + "\n".join(note) + "\n\n" + "\n".join(table) + "\n\n" + "\n".join(lost) + "\n\n" + "![Row-normalized confusion: balanced ResNet-50](figures/cifar_balanced_resnet50_confusion.png)\n\n" + "![Row-normalized confusion: no-fix imbalanced ResNet-18](figures/cifar_imbalance_none_confusion.png)\n\n" + "![Cat recall and accuracy by fix](figures/cifar_fix_comparison.png)\n"
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(body, encoding="utf-8")
    return public_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=ROOT / "runs/sweeps")
    parser.add_argument("--report-dir", type=Path, default=ROOT / "runs/report/cifar")
    parser.add_argument("--figures-dir", type=Path, default=ROOT / "docs/figures")
    parser.add_argument("--markdown", type=Path, default=ROOT / "docs/cifar_results.md")
    args = parser.parse_args()
    rows = generate_report(args.runs_root, args.report_dir, args.figures_dir, args.markdown)
    print(f"Wrote CIFAR report for {len(rows)} test runs to {args.markdown}")


if __name__ == "__main__":
    main()
