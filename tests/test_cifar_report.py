import json

import yaml

from scripts.cifar_report import generate_report


def test_cifar_report_reads_tiny_fake_runs_and_writes_outputs(tmp_path):
    run_root = tmp_path / "sweeps"
    run_specs = [("cifar_baseline", "resnet18", [42]), ("cifar_baseline", "resnet50", [42])]
    run_specs.extend((f"cifar_imb_{fix}", "resnet18", [0, 1, 2]) for fix in ("none", "weighted", "sampler", "focal"))
    for folder, backbone, seeds in run_specs:
        for seed in seeds:
            run = run_root / folder / f"{backbone}__fake" / f"seed_{seed}" / f"run_{seed}"
            run.mkdir(parents=True)
            (run / "config.yaml").write_text(yaml.safe_dump({"model": {"architecture": backbone}}), encoding="utf-8")
            matrix = [[0] * 10 for _ in range(10)]
            for label in range(10):
                matrix[label][label] = 999 if label == 3 else 1000
            matrix[3][5] = 1
            matrix[3][3] = 999
            metrics = {
                "count": 10000,
                "accuracy": 0.8,
                "macro_f1": 0.79,
                "macro_recall": 0.8,
                "per_class": {"cat": {"recall": 0.999, "precision": 0.9}},
                "confusion_matrix": matrix,
            }
            (run / "test_metrics.json").write_text(json.dumps(metrics), encoding="utf-8")

    report_dir = tmp_path / "report"
    figures_dir = tmp_path / "figures"
    markdown = tmp_path / "docs" / "cifar_results.md"
    rows = generate_report(run_root, report_dir, figures_dir, markdown)
    assert len(rows) == 14
    assert len((report_dir / "cifar_test_results.csv").read_text(encoding="utf-8").splitlines()) == 15
    result = json.loads((report_dir / "cifar_test_results.json").read_text(encoding="utf-8"))
    assert result["summaries"]["none"]["cat_recall"]["mean"] == 0.999
    assert result["cat_row_mean_counts"]["none"]["dog"] == 1
    assert len(result["confusion_matrices"]["balanced_resnet50"]) == 10
    assert markdown.is_file()
    assert (figures_dir / "cifar_fix_comparison.png").is_file()
