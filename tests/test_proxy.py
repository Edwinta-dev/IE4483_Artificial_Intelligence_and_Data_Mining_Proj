import csv
import json
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from ie4483.config import load_config
from ie4483.data_audit import audit_data
from ie4483.evaluate import evaluate
from ie4483.proxy import is_proxy_dataset
from ie4483.train import train


def _fake_cifar(root):
    folder = root / "cifar-10-batches-py"
    folder.mkdir(parents=True)
    # Two pixels patterns per class and batch make source identities auditable.
    for batch_num in range(1, 6):
        labels = [label for label in range(10) for _ in range(4)]
        images = np.zeros((len(labels), 3072), dtype=np.uint8)
        for i, label in enumerate(labels):
            images[i] = (label * 17 + batch_num * 7 + i) % 256
        with (folder / f"data_batch_{batch_num}").open("wb") as stream:
            pickle.dump({b"data": images, b"labels": labels}, stream)
    labels = [label for label in range(10) for _ in range(4)]
    images = np.stack([np.full(3072, i + 1, dtype=np.uint8) for i in range(len(labels))])
    with (folder / "test_batch").open("wb") as stream:
        pickle.dump({b"data": images, b"labels": labels}, stream)


def test_proxy_generator_layout_selection_and_audits(tmp_path):
    project = Path(__file__).resolve().parents[1]
    cifar = tmp_path / "cifar"
    out = tmp_path / "proxy"
    _fake_cifar(cifar)
    command = [sys.executable, str(project / "scripts/make_proxy_catdog.py"), "--cifar-root", str(cifar),
               "--out", str(out), "--n-train", "3", "--n-val", "1", "--seed", "81"]
    subprocess.run(command, check=True, cwd=project)
    second = tmp_path / "proxy-second"
    second_command = list(command)
    second_command[second_command.index(str(out))] = str(second)
    subprocess.run(second_command, check=True, cwd=project)

    assert {p.relative_to(out).as_posix() for p in out.rglob("*.png")} == {
        *(f"train/{name}/{i}.png" for name in ("cat", "dog") for i in range(3)),
        *(f"val/{name}/0.png" for name in ("cat", "dog")),
        *(f"test/{i}.png" for i in range(1, 9)),
    }
    for split, count in (("train", 3), ("val", 1)):
        for name in ("cat", "dog"):
            assert len(list((out / split / name).glob("*.png"))) == count
            assert [p.read_bytes() for p in sorted((out / split / name).glob("*.png"))] == [
                p.read_bytes() for p in sorted((second / split / name).glob("*.png"))]
    manifest = list(csv.DictReader((out / "manifest.csv").open(encoding="utf-8")))
    assert len({row["cifar_index"] for row in manifest if row["split"] != "test"}) == 8
    hashes = {}
    for row in manifest:
        if row["split"] != "test":
            hashes.setdefault(row["sha256"], set()).add(row["split"])
    assert all(len(splits) == 1 for splits in hashes.values())
    assert (out / "PROXY_DATASET.txt").is_file() and is_proxy_dataset(out)
    audit = audit_data(out)
    assert audit["layout_ok"] and not audit["missing_directories"]
    assert not audit["unreadable_files"]
    assert not audit["train_validation_duplicates"] and not audit["train_test_duplicates"]

    setup_script = project / "scripts/check_setup.py"
    result = subprocess.run([sys.executable, str(setup_script), "--data-root", str(out)], check=False, cwd=project,
                            capture_output=True, text=True)
    assert result.returncode == 0


def test_proxy_one_epoch_cpu_evaluate_and_predict(tmp_path, monkeypatch, request):
    from predict import predict

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    previous_threads = torch.get_num_threads()
    request.addfinalizer(lambda: torch.set_num_threads(previous_threads))
    torch.set_num_threads(1)
    cifar = tmp_path / "cifar"
    out = tmp_path / "proxy"
    _fake_cifar(cifar)
    project = Path(__file__).resolve().parents[1]
    subprocess.run([sys.executable, str(project / "scripts/make_proxy_catdog.py"), "--cifar-root", str(cifar),
                    "--out", str(out), "--n-train", "3", "--n-val", "1"], check=True, cwd=project)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"data:\n  root: '{out.as_posix()}'\n  image_size: 32\n  classes: [cat, dog]\n"
        "model:\n  architecture: resnet18\n  pretrained: false\n  freeze_backbone: true\n  num_classes: 2\n"
        "train:\n  seed: 4\n  epochs: 1\n  batch_size: 2\n  mixed_precision: false\n"
        "eval:\n  batch_size: 2\n", encoding="utf-8")
    run = train(load_config(config_path), {"feature_cache": "off"}, tmp_path / "runs")
    metrics = evaluate(run, tmp_path / "result.json")
    assert metrics["proxy"] is True
    assert json.loads((run / "val_metrics.json").read_text(encoding="utf-8"))["proxy"] is True
    predict(run)
    assert (run / "submission.csv").is_file()
