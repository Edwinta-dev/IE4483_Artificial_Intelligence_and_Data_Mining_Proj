import pickle

import numpy as np

from ie4483.cifar import CIFAR10_CLASSES, CIFAR10Dataset
from ie4483.evaluate import evaluate
from ie4483.train import load_training_config, train


def _make_cifar(root):
    folder = root / "cifar-10-batches-py"
    folder.mkdir(parents=True)
    for batch in range(1, 6):
        labels = [class_id for class_id in range(10) for _ in range(2)]
        pixels = np.random.default_rng(batch).integers(0, 256, size=(len(labels), 3072), dtype=np.uint8)
        with (folder / f"data_batch_{batch}").open("wb") as stream:
            pickle.dump({b"data": pixels, b"labels": labels}, stream)
    labels = [class_id for class_id in range(10) for _ in range(2)]
    with (folder / "test_batch").open("wb") as stream:
        pickle.dump({b"data": np.zeros((len(labels), 3072), dtype=np.uint8), b"labels": labels}, stream)


def test_cifar_loader_seeded_split_and_official_test(tmp_path):
    _make_cifar(tmp_path)
    first = CIFAR10Dataset(tmp_path, "train", seed=5, val_fraction=0.5)
    second = CIFAR10Dataset(tmp_path, "train", seed=5, val_fraction=0.5)
    val = CIFAR10Dataset(tmp_path, "val", seed=5, val_fraction=0.5)
    test = CIFAR10Dataset(tmp_path, "test")
    assert len(first) == 50 and len(val) == 50 and len(test) == 20
    assert first.labels.tolist() == second.labels.tolist()
    assert np.array_equal(first.images, second.images)
    assert not any(np.array_equal(image, held_out) for image in first.images for held_out in val.images)
    assert first[0][0].shape == (3, 32, 32)
    assert sorted(set(val.labels.tolist())) == list(range(10))
    assert len(CIFAR10_CLASSES) == 10


def test_cifar_one_epoch_cpu_and_test_evaluation(tmp_path, monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    _make_cifar(tmp_path)
    config_path = tmp_path / "cifar.yaml"
    config_path.write_text(
        "data:\n  root: '" + str(tmp_path).replace("\\", "/") + "'\n  image_size: 32\n"
        "  classes: [airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck]\n"
        "model:\n  architecture: resnet18\n  pretrained: false\n  freeze_backbone: true\n  num_classes: 10\n"
        "train:\n  seed: 7\n  epochs: 1\n  batch_size: 10\n  mixed_precision: false\n"
        "eval:\n  batch_size: 10\n", encoding="utf-8")
    config, extra = load_training_config(config_path)
    extra["feature_cache"] = "off"
    run = train(config, extra, tmp_path / "runs")
    result_path = tmp_path / "result.json"
    metrics = evaluate(run, result_path, "test")
    assert metrics["count"] == 20
    assert len(metrics["per_class"]) == 10
    assert len(metrics["confusion_matrix"]) == 10
    assert all(len(row) == 10 for row in metrics["confusion_matrix"])
    assert (run / "test_confusion_matrix.png").is_file()
    assert (run / "test_metrics.json").is_file()
    assert '"split": "test"' in result_path.read_text(encoding="utf-8")
