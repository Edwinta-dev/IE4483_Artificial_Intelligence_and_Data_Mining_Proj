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


def _cifar_config(tmp_path):
    config_path = tmp_path / "cifar.yaml"
    config_path.write_text(
        "data:\n  root: '" + str(tmp_path).replace("\\", "/") + "'\n  image_size: 32\n"
        "  classes: [airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck]\n"
        "model:\n  architecture: resnet18\n  pretrained: false\n  freeze_backbone: true\n  num_classes: 10\n"
        "train:\n  seed: 7\n  epochs: 1\n  batch_size: 10\n  mixed_precision: false\n"
        "eval:\n  batch_size: 10\n", encoding="utf-8")
    return load_training_config(config_path)


def test_frozen_cifar_without_augmentation_uses_feature_cache(tmp_path, monkeypatch):
    import ie4483.train as training

    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    _make_cifar(tmp_path)
    calls = []
    original = training.extract_features
    monkeypatch.setattr(training, "extract_features", lambda *a: calls.append(1) or original(*a))
    config, extra = _cifar_config(tmp_path)
    extra.update(cifar_augment=False, imbalance_fractions={0: 0.5})
    train(config, dict(extra), tmp_path / "runs")
    train(config, dict(extra), tmp_path / "runs")
    assert len(calls) == 2  # train + val built once, reused on the second run
    assert len(list((tmp_path / "runs" / "feature_cache").glob("*.pt"))) == 1


def test_cifar_cache_identities_are_stable_and_split_specific(tmp_path):
    _make_cifar(tmp_path)
    train_ids = [path for path, _ in CIFAR10Dataset(tmp_path, "train", seed=5).samples]
    val_ids = [path for path, _ in CIFAR10Dataset(tmp_path, "val", seed=5).samples]
    test_ids = [path for path, _ in CIFAR10Dataset(tmp_path, "test").samples]
    assert train_ids == [path for path, _ in CIFAR10Dataset(tmp_path, "train", seed=5).samples]
    assert not set(train_ids) & set(val_ids)
    assert not set(test_ids) & set(train_ids + val_ids)
