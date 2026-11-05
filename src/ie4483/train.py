"""Training loop for configured ResNet transfer-learning experiments."""

from __future__ import annotations

import csv
import dataclasses
import logging
import math
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import Subset, TensorDataset

from ie4483.cifar import CIFAR10Dataset, cifar_transform
from ie4483.config import Config, config_to_dict
from ie4483.data import ImageFolderDataset, make_dataloader
from ie4483.features import extract_features, feature_cache_key, load_or_create_cache
from ie4483.imbalance import (
    FocalLoss,
    balanced_sampler,
    class_weights,
    imbalanced_subset,
    write_class_counts,
)
from ie4483.model import build_model
from ie4483.runs import get_device
from ie4483.seed import seed_everything
from ie4483.transforms import (
    build_eval_transform,
    build_train_transform,
    describe_transforms,
)

LOGGER = logging.getLogger(__name__)


def _metric(output: torch.Tensor, target: torch.Tensor) -> tuple[float, int]:
    return float(nn.functional.cross_entropy(output.float(), target).item()), int((output.argmax(1) == target).sum())


def _manifest(dataset) -> list[tuple[str, int]]:
    if isinstance(dataset, Subset):
        base = _manifest(dataset.dataset)
        return [base[int(index)] for index in dataset.indices]
    return [(str(path), int(label)) for path, label in dataset.samples]


def _epoch(model, loader, criterion, optimizer, device, scaler, accumulation, train: bool) -> tuple[float, float]:
    model.train(train)
    loss_sum = 0.0
    correct = 0
    count = 0
    context = torch.enable_grad if train else torch.inference_mode
    with context():
        for step, (images, labels) in enumerate(loader):
            images, labels = images.to(device), labels.to(device)
            with torch.autocast(device_type="cuda", enabled=(device.type == "cuda" and scaler.is_enabled())):
                logits = model(images)
                loss = criterion(logits, labels)
            if train:
                scaler.scale(loss / accumulation).backward()
                if (step + 1) % accumulation == 0 or step + 1 == len(loader):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
            batch_loss, batch_correct = _metric(logits.detach(), labels)
            loss_sum += batch_loss * len(labels)
            correct += batch_correct
            count += len(labels)
    return (loss_sum / count, correct / count) if count else (math.nan, math.nan)


class _HeadOnly(nn.Module):
    def __init__(self, head):
        super().__init__()
        self.head = head

    def forward(self, features):
        return self.head(features)


def _rng_state(loader):
    # Plain lists/tensors only, so checkpoints load with ``weights_only=True`` (no ndarray unpickling).
    name, keys, position, has_gauss, cached_gaussian = np.random.get_state()
    return {"python": random.getstate(), "numpy": (name, keys.tolist(), position, has_gauss, cached_gaussian),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "loader": loader.generator.get_state() if loader.generator is not None else None}


def _restore_rng(state, loader):
    random.setstate(state["python"])
    name, keys, position, has_gauss, cached_gaussian = state["numpy"]
    np.random.set_state((name, np.array(keys, dtype=np.uint32), position, has_gauss, cached_gaussian))
    # The checkpoint is loaded with map_location=device, which moves these ByteTensors
    # onto the GPU; the RNG setters only accept CPU tensors.
    torch.set_rng_state(state["torch"].cpu())
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])
    if loader.generator is not None and state.get("loader") is not None:
        loader.generator.set_state(state["loader"].cpu())


def _atomic_save(payload, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _fit(model, train_loader, val_loader, cfg, run_dir, device, *, checkpoint_model=None, resume=False,
         class_weights=None):
    model.to(device)
    accumulation = cfg.train.gradient_accumulation_steps
    head_params = list(model.head.parameters())
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad] if hasattr(model, "backbone") else []
    groups = [{"params": head_params, "lr": cfg.train.learning_rate}]
    if backbone_params:
        groups.append({"params": backbone_params, "lr": cfg.train.backbone_learning_rate})
    if cfg.train.optimizer == "sgd":
        optimizer = torch.optim.SGD(groups, momentum=cfg.train.momentum, weight_decay=cfg.train.weight_decay)
    elif cfg.train.optimizer == "adamw":
        optimizer = torch.optim.AdamW(groups, weight_decay=cfg.train.weight_decay)
    else:
        raise ValueError("train.optimizer must be adamw or sgd")
    schedule = cfg.train.scheduler
    if schedule == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, cfg.train.epochs - cfg.train.warmup_epochs)
        )
    elif schedule == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.train.step_size, gamma=cfg.train.gamma)
    elif schedule == "none":
        scheduler = None
    else:
        raise ValueError("train.scheduler must be cosine, step, or none")
    weight = class_weights if class_weights is not None else cfg.train.class_weights
    weight_tensor = torch.tensor(weight, dtype=torch.float, device=device) if weight else None
    if cfg.train.focal_gamma is not None:
        criterion = FocalLoss(cfg.train.focal_gamma, weight_tensor, cfg.train.label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss(weight=weight_tensor, label_smoothing=cfg.train.label_smoothing)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and cfg.train.mixed_precision)
    checkpoint_path = run_dir / "last.pt"
    start_epoch, best_acc, min_val_loss, stale = 1, -1.0, math.inf, 0
    if resume:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["training_model"])
        if checkpoint_model is not None:
            checkpoint_model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if scheduler is not None and checkpoint["scheduler"] is not None:
            scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_acc, min_val_loss, stale = checkpoint["best_val_acc"], checkpoint["min_val_loss"], checkpoint["patience"]
        _restore_rng(checkpoint["rng"], train_loader)
    metrics_path = run_dir / "metrics.csv"
    mode = "a" if resume else "w"
    with metrics_path.open(mode, newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        if not resume:
            writer.writerow(("epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr", "seconds"))
        for epoch in range(start_epoch, cfg.train.epochs + 1):
            if epoch <= cfg.train.warmup_epochs:
                factor = epoch / max(1, cfg.train.warmup_epochs)
                base_rates = [cfg.train.learning_rate] + ([cfg.train.backbone_learning_rate] if backbone_params else [])
                for group, base_lr in zip(optimizer.param_groups, base_rates):
                    group["lr"] = base_lr * factor
            started = time.perf_counter()
            train_loss, train_acc = _epoch(model, train_loader, criterion, optimizer, device, scaler, accumulation, True)
            val_loss, val_acc = _epoch(model, val_loader, criterion, optimizer, device, scaler, accumulation, False)
            lr = optimizer.param_groups[0]["lr"]
            writer.writerow((epoch, train_loss, train_acc, val_loss, val_acc, lr, time.perf_counter() - started))
            stream.flush()
            if val_acc > best_acc:
                best_acc = val_acc
                saved_model = checkpoint_model if checkpoint_model is not None else model
                _atomic_save({"model": saved_model.state_dict(), "epoch": epoch, "val_acc": val_acc,
                              "val_loss": val_loss}, run_dir / "best.pt")
            if val_loss < min_val_loss:
                min_val_loss = val_loss
                stale = 0
            else:
                stale += 1
            if scheduler is not None and epoch > cfg.train.warmup_epochs:
                scheduler.step()
            _atomic_save({"model": (checkpoint_model or model).state_dict(), "training_model": model.state_dict(),
                          "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict() if scheduler else None,
                          "scaler": scaler.state_dict(), "epoch": epoch, "best_val_acc": best_acc,
                          "min_val_loss": min_val_loss, "patience": stale, "rng": _rng_state(train_loader)},
                         checkpoint_path)
            if stale > cfg.train.patience:
                LOGGER.info("Early stopping after epoch %d", epoch)
                break
    return run_dir


def _dataset_labels(dataset) -> np.ndarray:
    """Integer labels of a dataset, or of the selected rows of a ``Subset``."""
    if isinstance(dataset, Subset):
        return dataset.dataset.labels[dataset.indices]
    return dataset.labels


def _transform_description(config: Config) -> str:
    if config.dataset_kind == "cifar10":
        augmentation = "RandomCrop(padding=4) -> RandomHorizontalFlip" if config.train.cifar_augment else "none"
        return f"CIFAR-10: Resize({config.data.image_size}) -> CIFAR Normalize; train: {augmentation}"
    return describe_transforms(config)


def _uses_augmentation(config: Config) -> bool:
    if config.dataset_kind == "cifar10":
        return config.train.cifar_augment
    augment = config.augment
    return any((augment.random_resized_crop, augment.horizontal_flip, augment.rotation_degrees, augment.color_jitter))


def _cifar_datasets(config: Config, run_dir: Path):
    seed, size = config.train.seed, config.data.image_size
    train_ds = CIFAR10Dataset(config.data_root, "train", cifar_transform(size, config.train.cifar_augment), seed)
    val_ds = CIFAR10Dataset(config.data_root, "val", cifar_transform(size), seed)
    if config.train.imbalance_fractions:
        train_ds, counts = imbalanced_subset(train_ds, config.train.imbalance_fractions, seed)
    else:
        counts = {label: int((train_ds.labels == label).sum()) for label in range(len(config.data.classes))}
    write_class_counts(run_dir / "class_counts.json", {config.data.classes[label]: count for label, count in counts.items()})
    return train_ds, val_ds


def _image_folder_datasets(config: Config, run_dir: Path):
    seed, classes = config.train.seed, config.data.classes
    train_ds = ImageFolderDataset(config.data_root, "train", classes, build_train_transform(config, seed),
                                  seed=seed, manifest_path=run_dir / "manifest.csv")
    val_ds = ImageFolderDataset(config.data_root, "val", classes, build_eval_transform(config), seed=seed)
    return train_ds, val_ds


def _cached_features(config: Config, model, train_ds, val_ds, run_root: Path, device: torch.device):
    """Pooled frozen-backbone features for both splits, built once per distinct input set."""
    manifest = [entry for ds in (train_ds, val_ds) for entry in _manifest(ds)]
    key = feature_cache_key({"model": dataclasses.asdict(config.model), "data": dataclasses.asdict(config.data)}, manifest)
    path = run_root / "feature_cache" / f"{key}.pt"
    train_cfg, eval_cfg = config.train, config.eval
    payload, reused = load_or_create_cache(path, lambda: {
        "train": extract_features(model, make_dataloader(train_ds, train_cfg.batch_size, train_cfg.num_workers,
                                                         seed=train_cfg.seed), device),
        "val": extract_features(model, make_dataloader(val_ds, eval_cfg.batch_size, eval_cfg.num_workers), device),
    })
    LOGGER.info("Using %s frozen feature cache at %s", "reused" if reused else "built", path)
    return payload["train"], payload["val"]


def _prepare_run_dir(config: Config, run_root: Path, resume: str | Path | None) -> Path:
    run_config = config_to_dict(config)
    if resume is not None:
        run_dir = Path(resume)
        if not (run_dir / "last.pt").is_file():
            raise FileNotFoundError(f"no last.pt checkpoint in {run_dir}")
        saved_config = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
        if saved_config != run_config:
            raise ValueError("resume configuration differs from the saved run configuration")
        return run_dir
    run_dir = run_root / f"{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}-{config.model.architecture}"
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(run_config, sort_keys=False), encoding="utf-8")
    return run_dir


def train(config: Config, run_root: str | Path | None = None, resume: str | Path | None = None) -> Path:
    """Train ``config`` into a fresh run folder under ``run_root`` (or resume ``resume``) and return it."""
    settings = config.train
    seed_everything(settings.seed, deterministic=settings.deterministic)
    run_root = Path(run_root) if run_root else Path("runs")
    run_dir = _prepare_run_dir(config, run_root, resume)
    (run_dir / "transforms.txt").write_text(_transform_description(config), encoding="utf-8")
    device = get_device()
    model = build_model(config).to(device)
    train_ds, val_ds = (_cifar_datasets if config.dataset_kind == "cifar10" else _image_folder_datasets)(config, run_dir)

    sampler = balanced_sampler(_dataset_labels(train_ds), settings.seed) if settings.weighted_sampler else None
    class_weight_values = (class_weights(_dataset_labels(train_ds), len(config.data.classes)).tolist()
                           if settings.class_weighted else None)
    frozen = model.freeze_mode == "frozen"
    augment = _uses_augmentation(config)
    use_cache = settings.feature_cache == "auto" and frozen and not augment
    fit_args = dict(resume=resume is not None, class_weights=class_weight_values)
    if use_cache:
        train_data, val_data = _cached_features(config, model, train_ds, val_ds, run_root, device)
        # Cached features keep the training-set order, so the balanced sampler's weights still line up.
        train_loader = make_dataloader(TensorDataset(*train_data), settings.batch_size, shuffle=sampler is None,
                                       seed=settings.seed, sampler=sampler)
        val_loader = make_dataloader(TensorDataset(*val_data), config.eval.batch_size)
        return _fit(_HeadOnly(model.head), train_loader, val_loader, config, run_dir, device,
                    checkpoint_model=model, **fit_args)
    LOGGER.info("Using image training path (cache=%s, frozen=%s, augment=%s)", settings.feature_cache, frozen, augment)
    train_loader = make_dataloader(train_ds, settings.batch_size, settings.num_workers, shuffle=sampler is None,
                                   seed=settings.seed, sampler=sampler)
    val_loader = make_dataloader(val_ds, config.eval.batch_size, config.eval.num_workers)
    return _fit(model, train_loader, val_loader, config, run_dir, device, **fit_args)
