"""Training loop for configured ResNet transfer-learning experiments."""

from __future__ import annotations

import csv
import dataclasses
import logging
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ie4483.cifar import CIFAR10Dataset, cifar_transform
from ie4483.config import Config, TrainConfig, config_to_dict, load_config
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
from ie4483.seed import seed_everything
from ie4483.transforms import (
    build_eval_transform,
    build_train_transform,
    describe_transforms,
)

LOGGER = logging.getLogger(__name__)
_EXTRA_DEFAULTS = {
    "optimizer": "adamw", "backbone_learning_rate": 0.0001, "momentum": 0.9,
    "weight_decay": 0.01, "scheduler": "cosine", "step_size": 5,
    "gamma": 0.1, "warmup_epochs": 0, "patience": 3,
    "label_smoothing": 0.0, "class_weights": None, "num_workers": 0,
    "feature_cache": "auto",
    "imbalance_fractions": None, "class_weighted": False, "weighted_sampler": False,
    "focal_gamma": None, "cifar_augment": True,
}


def _apply_overrides(raw: dict[str, Any], overrides: list[str] | None) -> None:
    for expression in overrides or []:
        key, sep, value = expression.partition("=")
        if not sep:
            raise ValueError(f"override must be key=value: {expression!r}")
        node = raw
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = yaml.safe_load(value)


def load_training_config(path: str | Path, overrides: list[str] | None = None) -> tuple[Config, dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as stream:
        raw = yaml.safe_load(stream) or {}
    _apply_overrides(raw, overrides)
    extra = dict(_EXTRA_DEFAULTS)
    extra.update(raw.get("train", {}))
    # Validate the established typed configuration using its strict loader.
    typed_fields = {field.name for field in dataclasses.fields(TrainConfig)}
    filtered = {**raw, "train": {key: value for key, value in raw.get("train", {}).items() if key in typed_fields}}
    temporary = Path(path).with_name(f".{Path(path).stem}.validated.tmp.yaml")
    try:
        temporary.write_text(yaml.safe_dump(filtered), encoding="utf-8")
        config = load_config(temporary)
    finally:
        temporary.unlink(missing_ok=True)
    return config, extra


def _metric(output: torch.Tensor, target: torch.Tensor) -> tuple[float, int]:
    return float(nn.functional.cross_entropy(output.float(), target).item()), int((output.argmax(1) == target).sum())


def _manifest(dataset) -> list[tuple[str, int]]:
    if isinstance(dataset, torch.utils.data.Subset):
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


def _fit_head(model, train_data, val_data, cfg, extra, run_dir, device, steps):
    train_loader = DataLoader(TensorDataset(*train_data), batch_size=cfg.train.batch_size, shuffle=True,
                              generator=torch.Generator().manual_seed(cfg.train.seed))
    val_loader = DataLoader(TensorDataset(*val_data), batch_size=cfg.eval.batch_size)
    return _fit(_HeadOnly(model.head), train_loader, val_loader, cfg, extra, run_dir, device, steps,
                checkpoint_model=model)


def _fit(model, train_loader, val_loader, cfg, extra, run_dir, device, accumulation=None, checkpoint_model=None):
    model.to(device)
    accumulation = accumulation or cfg.train.gradient_accumulation_steps
    head_params = list(model.head.parameters())
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad] if hasattr(model, "backbone") else []
    groups = [{"params": head_params, "lr": cfg.train.learning_rate}]
    if backbone_params:
        groups.append({"params": backbone_params, "lr": extra["backbone_learning_rate"]})
    if extra["optimizer"].lower() == "sgd":
        optimizer = torch.optim.SGD(groups, momentum=extra["momentum"], weight_decay=extra["weight_decay"])
    elif extra["optimizer"].lower() == "adamw":
        optimizer = torch.optim.AdamW(groups, weight_decay=extra["weight_decay"])
    else:
        raise ValueError("train.optimizer must be adamw or sgd")
    schedule = extra["scheduler"].lower()
    if schedule == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, cfg.train.epochs))
    elif schedule == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=extra["step_size"], gamma=extra["gamma"])
    elif schedule == "none":
        scheduler = None
    else:
        raise ValueError("train.scheduler must be cosine, step, or none")
    weight = extra["class_weights"]
    weight_tensor = torch.tensor(weight, dtype=torch.float, device=device) if weight else None
    if extra.get("focal_gamma") is not None:
        criterion = FocalLoss(float(extra["focal_gamma"]), weight_tensor, extra["label_smoothing"])
    else:
        criterion = nn.CrossEntropyLoss(weight=weight_tensor, label_smoothing=extra["label_smoothing"])
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and cfg.train.mixed_precision)
    with (run_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr", "seconds"))
        best_acc, min_val_loss, stale = -1.0, math.inf, 0
        for epoch in range(1, cfg.train.epochs + 1):
            if epoch <= extra["warmup_epochs"]:
                factor = epoch / max(1, extra["warmup_epochs"])
                base_rates = [cfg.train.learning_rate] + ([extra["backbone_learning_rate"]] if backbone_params else [])
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
                torch.save({"model": saved_model.state_dict(), "epoch": epoch, "val_acc": val_acc,
                            "val_loss": val_loss}, run_dir / "best.pt")
            if val_loss < min_val_loss:
                min_val_loss = val_loss
                stale = 0
            else:
                stale += 1
            if scheduler is not None and epoch > extra["warmup_epochs"]:
                scheduler.step()
            if stale > extra["patience"]:
                LOGGER.info("Early stopping after epoch %d", epoch)
                break
    return run_dir


def train(config: Config, extra: dict[str, Any] | None = None, run_root: str | Path | None = None) -> Path:
    extra = {**_EXTRA_DEFAULTS, **(extra or {})}
    seed_everything(config.train.seed)
    run_root = Path(run_root) if run_root else Path("runs")
    name = getattr(config.model, "name", config.model.architecture)
    run_dir = run_root / f"{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}-{name}"
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "config.yaml").write_text(yaml.safe_dump({**config_to_dict(config), "train": {
        **config_to_dict(config)["train"], **extra}}, sort_keys=False), encoding="utf-8")
    cifar_train = "RandomCrop(padding=4) -> RandomHorizontalFlip" if extra["cifar_augment"] else "none"
    transform_description = (f"CIFAR-10: Resize({config.data.image_size}) -> CIFAR Normalize; train: {cifar_train}"
                             if len(config.data.classes) == 10 else describe_transforms(config))
    (run_dir / "transforms.txt").write_text(transform_description, encoding="utf-8")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config)
    model.to(device)
    manifest = run_dir / "manifest.csv"
    if len(config.data.classes) == 10:
        train_ds = CIFAR10Dataset(config.data_root, "train", cifar_transform(config.data.image_size, bool(extra["cifar_augment"])),
                                 config.train.seed)
        val_ds = CIFAR10Dataset(config.data_root, "val", cifar_transform(config.data.image_size), config.train.seed)
        if extra["imbalance_fractions"]:
            train_ds, counts = imbalanced_subset(train_ds, extra["imbalance_fractions"], config.train.seed)
        else:
            counts = {int(label): int((train_ds.labels == label).sum()) for label in range(len(config.data.classes))}
        write_class_counts(run_dir / "class_counts.json",
                           {config.data.classes[label]: count for label, count in counts.items()})
    else:
        train_transform = build_train_transform(config, config.train.seed)
        eval_transform = build_eval_transform(config)
        train_ds = ImageFolderDataset(config.data_root, "train", config.data.classes, train_transform,
                                      seed=config.train.seed, manifest_path=manifest)
        val_ds = ImageFolderDataset(config.data_root, "val", config.data.classes, eval_transform, seed=config.train.seed)
    batch = config.train.batch_size
    sampler = None
    if extra["weighted_sampler"]:
        labels = train_ds.dataset.labels[train_ds.indices] if isinstance(train_ds, torch.utils.data.Subset) else train_ds.labels
        sampler = balanced_sampler(labels, config.train.seed)
    train_loader = (DataLoader(train_ds, batch_size=batch, sampler=sampler, num_workers=extra["num_workers"])
                    if sampler is not None else make_dataloader(train_ds, batch, extra["num_workers"],
                                                                shuffle=True, seed=config.train.seed))
    if extra["class_weighted"]:
        labels = train_ds.dataset.labels[train_ds.indices] if isinstance(train_ds, torch.utils.data.Subset) else train_ds.labels
        weights = class_weights(labels, len(config.data.classes)).tolist()
        extra["class_weights"] = weights
    val_loader = make_dataloader(val_ds, config.eval.batch_size, extra["num_workers"])
    frozen = getattr(model, "freeze_mode", "frozen") == "frozen"
    is_cifar = len(config.data.classes) == 10
    augment = bool(extra["cifar_augment"]) if is_cifar else any((config.augment.random_resized_crop, config.augment.horizontal_flip,
                   config.augment.rotation_degrees, config.augment.color_jitter))
    cache_mode = extra["feature_cache"]
    if cache_mode not in {"auto", "off"}:
        raise ValueError("train.feature_cache must be auto or off")
    if cache_mode == "auto" and frozen and not augment:
        manifest_values = [entry for ds in (train_ds, val_ds) for entry in _manifest(ds)]
        key = feature_cache_key({"model": dataclasses.asdict(config.model), "data": dataclasses.asdict(config.data)}, manifest_values)
        cache_dir = run_root / "feature_cache"
        cache_path = cache_dir / f"{key}.pt"
        payload, reused = load_or_create_cache(cache_path, lambda: {
            "train": extract_features(model, make_dataloader(train_ds, batch, extra["num_workers"], seed=config.train.seed), device),
            "val": extract_features(model, make_dataloader(val_ds, config.eval.batch_size, extra["num_workers"]), device),
        })
        LOGGER.info("Using %s frozen feature cache at %s", "reused" if reused else "built", cache_path)
        return _fit_head(model, payload["train"], payload["val"], config, extra, run_dir, device,
                         config.train.gradient_accumulation_steps)
    LOGGER.info("Using image training path (cache=%s, frozen=%s, augment=%s)", cache_mode, frozen, augment)
    return _fit(model, train_loader, val_loader, config, extra, run_dir, device)
