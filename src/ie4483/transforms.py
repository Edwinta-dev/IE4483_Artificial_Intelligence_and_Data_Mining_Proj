"""Image preprocessing and reproducible training augmentation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from torchvision import transforms

from ie4483.config import Config
from ie4483.seed import seed_everything

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _read(cfg: Config | Mapping[str, Any], section: str, name: str, default: Any) -> Any:
    part = cfg.get(section, {}) if isinstance(cfg, Mapping) else getattr(cfg, section)
    return part.get(name, default) if isinstance(part, Mapping) else getattr(part, name, default)


def _data_size(cfg: Config | Mapping[str, Any]) -> int:
    return int(_read(cfg, "data", "image_size", 224))


def _normalize() -> list[Any]:
    return [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]


def build_eval_transform(cfg: Config | Mapping[str, Any]) -> transforms.Compose:
    """Resize the short edge, center crop, and apply ImageNet normalization."""
    size = _data_size(cfg)
    return transforms.Compose([transforms.Resize(size), transforms.CenterCrop(size), *_normalize()])


def build_train_transform(cfg: Config | Mapping[str, Any], seed: int | None = None) -> transforms.Compose:
    """Create the configured training pipeline; disabled augmentation is evaluation preprocessing."""
    aug = cfg.get("augment", {}) if isinstance(cfg, Mapping) else cfg.augment
    def value(name: str, default: Any) -> Any:
        return aug.get(name, default) if isinstance(aug, Mapping) else getattr(aug, name, default)

    seed_everything(int(_read(cfg, "train", "seed", 42) if seed is None else seed))
    size = _data_size(cfg)
    steps: list[Any] = []
    if value("random_resized_crop", False):
        steps.append(transforms.RandomResizedCrop(size, scale=tuple(value("random_resized_crop_scale", [0.8, 1.0]))))
    else:
        steps.extend([transforms.Resize(size), transforms.CenterCrop(size)])
    if value("horizontal_flip", False):
        steps.append(transforms.RandomHorizontalFlip())
    degrees = float(value("rotation_degrees", 0.0))
    if degrees:
        steps.append(transforms.RandomRotation(degrees))
    if value("color_jitter", False):
        steps.append(transforms.ColorJitter(
            brightness=float(value("color_jitter_brightness", 0.2)),
            contrast=float(value("color_jitter_contrast", 0.2)),
            saturation=float(value("color_jitter_saturation", 0.2)),
            hue=float(value("color_jitter_hue", 0.0)),
        ))
    steps.extend(_normalize())
    return transforms.Compose(steps)


def describe_transforms(cfg: Config | Mapping[str, Any]) -> str:
    """Return a readable, stable description of evaluation and training steps."""
    size = _data_size(cfg)
    aug = cfg.get("augment", {}) if isinstance(cfg, Mapping) else cfg.augment
    def value(name: str, default: Any) -> Any:
        return aug.get(name, default) if isinstance(aug, Mapping) else getattr(aug, name, default)
    if value("random_resized_crop", False):
        train_steps = [f"RandomResizedCrop(size={size}, scale={tuple(value('random_resized_crop_scale', [0.8, 1.0]))})"]
    else:
        train_steps = [f"Resize(shorter_side={size})", f"CenterCrop(size={size})"]
    if value("horizontal_flip", False):
        train_steps.append("RandomHorizontalFlip(p=0.5)")
    if value("rotation_degrees", 0.0):
        train_steps.append(f"RandomRotation(degrees={value('rotation_degrees', 0.0)})")
    if value("color_jitter", False):
        train_steps.append("ColorJitter(brightness={}, contrast={}, saturation={}, hue={})".format(
            value("color_jitter_brightness", 0.2), value("color_jitter_contrast", 0.2),
            value("color_jitter_saturation", 0.2), value("color_jitter_hue", 0.0)))
    norm = f"ToTensor() -> Normalize(mean={IMAGENET_MEAN}, std={IMAGENET_STD})"
    train_steps.append(norm)
    return (f"Eval: Resize(shorter_side={size}) -> CenterCrop(size={size}) -> {norm}\n"
            f"Train: {' -> '.join(train_steps)}")
