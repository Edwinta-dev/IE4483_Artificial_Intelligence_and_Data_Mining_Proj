"""Image preprocessing and reproducible training augmentation."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, NamedTuple

from torchvision import transforms

from ie4483.config import Config, get_setting
from ie4483.seed import seed_everything

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class _Step(NamedTuple):
    """One preprocessing step: its readable description and the factory that builds it."""

    text: str
    build: Callable[[], Any]


def _compose(steps: list[_Step]) -> transforms.Compose:
    """Build the steps into one flat ``Compose`` (a step may itself be a ``Compose``)."""
    built: list[Any] = []
    for step in steps:
        item = step.build()
        built.extend(item.transforms if isinstance(item, transforms.Compose) else [item])
    return transforms.Compose(built)


def _data_size(cfg: Config | Mapping[str, Any]) -> int:
    return int(get_setting(cfg, "data", "image_size", default=224))


def _normalize_step() -> _Step:
    return _Step(f"ToTensor() -> Normalize(mean={IMAGENET_MEAN}, std={IMAGENET_STD})",
                 lambda: transforms.Compose([transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]))


def _eval_steps(size: int) -> list[_Step]:
    return [_Step(f"Resize(shorter_side={size})", lambda: transforms.Resize(size)),
            _Step(f"CenterCrop(size={size})", lambda: transforms.CenterCrop(size))]


def _train_steps(cfg: Config | Mapping[str, Any]) -> list[_Step]:
    """The configured training pipeline before normalization; with augmentation off it equals evaluation."""
    size = _data_size(cfg)

    def value(name: str, default: Any) -> Any:
        return get_setting(cfg, "augment", name, default=default)

    steps: list[_Step] = []
    if value("random_resized_crop", False):
        scale = tuple(value("random_resized_crop_scale", [0.8, 1.0]))
        steps.append(_Step(f"RandomResizedCrop(size={size}, scale={scale})",
                           lambda: transforms.RandomResizedCrop(size, scale=scale)))
    else:
        steps.extend(_eval_steps(size))
    if value("horizontal_flip", False):
        steps.append(_Step("RandomHorizontalFlip(p=0.5)", lambda: transforms.RandomHorizontalFlip()))
    degrees = value("rotation_degrees", 0.0)
    if degrees:
        steps.append(_Step(f"RandomRotation(degrees={degrees})", lambda: transforms.RandomRotation(float(degrees))))
    if value("color_jitter", False):
        jitter = {name: value(f"color_jitter_{name}", default)
                  for name, default in (("brightness", 0.2), ("contrast", 0.2), ("saturation", 0.2), ("hue", 0.0))}
        steps.append(_Step(
            "ColorJitter(" + ", ".join(f"{name}={amount}" for name, amount in jitter.items()) + ")",
            lambda: transforms.ColorJitter(**{name: float(amount) for name, amount in jitter.items()})))
    return steps


def build_eval_transform(cfg: Config | Mapping[str, Any]) -> transforms.Compose:
    """Resize the short edge, center crop, and apply ImageNet normalization."""
    return _compose([*_eval_steps(_data_size(cfg)), _normalize_step()])


def build_train_transform(cfg: Config | Mapping[str, Any], seed: int | None = None) -> transforms.Compose:
    """Create the configured training pipeline; disabled augmentation is evaluation preprocessing."""
    seed_everything(int(get_setting(cfg, "train", "seed", default=42) if seed is None else seed))
    return _compose([*_train_steps(cfg), _normalize_step()])


def describe_transforms(cfg: Config | Mapping[str, Any]) -> str:
    """Return a readable, stable description of evaluation and training steps."""
    evaluation = " -> ".join(step.text for step in [*_eval_steps(_data_size(cfg)), _normalize_step()])
    training = " -> ".join(step.text for step in [*_train_steps(cfg), _normalize_step()])
    return f"Eval: {evaluation}\nTrain: {training}"
