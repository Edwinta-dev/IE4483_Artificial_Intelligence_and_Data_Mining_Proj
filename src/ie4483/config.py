"""Typed YAML configuration loading and resolved-config persistence."""

from __future__ import annotations

import dataclasses
import types
from collections.abc import Mapping
from pathlib import Path
from typing import Any, TypeVar, Union, get_args, get_origin, get_type_hints

import yaml

from ie4483.paths import resolve_data_root

T = TypeVar("T")

_MISSING = object()


def get_setting(source: Any, *path: str, default: Any = None) -> Any:
    """Follow ``path`` through attributes or mapping keys (a ``Config`` or a plain dict), else ``default``."""
    for key in path:
        source = source.get(key, _MISSING) if isinstance(source, Mapping) else getattr(source, key, _MISSING)
        if source is _MISSING:
            return default
    return source

FREEZE_MODES = ("frozen", "last_block", "full")
# Fields restricted to a fixed set of (lower-case) values: (section, field) -> allowed values.
CHOICES = {
    ("data", "dataset"): ("imagefolder", "cifar10"),
    ("model", "freeze_mode"): FREEZE_MODES,
    ("train", "optimizer"): ("adamw", "sgd"),
    ("train", "scheduler"): ("cosine", "step", "none"),
    ("train", "feature_cache"): ("auto", "off"),
}


def resolve_freeze_mode(freeze_mode: str | None, freeze_backbone: bool = True) -> str:
    """The effective freeze mode: ``freeze_mode`` wins; the legacy ``freeze_backbone`` flag is the fallback."""
    mode = freeze_mode if freeze_mode is not None else ("frozen" if freeze_backbone else "full")
    if mode not in FREEZE_MODES:
        raise ValueError(f"freeze_mode must be one of {', '.join(FREEZE_MODES)}")
    return mode


@dataclasses.dataclass
class DataConfig:
    root: str | None = None
    image_size: int = 224
    # "imagefolder" or "cifar10"; when unset it is inferred from the class count (ten classes means CIFAR-10).
    dataset: str | None = None
    classes: list[str] = dataclasses.field(default_factory=lambda: ["cat", "dog"])


@dataclasses.dataclass
class ModelConfig:
    architecture: str = "resnet18"
    pretrained: bool = True
    # Legacy flag: True means "frozen", False means "full". Superseded by freeze_mode.
    freeze_backbone: bool = True
    # Overrides freeze_backbone when set: "frozen", "last_block", or "full".
    freeze_mode: str | None = None
    num_classes: int = 2
    dropout: float = 0.0
    hidden_dim: int | None = None


@dataclasses.dataclass
class TrainConfig:
    batch_size: int = 32
    seed: int = 42
    epochs: int = 10
    learning_rate: float = 0.001
    mixed_precision: bool = True
    gradient_accumulation_steps: int = 1
    deterministic: bool = True
    # Optimisation.
    optimizer: str = "adamw"
    backbone_learning_rate: float = 0.0001
    momentum: float = 0.9
    weight_decay: float = 0.01
    scheduler: str = "cosine"
    step_size: int = 5
    gamma: float = 0.1
    warmup_epochs: int = 0
    patience: int = 3
    # Loss and class imbalance.
    label_smoothing: float = 0.0
    class_weights: list[float] | None = None
    class_weighted: bool = False
    weighted_sampler: bool = False
    focal_gamma: float | None = None
    imbalance_fractions: dict[str | int, float] | None = None
    # Data loading and caching.
    num_workers: int = 0
    feature_cache: str = "auto"
    cifar_augment: bool = True


@dataclasses.dataclass
class EvalConfig:
    batch_size: int = 32
    num_workers: int = 0


@dataclasses.dataclass
class AugmentConfig:
    random_resized_crop: bool = False
    random_resized_crop_scale: list[float] = dataclasses.field(default_factory=lambda: [0.8, 1.0])
    horizontal_flip: bool = False
    rotation_degrees: float = 0.0
    color_jitter: bool = False
    color_jitter_brightness: float = 0.2
    color_jitter_contrast: float = 0.2
    color_jitter_saturation: float = 0.2
    color_jitter_hue: float = 0.0


@dataclasses.dataclass
class Config:
    data: DataConfig = dataclasses.field(default_factory=DataConfig)
    model: ModelConfig = dataclasses.field(default_factory=ModelConfig)
    train: TrainConfig = dataclasses.field(default_factory=TrainConfig)
    eval: EvalConfig = dataclasses.field(default_factory=EvalConfig)
    augment: AugmentConfig = dataclasses.field(default_factory=AugmentConfig)

    @property
    def data_root(self) -> Path:
        """Resolved data directory using config, environment, then default."""
        return resolve_data_root(self.data.root)

    @property
    def dataset_kind(self) -> str:
        """``data.dataset`` if set, otherwise inferred from the class count."""
        return self.data.dataset or ("cifar10" if len(self.data.classes) == 10 else "imagefolder")


_INVALID = object()


def _convert(value: Any, annotation: Any) -> Any:
    """Return ``value`` coerced to ``annotation`` (ints become floats), or ``_INVALID``."""
    origin = get_origin(annotation)
    if origin in (Union, types.UnionType):
        for arm in get_args(annotation):
            converted = _convert(value, arm)
            if converted is not _INVALID:
                return converted
        return _INVALID
    if annotation is type(None):
        return None if value is None else _INVALID
    if origin is list:
        if not isinstance(value, list):
            return _INVALID
        items = [_convert(item, get_args(annotation)[0]) for item in value]
        return _INVALID if any(item is _INVALID for item in items) else items
    if origin is dict:
        if not isinstance(value, dict):
            return _INVALID
        key_type, value_type = get_args(annotation)
        pairs = [(_convert(k, key_type), _convert(v, value_type)) for k, v in value.items()]
        return _INVALID if any(k is _INVALID or v is _INVALID for k, v in pairs) else dict(pairs)
    if isinstance(value, bool):
        return value if annotation is bool else _INVALID
    if annotation is int:
        return value if isinstance(value, int) else _INVALID
    if annotation is float:
        return float(value) if isinstance(value, (int, float)) else _INVALID
    if annotation is str:
        return value if isinstance(value, str) else _INVALID
    return _INVALID


def _construct(cls: type[T], values: Any, path: str = "") -> T:
    if not isinstance(values, dict):
        raise TypeError(f"configuration section '{path or 'root'}' must be a mapping")
    hints = get_type_hints(cls)
    unknown = set(values) - {field.name for field in dataclasses.fields(cls)}
    if unknown:
        key = min(unknown)
        raise ValueError(f"unknown configuration key '{path + '.' if path else ''}{key}'")
    kwargs: dict[str, Any] = {}
    for name, value in values.items():
        if cls is Config:
            kwargs[name] = _construct(hints[name], value, name)
            continue
        converted = _convert(value, hints[name])
        if converted is _INVALID:
            raise ValueError(f"invalid type for '{path}.{name}'")
        choices = CHOICES.get((path, name))
        if isinstance(converted, str) and choices:
            converted = converted.lower()
        if choices and converted is not None and converted not in choices:
            raise ValueError(f"'{path}.{name}' must be one of {', '.join(choices)}")
        kwargs[name] = converted
    return cls(**kwargs)


def parse_overrides(overrides: list[str] | None) -> dict[str, Any]:
    """Parse dotted ``key=value`` overrides; values use YAML scalar syntax."""
    result: dict[str, Any] = {}
    for expression in overrides or []:
        if "=" not in expression:
            raise ValueError(f"override must be key=value: {expression!r}")
        key, raw = expression.split("=", 1)
        parts = key.split(".")
        if not all(parts):
            raise ValueError(f"invalid override key: {key!r}")
        current = result
        for part in parts[:-1]:
            child = current.setdefault(part, {})
            if not isinstance(child, dict):
                raise TypeError(f"conflicting override for {key!r}")
            current = child
        if parts[-1] in current:
            raise ValueError(f"duplicate override for {key!r}")
        current[parts[-1]] = yaml.safe_load(raw)
    return result


def _merge(base: dict[str, Any], changes: dict[str, Any]) -> dict[str, Any]:
    for key, value in changes.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: str | Path, overrides: list[str] | None = None) -> Config:
    """Load YAML into the typed config tree, applying optional dotted overrides."""
    with Path(path).open(encoding="utf-8") as stream:
        values = yaml.safe_load(stream) or {}
    if not isinstance(values, dict):
        raise TypeError("configuration root must be a mapping")
    return _construct(Config, _merge(values, parse_overrides(overrides)))


def config_to_dict(config: Config) -> dict[str, Any]:
    """Return a plain mapping suitable for YAML serialization."""
    return dataclasses.asdict(config)


def dump_config(config: Config, run_dir: str | Path) -> Path:
    """Write the resolved configuration to ``run_dir/config.yaml``."""
    destination = Path(run_dir) / "config.yaml"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(yaml.safe_dump(config_to_dict(config), sort_keys=False), encoding="utf-8")
    return destination
