"""Typed YAML configuration loading and resolved-config persistence."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, TypeVar, get_type_hints

import yaml

from ie4483.paths import resolve_data_root

T = TypeVar("T")

FREEZE_MODES = ("frozen", "last_block", "full")


@dataclasses.dataclass
class DataConfig:
    root: str | None = None
    image_size: int = 224
    classes: list[str] = dataclasses.field(default_factory=lambda: ["cat", "dog"])


@dataclasses.dataclass
class ModelConfig:
    architecture: str = "resnet18"
    pretrained: bool = True
    freeze_backbone: bool = True
    # Overrides freeze_backbone when set: "frozen", "last_block", or "full".
    freeze_mode: str | None = None
    num_classes: int = 2


@dataclasses.dataclass
class TrainConfig:
    batch_size: int = 32
    seed: int = 42
    epochs: int = 10
    learning_rate: float = 0.001
    mixed_precision: bool = True
    gradient_accumulation_steps: int = 1
    deterministic: bool = True


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


def _construct(cls: type[T], values: Any, path: str = "") -> T:
    if not isinstance(values, dict):
        raise TypeError(f"configuration section '{path or 'root'}' must be a mapping")
    hints = get_type_hints(cls)
    allowed = {field.name for field in dataclasses.fields(cls)}
    unknown = set(values) - allowed
    if unknown:
        key = min(unknown)
        raise ValueError(f"unknown configuration key '{path + '.' if path else ''}{key}'")
    kwargs: dict[str, Any] = {}
    nested = {"data": DataConfig, "model": ModelConfig, "train": TrainConfig, "eval": EvalConfig, "augment": AugmentConfig}
    for name, value in values.items():
        if cls is Config:
            kwargs[name] = _construct(nested[name], value, name)
        else:
            expected = hints[name]
            if name in {"root", "freeze_mode"} and value is None:
                kwargs[name] = value
                continue
            origin = getattr(expected, "__origin__", None)
            valid = (isinstance(value, list) and all(isinstance(item, expected.__args__[0]) and not isinstance(item, bool) for item in value)) if origin is list else (
                isinstance(value, bool) if expected is bool else
                isinstance(value, int) and not isinstance(value, bool) if expected is int else
                isinstance(value, (int, float)) and not isinstance(value, bool) if expected is float else
                isinstance(value, str)
            )
            if not valid:
                raise ValueError(f"invalid type for '{path}.{name}'")
            if name == "freeze_mode" and value not in FREEZE_MODES:
                raise ValueError(f"'{path}.freeze_mode' must be one of {', '.join(FREEZE_MODES)}")
            kwargs[name] = float(value) if expected is float else value
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
