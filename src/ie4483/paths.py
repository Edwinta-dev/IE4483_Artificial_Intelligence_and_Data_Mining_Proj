"""Shared paths used by owner setup and pipeline commands."""

from __future__ import annotations

import os
from pathlib import Path


def resolve_data_root(configured_root: str | Path | None = None) -> Path:
    """Resolve data root in config, environment, default precedence order."""
    root = configured_root or os.environ.get("IE4483_DATA") or "data"
    return Path(root).expanduser()


def runs_dir() -> Path:
    """Return the repository's run-output directory."""
    return Path(__file__).resolve().parents[2] / "runs"


def weights_cache_dir() -> Path:
    """Return torch hub's checkpoints directory, honoring TORCH_HOME."""
    torch_home = os.environ.get("TORCH_HOME")
    if torch_home:
        hub = Path(torch_home).expanduser() / "hub"
    else:
        hub = Path.home() / ".cache" / "torch" / "hub"
    return hub / "checkpoints"
