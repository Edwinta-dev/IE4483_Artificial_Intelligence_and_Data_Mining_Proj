"""Helpers shared by stages that reuse a finished training run."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from ie4483.config import Config
from ie4483.model import build_model


def get_device() -> torch.device:
    """CUDA when available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_best_model(run_dir: str | Path, config: Config, device: torch.device) -> nn.Module:
    """Rebuild the run's model, load ``best.pt`` (tensors only, no arbitrary unpickling) and set eval mode."""
    model = build_model(config).to(device)
    checkpoint = torch.load(Path(run_dir) / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    return model.eval()
