"""Report artifacts for the configured transfer-learning model."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch import nn

from ie4483.model import model_summary


def _module_name(module: nn.Module) -> str:
    if isinstance(module, nn.Linear):
        return f"Linear {module.in_features} → {module.out_features}"
    if isinstance(module, nn.Dropout):
        return f"Dropout p={module.p:g}"
    if isinstance(module, nn.ReLU):
        return "ReLU"
    return type(module).__name__


def model_architecture_figure(model: nn.Module, image_size: int, output_dir: str | Path) -> tuple[Path, Path]:
    """Draw input, ResNet stages, pooled features, head, and scores."""
    summary = model_summary(model, image_size)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    height, width = image_size, image_size
    blocks = [("Input", f"3 × {height} × {width}", False)]
    blocks.append(("Stem", f"64 × {height // 2} × {width // 2}", model.freeze_mode == "full"))
    h, w = (height + 3) // 4, (width + 3) // 4
    for index, channels in enumerate((model.feature_dim // 8, model.feature_dim // 4,
                                      model.feature_dim // 2, model.feature_dim), start=1):
        if index > 1:
            h, w = (h + 1) // 2, (w + 1) // 2
        trainable = model.freeze_mode == "full" or (model.freeze_mode == "last_block" and index == 4)
        blocks.append((f"Layer {index}", f"{channels} × {h} × {w}", trainable))
    blocks.append(("Adaptive pool", f"{model.feature_dim} × 1 × 1", False))
    blocks.append(("Flatten", f"{model.feature_dim}", False))
    for layer in model.head:
        blocks.append((_module_name(layer), "", any(p.requires_grad for p in layer.parameters())))
    blocks.append(("Class scores", f"{summary['output_shape'][1]}", True))

    fig, axis = plt.subplots(figsize=(max(12, len(blocks) * 1.25), 4.2))
    axis.set_xlim(0, len(blocks))
    axis.set_ylim(0, 1)
    axis.axis("off")
    for i, (title, shape, trainable) in enumerate(blocks):
        color = "#f4a261" if trainable else "#9ecae1"
        axis.text(i + .5, .56, f"{title}\n{shape}\n{'trainable' if trainable else 'frozen'}",
                  ha="center", va="center", fontsize=8,
                  bbox={"boxstyle": "round,pad=0.45", "facecolor": color, "edgecolor": "#34495e"})
        if i < len(blocks) - 1:
            axis.annotate("", (i + .95, .56), (i + .72, .56), arrowprops={"arrowstyle": "->", "color": "#34495e"})
    fig.suptitle(f"{type(model.backbone).__name__} with custom classifier head · freeze mode: {model.freeze_mode}")
    fig.tight_layout()
    paths = (output / "model_architecture.png", output / "model_architecture.pdf")
    for path in paths:
        fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return paths


def write_model_card(model: nn.Module, config: Any, training: dict[str, Any], output_dir: str | Path) -> Path:
    """Write a Markdown model card from the built model and resolved settings."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summary = model_summary(model, config.data.image_size)
    layers = " → ".join(_module_name(layer) for layer in model.head)
    weights_source = "Torchvision ImageNet default weights" if config.model.pretrained else "None (random initialization)"
    schedule = training["scheduler"]
    if schedule == "cosine":
        schedule = f"cosine annealing over {config.train.epochs} epochs"
    elif schedule == "step":
        schedule = f"StepLR, step_size={training['step_size']}, gamma={training['gamma']}"
    else:
        schedule = "none"
    text = f"""# Model card

## Model
- Backbone: {type(model.backbone).__name__}
- Pretrained source: {weights_source}
- Freeze mode: {model.freeze_mode}
- Input shape: `{summary['input_shape']}`
- Output shape: `{summary['output_shape']}`
- Head layers: {layers}
- Loss: CrossEntropyLoss (label smoothing {training['label_smoothing']})
- Optimizer: {training['optimizer'].upper()}
- Learning rate: {config.train.learning_rate:g} (head); backbone {training['backbone_learning_rate']:g}
- Schedule: {schedule}
- Selection: best.pt by validation accuracy; early stopping by validation loss.
- Parameters: {summary['trainable_parameters']:,} trainable / {summary['total_parameters']:,} total
"""
    card = output / "model_card.md"
    card.write_text(text, encoding="utf-8")
    return card
