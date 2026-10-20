"""Measure safe CUDA training batches for the supported ResNet freeze modes."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ie4483.model import build_model
from ie4483.seed import seed_everything
from ie4483.vram import (
    EFFECTIVE_BATCH_SIZE,
    accumulation_steps,
    check_fits_device,
    find_max_batch,
    write_probe_json,
)

MODES = ("frozen", "last_block", "full")
ARCHITECTURES = ("resnet18", "resnet50")


def _training_step(model: torch.nn.Module, batch_size: int, image_size: int, optimizer) -> int:
    device = torch.device("cuda")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    images = torch.rand((batch_size, 3, image_size, image_size), device=device)
    labels = torch.arange(batch_size, device=device).remainder(2)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        logits = model(images)
        loss = torch.nn.functional.cross_entropy(logits.float(), labels)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    del images, labels, logits, loss
    total = torch.cuda.get_device_properties(device).total_memory
    return check_fits_device(torch.cuda.max_memory_allocated(), total)


def _mode_for_config(model_cfg: dict[str, Any]) -> str:
    mode = model_cfg.get("freeze_mode")
    if mode is None:
        mode = "frozen" if model_cfg.get("freeze_backbone", True) else "full"
    if mode not in MODES:
        raise ValueError(f"unsupported freeze mode in config: {mode}")
    return mode


def _write_configs(paths: list[Path], results: dict[tuple[str, str], dict[str, Any]]) -> None:
    for path in paths:
        config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        model = config.setdefault("model", {})
        train = config.setdefault("train", {})
        key = (model.get("architecture", "resnet18"), _mode_for_config(model))
        result = results[key]
        if result["safe_batch_size"]:
            train["batch_size"] = result["safe_batch_size"]
            train["gradient_accumulation_steps"] = result["gradient_accumulation_steps"]
            path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def probe(
    image_size: int,
    start: int = 1,
    limit: int = 128,
    safety_margin: float = 0.8,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Run synthetic complete optimizer steps for every architecture and mode."""
    seed_everything(seed)
    results: list[dict[str, Any]] = []
    for architecture in ARCHITECTURES:
        for freeze_mode in MODES:
            model = build_model({
                "architecture": architecture,
                "pretrained": False,
                "freeze_mode": freeze_mode,
            }).to("cuda")
            parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
            optimizer = torch.optim.AdamW(parameters)
            peaks: dict[int, int] = {}

            def step(batch_size: int, model=model, optimizer=optimizer, peaks=peaks) -> None:
                torch.cuda.reset_peak_memory_stats()
                peaks[batch_size] = _training_step(model, batch_size, image_size, optimizer)

            safe = find_max_batch(step, start, limit, safety_margin)
            largest = max((size for size in peaks), default=0)
            results.append({
                "architecture": architecture,
                "freeze_mode": freeze_mode,
                "image_size": image_size,
                "largest_working_batch_size": largest,
                "safe_batch_size": safe,
                "peak_memory_bytes": peaks.get(largest),
                "gradient_accumulation_steps": accumulation_steps(safe),
            })
            del optimizer, model
            torch.cuda.empty_cache()
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", action="append", type=Path, default=[], help="YAML config; repeatable")
    parser.add_argument("--output", type=Path, default=Path("runs/vram_probe.json"))
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--limit", type=int, default=128)
    parser.add_argument("--safety-margin", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--write-configs", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        print("CUDA is unavailable; VRAM probe was not run.")
        return
    if args.write_configs and not args.config:
        parser.error("--write-configs requires at least one --config")
    configs = [yaml.safe_load(path.read_text(encoding="utf-8")) or {} for path in args.config]
    image_size = int(configs[0].get("data", {}).get("image_size", 224)) if configs else 224
    if any(int(config.get("data", {}).get("image_size", image_size)) != image_size for config in configs):
        parser.error("all supplied configs must use the same data.image_size")
    rows = probe(image_size, args.start, args.limit, args.safety_margin, args.seed)
    write_probe_json(args.output, rows, image_size, args.safety_margin, args.seed)
    print(f"Wrote {args.output} for effective batch {EFFECTIVE_BATCH_SIZE}.")
    if args.write_configs:
        result_map = {(row["architecture"], row["freeze_mode"]): row for row in rows}
        _write_configs(args.config, result_map)


if __name__ == "__main__":
    main()
