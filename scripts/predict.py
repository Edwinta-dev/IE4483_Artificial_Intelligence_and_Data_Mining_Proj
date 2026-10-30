"""Predict labels for a run's unlabeled test split."""

from __future__ import annotations

import argparse

import torch

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ie4483.data import UnlabelledTestDataset, make_dataloader
from ie4483.model import build_model
from ie4483.seed import seed_everything
from ie4483.submission import write_prediction_csvs
from ie4483.train import load_training_config
from ie4483.transforms import build_eval_transform


def predict(run: str, tta: str | None = None) -> None:
    """Predict the run's test images and write both CSV artifacts into its folder."""
    if tta not in (None, "flip"):
        raise ValueError("tta must be None or 'flip'")
    run_dir = Path(run)
    config, extra = load_training_config(run_dir / "config.yaml")
    classes = list(config.data.classes)
    if len(classes) != 2 or classes[1] != "dog":
        raise ValueError("run class order must place dog at index 1")
    seed_everything(config.train.seed)
    dataset = UnlabelledTestDataset(config.data_root, build_eval_transform(config))
    if dataset.unreadable_count:
        raise ValueError("test split contains unreadable images")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config).to(device)
    checkpoint = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    loader = make_dataloader(dataset, config.eval.batch_size, extra["num_workers"])
    image_ids: list[int] = []
    probabilities: list[list[float]] = []
    with torch.inference_mode():
        for images, batch_ids in loader:
            images = images.to(device)
            probs = model(images).float().softmax(dim=1)
            if tta == "flip":
                flipped = model(torch.flip(images, dims=(3,))).float().softmax(dim=1)
                probs = (probs + flipped) / 2
            image_ids.extend(int(value) for value in batch_ids.tolist())
            probabilities.extend(probs.cpu().tolist())
    write_prediction_csvs(run_dir, image_ids, probabilities, classes, len(dataset))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="run folder containing config.yaml and best.pt")
    parser.add_argument("--tta", choices=("flip",), default=None,
                        help="average probabilities for original and horizontally flipped images")
    args = parser.parse_args()
    predict(args.run, args.tta)


if __name__ == "__main__":
    main()
