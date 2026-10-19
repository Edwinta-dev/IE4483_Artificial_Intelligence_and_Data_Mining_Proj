"""Generate report architecture figures and a model card."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ie4483.figures import model_architecture_figure, write_model_card
from ie4483.model import build_model
from ie4483.seed import seed_everything
from ie4483.train import load_training_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="run folder for report artifacts")
    args = parser.parse_args()
    config, training = load_training_config(args.config)
    seed_everything(config.train.seed)
    model = build_model(config)
    model_architecture_figure(model, config.data.image_size, args.output)
    write_model_card(model, config, training, args.output)


if __name__ == "__main__":
    main()
