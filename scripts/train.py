"""Run a configured training job."""

from __future__ import annotations

import argparse
import logging

from ie4483.train import load_training_config, train


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--resume", metavar="RUN_DIR", help="resume training from RUN_DIR/last.pt")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config, extra = load_training_config(args.config, args.set)
    print(train(config, extra, resume=args.resume))


if __name__ == "__main__":
    main()
