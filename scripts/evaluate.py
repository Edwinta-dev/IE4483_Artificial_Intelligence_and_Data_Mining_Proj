"""Write validation metrics and report figures for a training run."""

from __future__ import annotations

import argparse

from ie4483.evaluate import evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="run folder containing config.yaml and best.pt")
    parser.add_argument("--results", default="results/latest.json", help="latest-result JSON path")
    parser.add_argument("--split", choices=("val", "test"), default="val", help="labeled split to evaluate")
    args = parser.parse_args()
    print(evaluate(args.run, args.results, args.split))


if __name__ == "__main__":
    main()
