"""Run and summarize training across fixed random seeds."""

from __future__ import annotations

import argparse
from datetime import datetime

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ie4483.experiments import run_seeds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--seeds", required=True, nargs="+", type=int)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    output = args.output or Path("runs") / f"seeds-{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}"
    for row in run_seeds(args.config, args.seeds, output, args.set):
        print(f"seed={row['seed']} val_accuracy={row['val_accuracy']:.6f} run={row['run']}")
    print(f"Summary: {Path(output) / 'summary.md'}")


if __name__ == "__main__":
    main()
