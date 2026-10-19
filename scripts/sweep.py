"""Run and summarize a resumable parameter grid."""

from __future__ import annotations

import argparse

from ie4483.experiments import sweep


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", required=True, help="sweep YAML file")
    parser.add_argument("--output", default=None, help="output folder (default runs/sweeps/<name>)")
    args = parser.parse_args()
    rows = sweep(args.sweep, args.output)
    print(f"Completed {len(rows)} seed runs; summary: {args.output or 'runs/sweeps/' + args.sweep.rsplit('/', 1)[-1].removesuffix('.yaml')}")


if __name__ == "__main__":
    main()
