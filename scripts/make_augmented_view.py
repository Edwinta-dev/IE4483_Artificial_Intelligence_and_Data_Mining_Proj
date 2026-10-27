"""Write a fixed augmented or mixed image-folder dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ie4483.augmented_view import make_augmented_view


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("augmented", "mixed"), required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    counts = make_augmented_view(args.input, args.output, args.mode, args.seed)
    print(f"training adjustments written: {sum(counts.values())}")
    for adjustment, count in counts.items():
        print(f"{adjustment}: {count}")


if __name__ == "__main__":
    main()
