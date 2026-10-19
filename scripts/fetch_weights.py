"""Owner-run command to fetch torchvision's default ResNet weights."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from torchvision.models import ResNet18_Weights, ResNet50_Weights

from ie4483.paths import weights_cache_dir


def main() -> None:
    for weights in (ResNet18_Weights.DEFAULT, ResNet50_Weights.DEFAULT):
        weights.get_state_dict(progress=True, check_hash=True)
    print(f"Cached ResNet-18 and ResNet-50 weights in {weights_cache_dir()}")


if __name__ == "__main__":
    main()
