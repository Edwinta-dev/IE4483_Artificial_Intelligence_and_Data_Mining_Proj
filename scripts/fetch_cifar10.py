"""Owner-run one-off CIFAR-10 download and extraction utility."""

from __future__ import annotations

import argparse
import tarfile
import urllib.request
from pathlib import Path

URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data", help="data root (IE4483_DATA is used by default)")
    args = parser.parse_args()
    import os
    root = Path(args.root if args.root != "data" else os.environ.get("IE4483_DATA", "data"))
    root.mkdir(parents=True, exist_ok=True)
    destination = root / "cifar-10-python.tar.gz"
    urllib.request.urlretrieve(URL, destination)
    with tarfile.open(destination, "r:gz") as archive:
        archive.extractall(root, filter="data")
    destination.unlink()


if __name__ == "__main__":
    main()
