"""Audit a cleaned dataset handoff and save JSON and Markdown reports."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ie4483.data_audit import audit_data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="cleaned data root to audit")
    parser.add_argument("--compare", type=Path, help="other data root to compare file listings with")
    args = parser.parse_args()
    report = audit_data(args.root, args.compare)
    print((args.root / "audit.md").read_text(encoding="utf-8"))
    if not report["layout_ok"] or report["train_validation_duplicates"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
