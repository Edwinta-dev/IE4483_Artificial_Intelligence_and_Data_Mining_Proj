"""Small shared type checks (``bool`` is deliberately not an ``int`` or a number here)."""

from __future__ import annotations

from typing import Any


def is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)
