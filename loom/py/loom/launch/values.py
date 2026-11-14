# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tolerant readers for documents whose publishers may add fields.

Published profiles outlive the reader: a card may omit a whole section, a
measurement record may add metrics or gates, and a hand-edited card may hold a
string where a count is expected. Every reader here answers "absent" instead of
raising, so one unexpected field cannot make an otherwise usable profile
unreadable.
"""

from __future__ import annotations


def dict_value(value: object) -> dict:
    """Returns `value` when it is an object, else an empty object."""
    return value if isinstance(value, dict) else {}


def list_value(value: object) -> list:
    """Returns `value` when it is an array, else an empty array."""
    return value if isinstance(value, list) else []


def str_value(value: object) -> str | None:
    """Returns a nonempty string, treating null and empty text as absent."""
    return value if isinstance(value, str) and value else None


def bool_value(value: object) -> bool | None:
    """Returns a real boolean; numbers are not booleans in these documents."""
    return value if isinstance(value, bool) else None


def int_value(value: object) -> int | None:
    """Returns an integral count, accepting an integral floating point value."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def number_value(value: object) -> float | None:
    """Returns a measurement as a float, if one is present."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def human_bytes(count: int | None, *, precision: int = 1) -> str:
    """Formats a byte count in binary units; an unknown size stays 'unknown'."""
    if count is None:
        return "unknown"
    value = float(count)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if value < 1024.0:
            if unit == "B":
                return f"{int(value)} B"
            return f"{value:.{precision}f} {unit}"
        value /= 1024.0
    return f"{value:.{precision}f} TiB"


def human_rate(value: float | None) -> str:
    """Formats a token rate; an unmeasured rate is a dash."""
    return "-" if value is None else f"{value:,.1f} tok/s"


def human_seconds(value: float | None) -> str:
    """Formats a duration at the precision a benchmark needs."""
    if value is None:
        return "-"
    if value < 1.0:
        return f"{value * 1000:.0f} ms"
    return f"{value:.2f} s"
