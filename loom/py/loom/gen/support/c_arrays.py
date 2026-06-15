# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""C array emission helpers shared by Loom generators."""

from __future__ import annotations

from collections.abc import Sequence


def append_value_array(
    lines: list[str],
    c_type: str,
    symbol_name: str,
    values: Sequence[str],
    *,
    trailing_blank: bool = True,
) -> None:
    """Appends a non-empty static const C array with scalar initializer rows."""
    if not values:
        return
    lines.append(f"static const {c_type} {symbol_name}[] = {{")
    lines.extend(f"    {value}," for value in values)
    lines.append("};")
    if trailing_blank:
        lines.append("")


def append_struct_array(
    lines: list[str],
    c_type: str,
    symbol_name: str,
    rows: Sequence[Sequence[str]],
    *,
    trailing_blank: bool = True,
) -> None:
    """Appends a non-empty static const C array with multi-line struct rows."""
    if not rows:
        return
    lines.append(f"static const {c_type} {symbol_name}[] = {{")
    for row in rows:
        lines.append("    {")
        lines.extend(f"        {line}" for line in row)
        lines.append("    },")
    lines.append("};")
    if trailing_blank:
        lines.append("")
