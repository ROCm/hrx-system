# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""C array emission helpers shared by Loom generators."""

from __future__ import annotations

from collections.abc import Sequence


class StaticArrayEmitter:
    """Emits exact C array initializers once within one generated source."""

    def __init__(self, lines: list[str]):
        self._lines = lines
        self._symbols_by_initializer: dict[tuple[str, str, tuple[object, ...]], str] = {}

    def append_value_array(
        self,
        c_type: str,
        symbol_name: str,
        values: Sequence[str],
        *,
        emit_empty: bool = False,
        trailing_blank: bool = True,
    ) -> str:
        """Emits or reuses a scalar array and returns its backing symbol."""
        if not values and not emit_empty:
            return symbol_name
        initializer = ("value", c_type, tuple(values))
        existing_symbol = self._symbols_by_initializer.get(initializer)
        if existing_symbol is not None:
            return existing_symbol
        if values:
            append_value_array(
                self._lines,
                c_type,
                symbol_name,
                values,
                trailing_blank=trailing_blank,
            )
        else:
            self._lines.extend(
                [
                    f"static const {c_type} {symbol_name}[] = {{",
                    "};",
                ]
            )
            if trailing_blank:
                self._lines.append("")
        self._symbols_by_initializer[initializer] = symbol_name
        return symbol_name

    def append_struct_array(
        self,
        c_type: str,
        symbol_name: str,
        rows: Sequence[Sequence[str]],
        *,
        emit_empty: bool = False,
        trailing_blank: bool = True,
    ) -> str:
        """Emits or reuses a struct array and returns its backing symbol."""
        if not rows and not emit_empty:
            return symbol_name
        initializer = (
            "struct",
            c_type,
            tuple(tuple(row) for row in rows),
        )
        existing_symbol = self._symbols_by_initializer.get(initializer)
        if existing_symbol is not None:
            return existing_symbol
        if rows:
            append_struct_array(
                self._lines,
                c_type,
                symbol_name,
                rows,
                trailing_blank=trailing_blank,
            )
        else:
            self._lines.extend(
                [
                    f"static const {c_type} {symbol_name}[] = {{",
                    "};",
                ]
            )
            if trailing_blank:
                self._lines.append("")
        self._symbols_by_initializer[initializer] = symbol_name
        return symbol_name


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
