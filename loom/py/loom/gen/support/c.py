# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""C text emission helpers shared by Loom generators."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Sequence
from enum import Enum


class CIdentifierCase(Enum):
    """Case transformation policy for generated C identifiers."""

    PRESERVE = "preserve"
    LOWER = "lower"
    UPPER = "upper"


def c_identifier_parts(value: str) -> tuple[str, ...]:
    """Returns non-empty ASCII identifier parts split on non-identifier chars."""
    return tuple(part for part in re.split(r"[^0-9A-Za-z]+", value) if part)


def c_identifier(
    value: str,
    *,
    case: CIdentifierCase = CIdentifierCase.PRESERVE,
    empty: str = "_",
) -> str:
    """Returns a valid C identifier from an arbitrary stable spelling."""
    parts = c_identifier_parts(value)
    identifier = "_".join(parts) if parts else empty
    if not identifier:
        raise ValueError("empty replacement identifier must not be empty")
    if identifier[0].isdigit():
        identifier = "_" + identifier
    if case is CIdentifierCase.LOWER:
        return identifier.lower()
    if case is CIdentifierCase.UPPER:
        return identifier.upper()
    return identifier


def c_pascal_identifier(value: str) -> str:
    """Returns a PascalCase C identifier suffix from an arbitrary spelling."""
    return "".join(part[:1].upper() + part[1:] for part in c_identifier_parts(value))


def c_string_literal(value: str) -> str:
    """Escapes string content for use between C double quotes."""
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")


def c_string_arg(value: str) -> str:
    """Returns a quoted C string literal expression."""
    return f'"{c_string_literal(value)}"'


def c_string_view(value: str, *, macro: str = "IREE_SVL") -> str:
    """Returns a C string-view literal expression."""
    return f'{macro}("{c_string_literal(value)}")'


def c_string_classifier_lines(
    cases: Sequence[tuple[str, str]],
    *,
    input_name: str,
    unmatched_result: str,
    indent: str = "",
) -> list[str]:
    """Emits a length-partitioned exact string classifier returning C values."""
    if not cases:
        raise ValueError("C string classifier must contain at least one case")
    if not input_name or not unmatched_result:
        raise ValueError("C string classifier expressions must not be empty")

    cases_by_length: dict[int, list[tuple[str, str]]] = defaultdict(list)
    seen_spellings: set[str] = set()
    for spelling, result in cases:
        if spelling in seen_spellings:
            raise ValueError(f"duplicate C string classifier spelling: {spelling!r}")
        if not result:
            raise ValueError(f"empty C result expression for spelling {spelling!r}")
        seen_spellings.add(spelling)
        cases_by_length[len(spelling)].append((spelling, result))

    lines = [f"{indent}switch ({input_name}.size) {{"]
    for spelling_length, length_cases in sorted(cases_by_length.items()):
        lines.append(f"{indent}  case {spelling_length}:")
        for spelling, result in length_cases:
            lines.append(f"{indent}    if (iree_string_view_equal({input_name}, IREE_SV({c_string_arg(spelling)}))) {{")
            lines.append(f"{indent}      return {result};")
            lines.append(f"{indent}    }}")
        lines.append(f"{indent}    break;")
    lines.append(f"{indent}}}")
    lines.append(f"{indent}return {unmatched_result};")
    return lines


def c_i64_literal(value: int) -> str:
    """Returns a portable C expression for a signed 64-bit integer."""
    if value < -(1 << 63) or value > (1 << 63) - 1:
        raise ValueError(f"signed 64-bit integer literal out of range: {value}")
    if value == -(1 << 63):
        return "INT64_MIN"
    if value < 0:
        return f"(-INT64_C({abs(value)}))"
    return f"INT64_C({value})"
