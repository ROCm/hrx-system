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
from functools import cache


class CIdentifierCase(Enum):
    """Case transformation policy for generated C identifiers."""

    PRESERVE = "preserve"
    LOWER = "lower"
    UPPER = "upper"


# C23 and C++20 keywords that can survive identifier normalization unchanged.
# Loom's generated public headers are C APIs that must also remain valid when
# included from C++, so identifiers avoid the union of both language sets.
_C_AND_CPP_KEYWORDS = frozenset(
    """
    alignas alignof and and_eq asm auto bitand bitor bool break case catch char
    char16_t char32_t char8_t class
    co_await co_return co_yield compl concept const const_cast consteval
    constexpr constinit continue decltype default delete do double
    dynamic_cast else enum explicit export extern false float for friend goto
    if inline int long mutable namespace new noexcept not not_eq nullptr
    operator or or_eq private protected public register reinterpret_cast
    requires restrict return short signed sizeof static static_assert
    static_cast struct switch template this thread_local throw true try typedef
    typeid typename typeof typeof_unqual union unsigned using virtual void
    volatile wchar_t while xor xor_eq
    """.split()  # noqa: SIM905 - compact language-spec word list
)


def c_identifier_parts(value: str) -> tuple[str, ...]:
    """Returns non-empty ASCII identifier parts split on non-identifier chars."""
    return tuple(part for part in re.split(r"[^0-9A-Za-z]+", value) if part)


@cache
def c_identifier(
    value: str,
    *,
    case: CIdentifierCase = CIdentifierCase.PRESERVE,
    empty: str = "_",
) -> str:
    """Returns a valid C/C++ identifier from an arbitrary stable spelling."""
    parts = c_identifier_parts(value)
    identifier = "_".join(parts) if parts else empty
    if not identifier:
        raise ValueError("empty replacement identifier must not be empty")
    if identifier[0].isdigit():
        identifier = "_" + identifier
    if case is CIdentifierCase.LOWER:
        identifier = identifier.lower()
    elif case is CIdentifierCase.UPPER:
        identifier = identifier.upper()
    if identifier in _C_AND_CPP_KEYWORDS:
        identifier += "_"
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
