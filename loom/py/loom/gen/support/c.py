# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""C text emission helpers shared by Loom generators."""

from __future__ import annotations

import re
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
