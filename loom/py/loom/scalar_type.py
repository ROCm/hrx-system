# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Scalar type ordinals and assembly spellings shared by Loom tooling."""

from __future__ import annotations

from enum import IntEnum, unique

__all__ = [
    "SCALAR_TYPE_NONE",
    "SCALAR_TYPE_SPELLINGS",
    "ScalarTypeKind",
    "parse_scalar_type_kind",
    "scalar_type_name",
]


@unique
class ScalarTypeKind(IntEnum):
    """Scalar element type kind.

    Values match ``loom_scalar_type_e``. These are internal compiler values,
    not bytecode-stable wire ordinals.

    Ordered: address types, integers by width, floats by width.
    """

    INDEX = 1
    OFFSET = 2
    I1 = 3
    I8 = 4
    I16 = 5
    I32 = 6
    I64 = 7
    F8E4M3 = 8
    F8E5M2 = 9
    F16 = 10
    BF16 = 11
    F32 = 12
    F64 = 13


# Explicit absence of a scalar type. This matches ``LOOM_SCALAR_TYPE_NONE`` and
# ensures zero-initialized storage cannot be mistaken for a concrete type.
SCALAR_TYPE_NONE = 0


# Canonical assembly spellings ordered by concrete ScalarTypeKind ordinal. C
# name and classification tables are generated directly from this declaration.
SCALAR_TYPE_SPELLINGS: tuple[str, ...] = (
    "index",
    "offset",
    "i1",
    "i8",
    "i16",
    "i32",
    "i64",
    "f8E4M3",
    "f8E5M2",
    "f16",
    "bf16",
    "f32",
    "f64",
)

if len(SCALAR_TYPE_SPELLINGS) != len(ScalarTypeKind):
    raise ValueError("scalar type spellings must cover every scalar type kind")
if any(kind.value != ordinal for ordinal, kind in enumerate(ScalarTypeKind, 1)):
    raise ValueError("concrete scalar type kinds must use contiguous non-zero ordinals")
if len(set(SCALAR_TYPE_SPELLINGS)) != len(SCALAR_TYPE_SPELLINGS):
    raise ValueError("scalar type spellings must be unique")

_SCALAR_TYPE_BY_SPELLING = {
    spelling: kind
    for kind, spelling in zip(ScalarTypeKind, SCALAR_TYPE_SPELLINGS, strict=True)
}


def scalar_type_name(kind: ScalarTypeKind) -> str:
    """Returns the canonical assembly spelling for ``kind``."""
    return SCALAR_TYPE_SPELLINGS[kind.value - 1]


def parse_scalar_type_kind(name: str) -> ScalarTypeKind | None:
    """Returns the scalar kind matching ``name``, if declared."""
    return _SCALAR_TYPE_BY_SPELLING.get(name)
