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

    INDEX = 0
    OFFSET = 1
    I1 = 2
    I8 = 3
    I16 = 4
    I32 = 5
    I64 = 6
    F8E4M3 = 7
    F8E5M2 = 8
    F16 = 9
    BF16 = 10
    F32 = 11
    F64 = 12


# Explicit absence of a scalar type. This matches ``LOOM_SCALAR_TYPE_NONE``
# and remains outside all concrete scalar type tables.
SCALAR_TYPE_NONE = len(ScalarTypeKind)


# Canonical assembly spellings indexed by ScalarTypeKind ordinal. C name and
# classification tables are generated directly from this declaration.
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
if any(kind.value != ordinal for ordinal, kind in enumerate(ScalarTypeKind)):
    raise ValueError("scalar type kinds must use contiguous declaration ordinals")
if len(set(SCALAR_TYPE_SPELLINGS)) != len(SCALAR_TYPE_SPELLINGS):
    raise ValueError("scalar type spellings must be unique")

_SCALAR_TYPE_BY_SPELLING = {
    spelling: ScalarTypeKind(ordinal)
    for ordinal, spelling in enumerate(SCALAR_TYPE_SPELLINGS)
}


def scalar_type_name(kind: ScalarTypeKind) -> str:
    """Returns the canonical assembly spelling for ``kind``."""
    return SCALAR_TYPE_SPELLINGS[kind]


def parse_scalar_type_kind(name: str) -> ScalarTypeKind | None:
    """Returns the scalar kind matching ``name``, if declared."""
    return _SCALAR_TYPE_BY_SPELLING.get(name)
