# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Scalar bitwise: bitwise ops, shifts, rotates, bit counting."""

from loom.assembly import ARROW, COLON, AttrDict, Ref, ResultType, TypeOf
from loom.dialect.scalar import IntOverflowFlags, scalar_ops
from loom.dsl import (
    ATTR_TYPE_I64,
    DISTRIBUTION_TRANSFER,
    INTEGER,
    PURE,
    AttrDef,
    BitRangeWithinElementWidth,
    ElementWidthAtLeastAttr,
    Op,
    Operand,
    OpPhase,
    Result,
    SameKind,
    binary_op,
    unary_op,
)

__all__ = ["ALL_BITWISE_OPS"]

# ============================================================================
# Bitwise logic
# ============================================================================

scalar_andi = binary_op(
    "scalar.andi",
    group=scalar_ops,
    phase=OpPhase.EXECUTABLE,
    type_constraint=INTEGER,
    doc="Bitwise AND.",
    commutative=True,
    facts="loom_scalar_andi_facts",
    traits=[DISTRIBUTION_TRANSFER],
    canonicalize="loom_scalar_andi_canonicalize",
    examples=["%result = scalar.andi %lhs, %rhs : i32"],
)
scalar_ori = binary_op(
    "scalar.ori",
    group=scalar_ops,
    phase=OpPhase.EXECUTABLE,
    type_constraint=INTEGER,
    doc="Bitwise OR.",
    commutative=True,
    facts="loom_scalar_ori_facts",
    traits=[DISTRIBUTION_TRANSFER],
    canonicalize="loom_scalar_ori_canonicalize",
    examples=["%result = scalar.ori %lhs, %rhs : i32"],
)
scalar_xori = binary_op(
    "scalar.xori",
    group=scalar_ops,
    phase=OpPhase.EXECUTABLE,
    type_constraint=INTEGER,
    doc="Bitwise XOR.",
    commutative=True,
    facts="loom_scalar_xori_facts",
    traits=[DISTRIBUTION_TRANSFER],
    canonicalize="loom_scalar_xori_canonicalize",
    examples=["%result = scalar.xori %lhs, %rhs : i32"],
)

# ============================================================================
# Shifts (with optional overflow flags on left shift)
# ============================================================================

scalar_shli = binary_op(
    "scalar.shli",
    group=scalar_ops,
    phase=OpPhase.EXECUTABLE,
    type_constraint=INTEGER,
    doc="Left shift.",
    flags=("overflow", IntOverflowFlags),
    facts="loom_scalar_shli_facts",
    traits=[DISTRIBUTION_TRANSFER],
    canonicalize="loom_scalar_shli_canonicalize",
    examples=["%result = scalar.shli %lhs, %rhs : i32"],
)
scalar_shrsi = binary_op(
    "scalar.shrsi",
    group=scalar_ops,
    phase=OpPhase.EXECUTABLE,
    type_constraint=INTEGER,
    doc="Arithmetic right shift (sign-extending).",
    facts="loom_scalar_shrsi_facts",
    traits=[DISTRIBUTION_TRANSFER],
    canonicalize="loom_scalar_shrsi_canonicalize",
    examples=["%result = scalar.shrsi %lhs, %rhs : i32"],
)
scalar_shrui = binary_op(
    "scalar.shrui",
    group=scalar_ops,
    phase=OpPhase.EXECUTABLE,
    type_constraint=INTEGER,
    doc="Logical right shift (zero-extending).",
    facts="loom_scalar_shrui_facts",
    traits=[DISTRIBUTION_TRANSFER],
    canonicalize="loom_scalar_shrui_canonicalize",
    examples=["%result = scalar.shrui %lhs, %rhs : i32"],
)

# ============================================================================
# Rotates
# ============================================================================

scalar_rotli = binary_op(
    "scalar.rotli",
    group=scalar_ops,
    type_constraint=INTEGER,
    doc="Left rotate.",
    facts="loom_scalar_rotli_facts",
    canonicalize="loom_scalar_rotli_canonicalize",
    examples=["%result = scalar.rotli %lhs, %rhs : i32"],
)
scalar_rotri = binary_op(
    "scalar.rotri",
    group=scalar_ops,
    type_constraint=INTEGER,
    doc="Right rotate.",
    facts="loom_scalar_rotri_facts",
    canonicalize="loom_scalar_rotri_canonicalize",
    examples=["%result = scalar.rotri %lhs, %rhs : i32"],
)

# ============================================================================
# Bit counting
# ============================================================================

scalar_ctlzi = unary_op(
    "scalar.ctlzi",
    group=scalar_ops,
    type_constraint=INTEGER,
    doc="Count leading zeros.",
    facts="loom_scalar_ctlzi_facts",
    examples=["%result = scalar.ctlzi %input : i32"],
)
scalar_cttzi = unary_op(
    "scalar.cttzi",
    group=scalar_ops,
    type_constraint=INTEGER,
    doc="Count trailing zeros.",
    facts="loom_scalar_cttzi_facts",
    examples=["%result = scalar.cttzi %input : i32"],
)
scalar_ctpopi = unary_op(
    "scalar.ctpopi",
    group=scalar_ops,
    type_constraint=INTEGER,
    doc="Population count (number of set bits).",
    facts="loom_scalar_ctpopi_facts",
    examples=["%result = scalar.ctpopi %input : i32"],
)

# ============================================================================
# Bitfield extraction
# ============================================================================


def _bitfield_attrs() -> list[AttrDef]:
    return [
        AttrDef(
            "offset",
            ATTR_TYPE_I64,
            doc="Least-significant bit position of the field within the source.",
        ),
        AttrDef(
            "width",
            ATTR_TYPE_I64,
            doc="Number of bits in the extracted field.",
        ),
    ]


scalar_bitfield_extractu = Op(
    "scalar.bitfield.extractu",
    group=scalar_ops,
    phase=OpPhase.EXECUTABLE,
    doc="Extract one fixed integer bitfield and zero-extend it into the result.",
    operands=[Operand("source", INTEGER)],
    results=[Result("result", INTEGER)],
    attrs=_bitfield_attrs(),
    constraints=[
        SameKind("source", "result"),
        BitRangeWithinElementWidth("source", "offset", "width"),
        ElementWidthAtLeastAttr("result", "width"),
    ],
    facts="loom_scalar_bitfield_extractu_facts",
    traits=[PURE, DISTRIBUTION_TRANSFER],
    format=[
        Ref("source"),
        AttrDict(),
        COLON,
        TypeOf("source"),
        ARROW,
        ResultType("result"),
    ],
    examples=[
        "%byte = scalar.bitfield.extractu %word {offset = 8, width = 8} : i32 -> i32",
    ],
)

scalar_bitfield_extracts = Op(
    "scalar.bitfield.extracts",
    group=scalar_ops,
    phase=OpPhase.EXECUTABLE,
    doc="Extract one fixed integer bitfield and sign-extend it into the result.",
    operands=[Operand("source", INTEGER)],
    results=[Result("result", INTEGER)],
    attrs=_bitfield_attrs(),
    constraints=[
        SameKind("source", "result"),
        BitRangeWithinElementWidth("source", "offset", "width"),
        ElementWidthAtLeastAttr("result", "width"),
    ],
    facts="loom_scalar_bitfield_extracts_facts",
    traits=[PURE, DISTRIBUTION_TRANSFER],
    format=[
        Ref("source"),
        AttrDict(),
        COLON,
        TypeOf("source"),
        ARROW,
        ResultType("result"),
    ],
    examples=[
        "%signed_byte = scalar.bitfield.extracts %word {offset = 24, width = 8} : i32 -> i32",
    ],
)

# ============================================================================
# Registry
# ============================================================================

ALL_BITWISE_OPS: tuple[Op, ...] = (
    scalar_andi,
    scalar_ori,
    scalar_xori,
    scalar_shli,
    scalar_shrsi,
    scalar_shrui,
    scalar_rotli,
    scalar_rotri,
    scalar_ctlzi,
    scalar_cttzi,
    scalar_ctpopi,
    scalar_bitfield_extractu,
    scalar_bitfield_extracts,
)
