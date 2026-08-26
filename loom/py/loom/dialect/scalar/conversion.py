# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Scalar conversion: type casts and constants."""

from loom.assembly import (
    COLON,
    Attr,
    ResultType,
    TypeOf,
)
from loom.dialect.scalar import scalar_ops
from loom.dsl import (
    CONSTANT_LIKE,
    DISTRIBUTION_TRANSFER,
    FLOAT,
    INTEGER,
    PAYLOAD_SCALAR,
    POISON,
    PURE,
    SCALAR,
    AttrDef,
    AttrMatchesElementType,
    ElementWidthGreaterThan,
    ElementWidthLessThan,
    Op,
    OperandRole,
    OpPhase,
    Result,
    TotalBitCountEqual,
    cast_op,
)

__all__ = ["ALL_CONVERSION_OPS"]

# ============================================================================
# Integer <-> float
# ============================================================================

scalar_sitofp = cast_op(
    "scalar.sitofp",
    group=scalar_ops,
    phase=OpPhase.EXECUTABLE,
    from_constraint=INTEGER,
    to_constraint=FLOAT,
    doc="Signed integer to floating-point.",
    facts="loom_scalar_sitofp_facts",
    examples=["%result = scalar.sitofp %input : i32 to f32"],
)
scalar_uitofp = cast_op(
    "scalar.uitofp",
    group=scalar_ops,
    phase=OpPhase.EXECUTABLE,
    from_constraint=INTEGER,
    to_constraint=FLOAT,
    doc="Unsigned integer to floating-point.",
    facts="loom_scalar_uitofp_facts",
    examples=["%result = scalar.uitofp %input : i32 to f32"],
)
scalar_fptosi = cast_op(
    "scalar.fptosi",
    group=scalar_ops,
    phase=OpPhase.EXECUTABLE,
    from_constraint=FLOAT,
    to_constraint=INTEGER,
    doc="Floating-point to signed integer (rounds toward zero).",
    facts="loom_scalar_fptosi_facts",
    examples=["%result = scalar.fptosi %input : f32 to i32"],
)
scalar_fptoui = cast_op(
    "scalar.fptoui",
    group=scalar_ops,
    phase=OpPhase.EXECUTABLE,
    from_constraint=FLOAT,
    to_constraint=INTEGER,
    doc="Floating-point to unsigned integer (rounds toward zero).",
    facts="loom_scalar_fptoui_facts",
    examples=["%result = scalar.fptoui %input : f32 to i32"],
)

# ============================================================================
# Float precision
# ============================================================================

scalar_extf = cast_op(
    "scalar.extf",
    group=scalar_ops,
    phase=OpPhase.EXECUTABLE,
    from_constraint=FLOAT,
    to_constraint=FLOAT,
    doc="Float precision extension to a strictly wider format: e.g. f16 to f32.",
    constraints=[ElementWidthGreaterThan("result", "input")],
    input_role=OperandRole.FLOAT_EXTENSION_SOURCE,
    canonicalize="loom_scalar_extf_canonicalize",
    facts="loom_scalar_extf_facts",
    examples=["%result = scalar.extf %input : f16 to f32"],
)
scalar_fptrunc = cast_op(
    "scalar.fptrunc",
    group=scalar_ops,
    phase=OpPhase.EXECUTABLE,
    from_constraint=FLOAT,
    to_constraint=FLOAT,
    doc=(
        "Float precision truncation using round-to-nearest, ties-to-even. "
        "The result format must be strictly narrower than the input format. "
        "Special values follow the destination format: f8E4M3 saturates "
        "finite overflow and infinities to its signed maximum finite value "
        "while preserving NaNs; IEEE formats preserve infinities and NaNs."
    ),
    constraints=[ElementWidthLessThan("result", "input")],
    canonicalize="loom_scalar_fptrunc_canonicalize",
    facts="loom_scalar_fptrunc_facts",
    examples=["%result = scalar.fptrunc %input : f32 to f16"],
)

# ============================================================================
# Integer width
# ============================================================================

scalar_extsi = cast_op(
    "scalar.extsi",
    group=scalar_ops,
    phase=OpPhase.EXECUTABLE,
    from_constraint=INTEGER,
    to_constraint=INTEGER,
    doc="Signed integer extension to a strictly wider type: e.g. i8 to i32.",
    constraints=[ElementWidthGreaterThan("result", "input")],
    canonicalize="loom_scalar_extsi_canonicalize",
    facts="loom_scalar_extsi_facts",
    traits=[DISTRIBUTION_TRANSFER],
    examples=["%result = scalar.extsi %input : i8 to i32"],
)
scalar_extui = cast_op(
    "scalar.extui",
    group=scalar_ops,
    phase=OpPhase.EXECUTABLE,
    from_constraint=INTEGER,
    to_constraint=INTEGER,
    doc="Unsigned integer extension to a strictly wider type: e.g. i8 to i32.",
    constraints=[ElementWidthGreaterThan("result", "input")],
    canonicalize="loom_scalar_extui_canonicalize",
    facts="loom_scalar_extui_facts",
    traits=[DISTRIBUTION_TRANSFER],
    examples=["%result = scalar.extui %input : i8 to i32"],
)
scalar_trunci = cast_op(
    "scalar.trunci",
    group=scalar_ops,
    phase=OpPhase.EXECUTABLE,
    from_constraint=INTEGER,
    to_constraint=INTEGER,
    doc="Integer truncation to a strictly narrower type: e.g. i32 to i8.",
    constraints=[ElementWidthLessThan("result", "input")],
    canonicalize="loom_scalar_trunci_canonicalize",
    facts="loom_scalar_trunci_facts",
    traits=[DISTRIBUTION_TRANSFER],
    examples=["%result = scalar.trunci %input : i32 to i8"],
)

# ============================================================================
# Special conversions
# ============================================================================

scalar_bitcast = cast_op(
    "scalar.bitcast",
    group=scalar_ops,
    phase=OpPhase.EXECUTABLE,
    from_constraint=PAYLOAD_SCALAR,
    to_constraint=PAYLOAD_SCALAR,
    doc=("Bitwise reinterpretation between scalar types with the same bit count. No numeric conversion is performed."),
    constraints=[TotalBitCountEqual("input", "result")],
    canonicalize="loom_scalar_bitcast_canonicalize",
    facts="loom_scalar_bitcast_facts",
    examples=["%result = scalar.bitcast %input : f32 to i32"],
)

# ============================================================================
# Constants
# ============================================================================

scalar_constant = Op(
    "scalar.constant",
    group=scalar_ops,
    phase=OpPhase.EXECUTABLE,
    doc=(
        "Materialize a compile-time integer or floating-point scalar value. "
        "Fixed-width integer literals use their signed value domain; unsigned "
        "operations interpret the resulting bit pattern. Logical coordinate "
        "and byte-offset constants use index.constant."
    ),
    results=[Result("result", PAYLOAD_SCALAR)],
    attrs=[AttrDef("value", "any", doc="The constant value.")],
    constraints=[AttrMatchesElementType("value", "result")],
    traits=[PURE, CONSTANT_LIKE],
    facts="loom_scalar_constant_facts",
    format=[Attr("value"), COLON, TypeOf("result")],
    examples=[
        "%c42 = scalar.constant 42 : i32",
        "%pi = scalar.constant 3.14159265358979 : f32",
        "%true = scalar.constant 1 : i1",
    ],
)

scalar_poison = Op(
    "scalar.poison",
    group=scalar_ops,
    doc=(
        "Materialize a typed Loom poison scalar. Poison represents an invalid "
        "scalar observation, such as extracting a lane proven not to exist. "
        "Pure scalar ops with any poison operand canonicalize to poison of the "
        "corresponding result type. Poison is not an LLVM poison value: it must "
        "be removed by dead-code elimination or diagnosed before it reaches a "
        "store, return, kernel boundary, or target-lowering boundary."
    ),
    results=[Result("result", SCALAR)],
    traits=[PURE, POISON],
    format=[COLON, ResultType("result")],
    examples=["%p = scalar.poison : f32"],
)

# ============================================================================
# Registry
# ============================================================================

ALL_CONVERSION_OPS: tuple[Op, ...] = (
    scalar_sitofp,
    scalar_uitofp,
    scalar_fptosi,
    scalar_fptoui,
    scalar_extf,
    scalar_fptrunc,
    scalar_extsi,
    scalar_extui,
    scalar_trunci,
    scalar_bitcast,
    scalar_constant,
    scalar_poison,
)
