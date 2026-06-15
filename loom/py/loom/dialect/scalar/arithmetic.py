# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Scalar arithmetic: integer and float arithmetic, min/max, div variants."""

# These are imported from __init__.py at declaration time, but we need
# forward references for the flag enums. Import from the parent.
from loom.assembly import (
    COLON,
    COMMA,
    Flags,
    Ref,
    TemplateParamFlags,
    TypeOf,
)
from loom.dialect.scalar import (
    ClampFMode,
    FastMathFlags,
    IntOverflowFlags,
    scalar_ops,
)
from loom.dsl import (
    ATTR_TYPE_ENUM,
    ATTR_TYPE_FLAGS,
    DISTRIBUTION_TRANSFER,
    FLOAT,
    IDEMPOTENT,
    INTEGER,
    INVOLUTION,
    PURE,
    AttrDef,
    Op,
    Operand,
    OpPhase,
    Result,
    SameType,
    binary_op,
    unary_op,
)

__all__ = ["ALL_ARITHMETIC_OPS"]

# ============================================================================
# Integer arithmetic (with optional IntOverflowFlags)
# ============================================================================

scalar_addi = binary_op(
    "scalar.addi",
    group=scalar_ops,
    phase=OpPhase.EXECUTABLE,
    type_constraint=INTEGER,
    doc="Integer addition.",
    commutative=True,
    flags=("overflow", IntOverflowFlags),
    facts="loom_scalar_addi_facts",
    traits=[DISTRIBUTION_TRANSFER],
    canonicalize="loom_scalar_addi_canonicalize",
    examples=[
        "%result = scalar.addi %lhs, %rhs : i32",
        "%result = scalar.addi<nuw> %lhs, %rhs : i32",
    ],
)

scalar_subi = binary_op(
    "scalar.subi",
    group=scalar_ops,
    phase=OpPhase.EXECUTABLE,
    type_constraint=INTEGER,
    doc="Integer subtraction.",
    flags=("overflow", IntOverflowFlags),
    facts="loom_scalar_subi_facts",
    traits=[DISTRIBUTION_TRANSFER],
    canonicalize="loom_scalar_subi_canonicalize",
    examples=["%result = scalar.subi %lhs, %rhs : i32"],
)

scalar_muli = binary_op(
    "scalar.muli",
    group=scalar_ops,
    phase=OpPhase.EXECUTABLE,
    type_constraint=INTEGER,
    doc="Integer multiplication.",
    commutative=True,
    flags=("overflow", IntOverflowFlags),
    facts="loom_scalar_muli_facts",
    traits=[DISTRIBUTION_TRANSFER],
    canonicalize="loom_scalar_muli_canonicalize",
    examples=["%result = scalar.muli %lhs, %rhs : i32"],
)

scalar_divsi = binary_op(
    "scalar.divsi",
    group=scalar_ops,
    phase=OpPhase.EXECUTABLE,
    type_constraint=INTEGER,
    doc="Signed integer division (rounds toward zero).",
    facts="loom_scalar_divsi_facts",
    traits=[DISTRIBUTION_TRANSFER],
    canonicalize="loom_scalar_divsi_canonicalize",
    examples=["%result = scalar.divsi %lhs, %rhs : i32"],
)

scalar_divui = binary_op(
    "scalar.divui",
    group=scalar_ops,
    phase=OpPhase.EXECUTABLE,
    type_constraint=INTEGER,
    doc="Unsigned integer division.",
    facts="loom_scalar_divui_facts",
    traits=[DISTRIBUTION_TRANSFER],
    canonicalize="loom_scalar_divui_canonicalize",
    examples=["%result = scalar.divui %lhs, %rhs : i32"],
)

scalar_remsi = binary_op(
    "scalar.remsi",
    group=scalar_ops,
    phase=OpPhase.EXECUTABLE,
    type_constraint=INTEGER,
    doc="Signed integer remainder.",
    facts="loom_scalar_remsi_facts",
    traits=[DISTRIBUTION_TRANSFER],
    canonicalize="loom_scalar_remsi_canonicalize",
    examples=["%result = scalar.remsi %lhs, %rhs : i32"],
)

scalar_remui = binary_op(
    "scalar.remui",
    group=scalar_ops,
    phase=OpPhase.EXECUTABLE,
    type_constraint=INTEGER,
    doc="Unsigned integer remainder.",
    facts="loom_scalar_remui_facts",
    traits=[DISTRIBUTION_TRANSFER],
    canonicalize="loom_scalar_remui_canonicalize",
    examples=["%result = scalar.remui %lhs, %rhs : i32"],
)

scalar_ceildivsi = binary_op(
    "scalar.ceildivsi",
    group=scalar_ops,
    type_constraint=INTEGER,
    doc="Signed integer division, rounding toward positive infinity.",
    canonicalize="loom_scalar_ceildivsi_canonicalize",
    examples=["%result = scalar.ceildivsi %lhs, %rhs : i32"],
)

scalar_ceildivui = binary_op(
    "scalar.ceildivui",
    group=scalar_ops,
    type_constraint=INTEGER,
    doc="Unsigned integer division, rounding toward positive infinity.",
    canonicalize="loom_scalar_ceildivui_canonicalize",
    examples=["%result = scalar.ceildivui %lhs, %rhs : i32"],
)

scalar_floordivsi = binary_op(
    "scalar.floordivsi",
    group=scalar_ops,
    type_constraint=INTEGER,
    doc="Signed integer division, rounding toward negative infinity.",
    canonicalize="loom_scalar_floordivsi_canonicalize",
    examples=["%result = scalar.floordivsi %lhs, %rhs : i32"],
)

scalar_negi = unary_op(
    "scalar.negi",
    group=scalar_ops,
    type_constraint=INTEGER,
    doc="Integer negation.",
    traits=[INVOLUTION, DISTRIBUTION_TRANSFER],
    facts="loom_scalar_negi_facts",
    canonicalize="loom_scalar_negi_canonicalize",
    examples=["%result = scalar.negi %input : i32"],
)

scalar_absi = unary_op(
    "scalar.absi",
    group=scalar_ops,
    type_constraint=INTEGER,
    doc="Integer absolute value.",
    traits=[IDEMPOTENT, DISTRIBUTION_TRANSFER],
    facts="loom_scalar_absi_facts",
    canonicalize="loom_scalar_absi_canonicalize",
    examples=["%result = scalar.absi %input : i32"],
)

scalar_minsi = binary_op(
    "scalar.minsi",
    group=scalar_ops,
    type_constraint=INTEGER,
    doc="Signed integer minimum.",
    commutative=True,
    facts="loom_scalar_minsi_facts",
    traits=[DISTRIBUTION_TRANSFER],
    canonicalize="loom_scalar_minsi_canonicalize",
    examples=["%result = scalar.minsi %lhs, %rhs : i32"],
)

scalar_maxsi = binary_op(
    "scalar.maxsi",
    group=scalar_ops,
    type_constraint=INTEGER,
    doc="Signed integer maximum.",
    commutative=True,
    facts="loom_scalar_maxsi_facts",
    traits=[DISTRIBUTION_TRANSFER],
    canonicalize="loom_scalar_maxsi_canonicalize",
    examples=["%result = scalar.maxsi %lhs, %rhs : i32"],
)

scalar_minui = binary_op(
    "scalar.minui",
    group=scalar_ops,
    type_constraint=INTEGER,
    doc="Unsigned integer minimum.",
    commutative=True,
    facts="loom_scalar_minui_facts",
    traits=[DISTRIBUTION_TRANSFER],
    canonicalize="loom_scalar_minui_canonicalize",
    examples=["%result = scalar.minui %lhs, %rhs : i32"],
)

scalar_maxui = binary_op(
    "scalar.maxui",
    group=scalar_ops,
    type_constraint=INTEGER,
    doc="Unsigned integer maximum.",
    commutative=True,
    facts="loom_scalar_maxui_facts",
    traits=[DISTRIBUTION_TRANSFER],
    canonicalize="loom_scalar_maxui_canonicalize",
    examples=["%result = scalar.maxui %lhs, %rhs : i32"],
)

# ============================================================================
# Integer fused multiply-add
# ============================================================================

scalar_fmai = Op(
    "scalar.fmai",
    group=scalar_ops,
    phase=OpPhase.EXECUTABLE,
    doc="Fused integer multiply-add: a*b + c with no intermediate overflow check.",
    operands=[
        Operand("a", INTEGER),
        Operand("b", INTEGER),
        Operand("c", INTEGER),
    ],
    results=[Result("result", INTEGER)],
    attrs=[AttrDef("overflow", ATTR_TYPE_FLAGS, optional=True, enum_def=IntOverflowFlags)],
    constraints=[SameType("a", "b", "c", "result")],
    traits=[PURE, DISTRIBUTION_TRANSFER],
    format=[
        Flags("overflow"),
        Ref("a"),
        COMMA,
        Ref("b"),
        COMMA,
        Ref("c"),
        COLON,
        TypeOf("result"),
    ],
    facts="loom_scalar_fmai_facts",
    canonicalize="loom_scalar_fmai_canonicalize",
    examples=["%result = scalar.fmai %a, %b, %c : i64"],
)

# ============================================================================
# Float arithmetic (with optional FastMathFlags)
# ============================================================================

scalar_addf = binary_op(
    "scalar.addf",
    group=scalar_ops,
    phase=OpPhase.EXECUTABLE,
    type_constraint=FLOAT,
    doc="Floating-point addition.",
    commutative=True,
    flags=("fastmath", FastMathFlags),
    facts="loom_scalar_addf_facts",
    canonicalize="loom_scalar_addf_canonicalize",
    examples=[
        "%result = scalar.addf %lhs, %rhs : f32",
        "%result = scalar.addf<fast> %lhs, %rhs : f32",
    ],
)

scalar_subf = binary_op(
    "scalar.subf",
    group=scalar_ops,
    phase=OpPhase.EXECUTABLE,
    type_constraint=FLOAT,
    doc="Floating-point subtraction.",
    flags=("fastmath", FastMathFlags),
    facts="loom_scalar_subf_facts",
    canonicalize="loom_scalar_subf_canonicalize",
    examples=["%result = scalar.subf %lhs, %rhs : f32"],
)

scalar_mulf = binary_op(
    "scalar.mulf",
    group=scalar_ops,
    phase=OpPhase.EXECUTABLE,
    type_constraint=FLOAT,
    doc="Floating-point multiplication.",
    commutative=True,
    flags=("fastmath", FastMathFlags),
    facts="loom_scalar_mulf_facts",
    canonicalize="loom_scalar_mulf_canonicalize",
    examples=["%result = scalar.mulf %lhs, %rhs : f32"],
)

scalar_divf = binary_op(
    "scalar.divf",
    group=scalar_ops,
    phase=OpPhase.EXECUTABLE,
    type_constraint=FLOAT,
    doc="Floating-point division.",
    flags=("fastmath", FastMathFlags),
    facts="loom_scalar_divf_facts",
    canonicalize="loom_scalar_divf_canonicalize",
    examples=["%result = scalar.divf %lhs, %rhs : f32"],
)

scalar_remf = binary_op(
    "scalar.remf",
    group=scalar_ops,
    phase=OpPhase.EXECUTABLE,
    type_constraint=FLOAT,
    doc="Floating-point remainder (C fmod semantics).",
    flags=("fastmath", FastMathFlags),
    facts="loom_scalar_remf_facts",
    examples=["%result = scalar.remf %lhs, %rhs : f32"],
)

scalar_negf = unary_op(
    "scalar.negf",
    group=scalar_ops,
    phase=OpPhase.EXECUTABLE,
    type_constraint=FLOAT,
    doc="Floating-point negation.",
    traits=[INVOLUTION],
    flags=("fastmath", FastMathFlags),
    facts="loom_scalar_negf_facts",
    canonicalize="loom_scalar_negf_canonicalize",
    examples=["%result = scalar.negf %input : f32"],
)

scalar_absf = unary_op(
    "scalar.absf",
    group=scalar_ops,
    type_constraint=FLOAT,
    doc="Floating-point absolute value.",
    traits=[IDEMPOTENT],
    flags=("fastmath", FastMathFlags),
    facts="loom_scalar_absf_facts",
    canonicalize="loom_scalar_absf_canonicalize",
    examples=["%result = scalar.absf %input : f32"],
)

scalar_minimumf = binary_op(
    "scalar.minimumf",
    group=scalar_ops,
    type_constraint=FLOAT,
    doc="IEEE 754 minimum (NaN propagates).",
    commutative=True,
    flags=("fastmath", FastMathFlags),
    facts="loom_scalar_minimumf_facts",
    examples=["%result = scalar.minimumf %lhs, %rhs : f32"],
)

scalar_maximumf = binary_op(
    "scalar.maximumf",
    group=scalar_ops,
    type_constraint=FLOAT,
    doc="IEEE 754 maximum (NaN propagates).",
    commutative=True,
    flags=("fastmath", FastMathFlags),
    facts="loom_scalar_maximumf_facts",
    examples=["%result = scalar.maximumf %lhs, %rhs : f32"],
)

scalar_minnumf = binary_op(
    "scalar.minnumf",
    group=scalar_ops,
    phase=OpPhase.EXECUTABLE,
    type_constraint=FLOAT,
    doc="C99 fmin (NaN ignored, returns the non-NaN operand).",
    commutative=True,
    flags=("fastmath", FastMathFlags),
    facts="loom_scalar_minnumf_facts",
    examples=["%result = scalar.minnumf %lhs, %rhs : f32"],
)

scalar_maxnumf = binary_op(
    "scalar.maxnumf",
    group=scalar_ops,
    phase=OpPhase.EXECUTABLE,
    type_constraint=FLOAT,
    doc="C99 fmax (NaN ignored, returns the non-NaN operand).",
    commutative=True,
    flags=("fastmath", FastMathFlags),
    facts="loom_scalar_maxnumf_facts",
    examples=["%result = scalar.maxnumf %lhs, %rhs : f32"],
)

scalar_clampf = Op(
    "scalar.clampf",
    group=scalar_ops,
    phase=OpPhase.EXECUTABLE,
    doc=(
        "Floating-point clamp with explicit NaN/comparison policy. The ordered mode preserves strict compare/select semantics, number mode uses minnum/maxnum semantics, and ieee mode propagates NaNs."
    ),
    operands=[
        Operand("value", FLOAT),
        Operand("lower", FLOAT),
        Operand("upper", FLOAT),
    ],
    results=[Result("result", FLOAT)],
    attrs=[
        AttrDef("mode", ATTR_TYPE_ENUM, enum_def=ClampFMode),
        AttrDef("fastmath", ATTR_TYPE_FLAGS, optional=True, enum_def=FastMathFlags),
    ],
    constraints=[SameType("value", "lower", "upper", "result")],
    traits=[PURE],
    facts="loom_scalar_clampf_facts",
    canonicalize="loom_scalar_clampf_canonicalize",
    format=[
        TemplateParamFlags("mode", "fastmath"),
        Ref("value"),
        COMMA,
        Ref("lower"),
        COMMA,
        Ref("upper"),
        COLON,
        TypeOf("result"),
    ],
    examples=[
        "%result = scalar.clampf<ordered> %value, %lower, %upper : f32",
        "%result = scalar.clampf<number, nnan|nsz> %value, %lower, %upper : f32",
        "%result = scalar.clampf<ieee> %value, %lower, %upper : f32",
    ],
)

scalar_copysignf = binary_op(
    "scalar.copysignf",
    group=scalar_ops,
    type_constraint=FLOAT,
    doc="Copy sign of rhs onto magnitude of lhs.",
    facts="loom_scalar_copysignf_facts",
    canonicalize="loom_scalar_copysignf_canonicalize",
    examples=["%result = scalar.copysignf %lhs, %rhs : f32"],
)

# ============================================================================
# Registry
# ============================================================================

ALL_ARITHMETIC_OPS: tuple[Op, ...] = (
    scalar_addi,
    scalar_subi,
    scalar_muli,
    scalar_divsi,
    scalar_divui,
    scalar_remsi,
    scalar_remui,
    scalar_ceildivsi,
    scalar_ceildivui,
    scalar_floordivsi,
    scalar_negi,
    scalar_absi,
    scalar_minsi,
    scalar_maxsi,
    scalar_minui,
    scalar_maxui,
    scalar_fmai,
    scalar_addf,
    scalar_subf,
    scalar_mulf,
    scalar_divf,
    scalar_remf,
    scalar_negf,
    scalar_absf,
    scalar_minimumf,
    scalar_maximumf,
    scalar_minnumf,
    scalar_maxnumf,
    scalar_clampf,
    scalar_copysignf,
)
