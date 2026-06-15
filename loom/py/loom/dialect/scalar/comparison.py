# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Scalar comparison: cmpi, cmpf, float predicates, sign."""

from loom.assembly import (
    COLON,
    Ref,
    TypeOf,
)
from loom.dialect.scalar import FastMathFlags, scalar_ops
from loom.dsl import (
    FLOAT,
    I1,
    INTEGER,
    PURE,
    EnumCase,
    EnumDef,
    Op,
    Operand,
    OpPhase,
    Result,
    SameType,
    comparison_op,
)

__all__ = [
    "CmpIPredicate",
    "CmpFPredicate",
    "ALL_COMPARISON_OPS",
]

# ============================================================================
# Predicate enums
# ============================================================================

CmpIPredicate = EnumDef(
    "CmpIPredicate",
    [
        EnumCase("eq", 0, doc="Equal."),
        EnumCase("ne", 1, doc="Not equal."),
        EnumCase("slt", 2, doc="Signed less than."),
        EnumCase("sle", 3, doc="Signed less or equal."),
        EnumCase("sgt", 4, doc="Signed greater than."),
        EnumCase("sge", 5, doc="Signed greater or equal."),
        EnumCase("ult", 6, doc="Unsigned less than."),
        EnumCase("ule", 7, doc="Unsigned less or equal."),
        EnumCase("ugt", 8, doc="Unsigned greater than."),
        EnumCase("uge", 9, doc="Unsigned greater or equal."),
    ],
    doc="Integer comparison predicates.",
)

CmpFPredicate = EnumDef(
    "CmpFPredicate",
    [
        EnumCase("oeq", 0, doc="Ordered and equal."),
        EnumCase("ogt", 1, doc="Ordered and greater than."),
        EnumCase("oge", 2, doc="Ordered and greater or equal."),
        EnumCase("olt", 3, doc="Ordered and less than."),
        EnumCase("ole", 4, doc="Ordered and less or equal."),
        EnumCase("one", 5, doc="Ordered and not equal."),
        EnumCase("ord", 6, doc="Ordered (no NaN)."),
        EnumCase("ueq", 7, doc="Unordered or equal."),
        EnumCase("ugt", 8, doc="Unordered or greater than."),
        EnumCase("uge", 9, doc="Unordered or greater or equal."),
        EnumCase("ult", 10, doc="Unordered or less than."),
        EnumCase("ule", 11, doc="Unordered or less or equal."),
        EnumCase("une", 12, doc="Unordered or not equal."),
        EnumCase("uno", 13, doc="Unordered (either is NaN)."),
    ],
    doc="Floating-point comparison predicates (IEEE 754 total order).",
)

# ============================================================================
# Comparison ops
# ============================================================================

scalar_cmpi = comparison_op(
    "scalar.cmpi",
    group=scalar_ops,
    phase=OpPhase.EXECUTABLE,
    type_constraint=INTEGER,
    predicates=CmpIPredicate,
    doc="Integer comparison.",
    canonicalize="loom_scalar_cmpi_canonicalize",
    facts="loom_scalar_cmpi_facts",
    examples=["%result = scalar.cmpi eq, %lhs, %rhs : i32"],
)

scalar_cmpf = comparison_op(
    "scalar.cmpf",
    group=scalar_ops,
    phase=OpPhase.EXECUTABLE,
    type_constraint=FLOAT,
    predicates=CmpFPredicate,
    doc="Floating-point comparison.",
    flags=("fastmath", FastMathFlags),
    canonicalize="loom_scalar_cmpf_canonicalize",
    facts="loom_scalar_cmpf_facts",
    examples=["%result = scalar.cmpf olt, %lhs, %rhs : f32"],
)

scalar_isnanf = Op(
    "scalar.isnanf",
    group=scalar_ops,
    doc="Returns true (i1) if the operand is NaN.",
    operands=[Operand("input", FLOAT)],
    results=[Result("result", I1)],
    traits=[PURE],
    format=[Ref("input"), COLON, TypeOf("input")],
    facts="loom_scalar_isnanf_facts",
    examples=["%result = scalar.isnanf %input : f32"],
)

scalar_isinff = Op(
    "scalar.isinff",
    group=scalar_ops,
    doc="Returns true (i1) if the operand is positive or negative infinity.",
    operands=[Operand("input", FLOAT)],
    results=[Result("result", I1)],
    traits=[PURE],
    format=[Ref("input"), COLON, TypeOf("input")],
    facts="loom_scalar_isinff_facts",
    examples=["%result = scalar.isinff %input : f32"],
)

scalar_isfinitef = Op(
    "scalar.isfinitef",
    group=scalar_ops,
    doc="Returns true (i1) if the operand is finite (not NaN and not infinity).",
    operands=[Operand("input", FLOAT)],
    results=[Result("result", I1)],
    traits=[PURE],
    format=[Ref("input"), COLON, TypeOf("input")],
    facts="loom_scalar_isfinitef_facts",
    examples=["%result = scalar.isfinitef %input : f32"],
)

scalar_signf = Op(
    "scalar.signf",
    group=scalar_ops,
    doc="Floating-point sign: returns -1.0, 0.0, or 1.0.",
    operands=[Operand("input", FLOAT)],
    results=[Result("result", FLOAT)],
    constraints=[SameType("input", "result")],
    traits=[PURE],
    format=[Ref("input"), COLON, TypeOf("result")],
    facts="loom_scalar_signf_facts",
    examples=["%result = scalar.signf %input : f32"],
)

scalar_signi = Op(
    "scalar.signi",
    group=scalar_ops,
    doc="Integer sign: returns -1, 0, or 1.",
    operands=[Operand("input", INTEGER)],
    results=[Result("result", INTEGER)],
    constraints=[SameType("input", "result")],
    traits=[PURE],
    format=[Ref("input"), COLON, TypeOf("result")],
    facts="loom_scalar_signi_facts",
    examples=["%result = scalar.signi %input : i32"],
)

# ============================================================================
# Registry
# ============================================================================

ALL_COMPARISON_OPS: tuple[Op, ...] = (
    scalar_cmpi,
    scalar_cmpf,
    scalar_isnanf,
    scalar_isinff,
    scalar_isfinitef,
    scalar_signf,
    scalar_signi,
)
