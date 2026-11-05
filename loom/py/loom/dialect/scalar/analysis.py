# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Scalar structural ops: ownership, predicates, and compile-time contracts.

These operations preserve authored requirements and ownership boundaries on
SSA values. Executable structural operations define their lowering contract;
compile-time queries instead diagnose if specialization leaves them unresolved.
"""

from loom.assembly import COLON, PredicateList, Ref, Refs, TypeOf, TypesOf
from loom.dialect.scalar.defs import scalar_ops
from loom.dsl import (
    FACT_IDENTITY,
    PAYLOAD_SCALAR,
    PURE,
    AttrDef,
    Op,
    Operand,
    OpPhase,
    Result,
    SameType,
    VariadicValuesMatch,
    value_duplication_op,
)

__all__ = [
    "scalar_assume",
    "scalar_constexpr",
    "scalar_duplicate",
    "ALL_ANALYSIS_OPS",
]

# ============================================================================
# scalar.duplicate — distinct ownership identity
# ============================================================================

scalar_duplicate = value_duplication_op(
    "scalar.duplicate",
    group=scalar_ops,
    carrier=PAYLOAD_SCALAR,
    doc=(
        "Create another local owner of a scalar value. The result has the "
        "same logical value as its source and a distinct SSA "
        "ownership identity. Allocation may coalesce non-overlapping source "
        "and result storage; the operation does not promise disjoint storage."
    ),
    examples=[
        "%copy = scalar.duplicate %value : i32",
    ],
)

# ============================================================================
# scalar.assume — predicate-constrained identity
# ============================================================================

scalar_assume = Op(
    "scalar.assume",
    group=scalar_ops,
    doc="Identity with predicate constraints on scalar payload results. Use index.assume for index or offset values.",
    operands=[Operand("values", PAYLOAD_SCALAR, variadic=True)],
    results=[Result("results", PAYLOAD_SCALAR, variadic=True)],
    attrs=[AttrDef("predicates", "predicate_list")],
    constraints=[VariadicValuesMatch("values", "results")],
    traits=[PURE, FACT_IDENTITY],
    facts="loom_scalar_assume_facts",
    format=[
        Refs("values"),
        PredicateList("predicates"),
        COLON,
        TypesOf("results"),
    ],
    examples=[
        "%n2 = scalar.assume %n [multiple_of(%n, 16)] : i64",
        "%count2 = scalar.assume %count [range(%count, 0, %capacity)] : i32",
        "%n2, %k2 = scalar.assume %n, %k [multiple_of(%n, 16), lt(%k, 1024)] : i64, i64",
        "%x2 = scalar.assume %x [finite(%x)] : f32",
    ],
)

# ============================================================================
# scalar.constexpr — required compile-time scalar value
# ============================================================================

scalar_constexpr = Op(
    "scalar.constexpr",
    group=scalar_ops,
    phase=OpPhase.COMPILE_TIME_QUERY,
    doc=(
        "Require a scalar payload value to become exact during specialization. "
        "Configured canonicalization replaces the result with an ordinary "
        "constant once exact facts are available; target lowering diagnoses "
        "an unresolved requirement."
    ),
    operands=[Operand("input", PAYLOAD_SCALAR)],
    results=[Result("result", PAYLOAD_SCALAR)],
    constraints=[SameType("input", "result")],
    traits=[PURE],
    facts="loom_scalar_constexpr_facts",
    format=[Ref("input"), COLON, TypeOf("result")],
    examples=["%tile_size = scalar.constexpr %configured : i32"],
)

# ============================================================================
# Registry
# ============================================================================

ALL_ANALYSIS_OPS: tuple[Op, ...] = (
    scalar_assume,
    scalar_constexpr,
    scalar_duplicate,
)
