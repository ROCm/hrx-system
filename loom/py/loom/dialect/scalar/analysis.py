# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Scalar analysis ops: constraints, assertions, and value properties.

These ops carry metadata about SSA values — they don't compute new
values, they assert or constrain properties of existing ones. The
optimizer and verifier use these constraints; codegen treats them as
identity (the output equals the input).
"""

from loom.assembly import COLON, PredicateList, Refs, TypesOf
from loom.dialect.scalar.defs import scalar_ops
from loom.dsl import FACT_IDENTITY, INTEGER, PURE, AttrDef, Op, Operand, Result

__all__ = [
    "scalar_assume",
    "ALL_ANALYSIS_OPS",
]

# ============================================================================
# scalar.assume — predicate-constrained identity
# ============================================================================

scalar_assume = Op(
    "scalar.assume",
    group=scalar_ops,
    doc="Identity with predicate constraints on integer payload results. Use index.assume for index or offset values.",
    operands=[Operand("values", INTEGER, variadic=True)],
    results=[Result("results", INTEGER, variadic=True)],
    attrs=[AttrDef("predicates", "predicate_list")],
    traits=[PURE, FACT_IDENTITY],
    facts="loom_scalar_assume_facts",
    verify="loom_scalar_assume_verify",
    format=[
        Refs("values"),
        PredicateList("predicates"),
        COLON,
        TypesOf("results"),
    ],
    examples=[
        "%n2 = scalar.assume %n [mul(%n, 16)] : i64",
        "%n2, %k2 = scalar.assume %n, %k [mul(%n, 16), lt(%k, 1024)] : i64, i64",
    ],
)

# ============================================================================
# Registry
# ============================================================================

ALL_ANALYSIS_OPS: tuple[Op, ...] = (scalar_assume,)
