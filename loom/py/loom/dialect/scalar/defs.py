# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Scalar dialect definitions: op group and flag enums.

Contains the scalar Dialect and its associated EnumDef types
(FastMathFlags, IntOverflowFlags) that are shared across all scalar
submodules.
"""

from loom.dsl import Dialect, EnumCase, EnumDef, OpCategory

__all__ = [
    "scalar_ops",
    "SCALAR_ANALYSIS_CATEGORY",
    "SCALAR_ARITHMETIC_CATEGORY",
    "SCALAR_BITWISE_CATEGORY",
    "SCALAR_COMPARISON_CATEGORY",
    "SCALAR_CONVERSION_CATEGORY",
    "SCALAR_MATH_CATEGORY",
    "SCALAR_OP_CATEGORIES",
    "ClampFMode",
    "FastMathFlags",
    "GeluVariant",
    "IntOverflowFlags",
]

SCALAR_ARITHMETIC_CATEGORY = OpCategory(
    "arithmetic",
    doc="Integer and floating-point scalar arithmetic operations.",
)
SCALAR_MATH_CATEGORY = OpCategory(
    "math",
    doc="Scalar transcendental, rounding, and special math operations.",
)
SCALAR_COMPARISON_CATEGORY = OpCategory(
    "comparison",
    doc="Scalar comparison, classification, and sign operations.",
)
SCALAR_CONVERSION_CATEGORY = OpCategory(
    "conversion",
    doc="Scalar type conversion, constants, and poison operations.",
)
SCALAR_BITWISE_CATEGORY = OpCategory(
    "bitwise",
    doc="Scalar bitwise logic, shifts, rotates, and bit counting operations.",
)
SCALAR_ANALYSIS_CATEGORY = OpCategory(
    "analysis",
    doc="Scalar fact and predicate refinement operations.",
)

SCALAR_OP_CATEGORIES = (
    SCALAR_ARITHMETIC_CATEGORY,
    SCALAR_MATH_CATEGORY,
    SCALAR_COMPARISON_CATEGORY,
    SCALAR_CONVERSION_CATEGORY,
    SCALAR_BITWISE_CATEGORY,
    SCALAR_ANALYSIS_CATEGORY,
)

scalar_ops = Dialect(
    "scalar",
    dialect_id=0x02,
    doc=("Scalar arithmetic, math, conversion, and typed poison ops. Loom poison is an invalid value sentinel introduced by canonicalization and diagnosed if it survives to an observation boundary."),
    categories=SCALAR_OP_CATEGORIES,
)

FastMathFlags = EnumDef(
    "FastMathFlags",
    [
        EnumCase("reassoc", 1, doc="Allow reassociation."),
        EnumCase("nnan", 2, doc="Assume no NaNs."),
        EnumCase("ninf", 4, doc="Assume no infinities."),
        EnumCase("nsz", 8, doc="Assume no signed zeros."),
        EnumCase("arcp", 16, doc="Allow reciprocal approximation."),
        EnumCase("contract", 32, doc="Allow contractions (FMA)."),
        EnumCase("afn", 64, doc="Approximate functions."),
        EnumCase("fast", 127, doc="All of the above."),
    ],
    doc="IEEE 754 fast-math relaxation flags for float operations.",
)

IntOverflowFlags = EnumDef(
    "IntOverflowFlags",
    [
        EnumCase("nsw", 1, doc="No signed wrap."),
        EnumCase("nuw", 2, doc="No unsigned wrap."),
    ],
    doc="Integer overflow behavior flags.",
)

ClampFMode = EnumDef(
    "ClampFMode",
    [
        EnumCase(
            "ordered",
            0,
            doc="Clamp with ordered comparisons and selects; NaN comparisons are false.",
        ),
        EnumCase(
            "number",
            1,
            doc="Clamp as minnum(maxnum(value, lower), upper); NaNs select the numeric peer.",
        ),
        EnumCase(
            "ieee",
            2,
            doc="Clamp as minimum(maximum(value, lower), upper); NaNs propagate.",
        ),
    ],
    doc="Floating-point clamp NaN and comparison policy.",
)

GeluVariant = EnumDef(
    "GeluVariant",
    [
        EnumCase("erf", 0, doc="Exact GELU formula using erf(x/sqrt(2))."),
        EnumCase("tanh", 1, doc="Tanh-polynomial GELU approximation."),
        EnumCase("logistic", 2, doc="Scaled logistic GELU approximation."),
    ],
    doc="GELU activation formula family.",
)
