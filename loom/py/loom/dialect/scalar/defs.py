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
    "FloatConversionFlags",
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
        EnumCase("nnan", 2, doc="Assume floating-point operands and results are not NaN, independently for every vector element."),
        EnumCase("ninf", 4, doc="Assume floating-point operands and results are not infinite, independently for every vector element."),
        EnumCase("nsz", 8, doc="Permit treating positive and negative zero as interchangeable; this does not exclude zero results."),
        EnumCase("arcp", 16, doc="Allow reciprocal approximation."),
        EnumCase("contract", 32, doc="Allow contractions (FMA)."),
        EnumCase("afn", 64, doc="Approximate functions."),
        EnumCase("fast", 127, doc="All of the above."),
    ],
    doc=(
        "IEEE 754 fast-math relaxation flags for float operations. nnan and "
        "ninf are independent author promises, not runtime checks or requests "
        "to clamp results. Together they guarantee finite floating-point "
        "results, but do not imply exact representability in a narrower type "
        "or absence of subnormals. Behavior outside an authored promise is "
        "not constrained by the operation's ordinary special-value contract. "
        "Flags do not refine unrelated uses of an operand."
    ),
)

TrigonometryMathFlags = EnumDef(
    "TrigonometryMathFlags",
    [
        *FastMathFlags.cases,
        EnumCase(
            "preserve",
            128,
            doc=("Require source-defined signs of zero results, independently of AFN approximation permission. This positive requirement overrides NSZ and is not included in fast."),
        ),
    ],
    doc=(
        "Trigonometric math permissions and requirements. AFN permits native approximation with target-dependent zero signs unless preserve is requested. Other fast-math permissions retain their meanings; preserve does not require correctly rounded nonzero approximations."
    ),
)

FloatConversionFlags = EnumDef(
    "FloatConversionFlags",
    [
        EnumCase(
            "daz",
            1,
            doc=("Denormals are zero: permit a subnormal input to be replaced by the corresponding signed zero before conversion."),
        ),
        EnumCase(
            "ftz",
            2,
            doc=("Flush to zero: permit a subnormal rounded result to be replaced by the corresponding signed zero."),
        ),
    ],
    doc="Independent DAZ and FTZ permissions for floating-point conversions.",
)

IntOverflowFlags = EnumDef(
    "IntOverflowFlags",
    [
        EnumCase("nsw", 1, doc="No signed wrap."),
        EnumCase("nuw", 2, doc="No unsigned wrap."),
    ],
    doc="Integer overflow behavior flags.",
    c_type="loom_integer_overflow_flags_t",
    c_const_prefix="LOOM_INTEGER_OVERFLOW_FLAG",
    c_include="loom/ops/op_defs.h",
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
