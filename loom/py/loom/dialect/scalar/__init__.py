# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Scalar dialect: primitive operations on scalar values.

Replaces MLIR's arith + math dialects with a single unified namespace.
These ops live inside tile.elementwise bodies, scf.for bounds, and
index computations. Exhaustively defined — every scalar operation the
compiler needs is declared here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

# defs MUST be imported before submodules — submodules import
# scalar_ops/FastMathFlags/IntOverflowFlags from this package.
from loom.dialect.scalar.defs import (  # isort: split
    SCALAR_ANALYSIS_CATEGORY,
    SCALAR_ARITHMETIC_CATEGORY,
    SCALAR_BITWISE_CATEGORY,
    SCALAR_COMPARISON_CATEGORY,
    SCALAR_CONVERSION_CATEGORY,
    SCALAR_MATH_CATEGORY,
    SCALAR_OP_CATEGORIES,
    ClampFMode,
    FastMathFlags,
    GeluVariant,
    IntOverflowFlags,
    scalar_ops,
)

from loom.dialect.scalar.analysis import ALL_ANALYSIS_OPS
from loom.dialect.scalar.arithmetic import ALL_ARITHMETIC_OPS
from loom.dialect.scalar.bitwise import ALL_BITWISE_OPS
from loom.dialect.scalar.comparison import ALL_COMPARISON_OPS
from loom.dialect.scalar.conversion import ALL_CONVERSION_OPS
from loom.dialect.scalar.math import ALL_MATH_OPS

if TYPE_CHECKING:
    from loom.dsl import Op, OpCategory

__all__ = [
    "scalar_ops",
    "SCALAR_ANALYSIS_CATEGORY",
    "SCALAR_ARITHMETIC_CATEGORY",
    "SCALAR_BITWISE_CATEGORY",
    "SCALAR_COMPARISON_CATEGORY",
    "SCALAR_CONVERSION_CATEGORY",
    "SCALAR_MATH_CATEGORY",
    "SCALAR_OP_CATEGORIES",
    "SCALAR_OP_CATEGORY_GROUPS",
    "ClampFMode",
    "FastMathFlags",
    "GeluVariant",
    "IntOverflowFlags",
    "ALL_SCALAR_OPS",
]

SCALAR_OP_CATEGORY_GROUPS: tuple[tuple[OpCategory, tuple[Op, ...]], ...] = (
    (SCALAR_ARITHMETIC_CATEGORY, ALL_ARITHMETIC_OPS),
    (SCALAR_MATH_CATEGORY, ALL_MATH_OPS),
    (SCALAR_COMPARISON_CATEGORY, ALL_COMPARISON_OPS),
    (SCALAR_CONVERSION_CATEGORY, ALL_CONVERSION_OPS),
    (SCALAR_BITWISE_CATEGORY, ALL_BITWISE_OPS),
    (SCALAR_ANALYSIS_CATEGORY, ALL_ANALYSIS_OPS),
)

ALL_SCALAR_OPS: tuple[Op, ...] = tuple(op for _, category_ops in SCALAR_OP_CATEGORY_GROUPS for op in category_ops)
