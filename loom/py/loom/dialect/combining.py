# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Shared algebraic combining operation vocabulary."""

from loom.dsl import EnumCase, EnumDef

CombiningKind = EnumDef(
    "CombiningKind",
    [
        EnumCase("addi", 0, doc="Integer addition."),
        EnumCase("addf", 1, doc="Floating-point addition."),
        EnumCase("muli", 2, doc="Integer multiplication."),
        EnumCase("mulf", 3, doc="Floating-point multiplication."),
        EnumCase("minsi", 4, doc="Signed integer minimum."),
        EnumCase("maxsi", 5, doc="Signed integer maximum."),
        EnumCase("minui", 6, doc="Unsigned integer minimum."),
        EnumCase("maxui", 7, doc="Unsigned integer maximum."),
        EnumCase("andi", 8, doc="Bitwise AND."),
        EnumCase("ori", 9, doc="Bitwise OR."),
        EnumCase("xori", 10, doc="Bitwise XOR."),
        EnumCase("minimumf", 11, doc="IEEE 754 floating-point minimum."),
        EnumCase("maximumf", 12, doc="IEEE 754 floating-point maximum."),
        EnumCase("minnumf", 13, doc="C99 fmin-style floating-point minimum."),
        EnumCase("maxnumf", 14, doc="C99 fmax-style floating-point maximum."),
    ],
    doc="Algebraic combining operations for reductions and scans.",
    c_type="loom_combining_kind_t",
    c_const_prefix="LOOM_COMBINING_KIND",
    c_include="loom/ops/combining.h",
)
