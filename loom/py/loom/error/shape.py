# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""SHAPE domain — rank mismatches and shape inconsistencies."""

from loom.errors import ErrorDef, ErrorDomain, ErrorParam, ParamKind, Severity

# ERR_SHAPE_001: RanksMatch constraint violated.
ERR_SHAPE_001 = ErrorDef(
    domain=ErrorDomain.SHAPE,
    code=1,
    severity=Severity.ERROR,
    summary="Operand ranks do not match.",
    message="'{operand_a}' rank ({rank_a}) does not match "
    "'{operand_b}' rank ({rank_b})",
    params=(
        ErrorParam("operand_a", ParamKind.STRING),
        ErrorParam("rank_a", ParamKind.I64),
        ErrorParam("operand_b", ParamKind.STRING),
        ErrorParam("rank_b", ParamKind.I64),
    ),
    fix_hint="'{operand_a}' and '{operand_b}' must have the same rank",
)

# ERR_SHAPE_002: SameShape constraint violated.
ERR_SHAPE_002 = ErrorDef(
    domain=ErrorDomain.SHAPE,
    code=2,
    severity=Severity.ERROR,
    summary="SameShape constraint violated.",
    message="'{field_a}' shape does not match '{field_b}' shape",
    params=(
        ErrorParam("field_a", ParamKind.STRING),
        ErrorParam("field_b", ParamKind.STRING),
    ),
    fix_hint="Ensure '{field_a}' and '{field_b}' have identical shapes",
)

# ERR_SHAPE_003: AllShapesMatch constraint violated.
ERR_SHAPE_003 = ErrorDef(
    domain=ErrorDomain.SHAPE,
    code=3,
    severity=Severity.ERROR,
    summary="Input tiles must have identical shapes.",
    message="input tile shapes do not match: input 0 has type "
    "{first_type}, input {other_index} has type {other_type}",
    params=(
        ErrorParam("first_type", ParamKind.TYPE),
        ErrorParam("other_index", ParamKind.U32),
        ErrorParam("other_type", ParamKind.TYPE),
    ),
    fix_hint="Ensure all input tiles have identical shapes, or use "
    "tile.broadcast first",
)

# ERR_SHAPE_004: Broadcast shape incompatibility.
ERR_SHAPE_004 = ErrorDef(
    domain=ErrorDomain.SHAPE,
    code=4,
    severity=Severity.ERROR,
    summary="Incompatible dimension sizes during broadcast.",
    message="incompatible dimension at source axis {source_axis} "
    "(size {source_size}) with result axis {result_axis} "
    "(size {result_size})",
    params=(
        ErrorParam("source_axis", ParamKind.U32),
        ErrorParam("source_size", ParamKind.I64),
        ErrorParam("result_axis", ParamKind.U32),
        ErrorParam("result_size", ParamKind.I64),
    ),
    fix_hint="Source axis {source_axis} (size {source_size}) must be 1 "
    "or match result axis {result_axis} (size {result_size})",
)

# ERR_SHAPE_005: Result shape does not match input shapes.
ERR_SHAPE_005 = ErrorDef(
    domain=ErrorDomain.SHAPE,
    code=5,
    severity=Severity.ERROR,
    summary="Result tile shape must match input tile shapes.",
    message="result tile type {result_type} does not match input "
    "tile type {input_type}",
    params=(
        ErrorParam("result_type", ParamKind.TYPE),
        ErrorParam("input_type", ParamKind.TYPE),
    ),
    fix_hint="Ensure result type has the same shape as the input tiles",
)

# ERR_SHAPE_006: Vector transform last-axis extent is dynamic.
ERR_SHAPE_006 = ErrorDef(
    domain=ErrorDomain.SHAPE,
    code=6,
    severity=Severity.ERROR,
    summary="Vector transform last-axis extent is dynamic.",
    message=("{pass_name} requires {op_name} last-axis transform extent to be static"),
    params=(
        ErrorParam("op_name", ParamKind.STRING),
        ErrorParam("pass_name", ParamKind.STRING),
    ),
    fix_hint=(
        "Refine the transform last-axis extent before vector scalarization or "
        "lower the transform through a target primitive"
    ),
)

# ERR_SHAPE_007: Static vector element count is not representable.
ERR_SHAPE_007 = ErrorDef(
    domain=ErrorDomain.SHAPE,
    code=7,
    severity=Severity.ERROR,
    summary="Static vector element count is not representable.",
    message=(
        "{op_name} cannot be scalarized by {pass_name} because static vector "
        "type {vector_type} has more lanes than scalarization can represent"
    ),
    params=(
        ErrorParam("op_name", ParamKind.STRING),
        ErrorParam("pass_name", ParamKind.STRING),
        ErrorParam("vector_type", ParamKind.TYPE),
    ),
    fix_hint=(
        "Refine the vector shape before scalarization or lower it with a "
        "target primitive that preserves the vector aggregate"
    ),
)

ALL_SHAPE_ERRORS: tuple[ErrorDef, ...] = (
    ERR_SHAPE_001,
    ERR_SHAPE_002,
    ERR_SHAPE_003,
    ERR_SHAPE_004,
    ERR_SHAPE_005,
    ERR_SHAPE_006,
    ERR_SHAPE_007,
)
