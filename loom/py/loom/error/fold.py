# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""FOLD domain — poison folding and canonicalization."""

from loom.errors import ErrorDef, ErrorDomain, ErrorParam, ParamKind, Severity

# ERR_FOLD_001: Operation folded to poison (generic).
ERR_FOLD_001 = ErrorDef(
    domain=ErrorDomain.FOLD,
    code=1,
    severity=Severity.REMARK,
    summary="Operation folded to poison.",
    message="'{op_name}' folded to poison: {poison_code}",
    params=(
        ErrorParam("op_name", ParamKind.STRING),
        ErrorParam("poison_code", ParamKind.STRING),
    ),
)

# ERR_FOLD_002: Slice source is poison.
ERR_FOLD_002 = ErrorDef(
    domain=ErrorDomain.FOLD,
    code=2,
    severity=Severity.REMARK,
    summary="Slice source operand is poison.",
    message="'{op_name}' folded to poison: source operand is poison",
    params=(
        ErrorParam("op_name", ParamKind.STRING),
        ErrorParam("source_type", ParamKind.TYPE),
    ),
)

# ERR_FOLD_003: Update target is poison.
ERR_FOLD_003 = ErrorDef(
    domain=ErrorDomain.FOLD,
    code=3,
    severity=Severity.REMARK,
    summary="Update target operand is poison.",
    message="'{op_name}' folded to poison: target operand is poison",
    params=(
        ErrorParam("op_name", ParamKind.STRING),
        ErrorParam("target_type", ParamKind.TYPE),
    ),
)

# ERR_FOLD_004: Subrange access out of bounds.
ERR_FOLD_004 = ErrorDef(
    domain=ErrorDomain.FOLD,
    code=4,
    severity=Severity.REMARK,
    summary="Subrange access extends beyond bounds.",
    message="'{op_name}' folded to poison: dimension {dim_index} "
    "access at offset {offset} + size {size} = {total} exceeds "
    "bound {bound}",
    params=(
        ErrorParam("op_name", ParamKind.STRING),
        ErrorParam("dim_index", ParamKind.I64),
        ErrorParam("offset", ParamKind.I64),
        ErrorParam("size", ParamKind.I64),
        ErrorParam("total", ParamKind.I64),
        ErrorParam("bound", ParamKind.I64),
    ),
)

# ERR_FOLD_005: Negative offset.
ERR_FOLD_005 = ErrorDef(
    domain=ErrorDomain.FOLD,
    code=5,
    severity=Severity.REMARK,
    summary="Subrange access has negative offset.",
    message="'{op_name}' folded to poison: dimension {dim_index} has "
    "negative offset {offset}",
    params=(
        ErrorParam("op_name", ParamKind.STRING),
        ErrorParam("dim_index", ParamKind.I64),
        ErrorParam("offset", ParamKind.I64),
    ),
)

ALL_FOLD_ERRORS: tuple[ErrorDef, ...] = (
    ERR_FOLD_001,
    ERR_FOLD_002,
    ERR_FOLD_003,
    ERR_FOLD_004,
    ERR_FOLD_005,
)
