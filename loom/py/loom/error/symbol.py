# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""SYMBOL domain — unresolved references."""

from loom.errors import ErrorDef, ErrorDomain, ErrorParam, ParamKind, Severity

# ERR_SYMBOL_001: Symbol reference out of range.
ERR_SYMBOL_001 = ErrorDef(
    domain=ErrorDomain.SYMBOL,
    code=1,
    severity=Severity.ERROR,
    summary="Symbol reference out of range.",
    message="symbol reference index {symbol_index} is out of range (max {max_index})",
    params=(
        ErrorParam("symbol_index", ParamKind.U32),
        ErrorParam("max_index", ParamKind.U32),
    ),
)

# ERR_SYMBOL_002: Symbol reference is unresolved.
ERR_SYMBOL_002 = ErrorDef(
    domain=ErrorDomain.SYMBOL,
    code=2,
    severity=Severity.ERROR,
    summary="Symbol reference is unresolved.",
    message="symbol '@{symbol_name}' is referenced but not defined",
    params=(ErrorParam("symbol_name", ParamKind.STRING),),
)

# ERR_SYMBOL_003: Symbol kind mismatch.
ERR_SYMBOL_003 = ErrorDef(
    domain=ErrorDomain.SYMBOL,
    code=3,
    severity=Severity.ERROR,
    summary="Symbol kind mismatch.",
    message="symbol '@{symbol_name}' has kind {actual_kind}, expected {expected_kind}",
    params=(
        ErrorParam("symbol_name", ParamKind.STRING),
        ErrorParam("actual_kind", ParamKind.STRING),
        ErrorParam("expected_kind", ParamKind.STRING),
    ),
)

# ERR_SYMBOL_004: Symbol references must be module-local in in-memory IR.
ERR_SYMBOL_004 = ErrorDef(
    domain=ErrorDomain.SYMBOL,
    code=4,
    severity=Severity.ERROR,
    summary="Symbol reference targets a non-local module.",
    message=(
        "symbol references in in-memory IR must use module_id 0, got "
        "module_id {module_id}"
    ),
    params=(ErrorParam("module_id", ParamKind.U32),),
)

# ERR_SYMBOL_005: Symbol is defined more than once.
ERR_SYMBOL_005 = ErrorDef(
    domain=ErrorDomain.SYMBOL,
    code=5,
    severity=Severity.ERROR,
    summary="Duplicate symbol definition.",
    message="symbol '@{symbol_name}' is already defined",
    params=(ErrorParam("symbol_name", ParamKind.STRING),),
)

ALL_SYMBOL_ERRORS: tuple[ErrorDef, ...] = (
    ERR_SYMBOL_001,
    ERR_SYMBOL_002,
    ERR_SYMBOL_003,
    ERR_SYMBOL_004,
    ERR_SYMBOL_005,
)
