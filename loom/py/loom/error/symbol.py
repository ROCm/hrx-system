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

# ERR_SYMBOL_006: A semantic relation between symbols is invalid.
ERR_SYMBOL_006 = ErrorDef(
    domain=ErrorDomain.SYMBOL,
    code=6,
    severity=Severity.ERROR,
    summary="Invalid symbol relation.",
    message=(
        "symbol relation '{relation}' from '@{source_symbol}' to "
        "'@{target_symbol}' is invalid: {reason}"
    ),
    params=(
        ErrorParam("relation", ParamKind.STRING),
        ErrorParam("source_symbol", ParamKind.STRING),
        ErrorParam("target_symbol", ParamKind.STRING),
        ErrorParam("reason", ParamKind.STRING),
    ),
)

# ERR_SYMBOL_007: Final compilation requires exact configuration definitions.
ERR_SYMBOL_007 = ErrorDef(
    domain=ErrorDomain.SYMBOL,
    code=7,
    severity=Severity.ERROR,
    summary="Configuration declaration remains unresolved.",
    message=(
        "config '@{symbol_name}' requires an exact definition for final "
        "compilation (unresolved count: {unresolved_count})"
    ),
    params=(
        ErrorParam("symbol_name", ParamKind.STRING),
        ErrorParam("unresolved_count", ParamKind.U64),
    ),
    fix_hint="Supply an exact config value before final compilation",
)

# ERR_SYMBOL_008: An exact symbol value must satisfy its declared predicates.
ERR_SYMBOL_008 = ErrorDef(
    domain=ErrorDomain.SYMBOL,
    code=8,
    severity=Severity.ERROR,
    summary="Exact value violates its symbol contract.",
    message=(
        "symbol '@{symbol_name}' value {value} of type {type} "
        "violates constraint '{predicate}'"
    ),
    params=(
        ErrorParam("symbol_name", ParamKind.STRING),
        ErrorParam("value", ParamKind.STRING),
        ErrorParam("type", ParamKind.TYPE),
        ErrorParam("predicate", ParamKind.STRING),
    ),
)

ERR_SYMBOL_009 = ErrorDef(
    domain=ErrorDomain.SYMBOL,
    code=9,
    severity=Severity.ERROR,
    summary="Configuration provider contains a program operation.",
    message="configuration providers must contain only config.def operations",
    params=(),
)

ERR_SYMBOL_010 = ErrorDef(
    domain=ErrorDomain.SYMBOL,
    code=10,
    severity=Severity.ERROR,
    summary="Configuration definition has an incompatible type.",
    message=(
        "config '@{symbol_name}' requires {expected_type}, "
        "but the supplied definition has type {actual_type}"
    ),
    params=(
        ErrorParam("symbol_name", ParamKind.STRING),
        ErrorParam("expected_type", ParamKind.TYPE),
        ErrorParam("actual_type", ParamKind.TYPE),
    ),
)

ERR_SYMBOL_011 = ErrorDef(
    domain=ErrorDomain.SYMBOL,
    code=11,
    severity=Severity.ERROR,
    summary="Configuration value cannot be represented by its declared type.",
    message="config '@{symbol_name}' value '{value}' cannot be represented as {type}",
    params=(
        ErrorParam("symbol_name", ParamKind.STRING),
        ErrorParam("value", ParamKind.STRING),
        ErrorParam("type", ParamKind.TYPE),
    ),
)

ERR_SYMBOL_012 = ErrorDef(
    domain=ErrorDomain.SYMBOL,
    code=12,
    severity=Severity.ERROR,
    summary="Configuration encoding has an incompatible role.",
    message="config '@{symbol_name}' expects {expected_role}, got {actual_role}",
    params=(
        ErrorParam("symbol_name", ParamKind.STRING),
        ErrorParam("expected_role", ParamKind.STRING),
        ErrorParam("actual_role", ParamKind.STRING),
    ),
)

ERR_SYMBOL_013 = ErrorDef(
    domain=ErrorDomain.SYMBOL,
    code=13,
    severity=Severity.ERROR,
    summary="Linked symbol contracts are incompatible.",
    message="symbol '@{symbol_name}' has incompatible contract field '{field}'",
    params=(
        ErrorParam("symbol_name", ParamKind.STRING),
        ErrorParam("field", ParamKind.STRING),
    ),
)

ERR_SYMBOL_014 = ErrorDef(
    domain=ErrorDomain.SYMBOL,
    code=14,
    severity=Severity.ERROR,
    summary="Linked function signatures have different lengths.",
    message=(
        "declaration for '@{symbol_name}' has {source_count} {field}, "
        "but the selected symbol has {selected_count}"
    ),
    params=(
        ErrorParam("symbol_name", ParamKind.STRING),
        ErrorParam("field", ParamKind.STRING),
        ErrorParam("source_count", ParamKind.U64),
        ErrorParam("selected_count", ParamKind.U64),
    ),
)

ERR_SYMBOL_015 = ErrorDef(
    domain=ErrorDomain.SYMBOL,
    code=15,
    severity=Severity.ERROR,
    summary="Linked function signatures have incompatible types.",
    message=(
        "declaration for '@{symbol_name}' has incompatible {field} type "
        "at index {ordinal}"
    ),
    params=(
        ErrorParam("symbol_name", ParamKind.STRING),
        ErrorParam("field", ParamKind.STRING),
        ErrorParam("ordinal", ParamKind.U64),
    ),
)

ERR_SYMBOL_016 = ErrorDef(
    domain=ErrorDomain.SYMBOL,
    code=16,
    severity=Severity.ERROR,
    summary="Linked value contract exceeds the predicate limit.",
    message=(
        "symbol '@{symbol_name}' has {count} merged predicates, "
        "exceeding the limit of {limit}"
    ),
    params=(
        ErrorParam("symbol_name", ParamKind.STRING),
        ErrorParam("count", ParamKind.U64),
        ErrorParam("limit", ParamKind.U64),
    ),
)

ERR_SYMBOL_017 = ErrorDef(
    domain=ErrorDomain.SYMBOL,
    code=17,
    severity=Severity.ERROR,
    summary="Requested root has no definition.",
    message=(
        "root symbol '@{symbol_name}' has no materialized definition or declaration"
    ),
    params=(ErrorParam("symbol_name", ParamKind.STRING),),
)

ERR_SYMBOL_018 = ErrorDef(
    domain=ErrorDomain.SYMBOL,
    code=18,
    severity=Severity.ERROR,
    summary="Requested root has an empty name.",
    message="root symbol name must not be empty",
    params=(),
)

ERR_SYMBOL_019 = ErrorDef(
    domain=ErrorDomain.SYMBOL,
    code=19,
    severity=Severity.ERROR,
    summary="Linked source snapshots disagree.",
    message="linked inputs contain different source contents for '{filename}'",
    params=(ErrorParam("filename", ParamKind.STRING),),
)

ALL_SYMBOL_ERRORS: tuple[ErrorDef, ...] = (
    ERR_SYMBOL_001,
    ERR_SYMBOL_002,
    ERR_SYMBOL_003,
    ERR_SYMBOL_004,
    ERR_SYMBOL_005,
    ERR_SYMBOL_006,
    ERR_SYMBOL_007,
    ERR_SYMBOL_008,
    ERR_SYMBOL_009,
    ERR_SYMBOL_010,
    ERR_SYMBOL_011,
    ERR_SYMBOL_012,
    ERR_SYMBOL_013,
    ERR_SYMBOL_014,
    ERR_SYMBOL_015,
    ERR_SYMBOL_016,
    ERR_SYMBOL_017,
    ERR_SYMBOL_018,
    ERR_SYMBOL_019,
)
