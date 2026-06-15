# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""SUBRANGE domain — offset/size counts and bounds violations."""

from loom.errors import ErrorDef, ErrorDomain, ErrorParam, ParamKind, Severity

# ERR_SUBRANGE_001: OffsetCountMatchesRank violated.
ERR_SUBRANGE_001 = ErrorDef(
    domain=ErrorDomain.SUBRANGE,
    code=1,
    severity=Severity.ERROR,
    summary="Offset count must match operand rank.",
    message="'{operand_name}' offset count ({offset_count}) does not "
    "match rank ({rank})",
    params=(
        ErrorParam("operand_name", ParamKind.STRING),
        ErrorParam("offset_count", ParamKind.U32),
        ErrorParam("rank", ParamKind.I64),
    ),
    fix_hint="Provide exactly {rank} offsets for '{operand_name}'",
)

# ERR_SUBRANGE_002: DimIndexInBounds violated.
ERR_SUBRANGE_002 = ErrorDef(
    domain=ErrorDomain.SUBRANGE,
    code=2,
    severity=Severity.ERROR,
    summary="Dimension index out of bounds.",
    message="dimension index {dim_index} is out of bounds for type with rank {rank}",
    params=(
        ErrorParam("dim_index", ParamKind.I64),
        ErrorParam("rank", ParamKind.I64),
    ),
    fix_hint="Use an index in range [0, {rank})",
)

# ERR_SUBRANGE_003: Size count does not match rank.
ERR_SUBRANGE_003 = ErrorDef(
    domain=ErrorDomain.SUBRANGE,
    code=3,
    severity=Severity.ERROR,
    summary="Size count must match operand rank.",
    message="'{operand_name}' size count ({size_count}) does not match rank ({rank})",
    params=(
        ErrorParam("operand_name", ParamKind.STRING),
        ErrorParam("size_count", ParamKind.U32),
        ErrorParam("rank", ParamKind.I64),
    ),
    fix_hint="Provide exactly {rank} sizes for '{operand_name}'",
)

# ERR_SUBRANGE_004: Static slice out of bounds.
ERR_SUBRANGE_004 = ErrorDef(
    domain=ErrorDomain.SUBRANGE,
    code=4,
    severity=Severity.ERROR,
    summary="Static slice extends beyond bounds.",
    message="dimension {dim_index}: offset ({offset}) + size ({size}) "
    "= {total} exceeds bound {bound}",
    params=(
        ErrorParam("dim_index", ParamKind.I64),
        ErrorParam("offset", ParamKind.I64),
        ErrorParam("size", ParamKind.I64),
        ErrorParam("total", ParamKind.I64),
        ErrorParam("bound", ParamKind.I64),
    ),
    fix_hint="Ensure offset + size <= {bound} for dimension {dim_index}",
)

# ERR_SUBRANGE_007: Vector memory view layout is unresolved.
ERR_SUBRANGE_007 = ErrorDef(
    domain=ErrorDomain.SUBRANGE,
    code=7,
    severity=Severity.ERROR,
    summary="Vector memory view layout is unresolved.",
    message="{op_name} footprint proof requires a resolved view layout",
    params=(ErrorParam("op_name", ParamKind.STRING),),
    fix_hint="Use a concrete view layout before vector footprint verification",
)

# ERR_SUBRANGE_008: Vector memory footprint origin is unresolved.
ERR_SUBRANGE_008 = ErrorDef(
    domain=ErrorDomain.SUBRANGE,
    code=8,
    severity=Severity.ERROR,
    summary="Vector memory footprint origin is unresolved.",
    message=(
        "{op_name} footprint origin {origin} is unresolved on view axis "
        "{view_axis} (vector axis {vector_axis})"
    ),
    params=(
        ErrorParam("op_name", ParamKind.STRING),
        ErrorParam("view_axis", ParamKind.I64),
        ErrorParam("vector_axis", ParamKind.I64),
        ErrorParam("origin", ParamKind.STRING),
    ),
    fix_hint="Use a static index or an index SSA value for the footprint origin",
)

# ERR_SUBRANGE_009: Vector memory origin lower bound is not proven.
ERR_SUBRANGE_009 = ErrorDef(
    domain=ErrorDomain.SUBRANGE,
    code=9,
    severity=Severity.ERROR,
    summary="Vector memory origin lower bound is not proven.",
    message=(
        "{op_name} footprint origin lower bound is not proven on view axis "
        "{view_axis} (vector axis {vector_axis}); origin {origin} must satisfy "
        "0 <= origin"
    ),
    params=(
        ErrorParam("op_name", ParamKind.STRING),
        ErrorParam("view_axis", ParamKind.I64),
        ErrorParam("vector_axis", ParamKind.I64),
        ErrorParam("origin", ParamKind.STRING),
    ),
    fix_hint="Prove that origin {origin} is non-negative at the memory op",
)

# ERR_SUBRANGE_010: Vector memory full-vector upper bound is not proven.
ERR_SUBRANGE_010 = ErrorDef(
    domain=ErrorDomain.SUBRANGE,
    code=10,
    severity=Severity.ERROR,
    summary="Vector memory full-vector upper bound is not proven.",
    message=(
        "{op_name} full-vector footprint upper bound is not proven on view axis "
        "{view_axis} (vector axis {vector_axis}); origin {origin} must satisfy "
        "origin + vector_extent <= view_bound"
    ),
    params=(
        ErrorParam("op_name", ParamKind.STRING),
        ErrorParam("view_axis", ParamKind.I64),
        ErrorParam("vector_axis", ParamKind.I64),
        ErrorParam("origin", ParamKind.STRING),
    ),
    fix_hint=(
        "Use a full-tile guard, vector.mask.range tail mask, padded view "
        "extent, or launch/view relation visible at the memory op"
    ),
)

# ERR_SUBRANGE_011: Vector memory scalar-axis upper bound is not proven.
ERR_SUBRANGE_011 = ErrorDef(
    domain=ErrorDomain.SUBRANGE,
    code=11,
    severity=Severity.ERROR,
    summary="Vector memory scalar-axis upper bound is not proven.",
    message=(
        "{op_name} scalar-axis footprint upper bound is not proven on view axis "
        "{view_axis} (vector axis {vector_axis}); origin {origin} must satisfy "
        "origin + 1 <= view_bound"
    ),
    params=(
        ErrorParam("op_name", ParamKind.STRING),
        ErrorParam("view_axis", ParamKind.I64),
        ErrorParam("vector_axis", ParamKind.I64),
        ErrorParam("origin", ParamKind.STRING),
    ),
    fix_hint=(
        "Use a guard, assume, launch/view relation, or padded view extent "
        "visible at the memory op"
    ),
)

# ERR_SUBRANGE_012: Vector memory linearized origin is unresolved.
ERR_SUBRANGE_012 = ErrorDef(
    domain=ErrorDomain.SUBRANGE,
    code=12,
    severity=Severity.ERROR,
    summary="Vector memory linearized origin is unresolved.",
    message="{op_name} linear footprint origin is unresolved",
    params=(ErrorParam("op_name", ParamKind.STRING),),
    fix_hint="Use dense layout facts and origin indices that can be linearized",
)

# ERR_SUBRANGE_013: Vector memory offset lane range is unresolved.
ERR_SUBRANGE_013 = ErrorDef(
    domain=ErrorDomain.SUBRANGE,
    code=13,
    severity=Severity.ERROR,
    summary="Vector memory offset lane range is unresolved.",
    message="{op_name} linear footprint offset lane range is unresolved",
    params=(ErrorParam("op_name", ParamKind.STRING),),
    fix_hint="Provide offset lane facts with known minimum and maximum values",
)

# ERR_SUBRANGE_014: Vector memory linear footprint lower bound is not proven.
ERR_SUBRANGE_014 = ErrorDef(
    domain=ErrorDomain.SUBRANGE,
    code=14,
    severity=Severity.ERROR,
    summary="Vector memory linear footprint lower bound is not proven.",
    message=(
        "{op_name} linear footprint lower bound is not proven; "
        "0 <= linearized_origin + min(offsets) must hold"
    ),
    params=(ErrorParam("op_name", ParamKind.STRING),),
    fix_hint="Prove the linearized origin plus the minimum offset lane is non-negative",
)

# ERR_SUBRANGE_015: Vector memory storage span is unresolved.
ERR_SUBRANGE_015 = ErrorDef(
    domain=ErrorDomain.SUBRANGE,
    code=15,
    severity=Severity.ERROR,
    summary="Vector memory storage span is unresolved.",
    message="{op_name} linear footprint storage span is unresolved",
    params=(ErrorParam("op_name", ParamKind.STRING),),
    fix_hint="Use a view layout whose storage span can be derived",
)

# ERR_SUBRANGE_016: Vector memory linear footprint upper bound is not proven.
ERR_SUBRANGE_016 = ErrorDef(
    domain=ErrorDomain.SUBRANGE,
    code=16,
    severity=Severity.ERROR,
    summary="Vector memory linear footprint upper bound is not proven.",
    message=(
        "{op_name} linear footprint upper bound is not proven; "
        "linearized_origin + max(offsets) + 1 <= view_storage_span must hold"
    ),
    params=(ErrorParam("op_name", ParamKind.STRING),),
    fix_hint="Prove the maximum offset lane remains inside the view storage span",
)

# ERR_SUBRANGE_017: Vector memory offset vector rank is unsupported.
ERR_SUBRANGE_017 = ErrorDef(
    domain=ErrorDomain.SUBRANGE,
    code=17,
    severity=Severity.ERROR,
    summary="Vector memory offset vector rank is unsupported.",
    message="{op_name} footprint proof requires rank-1 vector offsets",
    params=(ErrorParam("op_name", ParamKind.STRING),),
    fix_hint="Use rank-1 vector offsets for gather, scatter, and vector atomics",
)

# ERR_SUBRANGE_018: Vector memory config-declared upper bound is not proven.
ERR_SUBRANGE_018 = ErrorDef(
    domain=ErrorDomain.SUBRANGE,
    code=18,
    severity=Severity.ERROR,
    summary="Vector memory config-declared upper bound is not proven.",
    message=(
        "{op_name} footprint upper bound is not proven on view axis "
        "{view_axis} (vector axis {vector_axis}); view_bound comes from "
        "config.decl @{config_key}, whose where predicates do not prove "
        "origin {origin} remains in bounds"
    ),
    params=(
        ErrorParam("op_name", ParamKind.STRING),
        ErrorParam("view_axis", ParamKind.I64),
        ErrorParam("vector_axis", ParamKind.I64),
        ErrorParam("origin", ParamKind.STRING),
        ErrorParam("config_key", ParamKind.STRING),
        ErrorParam("constraint_key", ParamKind.STRING),
    ),
    fix_hint=(
        "Strengthen config.decl @{config_key} where predicates or add a "
        "launch/view relation proving {constraint_key}"
    ),
)

ALL_SUBRANGE_ERRORS: tuple[ErrorDef, ...] = (
    ERR_SUBRANGE_001,
    ERR_SUBRANGE_002,
    ERR_SUBRANGE_003,
    ERR_SUBRANGE_004,
    ERR_SUBRANGE_007,
    ERR_SUBRANGE_008,
    ERR_SUBRANGE_009,
    ERR_SUBRANGE_010,
    ERR_SUBRANGE_011,
    ERR_SUBRANGE_012,
    ERR_SUBRANGE_013,
    ERR_SUBRANGE_014,
    ERR_SUBRANGE_015,
    ERR_SUBRANGE_016,
    ERR_SUBRANGE_017,
    ERR_SUBRANGE_018,
)
