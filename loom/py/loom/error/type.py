# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""TYPE domain — type mismatches and constraint violations."""

from loom.errors import ErrorDef, ErrorDomain, ErrorParam, ParamKind, Severity

# ERR_TYPE_001: SameType constraint violated.
ERR_TYPE_001 = ErrorDef(
    domain=ErrorDomain.TYPE,
    code=1,
    severity=Severity.ERROR,
    summary="SameType constraint violated.",
    message="'{field_a}' type {type_a} does not match '{field_b}' type {type_b}",
    params=(
        ErrorParam("field_a", ParamKind.STRING),
        ErrorParam("type_a", ParamKind.TYPE),
        ErrorParam("field_b", ParamKind.STRING),
        ErrorParam("type_b", ParamKind.TYPE),
    ),
    fix_hint="Ensure '{field_a}' and '{field_b}' have the same type",
)

# ERR_TYPE_002: SameElementType constraint violated.
ERR_TYPE_002 = ErrorDef(
    domain=ErrorDomain.TYPE,
    code=2,
    severity=Severity.ERROR,
    summary="Operands must have the same element type.",
    message="element type mismatch: '{field_a}' has element type "
    "{element_type_a}, '{field_b}' has element type {element_type_b}",
    params=(
        ErrorParam("field_a", ParamKind.STRING),
        ErrorParam("element_type_a", ParamKind.TYPE),
        ErrorParam("field_b", ParamKind.STRING),
        ErrorParam("element_type_b", ParamKind.TYPE),
    ),
    fix_hint="Operands '{field_a}' and '{field_b}' must have the same "
    "element type; use explicit type casts if needed",
)

# ERR_TYPE_003: Operand type constraint violated.
ERR_TYPE_003 = ErrorDef(
    domain=ErrorDomain.TYPE,
    code=3,
    severity=Severity.ERROR,
    summary="Operand type constraint violated.",
    message="operand '{operand_name}' has type {actual_type}, "
    "expected {expected_constraint}",
    params=(
        ErrorParam("operand_name", ParamKind.STRING),
        ErrorParam("actual_type", ParamKind.TYPE),
        ErrorParam("expected_constraint", ParamKind.STRING),
    ),
    fix_hint="'{operand_name}' must satisfy type constraint '{expected_constraint}'",
)

# ERR_TYPE_004: Result type constraint violated.
ERR_TYPE_004 = ErrorDef(
    domain=ErrorDomain.TYPE,
    code=4,
    severity=Severity.ERROR,
    summary="Result type constraint violated.",
    message="result '{result_name}' has type {actual_type}, "
    "expected {expected_constraint}",
    params=(
        ErrorParam("result_name", ParamKind.STRING),
        ErrorParam("actual_type", ParamKind.TYPE),
        ErrorParam("expected_constraint", ParamKind.STRING),
    ),
    fix_hint="'{result_name}' must satisfy type constraint '{expected_constraint}'",
)

# ERR_TYPE_005: Attribute kind mismatch.
ERR_TYPE_005 = ErrorDef(
    domain=ErrorDomain.TYPE,
    code=5,
    severity=Severity.ERROR,
    summary="Attribute kind mismatch.",
    message="attribute '{attr_name}' has kind {actual_kind}, expected {expected_kind}",
    params=(
        ErrorParam("attr_name", ParamKind.STRING),
        ErrorParam("actual_kind", ParamKind.U32),
        ErrorParam("expected_kind", ParamKind.U32),
    ),
    fix_hint="'{attr_name}' must be a {expected_kind} attribute",
)

# ERR_TYPE_006: Fill value type mismatch.
ERR_TYPE_006 = ErrorDef(
    domain=ErrorDomain.TYPE,
    code=6,
    severity=Severity.ERROR,
    summary="Fill value type mismatch.",
    message="fill value has type {value_type}, but tile element type is {element_type}",
    params=(
        ErrorParam("value_type", ParamKind.TYPE),
        ErrorParam("element_type", ParamKind.TYPE),
    ),
    fix_hint="Cast or convert fill value to type {element_type}",
)

# ERR_TYPE_007: Constant attribute type mismatch.
ERR_TYPE_007 = ErrorDef(
    domain=ErrorDomain.TYPE,
    code=7,
    severity=Severity.ERROR,
    summary="Constant attribute type mismatch.",
    message="constant attribute has type {attr_type} but result tile "
    "has type {result_type}",
    params=(
        ErrorParam("attr_type", ParamKind.TYPE),
        ErrorParam("result_type", ParamKind.TYPE),
    ),
    fix_hint="Ensure constant attribute shape and element type match '{result_type}'",
)

# ERR_TYPE_008: Block arg type does not match input element type.
ERR_TYPE_008 = ErrorDef(
    domain=ErrorDomain.TYPE,
    code=8,
    severity=Severity.ERROR,
    summary="Block argument type does not match input element type.",
    message="block argument {arg_index} has type {arg_type}, expected "
    "element type {element_type}",
    params=(
        ErrorParam("arg_index", ParamKind.U32),
        ErrorParam("arg_type", ParamKind.TYPE),
        ErrorParam("element_type", ParamKind.TYPE),
    ),
    fix_hint="Block argument should have type {element_type} to match "
    "tile element type",
)

# ERR_TYPE_013: Block arg type does not match input type.
ERR_TYPE_013 = ErrorDef(
    domain=ErrorDomain.TYPE,
    code=13,
    severity=Severity.ERROR,
    summary="Block argument type does not match input type.",
    message="block argument {arg_index} has type {arg_type}, expected "
    "type {expected_type}",
    params=(
        ErrorParam("arg_index", ParamKind.U32),
        ErrorParam("arg_type", ParamKind.TYPE),
        ErrorParam("expected_type", ParamKind.TYPE),
    ),
    fix_hint="Block argument should have type {expected_type}",
)

# ERR_TYPE_014: Block argument type constraint violated.
ERR_TYPE_014 = ErrorDef(
    domain=ErrorDomain.TYPE,
    code=14,
    severity=Severity.ERROR,
    summary="Block argument type constraint violated.",
    message="block argument '{argument_name}' has type {actual_type}, "
    "expected {expected_constraint}",
    params=(
        ErrorParam("argument_name", ParamKind.STRING),
        ErrorParam("actual_type", ParamKind.TYPE),
        ErrorParam("expected_constraint", ParamKind.STRING),
    ),
    fix_hint=("'{argument_name}' must satisfy type constraint '{expected_constraint}'"),
)

# ERR_TYPE_015: Boundary facts conflict with a boundary value type.
ERR_TYPE_015 = ErrorDef(
    domain=ErrorDomain.TYPE,
    code=15,
    severity=Severity.ERROR,
    summary="Boundary facts conflict with a boundary value type.",
    message=(
        "{pass_name} found boundary facts for {op_name} that conflict with "
        "boundary type {actual_type}"
    ),
    params=(
        ErrorParam("op_name", ParamKind.STRING),
        ErrorParam("pass_name", ParamKind.STRING),
        ErrorParam("actual_type", ParamKind.TYPE),
    ),
    fix_hint=(
        "Refine the producer facts or specialize the function boundary so the "
        "boundary value type can represent the propagated facts"
    ),
)

# ERR_TYPE_016: Callee result type conflicts with a call result type.
ERR_TYPE_016 = ErrorDef(
    domain=ErrorDomain.TYPE,
    code=16,
    severity=Severity.ERROR,
    summary="Callee result type conflicts with a call result type.",
    message=(
        "{pass_name} found {op_name} with call result type {actual_type} "
        "conflicting with substituted callee result type {candidate_type}"
    ),
    params=(
        ErrorParam("op_name", ParamKind.STRING),
        ErrorParam("pass_name", ParamKind.STRING),
        ErrorParam("actual_type", ParamKind.TYPE),
        ErrorParam("candidate_type", ParamKind.TYPE),
    ),
    fix_hint=(
        "Specialize the callee or callsite so substituted boundary dimensions "
        "agree before refining call result types"
    ),
)

# ERR_TYPE_009: Yield type does not match result type.
ERR_TYPE_009 = ErrorDef(
    domain=ErrorDomain.TYPE,
    code=9,
    severity=Severity.ERROR,
    summary="Yielded value type does not match expected result type.",
    message="yielded value has type {yield_type}, but expected {expected_type}",
    params=(
        ErrorParam("yield_type", ParamKind.TYPE),
        ErrorParam("expected_type", ParamKind.TYPE),
    ),
    fix_hint="Convert yielded value to type {expected_type}",
)

# ERR_TYPE_010: Malformed type payload.
ERR_TYPE_010 = ErrorDef(
    domain=ErrorDomain.TYPE,
    code=10,
    severity=Severity.ERROR,
    summary="Malformed type payload.",
    message=("'{field_name}' has malformed type {actual_type}: {malformation_code}"),
    params=(
        ErrorParam("field_name", ParamKind.STRING),
        ErrorParam("actual_type", ParamKind.TYPE),
        ErrorParam("malformation_code", ParamKind.STRING),
    ),
    fix_hint="Construct a well-formed type before verification",
)

# ERR_TYPE_011: Integer code result type is too small.
ERR_TYPE_011 = ErrorDef(
    domain=ErrorDomain.TYPE,
    code=11,
    severity=Severity.ERROR,
    summary="Integer code result type is too small.",
    message="result '{result_name}' has type {actual_type}, but {threshold_count} "
    "thresholds require codes in [0, {max_code}]",
    params=(
        ErrorParam("result_name", ParamKind.STRING),
        ErrorParam("actual_type", ParamKind.TYPE),
        ErrorParam("threshold_count", ParamKind.I64),
        ErrorParam("max_code", ParamKind.I64),
    ),
    fix_hint="Use a result integer element type with enough bits for every code",
)

# ERR_TYPE_012: Poison value observed at a boundary.
ERR_TYPE_012 = ErrorDef(
    domain=ErrorDomain.TYPE,
    code=12,
    severity=Severity.ERROR,
    summary="Poison value observed at boundary.",
    message="poison value '{value_name}' cannot be used by {boundary_kind} '{op_name}'",
    params=(
        ErrorParam("value_name", ParamKind.STRING),
        ErrorParam("boundary_kind", ParamKind.STRING),
        ErrorParam("op_name", ParamKind.STRING),
    ),
    fix_hint="Canonicalize away dead poison or prevent the invalid observation "
    "before it reaches this boundary",
)

ALL_TYPE_ERRORS: tuple[ErrorDef, ...] = (
    ERR_TYPE_001,
    ERR_TYPE_002,
    ERR_TYPE_003,
    ERR_TYPE_004,
    ERR_TYPE_005,
    ERR_TYPE_006,
    ERR_TYPE_007,
    ERR_TYPE_008,
    ERR_TYPE_009,
    ERR_TYPE_010,
    ERR_TYPE_011,
    ERR_TYPE_012,
    ERR_TYPE_013,
    ERR_TYPE_014,
    ERR_TYPE_015,
    ERR_TYPE_016,
)
