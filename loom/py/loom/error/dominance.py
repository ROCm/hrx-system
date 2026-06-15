# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""DOMINANCE domain — undefined values and use-after-consume."""

from loom.errors import ErrorDef, ErrorDomain, ErrorParam, ParamKind, Severity

# ERR_DOMINANCE_001: Use of undefined value.
ERR_DOMINANCE_001 = ErrorDef(
    domain=ErrorDomain.DOMINANCE,
    code=1,
    severity=Severity.ERROR,
    summary="Use of undefined value.",
    message="use of undefined value '{value_name}'",
    params=(ErrorParam("value_name", ParamKind.STRING),),
)

# ERR_DOMINANCE_002: Use of consumed value (after tied result).
ERR_DOMINANCE_002 = ErrorDef(
    domain=ErrorDomain.DOMINANCE,
    code=2,
    severity=Severity.ERROR,
    summary="Use of consumed value.",
    message="use of consumed value '{value_name}' (consumed by tied "
    "result of '{consuming_op}')",
    params=(
        ErrorParam("value_name", ParamKind.STRING),
        ErrorParam("consuming_op", ParamKind.STRING),
    ),
    fix_hint="Use the tied result of '{consuming_op}' instead of '{value_name}'",
)

# ERR_DOMINANCE_003: Value ID out of range.
ERR_DOMINANCE_003 = ErrorDef(
    domain=ErrorDomain.DOMINANCE,
    code=3,
    severity=Severity.ERROR,
    summary="Value ID out of range.",
    message="value ID {value_id} is out of range (max {max_id})",
    params=(
        ErrorParam("value_id", ParamKind.U32),
        ErrorParam("max_id", ParamKind.U32),
    ),
)

# ERR_DOMINANCE_004: Tied result index out of range.
ERR_DOMINANCE_004 = ErrorDef(
    domain=ErrorDomain.DOMINANCE,
    code=4,
    severity=Severity.ERROR,
    summary="Tied result index out of range.",
    message="tied result index {result_index} is out of range for "
    "'{op_name}' which has {result_count} results",
    params=(
        ErrorParam("result_index", ParamKind.U32),
        ErrorParam("op_name", ParamKind.STRING),
        ErrorParam("result_count", ParamKind.U32),
    ),
)

# ERR_DOMINANCE_005: Tied operand index out of range.
ERR_DOMINANCE_005 = ErrorDef(
    domain=ErrorDomain.DOMINANCE,
    code=5,
    severity=Severity.ERROR,
    summary="Tied operand index out of range.",
    message="tied operand index {operand_index} is out of range for "
    "'{op_name}' which has {operand_count} operands",
    params=(
        ErrorParam("operand_index", ParamKind.U32),
        ErrorParam("op_name", ParamKind.STRING),
        ErrorParam("operand_count", ParamKind.U32),
    ),
)

# ERR_DOMINANCE_006: Duplicate tied result index.
ERR_DOMINANCE_006 = ErrorDef(
    domain=ErrorDomain.DOMINANCE,
    code=6,
    severity=Severity.ERROR,
    summary="Duplicate tied result index.",
    message="result {result_index} of '{op_name}' is tied more than once",
    params=(
        ErrorParam("result_index", ParamKind.U32),
        ErrorParam("op_name", ParamKind.STRING),
    ),
)

# ERR_DOMINANCE_007: Duplicate tied operand index.
ERR_DOMINANCE_007 = ErrorDef(
    domain=ErrorDomain.DOMINANCE,
    code=7,
    severity=Severity.ERROR,
    summary="Duplicate tied operand index.",
    message="operand {operand_index} of '{op_name}' is tied by more than one result",
    params=(
        ErrorParam("operand_index", ParamKind.U32),
        ErrorParam("op_name", ParamKind.STRING),
    ),
)

# ERR_DOMINANCE_008: Ambiguous tied operand value.
ERR_DOMINANCE_008 = ErrorDef(
    domain=ErrorDomain.DOMINANCE,
    code=8,
    severity=Severity.ERROR,
    summary="Ambiguous tied operand value.",
    message="operand {operand_index} of '{op_name}' ties value '%{value_name}', "
    "but that value appears in multiple operand slots",
    params=(
        ErrorParam("operand_index", ParamKind.U32),
        ErrorParam("op_name", ParamKind.STRING),
        ErrorParam("value_name", ParamKind.STRING),
    ),
)

# ERR_DOMINANCE_009: Value use count violates a structural lifetime contract.
ERR_DOMINANCE_009 = ErrorDef(
    domain=ErrorDomain.DOMINANCE,
    code=9,
    severity=Severity.ERROR,
    summary="Value use count violates a structural lifetime contract.",
    message="value '%{value_name}' has {actual_count} uses, expected "
    "{expected_constraint}",
    params=(
        ErrorParam("value_name", ParamKind.STRING),
        ErrorParam("actual_count", ParamKind.U32),
        ErrorParam("expected_constraint", ParamKind.STRING),
    ),
    fix_hint="Route '%{value_name}' through the required consumer shape.",
)

# ERR_DOMINANCE_010: Value user violates a structural lifetime contract.
ERR_DOMINANCE_010 = ErrorDef(
    domain=ErrorDomain.DOMINANCE,
    code=10,
    severity=Severity.ERROR,
    summary="Value user violates a structural lifetime contract.",
    message="value '%{value_name}' is used by '{user_op}', expected "
    "{expected_constraint}",
    params=(
        ErrorParam("value_name", ParamKind.STRING),
        ErrorParam("user_op", ParamKind.STRING),
        ErrorParam("expected_constraint", ParamKind.STRING),
    ),
    fix_hint="Use '%{value_name}' only where the lifetime contract allows it.",
)

# ERR_DOMINANCE_011: Low operation reads an undefined register part.
ERR_DOMINANCE_011 = ErrorDef(
    domain=ErrorDomain.DOMINANCE,
    code=11,
    severity=Severity.ERROR,
    summary="Low operation reads an undefined register part.",
    message=(
        "low function '@{function_name}' op '{op_name}' operand "
        "'{field_name}' requires register part mask {required_mask}, but "
        "the value defines only mask {defined_mask}"
    ),
    params=(
        ErrorParam("function_name", ParamKind.STRING),
        ErrorParam("op_name", ParamKind.STRING),
        ErrorParam("field_name", ParamKind.STRING),
        ErrorParam("required_mask", ParamKind.U32),
        ErrorParam("defined_mask", ParamKind.U32),
    ),
    fix_hint=(
        "Compose the missing register part with an explicitly tied partial "
        "write before using the value as a full-register operand"
    ),
)

# ERR_DOMINANCE_012: Ownership lifetime use after consume.
ERR_DOMINANCE_012 = ErrorDef(
    domain=ErrorDomain.DOMINANCE,
    code=12,
    severity=Severity.ERROR,
    summary="Ownership lifetime use after consume.",
    message=("{phase_name} uses consumed resource '%{value_name}' at '{op_name}'"),
    params=(
        ErrorParam("phase_name", ParamKind.STRING),
        ErrorParam("value_name", ParamKind.STRING),
        ErrorParam("op_name", ParamKind.STRING),
    ),
    fix_hint="Use the current owned carrier value or retain before consuming.",
)

# ERR_DOMINANCE_013: Ownership effect requires an owned resource.
ERR_DOMINANCE_013 = ErrorDef(
    domain=ErrorDomain.DOMINANCE,
    code=13,
    severity=Severity.ERROR,
    summary="Ownership effect requires an owned resource.",
    message=(
        "{phase_name} op '{op_name}' requires owned resource "
        "'%{value_name}' for {effect_name}, but the value is {state_name}"
    ),
    params=(
        ErrorParam("phase_name", ParamKind.STRING),
        ErrorParam("op_name", ParamKind.STRING),
        ErrorParam("value_name", ParamKind.STRING),
        ErrorParam("effect_name", ParamKind.STRING),
        ErrorParam("state_name", ParamKind.STRING),
    ),
    fix_hint=(
        "Pass an owned resource, retain a borrowed resource, or use a "
        "borrow-only operation."
    ),
)

# ERR_DOMINANCE_014: Owned resource reaches function exit.
ERR_DOMINANCE_014 = ErrorDef(
    domain=ErrorDomain.DOMINANCE,
    code=14,
    severity=Severity.ERROR,
    summary="Owned resource reaches function exit.",
    message=(
        "{phase_name} resource '%{value_name}' reaches function exit "
        "without consume, release, discard, or escape"
    ),
    params=(
        ErrorParam("phase_name", ParamKind.STRING),
        ErrorParam("value_name", ParamKind.STRING),
    ),
    fix_hint=(
        "Consume, release, discard, escape, or return the resource through an "
        "owned result summary."
    ),
)

# ERR_DOMINANCE_015: Ownership state mismatch at control-flow join.
ERR_DOMINANCE_015 = ErrorDef(
    domain=ErrorDomain.DOMINANCE,
    code=15,
    severity=Severity.ERROR,
    summary="Ownership state mismatch at control-flow join.",
    message=(
        "{phase_name} cannot merge resource '%{value_name}' at a "
        "control-flow join: incoming states are {current_state} and "
        "{incoming_state}"
    ),
    params=(
        ErrorParam("phase_name", ParamKind.STRING),
        ErrorParam("value_name", ParamKind.STRING),
        ErrorParam("current_state", ParamKind.STRING),
        ErrorParam("incoming_state", ParamKind.STRING),
    ),
    fix_hint=(
        "Balance ownership effects on all incoming paths or pass the owned "
        "value through a block argument."
    ),
)

ALL_DOMINANCE_ERRORS: tuple[ErrorDef, ...] = (
    ERR_DOMINANCE_001,
    ERR_DOMINANCE_002,
    ERR_DOMINANCE_003,
    ERR_DOMINANCE_004,
    ERR_DOMINANCE_005,
    ERR_DOMINANCE_006,
    ERR_DOMINANCE_007,
    ERR_DOMINANCE_008,
    ERR_DOMINANCE_009,
    ERR_DOMINANCE_010,
    ERR_DOMINANCE_011,
    ERR_DOMINANCE_012,
    ERR_DOMINANCE_013,
    ERR_DOMINANCE_014,
    ERR_DOMINANCE_015,
)
