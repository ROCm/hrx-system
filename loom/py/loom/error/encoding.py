# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""ENCODING domain — encoding mismatches."""

from loom.errors import ErrorDef, ErrorDomain, ErrorParam, ParamKind, Severity

# ERR_ENCODING_001: SameEncoding constraint violated.
ERR_ENCODING_001 = ErrorDef(
    domain=ErrorDomain.ENCODING,
    code=1,
    severity=Severity.ERROR,
    summary="Operands must have the same encoding.",
    message="encoding mismatch: '{field_a}' and '{field_b}' have different encodings",
    params=(
        ErrorParam("field_a", ParamKind.STRING),
        ErrorParam("field_b", ParamKind.STRING),
    ),
    fix_hint="Operands '{field_a}' and '{field_b}' must have the same "
    "encoding; use explicit encoding casts if needed",
)

# ERR_ENCODING_002: Encoding must be preserved across slice/view ops.
ERR_ENCODING_002 = ErrorDef(
    domain=ErrorDomain.ENCODING,
    code=2,
    severity=Severity.ERROR,
    summary="Encoding must be preserved across slice/view operations.",
    message="result encoding does not match source encoding for '{op_name}'",
    params=(ErrorParam("op_name", ParamKind.STRING),),
    fix_hint="Slice and view operations must preserve the source encoding",
)

# ERR_ENCODING_003: SSA encoding value_id out of range.
ERR_ENCODING_003 = ErrorDef(
    domain=ErrorDomain.ENCODING,
    code=3,
    severity=Severity.ERROR,
    summary="SSA encoding reference out of range.",
    message="{field_name} type references encoding value %{value_id}, "
    "but module has only {value_count} values",
    params=(
        ErrorParam("field_name", ParamKind.STRING),
        ErrorParam("value_id", ParamKind.U32),
        ErrorParam("value_count", ParamKind.U32),
    ),
    fix_hint="The encoding value_id in the type is invalid; this "
    "indicates a corrupted or malformed module",
)

# ERR_ENCODING_004: SSA encoding value not defined at point of use.
ERR_ENCODING_004 = ErrorDef(
    domain=ErrorDomain.ENCODING,
    code=4,
    severity=Severity.ERROR,
    summary="SSA encoding value not visible at point of use.",
    message="{field_name} type references encoding value '%{value_name}' "
    "which is not defined at this point",
    params=(
        ErrorParam("field_name", ParamKind.STRING),
        ErrorParam("value_name", ParamKind.STRING),
    ),
    fix_hint="The encoding value must be defined before it can be "
    "referenced in a type; ensure the encoding.define op dominates this use",
)

# ERR_ENCODING_005: SSA encoding value has wrong type.
ERR_ENCODING_005 = ErrorDef(
    domain=ErrorDomain.ENCODING,
    code=5,
    severity=Severity.ERROR,
    summary="SSA encoding reference has wrong type.",
    message="{field_name} type references encoding value '%{value_name}' "
    "which has type {actual_type}, expected 'encoding'",
    params=(
        ErrorParam("field_name", ParamKind.STRING),
        ErrorParam("value_name", ParamKind.STRING),
        ErrorParam("actual_type", ParamKind.TYPE),
    ),
    fix_hint="The encoding value must have type 'encoding'; "
    "it should be the result of an encoding.define op",
)

# ERR_ENCODING_006: encoding.define parameter is both static and dynamic.
ERR_ENCODING_006 = ErrorDef(
    domain=ErrorDomain.ENCODING,
    code=6,
    severity=Severity.ERROR,
    summary="Encoding parameter is both static and dynamic.",
    message="encoding.define parameter '{param_name}' is both static and dynamic",
    params=(ErrorParam("param_name", ParamKind.STRING),),
    fix_hint=(
        "Keep compile-time values in #family<...> and SSA values in the "
        "encoding.define operand dictionary; each parameter name must appear "
        "in exactly one place"
    ),
)

# ERR_ENCODING_007: encoding.define is missing a required family parameter.
ERR_ENCODING_007 = ErrorDef(
    domain=ErrorDomain.ENCODING,
    code=7,
    severity=Severity.ERROR,
    summary="Required encoding parameter is missing.",
    message="encoding '{encoding_name}' requires parameter '{param_name}'",
    params=(
        ErrorParam("encoding_name", ParamKind.STRING),
        ErrorParam("param_name", ParamKind.STRING),
    ),
    fix_hint="Add the required parameter to the encoding.define operand dictionary",
)

# ERR_ENCODING_008: encoding.define parameter is not supported by the family.
ERR_ENCODING_008 = ErrorDef(
    domain=ErrorDomain.ENCODING,
    code=8,
    severity=Severity.ERROR,
    summary="Unknown encoding parameter.",
    message="encoding '{encoding_name}' does not support parameter '{param_name}'",
    params=(
        ErrorParam("encoding_name", ParamKind.STRING),
        ErrorParam("param_name", ParamKind.STRING),
    ),
    fix_hint=(
        "Remove the unsupported parameter or use an encoding family that defines it"
    ),
)

# ERR_ENCODING_009: encoding.define parameter has the wrong SSA type.
ERR_ENCODING_009 = ErrorDef(
    domain=ErrorDomain.ENCODING,
    code=9,
    severity=Severity.ERROR,
    summary="Encoding parameter has wrong type.",
    message="encoding '{encoding_name}' parameter '{param_name}' has type "
    "{actual_type}, expected '{expected_type}'",
    params=(
        ErrorParam("encoding_name", ParamKind.STRING),
        ErrorParam("param_name", ParamKind.STRING),
        ErrorParam("actual_type", ParamKind.TYPE),
        ErrorParam("expected_type", ParamKind.STRING),
    ),
    fix_hint="Pass an SSA value whose type matches the encoding family contract",
)

# ERR_ENCODING_010: encoding.define static parameter has the wrong attr kind.
ERR_ENCODING_010 = ErrorDef(
    domain=ErrorDomain.ENCODING,
    code=10,
    severity=Severity.ERROR,
    summary="Encoding static parameter has wrong attribute kind.",
    message="encoding '{encoding_name}' parameter '{param_name}' has attribute "
    "kind {actual_kind}, expected '{expected_kind}'",
    params=(
        ErrorParam("encoding_name", ParamKind.STRING),
        ErrorParam("param_name", ParamKind.STRING),
        ErrorParam("actual_kind", ParamKind.U32),
        ErrorParam("expected_kind", ParamKind.STRING),
    ),
    fix_hint="Use a static parameter value whose kind matches the encoding contract",
)

# ERR_ENCODING_011: encoding.define parameter has the wrong semantic role.
ERR_ENCODING_011 = ErrorDef(
    domain=ErrorDomain.ENCODING,
    code=11,
    severity=Severity.ERROR,
    summary="Encoding parameter has wrong role.",
    message=(
        "encoding '{encoding_name}' parameter '{param_name}' must be {expected_role}"
    ),
    params=(
        ErrorParam("encoding_name", ParamKind.STRING),
        ErrorParam("param_name", ParamKind.STRING),
        ErrorParam("expected_role", ParamKind.STRING),
    ),
    fix_hint="Pass an encoding value with the role required by the family contract",
)

# ERR_ENCODING_012: encoding.define result type has the wrong role.
ERR_ENCODING_012 = ErrorDef(
    domain=ErrorDomain.ENCODING,
    code=12,
    severity=Severity.ERROR,
    summary="Encoding result has wrong role.",
    message=(
        "encoding '{encoding_name}' result has type {actual_type}, "
        "expected {expected_type}"
    ),
    params=(
        ErrorParam("encoding_name", ParamKind.STRING),
        ErrorParam("actual_type", ParamKind.TYPE),
        ErrorParam("expected_type", ParamKind.TYPE),
    ),
    fix_hint=(
        "Spell the encoding.define result type with the role produced by the family"
    ),
)

# ERR_ENCODING_013: encoding.define static parameter has an unsupported value.
ERR_ENCODING_013 = ErrorDef(
    domain=ErrorDomain.ENCODING,
    code=13,
    severity=Severity.ERROR,
    summary="Encoding static parameter has unsupported value.",
    message=(
        "encoding '{encoding_name}' parameter '{param_name}' has value "
        "'{actual_value}', expected {expected_values}"
    ),
    params=(
        ErrorParam("encoding_name", ParamKind.STRING),
        ErrorParam("param_name", ParamKind.STRING),
        ErrorParam("actual_value", ParamKind.STRING),
        ErrorParam("expected_values", ParamKind.STRING),
    ),
    fix_hint=(
        "Use one of the static parameter values supported by the encoding family"
    ),
)

# ERR_ENCODING_014: encoding.define dynamic parameters are mutually exclusive.
ERR_ENCODING_014 = ErrorDef(
    domain=ErrorDomain.ENCODING,
    code=14,
    severity=Severity.ERROR,
    summary="Encoding parameters are mutually exclusive.",
    message=(
        "encoding '{encoding_name}' supports only one of parameters "
        "'{param_a}' and '{param_b}'"
    ),
    params=(
        ErrorParam("encoding_name", ParamKind.STRING),
        ErrorParam("param_a", ParamKind.STRING),
        ErrorParam("param_b", ParamKind.STRING),
    ),
    fix_hint="Keep the single parameter that defines the intended encoding behavior",
)

# ERR_ENCODING_015: vector encoded auxiliary operand key is not supported.
ERR_ENCODING_015 = ErrorDef(
    domain=ErrorDomain.ENCODING,
    code=15,
    severity=Severity.ERROR,
    summary="Unknown encoded auxiliary operand key.",
    message="{op_name} auxiliary operand key '{key_name}' is not supported",
    params=(
        ErrorParam("op_name", ParamKind.STRING),
        ErrorParam("key_name", ParamKind.STRING),
    ),
    fix_hint="Use one of the encoded auxiliary keys understood by vector lowering",
)

# ERR_ENCODING_016: vector encoded auxiliary operand key is required.
ERR_ENCODING_016 = ErrorDef(
    domain=ErrorDomain.ENCODING,
    code=16,
    severity=Severity.ERROR,
    summary="Required encoded auxiliary operand is missing.",
    message="{op_name} auxiliary operands require key '{key_name}'",
    params=(
        ErrorParam("op_name", ParamKind.STRING),
        ErrorParam("key_name", ParamKind.STRING),
    ),
    fix_hint="Provide the required SSA operand in the keyed auxiliary dictionary",
)

# ERR_ENCODING_017: Vector transform descriptor is not locally decodable.
ERR_ENCODING_017 = ErrorDef(
    domain=ErrorDomain.ENCODING,
    code=17,
    severity=Severity.ERROR,
    summary="Vector transform descriptor is not locally decodable.",
    message=(
        "{pass_name} requires {op_name} transform to be a local "
        "#numeric_transform encoding.define"
    ),
    params=(
        ErrorParam("op_name", ParamKind.STRING),
        ErrorParam("pass_name", ParamKind.STRING),
    ),
    fix_hint=(
        "Specialize the transform descriptor into the function before vector "
        "scalarization or lower the transform through a target primitive"
    ),
)

# ERR_ENCODING_018: Vector transform permutation is not statically proven.
ERR_ENCODING_018 = ErrorDef(
    domain=ErrorDomain.ENCODING,
    code=18,
    severity=Severity.ERROR,
    summary="Vector transform permutation is not statically proven.",
    message=("{pass_name} requires {op_name} permutation lanes to be statically exact"),
    params=(
        ErrorParam("op_name", ParamKind.STRING),
        ErrorParam("pass_name", ParamKind.STRING),
    ),
    fix_hint=(
        "Refine the permutation lane facts before vector scalarization or "
        "specialize the transform through a target primitive"
    ),
)

# ERR_ENCODING_019: Vector transform permutation repeats a source lane.
ERR_ENCODING_019 = ErrorDef(
    domain=ErrorDomain.ENCODING,
    code=19,
    severity=Severity.ERROR,
    summary="Vector transform permutation repeats a source lane.",
    message=(
        "{pass_name} requires {op_name} permutation to name each source lane "
        "once per last-axis slice"
    ),
    params=(
        ErrorParam("op_name", ParamKind.STRING),
        ErrorParam("pass_name", ParamKind.STRING),
    ),
    fix_hint=(
        "Use a bijective permutation for each last-axis slice before vector "
        "scalarization"
    ),
)

# ERR_ENCODING_020: Vector transform permutation source lane is out of bounds.
ERR_ENCODING_020 = ErrorDef(
    domain=ErrorDomain.ENCODING,
    code=20,
    severity=Severity.ERROR,
    summary="Vector transform permutation source lane is out of bounds.",
    message=(
        "{pass_name} requires {op_name} permutation lanes to reference "
        "in-bounds source lanes"
    ),
    params=(
        ErrorParam("op_name", ParamKind.STRING),
        ErrorParam("pass_name", ParamKind.STRING),
    ),
    fix_hint="Clamp or specialize permutation lanes to the source last-axis extent",
)

ALL_ENCODING_ERRORS: tuple[ErrorDef, ...] = (
    ERR_ENCODING_001,
    ERR_ENCODING_002,
    ERR_ENCODING_003,
    ERR_ENCODING_004,
    ERR_ENCODING_005,
    ERR_ENCODING_006,
    ERR_ENCODING_007,
    ERR_ENCODING_008,
    ERR_ENCODING_009,
    ERR_ENCODING_010,
    ERR_ENCODING_011,
    ERR_ENCODING_012,
    ERR_ENCODING_013,
    ERR_ENCODING_014,
    ERR_ENCODING_015,
    ERR_ENCODING_016,
    ERR_ENCODING_017,
    ERR_ENCODING_018,
    ERR_ENCODING_019,
    ERR_ENCODING_020,
)
