# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""PARSE domain — syntax errors and tokenization."""

from loom.errors import ErrorDef, ErrorDomain, ErrorParam, ParamKind, Severity

# ERR_PARSE_001: Undefined SSA value.
ERR_PARSE_001 = ErrorDef(
    domain=ErrorDomain.PARSE,
    code=1,
    severity=Severity.ERROR,
    summary="Undefined SSA value.",
    message="undefined SSA value '%{value_name}'",
    params=(ErrorParam("value_name", ParamKind.STRING),),
)

# ERR_PARSE_002: Duplicate SSA name.
ERR_PARSE_002 = ErrorDef(
    domain=ErrorDomain.PARSE,
    code=2,
    severity=Severity.ERROR,
    summary="Duplicate SSA name.",
    message="SSA value '%{value_name}' is already defined",
    params=(ErrorParam("value_name", ParamKind.STRING),),
    fix_hint="Each SSA value name must be unique within its scope",
)

# ERR_PARSE_003: Unexpected token.
ERR_PARSE_003 = ErrorDef(
    domain=ErrorDomain.PARSE,
    code=3,
    severity=Severity.ERROR,
    summary="Unexpected token.",
    message="unexpected token '{actual_token}', expected {expected}",
    params=(
        ErrorParam("actual_token", ParamKind.STRING),
        ErrorParam("expected", ParamKind.STRING),
    ),
)

# ERR_PARSE_004: Invalid type syntax.
ERR_PARSE_004 = ErrorDef(
    domain=ErrorDomain.PARSE,
    code=4,
    severity=Severity.ERROR,
    summary="Invalid type syntax.",
    message="invalid type syntax: '{text}'",
    params=(ErrorParam("text", ParamKind.STRING),),
)

# ERR_PARSE_005: Unterminated string literal.
ERR_PARSE_005 = ErrorDef(
    domain=ErrorDomain.PARSE,
    code=5,
    severity=Severity.ERROR,
    summary="Unterminated string literal.",
    message="unterminated string literal",
    params=(),
)

# ERR_PARSE_006: Unknown op name.
ERR_PARSE_006 = ErrorDef(
    domain=ErrorDomain.PARSE,
    code=6,
    severity=Severity.ERROR,
    summary="Unknown op name.",
    message="unknown op '{op_name}'",
    params=(ErrorParam("op_name", ParamKind.STRING),),
    fix_hint=(
        "Check that the dialect is registered and the op name is spelled correctly"
    ),
)

# ERR_PARSE_007: Unknown type name.
ERR_PARSE_007 = ErrorDef(
    domain=ErrorDomain.PARSE,
    code=7,
    severity=Severity.ERROR,
    summary="Unknown type name.",
    message="unknown type '{type_name}'",
    params=(ErrorParam("type_name", ParamKind.STRING),),
)

# ERR_PARSE_008: Unknown encoding.
ERR_PARSE_008 = ErrorDef(
    domain=ErrorDomain.PARSE,
    code=8,
    severity=Severity.ERROR,
    summary="Unknown encoding.",
    message="unknown encoding '{encoding_name}'",
    params=(ErrorParam("encoding_name", ParamKind.STRING),),
)

# ERR_PARSE_009: Result count mismatch.
ERR_PARSE_009 = ErrorDef(
    domain=ErrorDomain.PARSE,
    code=9,
    severity=Severity.ERROR,
    summary="Result count mismatch.",
    message="op '{op_name}' expects {expected_count} results, got {actual_count}",
    params=(
        ErrorParam("op_name", ParamKind.STRING),
        ErrorParam("expected_count", ParamKind.U32),
        ErrorParam("actual_count", ParamKind.U32),
    ),
)

# ERR_PARSE_010: Operand count mismatch.
ERR_PARSE_010 = ErrorDef(
    domain=ErrorDomain.PARSE,
    code=10,
    severity=Severity.ERROR,
    summary="Operand count mismatch.",
    message="op '{op_name}' expects {expected_count} operands, got {actual_count}",
    params=(
        ErrorParam("op_name", ParamKind.STRING),
        ErrorParam("expected_count", ParamKind.U32),
        ErrorParam("actual_count", ParamKind.U32),
    ),
)

# ERR_PARSE_011: Invalid location syntax.
ERR_PARSE_011 = ErrorDef(
    domain=ErrorDomain.PARSE,
    code=11,
    severity=Severity.ERROR,
    summary="Invalid location syntax.",
    message="malformed loc() annotation: {detail}",
    params=(ErrorParam("detail", ParamKind.STRING),),
)

# ERR_PARSE_012: Too many errors.
ERR_PARSE_012 = ErrorDef(
    domain=ErrorDomain.PARSE,
    code=12,
    severity=Severity.ERROR,
    summary="Too many errors.",
    message="too many errors ({error_count}), stopping",
    params=(ErrorParam("error_count", ParamKind.U32),),
)

# ERR_PARSE_013: Unknown predicate kind.
ERR_PARSE_013 = ErrorDef(
    domain=ErrorDomain.PARSE,
    code=13,
    severity=Severity.ERROR,
    summary="Unknown predicate kind.",
    message="unknown predicate '{predicate_name}'",
    params=(ErrorParam("predicate_name", ParamKind.STRING),),
)

# ERR_PARSE_014: Invalid encoding alias.
ERR_PARSE_014 = ErrorDef(
    domain=ErrorDomain.PARSE,
    code=14,
    severity=Severity.ERROR,
    summary="Invalid encoding alias.",
    message="invalid encoding alias definition: {detail}",
    params=(ErrorParam("detail", ParamKind.STRING),),
)

# ERR_PARSE_015: Invalid integer literal.
ERR_PARSE_015 = ErrorDef(
    domain=ErrorDomain.PARSE,
    code=15,
    severity=Severity.ERROR,
    summary="Invalid integer literal.",
    message="invalid integer literal '{text}'",
    params=(ErrorParam("text", ParamKind.STRING),),
)

# ERR_PARSE_016: Invalid float literal.
ERR_PARSE_016 = ErrorDef(
    domain=ErrorDomain.PARSE,
    code=16,
    severity=Severity.ERROR,
    summary="Invalid float literal.",
    message="invalid float literal '{text}'",
    params=(ErrorParam("text", ParamKind.STRING),),
)

# ERR_PARSE_017: Unknown enum value.
ERR_PARSE_017 = ErrorDef(
    domain=ErrorDomain.PARSE,
    code=17,
    severity=Severity.ERROR,
    summary="Unknown enum value.",
    message="unknown {enum_name} value '{value}'",
    params=(
        ErrorParam("enum_name", ParamKind.STRING),
        ErrorParam("value", ParamKind.STRING),
    ),
)

# ERR_PARSE_018: Unknown flag or modifier.
ERR_PARSE_018 = ErrorDef(
    domain=ErrorDomain.PARSE,
    code=18,
    severity=Severity.ERROR,
    summary="Unknown flag or modifier.",
    message="unknown {context} '{value}'",
    params=(
        ErrorParam("context", ParamKind.STRING),
        ErrorParam("value", ParamKind.STRING),
    ),
)

# ERR_PARSE_019: Invalid UTF-8 in source.
ERR_PARSE_019 = ErrorDef(
    domain=ErrorDomain.PARSE,
    code=19,
    severity=Severity.ERROR,
    summary="Source contains invalid UTF-8.",
    message="invalid UTF-8 byte sequence at offset {byte_offset}",
    params=(ErrorParam("byte_offset", ParamKind.U32),),
    fix_hint="The source file must be valid UTF-8",
)

# ERR_PARSE_020: Duplicate dictionary attribute key.
ERR_PARSE_020 = ErrorDef(
    domain=ErrorDomain.PARSE,
    code=20,
    severity=Severity.ERROR,
    summary="Duplicate dictionary attribute key.",
    message="dictionary attribute key '{key_name}' is already defined",
    params=(ErrorParam("key_name", ParamKind.STRING),),
    fix_hint="Each key in an attribute dictionary must be unique",
)

# ERR_PARSE_021: Dictionary attribute nesting exceeds the supported depth.
ERR_PARSE_021 = ErrorDef(
    domain=ErrorDomain.PARSE,
    code=21,
    severity=Severity.ERROR,
    summary="Dictionary attribute nesting is too deep.",
    message="dictionary attribute nesting exceeds the maximum depth {max_depth}",
    params=(ErrorParam("max_depth", ParamKind.U32),),
    fix_hint="Flatten deeply nested attribute dictionaries",
)

# ERR_PARSE_022: Unresolved forward reference in a declaration scope.
ERR_PARSE_022 = ErrorDef(
    domain=ErrorDomain.PARSE,
    code=22,
    severity=Severity.ERROR,
    summary="Unresolved declaration-scope forward reference.",
    message="unresolved forward reference to '%{value_name}' in declaration scope",
    params=(ErrorParam("value_name", ParamKind.STRING),),
    fix_hint=(
        "Define the SSA name in the same Scope(...) signature, "
        "for example as a function argument or named symbol result"
    ),
)

# ERR_PARSE_023: Invalid escape sequence in a string literal.
ERR_PARSE_023 = ErrorDef(
    domain=ErrorDomain.PARSE,
    code=23,
    severity=Severity.ERROR,
    summary="Invalid string literal escape.",
    message="invalid string literal escape: {detail}",
    params=(ErrorParam("detail", ParamKind.STRING),),
)

# ERR_PARSE_024: Expected an identifier after a sigil prefix.
ERR_PARSE_024 = ErrorDef(
    domain=ErrorDomain.PARSE,
    code=24,
    severity=Severity.ERROR,
    summary="Missing identifier after a sigil prefix.",
    message="expected identifier after '{sigil}'",
    params=(ErrorParam("sigil", ParamKind.STRING),),
)

# ERR_PARSE_025: Unexpected character in source text.
ERR_PARSE_025 = ErrorDef(
    domain=ErrorDomain.PARSE,
    code=25,
    severity=Severity.ERROR,
    summary="Unexpected source character.",
    message="unexpected character",
    params=(),
)

# ERR_PARSE_026: Unterminated angle bracket.
ERR_PARSE_026 = ErrorDef(
    domain=ErrorDomain.PARSE,
    code=26,
    severity=Severity.ERROR,
    summary="Unterminated angle bracket.",
    message="unterminated angle bracket",
    params=(),
)

# ERR_PARSE_027: Duplicate operand dictionary key.
ERR_PARSE_027 = ErrorDef(
    domain=ErrorDomain.PARSE,
    code=27,
    severity=Severity.ERROR,
    summary="Duplicate operand dictionary key.",
    message="operand dictionary key '{key_name}' is already defined",
    params=(ErrorParam("key_name", ParamKind.STRING),),
    fix_hint="Each key in an operand dictionary must be unique",
)

# ERR_PARSE_028: Static encoding parameter used an SSA value.
ERR_PARSE_028 = ErrorDef(
    domain=ErrorDomain.PARSE,
    code=28,
    severity=Severity.ERROR,
    summary="Static encoding parameter uses an SSA value.",
    message="static encoding parameter '{param_name}' cannot use an SSA value",
    params=(ErrorParam("param_name", ParamKind.STRING),),
    fix_hint=(
        "Keep #family<...> parameters static; pass dynamic parameters with "
        "encoding.define #family<static attrs> {name = %value : type} : encoding<role>"
    ),
)

# ERR_PARSE_029: Duplicate encoding parameter.
ERR_PARSE_029 = ErrorDef(
    domain=ErrorDomain.PARSE,
    code=29,
    severity=Severity.ERROR,
    summary="Duplicate encoding parameter.",
    message="encoding parameter '{param_name}' is already defined",
    params=(ErrorParam("param_name", ParamKind.STRING),),
    fix_hint="Each static encoding parameter name must be unique",
)

# ERR_PARSE_030: Attribute table row width mismatch.
ERR_PARSE_030 = ErrorDef(
    domain=ErrorDomain.PARSE,
    code=30,
    severity=Severity.ERROR,
    summary="Attribute table row width mismatch.",
    message=(
        "attribute table row has {actual_count} values, expected {expected_count}"
    ),
    params=(
        ErrorParam("actual_count", ParamKind.U32),
        ErrorParam("expected_count", ParamKind.U32),
    ),
    fix_hint="Every explicit row and the default row must have the same width",
)

# ERR_PARSE_031: Predicate argument count mismatch.
ERR_PARSE_031 = ErrorDef(
    domain=ErrorDomain.PARSE,
    code=31,
    severity=Severity.ERROR,
    summary="Predicate argument count mismatch.",
    message=(
        "predicate '{predicate_name}' expects {expected_count} arguments, "
        "got {actual_count}"
    ),
    params=(
        ErrorParam("predicate_name", ParamKind.STRING),
        ErrorParam("expected_count", ParamKind.U32),
        ErrorParam("actual_count", ParamKind.U32),
    ),
    fix_hint="Use the predicate arity defined by the predicate vocabulary",
)

# ERR_PARSE_032: Undefined block label.
ERR_PARSE_032 = ErrorDef(
    domain=ErrorDomain.PARSE,
    code=32,
    severity=Severity.ERROR,
    summary="Undefined block label.",
    message="undefined block label '^{label_name}'",
    params=(ErrorParam("label_name", ParamKind.STRING),),
    fix_hint="Define the target block label in the same region",
)

# ERR_PARSE_033: Duplicate block label.
ERR_PARSE_033 = ErrorDef(
    domain=ErrorDomain.PARSE,
    code=33,
    severity=Severity.ERROR,
    summary="Duplicate block label.",
    message="block label '^{label_name}' is already defined in this region",
    params=(ErrorParam("label_name", ParamKind.STRING),),
    fix_hint="Each block label must be unique within its region",
)

# ERR_PARSE_034: Invalid low asm syntax.
ERR_PARSE_034 = ErrorDef(
    domain=ErrorDomain.PARSE,
    code=34,
    severity=Severity.ERROR,
    summary="Invalid low asm syntax.",
    message="invalid low asm syntax: {detail}",
    params=(ErrorParam("detail", ParamKind.STRING),),
)

ALL_PARSE_ERRORS: tuple[ErrorDef, ...] = (
    ERR_PARSE_001,
    ERR_PARSE_002,
    ERR_PARSE_003,
    ERR_PARSE_004,
    ERR_PARSE_005,
    ERR_PARSE_006,
    ERR_PARSE_007,
    ERR_PARSE_008,
    ERR_PARSE_009,
    ERR_PARSE_010,
    ERR_PARSE_011,
    ERR_PARSE_012,
    ERR_PARSE_013,
    ERR_PARSE_014,
    ERR_PARSE_015,
    ERR_PARSE_016,
    ERR_PARSE_017,
    ERR_PARSE_018,
    ERR_PARSE_019,
    ERR_PARSE_020,
    ERR_PARSE_021,
    ERR_PARSE_022,
    ERR_PARSE_023,
    ERR_PARSE_024,
    ERR_PARSE_025,
    ERR_PARSE_026,
    ERR_PARSE_027,
    ERR_PARSE_028,
    ERR_PARSE_029,
    ERR_PARSE_030,
    ERR_PARSE_031,
    ERR_PARSE_032,
    ERR_PARSE_033,
    ERR_PARSE_034,
)
