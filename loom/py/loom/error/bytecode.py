# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""BYTECODE domain — format errors and version mismatches."""

from loom.errors import ErrorDef, ErrorDomain, ErrorParam, ParamKind, Severity

# ERR_BYTECODE_001: Invalid magic bytes.
ERR_BYTECODE_001 = ErrorDef(
    domain=ErrorDomain.BYTECODE,
    code=1,
    severity=Severity.ERROR,
    summary="Invalid magic bytes.",
    message="invalid magic bytes: expected {expected_magic}, got {actual_magic}",
    params=(
        ErrorParam("expected_magic", ParamKind.STRING),
        ErrorParam("actual_magic", ParamKind.STRING),
    ),
)

# ERR_BYTECODE_002: Unsupported format version.
ERR_BYTECODE_002 = ErrorDef(
    domain=ErrorDomain.BYTECODE,
    code=2,
    severity=Severity.ERROR,
    summary="Unsupported format version.",
    message="unsupported format version {actual_version}, expected {expected_version}",
    params=(
        ErrorParam("actual_version", ParamKind.U32),
        ErrorParam("expected_version", ParamKind.U32),
    ),
)

# ERR_BYTECODE_003: Unexpected end of data.
ERR_BYTECODE_003 = ErrorDef(
    domain=ErrorDomain.BYTECODE,
    code=3,
    severity=Severity.ERROR,
    summary="Unexpected end of data.",
    message="unexpected end of data at offset {offset}, "
    "need {needed_bytes} bytes but only {available_bytes} remain",
    params=(
        ErrorParam("offset", ParamKind.U64),
        ErrorParam("needed_bytes", ParamKind.U64),
        ErrorParam("available_bytes", ParamKind.U64),
    ),
)

# ERR_BYTECODE_004: Unknown type kind.
ERR_BYTECODE_004 = ErrorDef(
    domain=ErrorDomain.BYTECODE,
    code=4,
    severity=Severity.ERROR,
    summary="Unknown type kind.",
    message="unknown type kind {type_kind} at offset {offset}",
    params=(
        ErrorParam("type_kind", ParamKind.U32),
        ErrorParam("offset", ParamKind.U64),
    ),
)

# ERR_BYTECODE_005: Unknown attribute kind.
ERR_BYTECODE_005 = ErrorDef(
    domain=ErrorDomain.BYTECODE,
    code=5,
    severity=Severity.ERROR,
    summary="Unknown attribute kind.",
    message="unknown attribute kind {attr_kind} at offset {offset}",
    params=(
        ErrorParam("attr_kind", ParamKind.U32),
        ErrorParam("offset", ParamKind.U64),
    ),
)

# ERR_BYTECODE_006: Invalid bytecode record field.
ERR_BYTECODE_006 = ErrorDef(
    domain=ErrorDomain.BYTECODE,
    code=6,
    severity=Severity.ERROR,
    summary="Invalid bytecode record field.",
    message="invalid field '{field_name}' in "
    "{section_name}/{table_name}[{record_index}] "
    "at offset {offset}: {failure_code}",
    params=(
        ErrorParam("section_name", ParamKind.STRING),
        ErrorParam("table_name", ParamKind.STRING),
        ErrorParam("record_index", ParamKind.U64),
        ErrorParam("field_name", ParamKind.STRING),
        ErrorParam("offset", ParamKind.U64),
        ErrorParam("failure_code", ParamKind.STRING),
    ),
)

# ERR_BYTECODE_007: Invalid bytecode range.
ERR_BYTECODE_007 = ErrorDef(
    domain=ErrorDomain.BYTECODE,
    code=7,
    severity=Severity.ERROR,
    summary="Invalid bytecode range.",
    message="invalid range '{range_name}' at offset {offset} with length {length}; "
    "container length is {container_length}",
    params=(
        ErrorParam("range_name", ParamKind.STRING),
        ErrorParam("offset", ParamKind.U64),
        ErrorParam("length", ParamKind.U64),
        ErrorParam("container_length", ParamKind.U64),
    ),
)

# ERR_BYTECODE_008: Malformed varint.
ERR_BYTECODE_008 = ErrorDef(
    domain=ErrorDomain.BYTECODE,
    code=8,
    severity=Severity.ERROR,
    summary="Malformed varint.",
    message="malformed varint at offset {offset}: {failure_code}",
    params=(
        ErrorParam("offset", ParamKind.U64),
        ErrorParam("failure_code", ParamKind.STRING),
    ),
)

# ERR_BYTECODE_009: Count exceeds bytecode limit.
ERR_BYTECODE_009 = ErrorDef(
    domain=ErrorDomain.BYTECODE,
    code=9,
    severity=Severity.ERROR,
    summary="Count exceeds bytecode limit.",
    message="{table_name} count {count} exceeds limit {limit}",
    params=(
        ErrorParam("table_name", ParamKind.STRING),
        ErrorParam("count", ParamKind.U64),
        ErrorParam("limit", ParamKind.U64),
    ),
)

# ERR_BYTECODE_010: Invalid string reference.
ERR_BYTECODE_010 = ErrorDef(
    domain=ErrorDomain.BYTECODE,
    code=10,
    severity=Severity.ERROR,
    summary="Invalid string reference.",
    message="invalid string reference in field '{field_name}': id {string_id}, "
    "string table has {string_count} entries",
    params=(
        ErrorParam("field_name", ParamKind.STRING),
        ErrorParam("string_id", ParamKind.U64),
        ErrorParam("string_count", ParamKind.U64),
    ),
)

# ERR_BYTECODE_011: Invalid enum value.
ERR_BYTECODE_011 = ErrorDef(
    domain=ErrorDomain.BYTECODE,
    code=11,
    severity=Severity.ERROR,
    summary="Invalid enum value.",
    message="invalid enum value for field '{field_name}': {actual_value}; "
    "valid case count is {case_count}",
    params=(
        ErrorParam("field_name", ParamKind.STRING),
        ErrorParam("actual_value", ParamKind.U64),
        ErrorParam("case_count", ParamKind.U64),
    ),
)

# ERR_BYTECODE_012: Invalid table reference.
ERR_BYTECODE_012 = ErrorDef(
    domain=ErrorDomain.BYTECODE,
    code=12,
    severity=Severity.ERROR,
    summary="Invalid table reference.",
    message="invalid {table_name} reference {ref_id}; table has {table_count} entries",
    params=(
        ErrorParam("table_name", ParamKind.STRING),
        ErrorParam("ref_id", ParamKind.U64),
        ErrorParam("table_count", ParamKind.U64),
    ),
)

# ERR_BYTECODE_013: Invalid location record.
ERR_BYTECODE_013 = ErrorDef(
    domain=ErrorDomain.BYTECODE,
    code=13,
    severity=Severity.ERROR,
    summary="Invalid location record.",
    message="invalid location record {location_id} at offset {offset}: {failure_code}",
    params=(
        ErrorParam("location_id", ParamKind.U64),
        ErrorParam("offset", ParamKind.U64),
        ErrorParam("failure_code", ParamKind.STRING),
    ),
)

# ERR_BYTECODE_014: Invalid encoding record.
ERR_BYTECODE_014 = ErrorDef(
    domain=ErrorDomain.BYTECODE,
    code=14,
    severity=Severity.ERROR,
    summary="Invalid encoding record.",
    message="invalid encoding record {encoding_id} at offset {offset}: {failure_code}",
    params=(
        ErrorParam("encoding_id", ParamKind.U64),
        ErrorParam("offset", ParamKind.U64),
        ErrorParam("failure_code", ParamKind.STRING),
    ),
)

# ERR_BYTECODE_015: Invalid attribute payload.
ERR_BYTECODE_015 = ErrorDef(
    domain=ErrorDomain.BYTECODE,
    code=15,
    severity=Severity.ERROR,
    summary="Invalid attribute payload.",
    message="invalid attribute payload at offset {offset}: {failure_code}",
    params=(
        ErrorParam("offset", ParamKind.U64),
        ErrorParam("failure_code", ParamKind.STRING),
    ),
)

# ERR_BYTECODE_016: Invalid IR body.
ERR_BYTECODE_016 = ErrorDef(
    domain=ErrorDomain.BYTECODE,
    code=16,
    severity=Severity.ERROR,
    summary="Invalid IR body.",
    message=(
        "invalid IR body for symbol '{symbol_name}' at offset {offset}: {failure_code}"
    ),
    params=(
        ErrorParam("symbol_name", ParamKind.STRING),
        ErrorParam("offset", ParamKind.U64),
        ErrorParam("failure_code", ParamKind.STRING),
    ),
)

# ERR_BYTECODE_017: Resource payload failure.
ERR_BYTECODE_017 = ErrorDef(
    domain=ErrorDomain.BYTECODE,
    code=17,
    severity=Severity.ERROR,
    summary="Resource payload failure.",
    message="resource {resource_id} payload at offset {offset} "
    "could not be read: {failure_code}",
    params=(
        ErrorParam("resource_id", ParamKind.U64),
        ErrorParam("offset", ParamKind.U64),
        ErrorParam("failure_code", ParamKind.STRING),
    ),
)

ALL_BYTECODE_ERRORS: tuple[ErrorDef, ...] = (
    ERR_BYTECODE_001,
    ERR_BYTECODE_002,
    ERR_BYTECODE_003,
    ERR_BYTECODE_004,
    ERR_BYTECODE_005,
    ERR_BYTECODE_006,
    ERR_BYTECODE_007,
    ERR_BYTECODE_008,
    ERR_BYTECODE_009,
    ERR_BYTECODE_010,
    ERR_BYTECODE_011,
    ERR_BYTECODE_012,
    ERR_BYTECODE_013,
    ERR_BYTECODE_014,
    ERR_BYTECODE_015,
    ERR_BYTECODE_016,
    ERR_BYTECODE_017,
)
