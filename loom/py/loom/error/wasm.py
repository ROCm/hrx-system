# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""WASM domain — WebAssembly-owned legality and lowering diagnostics."""

from loom.errors import ErrorDef, ErrorDomain, ErrorParam, ParamKind, Severity

_TARGET_CONTEXT_PARAMS = (
    ErrorParam("target_key", ParamKind.STRING),
    ErrorParam("export_name", ParamKind.STRING),
    ErrorParam("config_key", ParamKind.STRING),
    ErrorParam("function_name", ParamKind.STRING),
    ErrorParam("op_name", ParamKind.STRING),
)

# ERR_WASM_001: Wasm source value type is unsupported.
ERR_WASM_001 = ErrorDef(
    domain=ErrorDomain.WASM,
    code=1,
    severity=Severity.ERROR,
    summary="Wasm source value type is unsupported.",
    message=(
        "Wasm target '{target_key}' export '{export_name}' config "
        "'{config_key}' rejected '{op_name}' field '{field_name}' in "
        "'@{function_name}': type {actual_type} is not {required_type}"
    ),
    params=(
        *_TARGET_CONTEXT_PARAMS,
        ErrorParam("field_name", ParamKind.STRING),
        ErrorParam("actual_type", ParamKind.TYPE),
        ErrorParam("required_type", ParamKind.STRING),
    ),
)

# ERR_WASM_002: Wasm constant attribute kind is unsupported.
ERR_WASM_002 = ErrorDef(
    domain=ErrorDomain.WASM,
    code=2,
    severity=Severity.ERROR,
    summary="Wasm constant attribute kind is unsupported.",
    message=(
        "Wasm target '{target_key}' export '{export_name}' config "
        "'{config_key}' rejected '{op_name}' attribute '{field_name}' in "
        "'@{function_name}': expected an i64 constant attribute"
    ),
    params=(
        *_TARGET_CONTEXT_PARAMS,
        ErrorParam("field_name", ParamKind.STRING),
    ),
)

# ERR_WASM_003: Wasm constant value is outside the encodable range.
ERR_WASM_003 = ErrorDef(
    domain=ErrorDomain.WASM,
    code=3,
    severity=Severity.ERROR,
    summary="Wasm constant value is outside the encodable range.",
    message=(
        "Wasm target '{target_key}' export '{export_name}' config "
        "'{config_key}' rejected '{op_name}' attribute '{field_name}' in "
        "'@{function_name}': expected range [{minimum}, {maximum}]"
    ),
    params=(
        *_TARGET_CONTEXT_PARAMS,
        ErrorParam("field_name", ParamKind.STRING),
        ErrorParam("minimum", ParamKind.I64),
        ErrorParam("maximum", ParamKind.I64),
    ),
)

# ERR_WASM_005: Wasm source memory access is unsupported.
ERR_WASM_005 = ErrorDef(
    domain=ErrorDomain.WASM,
    code=5,
    severity=Severity.ERROR,
    summary="Wasm source memory access is unsupported.",
    message=(
        "Wasm target '{target_key}' export '{export_name}' config "
        "'{config_key}' rejected '{op_name}' source memory access in "
        "'@{function_name}': expected a scalar or contiguous SIMD "
        "linear-memory access with a 32-bit byte offset"
    ),
    params=_TARGET_CONTEXT_PARAMS,
)

# ERR_WASM_006: Wasm source memory byte offset is outside the target address width.
ERR_WASM_006 = ErrorDef(
    domain=ErrorDomain.WASM,
    code=6,
    severity=Severity.ERROR,
    summary="Wasm source memory byte offset is outside the target address width.",
    message=(
        "Wasm target '{target_key}' export '{export_name}' config "
        "'{config_key}' rejected '{op_name}' source memory access in "
        "'@{function_name}': effective byte displacement must fit a signed "
        "or unsigned {bit_count}-bit register"
    ),
    params=(
        *_TARGET_CONTEXT_PARAMS,
        ErrorParam("bit_count", ParamKind.I64),
    ),
)

# ERR_WASM_007: Wasm dynamic vector window exceeds the native lowering limit.
ERR_WASM_007 = ErrorDef(
    domain=ErrorDomain.WASM,
    code=7,
    severity=Severity.ERROR,
    summary="Wasm dynamic vector window exceeds the native lowering limit.",
    message=(
        "Wasm target '{target_key}' export '{export_name}' config "
        "'{config_key}' rejected '{op_name}' in '@{function_name}': "
        "materializing the runtime-selected packet window requires "
        "{required_operation_count} Low operations, exceeding the "
        "per-operation limit of {maximum_operation_count}"
    ),
    params=(
        *_TARGET_CONTEXT_PARAMS,
        ErrorParam("required_operation_count", ParamKind.U32),
        ErrorParam("maximum_operation_count", ParamKind.U32),
    ),
)

ALL_WASM_ERRORS = (
    ERR_WASM_001,
    ERR_WASM_002,
    ERR_WASM_003,
    ERR_WASM_005,
    ERR_WASM_006,
    ERR_WASM_007,
)
