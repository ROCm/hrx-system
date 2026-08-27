# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""SPIRV domain — SPIR-V-owned legality and lowering diagnostics."""

from loom.errors import ErrorDef, ErrorDomain, ErrorParam, ParamKind, Severity

_TARGET_CONTEXT_PARAMS = (
    ErrorParam("target_key", ParamKind.STRING),
    ErrorParam("export_name", ParamKind.STRING),
    ErrorParam("config_key", ParamKind.STRING),
    ErrorParam("function_name", ParamKind.STRING),
    ErrorParam("op_name", ParamKind.STRING),
)

# ERR_SPIRV_005: SPIR-V Low register type has no exact value representation.
ERR_SPIRV_005 = ErrorDef(
    domain=ErrorDomain.SPIRV,
    code=5,
    severity=Severity.ERROR,
    summary="Unsupported SPIR-V Low register type.",
    message=(
        "SPIR-V Low value '{value_name}' in '@{function_name}' has register "
        "type {actual_type}, which has no exact SPIR-V value representation"
    ),
    params=(
        ErrorParam("function_name", ParamKind.STRING),
        ErrorParam("value_name", ParamKind.STRING),
        ErrorParam("actual_type", ParamKind.TYPE),
    ),
    fix_hint="Attach the exact public value type to the SPIR-V Low register",
)

# ERR_SPIRV_006: SPIR-V raw-BDA HAL kernels cannot return values.
ERR_SPIRV_006 = ErrorDef(
    domain=ErrorDomain.SPIRV,
    code=6,
    severity=Severity.ERROR,
    summary="SPIR-V raw-BDA HAL kernel returns values.",
    message=(
        "SPIR-V raw-BDA HAL kernel '@{function_name}' cannot return values; "
        "writes must go through HAL binding resources"
    ),
    params=(ErrorParam("function_name", ParamKind.STRING),),
)

# ERR_SPIRV_007: SPIR-V raw-BDA HAL direct ABI value is unsupported.
ERR_SPIRV_007 = ErrorDef(
    domain=ErrorDomain.SPIRV,
    code=7,
    severity=Severity.ERROR,
    summary="Unsupported SPIR-V raw-BDA HAL direct ABI value.",
    message=(
        "SPIR-V raw-BDA HAL kernel '@{function_name}' direct value "
        "'{value_name}' has SPIR-V value type '{actual_value_type}', but only "
        "scalar, bool, and offset64 direct values are supported"
    ),
    params=(
        ErrorParam("function_name", ParamKind.STRING),
        ErrorParam("value_name", ParamKind.STRING),
        ErrorParam("actual_value_type", ParamKind.STRING),
    ),
)

# ERR_SPIRV_008: SPIR-V raw-BDA resource import is unsupported.
ERR_SPIRV_008 = ErrorDef(
    domain=ErrorDomain.SPIRV,
    code=8,
    severity=Severity.ERROR,
    summary="Unsupported SPIR-V raw-BDA resource import.",
    message=(
        "SPIR-V raw-BDA HAL kernel '@{function_name}' resource must import a "
        "HAL binding; import kind {import_kind} is not supported"
    ),
    params=(
        ErrorParam("function_name", ParamKind.STRING),
        ErrorParam("import_kind", ParamKind.U32),
    ),
)

# ERR_SPIRV_009: SPIR-V raw-BDA binding index is outside the encodable range.
ERR_SPIRV_009 = ErrorDef(
    domain=ErrorDomain.SPIRV,
    code=9,
    severity=Severity.ERROR,
    summary="SPIR-V raw-BDA binding index out of range.",
    message=(
        "SPIR-V raw-BDA HAL kernel '@{function_name}' binding index "
        "{binding_index} is outside the 16-bit binding-count range"
    ),
    params=(
        ErrorParam("function_name", ParamKind.STRING),
        ErrorParam("binding_index", ParamKind.I64),
    ),
)

# ERR_SPIRV_010: SPIR-V raw-BDA resource source type is unsupported.
ERR_SPIRV_010 = ErrorDef(
    domain=ErrorDomain.SPIRV,
    code=10,
    severity=Severity.ERROR,
    summary="Unsupported SPIR-V raw-BDA resource source type.",
    message=(
        "SPIR-V raw-BDA HAL kernel '@{function_name}' resource source type "
        "{actual_type} is not hal.buffer"
    ),
    params=(
        ErrorParam("function_name", ParamKind.STRING),
        ErrorParam("actual_type", ParamKind.TYPE),
    ),
)

# ERR_SPIRV_011: SPIR-V raw-BDA resource result type is unsupported.
ERR_SPIRV_011 = ErrorDef(
    domain=ErrorDomain.SPIRV,
    code=11,
    severity=Severity.ERROR,
    summary="Unsupported SPIR-V raw-BDA resource result type.",
    message=(
        "SPIR-V raw-BDA HAL kernel '@{function_name}' resource result "
        "'{value_name}' has SPIR-V value type '{actual_value_type}', but "
        "resources materialize as storage-buffer addresses"
    ),
    params=(
        ErrorParam("function_name", ParamKind.STRING),
        ErrorParam("value_name", ParamKind.STRING),
        ErrorParam("actual_value_type", ParamKind.STRING),
    ),
)

# ERR_SPIRV_012: SPIR-V descriptor has no binary packet row.
ERR_SPIRV_012 = ErrorDef(
    domain=ErrorDomain.SPIRV,
    code=12,
    severity=Severity.ERROR,
    summary="SPIR-V descriptor has no binary packet row.",
    message=(
        "SPIR-V descriptor '{descriptor_key}' in '@{function_name}' has no "
        "binary packet row for the selected target"
    ),
    params=(
        ErrorParam("function_name", ParamKind.STRING),
        ErrorParam("descriptor_key", ParamKind.STRING),
    ),
)

# ERR_SPIRV_013: SPIR-V packet exact value type mismatch.
ERR_SPIRV_013 = ErrorDef(
    domain=ErrorDomain.SPIRV,
    code=13,
    severity=Severity.ERROR,
    summary="SPIR-V packet value type mismatch.",
    message=(
        "SPIR-V descriptor '{descriptor_key}' in '@{function_name}' expects "
        "{field_kind} {field_index} to have value type '{expected_value_type}', "
        "but '{value_name}' has '{actual_value_type}'"
    ),
    params=(
        ErrorParam("function_name", ParamKind.STRING),
        ErrorParam("descriptor_key", ParamKind.STRING),
        ErrorParam("field_kind", ParamKind.STRING),
        ErrorParam("field_index", ParamKind.U32),
        ErrorParam("value_name", ParamKind.STRING),
        ErrorParam("expected_value_type", ParamKind.STRING),
        ErrorParam("actual_value_type", ParamKind.STRING),
    ),
)

# ERR_SPIRV_014: SPIR-V low return exact value type mismatch.
ERR_SPIRV_014 = ErrorDef(
    domain=ErrorDomain.SPIRV,
    code=14,
    severity=Severity.ERROR,
    summary="SPIR-V return value type mismatch.",
    message=(
        "SPIR-V return in '@{function_name}' expects result {result_index} "
        "to have value type '{expected_value_type}', but '{value_name}' has "
        "'{actual_value_type}'"
    ),
    params=(
        ErrorParam("function_name", ParamKind.STRING),
        ErrorParam("result_index", ParamKind.U32),
        ErrorParam("value_name", ParamKind.STRING),
        ErrorParam("expected_value_type", ParamKind.STRING),
        ErrorParam("actual_value_type", ParamKind.STRING),
    ),
)

# ERR_SPIRV_015: SPIR-V requires structured low control flow.
ERR_SPIRV_015 = ErrorDef(
    domain=ErrorDomain.SPIRV,
    code=15,
    severity=Severity.ERROR,
    summary="SPIR-V requires structured low control flow.",
    message=(
        "SPIR-V emission for '@{function_name}' requires low.scf control "
        "flow; '{op_name}' is CFG input"
    ),
    params=(
        ErrorParam("function_name", ParamKind.STRING),
        ErrorParam("op_name", ParamKind.STRING),
    ),
)

# ERR_SPIRV_016: SPIR-V shader-entry result value is unsupported.
ERR_SPIRV_016 = ErrorDef(
    domain=ErrorDomain.SPIRV,
    code=16,
    severity=Severity.ERROR,
    summary="Unsupported SPIR-V shader-entry result value.",
    message=(
        "SPIR-V shader-entry function '@{function_name}' result "
        "'{value_name}' has SPIR-V value type '{actual_value_type}', but only "
        "scalar, bool, and offset64 result values are supported"
    ),
    params=(
        ErrorParam("function_name", ParamKind.STRING),
        ErrorParam("value_name", ParamKind.STRING),
        ErrorParam("actual_value_type", ParamKind.STRING),
    ),
)

# ERR_SPIRV_017: SPIR-V low structural op is unsupported by binary emission.
ERR_SPIRV_017 = ErrorDef(
    domain=ErrorDomain.SPIRV,
    code=17,
    severity=Severity.ERROR,
    summary="Unsupported SPIR-V structural low op.",
    message=(
        "SPIR-V binary emission for '@{function_name}' does not support "
        "structural low op '{op_name}'"
    ),
    params=(
        ErrorParam("function_name", ParamKind.STRING),
        ErrorParam("op_name", ParamKind.STRING),
    ),
)

# ERR_SPIRV_018: SPIR-V low.resource requires the raw-BDA HAL ABI.
ERR_SPIRV_018 = ErrorDef(
    domain=ErrorDomain.SPIRV,
    code=18,
    severity=Severity.ERROR,
    summary="SPIR-V resource requires raw-BDA HAL ABI.",
    message=(
        "SPIR-V function '@{function_name}' uses low.resource, but resource "
        "materialization is only supported for raw-BDA HAL kernels"
    ),
    params=(ErrorParam("function_name", ParamKind.STRING),),
)

# ERR_SPIRV_019: SPIR-V low branch condition exact value type mismatch.
ERR_SPIRV_019 = ErrorDef(
    domain=ErrorDomain.SPIRV,
    code=19,
    severity=Severity.ERROR,
    summary="SPIR-V branch condition value type mismatch.",
    message=(
        "SPIR-V conditional branch in '@{function_name}' expects condition "
        "'{value_name}' to have value type 'bool', but it has "
        "'{actual_value_type}'"
    ),
    params=(
        ErrorParam("function_name", ParamKind.STRING),
        ErrorParam("value_name", ParamKind.STRING),
        ErrorParam("actual_value_type", ParamKind.STRING),
    ),
)

# ERR_SPIRV_020: SPIR-V low branch payload exact value type mismatch.
ERR_SPIRV_020 = ErrorDef(
    domain=ErrorDomain.SPIRV,
    code=20,
    severity=Severity.ERROR,
    summary="SPIR-V branch payload value type mismatch.",
    message=(
        "SPIR-V branch in '@{function_name}' forwards '{source_value_name}' "
        "to block argument '{target_value_name}', expected value type "
        "'{expected_value_type}' but got '{actual_value_type}'"
    ),
    params=(
        ErrorParam("function_name", ParamKind.STRING),
        ErrorParam("source_value_name", ParamKind.STRING),
        ErrorParam("target_value_name", ParamKind.STRING),
        ErrorParam("expected_value_type", ParamKind.STRING),
        ErrorParam("actual_value_type", ParamKind.STRING),
    ),
)

# ERR_SPIRV_022: SPIR-V shader-entry argument value is unsupported.
ERR_SPIRV_022 = ErrorDef(
    domain=ErrorDomain.SPIRV,
    code=22,
    severity=Severity.ERROR,
    summary="Unsupported SPIR-V shader-entry argument value.",
    message=(
        "SPIR-V shader-entry function '@{function_name}' argument "
        "'{value_name}' has SPIR-V value type '{actual_value_type}', but only "
        "scalar, bool, offset64, and storage-buffer-address argument values "
        "are supported"
    ),
    params=(
        ErrorParam("function_name", ParamKind.STRING),
        ErrorParam("value_name", ParamKind.STRING),
        ErrorParam("actual_value_type", ParamKind.STRING),
    ),
)

# ERR_SPIRV_023: SPIR-V raw-BDA direct constants exceed the ABI range.
ERR_SPIRV_023 = ErrorDef(
    domain=ErrorDomain.SPIRV,
    code=23,
    severity=Severity.ERROR,
    summary="SPIR-V raw-BDA direct constants out of range.",
    message=(
        "SPIR-V raw-BDA HAL kernel '@{function_name}' direct values require "
        "{constant_word_count} 32-bit constant word(s), which exceeds the "
        "16-bit ABI range"
    ),
    params=(
        ErrorParam("function_name", ParamKind.STRING),
        ErrorParam("constant_word_count", ParamKind.U32),
    ),
)

# ERR_SPIRV_024: SPIR-V target-low function body is not structured.
ERR_SPIRV_024 = ErrorDef(
    domain=ErrorDomain.SPIRV,
    code=24,
    severity=Severity.ERROR,
    summary="SPIR-V target-low function body is not structured.",
    message=(
        "SPIR-V emission for '@{function_name}' requires exactly one "
        "top-level low function block, but the body has {block_count}"
    ),
    params=(
        ErrorParam("function_name", ParamKind.STRING),
        ErrorParam("block_count", ParamKind.U32),
    ),
)

# ERR_SPIRV_025: SPIR-V target-low module mixes target contracts.
ERR_SPIRV_025 = ErrorDef(
    domain=ErrorDomain.SPIRV,
    code=25,
    severity=Severity.ERROR,
    summary="SPIR-V target-low module mixes target contracts.",
    message=(
        "SPIR-V module emission for '@{function_name}' selects target "
        "'@{target_name}', but the module already selected incompatible "
        "target '@{first_target_name}'"
    ),
    params=(
        ErrorParam("function_name", ParamKind.STRING),
        ErrorParam("target_name", ParamKind.STRING),
        ErrorParam("first_target_name", ParamKind.STRING),
    ),
)

# ERR_SPIRV_026: SPIR-V address conversion range is not proven.
ERR_SPIRV_026 = ErrorDef(
    domain=ErrorDomain.SPIRV,
    code=26,
    severity=Severity.ERROR,
    summary="SPIR-V address conversion range is not proven.",
    message=(
        "SPIR-V target '{target_key}' export '{export_name}' config "
        "'{config_key}' rejected '{op_name}' in '@{function_name}': "
        "conversion from {source_type} to {result_type} requires the source "
        "value to be proven in [{required_range_lo}, {required_range_hi}]; "
        "constraint '{constraint_key}' is not satisfied"
    ),
    params=(
        *_TARGET_CONTEXT_PARAMS,
        ErrorParam("source_type", ParamKind.TYPE),
        ErrorParam("result_type", ParamKind.TYPE),
        ErrorParam("required_range_lo", ParamKind.I64),
        ErrorParam("required_range_hi", ParamKind.I64),
        ErrorParam("constraint_key", ParamKind.STRING),
    ),
    fix_hint=(
        "Establish the value range at its producer or with an assumption "
        "after the corresponding runtime guard"
    ),
)

# ERR_SPIRV_027: SPIR-V index numeric operand range is not proven.
ERR_SPIRV_027 = ErrorDef(
    domain=ErrorDomain.SPIRV,
    code=27,
    severity=Severity.ERROR,
    summary="SPIR-V index numeric operand range is not proven.",
    message=(
        "SPIR-V target '{target_key}' export '{export_name}' config "
        "'{config_key}' rejected '{op_name}' field '{field_name}' in "
        "'@{function_name}': {value_type} value must be proven in "
        "[{required_range_lo}, {required_range_hi}]; constraint "
        "'{constraint_key}' is not satisfied"
    ),
    params=(
        *_TARGET_CONTEXT_PARAMS,
        ErrorParam("field_name", ParamKind.STRING),
        ErrorParam("value_type", ParamKind.TYPE),
        ErrorParam("required_range_lo", ParamKind.I64),
        ErrorParam("required_range_hi", ParamKind.I64),
        ErrorParam("constraint_key", ParamKind.STRING),
    ),
    fix_hint=(
        "Establish the operand range at its producer or with an assumption "
        "after the corresponding runtime guard"
    ),
)

# ERR_SPIRV_028: SPIR-V source-memory address index range is not proven.
ERR_SPIRV_028 = ErrorDef(
    domain=ErrorDomain.SPIRV,
    code=28,
    severity=Severity.ERROR,
    summary="SPIR-V source-memory address index range is not proven.",
    message=(
        "SPIR-V target '{target_key}' export '{export_name}' config "
        "'{config_key}' rejected '{op_name}' in '@{function_name}': every "
        "index contributing to the source-memory address must be proven in "
        "[{required_range_lo}, {required_range_hi}]; constraint "
        "'{constraint_key}' is not satisfied"
    ),
    params=(
        *_TARGET_CONTEXT_PARAMS,
        ErrorParam("required_range_lo", ParamKind.I64),
        ErrorParam("required_range_hi", ParamKind.I64),
        ErrorParam("constraint_key", ParamKind.STRING),
    ),
    fix_hint=(
        "Establish the index range at its producer or with an assumption "
        "after the corresponding runtime guard"
    ),
)

# ERR_SPIRV_029: SPIR-V Workgroup element address is not proven.
ERR_SPIRV_029 = ErrorDef(
    domain=ErrorDomain.SPIRV,
    code=29,
    severity=Severity.ERROR,
    summary="SPIR-V Workgroup element address is not proven.",
    message=(
        "SPIR-V target '{target_key}' export '{export_name}' config "
        "'{config_key}' rejected '{op_name}' in '@{function_name}': the "
        "complete root-relative byte address must be exactly divisible by "
        "the {element_byte_count}-byte element size and its element index "
        "must be proven in [{required_range_lo}, {required_range_hi}]; "
        "constraint '{constraint_key}' is not satisfied"
    ),
    params=(
        *_TARGET_CONTEXT_PARAMS,
        ErrorParam("element_byte_count", ParamKind.U32),
        ErrorParam("required_range_lo", ParamKind.I64),
        ErrorParam("required_range_hi", ParamKind.I64),
        ErrorParam("constraint_key", ParamKind.STRING),
    ),
    fix_hint=(
        "Establish aligned, non-negative element addressing at the view or "
        "index producer and prove the complete address range"
    ),
)

# ERR_SPIRV_030: SPIR-V low boundary carries target-specific ABI layout state.
ERR_SPIRV_030 = ErrorDef(
    domain=ErrorDomain.SPIRV,
    code=30,
    severity=Severity.ERROR,
    summary="SPIR-V low boundary has target-specific ABI layout metadata.",
    message=(
        "SPIR-V low function '@{function_name}' cannot carry target-specific "
        "abi_layout metadata; boundary value meaning must be encoded in "
        "structural register types"
    ),
    params=(ErrorParam("function_name", ParamKind.STRING),),
    fix_hint="Attach the exact public value type to each SPIR-V Low register",
)

ALL_SPIRV_ERRORS: tuple[ErrorDef, ...] = (
    ERR_SPIRV_005,
    ERR_SPIRV_006,
    ERR_SPIRV_007,
    ERR_SPIRV_008,
    ERR_SPIRV_009,
    ERR_SPIRV_010,
    ERR_SPIRV_011,
    ERR_SPIRV_012,
    ERR_SPIRV_013,
    ERR_SPIRV_014,
    ERR_SPIRV_015,
    ERR_SPIRV_016,
    ERR_SPIRV_017,
    ERR_SPIRV_018,
    ERR_SPIRV_019,
    ERR_SPIRV_020,
    ERR_SPIRV_022,
    ERR_SPIRV_023,
    ERR_SPIRV_024,
    ERR_SPIRV_025,
    ERR_SPIRV_026,
    ERR_SPIRV_027,
    ERR_SPIRV_028,
    ERR_SPIRV_029,
    ERR_SPIRV_030,
)
