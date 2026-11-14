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

# ERR_SPIRV_028: SPIR-V source-memory address cannot be represented.
ERR_SPIRV_028 = ErrorDef(
    domain=ErrorDomain.SPIRV,
    code=28,
    severity=Severity.ERROR,
    summary="SPIR-V source-memory address cannot be represented.",
    message=(
        "SPIR-V target '{target_key}' export '{export_name}' config "
        "'{config_key}' rejected '{op_name}' in '@{function_name}': index "
        "inputs must be proven in [{required_range_lo}, {required_range_hi}] "
        "and integer terms must have supported numeric carrier conversions; constraint "
        "'{constraint_key}' is not satisfied"
    ),
    params=(
        *_TARGET_CONTEXT_PARAMS,
        ErrorParam("required_range_lo", ParamKind.I64),
        ErrorParam("required_range_hi", ParamKind.I64),
        ErrorParam("constraint_key", ParamKind.STRING),
    ),
    fix_hint=(
        "Establish bounds on target index inputs and enable the integer "
        "capabilities required by the address terms"
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

# ERR_SPIRV_030: SPIR-V low boundary has an invalid ABI layout.
ERR_SPIRV_030 = ErrorDef(
    domain=ErrorDomain.SPIRV,
    code=30,
    severity=Severity.ERROR,
    summary="SPIR-V low boundary has invalid ABI layout metadata.",
    message=(
        "SPIR-V low function '@{function_name}' may carry only a resultless "
        "logical 'signature' for a HAL kernel, and that signature must match "
        "its structural direct arguments and retained binding resources"
    ),
    params=(ErrorParam("function_name", ParamKind.STRING),),
    fix_hint=(
        "Remove target-specific ABI layout state or regenerate the logical "
        "HAL signature and binding ordinals from the source kernel boundary"
    ),
)

# ERR_SPIRV_031: SPIR-V raw-BDA reflected native offset is out of range.
ERR_SPIRV_031 = ErrorDef(
    domain=ErrorDomain.SPIRV,
    code=31,
    severity=Severity.ERROR,
    summary="SPIR-V raw-BDA reflected parameter offset out of range.",
    message=(
        "SPIR-V raw-BDA HAL kernel '@{function_name}' direct value "
        "'{value_name}' begins at constant-table byte offset "
        "{constant_byte_offset}, producing native ABI byte offset "
        "{native_byte_offset}, which exceeds the 16-bit HAL parameter range"
    ),
    params=(
        ErrorParam("function_name", ParamKind.STRING),
        ErrorParam("value_name", ParamKind.STRING),
        ErrorParam("constant_byte_offset", ParamKind.U32),
        ErrorParam("native_byte_offset", ParamKind.U32),
    ),
)

# ERR_SPIRV_032: SPIR-V internal function recursion is unsupported.
ERR_SPIRV_032 = ErrorDef(
    domain=ErrorDomain.SPIRV,
    code=32,
    severity=Severity.ERROR,
    summary="Recursive SPIR-V internal function call.",
    message=(
        "SPIR-V internal function '@{function_name}' participates in a "
        "recursive call cycle, which Vulkan SPIR-V does not permit"
    ),
    params=(ErrorParam("function_name", ParamKind.STRING),),
    fix_hint="Rewrite the recursive helper as structured iteration",
)

# ERR_SPIRV_033: An opaque matrix fragment crosses a retained source callable.
ERR_SPIRV_033 = ErrorDef(
    domain=ErrorDomain.SPIRV,
    code=33,
    severity=Severity.ERROR,
    summary="Opaque matrix fragment crosses a retained SPIR-V call boundary.",
    message=(
        "'{op_name}' crosses the retained boundary of '@{callee_name}' with "
        "an opaque cooperative-matrix fragment spelled as {source_type}; "
        "the source vector signature does not describe its SPIR-V matrix role "
        "and shape"
    ),
    params=(
        ErrorParam("op_name", ParamKind.STRING),
        ErrorParam("callee_name", ParamKind.STRING),
        ErrorParam("source_type", ParamKind.TYPE),
    ),
    fix_hint=(
        "Inline the fragment helper (remove conflicting noinline), or keep "
        "fragment construction, computation, and storage inside the helper"
    ),
)

# ERR_SPIRV_034: A target profile omits a required feature fact.
ERR_SPIRV_034 = ErrorDef(
    domain=ErrorDomain.SPIRV,
    code=34,
    severity=Severity.ERROR,
    summary="SPIR-V target profile is missing a feature dependency.",
    message=(
        "SPIR-V target profile '{profile_name}' feature '{feature_name}' "
        "requires feature '{required_feature_name}'"
    ),
    params=(
        ErrorParam("profile_name", ParamKind.STRING),
        ErrorParam("feature_name", ParamKind.STRING),
        ErrorParam("required_feature_name", ParamKind.STRING),
    ),
)

ERR_SPIRV_035 = ErrorDef(
    domain=ErrorDomain.SPIRV,
    code=35,
    severity=Severity.ERROR,
    summary="SPIR-V profile observations contradict each other.",
    message=(
        "SPIR-V {fact_kind} '{fact_name}' has contradictory facts: "
        "'{first_provenance}' says {first_state} while '{second_provenance}' says"
        " {second_state}"
    ),
    params=(
        ErrorParam("fact_kind", ParamKind.STRING),
        ErrorParam("fact_name", ParamKind.STRING),
        ErrorParam("first_provenance", ParamKind.STRING),
        ErrorParam("first_state", ParamKind.BOOL),
        ErrorParam("second_provenance", ParamKind.STRING),
        ErrorParam("second_state", ParamKind.BOOL),
    ),
)

ERR_SPIRV_036 = ErrorDef(
    domain=ErrorDomain.SPIRV,
    code=36,
    severity=Severity.ERROR,
    summary="SPIR-V numeric profile observation is out of range.",
    message=(
        "SPIR-V {fact_kind} '{fact_name}' value {value} from '{provenance}' is "
        "outside the supported range [1, {maximum}]"
    ),
    params=(
        ErrorParam("fact_kind", ParamKind.STRING),
        ErrorParam("fact_name", ParamKind.STRING),
        ErrorParam("value", ParamKind.U64),
        ErrorParam("provenance", ParamKind.STRING),
        ErrorParam("maximum", ParamKind.U64),
    ),
)

ERR_SPIRV_037 = ErrorDef(
    domain=ErrorDomain.SPIRV,
    code=37,
    severity=Severity.ERROR,
    summary="SPIR-V numeric profile observations contradict each other.",
    message=(
        "SPIR-V {fact_kind} '{fact_name}' has contradictory values: "
        "'{first_provenance}' says {first_value} while '{second_provenance}' says"
        " {second_value}"
    ),
    params=(
        ErrorParam("fact_kind", ParamKind.STRING),
        ErrorParam("fact_name", ParamKind.STRING),
        ErrorParam("first_provenance", ParamKind.STRING),
        ErrorParam("first_value", ParamKind.U64),
        ErrorParam("second_provenance", ParamKind.STRING),
        ErrorParam("second_value", ParamKind.U64),
    ),
)

ERR_SPIRV_038 = ErrorDef(
    domain=ErrorDomain.SPIRV,
    code=38,
    severity=Severity.ERROR,
    summary="SPIR-V profile exceeds its observed environment version.",
    message=(
        "SPIR-V target profile requires SPIR-V version "
        "{required_major}.{required_minor}, but environment '{fact_name}' from "
        "'{provenance}' allows only {maximum_major}.{maximum_minor}"
    ),
    params=(
        ErrorParam("required_major", ParamKind.U32),
        ErrorParam("required_minor", ParamKind.U32),
        ErrorParam("fact_name", ParamKind.STRING),
        ErrorParam("provenance", ParamKind.STRING),
        ErrorParam("maximum_major", ParamKind.U32),
        ErrorParam("maximum_minor", ParamKind.U32),
    ),
)

ERR_SPIRV_039 = ErrorDef(
    domain=ErrorDomain.SPIRV,
    code=39,
    severity=Severity.ERROR,
    summary="SPIR-V subgroup width is not a power of two.",
    message=(
        "SPIR-V subgroup width '{fact_name}' value {value} from '{provenance}' "
        "must be a power of two"
    ),
    params=(
        ErrorParam("fact_name", ParamKind.STRING),
        ErrorParam("value", ParamKind.U32),
        ErrorParam("provenance", ParamKind.STRING),
    ),
)

ERR_SPIRV_040 = ErrorDef(
    domain=ErrorDomain.SPIRV,
    code=40,
    severity=Severity.ERROR,
    summary="SPIR-V subgroup widths contradict the observed range.",
    message=(
        "SPIR-V subgroup widths contradict the supported range: fixed={fixed}, "
        "default={default}, minimum={minimum}, maximum={maximum}"
    ),
    params=(
        ErrorParam("fixed", ParamKind.U32),
        ErrorParam("default", ParamKind.U32),
        ErrorParam("minimum", ParamKind.U32),
        ErrorParam("maximum", ParamKind.U32),
    ),
)

ERR_SPIRV_041 = ErrorDef(
    domain=ErrorDomain.SPIRV,
    code=41,
    severity=Severity.ERROR,
    summary="SPIR-V cooperative row observations contradict each other.",
    message=(
        "SPIR-V profile has contradictory cooperative {row_kind} row facts: "
        "'{first_name}' from '{first_provenance}' says {first_state} while "
        "'{second_name}' from '{second_provenance}' says {second_state}"
    ),
    params=(
        ErrorParam("row_kind", ParamKind.STRING),
        ErrorParam("first_name", ParamKind.STRING),
        ErrorParam("first_provenance", ParamKind.STRING),
        ErrorParam("first_state", ParamKind.BOOL),
        ErrorParam("second_name", ParamKind.STRING),
        ErrorParam("second_provenance", ParamKind.STRING),
        ErrorParam("second_state", ParamKind.BOOL),
    ),
)

ERR_SPIRV_042 = ErrorDef(
    domain=ErrorDomain.SPIRV,
    code=42,
    severity=Severity.ERROR,
    summary="SPIR-V cooperative row names identify different operations.",
    message=(
        "SPIR-V profile has contradictory cooperative {row_kind} row definitions "
        "for name '{name}': '{first_provenance}' and '{second_provenance}' "
        "describe different rows"
    ),
    params=(
        ErrorParam("row_kind", ParamKind.STRING),
        ErrorParam("name", ParamKind.STRING),
        ErrorParam("first_provenance", ParamKind.STRING),
        ErrorParam("second_provenance", ParamKind.STRING),
    ),
)

ERR_SPIRV_043 = ErrorDef(
    domain=ErrorDomain.SPIRV,
    code=43,
    severity=Severity.ERROR,
    summary="Unsupported Vulkan cooperative matrix construction.",
    message=(
        "target '{target_key}' export '{export_name}' config '{config_key}' "
        "cannot construct '{op_name}' from {input_type} in '@{function_name}': "
        "{reason}"
    ),
    params=(
        *_TARGET_CONTEXT_PARAMS,
        ErrorParam("input_type", ParamKind.TYPE),
        ErrorParam("reason", ParamKind.STRING),
    ),
    fix_hint=(
        "Use vector.fragment.load for a logical matrix tile, preserve a compatible "
        "native fragment, or initialize with a subgroup-uniform splat; Vulkan "
        "does not specify the mapping of matrix elements to per-thread registers"
    ),
)

ERR_SPIRV_044 = ErrorDef(
    domain=ErrorDomain.SPIRV,
    code=44,
    severity=Severity.ERROR,
    summary="SPIR-V logical storage requires a disabled scalar feature.",
    message=(
        "SPIR-V storage type {actual_type} in '@{function_name}' requires "
        "feature '{feature_name}', which is not enabled by the target profile"
    ),
    params=(
        ErrorParam("function_name", ParamKind.STRING),
        ErrorParam("actual_type", ParamKind.TYPE),
        ErrorParam("feature_name", ParamKind.STRING),
    ),
    fix_hint="Select a target profile that enables the required scalar feature",
)

ERR_SPIRV_045 = ErrorDef(
    domain=ErrorDomain.SPIRV,
    code=45,
    severity=Severity.ERROR,
    summary="SPIR-V logical storage pointer crosses a control-flow merge.",
    message=(
        "SPIR-V logical storage pointer '{value_name}' in '@{function_name}' "
        "cannot be selected or carried through a control-flow merge"
    ),
    params=(
        ErrorParam("function_name", ParamKind.STRING),
        ErrorParam("value_name", ParamKind.STRING),
    ),
    fix_hint="Carry an integer position and form the storage pointer at each access",
)

ALL_SPIRV_ERRORS: tuple[ErrorDef, ...] = (
    ERR_SPIRV_005,
    ERR_SPIRV_006,
    ERR_SPIRV_007,
    ERR_SPIRV_008,
    ERR_SPIRV_009,
    ERR_SPIRV_010,
    ERR_SPIRV_011,
    ERR_SPIRV_013,
    ERR_SPIRV_014,
    ERR_SPIRV_016,
    ERR_SPIRV_017,
    ERR_SPIRV_018,
    ERR_SPIRV_019,
    ERR_SPIRV_020,
    ERR_SPIRV_022,
    ERR_SPIRV_023,
    ERR_SPIRV_025,
    ERR_SPIRV_026,
    ERR_SPIRV_027,
    ERR_SPIRV_028,
    ERR_SPIRV_029,
    ERR_SPIRV_030,
    ERR_SPIRV_031,
    ERR_SPIRV_032,
    ERR_SPIRV_033,
    ERR_SPIRV_034,
    ERR_SPIRV_035,
    ERR_SPIRV_036,
    ERR_SPIRV_037,
    ERR_SPIRV_038,
    ERR_SPIRV_039,
    ERR_SPIRV_040,
    ERR_SPIRV_041,
    ERR_SPIRV_042,
    ERR_SPIRV_043,
    ERR_SPIRV_044,
    ERR_SPIRV_045,
)
