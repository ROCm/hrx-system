# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""AMDGPU domain — AMDGPU-owned legality and lowering diagnostics."""

from loom.errors import ErrorDef, ErrorDomain, ErrorParam, ParamKind, Severity

_TARGET_CONTEXT_PARAMS = (
    ErrorParam("target_key", ParamKind.STRING),
    ErrorParam("export_name", ParamKind.STRING),
    ErrorParam("config_key", ParamKind.STRING),
    ErrorParam("function_name", ParamKind.STRING),
    ErrorParam("op_name", ParamKind.STRING),
)

# ERR_AMDGPU_001: AMDGPU buffer view element storage is unsupported.
ERR_AMDGPU_001 = ErrorDef(
    domain=ErrorDomain.AMDGPU,
    code=1,
    severity=Severity.ERROR,
    summary="AMDGPU buffer view element storage is unsupported.",
    message=(
        "AMDGPU target '{target_key}' export '{export_name}' config "
        "'{config_key}' rejected '{op_name}' field '{field_name}' in "
        "'@{function_name}': type {actual_type} is not a typed view over "
        "{required_storage}"
    ),
    params=(
        *_TARGET_CONTEXT_PARAMS,
        ErrorParam("field_name", ParamKind.STRING),
        ErrorParam("actual_type", ParamKind.TYPE),
        ErrorParam("required_storage", ParamKind.STRING),
    ),
    fix_hint=(
        "Use a HAL buffer view whose element type has byte-addressable AMDGPU "
        "buffer storage"
    ),
)

# ERR_AMDGPU_003: AMDGPU processor override is unknown.
ERR_AMDGPU_003 = ErrorDef(
    domain=ErrorDomain.AMDGPU,
    code=3,
    severity=Severity.ERROR,
    summary="AMDGPU processor override is unknown.",
    message="AMDGPU processor override '{processor}' is not known",
    params=(ErrorParam("processor", ParamKind.STRING),),
    fix_hint="Use a known AMDGPU processor name such as gfx942, gfx950, or gfx1100",
)

# ERR_AMDGPU_004: AMDGPU processor override has no native descriptor set.
ERR_AMDGPU_004 = ErrorDef(
    domain=ErrorDomain.AMDGPU,
    code=4,
    severity=Severity.ERROR,
    summary="AMDGPU processor override has no native descriptor set.",
    message=(
        "AMDGPU processor override '{processor}' is known but has no "
        "native target-low descriptor set"
    ),
    params=(ErrorParam("processor", ParamKind.STRING),),
    fix_hint="Use an AMDGPU processor with native target-low descriptor coverage",
)

# ERR_AMDGPU_005: AMDGPU processor override changes descriptor set.
ERR_AMDGPU_005 = ErrorDef(
    domain=ErrorDomain.AMDGPU,
    code=5,
    severity=Severity.ERROR,
    summary="AMDGPU processor override changes descriptor set.",
    message=(
        "AMDGPU processor override '{processor}' selects descriptor set "
        "'{target_descriptor_set}' but target record '@{target_name}' uses "
        "descriptor set '{record_descriptor_set}'"
    ),
    params=(
        ErrorParam("processor", ParamKind.STRING),
        ErrorParam("target_descriptor_set", ParamKind.STRING),
        ErrorParam("target_name", ParamKind.STRING),
        ErrorParam("record_descriptor_set", ParamKind.STRING),
    ),
    fix_hint=(
        "Select a target record from the same AMDGPU descriptor-set family as "
        "the requested processor"
    ),
)

# ERR_AMDGPU_006: AMDGPU HAL-kernel ABI resource count overflows.
ERR_AMDGPU_006 = ErrorDef(
    domain=ErrorDomain.AMDGPU,
    code=6,
    severity=Severity.ERROR,
    summary="AMDGPU HAL-kernel ABI resource count overflows.",
    message=(
        "AMDGPU HAL-kernel ABI has {resource_count} HAL binding resources, "
        "but at most {max_resource_count} fit in the kernarg segment"
    ),
    params=(
        ErrorParam("resource_count", ParamKind.U64),
        ErrorParam("max_resource_count", ParamKind.U64),
    ),
    fix_hint="Split the kernel ABI or reduce the number of HAL binding resources",
)

# ERR_AMDGPU_007: AMDGPU HAL-kernel ABI resource import kind is unsupported.
ERR_AMDGPU_007 = ErrorDef(
    domain=ErrorDomain.AMDGPU,
    code=7,
    severity=Severity.ERROR,
    summary="AMDGPU HAL-kernel ABI resource import kind is unsupported.",
    message=(
        "AMDGPU HAL-kernel ABI requires low.resource import_kind enum value "
        "{expected_import_kind}, but found {actual_import_kind}"
    ),
    params=(
        ErrorParam("expected_import_kind", ParamKind.U32),
        ErrorParam("actual_import_kind", ParamKind.U32),
    ),
    fix_hint="Use low.resource<hal_binding> for AMDGPU HAL kernel resources",
)

# ERR_AMDGPU_008: AMDGPU HAL-kernel ABI resource result type is unsupported.
ERR_AMDGPU_008 = ErrorDef(
    domain=ErrorDomain.AMDGPU,
    code=8,
    severity=Severity.ERROR,
    summary="AMDGPU HAL-kernel ABI resource result type is unsupported.",
    message=(
        "AMDGPU HAL-kernel ABI requires hal_binding resources to produce "
        "descriptor register-class ID {expected_reg_class_id} with "
        "{expected_unit_count} unit(s), but found {actual_type}"
    ),
    params=(
        ErrorParam("actual_type", ParamKind.TYPE),
        ErrorParam("expected_reg_class_id", ParamKind.U32),
        ErrorParam("expected_unit_count", ParamKind.U32),
    ),
    fix_hint="Use a 64-bit pointer resource value for HAL binding imports",
)

# ERR_AMDGPU_009: AMDGPU HAL-kernel ABI binding index is negative.
ERR_AMDGPU_009 = ErrorDef(
    domain=ErrorDomain.AMDGPU,
    code=9,
    severity=Severity.ERROR,
    summary="AMDGPU HAL-kernel ABI binding index is negative.",
    message=(
        "AMDGPU HAL-kernel ABI requires non-negative binding indexes, but "
        "found {binding_index}"
    ),
    params=(ErrorParam("binding_index", ParamKind.I64),),
    fix_hint="Number HAL binding resources densely from zero",
)

# ERR_AMDGPU_010: AMDGPU HAL-kernel ABI binding index is not dense.
ERR_AMDGPU_010 = ErrorDef(
    domain=ErrorDomain.AMDGPU,
    code=10,
    severity=Severity.ERROR,
    summary="AMDGPU HAL-kernel ABI binding index is not dense.",
    message=(
        "AMDGPU HAL-kernel ABI found binding index {binding_index}, but "
        "{resource_count} resource(s) require indexes in [0, {resource_count})"
    ),
    params=(
        ErrorParam("binding_index", ParamKind.U64),
        ErrorParam("resource_count", ParamKind.U64),
    ),
    fix_hint="Number HAL binding resources densely from zero without gaps",
)

# ERR_AMDGPU_011: AMDGPU HAL-kernel ABI binding index is duplicated.
ERR_AMDGPU_011 = ErrorDef(
    domain=ErrorDomain.AMDGPU,
    code=11,
    severity=Severity.ERROR,
    summary="AMDGPU HAL-kernel ABI binding index is duplicated.",
    message=(
        "AMDGPU HAL-kernel ABI binding index {binding_index} is defined more than once"
    ),
    params=(ErrorParam("binding_index", ParamKind.U64),),
    fix_hint="Give each HAL binding resource a unique dense index",
)

# ERR_AMDGPU_012: AMDGPU HAL-kernel ABI binding index is missing.
ERR_AMDGPU_012 = ErrorDef(
    domain=ErrorDomain.AMDGPU,
    code=12,
    severity=Severity.ERROR,
    summary="AMDGPU HAL-kernel ABI binding index is missing.",
    message=(
        "AMDGPU HAL-kernel ABI requires binding index {binding_index}, but "
        "no low.resource defines it"
    ),
    params=(ErrorParam("binding_index", ParamKind.U64),),
    fix_hint="Define HAL binding resources densely from zero without gaps",
)

# ERR_AMDGPU_013: AMDGPU HAL buffer descriptor pseudo attribute is invalid.
ERR_AMDGPU_013 = ErrorDef(
    domain=ErrorDomain.AMDGPU,
    code=13,
    severity=Severity.ERROR,
    summary="AMDGPU HAL buffer descriptor pseudo attribute is invalid.",
    message=(
        "AMDGPU HAL buffer descriptor pseudo requires attribute "
        "'{attr_name}' with kind {expected_kind}, but found kind {actual_kind}"
    ),
    params=(
        ErrorParam("attr_name", ParamKind.STRING),
        ErrorParam("expected_kind", ParamKind.U32),
        ErrorParam("actual_kind", ParamKind.U32),
    ),
    fix_hint="Run low descriptor verification before AMDGPU HAL ABI verification",
)

# ERR_AMDGPU_014: AMDGPU HAL buffer descriptor cache swizzle is unsupported.
ERR_AMDGPU_014 = ErrorDef(
    domain=ErrorDomain.AMDGPU,
    code=14,
    severity=Severity.ERROR,
    summary="AMDGPU HAL buffer descriptor cache swizzle is unsupported.",
    message=(
        "AMDGPU descriptor set '{descriptor_set_key}' cannot encode HAL "
        "buffer descriptor cache_swizzle_stride {cache_swizzle_stride}"
    ),
    params=(
        ErrorParam("descriptor_set_key", ParamKind.STRING),
        ErrorParam("cache_swizzle_stride", ParamKind.U64),
    ),
    fix_hint=(
        "Use zero cache_swizzle_stride or select an AMDGPU descriptor set "
        "that supports stride14 cache swizzle"
    ),
)

# ERR_AMDGPU_015: AMDGPU HAL-kernel ABI live-in is duplicated.
ERR_AMDGPU_015 = ErrorDef(
    domain=ErrorDomain.AMDGPU,
    code=15,
    severity=Severity.ERROR,
    summary="AMDGPU HAL-kernel ABI live-in is duplicated.",
    message=(
        "AMDGPU HAL-kernel ABI live-in source '{source_name}' is defined more than once"
    ),
    params=(ErrorParam("source_name", ParamKind.STRING),),
    fix_hint="Keep at most one low.live_in for each AMDGPU ABI source",
)

# ERR_AMDGPU_016: AMDGPU HAL-kernel ABI workitem live-in forms are mixed.
ERR_AMDGPU_016 = ErrorDef(
    domain=ErrorDomain.AMDGPU,
    code=16,
    severity=Severity.ERROR,
    summary="AMDGPU HAL-kernel ABI workitem live-in forms are mixed.",
    message=(
        "AMDGPU HAL-kernel ABI cannot mix workitem live-in source "
        "'{source_name}' with '{conflicting_source_name}'"
    ),
    params=(
        ErrorParam("source_name", ParamKind.STRING),
        ErrorParam("conflicting_source_name", ParamKind.STRING),
    ),
    fix_hint="Use either packed or unpacked workitem-id live-ins, not both",
)

# ERR_AMDGPU_017: AMDGPU HAL-kernel ABI live-in result type is unsupported.
ERR_AMDGPU_017 = ErrorDef(
    domain=ErrorDomain.AMDGPU,
    code=17,
    severity=Severity.ERROR,
    summary="AMDGPU HAL-kernel ABI live-in result type is unsupported.",
    message=(
        "AMDGPU HAL-kernel ABI live-in source '{source_name}' requires "
        "descriptor register-class ID {expected_reg_class_id} with "
        "{expected_unit_count} unit(s), but found {actual_type}"
    ),
    params=(
        ErrorParam("source_name", ParamKind.STRING),
        ErrorParam("actual_type", ParamKind.TYPE),
        ErrorParam("expected_reg_class_id", ParamKind.U32),
        ErrorParam("expected_unit_count", ParamKind.U32),
    ),
    fix_hint="Use the register shape required by the AMDGPU ABI live-in source",
)

# ERR_AMDGPU_018: AMDGPU HAL-kernel ABI M0 live-in result type is unsupported.
ERR_AMDGPU_018 = ErrorDef(
    domain=ErrorDomain.AMDGPU,
    code=18,
    severity=Severity.ERROR,
    summary="AMDGPU HAL-kernel ABI M0 live-in result type is unsupported.",
    message=(
        "AMDGPU HAL-kernel ABI live-in source '{source_name}' must be a "
        "single unspillable physical register, but found {actual_type}"
    ),
    params=(
        ErrorParam("source_name", ParamKind.STRING),
        ErrorParam("actual_type", ParamKind.TYPE),
    ),
    fix_hint="Use the AMDGPU M0 physical register type for the M0 live-in",
)

# ERR_AMDGPU_019: AMDGPU packed memory payload does not fill registers.
ERR_AMDGPU_019 = ErrorDef(
    domain=ErrorDomain.AMDGPU,
    code=19,
    severity=Severity.ERROR,
    summary="AMDGPU packed memory payload footprint is unsupported.",
    message=(
        "AMDGPU target '{target_key}' export '{export_name}' config "
        "'{config_key}' rejected '{op_name}' in '@{function_name}': payload "
        "type {payload_type} has {payload_bit_count} payload bit(s), but the "
        "selected packed memory register footprint holds {register_bit_count} "
        "bit(s)"
    ),
    params=(
        *_TARGET_CONTEXT_PARAMS,
        ErrorParam("payload_type", ParamKind.TYPE),
        ErrorParam("payload_bit_count", ParamKind.U32),
        ErrorParam("register_bit_count", ParamKind.U32),
    ),
    fix_hint=(
        "Widen the packed payload to fill complete 32-bit AMDGPU memory "
        "registers or use a supported scalar D16 access"
    ),
)

# ERR_AMDGPU_020: AMDGPU flat dynamic memory address is unsupported.
ERR_AMDGPU_020 = ErrorDef(
    domain=ErrorDomain.AMDGPU,
    code=20,
    severity=Severity.ERROR,
    summary="AMDGPU flat dynamic memory address is unsupported.",
    message=(
        "AMDGPU target '{target_key}' export '{export_name}' config "
        "'{config_key}' rejected '{op_name}' in '@{function_name}': "
        "{operation_kind} in {memory_space} memory has flat dynamic address "
        "term {dynamic_term_index} with byte stride {byte_stride}, byte range "
        "[{byte_range_lo}, {byte_range_hi}], and byte shift {byte_shift}"
    ),
    params=(
        *_TARGET_CONTEXT_PARAMS,
        ErrorParam("operation_kind", ParamKind.STRING),
        ErrorParam("memory_space", ParamKind.STRING),
        ErrorParam("dynamic_term_index", ParamKind.U32),
        ErrorParam("byte_stride", ParamKind.I64),
        ErrorParam("byte_range_lo", ParamKind.I64),
        ErrorParam("byte_range_hi", ParamKind.I64),
        ErrorParam("byte_shift", ParamKind.U32),
    ),
    fix_hint=(
        "Prove a non-negative 32-bit dynamic byte address before AMDGPU "
        "flat memory lowering, or lower through an address form that supports "
        "the dynamic term"
    ),
)

# ERR_AMDGPU_021: AMDGPU matrix descriptor selection rejected a source contract.
ERR_AMDGPU_021 = ErrorDef(
    domain=ErrorDomain.AMDGPU,
    code=21,
    severity=Severity.ERROR,
    summary="AMDGPU matrix descriptor selection rejected a source contract.",
    message=(
        "AMDGPU target '{target_key}' export '{export_name}' config "
        "'{config_key}' rejected '{op_name}' in '@{function_name}': matrix "
        "constraint '{matrix_constraint}' is not satisfied "
        "(source_bits={source_rejection_bits}, target_bits={target_rejection_bits})"
    ),
    params=(
        *_TARGET_CONTEXT_PARAMS,
        ErrorParam("matrix_constraint", ParamKind.STRING),
        ErrorParam("source_rejection_bits", ParamKind.U32),
        ErrorParam("target_rejection_bits", ParamKind.U32),
    ),
    fix_hint=(
        "Select a matrix shape, payload, feature set, and fragment schema "
        "supported by the AMDGPU descriptor set"
    ),
)

# ERR_AMDGPU_022: AMDGPU matrix descriptor has no target-low packet mapping.
ERR_AMDGPU_022 = ErrorDef(
    domain=ErrorDomain.AMDGPU,
    code=22,
    severity=Severity.ERROR,
    summary="AMDGPU matrix descriptor has no target-low packet mapping.",
    message=(
        "AMDGPU target '{target_key}' export '{export_name}' config "
        "'{config_key}' rejected '{op_name}' descriptor '{descriptor_name}' "
        "in '@{function_name}': descriptor constraint '{descriptor_constraint}' "
        "is not satisfied"
    ),
    params=(
        *_TARGET_CONTEXT_PARAMS,
        ErrorParam("descriptor_name", ParamKind.STRING),
        ErrorParam("descriptor_constraint", ParamKind.STRING),
    ),
    fix_hint=(
        "Use an AMDGPU matrix descriptor that has a selected target-low packet "
        "mapping in the active descriptor set"
    ),
)

# ERR_AMDGPU_023: AMDGPU source-to-low constraint is not satisfied.
ERR_AMDGPU_023 = ErrorDef(
    domain=ErrorDomain.AMDGPU,
    code=23,
    severity=Severity.ERROR,
    summary="AMDGPU source-to-low constraint is not satisfied.",
    message=(
        "AMDGPU target '{target_key}' export '{export_name}' config "
        "'{config_key}' rejected '{op_name}' in '@{function_name}': "
        "source-to-low constraint '{constraint_key}' is not satisfied"
    ),
    params=(
        *_TARGET_CONTEXT_PARAMS,
        ErrorParam("constraint_key", ParamKind.STRING),
    ),
    fix_hint=(
        "Legalize, refine, or decompose the operation before AMDGPU "
        "source-to-low lowering"
    ),
)

# ERR_AMDGPU_024: AMDGPU memory cache policy is not encodable.
ERR_AMDGPU_024 = ErrorDef(
    domain=ErrorDomain.AMDGPU,
    code=24,
    severity=Severity.ERROR,
    summary="AMDGPU memory cache policy is not encodable.",
    message=(
        "AMDGPU target '{target_key}' export '{export_name}' config "
        "'{config_key}' rejected '{op_name}' in '@{function_name}': memory "
        "cache policy {cache_scope}/{cache_temporal} for {memory_space} "
        "memory is not encodable by descriptor set '{descriptor_set_key}' "
        "because constraint '{constraint_key}' is not satisfied"
    ),
    params=(
        *_TARGET_CONTEXT_PARAMS,
        ErrorParam("constraint_key", ParamKind.STRING),
        ErrorParam("memory_space", ParamKind.STRING),
        ErrorParam("cache_scope", ParamKind.STRING),
        ErrorParam("cache_temporal", ParamKind.STRING),
        ErrorParam("descriptor_set_key", ParamKind.STRING),
    ),
    fix_hint=(
        "Use an AMDGPU cache policy supported by the selected descriptor set "
        "or legalize the access before source-to-low lowering"
    ),
)

# ERR_AMDGPU_025: AMDGPU HAL-kernel ABI direct argument type is unsupported.
ERR_AMDGPU_025 = ErrorDef(
    domain=ErrorDomain.AMDGPU,
    code=25,
    severity=Severity.ERROR,
    summary="AMDGPU HAL-kernel ABI direct argument type is unsupported.",
    message=(
        "AMDGPU HAL-kernel ABI direct argument {argument_index} requires "
        "descriptor register-class ID {expected_reg_class_id} with "
        "1-{expected_unit_count} unit(s), but found {actual_type}"
    ),
    params=(
        ErrorParam("argument_index", ParamKind.U32),
        ErrorParam("actual_type", ParamKind.TYPE),
        ErrorParam("expected_reg_class_id", ParamKind.U32),
        ErrorParam("expected_unit_count", ParamKind.U32),
    ),
    fix_hint="Lower direct HAL kernel arguments to the ABI register shape",
)

# ERR_AMDGPU_026: AMDGPU target subgroup size is unsupported.
ERR_AMDGPU_026 = ErrorDef(
    domain=ErrorDomain.AMDGPU,
    code=26,
    severity=Severity.ERROR,
    summary="AMDGPU target subgroup size is unsupported.",
    message=(
        "AMDGPU target record '@{target_name}' selects subgroup_size "
        "{subgroup_size}, but processor '{processor}' does not support that "
        "wavefront size"
    ),
    params=(
        ErrorParam("target_name", ParamKind.STRING),
        ErrorParam("subgroup_size", ParamKind.U64),
        ErrorParam("processor", ParamKind.STRING),
    ),
    fix_hint="Use a wavefront size supported by the selected AMDGPU processor",
)

# ERR_AMDGPU_027: AMDGPU mixed FMA operand-form decision was recorded.
ERR_AMDGPU_027 = ErrorDef(
    domain=ErrorDomain.AMDGPU,
    code=27,
    severity=Severity.REMARK,
    summary="AMDGPU mixed FMA operand-form decision was recorded.",
    message=(
        "AMDGPU target '{target_key}' export '{export_name}' config "
        "'{config_key}' selected mixed FMA descriptor '{descriptor_name}' for "
        "'{op_name}' in '@{function_name}' and {decision} source operand "
        "{source_operand_index} constant form '{constant_form}' in descriptor "
        "set '{descriptor_set_name}': reason '{reason}'"
    ),
    params=(
        *_TARGET_CONTEXT_PARAMS,
        ErrorParam("descriptor_name", ParamKind.STRING),
        ErrorParam("source_operand_index", ParamKind.U32),
        ErrorParam("descriptor_set_name", ParamKind.STRING),
        ErrorParam("constant_form", ParamKind.STRING),
        ErrorParam("decision", ParamKind.STRING),
        ErrorParam("reason", ParamKind.STRING),
    ),
    fix_hint=(
        "Use the recorded descriptor and operand-form fields to explain why "
        "source-to-low selected or rejected the literal mixed-FMA form"
    ),
)

# ERR_AMDGPU_028: AMDGPU dot FMAC operand-form decision was recorded.
ERR_AMDGPU_028 = ErrorDef(
    domain=ErrorDomain.AMDGPU,
    code=28,
    severity=Severity.REMARK,
    summary="AMDGPU dot FMAC operand-form decision was recorded.",
    message=(
        "AMDGPU target '{target_key}' export '{export_name}' config "
        "'{config_key}' selected dot accumulation descriptor "
        "'{descriptor_name}' for '{op_name}' in '@{function_name}' and "
        "{decision} tied accumulator form at lane {lane_index} in descriptor "
        "set '{descriptor_set_name}': accumulator '{accumulator_kind}', "
        "reason '{reason}'"
    ),
    params=(
        *_TARGET_CONTEXT_PARAMS,
        ErrorParam("descriptor_name", ParamKind.STRING),
        ErrorParam("lane_index", ParamKind.U32),
        ErrorParam("descriptor_set_name", ParamKind.STRING),
        ErrorParam("accumulator_kind", ParamKind.STRING),
        ErrorParam("decision", ParamKind.STRING),
        ErrorParam("reason", ParamKind.STRING),
    ),
    fix_hint=(
        "Use the recorded descriptor, lane, and accumulator fields to explain "
        "why source-to-low selected or rejected tied FMAC accumulation"
    ),
)

# ERR_AMDGPU_029: AMDGPU half-result mixed FMA lane decision was recorded.
ERR_AMDGPU_029 = ErrorDef(
    domain=ErrorDomain.AMDGPU,
    code=29,
    severity=Severity.REMARK,
    summary="AMDGPU half-result mixed FMA lane decision was recorded.",
    message=(
        "AMDGPU target '{target_key}' export '{export_name}' config "
        "'{config_key}' {decision} half-result mixed FMA descriptor "
        "'{descriptor_name}' for '{op_name}' in '@{function_name}' at "
        "destination lane {destination_lane_index} ({result_half}) in "
        "descriptor set '{descriptor_set_name}': sources [{source0_kind}, "
        "{source1_kind}, {source2_kind}], rounding '{rounding_contract}', "
        "reason '{reason}'"
    ),
    params=(
        *_TARGET_CONTEXT_PARAMS,
        ErrorParam("descriptor_name", ParamKind.STRING),
        ErrorParam("destination_lane_index", ParamKind.U32),
        ErrorParam("result_half", ParamKind.STRING),
        ErrorParam("descriptor_set_name", ParamKind.STRING),
        ErrorParam("source0_kind", ParamKind.STRING),
        ErrorParam("source1_kind", ParamKind.STRING),
        ErrorParam("source2_kind", ParamKind.STRING),
        ErrorParam("rounding_contract", ParamKind.STRING),
        ErrorParam("decision", ParamKind.STRING),
        ErrorParam("reason", ParamKind.STRING),
    ),
    fix_hint=(
        "Use the recorded lane, source-kind, and rounding-contract fields to "
        "explain why source-to-low selected or rejected v_fma_mixlo/hi_f16 or "
        "v_mad_mixlo/hi_f16"
    ),
)

# ERR_AMDGPU_030: AMDGPU literal FMA operand-form decision was recorded.
ERR_AMDGPU_030 = ErrorDef(
    domain=ErrorDomain.AMDGPU,
    code=30,
    severity=Severity.REMARK,
    summary="AMDGPU literal FMA operand-form decision was recorded.",
    message=(
        "AMDGPU target '{target_key}' export '{export_name}' config "
        "'{config_key}' {decision} literal FMA descriptor "
        "'{descriptor_name}' for '{op_name}' in '@{function_name}' using "
        "{operand_form} source operand {source_operand_index} ({literal_role}) "
        "constant form '{constant_form}' in descriptor set "
        "'{descriptor_set_name}': reason '{reason}'"
    ),
    params=(
        *_TARGET_CONTEXT_PARAMS,
        ErrorParam("descriptor_name", ParamKind.STRING),
        ErrorParam("operand_form", ParamKind.STRING),
        ErrorParam("source_operand_index", ParamKind.U32),
        ErrorParam("literal_role", ParamKind.STRING),
        ErrorParam("descriptor_set_name", ParamKind.STRING),
        ErrorParam("constant_form", ParamKind.STRING),
        ErrorParam("decision", ParamKind.STRING),
        ErrorParam("reason", ParamKind.STRING),
    ),
    fix_hint=(
        "Use the recorded descriptor, operand-form, and literal-role fields to "
        "explain why source-to-low selected or rejected v_fmaak_f32 or "
        "v_fmamk_f32"
    ),
)

# ERR_AMDGPU_031: AMDGPU FMA alias has compatibility-only semantics.
ERR_AMDGPU_031 = ErrorDef(
    domain=ErrorDomain.AMDGPU,
    code=31,
    severity=Severity.ERROR,
    summary="AMDGPU FMA alias has compatibility-only semantics.",
    message=(
        "low function '@{function_name}' uses AMDGPU FMA alias "
        "'{descriptor_name}' in descriptor set '{descriptor_set_name}', but "
        "mnemonic '{alias_mnemonic}' has '{alias_semantics}' compatibility "
        "semantics and is not an ordinary f32 FMA; use descriptor "
        "'{replacement_descriptor_name}' with mnemonic "
        "'{replacement_mnemonic}' instead: {decision}, reason '{reason}'"
    ),
    params=(
        ErrorParam("function_name", ParamKind.STRING),
        ErrorParam("descriptor_name", ParamKind.STRING),
        ErrorParam("descriptor_set_name", ParamKind.STRING),
        ErrorParam("alias_mnemonic", ParamKind.STRING),
        ErrorParam("alias_semantics", ParamKind.STRING),
        ErrorParam("replacement_descriptor_name", ParamKind.STRING),
        ErrorParam("replacement_mnemonic", ParamKind.STRING),
        ErrorParam("decision", ParamKind.STRING),
        ErrorParam("reason", ParamKind.STRING),
    ),
    fix_hint=(
        "Use ordinary v_fma_f32 or v_fmac_f32 descriptors for IEEE f32 FMA; "
        "DX9-zero aliases are compatibility opcodes with different zero "
        "semantics"
    ),
)

# ERR_AMDGPU_032: AMDGPU FMA asm mnemonic has compatibility-only semantics.
ERR_AMDGPU_032 = ErrorDef(
    domain=ErrorDomain.AMDGPU,
    code=32,
    severity=Severity.ERROR,
    summary="AMDGPU FMA asm mnemonic has compatibility-only semantics.",
    message=(
        "low asm descriptor set '{descriptor_set_name}' rejects mnemonic "
        "'{alias_mnemonic}': it has '{alias_semantics}' compatibility "
        "semantics and is not an ordinary f32 FMA; use descriptor "
        "'{replacement_descriptor_name}' with mnemonic "
        "'{replacement_mnemonic}' instead: {decision}, reason '{reason}'"
    ),
    params=(
        ErrorParam("descriptor_set_name", ParamKind.STRING),
        ErrorParam("alias_mnemonic", ParamKind.STRING),
        ErrorParam("alias_semantics", ParamKind.STRING),
        ErrorParam("replacement_descriptor_name", ParamKind.STRING),
        ErrorParam("replacement_mnemonic", ParamKind.STRING),
        ErrorParam("decision", ParamKind.STRING),
        ErrorParam("reason", ParamKind.STRING),
    ),
    fix_hint=(
        "Use ordinary v_fma_f32 or v_fmac_f32 mnemonics for IEEE f32 FMA; "
        "DX9-zero aliases are compatibility opcodes with different zero "
        "semantics"
    ),
)

# ERR_AMDGPU_033: AMDGPU index cast source range does not fit target index width.
ERR_AMDGPU_033 = ErrorDef(
    domain=ErrorDomain.AMDGPU,
    code=33,
    severity=Severity.ERROR,
    summary="AMDGPU index cast source range does not fit target index width.",
    message=(
        "AMDGPU target '{target_key}' export '{export_name}' config "
        "'{config_key}' rejected '{op_name}' in '@{function_name}': "
        "index cast from {source_type} to {result_type} has source range "
        "[{source_range_lo}, {source_range_hi}], which is not proven to fit "
        "the {index_bitwidth}-bit target index domain "
        "[{required_range_lo}, {required_range_hi}]; constraint "
        "'{constraint_key}' is not satisfied; accepted proof sources are "
        "{accepted_proof_sources}"
    ),
    params=(
        *_TARGET_CONTEXT_PARAMS,
        ErrorParam("source_type", ParamKind.TYPE),
        ErrorParam("result_type", ParamKind.TYPE),
        ErrorParam("source_range_lo", ParamKind.I64),
        ErrorParam("source_range_hi", ParamKind.I64),
        ErrorParam("index_bitwidth", ParamKind.U32),
        ErrorParam("required_range_lo", ParamKind.I64),
        ErrorParam("required_range_hi", ParamKind.I64),
        ErrorParam("constraint_key", ParamKind.STRING),
        ErrorParam("accepted_proof_sources", ParamKind.STRING_LIST),
    ),
    fix_hint=(
        "Assume the integer source before the index cast, after any required "
        "runtime guard, so value facts prove it fits the target index width; "
        "use an offset path when the value is a byte address"
    ),
)

# ERR_AMDGPU_034: AMDGPU dynamic memory byte offset range is unsupported.
ERR_AMDGPU_034 = ErrorDef(
    domain=ErrorDomain.AMDGPU,
    code=34,
    severity=Severity.ERROR,
    summary="AMDGPU dynamic memory byte offset range is unsupported.",
    message=(
        "AMDGPU target '{target_key}' export '{export_name}' config "
        "'{config_key}' rejected '{op_name}' in '@{function_name}': "
        "{operation_kind} in {memory_space} memory has dynamic byte offset "
        "range [{byte_offset_range_lo}, {byte_offset_range_hi}], which is "
        "not proven to fit the non-negative 32-bit AMDGPU memory offset "
        "domain; dynamic term {dynamic_term_index} has byte stride "
        "{dynamic_term_byte_stride}, byte range "
        "[{dynamic_term_byte_range_lo}, {dynamic_term_byte_range_hi}], and "
        "byte shift {dynamic_term_byte_shift}; static byte offset is "
        "{static_byte_offset}; constraint '{constraint_key}' is not satisfied"
    ),
    params=(
        *_TARGET_CONTEXT_PARAMS,
        ErrorParam("operation_kind", ParamKind.STRING),
        ErrorParam("memory_space", ParamKind.STRING),
        ErrorParam("byte_offset_range_lo", ParamKind.I64),
        ErrorParam("byte_offset_range_hi", ParamKind.I64),
        ErrorParam("dynamic_term_index", ParamKind.U32),
        ErrorParam("dynamic_term_byte_stride", ParamKind.I64),
        ErrorParam("dynamic_term_byte_range_lo", ParamKind.I64),
        ErrorParam("dynamic_term_byte_range_hi", ParamKind.I64),
        ErrorParam("dynamic_term_byte_shift", ParamKind.U32),
        ErrorParam("static_byte_offset", ParamKind.I64),
        ErrorParam("constraint_key", ParamKind.STRING),
    ),
    fix_hint=(
        "Keep byte addresses in the offset domain, use index.scale when a "
        "logical index is multiplied by an offset byte stride, and prove the "
        "resulting byte offset range when target facts cannot infer it"
    ),
)

# ERR_AMDGPU_035: AMDGPU native emission metadata does not support register class.
ERR_AMDGPU_035 = ErrorDef(
    domain=ErrorDomain.AMDGPU,
    code=35,
    severity=Severity.ERROR,
    summary="AMDGPU native emission register metadata is unsupported.",
    message=(
        "AMDGPU target '{target_key}' export '{export_name}' config "
        "'{config_key}' cannot emit native metadata for '@{function_name}': "
        "{value_class} value '{value_name}' is assigned to register class "
        "'{register_class}', but {metadata_contract} metadata is not "
        "implemented for that class"
    ),
    params=(
        ErrorParam("target_key", ParamKind.STRING),
        ErrorParam("export_name", ParamKind.STRING),
        ErrorParam("config_key", ParamKind.STRING),
        ErrorParam("function_name", ParamKind.STRING),
        ErrorParam("value_name", ParamKind.STRING),
        ErrorParam("value_class", ParamKind.STRING),
        ErrorParam("register_class", ParamKind.STRING),
        ErrorParam("metadata_contract", ParamKind.STRING),
    ),
    fix_hint=(
        "Use register classes with complete AMDGPU native metadata support, "
        "or implement kernel descriptor metadata for the selected class"
    ),
)

ALL_AMDGPU_ERRORS = (
    ERR_AMDGPU_001,
    ERR_AMDGPU_003,
    ERR_AMDGPU_004,
    ERR_AMDGPU_005,
    ERR_AMDGPU_006,
    ERR_AMDGPU_007,
    ERR_AMDGPU_008,
    ERR_AMDGPU_009,
    ERR_AMDGPU_010,
    ERR_AMDGPU_011,
    ERR_AMDGPU_012,
    ERR_AMDGPU_013,
    ERR_AMDGPU_014,
    ERR_AMDGPU_015,
    ERR_AMDGPU_016,
    ERR_AMDGPU_017,
    ERR_AMDGPU_018,
    ERR_AMDGPU_019,
    ERR_AMDGPU_020,
    ERR_AMDGPU_021,
    ERR_AMDGPU_022,
    ERR_AMDGPU_023,
    ERR_AMDGPU_024,
    ERR_AMDGPU_025,
    ERR_AMDGPU_026,
    ERR_AMDGPU_027,
    ERR_AMDGPU_028,
    ERR_AMDGPU_029,
    ERR_AMDGPU_030,
    ERR_AMDGPU_031,
    ERR_AMDGPU_032,
    ERR_AMDGPU_033,
    ERR_AMDGPU_034,
    ERR_AMDGPU_035,
)
