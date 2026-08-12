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

# ERR_AMDGPU_006: AMDGPU HAL-kernel ABI resource count overflows.
ERR_AMDGPU_006 = ErrorDef(
    domain=ErrorDomain.AMDGPU,
    code=6,
    severity=Severity.ERROR,
    summary="AMDGPU HAL-kernel ABI resource count overflows.",
    message=(
        "AMDGPU HAL-kernel ABI has {resource_count} HAL binding resources, "
        "but at most {max_resource_count} fit in the ABI layout snapshot"
    ),
    params=(
        ErrorParam("resource_count", ParamKind.U64),
        ErrorParam("max_resource_count", ParamKind.U64),
    ),
    fix_hint=("Split the kernel ABI or reduce the number of HAL binding resources"),
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

# ERR_AMDGPU_022: AMDGPU matrix descriptor lacks required target support.
ERR_AMDGPU_022 = ErrorDef(
    domain=ErrorDomain.AMDGPU,
    code=22,
    severity=Severity.ERROR,
    summary="AMDGPU matrix descriptor lacks required target support.",
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
        "Use an AMDGPU matrix descriptor whose selected target support is "
        "complete in the active descriptor set"
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
        "'{op_name}' in '@{function_name}' with decision key '{decision_key}' "
        "for source operand "
        "{source_operand_index} constant form '{constant_form}' in descriptor "
        "set '{descriptor_set_name}': reason key '{reason_key}'"
    ),
    params=(
        *_TARGET_CONTEXT_PARAMS,
        ErrorParam("descriptor_name", ParamKind.STRING),
        ErrorParam("source_operand_index", ParamKind.U32),
        ErrorParam("descriptor_set_name", ParamKind.STRING),
        ErrorParam("constant_form", ParamKind.STRING),
        ErrorParam("decision_key", ParamKind.STRING),
        ErrorParam("reason_key", ParamKind.STRING),
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
        "recorded decision key '{decision_key}' for tied accumulator form at "
        "lane {lane_index} in descriptor "
        "set '{descriptor_set_name}': accumulator '{accumulator_kind}', "
        "reason key '{reason_key}'"
    ),
    params=(
        *_TARGET_CONTEXT_PARAMS,
        ErrorParam("descriptor_name", ParamKind.STRING),
        ErrorParam("lane_index", ParamKind.U32),
        ErrorParam("descriptor_set_name", ParamKind.STRING),
        ErrorParam("accumulator_kind", ParamKind.STRING),
        ErrorParam("decision_key", ParamKind.STRING),
        ErrorParam("reason_key", ParamKind.STRING),
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
        "'{config_key}' recorded decision key '{decision_key}' for "
        "half-result mixed FMA descriptor "
        "'{descriptor_name}' for '{op_name}' in '@{function_name}' at "
        "destination lane {destination_lane_index} ({result_half}) in "
        "descriptor set '{descriptor_set_name}': sources [{source0_kind}, "
        "{source1_kind}, {source2_kind}], rounding '{rounding_contract}', "
        "reason key '{reason_key}'"
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
        ErrorParam("decision_key", ParamKind.STRING),
        ErrorParam("reason_key", ParamKind.STRING),
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
        "'{config_key}' recorded decision key '{decision_key}' for literal "
        "FMA descriptor "
        "'{descriptor_name}' for '{op_name}' in '@{function_name}' using "
        "{operand_form} source operand {source_operand_index} ({literal_role}) "
        "constant form '{constant_form}' in descriptor set "
        "'{descriptor_set_name}': reason key '{reason_key}'"
    ),
    params=(
        *_TARGET_CONTEXT_PARAMS,
        ErrorParam("descriptor_name", ParamKind.STRING),
        ErrorParam("operand_form", ParamKind.STRING),
        ErrorParam("source_operand_index", ParamKind.U32),
        ErrorParam("literal_role", ParamKind.STRING),
        ErrorParam("descriptor_set_name", ParamKind.STRING),
        ErrorParam("constant_form", ParamKind.STRING),
        ErrorParam("decision_key", ParamKind.STRING),
        ErrorParam("reason_key", ParamKind.STRING),
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
        "'{replacement_mnemonic}' instead: decision key '{decision_key}', "
        "reason key '{reason_key}'"
    ),
    params=(
        ErrorParam("function_name", ParamKind.STRING),
        ErrorParam("descriptor_name", ParamKind.STRING),
        ErrorParam("descriptor_set_name", ParamKind.STRING),
        ErrorParam("alias_mnemonic", ParamKind.STRING),
        ErrorParam("alias_semantics", ParamKind.STRING),
        ErrorParam("replacement_descriptor_name", ParamKind.STRING),
        ErrorParam("replacement_mnemonic", ParamKind.STRING),
        ErrorParam("decision_key", ParamKind.STRING),
        ErrorParam("reason_key", ParamKind.STRING),
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
        "'{replacement_mnemonic}' instead: decision key '{decision_key}', "
        "reason key '{reason_key}'"
    ),
    params=(
        ErrorParam("descriptor_set_name", ParamKind.STRING),
        ErrorParam("alias_mnemonic", ParamKind.STRING),
        ErrorParam("alias_semantics", ParamKind.STRING),
        ErrorParam("replacement_descriptor_name", ParamKind.STRING),
        ErrorParam("replacement_mnemonic", ParamKind.STRING),
        ErrorParam("decision_key", ParamKind.STRING),
        ErrorParam("reason_key", ParamKind.STRING),
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

# ERR_AMDGPU_036: AMDGPU native emission storage space is unsupported.
ERR_AMDGPU_036 = ErrorDef(
    domain=ErrorDomain.AMDGPU,
    code=36,
    severity=Severity.ERROR,
    summary="AMDGPU native emission storage space is unsupported.",
    message=(
        "AMDGPU target '{target_key}' export '{export_name}' config "
        "'{config_key}' cannot emit native storage for '@{function_name}': "
        "storage value '{storage_value_name}' reserves '{storage_space}' "
        "storage, but AMDGPU native emission supports "
        "{supported_storage_spaces}"
    ),
    params=(
        ErrorParam("target_key", ParamKind.STRING),
        ErrorParam("export_name", ParamKind.STRING),
        ErrorParam("config_key", ParamKind.STRING),
        ErrorParam("function_name", ParamKind.STRING),
        ErrorParam("storage_value_name", ParamKind.STRING),
        ErrorParam("storage_space", ParamKind.STRING),
        ErrorParam("supported_storage_spaces", ParamKind.STRING_LIST),
    ),
    fix_hint=(
        "Use scratch, private, or workgroup storage for AMDGPU native emission, "
        "or add native ABI lowering for the selected storage space"
    ),
)

# ERR_AMDGPU_037: AMDGPU spill traffic storage space is unsupported.
ERR_AMDGPU_037 = ErrorDef(
    domain=ErrorDomain.AMDGPU,
    code=37,
    severity=Severity.ERROR,
    summary="AMDGPU spill traffic storage space is unsupported.",
    message=(
        "AMDGPU spill lowering rejected '{op_name}' in '@{function_name}': "
        "storage value '{storage_value_name}' reserves '{storage_space}' "
        "storage, but spill traffic lowering supports "
        "{supported_storage_spaces}"
    ),
    params=(
        ErrorParam("function_name", ParamKind.STRING),
        ErrorParam("op_name", ParamKind.STRING),
        ErrorParam("storage_value_name", ParamKind.STRING),
        ErrorParam("storage_space", ParamKind.STRING),
        ErrorParam("supported_storage_spaces", ParamKind.STRING_LIST),
    ),
    fix_hint=(
        "Use scratch or private storage for AMDGPU spill traffic, or let "
        "target allocation materialization create the spill storage"
    ),
)

# ERR_AMDGPU_038: AMDGPU spill traffic register type is unsupported.
ERR_AMDGPU_038 = ErrorDef(
    domain=ErrorDomain.AMDGPU,
    code=38,
    severity=Severity.ERROR,
    summary="AMDGPU spill traffic register type is unsupported.",
    message=(
        "AMDGPU spill lowering rejected '{op_name}' in '@{function_name}': "
        "value '{value_name}' has type {actual_type}, but spill traffic "
        "lowering supports AMDGPU SGPR or VGPR register values"
    ),
    params=(
        ErrorParam("function_name", ParamKind.STRING),
        ErrorParam("op_name", ParamKind.STRING),
        ErrorParam("value_name", ParamKind.STRING),
        ErrorParam("actual_type", ParamKind.TYPE),
    ),
    fix_hint=(
        "Spill only values allocated in the selected AMDGPU SGPR or VGPR "
        "descriptor register class"
    ),
)

# ERR_AMDGPU_039: AMDGPU spill traffic storage access is out of bounds.
ERR_AMDGPU_039 = ErrorDef(
    domain=ErrorDomain.AMDGPU,
    code=39,
    severity=Severity.ERROR,
    summary="AMDGPU spill traffic storage access is out of bounds.",
    message=(
        "AMDGPU spill lowering rejected '{op_name}' in '@{function_name}': "
        "access to storage value '{storage_value_name}' starts at byte offset "
        "{access_byte_offset} with byte length {access_byte_length}, but the "
        "resolved storage reference has byte length {storage_byte_length}"
    ),
    params=(
        ErrorParam("function_name", ParamKind.STRING),
        ErrorParam("op_name", ParamKind.STRING),
        ErrorParam("storage_value_name", ParamKind.STRING),
        ErrorParam("access_byte_offset", ParamKind.I64),
        ErrorParam("access_byte_length", ParamKind.U64),
        ErrorParam("storage_byte_length", ParamKind.U64),
    ),
    fix_hint=(
        "Use a non-negative spill/reload byte offset whose accessed range fits "
        "inside the selected storage reservation or storage view"
    ),
)

# ERR_AMDGPU_040: AMDGPU memory vector is wider than supported packets.
ERR_AMDGPU_040 = ErrorDef(
    domain=ErrorDomain.AMDGPU,
    code=40,
    severity=Severity.ERROR,
    summary="AMDGPU memory vector is wider than supported packets.",
    message=(
        "AMDGPU target '{target_key}' export '{export_name}' config "
        "'{config_key}' rejected {operation_kind} '{op_name}' in "
        "'@{function_name}': vector type {vector_type} has "
        "{vector_lane_count} lanes of {element_byte_count} bytes "
        "requiring {required_32bit_lane_count} 32-bit payload lanes, "
        "but native memory packets support {native_max_vector_lane_count} "
        "source lanes ({native_max_32bit_lane_count} 32-bit payload lanes) "
        "and scalarized fallback supports {scalarized_max_vector_lane_count} "
        "source lanes ({scalarized_max_32bit_lane_count} 32-bit payload lanes)"
    ),
    params=(
        *_TARGET_CONTEXT_PARAMS,
        ErrorParam("operation_kind", ParamKind.STRING),
        ErrorParam("vector_type", ParamKind.TYPE),
        ErrorParam("vector_lane_count", ParamKind.U32),
        ErrorParam("element_byte_count", ParamKind.U32),
        ErrorParam("required_32bit_lane_count", ParamKind.U32),
        ErrorParam("native_max_32bit_lane_count", ParamKind.U32),
        ErrorParam("native_max_vector_lane_count", ParamKind.U32),
        ErrorParam("scalarized_max_32bit_lane_count", ParamKind.U32),
        ErrorParam("scalarized_max_vector_lane_count", ParamKind.U32),
        ErrorParam("constraint_key", ParamKind.STRING),
    ),
    fix_hint=(
        "Decompose the vector memory operation before AMDGPU source-to-low "
        "lowering so each native packet fits the reported lane limit"
    ),
)

# ERR_AMDGPU_041: AMDGPU fragment repack strategy is unsupported.
ERR_AMDGPU_041 = ErrorDef(
    domain=ErrorDomain.AMDGPU,
    code=41,
    severity=Severity.ERROR,
    summary="AMDGPU fragment repack strategy is unsupported.",
    message=(
        "AMDGPU target '{target_key}' export '{export_name}' config "
        "'{config_key}' rejected fragment repack '{op_name}' in "
        "'@{function_name}': source role {source_role_key} with type "
        "{source_type} cannot produce result role {result_role_key} with type "
        "{result_type} using strategy '{strategy_key}' for reason "
        "'{reason_key}'"
    ),
    params=(
        *_TARGET_CONTEXT_PARAMS,
        ErrorParam("source_role_key", ParamKind.STRING),
        ErrorParam("result_role_key", ParamKind.STRING),
        ErrorParam("source_type", ParamKind.TYPE),
        ErrorParam("result_type", ParamKind.TYPE),
        ErrorParam("strategy_key", ParamKind.STRING),
        ErrorParam("reason_key", ParamKind.STRING),
    ),
    fix_hint=(
        "Use memory-backed fragment store/load until the target reports a "
        "native fragment repack strategy for this role and type transition"
    ),
)

# ERR_AMDGPU_042: AMDGPU sanitizer site table is too large.
ERR_AMDGPU_042 = ErrorDef(
    domain=ErrorDomain.AMDGPU,
    code=42,
    severity=Severity.ERROR,
    summary="AMDGPU sanitizer site table is too large.",
    message=(
        "AMDGPU target '{target_key}' export '{export_name}' config "
        "'{config_key}' rejected sanitizer metadata for '{op_name}' in "
        "'@{function_name}': current table has {current_site_count} site row(s) "
        "and this function contributes {function_site_count}, but the maximum "
        "encodable site id is {max_site_id}"
    ),
    params=(
        *_TARGET_CONTEXT_PARAMS,
        ErrorParam("current_site_count", ParamKind.U64),
        ErrorParam("function_site_count", ParamKind.U64),
        ErrorParam("max_site_id", ParamKind.U64),
    ),
    fix_hint=(
        "Split the sanitizer-instrumented module or reduce the number of "
        "sanitizer assertion sites before AMDGPU lowering"
    ),
)

# ERR_AMDGPU_043: AMDGPU sanitizer site table symbol is reserved.
ERR_AMDGPU_043 = ErrorDef(
    domain=ErrorDomain.AMDGPU,
    code=43,
    severity=Severity.ERROR,
    summary="AMDGPU sanitizer site table symbol is reserved.",
    message=(
        "AMDGPU target '{target_key}' export '{export_name}' config "
        "'{config_key}' rejected sanitizer metadata for '{op_name}' in "
        "'@{function_name}': module symbol '{symbol_name}' is already defined"
    ),
    params=(
        *_TARGET_CONTEXT_PARAMS,
        ErrorParam("symbol_name", ParamKind.STRING),
    ),
    fix_hint=(
        "Rename the user-defined symbol so AMDGPU sanitizer lowering can "
        "materialize its reserved site table"
    ),
)

# ERR_AMDGPU_044: AMDGPU DPP control encoding is reserved.
ERR_AMDGPU_044 = ErrorDef(
    domain=ErrorDomain.AMDGPU,
    code=44,
    severity=Severity.ERROR,
    summary="AMDGPU DPP control encoding is reserved.",
    message=(
        "AMDGPU descriptor '{descriptor_name}' in '@{function_name}' uses "
        "reserved DPP control value {immediate_value} for immediate "
        "'{immediate_name}' in descriptor set '{descriptor_set_name}' "
        "under contract '{constraint_key}' for reason '{reason_key}'"
    ),
    params=(
        ErrorParam("function_name", ParamKind.STRING),
        ErrorParam("descriptor_name", ParamKind.STRING),
        ErrorParam("descriptor_set_name", ParamKind.STRING),
        ErrorParam("immediate_name", ParamKind.STRING),
        ErrorParam("immediate_value", ParamKind.U32),
        ErrorParam("constraint_key", ParamKind.STRING),
        ErrorParam("reason_key", ParamKind.STRING),
    ),
    fix_hint=(
        "Use a legal quad, row, or wave DPP lane-control encoding for the "
        "selected descriptor"
    ),
)

# ERR_AMDGPU_045: AMDGPU HAL-kernel ABI fixed live-ins overlap.
ERR_AMDGPU_045 = ErrorDef(
    domain=ErrorDomain.AMDGPU,
    code=45,
    severity=Severity.ERROR,
    summary="AMDGPU HAL-kernel ABI fixed live-ins overlap.",
    message=(
        "AMDGPU HAL-kernel ABI live-in source '{source_name}' and "
        "'{conflicting_source_name}' both require fixed SGPR {fixed_location}"
    ),
    params=(
        ErrorParam("source_name", ParamKind.STRING),
        ErrorParam("conflicting_source_name", ParamKind.STRING),
        ErrorParam("fixed_location", ParamKind.U32),
    ),
    fix_hint=(
        "Use one live-in source that represents all fields imported from the "
        "fixed register"
    ),
)

# ERR_AMDGPU_047: target instruction constraint requires legalization.
ERR_AMDGPU_047 = ErrorDef(
    domain=ErrorDomain.AMDGPU,
    code=47,
    severity=Severity.ERROR,
    summary="AMDGPU instruction constraint requires legalization.",
    message=(
        "AMDGPU descriptor '{descriptor_name}' in '@{function_name}' violates "
        "{constraint_kind} constraint '{constraint_key}' for target "
        "'{target}' in descriptor set "
        "'{descriptor_set_name}' and requires legalization "
        "'{legalization_key}'"
    ),
    params=(
        ErrorParam("function_name", ParamKind.STRING),
        ErrorParam("descriptor_name", ParamKind.STRING),
        ErrorParam("descriptor_set_name", ParamKind.STRING),
        ErrorParam("target", ParamKind.STRING),
        ErrorParam("constraint_kind", ParamKind.STRING),
        ErrorParam("constraint_key", ParamKind.STRING),
        ErrorParam("legalization_key", ParamKind.STRING),
    ),
    fix_hint=("Apply the named target legalization before native emission"),
)

# ERR_AMDGPU_048: target feature requirement is unsupported by the processor.
ERR_AMDGPU_048 = ErrorDef(
    domain=ErrorDomain.AMDGPU,
    code=48,
    severity=Severity.ERROR,
    summary="AMDGPU target feature requirement is unsupported by the processor.",
    message=(
        "AMDGPU target '@{target_name}' requires target feature '{feature}' "
        "{required_state}, but processor '{processor}' does not support the "
        "feature"
    ),
    params=(
        ErrorParam("target_name", ParamKind.STRING),
        ErrorParam("feature", ParamKind.STRING),
        ErrorParam("required_state", ParamKind.STRING),
        ErrorParam("processor", ParamKind.STRING),
    ),
    fix_hint=(
        "Select an enabled or disabled state only for processor-supported "
        "target features"
    ),
)

# ERR_AMDGPU_049: AMDGPU storage address result type is unsupported.
ERR_AMDGPU_049 = ErrorDef(
    domain=ErrorDomain.AMDGPU,
    code=49,
    severity=Severity.ERROR,
    summary="AMDGPU storage address result type is unsupported.",
    message=(
        "AMDGPU target '{target_key}' export '{export_name}' config "
        "'{config_key}' rejected '{op_name}' in '@{function_name}': result "
        "'{result_name}' has type {actual_type}, but native storage addresses "
        "require descriptor register-class ID {expected_reg_class_id} with "
        "{expected_unit_count} unit"
    ),
    params=(
        *_TARGET_CONTEXT_PARAMS,
        ErrorParam("result_name", ParamKind.STRING),
        ErrorParam("actual_type", ParamKind.TYPE),
        ErrorParam("expected_reg_class_id", ParamKind.U32),
        ErrorParam("expected_unit_count", ParamKind.U32),
    ),
    fix_hint="Materialize AMDGPU storage addresses as one VGPR",
)

# ERR_AMDGPU_050: target feature assertion set is vacuous.
ERR_AMDGPU_050 = ErrorDef(
    domain=ErrorDomain.AMDGPU,
    code=50,
    severity=Severity.ERROR,
    summary="AMDGPU target feature assertion set is empty.",
    message="AMDGPU target '@{target_name}' has an empty feature assertion set",
    params=(ErrorParam("target_name", ParamKind.STRING),),
    fix_hint="Omit the optional features field when no assertions are required",
)

ALL_AMDGPU_ERRORS = (
    ERR_AMDGPU_001,
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
    ERR_AMDGPU_036,
    ERR_AMDGPU_037,
    ERR_AMDGPU_038,
    ERR_AMDGPU_039,
    ERR_AMDGPU_040,
    ERR_AMDGPU_041,
    ERR_AMDGPU_042,
    ERR_AMDGPU_043,
    ERR_AMDGPU_044,
    ERR_AMDGPU_045,
    ERR_AMDGPU_047,
    ERR_AMDGPU_048,
    ERR_AMDGPU_049,
    ERR_AMDGPU_050,
)
