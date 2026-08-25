# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


# ruff: noqa: F403, F405

"""Public AMDGPU descriptor construction API."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from functools import cache

from loom.target.arch.amdgpu.encoding import (
    AMDGPU_DPP_CONTROL_ENCODING_FORMAT_IDS,
    AMDGPU_ENCODING_FORMAT_NONE,
    AMDGPU_ENCODING_FORMAT_XML_NAMES_BY_ID,
    AMDGPU_GFX125X_VGPR_MSB_BANK_COUNT,
    AMDGPU_GFX125X_VGPR_MSB_WINDOW_SIZE,
    AmdgpuVgprMsbSlot,
    amdgpu_dpp_control_is_valid,
    amdgpu_encoding_field_id,
    amdgpu_encoding_format_id,
    amdgpu_gfx125x_vgpr_msb_slot,
    amdgpu_supplemental_encoding_format_names,
)
from loom.target.arch.amdgpu.target_info import (
    AMDGPU_DESCRIPTOR_SET_INFO_FLAG_NATIVE_PACKED_BF16_ARITHMETIC,
    AMDGPU_DESCRIPTOR_SET_INFO_FLAG_NATIVE_SCALAR_FLOAT_ARITHMETIC,
    AMDGPU_DESCRIPTOR_SET_INFO_FLAG_NATIVE_SCALAR_FLOAT_COMPARE,
    AMDGPU_DESCRIPTOR_SET_INFO_FLAG_NATIVE_SCALAR_FLOAT_CONVERSION,
    AMDGPU_MATRIX_COEXECUTION_PROFILE_NONE,
    AMDGPU_MATRIX_COEXECUTION_RULES_BY_PROFILE,
    AMDGPU_MATRIX_COEXECUTION_SOURCE_INFOS,
    AMDGPU_MATRIX_COEXECUTION_SOURCE_SWMMAC,
    AMDGPU_MATRIX_COEXECUTION_SOURCE_WMMA,
    AMDGPU_PROCESSOR_INFOS,
    AMDGPU_TARGET_INFOS,
    AmdgpuDescriptorSetInfo,
    amdgpu_descriptor_set_info_by_generator_target,
    amdgpu_descriptor_set_supported_target_contract_keys,
    amdgpu_target_descriptor_set_key,
)
from loom.target.low_descriptors import InstructionClass

from .categories import *
from .cluster import _gfx125x_cluster_descriptors
from .common import *
from .control import _s_delay_alu_descriptor, _s_wait_xcnt_descriptor
from .rdna4m import (
    _RDNA4M_IEEE_MINMAX_INSTRUCTION_ROWS,
    _RDNA4M_NUMERIC_MINMAX_INSTRUCTION_ROWS,
)
from .sets import *
from .tensor import _gfx125x_tensor_descriptors


def _descriptor_has_memory_effect(descriptor: Descriptor) -> bool:
    return any(
        effect.kind in (EffectKind.READ, EffectKind.WRITE)
        and effect.memory_space in (MemorySpace.GLOBAL, MemorySpace.WORKGROUP)
        for effect in descriptor.effects
    )


def _descriptor_address_offset_immediates(
    descriptor: Descriptor,
) -> tuple[Immediate, ...]:
    return tuple(
        immediate
        for immediate in descriptor.immediates
        if immediate.field_name in _ADDRESS_OFFSET_IMMEDIATE_FIELD_NAMES
    )


def _validate_address_immediate_units(descriptor_set: DescriptorSet) -> None:
    for descriptor in descriptor_set.descriptors:
        if not _descriptor_has_memory_effect(descriptor):
            continue
        offset_immediates = _descriptor_address_offset_immediates(descriptor)
        if not offset_immediates:
            continue
        for immediate in offset_immediates:
            if immediate.encoding_id not in _ADDRESS_OFFSET_IMMEDIATE_ENCODING_IDS:
                raise ValueError(
                    f"AMDGPU memory descriptor '{descriptor.key}' immediate "
                    f"'{immediate.field_name}' has no address-unit encoding"
                )
        split_offset_immediates = tuple(
            immediate
            for immediate in offset_immediates
            if immediate.field_name in ("offset0", "offset1")
        )
        if split_offset_immediates:
            if len(split_offset_immediates) != 2:
                raise ValueError(
                    f"AMDGPU memory descriptor '{descriptor.key}' has an "
                    "incomplete split address offset"
                )
            first_encoding_id = split_offset_immediates[0].encoding_id
            if any(
                immediate.encoding_id != first_encoding_id
                for immediate in split_offset_immediates[1:]
            ):
                raise ValueError(
                    f"AMDGPU memory descriptor '{descriptor.key}' has "
                    "inconsistent split address offset units"
                )


def _validate_descriptor_encoding_formats(
    target: str,
    spec: AmdgpuIsaFactSource,
    descriptor_set: DescriptorSet,
) -> None:
    supported_format_names = {
        *(encoding.name for encoding in spec.encodings),
        *amdgpu_supplemental_encoding_format_names(target),
    }
    supported_format_ids = {
        amdgpu_encoding_format_id(format_name) for format_name in supported_format_names
    }
    for descriptor in descriptor_set.descriptors:
        format_id = descriptor.encoding_format_id
        if (
            format_id == AMDGPU_ENCODING_FORMAT_NONE
            or format_id in supported_format_ids
        ):
            continue
        format_name = AMDGPU_ENCODING_FORMAT_XML_NAMES_BY_ID.get(
            format_id, f"id {format_id}"
        )
        raise ValueError(
            f"AMDGPU descriptor target '{target}' descriptor "
            f"'{descriptor.key}' uses unavailable encoding format "
            f"'{format_name}'"
        )


def _validate_dpp_control_fields(descriptor_set: DescriptorSet) -> None:
    dpp_control_field_id = amdgpu_encoding_field_id("DPP_CTRL")
    for descriptor in descriptor_set.descriptors:
        dpp_operands = tuple(
            operand
            for operand in descriptor.operands
            if operand.encoding_field_id == dpp_control_field_id
        )
        dpp_immediates = tuple(
            immediate
            for immediate in descriptor.immediates
            if immediate.encoding_field_id == dpp_control_field_id
            or any(
                encoding_slice.encoding_field_id == dpp_control_field_id
                for encoding_slice in immediate.encoding_slices
            )
        )
        dpp_fixed_values = tuple(
            field_value.value
            for field_value in descriptor.encoding_field_values
            if field_value.encoding_field_id == dpp_control_field_id
        )
        source_count = len(dpp_operands) + len(dpp_immediates) + len(dpp_fixed_values)
        is_dpp_control_format = (
            descriptor.encoding_format_id in AMDGPU_DPP_CONTROL_ENCODING_FORMAT_IDS
        )
        if not is_dpp_control_format:
            if source_count != 0:
                raise ValueError(
                    f"AMDGPU descriptor '{descriptor.key}' maps DPP_CTRL through "
                    "a non-DPP encoding format"
                )
            continue
        if source_count != 1:
            raise ValueError(
                f"AMDGPU DPP descriptor '{descriptor.key}' must define exactly "
                f"one DPP_CTRL source; found {source_count}"
            )
        if dpp_operands:
            raise ValueError(
                f"AMDGPU DPP descriptor '{descriptor.key}' maps DPP_CTRL from "
                "a register operand instead of an immediate or fixed value"
            )
        if dpp_fixed_values:
            if not amdgpu_dpp_control_is_valid(dpp_fixed_values[0]):
                raise ValueError(
                    f"AMDGPU DPP descriptor '{descriptor.key}' fixes DPP_CTRL "
                    f"to reserved value {dpp_fixed_values[0]}"
                )
            continue
        immediate = dpp_immediates[0]
        if (
            immediate.encoding_field_id != dpp_control_field_id
            or immediate.encoding_slices
        ):
            raise ValueError(
                f"AMDGPU DPP descriptor '{descriptor.key}' immediate "
                f"'{immediate.field_name}' must map DPP_CTRL directly"
            )
        if (
            immediate.kind is not ImmediateKind.UNSIGNED
            or immediate.bit_width != 9
            or immediate.unsigned_max != 0x1FF
        ):
            raise ValueError(
                f"AMDGPU DPP descriptor '{descriptor.key}' immediate "
                f"'{immediate.field_name}' must expose the unsigned 9-bit "
                "DPP_CTRL encoding domain"
            )
        if (
            ImmediateFlag.DEFAULT_VALUE in immediate.flags
            and not amdgpu_dpp_control_is_valid(immediate.default_value)
        ):
            raise ValueError(
                f"AMDGPU DPP descriptor '{descriptor.key}' immediate "
                f"'{immediate.field_name}' has reserved default value "
                f"{immediate.default_value}"
            )


_MATRIX_HIGH_HALF_SELECT_FIELD_IDS = frozenset(
    (
        amdgpu_encoding_field_id("OPSEL_HI"),
        amdgpu_encoding_field_id("OP_SEL_HI"),
    )
)


def _validate_matrix_high_half_select_fields(
    descriptor_set: DescriptorSet,
) -> None:
    for descriptor in descriptor_set.descriptors:
        if descriptor.encoding_format_id != AMDGPU_ENCODING_FORMAT_VOP3P or not (
            descriptor.semantic_tag.startswith("matrix.wmma.")
            or descriptor.semantic_tag.startswith("matrix.swmmac.")
        ):
            continue
        selectors = tuple(
            field_value
            for field_value in descriptor.encoding_field_values
            if field_value.encoding_field_id in _MATRIX_HIGH_HALF_SELECT_FIELD_IDS
        )
        if len(selectors) != 1:
            raise ValueError(
                f"AMDGPU VOP3P matrix descriptor '{descriptor.key}' must define "
                f"exactly one high-half selector; found {len(selectors)}"
            )
        if selectors[0].value not in (0x3, 0x7):
            raise ValueError(
                f"AMDGPU VOP3P matrix descriptor '{descriptor.key}' has "
                f"non-canonical high-half selector {selectors[0].value}"
            )


_NAMED_NATIVE_ASM_IMMEDIATE_FORMAT_IDS = frozenset(
    (
        AMDGPU_NATIVE_ASM_IMMEDIATE_FORMAT_NAMED_BIT_LIST,
        AMDGPU_NATIVE_ASM_IMMEDIATE_FORMAT_NAMED_FLAG,
        AMDGPU_NATIVE_ASM_IMMEDIATE_FORMAT_NAMED_I64,
        AMDGPU_NATIVE_ASM_IMMEDIATE_FORMAT_REQUIRED_NAMED_I64,
        AMDGPU_NATIVE_ASM_IMMEDIATE_FORMAT_GFX12_SCOPE,
        AMDGPU_NATIVE_ASM_IMMEDIATE_FORMAT_GFX12_LOAD_TEMPORAL,
    )
)


def _validate_native_asm_values(descriptor_set: DescriptorSet) -> None:
    for descriptor in descriptor_set.descriptors:
        immediate_by_name = {
            immediate.field_name: immediate for immediate in descriptor.immediates
        }
        for form in descriptor.asm_forms:
            saw_named_modifier = False
            for value in form.native_assembly_values:
                is_named_modifier = (
                    value.kind is NativeAsmValueKind.MODIFIER_LITERAL
                    or (
                        value.kind is NativeAsmValueKind.IMMEDIATE_TARGET_FORMAT
                        and value.target_format_id
                        in _NAMED_NATIVE_ASM_IMMEDIATE_FORMAT_IDS
                    )
                )
                if not is_named_modifier:
                    if saw_named_modifier:
                        raise ValueError(
                            f"AMDGPU descriptor '{descriptor.key}' native asm "
                            "form has a positional value after a named modifier"
                        )
                    continue
                saw_named_modifier = True
                if value.kind is NativeAsmValueKind.MODIFIER_LITERAL:
                    continue
                immediate = immediate_by_name.get(value.field_name)
                if immediate is None:
                    raise ValueError(
                        f"AMDGPU descriptor '{descriptor.key}' native asm "
                        f"modifier references unknown immediate '{value.field_name}'"
                    )
                if (
                    value.target_format_id
                    != AMDGPU_NATIVE_ASM_IMMEDIATE_FORMAT_REQUIRED_NAMED_I64
                    and ImmediateFlag.DEFAULT_VALUE not in immediate.flags
                ):
                    raise ValueError(
                        f"AMDGPU descriptor '{descriptor.key}' native asm "
                        f"modifier '{value.field_name}' has no default value"
                    )
                if value.literal is None or value.literal == "":
                    raise ValueError(
                        f"AMDGPU descriptor '{descriptor.key}' native asm "
                        f"modifier '{value.field_name}' has no spelling"
                    )
                if value.target_format_id == (
                    AMDGPU_NATIVE_ASM_IMMEDIATE_FORMAT_NAMED_BIT_LIST
                ):
                    if value.bit_width <= 0 or value.bit_width >= 64:
                        raise ValueError(
                            f"AMDGPU descriptor '{descriptor.key}' native asm "
                            f"bit-list modifier '{value.field_name}' has invalid width"
                        )
                elif value.bit_width != 0:
                    raise ValueError(
                        f"AMDGPU descriptor '{descriptor.key}' native asm "
                        f"modifier '{value.field_name}' unexpectedly has a bit width"
                    )
                if value.target_format_id == (
                    AMDGPU_NATIVE_ASM_IMMEDIATE_FORMAT_NAMED_FLAG
                ) and (
                    immediate.kind is not ImmediateKind.UNSIGNED
                    or immediate.default_value != 0
                    or immediate.unsigned_max != 1
                ):
                    raise ValueError(
                        f"AMDGPU descriptor '{descriptor.key}' native asm flag "
                        f"'{value.field_name}' is not a zero-default boolean"
                    )


def _operand_has_vgpr_alt(operand: Operand) -> bool:
    return any(reg_alt.reg_class == _REG_VGPR for reg_alt in operand.reg_alts)


_MATRIX_COEXECUTION_SOURCE_INFOS_BY_SOURCE = {
    info.source: info for info in AMDGPU_MATRIX_COEXECUTION_SOURCE_INFOS
}


_NATIVE_PACKED_BF16_ARITHMETIC_DESCRIPTOR_KEYS = frozenset(
    (
        "amdgpu.v_pk_add_bf16",
        "amdgpu.v_pk_mul_bf16",
        "amdgpu.v_pk_fma_bf16",
    )
)

_NATIVE_SCALAR_FLOAT_ARITHMETIC_DESCRIPTOR_KEYS = frozenset(
    f"amdgpu.s_{operation}_f{bit_width}"
    for bit_width in (16, 32)
    for operation in (
        "add",
        "sub",
        "mul",
        "min",
        "max",
        "ceil",
        "floor",
        "rndne",
        "trunc",
    )
)

_NATIVE_SCALAR_FLOAT_CONVERSION_DESCRIPTOR_KEYS = frozenset(
    f"amdgpu.s_cvt_{operation}"
    for operation in (
        "f16_f32",
        "f32_f16",
        "f32_i32",
        "f32_u32",
        "hi_f32_f16",
        "i32_f32",
        "pk_rtz_f16_f32",
        "u32_f32",
    )
)

_NATIVE_SCALAR_FLOAT_COMPARE_DESCRIPTOR_KEYS = frozenset(
    f"amdgpu.s_cmp_{predicate}_f{bit_width}"
    for bit_width in (16, 32)
    for predicate in (
        "oeq",
        "ogt",
        "oge",
        "olt",
        "ole",
        "one",
        "ord",
        "ueq",
        "ugt",
        "uge",
        "ult",
        "ule",
        "une",
        "uno",
    )
)


def _validate_descriptor_family_capability(
    *,
    generator_target: str,
    descriptor_keys: frozenset[str],
    family_descriptor_keys: frozenset[str],
    capability_name: str,
    declares_capability: bool,
) -> None:
    present_keys = descriptor_keys & family_descriptor_keys
    if present_keys and present_keys != family_descriptor_keys:
        missing_keys = sorted(family_descriptor_keys - present_keys)
        raise ValueError(
            f"AMDGPU descriptor target '{generator_target}' has an incomplete "
            f"{capability_name} family; missing: {', '.join(missing_keys)}"
        )
    has_capability = present_keys == family_descriptor_keys
    if declares_capability != has_capability:
        declared_state = "declares" if declares_capability else "omits"
        actual_state = "provides" if has_capability else "omits"
        raise ValueError(
            f"AMDGPU descriptor target '{generator_target}' {declared_state} "
            f"{capability_name} capability but its descriptor set "
            f"{actual_state} the instruction family"
        )


def _validate_descriptor_set_info_capabilities(
    generator_target: str, descriptor_set: DescriptorSet
) -> None:
    info = amdgpu_descriptor_set_info_by_generator_target(generator_target)
    descriptor_keys = frozenset(
        descriptor.key for descriptor in descriptor_set.descriptors
    )
    _validate_descriptor_family_capability(
        generator_target=generator_target,
        descriptor_keys=descriptor_keys,
        family_descriptor_keys=_NATIVE_PACKED_BF16_ARITHMETIC_DESCRIPTOR_KEYS,
        capability_name="native packed BF16 arithmetic",
        declares_capability=bool(
            info.flags & AMDGPU_DESCRIPTOR_SET_INFO_FLAG_NATIVE_PACKED_BF16_ARITHMETIC
        ),
    )
    _validate_descriptor_family_capability(
        generator_target=generator_target,
        descriptor_keys=descriptor_keys,
        family_descriptor_keys=_NATIVE_SCALAR_FLOAT_COMPARE_DESCRIPTOR_KEYS,
        capability_name="native scalar floating-point comparison",
        declares_capability=bool(
            info.flags & AMDGPU_DESCRIPTOR_SET_INFO_FLAG_NATIVE_SCALAR_FLOAT_COMPARE
        ),
    )
    _validate_descriptor_family_capability(
        generator_target=generator_target,
        descriptor_keys=descriptor_keys,
        family_descriptor_keys=_NATIVE_SCALAR_FLOAT_CONVERSION_DESCRIPTOR_KEYS,
        capability_name="native scalar floating-point conversion",
        declares_capability=bool(
            info.flags & AMDGPU_DESCRIPTOR_SET_INFO_FLAG_NATIVE_SCALAR_FLOAT_CONVERSION
        ),
    )
    _validate_descriptor_family_capability(
        generator_target=generator_target,
        descriptor_keys=descriptor_keys,
        family_descriptor_keys=_NATIVE_SCALAR_FLOAT_ARITHMETIC_DESCRIPTOR_KEYS,
        capability_name="native scalar floating-point arithmetic",
        declares_capability=bool(
            info.flags & AMDGPU_DESCRIPTOR_SET_INFO_FLAG_NATIVE_SCALAR_FLOAT_ARITHMETIC
        ),
    )


def _validate_matrix_coexecution_profile_coverage(
    generator_target: str,
    descriptor_set: DescriptorSet,
) -> None:
    descriptor_set_info = amdgpu_descriptor_set_info_by_generator_target(
        generator_target
    )
    processors_by_name = {
        processor.processor: processor for processor in AMDGPU_PROCESSOR_INFOS
    }
    targets = tuple(
        (target, processors_by_name[target.processor])
        for target in AMDGPU_TARGET_INFOS
        if amdgpu_target_descriptor_set_key(
            target, processors_by_name[target.processor]
        )
        == descriptor_set_info.key
    )
    if not targets:
        raise ValueError(
            f"AMDGPU descriptor target '{generator_target}' has no compiler target "
            "using its descriptor set"
        )

    resources = {resource.name: resource for resource in descriptor_set.resources}
    schedule_classes = {
        schedule_class.name: schedule_class
        for schedule_class in descriptor_set.schedule_classes
    }
    requirements: dict[tuple[str, int], list[str]] = {}
    for descriptor in descriptor_set.descriptors:
        schedule_class = schedule_classes[descriptor.schedule_class]
        source_resources = tuple(
            resources[issue_use.resource]
            for issue_use in schedule_class.issue_uses
            if ResourceFlag.MATRIX_COEXECUTION_SOURCE
            in resources[issue_use.resource].flags
        )
        if not source_resources:
            continue
        if len(source_resources) != 1:
            raise ValueError(
                f"AMDGPU descriptor target '{generator_target}' descriptor "
                f"'{descriptor.key}' must issue on one matrix coexecution "
                "resource"
            )
        source_resource = source_resources[0]
        if ResourceFlag.VECTOR_ISSUE not in source_resource.flags:
            raise ValueError(
                f"AMDGPU descriptor target '{generator_target}' descriptor "
                f"'{descriptor.key}' matrix coexecution resource "
                f"'{source_resource.name}' is not a vector issue"
            )
        if schedule_class.latency_kind is LatencyKind.VARIABLE:
            raise ValueError(
                f"AMDGPU descriptor target '{generator_target}' descriptor "
                f"'{descriptor.key}' matrix coexecution latency cannot select "
                "a stable release rule"
            )
        source_families = tuple(
            source
            for instruction_class, source in (
                (
                    InstructionClass.WMMA,
                    AMDGPU_MATRIX_COEXECUTION_SOURCE_WMMA,
                ),
                (
                    InstructionClass.SWMMAC,
                    AMDGPU_MATRIX_COEXECUTION_SOURCE_SWMMAC,
                ),
            )
            if instruction_class in descriptor.instruction_classes
        )
        if len(source_families) != 1:
            raise ValueError(
                f"AMDGPU descriptor target '{generator_target}' descriptor "
                f"'{descriptor.key}' must belong to one matrix source family"
            )
        source_family = source_families[0]
        source_info = _MATRIX_COEXECUTION_SOURCE_INFOS_BY_SOURCE[source_family]
        source_operand_end = (
            source_info.source_operand_start + source_info.source_operand_count
        )
        required_operand_count = max(
            source_info.result_operand_index + 1, source_operand_end
        )
        if len(descriptor.operands) < required_operand_count:
            raise ValueError(
                f"AMDGPU descriptor target '{generator_target}' descriptor "
                f"'{descriptor.key}' does not provide the {source_family} "
                "coexecution operand layout"
            )
        operand_layout = (
            (
                source_info.result_operand_index,
                OperandRole.RESULT,
                amdgpu_encoding_field_id("VDST"),
            ),
            *(
                (
                    source_info.source_operand_start + source_index,
                    OperandRole.OPERAND,
                    amdgpu_encoding_field_id(f"SRC{source_index}"),
                )
                for source_index in range(source_info.source_operand_count)
            ),
        )
        for operand_index, expected_role, expected_encoding_field_id in operand_layout:
            operand = descriptor.operands[operand_index]
            if operand.role is not expected_role:
                raise ValueError(
                    f"AMDGPU descriptor target '{generator_target}' descriptor "
                    f"'{descriptor.key}' coexecution operand {operand_index} "
                    f"has role {operand.role.name}; expected {expected_role.name}"
                )
            if operand.encoding_field_id != expected_encoding_field_id:
                raise ValueError(
                    f"AMDGPU descriptor target '{generator_target}' descriptor "
                    f"'{descriptor.key}' coexecution operand {operand_index} "
                    "does not map the expected encoding field"
                )
            if not _operand_has_vgpr_alt(operand):
                raise ValueError(
                    f"AMDGPU descriptor target '{generator_target}' descriptor "
                    f"'{descriptor.key}' coexecution operand {operand_index} "
                    "cannot use a VGPR"
                )
        requirement = (source_family, schedule_class.latency_cycles)
        requirements.setdefault(requirement, []).append(descriptor.key)

    for target, processor in targets:
        profile = processor.features.matrix_coexecution
        if not requirements:
            if profile != AMDGPU_MATRIX_COEXECUTION_PROFILE_NONE:
                raise ValueError(
                    f"AMDGPU target '{target.target}' selects matrix "
                    f"coexecution profile '{profile}' without any source "
                    "descriptors"
                )
            continue
        if profile == AMDGPU_MATRIX_COEXECUTION_PROFILE_NONE:
            raise ValueError(
                f"AMDGPU target '{target.target}' has matrix "
                "coexecution source descriptors but no release profile"
            )
        rules = AMDGPU_MATRIX_COEXECUTION_RULES_BY_PROFILE.get(profile)
        if rules is None:
            raise ValueError(
                f"AMDGPU target '{target.target}' selects unknown "
                f"matrix coexecution profile '{profile}'"
            )
        covered_requirements = {(rule.source, rule.latency_cycles) for rule in rules}
        for requirement, descriptor_keys in requirements.items():
            if requirement in covered_requirements:
                continue
            source, latency_cycles = requirement
            examples = ", ".join(descriptor_keys[:3])
            raise ValueError(
                f"AMDGPU target '{target.target}' matrix "
                f"coexecution profile '{profile}' has no {source} "
                f"{latency_cycles}-cycle fallback rule required by "
                f"descriptor(s): {examples}"
            )


@cache
def amdgpu_descriptor_ref_keys() -> tuple[str, ...]:
    """Returns descriptor keys known to the AMDGPU target family."""

    keys = _amdgpu_descriptor_ref_key_set()
    for builder in _AMDGPU_CORE_DESCRIPTOR_SET_BUILDERS.values():
        keys.update(descriptor.key for descriptor in builder.extra_descriptors)
    return tuple(sorted(keys))


def amdgpu_descriptor_id_keys() -> tuple[str, ...]:
    """Returns descriptor keys that still need stable-ID compatibility refs."""

    return amdgpu_descriptor_ref_keys()


def amdgpu_immediate_encoding_id_items() -> tuple[tuple[str, int], ...]:
    """Returns target-owned immediate encoding IDs used by AMDGPU descriptors."""

    return (
        ("address_offset_byte", _ADDRESS_OFFSET_BYTE_ENCODING_ID),
        ("address_offset_dword", _ADDRESS_OFFSET_DWORD_ENCODING_ID),
        ("address_offset_qword", _ADDRESS_OFFSET_QWORD_ENCODING_ID),
        ("address_offset_dword_stride64", _ADDRESS_OFFSET_DWORD_STRIDE64_ENCODING_ID),
        ("address_offset_qword_stride64", _ADDRESS_OFFSET_QWORD_STRIDE64_ENCODING_ID),
        ("address_offset_ds16", _ADDRESS_OFFSET_DS16_ENCODING_ID),
        ("source_inline_u32", _SOURCE_INLINE_U32_ENCODING_ID),
        ("source_inline_f32", _SOURCE_INLINE_F32_ENCODING_ID),
        ("wait_counter_vmem", _WAIT_COUNTER_VMEM_ENCODING_ID),
        ("wait_counter_lgkm", _WAIT_COUNTER_LGKM_ENCODING_ID),
        ("wait_counter_vmem_load", _WAIT_COUNTER_VMEM_LOAD_ENCODING_ID),
        ("wait_counter_vmem_store", _WAIT_COUNTER_VMEM_STORE_ENCODING_ID),
        ("wait_counter_lds", _WAIT_COUNTER_LDS_ENCODING_ID),
        ("wait_counter_smem", _WAIT_COUNTER_SMEM_ENCODING_ID),
        ("wait_counter_alu", _WAIT_COUNTER_ALU_ENCODING_ID),
        ("wait_counter_tensor", _WAIT_COUNTER_TENSOR_ENCODING_ID),
        ("wait_counter_async", _WAIT_COUNTER_ASYNC_ENCODING_ID),
        ("wait_counter_x", _WAIT_COUNTER_X_ENCODING_ID),
    )


def amdgpu_common_reg_class_ids() -> tuple[tuple[str, int], ...]:
    """Returns descriptor-set-local register-class IDs shared by all AMDGPU sets."""

    result: list[tuple[str, int]] = []
    for reg_class_name in (_REG_SGPR, _REG_VGPR, _REG_SCC, _REG_EXEC, _REG_VCC):
        expected_reg_class_id: int | None = None
        for descriptor_set in _amdgpu_core_descriptor_set_bases():
            reg_class_id = next(
                i
                for i, reg_class in enumerate(descriptor_set.reg_classes)
                if reg_class.name == reg_class_name
            )
            if expected_reg_class_id is None:
                expected_reg_class_id = reg_class_id
            elif expected_reg_class_id != reg_class_id:
                raise ValueError(
                    f"AMDGPU common register class '{reg_class_name}' has "
                    "inconsistent descriptor-set-local IDs"
                )
        if expected_reg_class_id is None:
            raise ValueError(
                f"AMDGPU common register class '{reg_class_name}' is missing"
            )
        result.append((reg_class_name, expected_reg_class_id))
    return tuple(result)


def _with_overlay_descriptors(
    base: DescriptorSet,
    spec: AmdgpuIsaFactSource,
    overlay_descriptors: tuple[Descriptor, ...],
    extra_descriptors: tuple[Descriptor, ...] = (),
) -> DescriptorSet:
    manual_descriptors = _manual_scalar_descriptors(spec)
    descriptor_set = replace(
        base,
        descriptors=_categorize_amdgpu_descriptors(
            (
                manual_descriptors[0],
                *overlay_descriptors,
                *extra_descriptors,
                *manual_descriptors[1:],
                *_hal_buffer_descriptor_pseudos(),
                *base.descriptors,
            )
        ),
    )
    _validate_address_immediate_units(descriptor_set)
    return descriptor_set


def _operand_is_explicit_register(operand: Operand) -> bool:
    return OperandFlag.IMPLICIT not in operand.flags and operand.role in (
        OperandRole.RESULT,
        OperandRole.OPERAND,
        OperandRole.OPERAND_RESULT,
        OperandRole.PREDICATE,
        OperandRole.RESOURCE,
    )


def _gfx125x_operand_encoding_field_name(operand: Operand) -> str | None:
    if operand.encoding_field_id == 0:
        return None
    return amdgpu_encoding_field_name(operand.encoding_field_id)


def _gfx125x_operand_vgpr_msb_slot(
    descriptor: Descriptor, operand: Operand
) -> AmdgpuVgprMsbSlot:
    field_name = _gfx125x_operand_encoding_field_name(operand)
    if field_name is None:
        return AmdgpuVgprMsbSlot.NONE
    return amdgpu_gfx125x_vgpr_msb_slot(
        descriptor.key, descriptor.encoding_format_id, field_name
    )


def _gfx125x_operand_uses_vgpr_msb_state(
    descriptor: Descriptor, operand: Operand
) -> bool:
    return _gfx125x_operand_vgpr_msb_slot(descriptor, operand) != AmdgpuVgprMsbSlot.NONE


def _with_gfx125x_operand_address_state(
    descriptor: Descriptor, operand: Operand
) -> Operand:
    if not _operand_is_explicit_register(operand) or not _operand_has_vgpr_alt(operand):
        return operand
    address_state_slot = _gfx125x_operand_vgpr_msb_slot(descriptor, operand)
    if operand.address_map_kind is OperandAddressMapKind.LOW_SUBSET:
        if (
            address_state_slot == AmdgpuVgprMsbSlot.NONE
            or operand.address_state_slot != 0
        ):
            return operand
        return replace(operand, address_state_slot=int(address_state_slot))
    if operand.address_map_kind is not OperandAddressMapKind.DIRECT:
        return operand
    if address_state_slot != AmdgpuVgprMsbSlot.NONE:
        return replace(
            operand,
            address_map_kind=OperandAddressMapKind.TARGET_STATE,
            addressable_unit_count=AMDGPU_GFX125X_VGPR_MSB_WINDOW_SIZE,
            address_state_slot=int(address_state_slot),
        )
    return replace(
        operand,
        address_map_kind=OperandAddressMapKind.LOW_SUBSET,
        addressable_unit_count=AMDGPU_GFX125X_VGPR_MSB_WINDOW_SIZE,
    )


def _with_gfx125x_vgpr_msb_address_state(descriptor: Descriptor) -> Descriptor:
    operands: list[Operand] = []
    operands_changed = False
    has_address_state = False
    for operand in descriptor.operands:
        projected_operand = _with_gfx125x_operand_address_state(descriptor, operand)
        operands.append(projected_operand)
        operands_changed |= projected_operand is not operand
        has_address_state |= projected_operand.address_state_slot != 0

    updated_descriptor = descriptor
    if operands_changed:
        updated_descriptor = replace(descriptor, operands=tuple(operands))
    if has_address_state:
        updated_descriptor = _with_mode_state_read(updated_descriptor)
    return updated_descriptor


def _with_gfx125x_vgpr_msb_address_states(
    descriptor_set: DescriptorSet,
) -> DescriptorSet:
    descriptors: list[Descriptor] = []
    descriptors_changed = False
    for descriptor in descriptor_set.descriptors:
        projected_descriptor = _with_gfx125x_vgpr_msb_address_state(descriptor)
        descriptors.append(projected_descriptor)
        descriptors_changed |= projected_descriptor is not descriptor
    if descriptors_changed:
        descriptor_set = replace(descriptor_set, descriptors=tuple(descriptors))
    _validate_gfx125x_vgpr_msb_address_state(descriptor_set)
    return descriptor_set


def _descriptor_writes_mode_state(descriptor: Descriptor) -> bool:
    return any(
        OperandFlag.STATE_WRITE in operand.flags
        and len(operand.reg_alts) == 1
        and operand.reg_alts[0].reg_class == _REG_MODE
        for operand in descriptor.operands
    )


def _descriptor_tied_operand_roots(descriptor: Descriptor) -> tuple[int, ...]:
    roots = list(range(len(descriptor.operands)))

    def find_root(operand_index: int) -> int:
        while roots[operand_index] != operand_index:
            roots[operand_index] = roots[roots[operand_index]]
            operand_index = roots[operand_index]
        return operand_index

    for constraint in descriptor.constraints:
        if constraint.kind is not ConstraintKind.TIED:
            continue
        if constraint.rhs_operand_index is None:
            raise ValueError(
                f"gfx125x descriptor '{descriptor.key}' has a tied constraint "
                "without a right-hand operand"
            )
        lhs_root = find_root(constraint.lhs_operand_index)
        rhs_root = find_root(constraint.rhs_operand_index)
        roots[rhs_root] = lhs_root
    return tuple(find_root(i) for i in range(len(roots)))


def _validate_gfx125x_vgpr_msb_address_state(descriptor_set: DescriptorSet) -> None:
    vgpr_reg_class = next(
        (
            reg_class
            for reg_class in descriptor_set.reg_classes
            if reg_class.name == _REG_VGPR
        ),
        None,
    )
    if vgpr_reg_class is not None:
        maximum_vgpr_count = (
            AMDGPU_GFX125X_VGPR_MSB_WINDOW_SIZE * AMDGPU_GFX125X_VGPR_MSB_BANK_COUNT
        )
        fixed_vgpr_end = (
            vgpr_reg_class.fixed_location_base + vgpr_reg_class.fixed_location_count
        )
        if (
            vgpr_reg_class.allocatable_count > maximum_vgpr_count
            or fixed_vgpr_end > maximum_vgpr_count
        ):
            raise ValueError(
                f"gfx125x VGPR register class exceeds the "
                f"{AMDGPU_GFX125X_VGPR_MSB_BANK_COUNT}-bank "
                f"S_SET_VGPR_MSB capacity of {maximum_vgpr_count} registers"
            )

    descriptors_by_key = {
        descriptor.key: descriptor for descriptor in descriptor_set.descriptors
    }
    try:
        mode_descriptor = descriptors_by_key["amdgpu.s_set_vgpr_msb"]
    except KeyError as exc:
        raise ValueError(
            "gfx125x VGPR-MSB address-state operands require 'amdgpu.s_set_vgpr_msb'"
        ) from exc
    if not _descriptor_writes_mode_state(mode_descriptor):
        raise ValueError(
            "gfx125x descriptor 'amdgpu.s_set_vgpr_msb' must write MODE state"
        )
    for descriptor in descriptor_set.descriptors:
        has_address_state_operand = False
        tied_operand_roots = _descriptor_tied_operand_roots(descriptor)
        address_state_slot_operands: dict[int, int] = {}
        for operand_index, operand in enumerate(descriptor.operands):
            if (
                operand.address_map_kind is not OperandAddressMapKind.TARGET_STATE
                and operand.address_state_slot == 0
            ):
                continue
            has_address_state_operand = True
            expected_slot = _gfx125x_operand_vgpr_msb_slot(descriptor, operand)
            if expected_slot == AmdgpuVgprMsbSlot.NONE:
                raise ValueError(
                    f"gfx125x descriptor '{descriptor.key}' marks operand "
                    f"'{operand.field_name}' as VGPR-MSB address-state, but "
                    "the operand encoding field has no S_SET_VGPR_MSB slot"
                )
            if operand.address_state_slot != int(expected_slot):
                raise ValueError(
                    f"gfx125x descriptor '{descriptor.key}' marks operand "
                    f"'{operand.field_name}' with S_SET_VGPR_MSB slot "
                    f"{operand.address_state_slot}; expected {int(expected_slot)}"
                )
            previous_operand_index = address_state_slot_operands.get(
                operand.address_state_slot
            )
            if (
                previous_operand_index is not None
                and tied_operand_roots[previous_operand_index]
                != tied_operand_roots[operand_index]
            ):
                raise ValueError(
                    f"gfx125x descriptor '{descriptor.key}' assigns multiple "
                    f"untied operands to S_SET_VGPR_MSB slot "
                    f"{operand.address_state_slot}"
                )
            address_state_slot_operands.setdefault(
                operand.address_state_slot, operand_index
            )
            if operand.address_map_kind is OperandAddressMapKind.TARGET_STATE:
                if (
                    operand.addressable_unit_count
                    != AMDGPU_GFX125X_VGPR_MSB_WINDOW_SIZE
                ):
                    raise ValueError(
                        f"gfx125x descriptor '{descriptor.key}' marks operand "
                        f"'{operand.field_name}' as VGPR-MSB target-state with "
                        f"{operand.addressable_unit_count} addressable units; "
                        f"expected {AMDGPU_GFX125X_VGPR_MSB_WINDOW_SIZE}"
                    )
            elif operand.address_map_kind is OperandAddressMapKind.LOW_SUBSET:
                if not (
                    0
                    < operand.addressable_unit_count
                    <= AMDGPU_GFX125X_VGPR_MSB_WINDOW_SIZE
                ):
                    raise ValueError(
                        f"gfx125x descriptor '{descriptor.key}' marks operand "
                        f"'{operand.field_name}' as VGPR-MSB low-subset with "
                        f"{operand.addressable_unit_count} addressable units; "
                        f"expected between 1 and "
                        f"{AMDGPU_GFX125X_VGPR_MSB_WINDOW_SIZE}"
                    )
            else:
                raise ValueError(
                    f"gfx125x descriptor '{descriptor.key}' marks operand "
                    f"'{operand.field_name}' with S_SET_VGPR_MSB slot "
                    f"{operand.address_state_slot} but uses the "
                    f"'{operand.address_map_kind.name.lower()}' address map"
                )
        if has_address_state_operand and not any(
            _is_mode_state_read(operand) for operand in descriptor.operands
        ):
            raise ValueError(
                f"gfx125x descriptor '{descriptor.key}' uses VGPR-MSB "
                "address-state operands but does not read MODE state"
            )


_AMDGPU_WAIT_COUNTER_MASKS = {
    _COUNTER_VMEM_LOAD: 1 << 0,
    _COUNTER_VMEM_STORE: 1 << 1,
    _COUNTER_LDS: 1 << 2,
    _COUNTER_SMEM: 1 << 3,
    _COUNTER_ALU: 1 << 4,
    _COUNTER_TENSOR: 1 << 5,
    _COUNTER_ASYNC: 1 << 6,
    _COUNTER_X: 1 << 7,
}

_AMDGPU_READ_COUNTER_MASK = (
    _AMDGPU_WAIT_COUNTER_MASKS[_COUNTER_VMEM_LOAD]
    | _AMDGPU_WAIT_COUNTER_MASKS[_COUNTER_LDS]
    | _AMDGPU_WAIT_COUNTER_MASKS[_COUNTER_SMEM]
    | _AMDGPU_WAIT_COUNTER_MASKS[_COUNTER_TENSOR]
    | _AMDGPU_WAIT_COUNTER_MASKS[_COUNTER_ASYNC]
)

_AMDGPU_WRITE_COUNTER_MASK = (
    _AMDGPU_WAIT_COUNTER_MASKS[_COUNTER_VMEM_STORE]
    | _AMDGPU_WAIT_COUNTER_MASKS[_COUNTER_LDS]
    | _AMDGPU_WAIT_COUNTER_MASKS[_COUNTER_SMEM]
    | _AMDGPU_WAIT_COUNTER_MASKS[_COUNTER_TENSOR]
    | _AMDGPU_WAIT_COUNTER_MASKS[_COUNTER_ASYNC]
)

_AMDGPU_STORAGE_LEASE_MEMORY_SPACES = frozenset(
    (
        MemorySpace.GENERIC,
        MemorySpace.GLOBAL,
        MemorySpace.STACK,
        MemorySpace.WORKGROUP,
    )
)

_AMDGPU_WAIT_COUNTER_PROGRESS_CLASS_NAMES = {
    _COUNTER_VMEM_LOAD: "amdgpu.vmem_load",
    _COUNTER_VMEM_STORE: "amdgpu.vmem_store",
    _COUNTER_LDS: "amdgpu.lds",
    _COUNTER_SMEM: "amdgpu.smem",
    _COUNTER_ALU: "amdgpu.alu",
    _COUNTER_TENSOR: "amdgpu.tensor",
    _COUNTER_ASYNC: "amdgpu.async",
    _COUNTER_X: "amdgpu.x",
}

_GFX125X_XCNT_SCHEDULE_CLASSES = frozenset(
    (
        _SCHEDULE_SMEM_LOAD,
        _SCHEDULE_SMEM_STORE,
        _SCHEDULE_VMEM_LOAD,
        _SCHEDULE_VMEM_LOAD_LDS,
        _SCHEDULE_VMEM_STORE,
        _SCHEDULE_VMEM_ATOMIC_RETURN,
        _SCHEDULE_VMEM_ATOMIC_NO_RETURN,
        _SCHEDULE_CLUSTER_LOAD_LDS,
    )
)

_AMDGPU_WAIT_PLAN_RESIDUAL_ACTION_WAIT_PACKET = 1
_AMDGPU_WAIT_PLAN_RESIDUAL_ACTION_WAIT_PACKET_NAME = "amdgpu.wait_packet"
_AMDGPU_WAIT_PLAN_REASON_READ_RESULT_REUSE = 5
_AMDGPU_WAIT_PLAN_REASON_READ_RESULT_REUSE_NAME = "amdgpu.read_result_reuse"
_AMDGPU_WAIT_PLAN_REASON_MEMORY_SOURCE_REUSE = 10
_AMDGPU_WAIT_PLAN_REASON_MEMORY_SOURCE_REUSE_NAME = "amdgpu.memory_source_reuse"
_AMDGPU_STORAGE_LEASE_FLAGS = (
    StorageLeaseFlag.STARTS_AT_ISSUE,
    StorageLeaseFlag.MAY_CARRY_ACROSS_BOUNDARY,
)
_AMDGPU_PRESSURE_STORAGE_LEASE_FLAGS = (
    StorageLeaseFlag.STARTS_AT_ISSUE,
    StorageLeaseFlag.RELEASE_BEFORE_BOUNDARY,
    StorageLeaseFlag.RELEASE_FOR_PRESSURE,
)


def _amdgpu_wait_counter_mask(counter_id: int) -> int:
    try:
        return _AMDGPU_WAIT_COUNTER_MASKS[counter_id]
    except KeyError as exc:
        raise ValueError(f"unknown AMDGPU wait counter id {counter_id}") from exc


def _amdgpu_descriptor_hazard_counter_mask(
    schedule_classes: dict[str, ScheduleClass],
    descriptor: Descriptor,
) -> int:
    schedule_class = schedule_classes[descriptor.schedule_class]
    counter_mask = 0
    for hazard in schedule_class.hazards:
        if hazard.kind is not HazardKind.WAIT_COUNTER:
            continue
        if hazard.counter_id is None:
            raise ValueError(
                f"AMDGPU descriptor '{descriptor.key}' schedule class "
                f"'{schedule_class.name}' has a wait-counter hazard without a "
                "counter id"
            )
        counter_mask |= _amdgpu_wait_counter_mask(hazard.counter_id)
    return counter_mask


def _amdgpu_storage_lease_effect_is_dependency_memory(effect: Effect) -> bool:
    return (
        EffectFlag.DEPENDENCY in effect.flags
        and effect.memory_space in _AMDGPU_STORAGE_LEASE_MEMORY_SPACES
    )


def _amdgpu_effect_counter_mask(
    descriptor: Descriptor,
    effect: Effect,
    default_counter_mask: int,
    allowed_counter_mask: int,
) -> int:
    if effect.counter_id == 0:
        counter_mask = default_counter_mask & allowed_counter_mask
        if counter_mask == 0:
            raise ValueError(
                f"AMDGPU dependency memory effect on descriptor "
                f"'{descriptor.key}' has no matching wait-counter hazard"
            )
        return counter_mask
    return _amdgpu_wait_counter_mask(effect.counter_id)


def _amdgpu_storage_lease_counter_masks(
    schedule_classes: dict[str, ScheduleClass],
    descriptor: Descriptor,
) -> tuple[int, int]:
    hazard_counter_mask = _amdgpu_descriptor_hazard_counter_mask(
        schedule_classes, descriptor
    )
    read_counter_mask = 0
    write_counter_mask = 0
    for effect in descriptor.effects:
        if not _amdgpu_storage_lease_effect_is_dependency_memory(effect):
            continue
        if effect.kind is EffectKind.READ:
            read_counter_mask |= _amdgpu_effect_counter_mask(
                descriptor,
                effect,
                hazard_counter_mask,
                _AMDGPU_READ_COUNTER_MASK,
            )
        elif effect.kind is EffectKind.WRITE:
            write_counter_mask |= _amdgpu_effect_counter_mask(
                descriptor,
                effect,
                hazard_counter_mask,
                _AMDGPU_WRITE_COUNTER_MASK,
            )
    return read_counter_mask, write_counter_mask


def _amdgpu_storage_lease(
    *,
    kind: StorageLeaseKind,
    attachment: StorageLeaseAttachment,
    attachment_index: int,
    unit_count: int,
    release_class_id: int,
    release_reason_id: int,
    release_reason_name: str,
    flags: tuple[StorageLeaseFlag, ...],
) -> StorageLease:
    return StorageLease(
        kind=kind,
        attachment=attachment,
        attachment_index=attachment_index,
        unit_offset=0,
        unit_count=unit_count,
        release_scope=StorageLeaseReleaseScope.PROGRESS_CLASS,
        release_class_id=release_class_id,
        release_class_name=_AMDGPU_WAIT_COUNTER_PROGRESS_CLASS_NAMES[release_class_id],
        release_action_id=_AMDGPU_WAIT_PLAN_RESIDUAL_ACTION_WAIT_PACKET,
        release_action_name=_AMDGPU_WAIT_PLAN_RESIDUAL_ACTION_WAIT_PACKET_NAME,
        release_reason_id=release_reason_id,
        release_reason_name=release_reason_name,
        flags=flags,
    )


def _amdgpu_operand_is_packet_input(operand: Operand) -> bool:
    return operand.role in (
        OperandRole.OPERAND,
        OperandRole.PREDICATE,
        OperandRole.RESOURCE,
    )


def _amdgpu_operand_accepts_sgpr(operand: Operand) -> bool:
    return any(
        reg_alt.reg_class == _REG_SGPR
        and RegClassAltFlag.IMMEDIATE not in reg_alt.flags
        for reg_alt in operand.reg_alts
    )


def _amdgpu_append_memory_source_leases(
    storage_leases: list[StorageLease],
    operand: Operand,
    packet_operand_index: int,
    counter_mask: int,
) -> None:
    for counter_id, counter_bit in _AMDGPU_WAIT_COUNTER_MASKS.items():
        if (counter_mask & counter_bit) == 0:
            continue
        storage_leases.append(
            _amdgpu_storage_lease(
                kind=StorageLeaseKind.SOURCE_READ,
                attachment=StorageLeaseAttachment.OPERAND,
                attachment_index=packet_operand_index,
                unit_count=operand.unit_count,
                release_class_id=counter_id,
                release_reason_id=_AMDGPU_WAIT_PLAN_REASON_MEMORY_SOURCE_REUSE,
                release_reason_name=_AMDGPU_WAIT_PLAN_REASON_MEMORY_SOURCE_REUSE_NAME,
                flags=_AMDGPU_STORAGE_LEASE_FLAGS,
            )
        )


def _amdgpu_descriptor_storage_leases(
    schedule_classes: dict[str, ScheduleClass],
    descriptor: Descriptor,
    *,
    enable_gfx125x_xcnt: bool,
) -> tuple[StorageLease, ...]:
    if descriptor.storage_leases:
        raise ValueError(
            f"AMDGPU descriptor '{descriptor.key}' already has storage lease rows"
        )
    read_counter_mask, write_counter_mask = _amdgpu_storage_lease_counter_masks(
        schedule_classes, descriptor
    )
    storage_leases: list[StorageLease] = []
    for result_index, result in enumerate(
        descriptor.operands[: _descriptor_result_count(descriptor)]
    ):
        if result.unit_count == 0:
            continue
        for counter_id, counter_mask in _AMDGPU_WAIT_COUNTER_MASKS.items():
            if (read_counter_mask & counter_mask) == 0:
                continue
            storage_leases.append(
                _amdgpu_storage_lease(
                    kind=StorageLeaseKind.RESULT_WRITE,
                    attachment=StorageLeaseAttachment.RESULT,
                    attachment_index=result_index,
                    unit_count=result.unit_count,
                    release_class_id=counter_id,
                    release_reason_id=_AMDGPU_WAIT_PLAN_REASON_READ_RESULT_REUSE,
                    release_reason_name=(
                        _AMDGPU_WAIT_PLAN_REASON_READ_RESULT_REUSE_NAME
                    ),
                    flags=_AMDGPU_PRESSURE_STORAGE_LEASE_FLAGS,
                )
            )
    memory_source_read_counter_mask = read_counter_mask & (
        _AMDGPU_WAIT_COUNTER_MASKS[_COUNTER_VMEM_LOAD]
        | _AMDGPU_WAIT_COUNTER_MASKS[_COUNTER_TENSOR]
    )
    memory_source_write_counter_mask = write_counter_mask & (
        _AMDGPU_WAIT_COUNTER_MASKS[_COUNTER_VMEM_STORE]
        | _AMDGPU_WAIT_COUNTER_MASKS[_COUNTER_TENSOR]
    )
    xcnt_source_counter_mask = (
        _AMDGPU_WAIT_COUNTER_MASKS[_COUNTER_X]
        if enable_gfx125x_xcnt
        and descriptor.schedule_class in _GFX125X_XCNT_SCHEDULE_CLASSES
        else 0
    )
    if (
        memory_source_read_counter_mask != 0
        or memory_source_write_counter_mask != 0
        or xcnt_source_counter_mask != 0
    ):
        packet_operand_index = 0
        for operand in descriptor.operands[_descriptor_result_count(descriptor) :]:
            if not _amdgpu_operand_is_packet_input(operand):
                continue
            current_packet_operand_index = packet_operand_index
            packet_operand_index += 1
            if operand.unit_count == 0:
                continue
            if xcnt_source_counter_mask != 0:
                _amdgpu_append_memory_source_leases(
                    storage_leases,
                    operand,
                    current_packet_operand_index,
                    xcnt_source_counter_mask,
                )
            if _amdgpu_operand_accepts_sgpr(operand):
                _amdgpu_append_memory_source_leases(
                    storage_leases,
                    operand,
                    current_packet_operand_index,
                    memory_source_read_counter_mask | memory_source_write_counter_mask,
                )
    return tuple(storage_leases)


def _descriptor_result_count(descriptor: Descriptor) -> int:
    result_count = 0
    for operand in descriptor.operands:
        if operand.role is not OperandRole.RESULT:
            break
        result_count += 1
    return result_count


def _with_storage_lease_rows(
    descriptor_set: DescriptorSet, *, enable_gfx125x_xcnt: bool = False
) -> DescriptorSet:
    schedule_classes = {
        schedule_class.name: schedule_class
        for schedule_class in descriptor_set.schedule_classes
    }
    descriptors: list[Descriptor] = []
    descriptors_changed = False
    for descriptor in descriptor_set.descriptors:
        storage_leases = _amdgpu_descriptor_storage_leases(
            schedule_classes,
            descriptor,
            enable_gfx125x_xcnt=enable_gfx125x_xcnt,
        )
        if storage_leases == descriptor.storage_leases:
            descriptors.append(descriptor)
            continue
        descriptors.append(replace(descriptor, storage_leases=storage_leases))
        descriptors_changed = True
    if not descriptors_changed:
        return descriptor_set
    return replace(descriptor_set, descriptors=tuple(descriptors))


_AMDGPU_SCHEDULE_INSTRUCTION_CLASSES = {
    _SCHEDULE_SMEM_LOAD: (InstructionClass.SCALAR_MEMORY,),
    _SCHEDULE_SMEM_STORE: (InstructionClass.SCALAR_MEMORY,),
    _SCHEDULE_VMEM_LOAD: (InstructionClass.GLOBAL_MEMORY,),
    _SCHEDULE_VMEM_LOAD_LDS: (
        InstructionClass.GLOBAL_MEMORY,
        InstructionClass.LOCAL_MEMORY,
        InstructionClass.LDSDMA,
    ),
    _SCHEDULE_VMEM_STORE: (InstructionClass.GLOBAL_MEMORY,),
    _SCHEDULE_VMEM_ATOMIC_RETURN: (InstructionClass.GLOBAL_MEMORY,),
    _SCHEDULE_VMEM_ATOMIC_NO_RETURN: (InstructionClass.GLOBAL_MEMORY,),
    _SCHEDULE_LDS_LOAD: (InstructionClass.LOCAL_MEMORY,),
    _SCHEDULE_LDS_STORE: (InstructionClass.LOCAL_MEMORY,),
    _SCHEDULE_LDS_ATOMIC: (InstructionClass.LOCAL_MEMORY,),
    _SCHEDULE_LDS_CROSSLANE: (InstructionClass.LOCAL_MEMORY,),
    _SCHEDULE_TENSOR_LOAD_LDS: (
        InstructionClass.GLOBAL_MEMORY,
        InstructionClass.LOCAL_MEMORY,
        InstructionClass.LDSDMA,
    ),
    _SCHEDULE_CLUSTER_LOAD_LDS: (
        InstructionClass.GLOBAL_MEMORY,
        InstructionClass.LOCAL_MEMORY,
        InstructionClass.LDSDMA,
    ),
    _SCHEDULE_MFMA: (InstructionClass.MFMA,),
    _SCHEDULE_WMMA: (InstructionClass.WMMA,),
    _SCHEDULE_WMMA_SCALE: (InstructionClass.WMMA,),
    _SCHEDULE_SWMMAC: (InstructionClass.SWMMAC,),
}

_AMDGPU_KEY_INSTRUCTION_CLASSES = (
    ("amdgpu.global_load_", InstructionClass.GLOBAL_LOAD),
    ("amdgpu.global_store_", InstructionClass.GLOBAL_STORE),
    ("amdgpu.buffer_load_", InstructionClass.BUFFER_LOAD),
    ("amdgpu.buffer_store_", InstructionClass.BUFFER_STORE),
    ("amdgpu.flat_load_", InstructionClass.FLAT_MEMORY),
    ("amdgpu.flat_store_", InstructionClass.FLAT_MEMORY),
)


def _with_instruction_classes(descriptor_set: DescriptorSet) -> DescriptorSet:
    descriptors: list[Descriptor] = []
    descriptors_changed = False
    for descriptor in descriptor_set.descriptors:
        instruction_classes = set(descriptor.instruction_classes)
        schedule_instruction_classes = _AMDGPU_SCHEDULE_INSTRUCTION_CLASSES.get(
            descriptor.schedule_class, ()
        )
        if descriptor.schedule_class.startswith(_SCHEDULE_MFMA_QUALIFIED_PREFIX):
            schedule_instruction_classes = (InstructionClass.MFMA,)
        elif descriptor.schedule_class.startswith(f"{_SCHEDULE_MATRIX}."):
            semantic_tag = descriptor.semantic_tag or ""
            if semantic_tag.startswith("matrix.wmma."):
                schedule_instruction_classes = (InstructionClass.WMMA,)
            elif semantic_tag.startswith("matrix.swmmac."):
                schedule_instruction_classes = (InstructionClass.SWMMAC,)
        instruction_classes.update(schedule_instruction_classes)
        semantic_tag = descriptor.semantic_tag or ""
        if semantic_tag.startswith(("memory.stack.", "memory.private.")):
            instruction_classes.discard(InstructionClass.GLOBAL_MEMORY)
        for key_prefix, instruction_class in _AMDGPU_KEY_INSTRUCTION_CLASSES:
            if descriptor.key.startswith(key_prefix):
                instruction_classes.add(instruction_class)
                break
        projected_instruction_classes = tuple(
            instruction_class
            for instruction_class in InstructionClass
            if instruction_class in instruction_classes
        )
        if projected_instruction_classes == descriptor.instruction_classes:
            descriptors.append(descriptor)
            continue
        descriptors.append(
            replace(
                descriptor,
                instruction_classes=projected_instruction_classes,
            )
        )
        descriptors_changed = True
    if not descriptors_changed:
        return descriptor_set
    return replace(descriptor_set, descriptors=tuple(descriptors))


_AMDGPU_CORE_DESCRIPTOR_SET_BUILDER_FLAG_GFX125X = 1 << 0

_AMDGPU_CORE_INSTRUCTION_FACT_NAMES = (
    "S_GETPC_B64",
    "S_MOV_B32",
    "S_MOV_B64",
    "S_XOR_B64",
)


def _preserve_overlay_descriptors(
    descriptors: tuple[Descriptor, ...],
) -> tuple[Descriptor, ...]:
    return descriptors


_AmdgpuOverlayMaterializer = Callable[[AmdgpuIsaFactSource], tuple[Descriptor, ...]]
_AmdgpuOverlayProjection = Callable[[tuple[Descriptor, ...]], tuple[Descriptor, ...]]


@dataclass(frozen=True, slots=True)
class _AmdgpuCoreDescriptorSetBuilder:
    # Static target-low descriptor-set skeleton.
    base: DescriptorSet
    # Pure source overlay rows owned by the target contract.
    overlay_rows: Callable[[], tuple[AmdgpuDescriptorOverlay, ...]]
    # Materializer shared by targets that derive from the same ISA facts.
    overlay_descriptors: _AmdgpuOverlayMaterializer
    # Target-specific projection of the shared materialized descriptors.
    project_overlay_descriptors: _AmdgpuOverlayProjection = (
        _preserve_overlay_descriptors
    )
    # XML donor instructions needed to synthesize target-owned facts.
    source_instruction_names: tuple[str, ...] = ()
    # Target-owned descriptors that have no XML overlay.
    extra_descriptors: tuple[Descriptor, ...] = ()
    # Builder behavior flags.
    flags: int = 0


_GFX125X_EXTRA_DESCRIPTORS = (
    _s_delay_alu_descriptor(),
    _s_wait_xcnt_descriptor(),
    *_gfx125x_cluster_descriptors(),
    *_gfx125x_tensor_descriptors(),
)


_AMDGPU_CORE_DESCRIPTOR_SET_BUILDERS = {
    "cdna3": _AmdgpuCoreDescriptorSetBuilder(
        base=_AMDGPU_CDNA3_CORE_DESCRIPTOR_SET_BASE,
        overlay_rows=_gfx940_core_overlays,
        overlay_descriptors=_gfx940_core_overlay_descriptors,
    ),
    "cdna4": _AmdgpuCoreDescriptorSetBuilder(
        base=_AMDGPU_CDNA4_CORE_DESCRIPTOR_SET_BASE,
        overlay_rows=_gfx950_core_overlays,
        overlay_descriptors=_gfx950_core_overlay_descriptors,
    ),
    "gfx9_4_generic": _AmdgpuCoreDescriptorSetBuilder(
        base=_AMDGPU_GFX9_4_GENERIC_CORE_DESCRIPTOR_SET_BASE,
        overlay_rows=_gfx9_4_generic_core_overlays,
        overlay_descriptors=_gfx9_4_generic_core_overlay_descriptors,
    ),
    "rdna3": _AmdgpuCoreDescriptorSetBuilder(
        base=_AMDGPU_RDNA3_CORE_DESCRIPTOR_SET_BASE,
        overlay_rows=_gfx11_core_overlays,
        overlay_descriptors=_gfx11_core_overlay_descriptors,
        extra_descriptors=(_s_delay_alu_descriptor(),),
    ),
    "gfx11_generic": _AmdgpuCoreDescriptorSetBuilder(
        base=_AMDGPU_GFX11_GENERIC_CORE_DESCRIPTOR_SET_BASE,
        overlay_rows=_gfx11_core_overlays,
        overlay_descriptors=_gfx11_core_overlay_descriptors,
        extra_descriptors=(_s_delay_alu_descriptor(),),
    ),
    "gfx12_generic": _AmdgpuCoreDescriptorSetBuilder(
        base=_AMDGPU_GFX12_GENERIC_CORE_DESCRIPTOR_SET_BASE,
        overlay_rows=_gfx12_core_overlays,
        overlay_descriptors=_gfx12_core_overlay_descriptors,
        extra_descriptors=(_s_delay_alu_descriptor(),),
    ),
    "gfx12_5_generic": _AmdgpuCoreDescriptorSetBuilder(
        base=_AMDGPU_GFX12_5_GENERIC_CORE_DESCRIPTOR_SET_BASE,
        overlay_rows=_gfx12_5_generic_core_overlays,
        overlay_descriptors=_gfx12_5_generic_core_overlay_descriptors,
        extra_descriptors=_GFX125X_EXTRA_DESCRIPTORS,
        flags=_AMDGPU_CORE_DESCRIPTOR_SET_BUILDER_FLAG_GFX125X,
    ),
    "rdna3_5": _AmdgpuCoreDescriptorSetBuilder(
        base=_AMDGPU_RDNA3_5_CORE_DESCRIPTOR_SET_BASE,
        overlay_rows=_gfx115x_core_overlays,
        overlay_descriptors=_gfx115x_core_overlay_descriptors,
        extra_descriptors=(_s_delay_alu_descriptor(),),
    ),
    "rdna4m": _AmdgpuCoreDescriptorSetBuilder(
        base=_AMDGPU_RDNA4M_CORE_DESCRIPTOR_SET_BASE,
        overlay_rows=_rdna4m_core_overlays,
        overlay_descriptors=_rdna4m_core_overlay_descriptors,
        source_instruction_names=tuple(
            sorted(
                {
                    row[0]
                    for row in (
                        *_RDNA4M_NUMERIC_MINMAX_INSTRUCTION_ROWS,
                        *_RDNA4M_IEEE_MINMAX_INSTRUCTION_ROWS,
                    )
                }
            )
        ),
        extra_descriptors=(_s_delay_alu_descriptor(),),
    ),
    "rdna4": _AmdgpuCoreDescriptorSetBuilder(
        base=_AMDGPU_RDNA4_CORE_DESCRIPTOR_SET_BASE,
        overlay_rows=_gfx12_core_overlays,
        overlay_descriptors=_gfx12_core_overlay_descriptors,
        extra_descriptors=(_s_delay_alu_descriptor(),),
    ),
    "rdna4_gfx125x": _AmdgpuCoreDescriptorSetBuilder(
        base=_AMDGPU_RDNA4_GFX125X_CORE_DESCRIPTOR_SET_BASE,
        overlay_rows=_gfx125x_core_overlays,
        overlay_descriptors=_gfx125x_core_overlay_descriptors,
        extra_descriptors=_GFX125X_EXTRA_DESCRIPTORS,
        flags=_AMDGPU_CORE_DESCRIPTOR_SET_BUILDER_FLAG_GFX125X,
    ),
    "rdna4_gfx1250_a0": _AmdgpuCoreDescriptorSetBuilder(
        base=_AMDGPU_RDNA4_GFX1250_A0_CORE_DESCRIPTOR_SET_BASE,
        overlay_rows=_gfx125x_core_overlays,
        overlay_descriptors=_gfx125x_core_overlay_descriptors,
        project_overlay_descriptors=_gfx1250_a0_core_overlay_projection,
        extra_descriptors=_GFX125X_EXTRA_DESCRIPTORS,
        flags=_AMDGPU_CORE_DESCRIPTOR_SET_BUILDER_FLAG_GFX125X,
    ),
    "rdna4_gfx1251": _AmdgpuCoreDescriptorSetBuilder(
        base=_AMDGPU_RDNA4_GFX1251_CORE_DESCRIPTOR_SET_BASE,
        overlay_rows=_gfx125x_core_overlays,
        overlay_descriptors=_gfx125x_core_overlay_descriptors,
        project_overlay_descriptors=_gfx1251_core_overlay_projection,
        extra_descriptors=_GFX125X_EXTRA_DESCRIPTORS,
        flags=_AMDGPU_CORE_DESCRIPTOR_SET_BUILDER_FLAG_GFX125X,
    ),
}

AMDGPU_DESCRIPTOR_SET_GENERATOR_TARGETS = tuple(
    sorted(_AMDGPU_CORE_DESCRIPTOR_SET_BUILDERS)
)


@cache
def amdgpu_core_descriptor_set_instruction_names(target: str) -> tuple[str, ...]:
    """Returns the imported instruction facts needed by one target contract."""

    try:
        builder = _AMDGPU_CORE_DESCRIPTOR_SET_BUILDERS[target]
    except KeyError as exc:
        supported = ", ".join(AMDGPU_DESCRIPTOR_SET_GENERATOR_TARGETS)
        raise ValueError(
            f"unsupported AMDGPU descriptor target '{target}'; "
            f"expected one of: {supported}"
        ) from exc
    names = set(_AMDGPU_CORE_INSTRUCTION_FACT_NAMES)
    names.update(overlay.instruction_name for overlay in builder.overlay_rows())
    names.update(builder.source_instruction_names)
    return tuple(sorted(names))


def amdgpu_core_descriptor_set_instruction_names_by_isa_key(
    descriptor_set_infos: Iterable[AmdgpuDescriptorSetInfo],
) -> dict[str, tuple[str, ...]]:
    """Returns the union of imported instruction facts required per ISA XML."""

    names_by_isa_key: dict[str, set[str]] = {}
    for info in descriptor_set_infos:
        instruction_names = amdgpu_core_descriptor_set_instruction_names(
            info.generator_target
        )
        for isa_info in info.isa_infos:
            names_by_isa_key.setdefault(isa_info.isa_xml_key, set()).update(
                instruction_names
            )
    return {
        isa_key: tuple(sorted(instruction_names))
        for isa_key, instruction_names in sorted(names_by_isa_key.items())
    }


def _build_amdgpu_core_descriptor_set_from_spec(
    target: str,
    builder: _AmdgpuCoreDescriptorSetBuilder,
    spec: AmdgpuIsaFactSource,
    materialized_overlay_descriptors: tuple[Descriptor, ...],
) -> DescriptorSet:
    descriptor_set = _with_overlay_descriptors(
        builder.base,
        spec,
        builder.project_overlay_descriptors(materialized_overlay_descriptors),
        builder.extra_descriptors,
    )
    is_gfx125x = bool(builder.flags & _AMDGPU_CORE_DESCRIPTOR_SET_BUILDER_FLAG_GFX125X)
    if is_gfx125x:
        descriptor_set = _with_gfx125x_vgpr_msb_address_states(descriptor_set)
    descriptor_set = _with_storage_lease_rows(
        descriptor_set, enable_gfx125x_xcnt=is_gfx125x
    )
    descriptor_set = _with_instruction_classes(descriptor_set)
    _validate_descriptor_encoding_formats(target, spec, descriptor_set)
    _validate_dpp_control_fields(descriptor_set)
    _validate_matrix_high_half_select_fields(descriptor_set)
    _validate_native_asm_values(descriptor_set)
    _validate_descriptor_set_info_capabilities(target, descriptor_set)
    _validate_matrix_coexecution_profile_coverage(target, descriptor_set)
    return descriptor_set


def _amdgpu_core_descriptor_set_builder(
    target: str,
) -> _AmdgpuCoreDescriptorSetBuilder:
    try:
        return _AMDGPU_CORE_DESCRIPTOR_SET_BUILDERS[target]
    except KeyError as exc:
        supported = ", ".join(AMDGPU_DESCRIPTOR_SET_GENERATOR_TARGETS)
        raise ValueError(
            f"unsupported AMDGPU descriptor target '{target}'; "
            f"expected one of: {supported}"
        ) from exc


def amdgpu_core_descriptor_set_overlay_rows(
    target: str,
) -> tuple[AmdgpuDescriptorOverlay, ...]:
    """Returns the XML-free descriptor overlay rows for a generator target."""

    return _amdgpu_core_descriptor_set_builder(target).overlay_rows()


def _build_amdgpu_core_descriptor_set_from_specs(
    target: str,
    builder: _AmdgpuCoreDescriptorSetBuilder,
    info: AmdgpuDescriptorSetInfo,
    specs: Mapping[str, AmdgpuIsaFactSource],
    materialized_descriptors_by_source_and_builder: dict[
        tuple[str, _AmdgpuOverlayMaterializer],
        tuple[Descriptor, ...],
    ],
) -> DescriptorSet:
    descriptor_sets: list[DescriptorSet] = []
    for isa_info in info.isa_infos:
        try:
            spec = specs[isa_info.isa_xml_key]
        except KeyError as exc:
            raise ValueError(
                f"AMDGPU descriptor target '{target}' is missing ISA XML key "
                f"'{isa_info.isa_xml_key}'"
            ) from exc
        if (
            spec.architecture_name != isa_info.isa_architecture_name
            or spec.architecture_id != isa_info.isa_architecture_id
        ):
            raise ValueError(
                f"{spec.source_name}: AMDGPU descriptor set {info.key} ISA XML "
                f"key '{isa_info.isa_xml_key}' expects "
                f"{isa_info.isa_architecture_name} architecture id "
                f"{isa_info.isa_architecture_id}, found "
                f"{spec.architecture_name} architecture id "
                f"{spec.architecture_id}"
            )
        materialization_key = (
            isa_info.isa_xml_key,
            builder.overlay_descriptors,
        )
        materialized_overlay_descriptors = (
            materialized_descriptors_by_source_and_builder.get(materialization_key)
        )
        if materialized_overlay_descriptors is None:
            materialized_overlay_descriptors = builder.overlay_descriptors(spec)
            materialized_descriptors_by_source_and_builder[materialization_key] = (
                materialized_overlay_descriptors
            )
        descriptor_sets.append(
            _build_amdgpu_core_descriptor_set_from_spec(
                target,
                builder,
                spec,
                materialized_overlay_descriptors,
            )
        )
    descriptor_set = descriptor_sets[0]
    for isa_info, member_descriptor_set in zip(
        info.isa_infos[1:],
        descriptor_sets[1:],
        strict=True,
    ):
        if member_descriptor_set != descriptor_set:
            raise ValueError(
                f"AMDGPU descriptor target '{target}' does not have a common "
                f"contract across ISA XML keys '{info.isa_infos[0].isa_xml_key}' "
                f"and '{isa_info.isa_xml_key}'"
            )
    return replace(
        descriptor_set,
        supported_target_contract_keys=(
            amdgpu_descriptor_set_supported_target_contract_keys(info)
        ),
    )


def build_amdgpu_core_descriptor_sets_from_specs(
    targets: Sequence[str],
    specs: Mapping[str, AmdgpuIsaFactSource],
) -> dict[str, DescriptorSet]:
    """Builds related descriptor sets while sharing common ISA materialization."""

    if not targets:
        raise ValueError("AMDGPU descriptor family requires at least one target")
    if len(set(targets)) != len(targets):
        raise ValueError("AMDGPU descriptor family targets must be unique")

    builders_and_infos = tuple(
        (
            target,
            _amdgpu_core_descriptor_set_builder(target),
            amdgpu_descriptor_set_info_by_generator_target(target),
        )
        for target in targets
    )
    materialized_descriptors_by_source_and_builder: dict[
        tuple[str, _AmdgpuOverlayMaterializer],
        tuple[Descriptor, ...],
    ] = {}
    return {
        target: _build_amdgpu_core_descriptor_set_from_specs(
            target,
            builder,
            info,
            specs,
            materialized_descriptors_by_source_and_builder,
        )
        for target, builder, info in builders_and_infos
    }


def build_amdgpu_core_descriptor_set_from_specs(
    target: str,
    specs: Mapping[str, AmdgpuIsaFactSource],
) -> DescriptorSet:
    return build_amdgpu_core_descriptor_sets_from_specs((target,), specs)[target]


def build_amdgpu_core_descriptor_set_from_spec(
    target: str,
    spec: AmdgpuIsaFactSource,
) -> DescriptorSet:
    info = amdgpu_descriptor_set_info_by_generator_target(target)
    if len(info.isa_infos) != 1:
        keys = ", ".join(isa_info.isa_xml_key for isa_info in info.isa_infos)
        raise ValueError(
            f"AMDGPU descriptor target '{target}' requires ISA XML keys: {keys}"
        )
    return build_amdgpu_core_descriptor_set_from_specs(
        target,
        {info.isa_infos[0].isa_xml_key: spec},
    )


def build_amdgpu_core_descriptor_set(
    target: str,
    xml_path: str | Path,
) -> DescriptorSet:
    return build_amdgpu_core_descriptor_set_from_spec(
        target, parse_amdgpu_isa_xml_path(xml_path)
    )


__all__ = (
    "AMDGPU_DESCRIPTOR_SET_GENERATOR_TARGETS",
    "_AMDGPU_CORE_DESCRIPTOR_SET_BUILDERS",
    "_AmdgpuCoreDescriptorSetBuilder",
    "_descriptor_address_offset_immediates",
    "_descriptor_has_memory_effect",
    "_gfx125x_operand_uses_vgpr_msb_state",
    "_with_gfx125x_vgpr_msb_address_state",
    "_with_gfx125x_vgpr_msb_address_states",
    "_validate_address_immediate_units",
    "_validate_descriptor_encoding_formats",
    "_validate_dpp_control_fields",
    "_with_overlay_descriptors",
    "amdgpu_common_reg_class_ids",
    "amdgpu_core_descriptor_set_instruction_names",
    "amdgpu_core_descriptor_set_instruction_names_by_isa_key",
    "amdgpu_core_descriptor_set_overlay_rows",
    "amdgpu_descriptor_id_keys",
    "amdgpu_descriptor_ref_keys",
    "amdgpu_immediate_encoding_id_items",
    "build_amdgpu_core_descriptor_set",
    "build_amdgpu_core_descriptor_set_from_spec",
    "build_amdgpu_core_descriptor_set_from_specs",
    "build_amdgpu_core_descriptor_sets_from_specs",
)
