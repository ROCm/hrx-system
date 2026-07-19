# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


# ruff: noqa: F403, F405

"""Public AMDGPU descriptor construction API."""

from __future__ import annotations

from loom.target.arch.amdgpu.encoding import (
    AMDGPU_DPP_CONTROL_ENCODING_FORMAT_IDS,
    AMDGPU_ENCODING_FORMAT_NONE,
    AMDGPU_ENCODING_FORMAT_XML_NAMES_BY_ID,
    AmdgpuVgprMsbSlot,
    amdgpu_dpp_control_is_valid,
    amdgpu_encoding_field_id,
    amdgpu_encoding_format_id,
    amdgpu_gfx125x_vgpr_msb_slot,
    amdgpu_supplemental_encoding_format_names,
)
from loom.target.low_descriptors import InstructionClass

from .categories import *
from .cluster import _gfx125x_cluster_descriptors
from .common import *
from .control import _s_delay_alu_descriptor
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
                    value.kind is NativeAsmValueKind.IMMEDIATE_TARGET_FORMAT
                    and value.target_format_id in _NAMED_NATIVE_ASM_IMMEDIATE_FORMAT_IDS
                )
                if not is_named_modifier:
                    if saw_named_modifier:
                        raise ValueError(
                            f"AMDGPU descriptor '{descriptor.key}' native asm "
                            "form has a positional value after a named modifier"
                        )
                    continue
                saw_named_modifier = True
                immediate = immediate_by_name.get(value.field_name)
                if immediate is None:
                    raise ValueError(
                        f"AMDGPU descriptor '{descriptor.key}' native asm "
                        f"modifier references unknown immediate '{value.field_name}'"
                    )
                if ImmediateFlag.DEFAULT_VALUE not in immediate.flags:
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


_GFX125X_VGPR_MSB_ADDRESSABLE_UNIT_COUNT = 256


def _operand_has_vgpr_alt(operand: Operand) -> bool:
    return any(reg_alt.reg_class == _REG_VGPR for reg_alt in operand.reg_alts)


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
    if operand.address_map_kind is not OperandAddressMapKind.DIRECT:
        return operand
    address_state_slot = _gfx125x_operand_vgpr_msb_slot(descriptor, operand)
    if address_state_slot != AmdgpuVgprMsbSlot.NONE:
        return replace(
            operand,
            address_map_kind=OperandAddressMapKind.TARGET_STATE,
            addressable_unit_count=_GFX125X_VGPR_MSB_ADDRESSABLE_UNIT_COUNT,
            address_state_slot=int(address_state_slot),
        )
    return replace(
        operand,
        address_map_kind=OperandAddressMapKind.LOW_SUBSET,
        addressable_unit_count=_GFX125X_VGPR_MSB_ADDRESSABLE_UNIT_COUNT,
    )


def _with_gfx125x_vgpr_msb_address_state(descriptor: Descriptor) -> Descriptor:
    operands = tuple(
        _with_gfx125x_operand_address_state(descriptor, operand)
        for operand in descriptor.operands
    )
    updated_descriptor = replace(descriptor, operands=operands)
    if any(
        operand.address_map_kind is OperandAddressMapKind.TARGET_STATE
        for operand in operands
    ):
        updated_descriptor = _with_mode_state_read(updated_descriptor)
    return updated_descriptor


def _with_gfx125x_vgpr_msb_address_states(
    descriptor_set: DescriptorSet,
) -> DescriptorSet:
    descriptor_set = replace(
        descriptor_set,
        descriptors=tuple(
            _with_gfx125x_vgpr_msb_address_state(descriptor)
            for descriptor in descriptor_set.descriptors
        ),
    )
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
    descriptors_by_key = {
        descriptor.key: descriptor for descriptor in descriptor_set.descriptors
    }
    try:
        mode_descriptor = descriptors_by_key["amdgpu.s_set_vgpr_msb"]
    except KeyError as exc:
        raise ValueError(
            "gfx125x VGPR-MSB target-state operands require 'amdgpu.s_set_vgpr_msb'"
        ) from exc
    if not _descriptor_writes_mode_state(mode_descriptor):
        raise ValueError(
            "gfx125x descriptor 'amdgpu.s_set_vgpr_msb' must write MODE state"
        )
    for descriptor in descriptor_set.descriptors:
        has_target_state_operand = False
        tied_operand_roots = _descriptor_tied_operand_roots(descriptor)
        address_state_slot_operands: dict[int, int] = {}
        for operand_index, operand in enumerate(descriptor.operands):
            if operand.address_map_kind is not OperandAddressMapKind.TARGET_STATE:
                continue
            has_target_state_operand = True
            expected_slot = _gfx125x_operand_vgpr_msb_slot(descriptor, operand)
            if expected_slot == AmdgpuVgprMsbSlot.NONE:
                raise ValueError(
                    f"gfx125x descriptor '{descriptor.key}' marks operand "
                    f"'{operand.field_name}' as VGPR-MSB target-state, but "
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
            if (
                operand.addressable_unit_count
                != _GFX125X_VGPR_MSB_ADDRESSABLE_UNIT_COUNT
            ):
                raise ValueError(
                    f"gfx125x descriptor '{descriptor.key}' marks operand "
                    f"'{operand.field_name}' as VGPR-MSB target-state with "
                    f"{operand.addressable_unit_count} addressable units; "
                    f"expected {_GFX125X_VGPR_MSB_ADDRESSABLE_UNIT_COUNT}"
                )
        if has_target_state_operand and not any(
            _is_mode_state_read(operand) for operand in descriptor.operands
        ):
            raise ValueError(
                f"gfx125x descriptor '{descriptor.key}' uses VGPR-MSB "
                "target-state operands but does not read MODE state"
            )


_AMDGPU_WAIT_COUNTER_MASKS = {
    _COUNTER_VMEM_LOAD: 1 << 0,
    _COUNTER_VMEM_STORE: 1 << 1,
    _COUNTER_LDS: 1 << 2,
    _COUNTER_SMEM: 1 << 3,
    _COUNTER_ALU: 1 << 4,
    _COUNTER_TENSOR: 1 << 5,
    _COUNTER_ASYNC: 1 << 6,
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
}

_AMDGPU_WAIT_PLAN_RESIDUAL_ACTION_WAIT_PACKET = 1
_AMDGPU_WAIT_PLAN_RESIDUAL_ACTION_WAIT_PACKET_NAME = "amdgpu.wait_packet"
_AMDGPU_WAIT_PLAN_REASON_STORE_SOURCE_REUSE = 4
_AMDGPU_WAIT_PLAN_REASON_STORE_SOURCE_REUSE_NAME = "amdgpu.store_source_reuse"
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
    return OperandFlag.IMPLICIT not in operand.flags and operand.role in (
        OperandRole.OPERAND,
        OperandRole.PREDICATE,
        OperandRole.RESOURCE,
    )


def _amdgpu_operand_accepts_vgpr(operand: Operand) -> bool:
    return any(
        reg_alt.reg_class == _REG_VGPR
        and RegClassAltFlag.IMMEDIATE not in reg_alt.flags
        for reg_alt in operand.reg_alts
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
    vmem_write_counter_mask = (
        write_counter_mask & _AMDGPU_WAIT_COUNTER_MASKS[_COUNTER_VMEM_STORE]
    )
    if memory_source_read_counter_mask != 0 or memory_source_write_counter_mask != 0:
        packet_operand_index = 0
        for operand in descriptor.operands[_descriptor_result_count(descriptor) :]:
            if not _amdgpu_operand_is_packet_input(operand):
                continue
            current_packet_operand_index = packet_operand_index
            packet_operand_index += 1
            if operand.unit_count == 0:
                continue
            if _amdgpu_operand_accepts_vgpr(operand) and vmem_write_counter_mask != 0:
                storage_leases.append(
                    _amdgpu_storage_lease(
                        kind=StorageLeaseKind.SOURCE_READ,
                        attachment=StorageLeaseAttachment.OPERAND,
                        attachment_index=current_packet_operand_index,
                        unit_count=operand.unit_count,
                        release_class_id=_COUNTER_VMEM_STORE,
                        release_reason_id=_AMDGPU_WAIT_PLAN_REASON_STORE_SOURCE_REUSE,
                        release_reason_name=(
                            _AMDGPU_WAIT_PLAN_REASON_STORE_SOURCE_REUSE_NAME
                        ),
                        flags=_AMDGPU_STORAGE_LEASE_FLAGS,
                    )
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


def _with_storage_lease_rows(descriptor_set: DescriptorSet) -> DescriptorSet:
    schedule_classes = {
        schedule_class.name: schedule_class
        for schedule_class in descriptor_set.schedule_classes
    }
    return replace(
        descriptor_set,
        descriptors=tuple(
            replace(
                descriptor,
                storage_leases=_amdgpu_descriptor_storage_leases(
                    schedule_classes, descriptor
                ),
            )
            for descriptor in descriptor_set.descriptors
        ),
    )


_AMDGPU_SCHEDULE_INSTRUCTION_CLASSES = {
    _SCHEDULE_SMEM_LOAD: (InstructionClass.SCALAR_MEMORY,),
    _SCHEDULE_SMEM_STORE: (InstructionClass.SCALAR_MEMORY,),
    _SCHEDULE_VMEM_LOAD: (InstructionClass.GLOBAL_MEMORY,),
    _SCHEDULE_VMEM_LOAD_LDS: (
        InstructionClass.GLOBAL_MEMORY,
        InstructionClass.LOCAL_MEMORY,
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
    ),
    _SCHEDULE_CLUSTER_LOAD_LDS: (
        InstructionClass.GLOBAL_MEMORY,
        InstructionClass.LOCAL_MEMORY,
    ),
    _SCHEDULE_MFMA: (InstructionClass.MFMA,),
    _SCHEDULE_WMMA: (InstructionClass.WMMA,),
    _SCHEDULE_WMMA_SCALE: (InstructionClass.WMMA,),
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
    descriptors = []
    for descriptor in descriptor_set.descriptors:
        instruction_classes = set(descriptor.instruction_classes)
        instruction_classes.update(
            _AMDGPU_SCHEDULE_INSTRUCTION_CLASSES.get(descriptor.schedule_class, ())
        )
        semantic_tag = descriptor.semantic_tag or ""
        if semantic_tag.startswith(("memory.stack.", "memory.private.")):
            instruction_classes.discard(InstructionClass.GLOBAL_MEMORY)
        for key_prefix, instruction_class in _AMDGPU_KEY_INSTRUCTION_CLASSES:
            if descriptor.key.startswith(key_prefix):
                instruction_classes.add(instruction_class)
                break
        descriptors.append(
            replace(
                descriptor,
                instruction_classes=tuple(
                    instruction_class
                    for instruction_class in InstructionClass
                    if instruction_class in instruction_classes
                ),
            )
        )
    return replace(descriptor_set, descriptors=tuple(descriptors))


@dataclass(frozen=True, slots=True)
class _AmdgpuCoreDescriptorSetBuilder:
    base: DescriptorSet
    overlay_descriptors: Callable[[AmdgpuIsaFactSource], tuple[Descriptor, ...]]
    extra_descriptors: tuple[Descriptor, ...] = ()


_AMDGPU_CORE_DESCRIPTOR_SET_BUILDERS = {
    "cdna3": _AmdgpuCoreDescriptorSetBuilder(
        base=_AMDGPU_CDNA3_CORE_DESCRIPTOR_SET_BASE,
        overlay_descriptors=_gfx940_core_overlay_descriptors,
    ),
    "cdna4": _AmdgpuCoreDescriptorSetBuilder(
        base=_AMDGPU_CDNA4_CORE_DESCRIPTOR_SET_BASE,
        overlay_descriptors=_gfx950_core_overlay_descriptors,
    ),
    "rdna3": _AmdgpuCoreDescriptorSetBuilder(
        base=_AMDGPU_RDNA3_CORE_DESCRIPTOR_SET_BASE,
        overlay_descriptors=_gfx11_core_overlay_descriptors,
        extra_descriptors=(_s_delay_alu_descriptor(),),
    ),
    "rdna3_5": _AmdgpuCoreDescriptorSetBuilder(
        base=_AMDGPU_RDNA3_5_CORE_DESCRIPTOR_SET_BASE,
        overlay_descriptors=_gfx117x_core_overlay_descriptors,
        extra_descriptors=(_s_delay_alu_descriptor(),),
    ),
    "rdna4": _AmdgpuCoreDescriptorSetBuilder(
        base=_AMDGPU_RDNA4_CORE_DESCRIPTOR_SET_BASE,
        overlay_descriptors=_gfx12_core_overlay_descriptors,
        extra_descriptors=(_s_delay_alu_descriptor(),),
    ),
    "rdna4_gfx125x": _AmdgpuCoreDescriptorSetBuilder(
        base=_AMDGPU_RDNA4_GFX125X_CORE_DESCRIPTOR_SET_BASE,
        overlay_descriptors=_gfx1250_core_overlay_descriptors,
        extra_descriptors=(
            _s_delay_alu_descriptor(),
            *_gfx125x_cluster_descriptors(),
            *_gfx125x_tensor_descriptors(),
        ),
    ),
}

AMDGPU_DESCRIPTOR_SET_GENERATOR_TARGETS = tuple(
    sorted(_AMDGPU_CORE_DESCRIPTOR_SET_BUILDERS)
)


def build_amdgpu_core_descriptor_set_from_spec(
    target: str,
    spec: AmdgpuIsaFactSource,
) -> DescriptorSet:
    try:
        builder = _AMDGPU_CORE_DESCRIPTOR_SET_BUILDERS[target]
    except KeyError as exc:
        supported = ", ".join(AMDGPU_DESCRIPTOR_SET_GENERATOR_TARGETS)
        raise ValueError(
            f"unsupported AMDGPU descriptor target '{target}'; "
            f"expected one of: {supported}"
        ) from exc
    validate_amdgpu_descriptor_set_isa_xml(
        amdgpu_descriptor_set_info_by_generator_target(target), spec
    )
    descriptor_set = _with_overlay_descriptors(
        builder.base,
        spec,
        builder.overlay_descriptors(spec),
        builder.extra_descriptors,
    )
    if target == "rdna4_gfx125x":
        descriptor_set = _with_gfx125x_vgpr_msb_address_states(descriptor_set)
    descriptor_set = _with_storage_lease_rows(descriptor_set)
    descriptor_set = _with_instruction_classes(descriptor_set)
    _validate_descriptor_encoding_formats(target, spec, descriptor_set)
    _validate_dpp_control_fields(descriptor_set)
    _validate_matrix_high_half_select_fields(descriptor_set)
    _validate_native_asm_values(descriptor_set)
    return descriptor_set


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
    "amdgpu_descriptor_id_keys",
    "amdgpu_descriptor_ref_keys",
    "amdgpu_immediate_encoding_id_items",
    "build_amdgpu_core_descriptor_set",
    "build_amdgpu_core_descriptor_set_from_spec",
)
