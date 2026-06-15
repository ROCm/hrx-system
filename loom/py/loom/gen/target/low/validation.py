# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Validation helpers for low descriptor generator inputs."""

from __future__ import annotations

from collections.abc import Sequence

from loom.target.low_descriptors import (
    LOW_DESCRIPTOR_ENCODING_ID_NONE,
    AsmForm,
    ConstraintKind,
    Descriptor,
    DescriptorAsmSurface,
    DescriptorFlag,
    DescriptorSet,
    EnumDomain,
    EnumValue,
    Hazard,
    Immediate,
    ImmediateFlag,
    ImmediateKind,
    OperandAddressMapKind,
    OperandFlag,
    OperandRole,
    RegisterPart,
    StorageLeaseAttachment,
)


def validate_u16(value: int, description: str) -> None:
    if value < 0 or value > 0xFFFF:
        raise ValueError(f"{description} does not fit u16")


def validate_u32(value: int, description: str) -> None:
    if value < 0 or value > 0xFFFFFFFF:
        raise ValueError(f"{description} does not fit u32")


def validate_u16_table_count(count: int, description: str) -> None:
    if count > 0xFFFF:
        raise ValueError(f"{description} count does not fit u16 descriptor references")


def validate_u64(value: int, description: str) -> None:
    if value < 0 or value > 0xFFFFFFFFFFFFFFFF:
        raise ValueError(f"{description} does not fit u64")


def validate_i64(value: int, description: str) -> None:
    if value < -(1 << 63) or value > (1 << 63) - 1:
        raise ValueError(f"{description} does not fit i64")


def _descriptor_asm_surface_description(
    surface_name: str,
    spec: DescriptorSet,
    descriptor: Descriptor,
) -> str:
    return f"{surface_name} '{spec.key}' descriptor '{descriptor.key}'"


def validate_descriptor_asm_surface(
    spec: DescriptorSet,
    descriptors: Sequence[Descriptor],
    *,
    surface_name: str = "descriptor set",
) -> None:
    if not spec.requires_explicit_asm_surface:
        return
    for descriptor in descriptors:
        description = _descriptor_asm_surface_description(
            surface_name,
            spec,
            descriptor,
        )
        if descriptor.asm_surface is DescriptorAsmSurface.AUTHORABLE:
            if descriptor.asm_surface_reason:
                raise ValueError(f"{description} is authorable asm but has an asm surface reason")
            if len(descriptor.asm_forms) != 1:
                raise ValueError(f"{description} is authorable asm but does not declare exactly one canonical asm form; found {len(descriptor.asm_forms)}")
            if DescriptorFlag.PSEUDO in descriptor.flags:
                raise ValueError(f"{description} is pseudo but classified as authorable asm")
            continue

        if not descriptor.asm_surface_reason:
            raise ValueError(f"{description} is {descriptor.asm_surface.value} asm but does not explain the non-authorable surface")
        if descriptor.asm_forms:
            raise ValueError(f"{description} is {descriptor.asm_surface.value} asm but still declares {len(descriptor.asm_forms)} asm form(s)")
        if DescriptorFlag.PSEUDO in descriptor.flags and descriptor.asm_surface is not DescriptorAsmSurface.GENERATED_ONLY:
            raise ValueError(f"{description} is pseudo but not classified as generated-only asm")


def bit_mask(bit_count: int) -> int:
    if bit_count == 0:
        return 0
    return (1 << bit_count) - 1


def validate_immediate_encoding(descriptor: Descriptor, immediate: Immediate) -> None:
    if immediate.encoding_field_id and immediate.encoding_slices:
        raise ValueError(f"descriptor '{descriptor.key}' immediate '{immediate.field_name}' uses both direct and sliced encoding fields")
    if not immediate.encoding_slices:
        return
    covered_bits = 0
    for slice_index, encoding_slice in enumerate(immediate.encoding_slices):
        slice_description = f"descriptor '{descriptor.key}' immediate '{immediate.field_name}' encoding slice {slice_index}"
        validate_u16(
            encoding_slice.encoding_field_id,
            f"{slice_description} field id",
        )
        if encoding_slice.encoding_field_id == 0:
            raise ValueError(f"{slice_description} has field id zero")
        if encoding_slice.bit_count <= 0 or encoding_slice.bit_count > 64:
            raise ValueError(f"{slice_description} has invalid bit count {encoding_slice.bit_count}")
        if encoding_slice.source_bit_offset < 0 or encoding_slice.source_bit_offset > 255:
            raise ValueError(f"{slice_description} source bit offset does not fit u8")
        source_end = encoding_slice.source_bit_offset + encoding_slice.bit_count
        if source_end > immediate.bit_width:
            raise ValueError(f"{slice_description} source range [{encoding_slice.source_bit_offset}, {source_end}) exceeds {immediate.bit_width} bits")
        slice_bits = bit_mask(encoding_slice.bit_count) << encoding_slice.source_bit_offset
        if covered_bits & slice_bits:
            raise ValueError(f"{slice_description} overlaps another slice")
        covered_bits |= slice_bits
    expected_bits = bit_mask(immediate.bit_width)
    if covered_bits != expected_bits:
        raise ValueError(f"descriptor '{descriptor.key}' immediate '{immediate.field_name}' encoding slices cover 0x{covered_bits:x} instead of 0x{expected_bits:x}")


def hazard_reference_count(hazard: Hazard) -> int:
    return sum(reference is not None for reference in (hazard.resource, hazard.counter_id, hazard.target_id))


def validate_descriptor_operands(descriptor: Descriptor) -> int:
    result_count = 0
    seen_non_result = False
    for operand in descriptor.operands:
        if operand.role is OperandRole.OPERAND_RESULT:
            raise ValueError(f"descriptor '{descriptor.key}' operand '{operand.field_name}' uses OPERAND_RESULT; use separate result and operand rows plus an explicit constraint")
        is_result = operand.role is OperandRole.RESULT
        if is_result and seen_non_result:
            raise ValueError(f"descriptor '{descriptor.key}' has result operand '{operand.field_name}' after non-result operands")
        if is_result:
            result_count += 1
        else:
            seen_non_result = True
        if operand.role is OperandRole.IMPLICIT and OperandFlag.IMPLICIT not in operand.flags:
            raise ValueError(f"descriptor '{descriptor.key}' implicit operand '{operand.field_name}' must set the implicit flag")
        if not operand.reg_alts:
            raise ValueError(f"descriptor '{descriptor.key}' operand '{operand.field_name}' has no register-class alternatives")
        validate_u16(
            operand.addressable_unit_count,
            f"descriptor '{descriptor.key}' operand '{operand.field_name}' addressable unit count",
        )
        is_explicit_packet_value = operand_role_is_packet_input(operand.role) and OperandFlag.IMPLICIT not in operand.flags
        has_addressable_assignment = is_result or is_explicit_packet_value
        if operand.address_map_kind is OperandAddressMapKind.DIRECT:
            if operand.addressable_unit_count != 0:
                raise ValueError(f"descriptor '{descriptor.key}' operand '{operand.field_name}' direct address map must not set an addressable unit count")
        elif operand.address_map_kind in (OperandAddressMapKind.LOW_SUBSET, OperandAddressMapKind.TARGET_STATE):
            if not has_addressable_assignment:
                raise ValueError(f"descriptor '{descriptor.key}' operand '{operand.field_name}' bounded address map must apply to an explicit value operand")
            if operand.addressable_unit_count == 0:
                raise ValueError(f"descriptor '{descriptor.key}' operand '{operand.field_name}' bounded address map must set an addressable unit count")
            if operand.addressable_unit_count < operand.unit_count:
                raise ValueError(f"descriptor '{descriptor.key}' operand '{operand.field_name}' bounded address map covers fewer units than the operand consumes")
            if not any(reg_alt.reg_class is not None for reg_alt in operand.reg_alts):
                raise ValueError(f"descriptor '{descriptor.key}' operand '{operand.field_name}' bounded address map requires a concrete register-class alternative")
        else:
            raise ValueError(f"descriptor '{descriptor.key}' operand '{operand.field_name}' has unknown address map kind '{operand.address_map_kind}'")
        state_flags = {
            OperandFlag.STATE_READ,
            OperandFlag.STATE_WRITE,
        }.intersection(operand.flags)
        if OperandFlag.SCHEDULE_ONLY_STATE in operand.flags:
            if OperandFlag.IMPLICIT not in operand.flags:
                raise ValueError(f"descriptor '{descriptor.key}' operand '{operand.field_name}' uses schedule-only state without the implicit flag")
            if OperandFlag.STATE_READ not in state_flags:
                raise ValueError(f"descriptor '{descriptor.key}' operand '{operand.field_name}' uses schedule-only state without a state read flag")
            if OperandFlag.STATE_WRITE in state_flags:
                raise ValueError(f"descriptor '{descriptor.key}' operand '{operand.field_name}' uses schedule-only state with a state write flag")
        if state_flags:
            if len(operand.reg_alts) != 1:
                raise ValueError(f"descriptor '{descriptor.key}' state operand '{operand.field_name}' must name exactly one register-class alternative")
            if operand.reg_alts[0].reg_class is None:
                raise ValueError(f"descriptor '{descriptor.key}' state operand '{operand.field_name}' must name a concrete register class")
    return result_count


def validate_register_part(part: RegisterPart) -> None:
    if part.mask == 0:
        raise ValueError(f"register part '{part.name}' has an empty mask")
    if part.mask < 0 or part.mask > (2**32) - 1:
        raise ValueError(f"register part '{part.name}' mask does not fit u32")


def descriptor_has_tied_constraint(
    descriptor: Descriptor,
    result_index: int,
    operand_index: int,
) -> bool:
    for constraint in descriptor.constraints:
        if constraint.kind is not ConstraintKind.TIED:
            continue
        if constraint.lhs_operand_index != result_index:
            continue
        if constraint.rhs_operand_index != operand_index:
            continue
        return True
    return False


def operand_role_is_packet_input(role: OperandRole) -> bool:
    return role in (
        OperandRole.OPERAND,
        OperandRole.PREDICATE,
        OperandRole.RESOURCE,
    )


def operands_may_share_encoding_field(
    descriptor: Descriptor,
    lhs_index: int,
    rhs_index: int,
) -> bool:
    lhs = descriptor.operands[lhs_index]
    rhs = descriptor.operands[rhs_index]
    if lhs.role is OperandRole.RESULT and operand_role_is_packet_input(rhs.role):
        return descriptor_has_tied_constraint(descriptor, lhs_index, rhs_index)
    if rhs.role is OperandRole.RESULT and operand_role_is_packet_input(lhs.role):
        return descriptor_has_tied_constraint(descriptor, rhs_index, lhs_index)
    return False


def validate_descriptor_encoding_fields(descriptor: Descriptor) -> None:
    fixed_fields: set[int] = set()
    for field_value in descriptor.encoding_field_values:
        if field_value.encoding_field_id != 0:
            fixed_fields.add(field_value.encoding_field_id)
    for operand_index, operand in enumerate(descriptor.operands):
        if operand.encoding_field_id == 0:
            continue
        if operand.encoding_field_id in fixed_fields:
            raise ValueError(f"descriptor '{descriptor.key}' operand '{operand.field_name}' shares fixed encoding field id {operand.encoding_field_id}")
        for previous_index, previous_operand in enumerate(descriptor.operands[:operand_index]):
            if previous_operand.encoding_field_id != operand.encoding_field_id:
                continue
            if operands_may_share_encoding_field(
                descriptor,
                previous_index,
                operand_index,
            ):
                continue
            raise ValueError(
                f"descriptor '{descriptor.key}' operands '{previous_operand.field_name}' and '{operand.field_name}' share encoding field id {operand.encoding_field_id} without a tied constraint"
            )


def validate_immediate_default(descriptor: Descriptor, immediate: Immediate, enum_domains: dict[str, EnumDomain]) -> None:
    if ImmediateFlag.DEFAULT_VALUE not in immediate.flags:
        if immediate.default_value != 0:
            raise ValueError(f"descriptor '{descriptor.key}' immediate '{immediate.field_name}' has a default value without the default-value flag")
        return
    match immediate.kind:
        case ImmediateKind.SIGNED:
            maximum = min(immediate.unsigned_max, (1 << 63) - 1)
            if immediate.default_value < immediate.signed_min or immediate.default_value > maximum:
                raise ValueError(f"descriptor '{descriptor.key}' immediate '{immediate.field_name}' default value is out of signed range")
        case ImmediateKind.UNSIGNED | ImmediateKind.ORDINAL:
            if immediate.default_value < 0 or immediate.default_value > immediate.unsigned_max:
                raise ValueError(f"descriptor '{descriptor.key}' immediate '{immediate.field_name}' default value is out of unsigned range")
        case ImmediateKind.ENUM:
            assert immediate.enum_domain is not None
            domain = enum_domains[immediate.enum_domain]
            if all(value.value != immediate.default_value for value in domain.values):
                raise ValueError(f"descriptor '{descriptor.key}' immediate '{immediate.field_name}' default value is not in enum domain '{domain.name}'")


def asm_form_mnemonic(descriptor: Descriptor, asm_form: AsmForm) -> str:
    mnemonic = descriptor.mnemonic if asm_form.mnemonic is None else asm_form.mnemonic
    if mnemonic is None:
        raise ValueError(f"descriptor '{descriptor.key}' asm form must specify a mnemonic because the descriptor has no default mnemonic")
    if mnemonic == "":
        raise ValueError(f"descriptor '{descriptor.key}' asm form specifies an empty mnemonic")
    if len(mnemonic.encode()) > 255:
        raise ValueError(f"descriptor '{descriptor.key}' asm mnemonic '{mnemonic}' exceeds 255 bytes")
    return mnemonic


def descriptor_packet_operand_indices(descriptor: Descriptor) -> tuple[int, ...]:
    return tuple(i for i, operand in enumerate(descriptor.operands) if operand_role_is_packet_input(operand.role) and OperandFlag.IMPLICIT not in operand.flags)


def validate_storage_lease_name(value: str, description: str) -> None:
    if not value:
        raise ValueError(f"{description} must not be empty")
    if len(value.encode()) > 255:
        raise ValueError(f"{description} exceeds 255 bytes")


def validate_descriptor_storage_leases(
    descriptor: Descriptor,
    result_count: int,
) -> None:
    packet_operand_indices = descriptor_packet_operand_indices(descriptor)
    attachment_unit_counts: dict[tuple[StorageLeaseAttachment, int], int] = {}
    for result_index in range(result_count):
        attachment_unit_counts[(StorageLeaseAttachment.RESULT, result_index)] = descriptor.operands[result_index].unit_count
    for packet_index, descriptor_operand_index in enumerate(packet_operand_indices):
        attachment_unit_counts[(StorageLeaseAttachment.OPERAND, packet_index)] = descriptor.operands[descriptor_operand_index].unit_count
    for lease_index, lease in enumerate(descriptor.storage_leases):
        description = f"descriptor '{descriptor.key}' storage lease {lease_index}"
        validate_u16(lease.attachment_index, f"{description} attachment index")
        validate_u32(lease.unit_offset, f"{description} unit offset")
        validate_u32(lease.unit_count, f"{description} unit count")
        validate_u16(lease.release_class_id, f"{description} release class id")
        validate_u16(lease.release_action_id, f"{description} release action id")
        validate_u16(lease.release_reason_id, f"{description} release reason id")
        validate_storage_lease_name(lease.release_class_name, f"{description} release class name")
        validate_storage_lease_name(lease.release_action_name, f"{description} release action name")
        validate_storage_lease_name(lease.release_reason_name, f"{description} release reason name")
        if lease.unit_count == 0:
            raise ValueError(f"{description} has zero unit count")
        if lease.release_class_id == LOW_DESCRIPTOR_ENCODING_ID_NONE:
            raise ValueError(f"{description} has no release class id")
        if lease.release_action_id == 0:
            raise ValueError(f"{description} has zero release action id")
        if lease.release_reason_id == LOW_DESCRIPTOR_ENCODING_ID_NONE:
            raise ValueError(f"{description} has no release reason id")
        unit_count = attachment_unit_counts.get((lease.attachment, lease.attachment_index))
        if unit_count is None:
            raise ValueError(f"{description} references {lease.attachment.name.lower()} {lease.attachment_index}, which is not attached to the packet")
        if lease.unit_offset > unit_count or lease.unit_count > unit_count - lease.unit_offset:
            raise ValueError(f"{description} unit range [{lease.unit_offset}, {lease.unit_offset + lease.unit_count}) exceeds attached value unit count {unit_count}")


def immediate_accepts_i64_assignment(immediate: Immediate) -> bool:
    return immediate.kind in (
        ImmediateKind.SIGNED,
        ImmediateKind.UNSIGNED,
        ImmediateKind.ORDINAL,
        ImmediateKind.ENUM,
    )


def immediate_accepts_i64_arithmetic(immediate: Immediate) -> bool:
    return immediate.kind in (
        ImmediateKind.SIGNED,
        ImmediateKind.UNSIGNED,
        ImmediateKind.ORDINAL,
    )


def validate_unique_asm_fields(descriptor: Descriptor, asm_form: AsmForm, mnemonic: str) -> None:
    seen_fields: set[str] = set()
    for field_name in (
        *asm_form.results,
        *asm_form.operands,
        *(immediate.field_name for immediate in asm_form.immediates),
    ):
        if field_name in seen_fields:
            raise ValueError(f"descriptor '{descriptor.key}' asm form '{mnemonic}' references field '{field_name}' more than once")
        seen_fields.add(field_name)


def validate_enum_domain(domain: EnumDomain) -> tuple[EnumValue, ...]:
    if not domain.values:
        raise ValueError(f"enum domain '{domain.name}' has no values")
    by_token: dict[str, EnumValue] = {}
    for value in domain.values:
        if value.token in by_token:
            raise ValueError(f"enum domain '{domain.name}' repeats token '{value.token}'")
        by_token[value.token] = value
        if value.value < -(1 << 63) or value.value > (1 << 63) - 1:
            raise ValueError(f"enum domain '{domain.name}' token '{value.token}' value does not fit i64")
    return tuple(sorted(domain.values, key=lambda value: value.token))
