# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Validation helpers for low descriptor generator inputs."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise, permutations

from loom.target.low_descriptors import (
    LOW_DESCRIPTOR_ENCODING_ID_NONE,
    AsmForm,
    ConstraintKind,
    Descriptor,
    DescriptorAsmSurface,
    DescriptorCarrier,
    DescriptorFlag,
    DescriptorSet,
    EnumDomain,
    EnumValue,
    Hazard,
    Immediate,
    ImmediateFlag,
    ImmediateKind,
    Operand,
    OperandAddressMapKind,
    OperandFlag,
    OperandRole,
    RegClass,
    RegClassFlag,
    RegisterPart,
    StorageLeaseAttachment,
    StorageLeaseFlag,
)


@dataclass(frozen=True, slots=True)
class _PhysicalDescriptorLimits:
    """Reserved representation and exact-solver admission envelope."""

    # Maximum number of independent physical storage namespaces.
    canonical_resource_count: int = 15
    # Maximum allocation-unit capacity of one storage namespace.
    maximum_resource_capacity: int = 2048
    # Maximum allocation-unit capacity summed across all namespaces.
    aggregate_resource_capacity: int = 2304
    # Maximum number of physical operand rows in one descriptor.
    maximum_physical_binding_count: int = 15
    # Maximum number of untied components competing for one namespace.
    maximum_resource_component_count: int = 8
    # Maximum pairings of complete pre- and post-phase spatial orders.
    maximum_phase_order_pair_count: int = 1440
    # Maximum allocation-unit width of one physical operand.
    maximum_component_unit_count: int = 64
    # Maximum target-owned address-window selector identity.
    maximum_address_state_slot: int = 15


_PHYSICAL_DESCRIPTOR_LIMITS = _PhysicalDescriptorLimits()


@dataclass(frozen=True, slots=True)
class _PhysicalComponent:
    # Descriptor operand rows collapsed into this tied component.
    operand_indices: tuple[int, ...]
    # Canonical resources accepted by every row in this tied component.
    resource_keys: frozenset[str]
    # Allocation units occupied before the packet executes.
    pre_width: int
    # Allocation units occupied after the packet executes.
    post_width: int


@dataclass(frozen=True, slots=True)
class _PhysicalAddressWindow:
    # Allocation units in each independently addressable window.
    window_size: int
    # Largest base remainder that keeps the component inside one window.
    maximum_base_remainder: int


@dataclass(frozen=True, slots=True)
class _PhysicalBaseDomain:
    # Largest base that keeps the complete component inside the resource.
    maximum_base: int
    # Periodic address-window restrictions applied to candidate bases.
    address_windows: tuple[_PhysicalAddressWindow, ...]

    def first_at_or_above(self, lower_bound: int) -> int | None:
        """Returns the least admissible base at or above `lower_bound`."""

        candidate = lower_bound
        while candidate <= self.maximum_base:
            previous_candidate = candidate
            for window in self.address_windows:
                remainder = candidate % window.window_size
                if remainder > window.maximum_base_remainder:
                    candidate += window.window_size - remainder
            if candidate == previous_candidate:
                return candidate
        return None


def _physical_resource_key(register_class: RegClass) -> str:
    if register_class.alias_set_id:
        return f"alias:{register_class.alias_set_id}"
    return f"class:{register_class.name}"


def _operand_physical_register_classes(
    operand: Operand,
    register_classes: dict[str, RegClass],
) -> tuple[RegClass, ...]:
    physical_register_classes = []
    for alternative in operand.reg_alts:
        if alternative.reg_class is None:
            continue
        register_class = register_classes[alternative.reg_class]
        if RegClassFlag.PHYSICAL in register_class.flags:
            physical_register_classes.append(register_class)
    return tuple(physical_register_classes)


def _operand_phase_accesses(
    operand: Operand,
    *,
    is_early_clobber: bool,
) -> tuple[bool, bool]:
    is_result = operand.role is OperandRole.RESULT
    is_implicit_state = operand.role is OperandRole.IMPLICIT
    has_state_read = OperandFlag.STATE_READ in operand.flags
    has_state_write = OperandFlag.STATE_WRITE in operand.flags
    if is_implicit_state and not has_state_read and not has_state_write:
        return False, False
    reads_pre = has_state_read or is_early_clobber
    if not is_result and not is_implicit_state:
        reads_pre = True
    writes_post = is_result or has_state_write
    return reads_pre, writes_post


def _descriptor_tied_operand_roots(descriptor: Descriptor) -> tuple[int, ...]:
    parent = list(range(len(descriptor.operands)))

    def find(operand_index: int) -> int:
        while parent[operand_index] != operand_index:
            parent[operand_index] = parent[parent[operand_index]]
            operand_index = parent[operand_index]
        return operand_index

    for constraint in descriptor.constraints:
        if constraint.kind is not ConstraintKind.TIED:
            continue
        assert constraint.rhs_operand_index is not None
        lhs_root = find(constraint.lhs_operand_index)
        rhs_root = find(constraint.rhs_operand_index)
        parent[rhs_root] = lhs_root
    return tuple(find(operand_index) for operand_index in range(len(parent)))


def _component_base_domain(
    descriptor: Descriptor,
    component: _PhysicalComponent,
    resource_capacity: int,
    early_clobber_results: frozenset[int],
) -> _PhysicalBaseDomain:
    component_width = max(component.pre_width, component.post_width)
    maximum_base = resource_capacity - component_width
    address_windows = []
    for operand_index in component.operand_indices:
        operand = descriptor.operands[operand_index]
        reads_pre, writes_post = _operand_phase_accesses(
            operand,
            is_early_clobber=operand_index in early_clobber_results,
        )
        if not reads_pre and not writes_post:
            continue
        if operand.address_map_kind is OperandAddressMapKind.LOW_SUBSET:
            maximum_base = min(
                maximum_base,
                operand.addressable_unit_count - operand.unit_count,
            )
        elif operand.address_map_kind is OperandAddressMapKind.TARGET_STATE:
            address_windows.append(
                _PhysicalAddressWindow(
                    window_size=operand.addressable_unit_count,
                    maximum_base_remainder=(operand.addressable_unit_count - operand.unit_count),
                )
            )
    return _PhysicalBaseDomain(
        maximum_base=maximum_base,
        address_windows=tuple(address_windows),
    )


def _solve_physical_component_order_pair(
    components: Sequence[_PhysicalComponent],
    base_domains: Sequence[_PhysicalBaseDomain],
    pre_order: tuple[int, ...],
    post_order: tuple[int, ...],
) -> bool:
    edge_widths: dict[tuple[int, int], int] = {}
    for phase, order in (("pre", pre_order), ("post", post_order)):
        for source_index, destination_index in pairwise(order):
            width = getattr(components[source_index], f"{phase}_width")
            edge = (source_index, destination_index)
            edge_widths[edge] = max(edge_widths.get(edge, 0), width)

    successors: list[list[tuple[int, int]]] = [[] for _ in components]
    predecessor_counts = [0] * len(components)
    for (source_index, destination_index), width in edge_widths.items():
        successors[source_index].append((destination_index, width))
        predecessor_counts[destination_index] += 1

    ready = [component_index for component_index, predecessor_count in enumerate(predecessor_counts) if predecessor_count == 0]
    lower_bounds = [0] * len(components)
    visited_count = 0
    while ready:
        component_index = ready.pop()
        visited_count += 1
        base = base_domains[component_index].first_at_or_above(lower_bounds[component_index])
        if base is None:
            return False
        for successor_index, width in successors[component_index]:
            lower_bounds[successor_index] = max(lower_bounds[successor_index], base + width)
            predecessor_counts[successor_index] -= 1
            if predecessor_counts[successor_index] == 0:
                ready.append(successor_index)
    return visited_count == len(components)


def _physical_components_admit_placement(
    components: Sequence[_PhysicalComponent],
    base_domains: Sequence[_PhysicalBaseDomain],
) -> bool:
    pre_components = tuple(index for index, component in enumerate(components) if component.pre_width)
    post_components = tuple(index for index, component in enumerate(components) if component.post_width)
    for pre_order in permutations(pre_components):
        for post_order in permutations(post_components):
            if _solve_physical_component_order_pair(
                components,
                base_domains,
                pre_order,
                post_order,
            ):
                return True
    return False


def _validate_physical_metric(
    descriptor_set_key: str,
    metric_name: str,
    value: int,
    limit: int,
    *,
    descriptor_key: str | None = None,
) -> None:
    if value <= limit:
        return
    description = f"descriptor set '{descriptor_set_key}'"
    if descriptor_key is not None:
        description += f" descriptor '{descriptor_key}'"
    raise ValueError(f"{description} physical {metric_name} {value} exceeds generation bound {limit}")


def validate_physical_descriptor_set(
    descriptor_set: DescriptorSet,
) -> None:
    """Proves bounded packet-local physical storage for a descriptor set.

    Register classes and descriptor operands and constraints must already have
    passed their general validators.
    """

    register_classes = {register_class.name: register_class for register_class in descriptor_set.reg_classes}
    physical_register_classes = tuple(register_class for register_class in descriptor_set.reg_classes if RegClassFlag.PHYSICAL in register_class.flags)
    resources: dict[str, int] = {}
    for register_class in physical_register_classes:
        if register_class.allocatable_count == 0:
            raise ValueError(f"descriptor set '{descriptor_set.key}' physical register class '{register_class.name}' has zero allocation capacity")
        resource_key = _physical_resource_key(register_class)
        resources.setdefault(resource_key, register_class.allocatable_count)

    limits = _PHYSICAL_DESCRIPTOR_LIMITS
    maximum_resource_capacity = max(resources.values(), default=0)
    aggregate_resource_capacity = sum(resources.values())
    _validate_physical_metric(
        descriptor_set.key,
        "canonical resource count",
        len(resources),
        limits.canonical_resource_count,
    )
    _validate_physical_metric(
        descriptor_set.key,
        "maximum resource capacity",
        maximum_resource_capacity,
        limits.maximum_resource_capacity,
    )
    _validate_physical_metric(
        descriptor_set.key,
        "aggregate resource capacity",
        aggregate_resource_capacity,
        limits.aggregate_resource_capacity,
    )

    maximum_address_state_slot = 0
    address_state_slot_windows: dict[int, int] = {}

    for descriptor in descriptor_set.descriptors:
        for operand in descriptor.operands:
            for alternative in operand.reg_alts:
                if alternative.reg_class is not None and alternative.reg_class not in register_classes:
                    raise ValueError(f"descriptor '{descriptor.key}' operand '{operand.field_name}' references unknown register class '{alternative.reg_class}'")
        early_clobber_results = frozenset(constraint.lhs_operand_index for constraint in descriptor.constraints if constraint.kind is ConstraintKind.EARLY_CLOBBER)
        tied_operand_roots = _descriptor_tied_operand_roots(descriptor)

        physical_rows: list[tuple[int, Operand, tuple[RegClass, ...]]] = []
        for operand_index, operand in enumerate(descriptor.operands):
            physical_classes = _operand_physical_register_classes(operand, register_classes)
            if not physical_classes:
                continue
            if operand.role is OperandRole.IMPLICIT and OperandFlag.STATE_READ not in operand.flags and OperandFlag.STATE_WRITE not in operand.flags:
                raise ValueError(f"descriptor set '{descriptor_set.key}' descriptor '{descriptor.key}' physical implicit operand '{operand.field_name}' has no state read or write phase")
            _validate_physical_metric(
                descriptor_set.key,
                f"operand '{operand.field_name}' unit count",
                operand.unit_count,
                limits.maximum_component_unit_count,
                descriptor_key=descriptor.key,
            )
            if operand.address_state_slot != 0:
                maximum_address_state_slot = max(
                    maximum_address_state_slot,
                    operand.address_state_slot,
                )
            if operand.address_map_kind is OperandAddressMapKind.TARGET_STATE:
                previous_window = address_state_slot_windows.setdefault(
                    operand.address_state_slot,
                    operand.addressable_unit_count,
                )
                if previous_window != operand.addressable_unit_count:
                    raise ValueError(f"descriptor set '{descriptor_set.key}' address-state slot {operand.address_state_slot} has inconsistent window width")
            physical_rows.append((operand_index, operand, physical_classes))

        rows_by_component: dict[int, list[tuple[int, Operand, tuple[RegClass, ...]]]] = {}
        for operand_index, operand, physical_classes in physical_rows:
            rows_by_component.setdefault(tied_operand_roots[operand_index], []).append((operand_index, operand, physical_classes))

        address_state_slot_components: dict[int, int] = {}
        for component_root, component_rows in rows_by_component.items():
            for _, operand, _ in component_rows:
                if operand.address_state_slot == 0:
                    continue
                previous_root = address_state_slot_components.setdefault(operand.address_state_slot, component_root)
                if previous_root != component_root:
                    raise ValueError(
                        f"descriptor set '{descriptor_set.key}' descriptor '{descriptor.key}' assigns address-state slot {operand.address_state_slot} to multiple untied physical components"
                    )

        components: list[_PhysicalComponent] = []
        for component_rows in rows_by_component.values():
            legal_resource_keys: set[str] | None = None
            pre_width = 0
            post_width = 0
            for operand_index, operand, physical_classes in component_rows:
                operand_resource_keys = {_physical_resource_key(register_class) for register_class in physical_classes}
                if legal_resource_keys is None:
                    legal_resource_keys = operand_resource_keys
                else:
                    legal_resource_keys.intersection_update(operand_resource_keys)
                reads_pre, writes_post = _operand_phase_accesses(
                    operand,
                    is_early_clobber=operand_index in early_clobber_results,
                )
                if reads_pre:
                    pre_width = max(pre_width, operand.unit_count)
                if writes_post:
                    post_width = max(post_width, operand.unit_count)
            if not legal_resource_keys:
                operand_names = ", ".join(operand.field_name for _, operand, _ in component_rows)
                raise ValueError(f"descriptor set '{descriptor_set.key}' descriptor '{descriptor.key}' tied physical component [{operand_names}] has no common storage resource")
            components.append(
                _PhysicalComponent(
                    operand_indices=tuple(operand_index for operand_index, _, _ in component_rows),
                    resource_keys=frozenset(legal_resource_keys),
                    pre_width=pre_width,
                    post_width=post_width,
                )
            )

        _validate_physical_metric(
            descriptor_set.key,
            "binding count",
            len(physical_rows),
            limits.maximum_physical_binding_count,
            descriptor_key=descriptor.key,
        )
        # Register-class alternatives are selected by the types in low IR, not
        # by this proof. Every independent component that accepts a resource can
        # therefore select it in the same packet and must fit simultaneously.
        for resource_key, resource_capacity in resources.items():
            resource_components = [component for component in components if resource_key in component.resource_keys]
            if not resource_components:
                continue
            _validate_physical_metric(
                descriptor_set.key,
                f"resource '{resource_key}' component count",
                len(resource_components),
                limits.maximum_resource_component_count,
                descriptor_key=descriptor.key,
            )
            pre_component_count = sum(component.pre_width != 0 for component in resource_components)
            post_component_count = sum(component.post_width != 0 for component in resource_components)
            phase_order_pair_count = math.factorial(pre_component_count) * math.factorial(post_component_count)
            _validate_physical_metric(
                descriptor_set.key,
                f"resource '{resource_key}' phase-order pair count",
                phase_order_pair_count,
                limits.maximum_phase_order_pair_count,
                descriptor_key=descriptor.key,
            )
            base_domains = tuple(
                _component_base_domain(
                    descriptor,
                    component,
                    resource_capacity,
                    early_clobber_results,
                )
                for component in resource_components
            )
            if any(domain.first_at_or_above(0) is None for domain in base_domains) or not _physical_components_admit_placement(resource_components, base_domains):
                raise ValueError(
                    f"descriptor set '{descriptor_set.key}' descriptor '{descriptor.key}' physical resource '{resource_key}' does not admit a legal pre/post placement within {resource_capacity} units"
                )

    _validate_physical_metric(
        descriptor_set.key,
        "maximum address-state slot",
        maximum_address_state_slot,
        limits.maximum_address_state_slot,
    )


def validate_u16(value: int, description: str) -> None:
    if value < 0 or value > 0xFFFF:
        raise ValueError(f"{description} does not fit u16")


def validate_u32(value: int, description: str) -> None:
    if value < 0 or value > 0xFFFFFFFF:
        raise ValueError(f"{description} does not fit u32")


def validate_u16_table_count(count: int, description: str) -> None:
    if count > 0xFFFF:
        raise ValueError(f"{description} count does not fit a u16-indexed table")


def validate_u64(value: int, description: str) -> None:
    if value < 0 or value > 0xFFFFFFFFFFFFFFFF:
        raise ValueError(f"{description} does not fit u64")


def validate_i64(value: int, description: str) -> None:
    if value < -(1 << 63) or value > (1 << 63) - 1:
        raise ValueError(f"{description} does not fit i64")


def validate_register_classes(
    descriptor_set_key: str,
    register_classes: Sequence[RegClass],
) -> None:
    """Validates register classes and their shared storage namespaces."""
    alias_sets: dict[int, list[RegClass]] = {}
    for register_class in register_classes:
        description = f"descriptor set '{descriptor_set_key}' register class '{register_class.name}'"
        if register_class.alloc_unit_bits <= 0:
            raise ValueError(f"{description} allocation unit width must be positive")
        validate_u16(
            register_class.alloc_unit_bits,
            f"{description} allocation unit width",
        )
        validate_u16(
            register_class.target_bank_id,
            f"{description} target bank ID",
        )
        validate_u16(
            register_class.allocatable_count,
            f"{description} allocatable count",
        )
        validate_u16(
            register_class.fixed_location_base,
            f"{description} fixed-location base",
        )
        validate_u16(
            register_class.fixed_location_count,
            f"{description} fixed-location count",
        )
        validate_u16(
            register_class.alias_set_id,
            f"{description} alias-set ID",
        )
        if RegClassFlag.PHYSICAL in register_class.flags and RegClassFlag.VIRTUAL_ONLY in register_class.flags:
            raise ValueError(f"{description} cannot be both physical and virtual-only")
        if register_class.allocatable_count != 0 and RegClassFlag.VIRTUAL_ONLY in register_class.flags:
            raise ValueError(f"{description} has a physical allocation capacity but is virtual-only")
        if (register_class.fixed_location_base != 0 or register_class.fixed_location_count != 0) and RegClassFlag.VIRTUAL_ONLY in register_class.flags:
            raise ValueError(f"{description} has fixed physical locations but is virtual-only")
        if register_class.fixed_location_count == 0 and register_class.fixed_location_base != 0:
            raise ValueError(f"{description} has a fixed-location base without a count")
        fixed_location_end = register_class.fixed_location_base + register_class.fixed_location_count
        if fixed_location_end > 0x10000:
            raise ValueError(f"{description} fixed-location range exceeds the u16 location namespace")
        if register_class.fixed_location_count != 0 and register_class.fixed_location_base < register_class.allocatable_count:
            raise ValueError(f"{description} fixed-location range overlaps its allocatable locations")
        if register_class.alias_set_id != 0:
            alias_sets.setdefault(register_class.alias_set_id, []).append(register_class)

    alias_set_ids = sorted(alias_sets)
    expected_alias_set_ids = list(range(1, len(alias_set_ids) + 1))
    if alias_set_ids != expected_alias_set_ids:
        raise ValueError(f"descriptor set '{descriptor_set_key}' alias-set IDs must be dense from 1; found {alias_set_ids}")

    for alias_set_id, members in alias_sets.items():
        reference = members[0]
        reference_is_physical = reference.allocatable_count != 0 or RegClassFlag.PHYSICAL in reference.flags
        for member in members[1:]:
            member_is_physical = member.allocatable_count != 0 or RegClassFlag.PHYSICAL in member.flags
            if member_is_physical != reference_is_physical:
                raise ValueError(f"descriptor set '{descriptor_set_key}' alias set {alias_set_id} mixes physical and virtual location classes '{reference.name}' and '{member.name}'")
            if member.target_bank_id != reference.target_bank_id:
                raise ValueError(f"descriptor set '{descriptor_set_key}' alias set {alias_set_id} classes '{reference.name}' and '{member.name}' use different target banks")
            if member.allocatable_count != reference.allocatable_count:
                raise ValueError(f"descriptor set '{descriptor_set_key}' alias set {alias_set_id} classes '{reference.name}' and '{member.name}' have different allocatable counts")
            if (member.fixed_location_base, member.fixed_location_count) != (reference.fixed_location_base, reference.fixed_location_count):
                raise ValueError(f"descriptor set '{descriptor_set_key}' alias set {alias_set_id} classes '{reference.name}' and '{member.name}' have different fixed-location ranges")


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


@dataclass(frozen=True, slots=True)
class DescriptorOperandLayout:
    result_count: int
    minimum_packet_operand_count: int
    has_variadic_operands: bool


def validate_descriptor_operands(descriptor: Descriptor) -> DescriptorOperandLayout:
    validate_u16(
        len(descriptor.operands),
        f"descriptor '{descriptor.key}' operand count",
    )
    result_count = 0
    minimum_packet_operand_count = 0
    seen_non_result = False
    variadic_operand_index: int | None = None
    for operand_index, operand in enumerate(descriptor.operands):
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
        if OperandFlag.VARIADIC in operand.flags:
            if variadic_operand_index is not None:
                raise ValueError(f"descriptor '{descriptor.key}' has more than one variadic operand")
            variadic_operand_index = operand_index
            if not operand_role_is_packet_input(operand.role):
                raise ValueError(f"descriptor '{descriptor.key}' variadic operand '{operand.field_name}' is not an explicit packet operand")
            if operand_index != len(descriptor.operands) - 1:
                raise ValueError(f"descriptor '{descriptor.key}' variadic operand '{operand.field_name}' must be the final descriptor operand")
            other_flags = set(operand.flags) - {OperandFlag.VARIADIC}
            if other_flags:
                names = ", ".join(sorted(flag.name.lower() for flag in other_flags))
                raise ValueError(f"descriptor '{descriptor.key}' variadic operand '{operand.field_name}' has unsupported flags: {names}")
            if operand.encoding_field_id != 0 or operand.register_part is not None:
                raise ValueError(f"descriptor '{descriptor.key}' variadic operand '{operand.field_name}' cannot participate in a fixed instruction encoding")
        elif operand_role_is_packet_input(operand.role):
            minimum_packet_operand_count += 1
        if operand.unit_count <= 0:
            raise ValueError(f"descriptor '{descriptor.key}' operand '{operand.field_name}' unit count must be nonzero")
        if OperandFlag.VARIABLE_UNIT_COUNT in operand.flags:
            if operand.role not in (OperandRole.RESULT, OperandRole.OPERAND):
                raise ValueError(f"descriptor '{descriptor.key}' variable-unit operand '{operand.field_name}' must be a result or ordinary operand")
            if OperandFlag.VARIADIC in operand.flags:
                raise ValueError(f"descriptor '{descriptor.key}' variable-unit operand '{operand.field_name}' cannot be variadic")
        if not operand.reg_alts:
            raise ValueError(f"descriptor '{descriptor.key}' operand '{operand.field_name}' has no register-class alternatives")
        validate_u16(
            operand.addressable_unit_count,
            f"descriptor '{descriptor.key}' operand '{operand.field_name}' addressable unit count",
        )
        validate_u16(
            operand.address_state_slot,
            f"descriptor '{descriptor.key}' operand '{operand.field_name}' address state slot",
        )
        is_packet_value = operand_role_is_packet_input(operand.role)
        has_addressable_assignment = is_result or is_packet_value
        if operand.address_map_kind is OperandAddressMapKind.DIRECT:
            if operand.addressable_unit_count != 0:
                raise ValueError(f"descriptor '{descriptor.key}' operand '{operand.field_name}' direct address map must not set an addressable unit count")
            if operand.address_state_slot != 0:
                raise ValueError(f"descriptor '{descriptor.key}' operand '{operand.field_name}' direct address map must not set an address state slot")
        elif operand.address_map_kind in (OperandAddressMapKind.LOW_SUBSET, OperandAddressMapKind.TARGET_STATE):
            if not has_addressable_assignment:
                raise ValueError(f"descriptor '{descriptor.key}' operand '{operand.field_name}' bounded address map must apply to an SSA value operand")
            if operand.addressable_unit_count == 0:
                raise ValueError(f"descriptor '{descriptor.key}' operand '{operand.field_name}' bounded address map must set an addressable unit count")
            if operand.addressable_unit_count < operand.unit_count:
                raise ValueError(f"descriptor '{descriptor.key}' operand '{operand.field_name}' bounded address map covers fewer units than the operand consumes")
            if not any(reg_alt.reg_class is not None for reg_alt in operand.reg_alts):
                raise ValueError(f"descriptor '{descriptor.key}' operand '{operand.field_name}' bounded address map requires a concrete register-class alternative")
            if operand.address_map_kind is OperandAddressMapKind.TARGET_STATE and operand.address_state_slot == 0:
                raise ValueError(f"descriptor '{descriptor.key}' operand '{operand.field_name}' target-state address map must set an address state slot")
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
    if variadic_operand_index is not None:
        if descriptor.constraints:
            raise ValueError(f"descriptor '{descriptor.key}' with variadic operands cannot declare descriptor constraints")
        if descriptor.storage_leases:
            raise ValueError(f"descriptor '{descriptor.key}' with variadic operands cannot declare storage leases")
        if descriptor.operand_forms:
            raise ValueError(f"descriptor '{descriptor.key}' with variadic operands cannot declare operand forms")
    return DescriptorOperandLayout(
        result_count=result_count,
        minimum_packet_operand_count=minimum_packet_operand_count,
        has_variadic_operands=variadic_operand_index is not None,
    )


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


def validate_descriptor_carrier(descriptor: Descriptor, result_count: int) -> None:
    """Validates the canonical Low carrier selected by a descriptor."""
    if descriptor.carrier is DescriptorCarrier.BRANCH:
        if result_count != 0:
            raise ValueError(f"descriptor '{descriptor.key}' uses low.br but declares {result_count} results")
        if descriptor.immediates:
            raise ValueError(f"descriptor '{descriptor.key}' uses low.br but declares ordinary immediates")
        if descriptor.operands:
            raise ValueError(f"descriptor '{descriptor.key}' uses low.br but declares {len(descriptor.operands)} descriptor operands")
        return
    if descriptor.carrier is DescriptorCarrier.SWITCH:
        if result_count != 0:
            raise ValueError(f"descriptor '{descriptor.key}' uses low.switch but declares {result_count} results")
        if descriptor.immediates:
            raise ValueError(f"descriptor '{descriptor.key}' uses low.switch but declares ordinary immediates")
        if (
            len(descriptor.operands) != 1
            or not operand_role_is_packet_input(descriptor.operands[0].role)
            or OperandFlag.OPTIONAL in descriptor.operands[0].flags
            or OperandFlag.VARIADIC in descriptor.operands[0].flags
        ):
            raise ValueError(f"descriptor '{descriptor.key}' uses low.switch but declares a descriptor operand shape other than one required selector")
        return
    if descriptor.carrier not in (DescriptorCarrier.OP, DescriptorCarrier.CONST):
        raise ValueError(f"descriptor '{descriptor.key}' has unknown Low carrier '{descriptor.carrier}'")

    if descriptor.carrier is DescriptorCarrier.OP:
        return
    if result_count != 1:
        raise ValueError(f"descriptor '{descriptor.key}' uses low.const but declares {result_count} results instead of exactly one")
    if any(operand_role_is_packet_input(operand.role) for operand in descriptor.operands):
        raise ValueError(f"descriptor '{descriptor.key}' uses low.const but declares packet operands")
    if descriptor.effects:
        raise ValueError(f"descriptor '{descriptor.key}' uses low.const but declares effects")
    for asm_form in descriptor.asm_forms:
        mnemonic = asm_form_mnemonic(descriptor, asm_form)
        if len(asm_form.results) != 1 or asm_form.operands or asm_form.operand_segments:
            raise ValueError(f"descriptor '{descriptor.key}' low.const asm form '{mnemonic}' must expose exactly one result and no operands")


def descriptor_operand_source_value_indices(
    descriptor: Descriptor,
    result_count: int,
) -> tuple[int | None, ...]:
    """Derives and validates the source value coordinate for every operand row."""
    result_index = 0
    packet_operand_index = 0
    source_value_indices: list[int | None] = []
    for operand in descriptor.operands:
        if operand.role is OperandRole.RESULT:
            source_value_index = result_index
            result_index += 1
        elif operand_role_is_packet_input(operand.role):
            source_value_index = packet_operand_index
            packet_operand_index += 1
        elif operand.role is OperandRole.IMPLICIT:
            source_value_indices.append(None)
            continue
        else:
            raise ValueError(f"descriptor '{descriptor.key}' operand '{operand.field_name}' role {operand.role.name} has no source value coordinate")
        if source_value_index >= 0xFFFF:
            raise ValueError(f"descriptor '{descriptor.key}' operand '{operand.field_name}' source value index {source_value_index} collides with the absent-index sentinel")
        source_value_indices.append(source_value_index)
    if result_index != result_count:
        raise ValueError(f"descriptor '{descriptor.key}' derived {result_index} source result indices for {result_count} results")
    return tuple(source_value_indices)


def _validate_binary_constraint(
    descriptor: Descriptor,
    constraint_index: int,
    constraint_name: str,
    lhs_operand_index: int,
    rhs_operand_index: int | None,
) -> int:
    description = f"descriptor '{descriptor.key}' {constraint_name} constraint {constraint_index}"
    if rhs_operand_index is None:
        raise ValueError(f"{description} requires an rhs operand")
    if lhs_operand_index == rhs_operand_index:
        raise ValueError(f"{description} cannot reference the same operand twice")
    return rhs_operand_index


def _validate_rematerializable_result(
    descriptor: Descriptor,
    result_index: int,
) -> None:
    description = f"descriptor '{descriptor.key}' rematerializable result {result_index}"
    if DescriptorFlag.DEAD_REMOVABLE not in descriptor.flags:
        raise ValueError(f"{description} requires the dead-removable flag")
    forbidden_flags = {
        DescriptorFlag.SIDE_EFFECTING,
        DescriptorFlag.TERMINATOR,
    }.intersection(descriptor.flags)
    if forbidden_flags:
        names = ", ".join(sorted(flag.name.lower() for flag in forbidden_flags))
        raise ValueError(f"{description} has incompatible descriptor flags: {names}")
    if descriptor.effects:
        raise ValueError(f"{description} requires an effect-free descriptor")

    for operand_index, operand in enumerate(descriptor.operands):
        state_flags = {
            OperandFlag.STATE_READ,
            OperandFlag.STATE_WRITE,
        }.intersection(operand.flags)
        if not state_flags:
            continue
        if OperandFlag.SCHEDULE_ONLY_STATE in operand.flags:
            continue
        if state_flags != {OperandFlag.STATE_WRITE} or operand.role is not OperandRole.RESULT or operand_index != result_index:
            raise ValueError(f"{description} cannot replay target state operand '{operand.field_name}'")


def validate_descriptor_constraints(
    descriptor: Descriptor,
) -> tuple[int, ...]:
    """Validates constraints and returns rematerializable result indices."""

    rematerializable_results: set[int] = set()
    for constraint_index, constraint in enumerate(descriptor.constraints):
        lhs_operand_index = constraint.lhs_operand_index
        rhs_operand_index = constraint.rhs_operand_index
        description = f"descriptor '{descriptor.key}' constraint {constraint_index}"
        if lhs_operand_index < 0 or lhs_operand_index >= len(descriptor.operands):
            raise ValueError(f"{description} lhs operand {lhs_operand_index} is out of range")
        if rhs_operand_index is not None and (rhs_operand_index < 0 or rhs_operand_index >= len(descriptor.operands)):
            raise ValueError(f"{description} rhs operand {rhs_operand_index} is out of range")

        lhs = descriptor.operands[lhs_operand_index]
        if constraint.kind in (
            ConstraintKind.TIED,
            ConstraintKind.DESTRUCTIVE,
        ):
            constraint_name = constraint.kind.name.lower()
            rhs_operand_index = _validate_binary_constraint(
                descriptor,
                constraint_index,
                constraint_name,
                lhs_operand_index,
                rhs_operand_index,
            )
            rhs = descriptor.operands[rhs_operand_index]
            if lhs.role is not OperandRole.RESULT or not operand_role_is_packet_input(rhs.role):
                raise ValueError(f"descriptor '{descriptor.key}' {constraint_name} constraint requires a result lhs and packet operand rhs")
        elif constraint.kind is ConstraintKind.COMMUTABLE:
            rhs_operand_index = _validate_binary_constraint(
                descriptor,
                constraint_index,
                "commutable",
                lhs_operand_index,
                rhs_operand_index,
            )
            rhs = descriptor.operands[rhs_operand_index]
            if lhs.role is not OperandRole.OPERAND or rhs.role is not OperandRole.OPERAND:
                raise ValueError(f"descriptor '{descriptor.key}' commutable constraint requires two operand rows")
        elif constraint.kind in (
            ConstraintKind.EARLY_CLOBBER,
            ConstraintKind.REMATERIALIZABLE,
            ConstraintKind.FOLDABLE,
        ):
            if rhs_operand_index is not None or lhs.role is not OperandRole.RESULT:
                raise ValueError(f"descriptor '{descriptor.key}' {constraint.kind.name.lower()} constraint requires one result operand")
            if constraint.kind is ConstraintKind.REMATERIALIZABLE:
                if lhs_operand_index in rematerializable_results:
                    raise ValueError(f"descriptor '{descriptor.key}' repeats rematerializable result {lhs_operand_index}")
                _validate_rematerializable_result(descriptor, lhs_operand_index)
                rematerializable_results.add(lhs_operand_index)

    return tuple(sorted(rematerializable_results))


def validate_descriptor_storage_continuations(
    descriptor: Descriptor,
    register_parts: Mapping[str, RegisterPart],
) -> None:
    """Validates tied partial writes that preserve disjoint source storage."""

    for operand_index, operand in enumerate(descriptor.operands):
        if OperandFlag.STORAGE_CONTINUATION not in operand.flags:
            continue
        description = f"descriptor '{descriptor.key}' storage-continuation operand '{operand.field_name}'"
        if operand.role is OperandRole.RESULT:
            raise ValueError(f"{description} authors a result projection; mark only the tied packet input")
        if not operand_role_is_packet_input(operand.role):
            raise ValueError(f"{description} must be a packet input")
        if OperandFlag.IMPLICIT not in operand.flags:
            raise ValueError(f"{description} must be implicit")

        tied_results = [
            descriptor.operands[constraint.lhs_operand_index] for constraint in descriptor.constraints if constraint.kind is ConstraintKind.TIED and constraint.rhs_operand_index == operand_index
        ]
        if len(tied_results) != 1:
            raise ValueError(f"{description} must be tied to exactly one result")
        result = tied_results[0]
        if result.unit_count != operand.unit_count:
            raise ValueError(f"{description} and tied result must have equal unit counts")
        if operand.register_part is None or result.register_part is None:
            raise ValueError(f"{description} and tied result must name register parts")
        source_part = register_parts.get(operand.register_part)
        result_part = register_parts.get(result.register_part)
        if source_part is None:
            raise ValueError(f"{description} references unknown register part '{operand.register_part}'")
        if result_part is None:
            raise ValueError(f"{description} tied result references unknown register part '{result.register_part}'")
        if source_part.reg_class != result_part.reg_class:
            raise ValueError(f"{description} and tied result use different register classes")
        if source_part.mask & result_part.mask:
            raise ValueError(f"{description} and tied result have overlapping register parts")


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
    return tuple(i for i, operand in enumerate(descriptor.operands) if operand_role_is_packet_input(operand.role))


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
        if StorageLeaseFlag.RELEASE_BEFORE_BOUNDARY in lease.flags and StorageLeaseFlag.MAY_CARRY_ACROSS_BOUNDARY in lease.flags:
            raise ValueError(f"{description} cannot both release before and carry across a boundary")
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
    operand_fields = asm_form.operands
    if asm_form.operand_segments:
        operand_fields = tuple(field_name for segment in asm_form.operand_segments for field_name in segment.operands)
    for field_name in (
        *asm_form.results,
        *operand_fields,
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
