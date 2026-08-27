# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import re
from dataclasses import replace
from itertools import combinations_with_replacement

import pytest

from loom.gen.target.low import compiler, validation
from loom.gen.target.low.compiled import DescriptorAllowlist
from loom.target.low_descriptors import (
    AsmForm,
    Constraint,
    ConstraintKind,
    Descriptor,
    DescriptorSet,
    Operand,
    OperandAddressMapKind,
    OperandFlag,
    OperandRole,
    RegClass,
    RegClassAlt,
    RegClassFlag,
    SpillSlotSpace,
)
from loom.target.test.descriptors import (
    TEST_LOW_ADD_I32_DESCRIPTOR,
    TEST_LOW_CORE_DESCRIPTOR_SET,
)


def _physical_operand(
    field_name: str,
    role: OperandRole,
    register_class: str,
    *,
    flags: tuple[OperandFlag, ...] = (),
    unit_count: int = 1,
    address_map_kind: OperandAddressMapKind = OperandAddressMapKind.DIRECT,
    addressable_unit_count: int = 0,
    address_state_slot: int = 0,
) -> Operand:
    return Operand(
        field_name,
        role,
        (RegClassAlt(register_class),),
        flags=flags,
        unit_count=unit_count,
        address_map_kind=address_map_kind,
        addressable_unit_count=addressable_unit_count,
        address_state_slot=address_state_slot,
    )


def _descriptor(
    key: str,
    operands: tuple[Operand, ...],
    *,
    constraints: tuple[Constraint, ...] = (),
) -> Descriptor:
    result_names = tuple(operand.field_name for operand in operands if operand.role is OperandRole.RESULT)
    operand_names = tuple(
        operand.field_name for operand in operands if operand.role in (OperandRole.OPERAND, OperandRole.PREDICATE, OperandRole.RESOURCE) and OperandFlag.IMPLICIT not in operand.flags
    )
    return replace(
        TEST_LOW_ADD_I32_DESCRIPTOR,
        key=key,
        mnemonic=key,
        semantic_tag=key,
        operands=operands,
        constraints=constraints,
        asm_forms=(AsmForm(results=result_names, operands=operand_names),),
    )


def _descriptor_set(
    *descriptors: Descriptor,
    register_classes: tuple[RegClass, ...] | None = None,
) -> DescriptorSet:
    return replace(
        TEST_LOW_CORE_DESCRIPTOR_SET,
        reg_classes=(TEST_LOW_CORE_DESCRIPTOR_SET.reg_classes if register_classes is None else register_classes),
        register_parts=(),
        descriptors=descriptors,
    )


def test_physical_base_domain_matches_enumerated_contract() -> None:
    windows = tuple(
        validation._PhysicalAddressWindow(
            window_size=window_size,
            maximum_base_remainder=window_size - operand_width,
        )
        for window_size in range(1, 9)
        for operand_width in range(1, window_size + 1)
    )
    window_sets = [()]
    window_sets.extend((window,) for window in windows)
    window_sets.extend(combinations_with_replacement(windows, 2))

    for maximum_base in (-1, 0, 1, 7, 8, 15, 31):
        for address_windows in window_sets:
            expected_bases = tuple(base for base in range(maximum_base + 1) if all(base % window.window_size <= window.maximum_base_remainder for window in address_windows))
            domain = validation._PhysicalBaseDomain(
                maximum_base=maximum_base,
                address_windows=tuple(address_windows),
            )
            for lower_bound in range(maximum_base + 2):
                expected = next(
                    (base for base in expected_bases if base >= lower_bound),
                    None,
                )
                assert domain.first_at_or_above(lower_bound) == expected


def test_physical_component_base_domain_combines_address_maps() -> None:
    descriptor = _descriptor(
        "test.address.maps",
        (
            _physical_operand(
                "dst",
                OperandRole.RESULT,
                "test.phys",
                unit_count=2,
            ),
            _physical_operand(
                "bounded",
                OperandRole.OPERAND,
                "test.phys",
                unit_count=2,
                address_map_kind=OperandAddressMapKind.LOW_SUBSET,
                addressable_unit_count=6,
            ),
            _physical_operand(
                "windowed",
                OperandRole.OPERAND,
                "test.phys",
                unit_count=2,
                address_map_kind=OperandAddressMapKind.TARGET_STATE,
                addressable_unit_count=4,
                address_state_slot=1,
            ),
        ),
        constraints=(
            Constraint(ConstraintKind.TIED, 0, 1),
            Constraint(ConstraintKind.TIED, 0, 2),
        ),
    )
    component = validation._PhysicalComponent(
        operand_indices=(0, 1, 2),
        resource_keys=frozenset(("class:test.phys",)),
        pre_width=2,
        post_width=2,
    )

    domain = validation._component_base_domain(
        descriptor,
        component,
        resource_capacity=16,
        early_clobber_results=frozenset(),
    )

    assert domain.maximum_base == 4
    assert domain.first_at_or_above(0) == 0
    assert domain.first_at_or_above(2) == 2
    assert domain.first_at_or_above(3) == 4
    assert domain.first_at_or_above(5) is None


def test_physical_descriptor_set_accepts_reserved_resource_envelope() -> None:
    register_classes = (
        RegClass(
            "envelope.large",
            32,
            SpillSlotSpace.PRIVATE,
            flags=(RegClassFlag.PHYSICAL,),
            allocatable_count=2048,
        ),
        *(
            RegClass(
                f"envelope.small{index}",
                32,
                SpillSlotSpace.PRIVATE,
                flags=(RegClassFlag.PHYSICAL,),
                allocatable_count=18,
            )
            for index in range(13)
        ),
        RegClass(
            "envelope.tail",
            32,
            SpillSlotSpace.PRIVATE,
            flags=(RegClassFlag.PHYSICAL,),
            allocatable_count=22,
        ),
    )
    descriptor = _descriptor(
        "test.envelope",
        tuple(
            _physical_operand(
                f"src{index}",
                OperandRole.OPERAND,
                register_class.name,
                unit_count=64 if index == 0 else 1,
            )
            for index, register_class in enumerate(register_classes)
        ),
    )

    validation.validate_physical_descriptor_set(_descriptor_set(descriptor, register_classes=register_classes))


@pytest.mark.parametrize(
    ("register_classes", "message"),
    [
        (
            tuple(
                RegClass(
                    f"resource{index}",
                    32,
                    SpillSlotSpace.PRIVATE,
                    flags=(RegClassFlag.PHYSICAL,),
                    allocatable_count=1,
                )
                for index in range(16)
            ),
            r"canonical resource count 16 exceeds generation bound 15",
        ),
        (
            (
                RegClass(
                    "resource.large",
                    32,
                    SpillSlotSpace.PRIVATE,
                    flags=(RegClassFlag.PHYSICAL,),
                    allocatable_count=2049,
                ),
            ),
            r"maximum resource capacity 2049 exceeds generation bound 2048",
        ),
        (
            (
                RegClass(
                    "resource.large",
                    32,
                    SpillSlotSpace.PRIVATE,
                    flags=(RegClassFlag.PHYSICAL,),
                    allocatable_count=2048,
                ),
                RegClass(
                    "resource.tail",
                    32,
                    SpillSlotSpace.PRIVATE,
                    flags=(RegClassFlag.PHYSICAL,),
                    allocatable_count=257,
                ),
            ),
            r"aggregate resource capacity 2305 exceeds generation bound 2304",
        ),
    ],
)
def test_physical_descriptor_set_rejects_resource_envelope_overflow(
    register_classes: tuple[RegClass, ...],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        validation.validate_physical_descriptor_set(_descriptor_set(register_classes=register_classes))


def test_physical_descriptor_set_rejects_zero_capacity_resource() -> None:
    register_class = RegClass(
        "test.zero_capacity",
        32,
        SpillSlotSpace.PRIVATE,
        flags=(RegClassFlag.PHYSICAL,),
    )
    descriptor = _descriptor(
        "test.zero_capacity",
        (_physical_operand("dst", OperandRole.RESULT, register_class.name),),
    )

    with pytest.raises(
        ValueError,
        match=r"physical register class 'test.zero_capacity' has zero allocation capacity",
    ):
        compiler.compile_descriptor_set(_descriptor_set(descriptor, register_classes=(register_class,)))


def test_physical_descriptor_set_rejects_implicit_row_without_phase() -> None:
    descriptor = _descriptor(
        "test.bad.implicit",
        (
            _physical_operand(
                "state",
                OperandRole.IMPLICIT,
                "test.special",
                flags=(OperandFlag.IMPLICIT,),
            ),
        ),
    )

    with pytest.raises(
        ValueError,
        match=re.escape("descriptor set 'test.low.core' descriptor 'test.bad.implicit' physical implicit operand 'state' has no state read or write phase"),
    ):
        compiler.compile_descriptor_set(_descriptor_set(descriptor))


def test_physical_descriptor_set_accepts_native_implicit_packet_operand() -> None:
    descriptor = _descriptor(
        "test.assembly.implicit",
        (
            _physical_operand("dst", OperandRole.RESULT, "test.special"),
            _physical_operand(
                "src",
                OperandRole.OPERAND,
                "test.special",
                flags=(OperandFlag.IMPLICIT,),
            ),
        ),
    )

    compiler.compile_descriptor_set(_descriptor_set(descriptor))


def test_physical_descriptor_set_accounts_for_native_implicit_packet_operands() -> None:
    descriptor = _descriptor(
        "test.bad.assembly.implicit",
        (
            TEST_LOW_ADD_I32_DESCRIPTOR.operands[0],
            _physical_operand(
                "lhs",
                OperandRole.OPERAND,
                "test.special",
                flags=(OperandFlag.IMPLICIT,),
            ),
            _physical_operand(
                "rhs",
                OperandRole.OPERAND,
                "test.special",
                flags=(OperandFlag.IMPLICIT,),
            ),
        ),
    )

    with pytest.raises(
        ValueError,
        match=r"physical resource 'class:test.special' does not admit a legal pre/post placement",
    ):
        compiler.compile_descriptor_set(_descriptor_set(descriptor))


def test_physical_descriptor_set_rejects_component_width_overflow() -> None:
    descriptor = _descriptor(
        "test.bad.component.width",
        (
            _physical_operand(
                "dst",
                OperandRole.RESULT,
                "test.phys",
                unit_count=65,
            ),
        ),
    )

    with pytest.raises(
        ValueError,
        match=r"operand 'dst' unit count 65 exceeds generation bound 64",
    ):
        compiler.compile_descriptor_set(_descriptor_set(descriptor))


def test_physical_descriptor_set_rejects_packet_binding_overflow() -> None:
    descriptor = _descriptor(
        "test.bad.binding.count",
        tuple(
            _physical_operand(
                f"src{index}",
                OperandRole.OPERAND,
                "test.phys",
            )
            for index in range(16)
        ),
    )

    with pytest.raises(
        ValueError,
        match=r"binding count 16 exceeds generation bound 15",
    ):
        compiler.compile_descriptor_set(_descriptor_set(descriptor))


def test_physical_descriptor_set_rejects_resource_component_overflow() -> None:
    descriptor = _descriptor(
        "test.bad.resource.component.count",
        tuple(
            _physical_operand(
                f"src{index}",
                OperandRole.OPERAND,
                "test.phys",
            )
            for index in range(9)
        ),
    )

    with pytest.raises(
        ValueError,
        match=r"resource 'class:test.phys' component count 9 exceeds generation bound 8",
    ):
        compiler.compile_descriptor_set(_descriptor_set(descriptor))


def test_physical_descriptor_set_proves_every_register_alternative() -> None:
    register_classes = tuple(
        RegClass(
            f"test.alternative.{name}",
            32,
            SpillSlotSpace.PRIVATE,
            flags=(RegClassFlag.PHYSICAL,),
            allocatable_count=capacity,
        )
        for name, capacity in (("small", 1), ("large", 2))
    )
    alternatives = tuple(RegClassAlt(register_class.name) for register_class in register_classes)
    descriptor = _descriptor(
        "test.bad.alternative.capacity",
        tuple(
            replace(
                _physical_operand(
                    f"src{index}",
                    OperandRole.OPERAND,
                    register_classes[0].name,
                ),
                reg_alts=alternatives,
            )
            for index in range(2)
        ),
    )

    with pytest.raises(
        ValueError,
        match=(
            r"physical resource 'class:test.alternative.small' does not "
            r"admit a legal pre/post placement"
        ),
    ):
        compiler.compile_descriptor_set(_descriptor_set(descriptor, register_classes=register_classes))


def test_physical_descriptor_set_rejects_cross_resource_tie() -> None:
    descriptor = _descriptor(
        "test.bad.tie",
        (
            _physical_operand("dst", OperandRole.RESULT, "test.phys"),
            _physical_operand("src", OperandRole.OPERAND, "test.special"),
        ),
        constraints=(Constraint(ConstraintKind.TIED, 0, 1),),
    )

    with pytest.raises(
        ValueError,
        match=r"tied physical component \[dst, src\] has no common storage resource",
    ):
        compiler.compile_descriptor_set(_descriptor_set(descriptor))


def test_physical_descriptor_set_rejects_impossible_phase_capacity() -> None:
    descriptor = _descriptor(
        "test.bad.capacity",
        (
            TEST_LOW_ADD_I32_DESCRIPTOR.operands[0],
            _physical_operand("lhs", OperandRole.OPERAND, "test.special"),
            _physical_operand("rhs", OperandRole.OPERAND, "test.special"),
        ),
    )

    with pytest.raises(
        ValueError,
        match=(
            r"descriptor 'test.bad.capacity' physical resource "
            r"'class:test.special' does not admit a legal pre/post placement"
        ),
    ):
        compiler.compile_descriptor_set(_descriptor_set(descriptor))


def test_physical_descriptor_set_accounts_for_early_clobber_results() -> None:
    register_class = RegClass(
        "test.early_clobber",
        32,
        SpillSlotSpace.PRIVATE,
        flags=(RegClassFlag.PHYSICAL,),
        allocatable_count=1,
    )
    descriptor = _descriptor(
        "test.bad.early_clobber",
        (
            _physical_operand("dst", OperandRole.RESULT, register_class.name),
            _physical_operand("src", OperandRole.OPERAND, register_class.name),
        ),
        constraints=(Constraint(ConstraintKind.EARLY_CLOBBER, 0),),
    )

    with pytest.raises(
        ValueError,
        match=r"physical resource 'class:test.early_clobber' does not admit a legal pre/post placement",
    ):
        compiler.compile_descriptor_set(_descriptor_set(descriptor, register_classes=(register_class,)))


def test_physical_descriptor_set_accepts_tied_width_transition() -> None:
    descriptor = _descriptor(
        "test.tied.width",
        (
            _physical_operand("dst", OperandRole.RESULT, "test.phys", unit_count=4),
            _physical_operand("src", OperandRole.OPERAND, "test.phys", unit_count=2),
        ),
        constraints=(Constraint(ConstraintKind.TIED, 0, 1),),
    )

    validation.validate_physical_descriptor_set(_descriptor_set(descriptor))


def test_physical_descriptor_set_rejects_incompatible_address_windows() -> None:
    descriptor = _descriptor(
        "test.bad.address.window",
        (
            TEST_LOW_ADD_I32_DESCRIPTOR.operands[0],
            _physical_operand(
                "lhs",
                OperandRole.OPERAND,
                "test.phys",
                unit_count=2,
                address_map_kind=OperandAddressMapKind.LOW_SUBSET,
                addressable_unit_count=2,
            ),
            _physical_operand(
                "rhs",
                OperandRole.OPERAND,
                "test.phys",
                unit_count=2,
                address_map_kind=OperandAddressMapKind.LOW_SUBSET,
                addressable_unit_count=2,
            ),
        ),
    )

    with pytest.raises(
        ValueError,
        match=r"physical resource 'class:test.phys' does not admit a legal pre/post placement",
    ):
        compiler.compile_descriptor_set(_descriptor_set(descriptor))


def test_physical_descriptor_set_rejects_shared_address_state_slot() -> None:
    descriptor = _descriptor(
        "test.bad.shared.state.slot",
        (
            _physical_operand(
                "dst",
                OperandRole.RESULT,
                "test.phys",
                address_map_kind=OperandAddressMapKind.LOW_SUBSET,
                addressable_unit_count=8,
                address_state_slot=1,
            ),
            _physical_operand(
                "src",
                OperandRole.OPERAND,
                "test.phys",
                address_map_kind=OperandAddressMapKind.TARGET_STATE,
                addressable_unit_count=8,
                address_state_slot=1,
            ),
        ),
    )

    with pytest.raises(
        ValueError,
        match=(
            r"assigns address-state slot 1 to multiple untied physical "
            r"components"
        ),
    ):
        compiler.compile_descriptor_set(_descriptor_set(descriptor))


def test_physical_descriptor_set_accepts_tied_target_state_slot() -> None:
    descriptor = _descriptor(
        "test.tied.state.slot",
        (
            _physical_operand(
                "dst",
                OperandRole.RESULT,
                "test.phys",
                address_map_kind=OperandAddressMapKind.TARGET_STATE,
                addressable_unit_count=8,
                address_state_slot=15,
            ),
            _physical_operand(
                "src",
                OperandRole.OPERAND,
                "test.phys",
                address_map_kind=OperandAddressMapKind.TARGET_STATE,
                addressable_unit_count=8,
                address_state_slot=15,
            ),
        ),
        constraints=(Constraint(ConstraintKind.TIED, 0, 1),),
    )

    validation.validate_physical_descriptor_set(_descriptor_set(descriptor))


def test_physical_descriptor_set_rejects_inconsistent_target_state_window() -> None:
    descriptors = tuple(
        _descriptor(
            f"test.state.window{window_width}",
            (
                _physical_operand(
                    "dst",
                    OperandRole.RESULT,
                    "test.phys",
                    address_map_kind=OperandAddressMapKind.TARGET_STATE,
                    addressable_unit_count=window_width,
                    address_state_slot=1,
                ),
            ),
        )
        for window_width in (8, 16)
    )

    with pytest.raises(
        ValueError,
        match=r"address-state slot 1 has inconsistent window width",
    ):
        compiler.compile_descriptor_set(_descriptor_set(*descriptors))


def test_physical_descriptor_set_rejects_address_state_slot_overflow() -> None:
    descriptor = _descriptor(
        "test.bad.state.slot",
        (
            _physical_operand(
                "dst",
                OperandRole.RESULT,
                "test.phys",
                address_map_kind=OperandAddressMapKind.LOW_SUBSET,
                addressable_unit_count=8,
                address_state_slot=16,
            ),
        ),
    )

    with pytest.raises(
        ValueError,
        match=r"maximum address-state slot 16 exceeds generation bound 15",
    ):
        compiler.compile_descriptor_set(_descriptor_set(descriptor))


def test_physical_descriptor_set_rejects_phase_order_work_overflow() -> None:
    operands = (*(_physical_operand(f"dst{index}", OperandRole.RESULT, "test.phys") for index in range(5)), *(_physical_operand(f"src{index}", OperandRole.OPERAND, "test.phys") for index in range(5)))
    constraints = tuple(Constraint(ConstraintKind.TIED, result_index, result_index + 5) for result_index in range(5))
    descriptor = _descriptor(
        "test.bad.phase.work",
        operands,
        constraints=constraints,
    )

    with pytest.raises(
        ValueError,
        match=r"phase-order pair count 14400 exceeds generation bound 1440",
    ):
        compiler.compile_descriptor_set(_descriptor_set(descriptor))


def test_physical_descriptor_set_accepts_phase_order_work_envelope() -> None:
    descriptor = _descriptor(
        "test.phase.work.envelope",
        (
            *(_physical_operand(f"dst{index}", OperandRole.RESULT, "test.phys") for index in range(2)),
            *(_physical_operand(f"src{index}", OperandRole.OPERAND, "test.phys") for index in range(6)),
        ),
    )

    validation.validate_physical_descriptor_set(_descriptor_set(descriptor))


def test_physical_descriptor_set_validates_unselected_descriptors() -> None:
    good_descriptor = _descriptor(
        "test.good.selected",
        (TEST_LOW_ADD_I32_DESCRIPTOR.operands[0],),
    )
    bad_descriptor = _descriptor(
        "test.bad.unselected",
        (
            TEST_LOW_ADD_I32_DESCRIPTOR.operands[0],
            _physical_operand("lhs", OperandRole.OPERAND, "test.special"),
            _physical_operand("rhs", OperandRole.OPERAND, "test.special"),
        ),
    )

    with pytest.raises(ValueError, match=r"descriptor 'test.bad.unselected'"):
        compiler.compile_descriptor_set(
            _descriptor_set(good_descriptor, bad_descriptor),
            DescriptorAllowlist(keys=(good_descriptor.key,)),
        )
