# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from itertools import combinations

from loom.target.arch.amd.xdna.aie.schedule import (
    PipelineStageKind,
    pipeline_uses,
)
from loom.target.arch.amd.xdna.aie2p.core_descriptors import (
    _BUNDLE_SLOT_EXCLUSIONS,
    _DESCRIPTOR_SPECS,
    _INSTRUCTION_ENCODINGS,
    _SCHEDULE_CLASS_NAMES,
    _SLOT_RESOURCE_KINDS,
    AIE2P_CORE_DESCRIPTOR_SET,
    _bundle_exclusion_resource_name,
    _constraints,
    _itinerary,
    _low_register_class_name,
    _pipeline_resource_name,
    _slot_resource_name,
)
from loom.target.arch.amd.xdna.aie2p.core_encoding_data import CORE_ENCODING_TABLE
from loom.target.arch.amd.xdna.aie2p.core_machine_data import CORE_MACHINE_TABLE
from loom.target.arch.amd.xdna.aie2p.core_schedule_data import CORE_SCHEDULE_TABLE
from loom.target.low_descriptors import (
    ConstraintKind,
    EffectKind,
    IssueUseKind,
    OperandFlag,
    OperandRole,
    RegClassAltFlag,
    RegClassFlag,
    ResourceKind,
)


def test_core_descriptor_closure_is_complete() -> None:
    descriptor_set = AIE2P_CORE_DESCRIPTOR_SET
    assert len(descriptor_set.physical_registers) == 359
    assert len(descriptor_set.reg_classes) == 11
    assert len(descriptor_set.descriptors) == 57
    assert tuple(row.name for row in descriptor_set.physical_registers) == tuple(
        row.name for row in CORE_MACHINE_TABLE.physical_registers
    )
    assert tuple(
        row.atomic_units for row in descriptor_set.physical_registers
    ) == tuple(row.atomic_units for row in CORE_MACHINE_TABLE.physical_registers)


def test_complete_schedule_domain_drives_selected_low_descriptors() -> None:
    descriptor_set = AIE2P_CORE_DESCRIPTOR_SET
    assert len(descriptor_set.resources) == 90
    assert len(descriptor_set.timing_events) == 38
    assert len(descriptor_set.event_separations) == 651
    assert len(descriptor_set.schedule_classes) == 23
    assert {
        resource.name
        for resource in descriptor_set.resources
        if resource.kind is ResourceKind.PIPELINE
    } == {
        *(
            _pipeline_resource_name(resource)
            for resource in CORE_SCHEDULE_TABLE.resources
        ),
        *(
            _bundle_exclusion_resource_name(exclusion)
            for exclusion in _BUNDLE_SLOT_EXCLUSIONS
        ),
    }

    schedule_classes = {
        schedule_class.name: schedule_class
        for schedule_class in descriptor_set.schedule_classes
    }
    for spec in _DESCRIPTOR_SPECS:
        schedule_class = schedule_classes[
            _SCHEDULE_CLASS_NAMES[(spec.form_name, spec.itinerary)]
        ]
        slot = _INSTRUCTION_ENCODINGS[spec.form_name].slot
        assert schedule_class.issue_uses[0].resource == _slot_resource_name(slot)
        assert schedule_class.issue_uses[0].stage == 0
        assert schedule_class.issue_uses[0].cycles == 1
        assert schedule_class.issue_uses[0].kind is IssueUseKind.REQUIRED
        expected_exclusions = tuple(
            exclusion for exclusion in _BUNDLE_SLOT_EXCLUSIONS if slot in exclusion
        )
        exclusion_uses = schedule_class.issue_uses[1 : 1 + len(expected_exclusions)]
        assert tuple(use.resource for use in exclusion_uses) == tuple(
            _bundle_exclusion_resource_name(exclusion)
            for exclusion in expected_exclusions
        )
        assert all(use.stage == 0 for use in exclusion_uses)
        assert all(use.cycles == 1 for use in exclusion_uses)
        assert all(use.units == 1 for use in exclusion_uses)
        assert all(use.kind is IssueUseKind.REQUIRED for use in exclusion_uses)
        expected_pipeline_uses = pipeline_uses(_itinerary(spec))
        assert len(schedule_class.issue_uses) == (
            len(expected_pipeline_uses) + len(expected_exclusions) + 1
        )
        for actual, expected in zip(
            schedule_class.issue_uses[1 + len(expected_exclusions) :],
            expected_pipeline_uses,
            strict=True,
        ):
            assert len(expected.resources) == 1
            assert actual.resource == _pipeline_resource_name(expected.resources[0])
            assert actual.stage == expected.start_cycle
            assert actual.cycles == expected.cycles
            assert actual.units == 1
            assert actual.kind is (
                IssueUseKind.REQUIRED
                if expected.kind is PipelineStageKind.REQUIRED
                else IssueUseKind.RESERVED
            )


def test_bundle_resources_exactly_model_every_extendable_physical_slot_set() -> None:
    descriptor_set = AIE2P_CORE_DESCRIPTOR_SET
    resources = {resource.name: resource for resource in descriptor_set.resources}
    for exclusion in _BUNDLE_SLOT_EXCLUSIONS:
        resource = resources[_bundle_exclusion_resource_name(exclusion)]
        assert resource.capacity_per_cycle == len(exclusion) - 1
        assert resource.kind is ResourceKind.PIPELINE

    legal_signatures = {
        frozenset(field.slot for field in bundle_format.fields)
        for bundle_format in CORE_ENCODING_TABLE.bundle_formats
    }
    slots = tuple(sorted(_SLOT_RESOURCE_KINDS))
    for slot_count in range(1, len(slots) + 1):
        for candidate in combinations(slots, slot_count):
            admitted_by_resources = all(
                len(set(candidate).intersection(exclusion)) < len(exclusion)
                for exclusion in _BUNDLE_SLOT_EXCLUSIONS
            )
            extendable_to_bundle = any(
                frozenset(candidate).issubset(signature)
                for signature in legal_signatures
            )
            assert admitted_by_resources == extendable_to_bundle


def test_low_register_classes_retain_machine_candidate_order() -> None:
    machine_classes = {
        _low_register_class_name(row.name): row
        for row in CORE_MACHINE_TABLE.register_classes
    }
    assert len(machine_classes) == len(CORE_MACHINE_TABLE.register_classes)
    for register_class in AIE2P_CORE_DESCRIPTOR_SET.reg_classes:
        if register_class.name.startswith("aie2p.state."):
            assert len(register_class.physical_registers) == 1
        else:
            assert (
                register_class.physical_registers
                == machine_classes[register_class.name].candidates
            )
        assert RegClassFlag.PHYSICAL in register_class.flags
        assert RegClassFlag.EXPLICIT_PHYSICAL_REGISTERS in register_class.flags
        assert RegClassFlag.UNSPILLABLE in register_class.flags


def test_vector_encoding_roles_share_one_low_storage_class() -> None:
    descriptors = {
        descriptor.key: descriptor
        for descriptor in AIE2P_CORE_DESCRIPTOR_SET.descriptors
    }
    vector_keys = (
        "amd.xdna.aie2p.load.a.i8x64.indexed.immediate",
        "amd.xdna.aie2p.load.b.i8x64.indexed.immediate",
        "amd.xdna.aie2p.add.i8x64",
        "amd.xdna.aie2p.store.i8x64.indexed.immediate",
    )
    vector_register_classes = [
        alternative.reg_class
        for key in vector_keys
        for operand in descriptors[key].operands
        if operand.encoding_adapter_id != 0
        for alternative in operand.reg_alts
    ]
    assert vector_register_classes
    assert set(vector_register_classes) == {"aie2p.vec512"}
    assert (
        len(
            {
                operand.encoding_adapter_id
                for key in vector_keys
                for operand in descriptors[key].operands
                if operand.encoding_adapter_id != 0
            }
        )
        == 5
    )


def test_descriptor_encoding_ids_and_adapters_are_materialized() -> None:
    descriptors = {
        descriptor.key: descriptor
        for descriptor in AIE2P_CORE_DESCRIPTOR_SET.descriptors
    }
    vector_add = descriptors["amd.xdna.aie2p.add.i32x16"]
    assert vector_add.encoding_id != 0
    assert [operand.field_name for operand in vector_add.operands] == [
        "d",
        "s1",
        "s2",
    ]
    assert all(operand.encoding_field_id != 0 for operand in vector_add.operands)
    assert all(operand.encoding_adapter_id != 0 for operand in vector_add.operands)
    assert vector_add.operands[0].ready_stage == 2
    assert vector_add.operands[1].read_stage == 1

    vector_load = descriptors["amd.xdna.aie2p.load.a.i8x64.indexed.immediate"]
    assert vector_load.operands[0].ready_stage == 7

    short_constant = descriptors["amd.xdna.aie2p.constant.i32.short"]
    full_constant = descriptors["amd.xdna.aie2p.constant.i32"]
    for descriptor in (short_constant, full_constant):
        assert descriptor.operands[0].reg_alts[0].reg_class == "aie2p.er"
        assert descriptor.operands[0].encoding_adapter_id != 0
        assert descriptor.operands[0].ready_stage == 1

    scalar_move = descriptors["amd.xdna.aie2p.move.scalar"]
    assert [operand.field_name for operand in scalar_move.operands] == [
        "dst",
        "src",
    ]
    assert all(
        operand.reg_alts[0].reg_class == "aie2p.er" for operand in scalar_move.operands
    )
    assert all(operand.encoding_adapter_id != 0 for operand in scalar_move.operands)
    assert scalar_move.operands[0].ready_stage == 1
    assert scalar_move.operands[1].read_stage == 1

    predicate_compare = descriptors["amd.xdna.aie2p.cmp.eqz.i8x64"]
    assert [operand.field_name for operand in predicate_compare.operands] == [
        "cmp",
        "s2",
    ]
    assert predicate_compare.operands[0].reg_alts[0].reg_class == "aie2p.el"
    assert predicate_compare.operands[1].reg_alts[0].reg_class == "aie2p.vec512"

    byte_select = descriptors["amd.xdna.aie2p.select.i8x64"]
    word_select = descriptors["amd.xdna.aie2p.select.i32x16"]
    assert [operand.field_name for operand in byte_select.operands] == [
        "d",
        "s1",
        "s2",
        "sel",
    ]
    assert [operand.reg_alts[0].reg_class for operand in byte_select.operands] == [
        "aie2p.vec512",
        "aie2p.vec512",
        "aie2p.vec512",
        "aie2p.el",
    ]
    assert word_select.operands[-1].reg_alts[0].reg_class == "aie2p.ers16"

    scalar_selector = descriptors["amd.xdna.aie2p.select.mask.i32"]
    assert scalar_selector.operands[0].field_name == "d0"
    assert scalar_selector.operands[0].reg_alts[0].reg_class == "aie2p.ers16"
    assert scalar_selector.operands[1].field_name == "s0"
    assert scalar_selector.operands[1].reg_alts[0].reg_class == "aie2p.er"


def test_implicit_registers_and_machine_ties_reach_low() -> None:
    descriptors = {
        descriptor.key: descriptor
        for descriptor in AIE2P_CORE_DESCRIPTOR_SET.descriptors
    }
    for key in (
        "amd.xdna.aie2p.add.i32.immediate",
        "amd.xdna.aie2p.add.i32",
        "amd.xdna.aie2p.sub.i32",
    ):
        carry = next(
            operand
            for operand in descriptors[key].operands
            if operand.field_name == "implicit_def_srcarry"
        )
        assert carry.role is OperandRole.IMPLICIT
        assert set(carry.flags) == {
            OperandFlag.IMPLICIT,
            OperandFlag.STATE_WRITE,
        }
        assert carry.reg_alts[0].reg_class == "aie2p.state.srcarry"
        assert carry.reg_alts[0].flags == (RegClassAltFlag.PHYSICAL_ONLY,)
        assert carry.ready_stage == 1

    link_register = descriptors["amd.xdna.aie2p.return"].operands[0]
    assert link_register.field_name == "implicit_use_lr"
    assert set(link_register.flags) == {
        OperandFlag.IMPLICIT,
        OperandFlag.STATE_READ,
    }
    assert link_register.reg_alts[0].reg_class == "aie2p.state.lr"

    vector_extract = descriptors["amd.xdna.aie2p.extract.i8.immediate"]
    vector_add_sign = next(
        operand
        for operand in vector_extract.operands
        if operand.field_name == "implicit_use_vaddsign0"
    )
    assert vector_add_sign.reg_alts[0].reg_class == "aie2p.state.vaddsign0"
    assert vector_add_sign.reg_alts[0].flags == (RegClassAltFlag.PHYSICAL_ONLY,)
    vector_add_sign_class = next(
        register_class
        for register_class in AIE2P_CORE_DESCRIPTOR_SET.reg_classes
        if register_class.name == "aie2p.state.vaddsign0"
    )
    assert vector_add_sign_class.alloc_unit_bits == 32
    assert vector_add_sign_class.physical_registers == ("vaddSign0",)

    vector_insert = descriptors["amd.xdna.aie2p.insert.i32.register"]
    insert_index = next(
        operand for operand in vector_insert.operands if operand.field_name == "idx"
    )
    assert insert_index.reg_alts[0].reg_class == "aie2p.mr29_insert"
    insert_index_class = next(
        register_class
        for register_class in AIE2P_CORE_DESCRIPTOR_SET.reg_classes
        if register_class.name == "aie2p.mr29_insert"
    )
    assert insert_index_class.physical_registers == ("r29",)

    divs = next(form for form in CORE_MACHINE_TABLE.forms if form.name == "DIVS")
    constraints = _constraints(divs, (*divs.outputs, *divs.inputs))
    assert len(constraints) == 1
    assert constraints[0].kind is ConstraintKind.TIED
    assert constraints[0].lhs_operand_index == 1
    assert constraints[0].rhs_operand_index == 2


def test_seed_schedule_contract_retains_endpoint_events_and_separations() -> None:
    descriptor_set = AIE2P_CORE_DESCRIPTOR_SET
    descriptors = {row.key: row for row in descriptor_set.descriptors}
    separations = {
        (row.producer_event, row.consumer_event): row.minimum_issue_separation_cycles
        for row in descriptor_set.event_separations
    }

    scalar_add = descriptors["amd.xdna.aie2p.add.i32.immediate"]
    scalar_write = next(
        row.write_event for row in scalar_add.operands if row.role is OperandRole.RESULT
    )
    scalar_read = next(
        row.read_event for row in scalar_add.operands if row.role is OperandRole.OPERAND
    )
    scalar_store = descriptors["amd.xdna.aie2p.store.scalar.indexed.immediate"]
    scalar_store_read = next(
        row.read_event for row in scalar_store.operands if row.field_name == "src"
    )
    assert separations[scalar_write, scalar_read] == 1
    assert separations[scalar_write, scalar_write] == 1
    assert separations[scalar_read, scalar_write] == 0
    assert separations[scalar_write, scalar_store_read] == 1

    scalar_mul = descriptors["amd.xdna.aie2p.mul.i32"]
    multiply_write = next(
        row.write_event for row in scalar_mul.operands if row.role is OperandRole.RESULT
    )
    assert scalar_mul.operands[0].ready_stage == 2
    assert separations[multiply_write, scalar_read] == 2

    vector_add = descriptors["amd.xdna.aie2p.add.i32x16"]
    vector_write = next(
        row.write_event for row in vector_add.operands if row.role is OperandRole.RESULT
    )
    vector_read = next(
        row.read_event for row in vector_add.operands if row.role is OperandRole.OPERAND
    )
    vector_load = descriptors["amd.xdna.aie2p.load.a.i8x64.indexed.immediate"]
    load_write = next(
        row.write_event
        for row in vector_load.operands
        if row.role is OperandRole.RESULT
    )
    vector_store = descriptors["amd.xdna.aie2p.store.i8x64.indexed.immediate"]
    vector_store_read = next(
        row.read_event for row in vector_store.operands if row.field_name == "src"
    )
    assert separations[load_write, vector_read] == 7
    assert separations[vector_write, vector_read] == 1
    assert separations[vector_write, vector_write] == 1
    assert separations[vector_read, vector_write] == 0
    assert separations[vector_write, vector_store_read] == 2

    memory_write = next(
        row.timing_event for row in vector_store.effects if row.kind is EffectKind.WRITE
    )
    memory_read = next(
        row.timing_event for row in vector_load.effects if row.kind is EffectKind.READ
    )
    assert separations[memory_write, memory_read] == 1
    assert vector_load.immediates[0].encoding_field_id != 0
    assert vector_load.immediates[0].encoding_id != 0
    assert vector_load.immediates[0].value_step == 64

    scalar_add = descriptors["amd.xdna.aie2p.add.i32.immediate"]
    assert scalar_add.immediates[0].value_step == 1
