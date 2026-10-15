# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""AIE2P native itineraries, issue resources, and event timing."""

from __future__ import annotations

from dataclasses import replace
from itertools import combinations, product

import pytest

from loom.target.arch.amd.xdna.aie.machine import has_property
from loom.target.arch.amd.xdna.aie.schedule import (
    DependencyKind,
    PipelineStageKind,
    bypass_class,
    dependency_separation,
    itinerary_payload,
    pipeline_uses,
)
from loom.target.arch.amd.xdna.aie2p.core_descriptor_specs import (
    _DESCRIPTOR_SPECS,
    _LOCK_EFFECT,
    _MACHINE_FORMS,
)
from loom.target.arch.amd.xdna.aie2p.core_descriptors import (
    _BUNDLE_SLOT_EXCLUSIONS,
    _INSTRUCTION_ENCODINGS,
    _REGISTER_ENDPOINTS,
    _SCHEDULE_CLASS_NAMES,
    _SLOT_RESOURCE_KINDS,
    AIE2P_CORE_DESCRIPTOR_SET,
    _bundle_exclusion_resource_name,
    _itinerary,
    _memory_event_name,
    _pipeline_resource_name,
    _register_event_name,
    _slot_resource_name,
    _validate_control_issue_timing,
)
from loom.target.arch.amd.xdna.aie2p.core_encoding_data import CORE_ENCODING_TABLE
from loom.target.arch.amd.xdna.aie2p.core_machine_data import CORE_MACHINE_TABLE
from loom.target.arch.amd.xdna.aie2p.core_schedule_data import CORE_SCHEDULE_TABLE
from loom.target.low_descriptors import (
    DescriptorFlag,
    EffectKind,
    ImmediateFlag,
    ImmediateKind,
    InstructionClass,
    IssueUseKind,
    OperandFlag,
    OperandRole,
    ResourceKind,
    ScheduleClassFlag,
)


@pytest.mark.parametrize(
    "suffix", ["return", "branch.direct", "branch.zero", "branch.nonzero"]
)
def test_control_timing_requires_issue_stage_only_resources(suffix: str) -> None:
    descriptor_set = AIE2P_CORE_DESCRIPTOR_SET
    control = next(
        row for row in descriptor_set.descriptors if row.key.endswith("." + suffix)
    )
    changed = tuple(
        replace(
            row, issue_uses=(replace(row.issue_uses[0], stage=1), *row.issue_uses[1:])
        )
        if row.name == control.schedule_class
        else row
        for row in descriptor_set.schedule_classes
    )
    with pytest.raises(
        ValueError, match="control sharing requires issue-stage-only resources"
    ):
        _validate_control_issue_timing(
            replace(descriptor_set, schedule_classes=changed)
        )


@pytest.mark.parametrize("suffix", ["return", "branch.zero", "branch.nonzero"])
def test_control_tail_placement_requires_read_only_registers(suffix: str) -> None:
    descriptor_set = AIE2P_CORE_DESCRIPTOR_SET
    changed = tuple(
        replace(
            row,
            operands=(
                replace(row.operands[0], write_event=row.operands[0].read_event),
            ),
        )
        if row.key.endswith("." + suffix)
        else row
        for row in descriptor_set.descriptors
    )
    with pytest.raises(
        ValueError, match="control tail placement requires read-only registers"
    ):
        _validate_control_issue_timing(replace(descriptor_set, descriptors=changed))


def test_control_window_covers_its_outgoing_register_events() -> None:
    descriptor_set = AIE2P_CORE_DESCRIPTOR_SET
    returned = next(
        row for row in descriptor_set.descriptors if row.key.endswith(".return")
    )
    event = returned.operands[0].read_event
    changed = tuple(
        replace(row, minimum_issue_separation_cycles=7)
        if row.producer_event == event
        else row
        for row in descriptor_set.event_separations
    )
    with pytest.raises(
        ValueError, match="control window does not cover its outgoing events"
    ):
        _validate_control_issue_timing(
            replace(descriptor_set, event_separations=changed)
        )


def test_complete_schedule_domain_drives_selected_low_descriptors() -> None:
    descriptor_set = AIE2P_CORE_DESCRIPTOR_SET
    assert {
        resource.name
        for resource in descriptor_set.resources
        if resource.kind is ResourceKind.PIPELINE
    } == {
        *(_slot_resource_name(slot) for slot in _SLOT_RESOURCE_KINDS),
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


@pytest.mark.parametrize(
    ("key", "destination_class", "source_class"),
    [
        ("vector512", "mXm", "mXm"),
        ("vector512.to.accumulator512", "mBMs", "mXm"),
        ("accumulator512.to.vector512", "mXm", "mBMs"),
        ("accumulator512", "mBMs", "mBMs"),
    ],
)
def test_vector_move_itineraries_cover_complete_storage_domains(
    key: str, destination_class: str, source_class: str
) -> None:
    specifications = {spec.key: spec for spec in _DESCRIPTOR_SPECS}
    spec = specifications[f"amd.xdna.aie2p.move.{key}"]
    classes = {row.name: row for row in CORE_MACHINE_TABLE.register_classes}
    adapters = {row.name: row for row in CORE_MACHINE_TABLE.register_adapters}
    itineraries = {row.name: row for row in CORE_SCHEDULE_TABLE.itineraries}
    domains = {
        "mXm": ("eXe", "eXo"),
        "mBMs": ("eBMLL", "eBMLH", "eBMHL", "eBMHH"),
    }
    for operand, register_class in zip(
        ("dst", "src"), (destination_class, source_class), strict=True
    ):
        adapter = adapters[dict(spec.encoding_adapter_overrides)[operand]]
        assert adapter.register_class == register_class
        assert {
            register
            for domain in domains[register_class]
            for register in classes[domain].candidates
        } == set(classes[register_class].candidates)

    # Every subview/parity pair shares the complete payload, so the selected
    # representative does not restrict the physical register domain.
    selected_payload = itinerary_payload(_itinerary(spec))
    for destination, source in product(
        domains[destination_class], domains[source_class]
    ):
        native = itineraries[f"II_VMOV_alu_mv_mv_x_{destination}_{source}"]
        assert selected_payload == itinerary_payload(native)


@pytest.mark.parametrize(
    ("key", "itinerary_suffix", "bypasses", "resources", "allocation_move"),
    [
        ("vector512", "x_eXe_eXe", ("MV_Bypass", "MV_Bypass"), (), False),
        (
            "vector512.to.accumulator512",
            "x_eBMLL_eXe",
            (None, "MV_Bypass"),
            (("DM_WM_L0_PORT", 1),),
            False,
        ),
        (
            "accumulator512.to.vector512",
            "x_eXe_eBMLL",
            ("MV_Bypass", None),
            (("DM_RM_L0_PORT", 0),),
            False,
        ),
        (
            "accumulator512",
            "x_eBMLL_eBMLL",
            (None, None),
            (("DM_RM_L0_PORT", 0), ("DM_WM_L0_PORT", 1)),
            True,
        ),
        ("vec256", "w", ("MV_Bypass", "MV_Bypass"), (), True),
        ("bfp576", "ex", ("MV_Bypass", "MV_Bypass"), (), True),
    ],
)
def test_vector_move_descriptors_retain_native_endpoints_and_resources(
    key: str,
    itinerary_suffix: str,
    bypasses: tuple[str | None, str | None],
    resources: tuple[tuple[str, int], ...],
    allocation_move: bool,
) -> None:
    specifications = {spec.key: spec for spec in _DESCRIPTOR_SPECS}
    descriptors = {row.key: row for row in AIE2P_CORE_DESCRIPTOR_SET.descriptors}
    spec = specifications[f"amd.xdna.aie2p.move.{key}"]
    descriptor = descriptors[spec.key]
    assert spec.itinerary == f"II_VMOV_alu_mv_mv_{itinerary_suffix}"
    itinerary = _itinerary(spec)
    assert itinerary.operand_cycles == (2, 1)
    assert tuple(bypass_class(itinerary, index) for index in range(2)) == bypasses
    assert itinerary.memory is None
    assert itinerary.micro_ops == 1
    assert [
        (use.resources, use.start_cycle, use.cycles, use.kind)
        for use in pipeline_uses(itinerary)
    ] == [
        ((resource,), stage, 1, PipelineStageKind.REQUIRED)
        for resource, stage in resources
    ]
    assert [
        (
            operand.read_stage,
            operand.ready_stage,
            operand.read_event,
            operand.write_event,
        )
        for operand in descriptor.operands
    ] == [
        (0, 2, None, _register_event_name("write", 2, bypasses[0])),
        (1, 0, _register_event_name("read", 1, bypasses[1]), None),
    ]
    assert (DescriptorFlag.ALLOCATION_MOVE in descriptor.flags) == allocation_move


def test_vector_move_reverse_bypass_prevents_early_overwrite() -> None:
    specifications = {spec.key: spec for spec in _DESCRIPTOR_SPECS}
    reader = _itinerary(
        specifications["amd.xdna.aie2p.move.vector512.to.accumulator512"]
    )
    writer = _itinerary(
        specifications["amd.xdna.aie2p.move.accumulator512.to.vector512"]
    )
    # The X source observes the replacement through the move bypass. Equal
    # nominal read/write endpoints therefore do not permit earlier writer issue.
    assert dependency_separation(reader, 1, writer, 0, DependencyKind.WAR) == 0
    assert dependency_separation(writer, 0, reader, 1, DependencyKind.RAW) == 1


def test_scalar_memory_forms_use_storage_specialized_itineraries() -> None:
    specifications = {spec.key: spec for spec in _DESCRIPTOR_SPECS}
    expected_itineraries = {
        "i8": (
            "II_LDA_u8_idx_imm",
            "II_LDA_u8_idx",
            "II_ST_s8_idx_imm",
            "II_ST_s8_idx",
        ),
        "i16": (
            "II_LDA_u16_idx_imm",
            "II_LDA_u16_idx",
            "II_ST_s16_idx_imm",
            "II_ST_s16_idx",
        ),
        "i32": (
            "II_LDA_dms_lda_idx_imm_eR",
            "II_LDA_dms_lda_idx_eR",
            "II_ST_dms_sts_idx_imm_eR",
            "II_ST_dms_sts_idx_eR",
        ),
    }
    for element_type, expected in expected_itineraries.items():
        assert (
            tuple(
                specifications[
                    f"amd.xdna.aie2p.{operation}.scalar.{element_type}.indexed.{address}"
                ].itinerary
                for operation, address in (
                    ("load", "immediate"),
                    ("load", "register"),
                    ("store", "immediate"),
                    ("store", "register"),
                )
            )
            == expected
        )
    assert (
        specifications["amd.xdna.aie2p.move.to.address-index"].itinerary
        == "II_MOVS_eDJ_eR"
    )


def test_direct_branch_forms_retain_exact_control_contracts() -> None:
    descriptors = {
        descriptor.key: descriptor
        for descriptor in AIE2P_CORE_DESCRIPTOR_SET.descriptors
    }
    specifications = {spec.key: spec for spec in _DESCRIPTOR_SPECS}
    schedule_classes = {
        schedule_class.name: schedule_class
        for schedule_class in AIE2P_CORE_DESCRIPTOR_SET.schedule_classes
    }
    expected = {
        "amd.xdna.aie2p.branch.direct": ("J_lng", "II_J_lng", "j", 0),
        "amd.xdna.aie2p.branch.nonzero": ("JNZ", "II_JNZ", "jnz", 1),
        "amd.xdna.aie2p.branch.zero": ("JZ", "II_JZ", "jz", 1),
    }
    for key, (form_name, itinerary, mnemonic, operand_count) in expected.items():
        specification = specifications[key]
        descriptor = descriptors[key]
        assert specification.form_name == form_name
        assert specification.itinerary == itinerary
        assert descriptor.mnemonic == mnemonic
        assert descriptor.asm_forms[0].mnemonic == mnemonic
        assert len(descriptor.operands) == operand_count
        assert descriptor.asm_forms[0].results == ()
        assert descriptor.asm_forms[0].operands == tuple(
            operand.field_name for operand in descriptor.operands
        )
        assert len(descriptor.immediates) == 1
        immediate = descriptor.immediates[0]
        assert immediate.field_name == "i"
        assert immediate.kind is ImmediateKind.ORDINAL
        assert immediate.flags == (ImmediateFlag.SYMBOLIC,)
        assert immediate.bit_width == 20
        assert immediate.value_step == 1
        assert immediate.signed_min == 0
        assert immediate.unsigned_max == (1 << 20) - 1
        assert immediate.encoding_field_id != 0
        assert immediate.encoding_id != 0
        assert descriptor.asm_forms[0].immediates[0].field_name == "i"
        assert descriptor.effects[0].kind is EffectKind.CONTROL
        assert descriptor.instruction_classes == (InstructionClass.CONTROL,)
        assert DescriptorFlag.SIDE_EFFECTING in descriptor.flags
        assert DescriptorFlag.TERMINATOR in descriptor.flags
        assert DescriptorFlag.DEAD_REMOVABLE not in descriptor.flags
        schedule_class = schedule_classes[descriptor.schedule_class]
        assert schedule_class.flags == (ScheduleClassFlag.CONTROL,)
        assert schedule_class.instruction_classes == (InstructionClass.CONTROL,)
        assert _INSTRUCTION_ENCODINGS[form_name].delay_slot_count == 5


def test_lock_memory_timing_matches_aie2p_stall_and_resume_oracle() -> None:
    descriptors = {
        descriptor.key: descriptor
        for descriptor in AIE2P_CORE_DESCRIPTOR_SET.descriptors
    }
    separations = {
        (row.producer_event, row.consumer_event): row.minimum_issue_separation_cycles
        for row in AIE2P_CORE_DESCRIPTOR_SET.event_separations
    }
    assert _LOCK_EFFECT.producer_event is not None
    assert _LOCK_EFFECT.consumer_event is not None
    assert separations[_LOCK_EFFECT.producer_event, _LOCK_EFFECT.consumer_event] == 4

    # The pinned AIE2P model uses memory cycles 5 for normal accesses, 6 for
    # FIFO stores, 7 for SRS stores, 8 for converting FIFO stores, and 5/11 for
    # partword read-modify-write stores. Core resume at 8 and stall at 2 require
    # these forward/backward issue separations.
    expected_separations_by_cycles = {
        (5,): (4, 4),
        (6,): (3, 5),
        (7,): (2, 6),
        (8,): (1, 7),
        (5, 11): (4, 10),
    }

    memory_spec_count = 0
    read_modify_write_spec_count = 0
    for spec in _DESCRIPTOR_SPECS:
        form = _MACHINE_FORMS[spec.form_name]
        may_load = has_property(form, "mayLoad")
        may_store = has_property(form, "mayStore")
        if not may_load and not may_store:
            continue
        memory_spec_count += 1
        read_modify_write_spec_count += int(may_load and may_store)
        memory = _itinerary(spec).memory
        assert memory is not None
        expected_resume_separation, expected_stall_separation = (
            expected_separations_by_cycles[memory.cycles]
        )
        descriptor = descriptors[spec.key]
        for effect in descriptor.effects:
            if effect.kind not in (EffectKind.READ, EffectKind.WRITE):
                continue
            access = "read" if effect.kind is EffectKind.READ else "write"
            memory_event = _memory_event_name(access, memory)
            assert effect.producer_event == memory_event
            assert effect.consumer_event == memory_event
            assert (
                separations[_LOCK_EFFECT.producer_event, memory_event]
                == expected_resume_separation
            )
            assert (
                separations[memory_event, _LOCK_EFFECT.consumer_event]
                == expected_stall_separation
            )

    assert memory_spec_count != 0
    assert read_modify_write_spec_count != 0


def test_scalar_stream_transfers_preserve_protocol_and_status_dependencies() -> None:
    descriptors = {row.key: row for row in AIE2P_CORE_DESCRIPTOR_SET.descriptors}
    classes = {row.name: row for row in AIE2P_CORE_DESCRIPTOR_SET.reg_classes}
    separations = {
        (row.producer_event, row.consumer_event): row.minimum_issue_separation_cycles
        for row in AIE2P_CORE_DESCRIPTOR_SET.event_separations
    }
    source_adapter = next(
        row
        for row in CORE_MACHINE_TABLE.register_adapters
        if row.name == "OP_mMvSclSrc"
    )
    source_encodings = dict(source_adapter.effective_register_encodings)
    for spec in _DESCRIPTOR_SPECS:
        form = _MACHINE_FORMS[spec.form_name]
        status_registers = {"srMS0", "srSS0"} & set(form.implicit_defs)
        if not status_registers:
            continue
        descriptor = descriptors[spec.key]
        assert DescriptorFlag.SIDE_EFFECTING in descriptor.flags
        assert DescriptorFlag.DEAD_REMOVABLE not in descriptor.flags
        assert [effect.kind for effect in descriptor.effects] == [EffectKind.BARRIER]
        assert _itinerary(spec).memory is None
        register = next(iter(status_registers))
        direction = "read" if register == "srSS0" else "write"
        state_write = descriptor.operands[-1]
        assert state_write.flags == (OperandFlag.IMPLICIT, OperandFlag.STATE_WRITE)
        state_class = state_write.reg_alts[0].reg_class
        assert classes[state_class].physical_registers == (register,)
        status = descriptors[f"amd.xdna.aie2p.stream.{direction}.status"]
        state_read = status.operands[-1]
        assert state_read.flags == (OperandFlag.IMPLICIT, OperandFlag.STATE_READ)
        assert state_read.reg_alts[0].reg_class == state_class
        assert status.asm_forms[0].operands == ()
        assert status.encoding_field_values[0].value == source_encodings[register]
        assert separations[state_write.write_event, state_read.read_event] == (
            8 if direction == "read" else 3
        )


def test_cascade_transfers_preserve_protocol_and_enable_dependencies() -> None:
    descriptors = {row.key: row for row in AIE2P_CORE_DESCRIPTOR_SET.descriptors}
    classes = {row.name: row for row in AIE2P_CORE_DESCRIPTOR_SET.reg_classes}
    separations = {
        (row.producer_event, row.consumer_event): row.minimum_issue_separation_cycles
        for row in AIE2P_CORE_DESCRIPTOR_SET.event_separations
    }
    for direction, port, register in (
        ("read", "scd", "crSCDEn"),
        ("write", "mcd", "crMCDEn"),
    ):
        enable = descriptors[f"amd.xdna.aie2p.state.{port}-enable.immediate"]
        state_write = enable.operands[0]
        assert state_write.flags == (OperandFlag.IMPLICIT, OperandFlag.STATE_WRITE)
        for payload in ("vector", "accumulator"):
            transfer = descriptors[f"amd.xdna.aie2p.cascade.{direction}.{payload}.512"]
            assert DescriptorFlag.SIDE_EFFECTING in transfer.flags
            assert DescriptorFlag.DEAD_REMOVABLE not in transfer.flags
            assert [effect.kind for effect in transfer.effects] == [EffectKind.BARRIER]
            value, state_read = transfer.operands
            assert (
                value.unit_count * classes[value.reg_alts[0].reg_class].alloc_unit_bits
                == 512
            )
            assert state_read.flags == (OperandFlag.IMPLICIT, OperandFlag.STATE_READ)
            state_class = state_read.reg_alts[0].reg_class
            assert classes[state_class].physical_registers == (register,)
            assert state_write.reg_alts[0].reg_class == state_class
            assert (state_write.write_event, state_read.read_event) in separations


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
    scalar_store = descriptors["amd.xdna.aie2p.store.scalar.i32.indexed.immediate"]
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

    # Without forwarding, an old read may observe storage at the replacement's
    # write cycle. Physical issue order need not match access publication order.
    assert (
        separations[
            _register_event_name("read", 1, None),
            _register_event_name("write", 2, None),
        ]
        == -1
    )

    memory_write = next(
        row.producer_event
        for row in vector_store.effects
        if row.kind is EffectKind.WRITE
    )
    memory_read = next(
        row.consumer_event for row in vector_load.effects if row.kind is EffectKind.READ
    )
    assert separations[memory_write, memory_read] == 1
    assert vector_load.immediates[0].encoding_field_id != 0
    assert vector_load.immediates[0].encoding_id != 0
    assert vector_load.immediates[0].value_step == 64
    assert scalar_add.immediates[0].value_step == 1


def test_physical_register_events_preserve_signed_observation_times() -> None:
    separations = {
        (row.producer_event, row.consumer_event): row.minimum_issue_separation_cycles
        for row in AIE2P_CORE_DESCRIPTOR_SET.event_separations
    }
    for producer, consumer in product(_REGISTER_ENDPOINTS, repeat=2):
        producer_cycle, producer_bypass = producer
        consumer_cycle, consumer_bypass = consumer
        forwarded = int(
            producer_bypass is not None and producer_bypass == consumer_bypass
        )
        difference = producer_cycle - consumer_cycle
        assert (
            separations[
                _register_event_name("write", *producer),
                _register_event_name("read", *consumer),
            ]
            == difference + 1 - forwarded
        )
        assert (
            separations[
                _register_event_name("read", *producer),
                _register_event_name("write", *consumer),
            ]
            == difference + forwarded
        )
        assert (
            separations[
                _register_event_name("write", *producer),
                _register_event_name("write", *consumer),
            ]
            == difference + 1
        )

    # A delayed load may replace storage read by an earlier BFP conversion.
    assert (
        separations[
            _register_event_name("read", 1, None),
            _register_event_name("write", 7, None),
        ]
        == -6
    )
    # Matching move bypass observes the new value at endpoint equality. The
    # old X-to-BM read must therefore prohibit an earlier BM-to-X overwrite.
    assert (
        separations[
            _register_event_name("read", 1, "MV_Bypass"),
            _register_event_name("write", 2, "MV_Bypass"),
        ]
        == 0
    )
    # Immediate writes still wait for a late store's old-value read.
    assert (
        separations[
            _register_event_name("read", 7, None),
            _register_event_name("write", 1, None),
        ]
        == 6
    )
