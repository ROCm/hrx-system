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
    DescriptorFlag,
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
    assert len(descriptor_set.reg_classes) == 22
    assert len(descriptor_set.physical_register_views) == 12
    assert len(descriptor_set.register_parts) == 4
    assert len(descriptor_set.descriptors) == 169
    assert tuple(row.name for row in descriptor_set.physical_registers) == tuple(
        row.name for row in CORE_MACHINE_TABLE.physical_registers
    )
    assert tuple(
        row.atomic_units for row in descriptor_set.physical_registers
    ) == tuple(row.atomic_units for row in CORE_MACHINE_TABLE.physical_registers)
    assert {
        (view.physical_register, view.reg_class): view.units
        for view in descriptor_set.physical_register_views
    } == {
        (f"x{index}", "aie2p.vec256"): (f"wl{index}", f"wh{index}")
        for index in range(12)
    }


def test_complete_schedule_domain_drives_selected_low_descriptors() -> None:
    descriptor_set = AIE2P_CORE_DESCRIPTOR_SET
    assert len(descriptor_set.resources) == 90
    assert len(descriptor_set.timing_events) == 38
    assert len(descriptor_set.event_separations) == 651
    assert len(descriptor_set.schedule_classes) == 36
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


def test_scalar_memory_forms_use_storage_specialized_itineraries() -> None:
    specifications = {spec.key: spec for spec in _DESCRIPTOR_SPECS}
    assert {
        key: specifications[key].itinerary
        for key in (
            "amd.xdna.aie2p.load.scalar.indexed.immediate",
            "amd.xdna.aie2p.load.scalar.indexed.register",
            "amd.xdna.aie2p.store.scalar.indexed.immediate",
            "amd.xdna.aie2p.store.scalar.indexed.register",
            "amd.xdna.aie2p.move.to.address-index",
        )
    } == {
        "amd.xdna.aie2p.load.scalar.indexed.immediate": ("II_LDA_dms_lda_idx_imm_eR"),
        "amd.xdna.aie2p.load.scalar.indexed.register": "II_LDA_dms_lda_idx_eR",
        "amd.xdna.aie2p.store.scalar.indexed.immediate": ("II_ST_dms_sts_idx_imm_eR"),
        "amd.xdna.aie2p.store.scalar.indexed.register": "II_ST_dms_sts_idx_eR",
        "amd.xdna.aie2p.move.to.address-index": "II_MOVS_eDJ_eR",
    }


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

    el_class = next(
        register_class
        for register_class in AIE2P_CORE_DESCRIPTOR_SET.reg_classes
        if register_class.name == "aie2p.elpredicate"
    )
    assert el_class.full_register_part_mask == 0x3
    vec256_class = next(
        register_class
        for register_class in AIE2P_CORE_DESCRIPTOR_SET.reg_classes
        if register_class.name == "aie2p.vec256"
    )
    assert vec256_class.alloc_unit_bits == 256
    assert vec256_class.full_register_part_mask == 0x3
    assert {
        (part.name, part.reg_class, part.mask)
        for part in AIE2P_CORE_DESCRIPTOR_SET.register_parts
    } == {
        ("aie2p.elpredicate.low32", "aie2p.elpredicate", 0x1),
        ("aie2p.elpredicate.high32", "aie2p.elpredicate", 0x2),
        ("aie2p.vec256.low128", "aie2p.vec256", 0x1),
        ("aie2p.vec256.high128", "aie2p.vec256", 0x2),
    }


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
    assert set(vector_register_classes) == {"aie2p.vec256"}
    assert {
        operand.unit_count
        for key in vector_keys
        for operand in descriptors[key].operands
        if operand.reg_alts[0].reg_class == "aie2p.vec256"
    } == {2}
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
    assert all(operand.unit_count == 2 for operand in vector_add.operands)
    assert vector_add.operands[0].ready_stage == 2
    assert vector_add.operands[1].read_stage == 1

    vector_load = descriptors["amd.xdna.aie2p.load.a.i8x64.indexed.immediate"]
    assert vector_load.operands[0].ready_stage == 7
    assert vector_load.effects[0].width_bits == 512
    assert vector_load.schedule_alternatives == (
        "amd.xdna.aie2p.load.b.i8x64.indexed.immediate",
    )

    vector_store = descriptors["amd.xdna.aie2p.store.i8x64.indexed.immediate"]
    assert vector_store.effects[0].width_bits == 512

    vector_register_load = descriptors["amd.xdna.aie2p.load.a.i8x64.indexed.register"]
    vector_register_store = descriptors["amd.xdna.aie2p.store.i8x64.indexed.register"]
    assert [
        operand.reg_alts[0].reg_class for operand in vector_register_load.operands
    ] == ["aie2p.vec256", "aie2p.ep", "aie2p.edj"]
    assert [
        operand.reg_alts[0].reg_class for operand in vector_register_store.operands
    ] == ["aie2p.vec256", "aie2p.ep", "aie2p.edj"]
    assert vector_register_load.operands[0].unit_count == 2
    assert vector_register_store.operands[0].unit_count == 2
    assert vector_register_load.schedule_alternatives == (
        "amd.xdna.aie2p.load.b.i8x64.indexed.register",
    )
    assert vector_register_load.asm_forms[0].mnemonic == "vlda.512.i8x64.index"
    assert vector_register_store.asm_forms[0].mnemonic == "vst.512.i8x64.index"

    scalar_load = descriptors["amd.xdna.aie2p.load.scalar.indexed.immediate"]
    assert scalar_load.effects[0].width_bits == 32

    scalar_store = descriptors["amd.xdna.aie2p.store.scalar.indexed.immediate"]
    assert scalar_store.effects[0].width_bits == 32

    scalar_register_load = descriptors["amd.xdna.aie2p.load.scalar.indexed.register"]
    scalar_register_store = descriptors["amd.xdna.aie2p.store.scalar.indexed.register"]
    address_index_move = descriptors["amd.xdna.aie2p.move.to.address-index"]
    assert [
        operand.reg_alts[0].reg_class for operand in scalar_register_load.operands
    ] == ["aie2p.er", "aie2p.ep", "aie2p.edj"]
    assert [
        operand.reg_alts[0].reg_class for operand in scalar_register_store.operands
    ] == ["aie2p.er", "aie2p.ep", "aie2p.edj"]
    assert scalar_register_load.asm_forms[0].mnemonic == "lda.index"
    assert scalar_register_store.asm_forms[0].mnemonic == "st.index"
    assert [
        operand.reg_alts[0].reg_class for operand in address_index_move.operands
    ] == ["aie2p.edj", "aie2p.er"]
    assert address_index_move.asm_forms[0].mnemonic == "mov.address-index"

    static_offset = descriptors["amd.xdna.aie2p.materialize.static-byte-offset.i32"]
    assert static_offset.asm_forms[0].mnemonic == "mov.static-byte-offset"
    assert static_offset.operands[0].reg_alts[0].reg_class == "aie2p.er"

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
    assert predicate_compare.operands[0].reg_alts[0].reg_class == "aie2p.elpredicate"
    assert predicate_compare.operands[1].reg_alts[0].reg_class == "aie2p.vec256"
    assert predicate_compare.operands[1].unit_count == 2

    byte_select = descriptors["amd.xdna.aie2p.select.i8x64"]
    word_select = descriptors["amd.xdna.aie2p.select.i32x16"]
    assert [operand.field_name for operand in byte_select.operands] == [
        "d",
        "s1",
        "s2",
        "sel",
    ]
    assert [operand.reg_alts[0].reg_class for operand in byte_select.operands] == [
        "aie2p.vec256",
        "aie2p.vec256",
        "aie2p.vec256",
        "aie2p.elpredicate",
    ]
    assert all(operand.unit_count == 2 for operand in byte_select.operands[:3])
    assert word_select.operands[-1].reg_alts[0].reg_class == "aie2p.ers16"

    scalar_selector = descriptors["amd.xdna.aie2p.select.mask.i32"]
    assert scalar_selector.operands[0].field_name == "d0"
    assert scalar_selector.operands[0].reg_alts[0].reg_class == "aie2p.ers16"
    assert scalar_selector.operands[1].field_name == "s0"
    assert scalar_selector.operands[1].reg_alts[0].reg_class == "aie2p.er"

    for width in (8, 16, 32):
        for operation, signedness, sign_register in (
            ("min", "signed", "vaddsign1"),
            ("max", "signed", "vaddsign1"),
            ("min", "unsigned", "vaddsign0"),
            ("max", "unsigned", "vaddsign0"),
        ):
            key = f"amd.xdna.aie2p.{operation}.{signedness}.i{width}x{512 // width}"
            descriptor = descriptors[key]
            assert [operand.field_name for operand in descriptor.operands[:3]] == [
                "d",
                "s1",
                "s2",
            ]
            assert [
                operand.reg_alts[0].reg_class for operand in descriptor.operands[:3]
            ] == [
                "aie2p.vec256",
                "aie2p.vec256",
                "aie2p.vec256",
            ]
            assert all(operand.unit_count == 2 for operand in descriptor.operands[:3])
            hardwired_compare = descriptor.operands[3]
            assert hardwired_compare.field_name == "implicit_output_cmp"
            assert hardwired_compare.role is OperandRole.IMPLICIT
            assert hardwired_compare.reg_alts[0].reg_class == (
                "aie2p.ml8m" if width == 8 else "aie2p.mr16_vcompare"
            )
            assert hardwired_compare.reg_alts[0].flags == (
                RegClassAltFlag.PHYSICAL_ONLY,
            )
            assert set(hardwired_compare.flags) == {
                OperandFlag.IMPLICIT,
                OperandFlag.STATE_WRITE,
            }
            assert hardwired_compare.encoding_field_id == 0
            assert descriptor.asm_forms[0].results == ("d",)
            assert descriptor.operands[-1].field_name == (
                f"implicit_use_{sign_register}"
            )
            assert descriptor.operands[-1].reg_alts[0].reg_class == (
                f"aie2p.state.{sign_register}"
            )


def test_vector_memory_descriptors_cover_each_native_width_and_integer_shape() -> None:
    descriptors = {
        descriptor.key: descriptor
        for descriptor in AIE2P_CORE_DESCRIPTOR_SET.descriptors
    }
    for width_bits in (128, 256, 512):
        unit_count = 2 if width_bits == 512 else 1
        immediate_step = width_bits // 8
        for element_bits in (8, 16, 32):
            shape = f"i{element_bits}x{width_bits // element_bits}"
            for load_pipe in ("a", "b"):
                for address_form in ("immediate", "register"):
                    descriptor = descriptors[
                        f"amd.xdna.aie2p.load.{load_pipe}.{shape}.indexed."
                        f"{address_form}"
                    ]
                    payload = descriptor.operands[0]
                    assert payload.reg_alts[0].reg_class == "aie2p.vec256"
                    assert payload.unit_count == unit_count
                    assert descriptor.effects[0].width_bits == width_bits
                    if address_form == "immediate":
                        assert descriptor.immediates[0].value_step == immediate_step
            for address_form in ("immediate", "register"):
                descriptor = descriptors[
                    f"amd.xdna.aie2p.store.{shape}.indexed.{address_form}"
                ]
                payload = descriptor.operands[0]
                assert payload.reg_alts[0].reg_class == "aie2p.vec256"
                assert payload.unit_count == unit_count
                assert descriptor.effects[0].width_bits == width_bits
                if address_form == "immediate":
                    assert descriptor.immediates[0].value_step == immediate_step

            load = descriptors[f"amd.xdna.aie2p.load.a.{shape}.indexed.immediate"]
            store = descriptors[f"amd.xdna.aie2p.store.{shape}.indexed.immediate"]
            if width_bits == 128:
                assert load.operands[0].register_part == "aie2p.vec256.low128"
                assert store.operands[0].register_part == "aie2p.vec256.low128"
            else:
                assert load.operands[0].register_part is None
                assert store.operands[0].register_part is None


def test_scalar_address_descriptors_expose_fixed_register_state() -> None:
    descriptors = {
        descriptor.key: descriptor
        for descriptor in AIE2P_CORE_DESCRIPTOR_SET.descriptors
    }

    move_to_state = descriptors["amd.xdna.aie2p.move.to.division-state"]
    move_from_state = descriptors["amd.xdna.aie2p.move.from.division-state"]
    assert [operand.reg_alts[0].reg_class for operand in move_to_state.operands] == [
        "aie2p.mr31_divs",
        "aie2p.er",
    ]
    assert [operand.reg_alts[0].reg_class for operand in move_from_state.operands] == [
        "aie2p.er",
        "aie2p.mr31_divs",
    ]
    assert move_to_state.asm_forms[0].mnemonic == "mov.dividend"
    assert move_from_state.asm_forms[0].mnemonic == "mov.quotient"

    divide_step = descriptors["amd.xdna.aie2p.divide.step.unsigned.i32"]
    assert [operand.field_name for operand in divide_step.operands] == [
        "d0",
        "sd_out",
        "sd",
        "s0",
        "s1",
    ]
    assert [operand.reg_alts[0].reg_class for operand in divide_step.operands] == [
        "aie2p.er",
        "aie2p.mr31_divs",
        "aie2p.mr31_divs",
        "aie2p.er",
        "aie2p.er",
    ]
    assert len(divide_step.constraints) == 1
    assert divide_step.constraints[0].kind is ConstraintKind.TIED
    assert divide_step.constraints[0].lhs_operand_index == 1
    assert divide_step.constraints[0].rhs_operand_index == 2
    assert divide_step.asm_forms[0].results == ("d0", "sd_out")
    assert divide_step.asm_forms[0].operands == ("sd", "s0", "s1")

    multiply_add = descriptors["amd.xdna.aie2p.madd.i32"]
    assert [operand.field_name for operand in multiply_add.operands] == [
        "d0",
        "a0",
        "s0",
        "s1",
    ]
    assert len(multiply_add.constraints) == 1
    assert multiply_add.constraints[0].kind is ConstraintKind.TIED
    assert multiply_add.constraints[0].lhs_operand_index == 0
    assert multiply_add.constraints[0].rhs_operand_index == 1

    for key in (
        "amd.xdna.aie2p.select.zero.i32",
        "amd.xdna.aie2p.select.nonzero.i32",
    ):
        select = descriptors[key]
        assert select.operands[-1].field_name == "s2"
        assert select.operands[-1].reg_alts[0].reg_class == "aie2p.mr27_select"
        assert select.operands[-1].encoding_field_id == 0

    for key in (
        "amd.xdna.aie2p.cmp.slt.i32.select",
        "amd.xdna.aie2p.cmp.ult.i32.select",
    ):
        compare = descriptors[key]
        assert compare.operands[0].reg_alts[0].reg_class == "aie2p.mr27_select"

    fixed_classes = {
        register_class.name: register_class
        for register_class in AIE2P_CORE_DESCRIPTOR_SET.reg_classes
        if register_class.name in ("aie2p.mr27_select", "aie2p.mr31_divs")
    }
    assert fixed_classes["aie2p.mr27_select"].physical_registers == ("r27",)
    assert fixed_classes["aie2p.mr31_divs"].physical_registers == ("r31",)


def test_vector_predicates_use_one_partially_addressable_el_value() -> None:
    descriptors = {
        descriptor.key: descriptor
        for descriptor in AIE2P_CORE_DESCRIPTOR_SET.descriptors
    }
    for width in (16, 32):
        compare = descriptors[
            f"amd.xdna.aie2p.cmp.lt.signed.i{width}x{512 // width}.el.low32"
        ]
        assert compare.operands[0].reg_alts[0].reg_class == "aie2p.elpredicate"
        assert compare.operands[0].register_part == "aie2p.elpredicate.low32"
        assert compare.operands[0].encoding_adapter_id != 0

        select = descriptors[f"amd.xdna.aie2p.select.i{width}x{512 // width}.mask64"]
        assert select.operands[-1].field_name == "sel"
        assert select.operands[-1].reg_alts[0].reg_class == "aie2p.elpredicate"
        assert select.operands[-1].register_part == "aie2p.elpredicate.low32"
        assert select.operands[-1].encoding_adapter_id != 0

    for operation in ("and", "or", "xor"):
        low = descriptors[f"amd.xdna.aie2p.predicate.{operation}.low32"]
        high = descriptors[f"amd.xdna.aie2p.predicate.{operation}.high32"]
        assert all(
            operand.register_part == "aie2p.elpredicate.low32"
            for operand in low.operands
        )
        assert all(
            operand.register_part == "aie2p.elpredicate.high32"
            for operand in high.operands[:3]
        )
        continuation = high.operands[3]
        assert continuation.field_name == "storage"
        assert continuation.register_part == "aie2p.elpredicate.low32"
        assert set(continuation.flags) == {
            OperandFlag.IMPLICIT,
            OperandFlag.STORAGE_CONTINUATION,
        }
        assert len(high.constraints) == 1
        assert high.constraints[0].kind is ConstraintKind.TIED
        assert high.constraints[0].lhs_operand_index == 0
        assert high.constraints[0].rhs_operand_index == 3
        assert high.asm_forms[0].operands == ("s0", "s1", "storage")

    complete = descriptors["amd.xdna.aie2p.predicate.complete.zero.high32"]
    assert complete.operands[0].register_part == "aie2p.elpredicate.high32"
    assert complete.operands[1].register_part == "aie2p.elpredicate.low32"
    assert OperandFlag.STORAGE_CONTINUATION in complete.operands[1].flags
    assert complete.constraints[0].kind is ConstraintKind.TIED
    assert complete.constraints[0].lhs_operand_index == 0
    assert complete.constraints[0].rhs_operand_index == 1
    assert complete.asm_forms[0].operands == ("storage",)


def test_vector_multiply_descriptors_own_configuration_state() -> None:
    descriptors = {
        descriptor.key: descriptor
        for descriptor in AIE2P_CORE_DESCRIPTOR_SET.descriptors
    }

    config_constant = descriptors["amd.xdna.aie2p.constant.i32.mova"]
    shift_constant = descriptors["amd.xdna.aie2p.constant.i32.shift"]
    assert config_constant.operands[0].reg_alts[0].reg_class == "aie2p.er"
    assert shift_constant.operands[0].reg_alts[0].reg_class == "aie2p.es"

    multiply = descriptors["amd.xdna.aie2p.multiply.i16x32.configured"]
    assert [operand.field_name for operand in multiply.operands] == [
        "dst",
        "s1",
        "s2",
        "acc",
    ]
    assert [operand.reg_alts[0].reg_class for operand in multiply.operands] == [
        "aie2p.edm",
        "aie2p.vec256",
        "aie2p.vec256",
        "aie2p.er",
    ]
    assert [operand.unit_count for operand in multiply.operands] == [1, 2, 2, 1]

    narrow = descriptors["amd.xdna.aie2p.narrow.trunc.signed.i16x32"]
    assert [operand.reg_alts[0].reg_class for operand in narrow.operands[:3]] == [
        "aie2p.vec256",
        "aie2p.edm",
        "aie2p.es",
    ]
    assert narrow.operands[0].unit_count == 2
    narrow_state_classes = {
        operand.field_name: operand.reg_alts[0].reg_class
        for operand in narrow.operands[3:]
    }

    for key, state_field, register_class, encoded_register in (
        (
            "amd.xdna.aie2p.state.rounding.immediate",
            "implicit_use_crrnd",
            "aie2p.mcrrnd",
            123,
        ),
        (
            "amd.xdna.aie2p.state.srs-mode.immediate",
            "implicit_use_crsrsmode",
            "aie2p.mcrsrsmode",
            71,
        ),
        (
            "amd.xdna.aie2p.state.saturation.immediate",
            "implicit_use_crsat",
            "aie2p.mcrsat",
            39,
        ),
    ):
        setter = descriptors[key]
        assert setter.asm_forms[0].results == ()
        assert len(setter.operands) == 1
        state_write = setter.operands[0]
        assert state_write.role is OperandRole.IMPLICIT
        assert state_write.reg_alts[0].reg_class == register_class
        assert set(state_write.flags) == {
            OperandFlag.IMPLICIT,
            OperandFlag.STATE_WRITE,
        }
        assert state_write.encoding_field_id == 0
        assert len(setter.encoding_field_values) == 1
        assert setter.encoding_field_values[0].value == encoded_register
        assert DescriptorFlag.SIDE_EFFECTING in setter.flags
        assert DescriptorFlag.DEAD_REMOVABLE not in setter.flags
        assert narrow_state_classes[state_field] == register_class


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
