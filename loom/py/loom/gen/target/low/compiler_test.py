# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from dataclasses import replace
from itertools import permutations

import pytest

from loom.gen.target.low import compiler
from loom.target.low_descriptors import (
    DescriptorFlag,
    EncodingFieldValue,
    EnumDomain,
    EnumValue,
    ImmediateFlag,
    ImmediateKind,
    InstructionClass,
    IssueUse,
    IssueUseKind,
    LatencyKind,
    ModelQuality,
    Resource,
    ResourceKind,
    ScheduleClass,
    ScheduleClassFlag,
)
from loom.target.test.descriptors import (
    TEST_LOW_ADD_I32_DESCRIPTOR,
    TEST_LOW_CONST_I32_DESCRIPTOR,
    TEST_LOW_CORE_DESCRIPTOR_SET,
)


@pytest.mark.parametrize("names", permutations(("zulu", "alpha", "i32_value", "beta")))
def test_immediate_identity_survives_declaration_order(names) -> None:
    base = TEST_LOW_CONST_I32_DESCRIPTOR
    fields = tuple(replace(base.immediates[0], field_name=name) for name in names)
    descriptor = replace(base, immediates=fields)
    compiled = compiler.compile_descriptor_set(replace(TEST_LOW_CORE_DESCRIPTOR_SET, descriptors=(descriptor,)))
    masks = {field.field_name: mask for field, mask in zip(compiled.immediates, compiled.immediate_attribute_masks, strict=True)}
    assert masks == {"alpha": 1, "beta": 2, "i32_value": 4, "zulu": 8}


def test_immediate_identity_is_part_of_interned_layout() -> None:
    base = TEST_LOW_CONST_I32_DESCRIPTOR
    expanded = replace(
        base,
        key="test.const.expanded.i32",
        mnemonic="test.const.expanded.i32",
        immediates=(replace(base.immediates[0], field_name="alpha", flags=(ImmediateFlag.DEFAULT_VALUE,)), *base.immediates),
    )
    compiled = compiler.compile_descriptor_set(replace(TEST_LOW_CORE_DESCRIPTOR_SET, descriptors=(base, expanded)))
    value_masks = [mask for field, mask in zip(compiled.immediates, compiled.immediate_attribute_masks, strict=True) if field.field_name == "i32_value"]
    assert sorted(value_masks) == [1, 2]


@pytest.mark.parametrize("count", [32, 33])
def test_immediate_presence_capacity(count) -> None:
    base = TEST_LOW_CONST_I32_DESCRIPTOR
    fields = (*base.immediates, *(replace(base.immediates[0], field_name=f"field_{index:02}") for index in range(count - 1)))
    descriptor_set = replace(TEST_LOW_CORE_DESCRIPTOR_SET, descriptors=(replace(base, immediates=fields),))
    if count == 33:
        with pytest.raises(ValueError, match="exceeds 32 immediate fields"):
            compiler.compile_descriptor_set(descriptor_set)
    else:
        compiled = compiler.compile_descriptor_set(descriptor_set)
        assert sorted(compiled.immediate_attribute_masks) == [1 << index for index in range(32)]


def test_enum_immediate_projection_retains_semantic_values() -> None:
    enum_descriptor = replace(
        TEST_LOW_CONST_I32_DESCRIPTOR,
        immediates=(replace(TEST_LOW_CONST_I32_DESCRIPTOR.immediates[0], kind=ImmediateKind.ENUM, enum_domain="direction"),),
    )
    compiled = compiler.compile_descriptor_set(
        replace(
            TEST_LOW_CORE_DESCRIPTOR_SET,
            descriptors=(TEST_LOW_ADD_I32_DESCRIPTOR, enum_descriptor),
            enum_domains=(EnumDomain("direction", (EnumValue("reverse", -5), EnumValue("forward", 7))),),
        )
    )
    descriptors = {descriptor.key: descriptor for descriptor in compiled.descriptors}
    assert DescriptorFlag.ENUM_IMMEDIATES not in descriptors[TEST_LOW_ADD_I32_DESCRIPTOR.key].flags
    assert DescriptorFlag.ENUM_IMMEDIATES in descriptors[enum_descriptor.key].flags
    assert {value.token: value.value for value in compiled.enum_values} == {"reverse": -5, "forward": 7}


@pytest.mark.parametrize(
    ("second_uses", "expected_units"),
    [
        ((IssueUse("test.issue.alias", cycles=1, units=1),), 1),
        ((IssueUse("test.issue.alias", cycles=1, units=2),), 2),
        ((IssueUse("test.issue.alias", cycles=1, units=1, stage=1),), 0),
        ((IssueUse("test.issue.alias", cycles=1, units=1, kind=IssueUseKind.RESERVED),), 0),
        ((), 0),
    ],
)
def test_resource_calendar_retains_common_issue_demand(second_uses, expected_units) -> None:
    descriptor_set = replace(
        TEST_LOW_CORE_DESCRIPTOR_SET,
        resources=(
            Resource("test.issue", capacity_per_cycle=4, kind=ResourceKind.PIPELINE, contention_group_id=1),
            Resource("test.issue.alias", capacity_per_cycle=4, kind=ResourceKind.PIPELINE, contention_group_id=1),
        ),
        schedule_classes=(
            ScheduleClass(
                "test.first", latency_kind=LatencyKind.EXACT, model_quality=ModelQuality.EXACT, issue_uses=(IssueUse("test.issue", cycles=1, units=1), IssueUse("test.issue.alias", cycles=1, units=1))
            ),
            ScheduleClass("test.second", latency_kind=LatencyKind.EXACT, model_quality=ModelQuality.EXACT, issue_uses=second_uses),
        ),
        descriptors=(
            replace(TEST_LOW_ADD_I32_DESCRIPTOR, schedule_class="test.first", instruction_classes=(InstructionClass.SCALAR_ALU,)),
            replace(TEST_LOW_CONST_I32_DESCRIPTOR, schedule_class="test.second", instruction_classes=(InstructionClass.SCALAR_ALU,)),
        ),
    )
    compiled = compiler.compile_descriptor_set(descriptor_set)
    assert compiled.resource_calendars
    assert all(calendar.minimum_issue_units == expected_units for calendar in compiled.resource_calendars)
    classes = {schedule_class.name: schedule_class for schedule_class in compiled.schedule_classes}
    assert ScheduleClassFlag.DISJOINT_ISSUE_USES not in classes["test.first"].flags
    expected_flags = (ScheduleClassFlag.DISJOINT_ISSUE_USES,) if second_uses else ()
    assert classes["test.second"].flags == expected_flags


@pytest.mark.parametrize("second_resource", ["test.issue", "test.alias", "test.other"])
@pytest.mark.parametrize("second_stage", [0, 1, 2, 4])
@pytest.mark.parametrize("first_kind", [IssueUseKind.REQUIRED, IssueUseKind.RESERVED])
@pytest.mark.parametrize("second_kind", [IssueUseKind.REQUIRED, IssueUseKind.RESERVED])
def test_disjoint_issue_uses_respect_aliases_and_stage_intervals(second_resource, second_stage, first_kind, second_kind) -> None:
    schedule_class = ScheduleClass(
        "test.staged",
        latency_kind=LatencyKind.EXACT,
        model_quality=ModelQuality.EXACT,
        issue_uses=(
            IssueUse("test.issue", cycles=2, units=1, kind=first_kind),
            IssueUse(second_resource, cycles=2, units=1, stage=second_stage, kind=second_kind),
        ),
    )
    compiled = compiler.compile_descriptor_set(
        replace(
            TEST_LOW_CORE_DESCRIPTOR_SET,
            resources=(
                Resource("test.issue", capacity_per_cycle=4, kind=ResourceKind.PIPELINE, contention_group_id=1),
                Resource("test.alias", capacity_per_cycle=4, kind=ResourceKind.PIPELINE, contention_group_id=1),
                Resource("test.other", capacity_per_cycle=4, kind=ResourceKind.PIPELINE),
            ),
            schedule_classes=(schedule_class,),
            descriptors=(replace(TEST_LOW_ADD_I32_DESCRIPTOR, schedule_class=schedule_class.name, instruction_classes=(InstructionClass.SCALAR_ALU,)),),
        )
    )
    disjoint = ScheduleClassFlag.DISJOINT_ISSUE_USES in compiled.schedule_classes[0].flags
    assert disjoint == (second_resource == "test.other" or second_stage >= 2)


def test_schedule_class_cannot_author_disjoint_issue_use_proof() -> None:
    descriptor_set = replace(
        TEST_LOW_CORE_DESCRIPTOR_SET,
        schedule_classes=tuple(replace(schedule_class, flags=(*schedule_class.flags, ScheduleClassFlag.DISJOINT_ISSUE_USES)) for schedule_class in TEST_LOW_CORE_DESCRIPTOR_SET.schedule_classes),
    )
    with pytest.raises(ValueError, match="authors the derived disjoint-issue-uses flag"):
        compiler.compile_descriptor_set(descriptor_set)


@pytest.mark.parametrize("candidate_names", permutations(("test.r0", "test.r1", "test.r2", "test.r3")))
def test_physical_packing_order_preserves_pairs_and_semantic_ordinals(candidate_names) -> None:
    descriptor_set = replace(
        TEST_LOW_CORE_DESCRIPTOR_SET,
        reg_classes=tuple(replace(reg_class, physical_registers=candidate_names) if reg_class.name == "test.explicit32" else reg_class for reg_class in TEST_LOW_CORE_DESCRIPTOR_SET.reg_classes),
    )
    compiled = compiler.compile_descriptor_set(descriptor_set)
    class_id = compiled.reg_class_ids["test.explicit32"]
    start = compiled.physical_register_candidate_starts[class_id]
    semantic_ids = compiled.physical_register_candidate_ids[start : start + 4]
    assert tuple(compiled.physical_registers[index].name for index in semantic_ids) == candidate_names
    lookup = compiled.physical_register_candidate_lookups[class_id]
    reverse = compiled.physical_register_candidate_ordinals[lookup.ordinal_start : lookup.ordinal_start + lookup.register_count]
    for physical_id in range(lookup.register_base, lookup.register_base + lookup.register_count):
        expected = semantic_ids.index(physical_id) if physical_id in semantic_ids else 0xFFFF
        assert reverse[physical_id - lookup.register_base] == expected
    allocation_order = compiled.physical_register_allocation_ordinals[start : start + 4]
    assert sorted(allocation_order) == list(range(4))
    packed_names = [candidate_names[ordinal] for ordinal in allocation_order]
    assert {frozenset(packed_names[:2]), frozenset(packed_names[2:])} == {frozenset(("test.r0", "test.r2")), frozenset(("test.r1", "test.r3"))}
    ranks = {ordinal: rank for rank, ordinal in enumerate(allocation_order)}
    for view in compiled.physical_register_views:
        if view.reg_class_id != class_id:
            continue
        units = compiled.physical_register_view_unit_candidate_ordinals[view.unit_candidate_ordinal_start : view.unit_candidate_ordinal_start + view.unit_count]
        assert view.packing_rank == min(ranks[ordinal] for ordinal in units)


def test_physical_packing_order_is_independent_of_view_declaration_order() -> None:
    compiled = compiler.compile_descriptor_set(TEST_LOW_CORE_DESCRIPTOR_SET)
    reversed_views = compiler.compile_descriptor_set(
        replace(
            TEST_LOW_CORE_DESCRIPTOR_SET,
            physical_register_views=tuple(reversed(TEST_LOW_CORE_DESCRIPTOR_SET.physical_register_views)),
        )
    )
    assert compiled.physical_register_allocation_ordinals == reversed_views.physical_register_allocation_ordinals
    assert compiled.physical_register_views == reversed_views.physical_register_views
    assert compiled.physical_register_view_lookups == reversed_views.physical_register_view_lookups
    assert compiled.physical_register_view_ordinals == reversed_views.physical_register_view_ordinals
    for class_id, reg_class in enumerate(compiled.reg_classes):
        start = compiled.physical_register_candidate_starts[class_id]
        count = len(reg_class.physical_registers)
        order = compiled.physical_register_allocation_ordinals[start : start + count]
        assert sorted(order) == list(range(count))
        lookup = compiled.physical_register_candidate_lookups[class_id]
        if not count:
            assert lookup.register_count == 0
        if reg_class.name == "test.packed.narrow":
            # No aggregate spans this class's two candidates. Its source
            # preference order is already the packing order.
            assert order == [0, 1]


def test_compiler_interns_exact_descriptor_and_asm_spans() -> None:
    add_copy = replace(
        TEST_LOW_ADD_I32_DESCRIPTOR,
        key="test.add.copy.i32",
        mnemonic="test.add.copy.i32",
        semantic_tag="integer.add.copy.i32",
    )
    const = replace(
        TEST_LOW_CONST_I32_DESCRIPTOR,
        feature_mask_words=(0x5,),
        encoding_field_values=(EncodingFieldValue(7, 11),),
    )
    const_copy = replace(
        const,
        key="test.const.copy.i32",
        mnemonic="test.const.copy.i32",
        semantic_tag="integer.const.copy.i32",
    )
    materialized_const = replace(
        const,
        key="test.const.materialized.i32",
        mnemonic="test.const.materialized.i32",
        semantic_tag="integer.const.materialized.i32",
        constraints=(),
    )
    descriptor_set = replace(
        TEST_LOW_CORE_DESCRIPTOR_SET,
        descriptors=(
            TEST_LOW_ADD_I32_DESCRIPTOR,
            add_copy,
            const,
            const_copy,
            materialized_const,
        ),
    )

    compiled = compiler.compile_descriptor_set(descriptor_set)
    rows_by_key = {
        descriptor.key: row
        for descriptor, row in zip(
            compiled.descriptors,
            compiled.descriptor_rows,
            strict=True,
        )
    }
    add_row = rows_by_key[TEST_LOW_ADD_I32_DESCRIPTOR.key]
    add_copy_row = rows_by_key[add_copy.key]
    const_row = rows_by_key[const.key]
    const_copy_row = rows_by_key[const_copy.key]
    materialized_const_row = rows_by_key[materialized_const.key]

    assert add_row["operand_start"] == add_copy_row["operand_start"]
    assert const_row["operand_start"] == const_copy_row["operand_start"]
    assert const_row["operand_start"] != materialized_const_row["operand_start"]
    assert const_row["immediate_start"] == const_copy_row["immediate_start"]
    assert const_row["immediate_start"] == materialized_const_row["immediate_start"]
    assert const_row["constraint_start"] == const_copy_row["constraint_start"]
    assert const_row["feature_mask_word_start"] == const_copy_row["feature_mask_word_start"]
    assert const_row["encoding_field_value_start"] == const_copy_row["encoding_field_value_start"]
    assert len(compiled.operands) == 5
    assert len(compiled.immediates) == 1
    assert len(compiled.constraints) == 1
    assert compiled.feature_mask_words == [0x5]
    assert compiled.encoding_field_values == [EncodingFieldValue(7, 11)]

    asm_forms_by_descriptor = {form.descriptor_ordinal: form for form in compiled.asm_forms}
    add_form = asm_forms_by_descriptor[0]
    add_copy_form = asm_forms_by_descriptor[1]
    const_form = asm_forms_by_descriptor[2]
    const_copy_form = asm_forms_by_descriptor[3]
    assert add_form.result_index_start == add_copy_form.result_index_start
    assert add_form.operand_index_start == add_copy_form.operand_index_start
    assert const_form.result_index_start == const_copy_form.result_index_start
    assert const_form.immediate_start == const_copy_form.immediate_start
    assert compiled.asm_table_storage.operand_indices == [0, 1, 2]
    assert len(compiled.asm_table_storage.immediates) == 1


def test_physical_view_lookup_preserves_exact_class_and_unit_relations() -> None:
    compiled = compiler.compile_descriptor_set(TEST_LOW_CORE_DESCRIPTOR_SET)
    expected = {(view.physical_register_id, view.reg_class_id): ordinal for ordinal, view in enumerate(compiled.physical_register_views)}
    for physical_id, lookup in enumerate(compiled.physical_register_view_lookups):
        ordinals = compiled.physical_register_view_ordinals[lookup.ordinal_start : lookup.ordinal_start + lookup.class_count]
        for class_id in range(len(compiled.reg_classes)):
            offset = class_id - lookup.class_base
            actual = ordinals[offset] if 0 <= offset < lookup.class_count else 0xFFFFFFFF
            assert actual == expected.get((physical_id, class_id), 0xFFFFFFFF)
