# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from dataclasses import replace
from itertools import permutations

import pytest

from loom.gen.target.low import compiler
from loom.ir import ScalarTypeKind
from loom.target.low_descriptors import (
    AsmForm,
    AsmImmediate,
    AsmOperandSegment,
    AsmOperandSegmentDelimiter,
    AsmResultValueType,
    Constraint,
    ConstraintKind,
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
    MemorySpace,
    ModelQuality,
    NativeAsmValue,
    NativeAsmValueKind,
    OperandForm,
    OperandFormMatch,
    OperandFormMatchKind,
    PhysicalRegisterView,
    RegClassAlt,
    RegClassAltFlag,
    Resource,
    ResourceKind,
    ScheduleClass,
    ScheduleClassFlag,
)
from loom.target.test.descriptors import (
    TEST_LOW_ADD_I32_DESCRIPTOR,
    TEST_LOW_CONST_I32_DESCRIPTOR,
    TEST_LOW_CORE_DESCRIPTOR_SET,
    TEST_LOW_LOAD_V4I32_DESCRIPTOR,
    TEST_LOW_WRITE_LOW16_I32_DESCRIPTOR,
)


@pytest.mark.parametrize("alignment", [0, 3, 6, 65536])
def test_operand_alignment_requires_representable_power_of_two(alignment) -> None:
    base = TEST_LOW_ADD_I32_DESCRIPTOR
    operand = base.operands[0]
    alternative = replace(operand.reg_alts[0], unit_alignment=alignment)
    descriptor = replace(base, operands=(replace(operand, reg_alts=(alternative,)), *base.operands[1:]))
    with pytest.raises(ValueError, match="register alignment"):
        compiler.compile_descriptor_set(replace(TEST_LOW_CORE_DESCRIPTOR_SET, descriptors=(descriptor,)))


def test_operand_alignment_is_part_of_interned_alternative() -> None:
    base = TEST_LOW_ADD_I32_DESCRIPTOR
    operands = tuple(replace(operand, reg_alts=(replace(operand.reg_alts[0], unit_alignment=alignment),)) for operand, alignment in zip(base.operands, (1, 2, 4), strict=True))
    compiled = compiler.compile_descriptor_set(replace(TEST_LOW_CORE_DESCRIPTOR_SET, descriptors=(replace(base, operands=operands),)))
    assert [row[3] for row in compiled.reg_class_alts] == [0, 1, 2]


def test_register_part_is_owned_by_each_register_alternative() -> None:
    base = TEST_LOW_WRITE_LOW16_I32_DESCRIPTOR
    mixed_result = replace(
        base.operands[0],
        reg_alts=(
            base.operands[0].reg_alts[0],
            RegClassAlt("test.i64"),
        ),
    )
    descriptor = replace(base, operands=(mixed_result, *base.operands[1:]))
    compiled = compiler.compile_descriptor_set(replace(TEST_LOW_CORE_DESCRIPTOR_SET, descriptors=(descriptor,)))
    result_alt_start = compiled.operand_alt_starts[0]
    result_alternatives = compiled.reg_class_alts[result_alt_start : result_alt_start + 2]
    assert [row[0] for row in result_alternatives] == [
        compiled.reg_class_ids["test.i32"],
        compiled.reg_class_ids["test.i64"],
    ]
    assert [row[1] for row in result_alternatives] == [
        compiled.register_part_ids[base.operands[0].reg_alts[0].register_part],
        None,
    ]


def test_register_alternative_rejects_part_for_another_class() -> None:
    base = TEST_LOW_WRITE_LOW16_I32_DESCRIPTOR
    result = replace(
        base.operands[0],
        reg_alts=(replace(base.operands[0].reg_alts[0], reg_class="test.i64"),),
    )
    descriptor = replace(base, operands=(result, *base.operands[1:]))
    with pytest.raises(ValueError, match=r"uses register part .* for register class"):
        compiler.compile_descriptor_set(replace(TEST_LOW_CORE_DESCRIPTOR_SET, descriptors=(descriptor,)))


def test_immediate_alternative_rejects_register_part() -> None:
    base = TEST_LOW_ADD_I32_DESCRIPTOR
    operand = replace(
        base.operands[1],
        reg_alts=(
            RegClassAlt(
                None,
                flags=(RegClassAltFlag.IMMEDIATE,),
                register_part=TEST_LOW_WRITE_LOW16_I32_DESCRIPTOR.operands[0].reg_alts[0].register_part,
            ),
        ),
    )
    descriptor = replace(base, operands=(base.operands[0], operand, *base.operands[2:]))
    with pytest.raises(ValueError, match="immediate alternative cannot name"):
        compiler.compile_descriptor_set(replace(TEST_LOW_CORE_DESCRIPTOR_SET, descriptors=(descriptor,)))


def test_immediate_alternative_has_no_register_alignment() -> None:
    base = TEST_LOW_ADD_I32_DESCRIPTOR
    operand = replace(base.operands[1], reg_alts=(RegClassAlt(None, (RegClassAltFlag.IMMEDIATE,), unit_alignment=2),))
    descriptor = replace(base, operands=(base.operands[0], operand, *base.operands[2:]))
    with pytest.raises(ValueError, match="immediate alternative cannot require register alignment"):
        compiler.compile_descriptor_set(replace(TEST_LOW_CORE_DESCRIPTOR_SET, descriptors=(descriptor,)))


def test_late_read_alternative_is_interned_with_its_execution_mode() -> None:
    base = TEST_LOW_ADD_I32_DESCRIPTOR
    operands = tuple(replace(operand, reg_alts=(replace(operand.reg_alts[0], late_read_subgroup_size=width),)) for operand, width in zip(base.operands, (None, 0, 64), strict=True))
    compiled = compiler.compile_descriptor_set(replace(TEST_LOW_CORE_DESCRIPTOR_SET, descriptors=(replace(base, operands=operands),)))
    assert [(RegClassAltFlag.LATE_READ in row[2], row[4]) for row in compiled.reg_class_alts] == [(False, 0), (True, 0), (True, 64)]
    assert DescriptorFlag.LATE_READ in compiled.descriptors[0].flags


@pytest.mark.parametrize("width", [-1, 65536])
def test_late_read_subgroup_size_is_representable(width) -> None:
    base = TEST_LOW_ADD_I32_DESCRIPTOR
    operand = base.operands[1]
    operand = replace(operand, reg_alts=(replace(operand.reg_alts[0], late_read_subgroup_size=width),))
    descriptor = replace(base, operands=(base.operands[0], operand, base.operands[2]))
    with pytest.raises(ValueError, match="late-read subgroup size"):
        compiler.compile_descriptor_set(replace(TEST_LOW_CORE_DESCRIPTOR_SET, descriptors=(descriptor,)))


@pytest.mark.parametrize("operand_index", [0, 1])
def test_late_read_requires_a_register_input(operand_index) -> None:
    base = TEST_LOW_ADD_I32_DESCRIPTOR
    operand = base.operands[operand_index]
    alternative = replace(operand.reg_alts[0], late_read_subgroup_size=0) if operand_index == 0 else RegClassAlt(None, (RegClassAltFlag.IMMEDIATE,), late_read_subgroup_size=0)
    operands = list(base.operands)
    operands[operand_index] = replace(operand, reg_alts=(alternative,))
    with pytest.raises(ValueError, match="late read requires an explicit register input"):
        compiler.compile_descriptor_set(replace(TEST_LOW_CORE_DESCRIPTOR_SET, descriptors=(replace(base, operands=tuple(operands)),)))


def test_late_read_cannot_be_tied_to_a_result() -> None:
    base = TEST_LOW_ADD_I32_DESCRIPTOR
    operand = base.operands[1]
    operand = replace(operand, reg_alts=(replace(operand.reg_alts[0], late_read_subgroup_size=0),))
    descriptor = replace(base, operands=(base.operands[0], operand, base.operands[2]), constraints=(Constraint(ConstraintKind.TIED, 0, 1),))
    with pytest.raises(ValueError, match="late read cannot share a tied result"):
        compiler.compile_descriptor_set(replace(TEST_LOW_CORE_DESCRIPTOR_SET, descriptors=(descriptor,)))


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


@pytest.mark.parametrize(
    ("immediate", "message"),
    [
        (
            replace(
                TEST_LOW_CONST_I32_DESCRIPTOR.immediates[0],
                flags=(ImmediateFlag.SYMBOLIC, ImmediateFlag.READ_ONLY_DATA),
            ),
            "marks read-only data on a non-ordinal",
        ),
        (
            replace(
                TEST_LOW_CONST_I32_DESCRIPTOR.immediates[0],
                kind=ImmediateKind.ORDINAL,
                flags=(ImmediateFlag.READ_ONLY_DATA,),
            ),
            "marks read-only data without symbolic",
        ),
    ],
)
def test_read_only_data_immediate_requires_symbolic_ordinal(immediate, message) -> None:
    descriptor = replace(
        TEST_LOW_CONST_I32_DESCRIPTOR,
        immediates=(immediate,),
    )
    with pytest.raises(ValueError, match=message):
        compiler.compile_descriptor_set(replace(TEST_LOW_CORE_DESCRIPTOR_SET, descriptors=(descriptor,)))


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


@pytest.mark.parametrize(
    ("view_units", "expected_order"),
    [
        (("test.r2", "test.r3"), [0, 1, 2, 3]),
        (("test.r1", "test.r3"), [0, 1, 3, 2]),
    ],
)
def test_physical_packing_order_groups_partial_views_in_place(view_units, expected_order) -> None:
    reg_class = next(row for row in TEST_LOW_CORE_DESCRIPTOR_SET.reg_classes if row.name == "test.explicit32")
    views = (
        PhysicalRegisterView(
            physical_register="test.partial",
            reg_class=reg_class.name,
            units=view_units,
        ),
    )

    assert compiler._physical_register_packing_order(reg_class, views) == expected_order


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
    assert len(compiled.operands) == 4
    assert compiled.operand_indices == [0, 1, 2, 3, 0]
    assert len(compiled.immediates) == 1
    assert len(compiled.constraints) == 1
    assert compiled.feature_mask_words == [0x5]
    assert compiled.encoding_field_values == [EncodingFieldValue(7, 11)]

    asm_forms_by_descriptor = {form.descriptor_ordinal: form for form in compiled.asm_forms}
    add_form = asm_forms_by_descriptor[0]
    add_copy_form = asm_forms_by_descriptor[1]
    const_form = asm_forms_by_descriptor[2]
    const_copy_form = asm_forms_by_descriptor[3]
    assert add_form.layout_index == add_copy_form.layout_index
    assert const_form.layout_index == const_copy_form.layout_index
    assert add_form.layout_index != const_form.layout_index
    assert len(compiled.asm_table_storage.layouts) == 2
    assert compiled.asm_table_storage.operand_indices == [0, 1, 2]
    assert len(compiled.asm_table_storage.immediates) == 1


def test_shared_operand_rows_keep_descriptor_local_source_positions() -> None:
    base = TEST_LOW_ADD_I32_DESCRIPTOR
    reversed_sources = replace(
        base,
        key="test.reversed.i32",
        mnemonic="test.reversed.i32",
        operands=(base.operands[0], base.operands[2], base.operands[1]),
    )
    compiled = compiler.compile_descriptor_set(replace(TEST_LOW_CORE_DESCRIPTOR_SET, descriptors=(base, reversed_sources)))

    assert compiled.operand_indices == [0, 1, 2, 0, 3, 4]
    assert compiled.operands[1] == compiled.operands[4]
    assert compiled.operands[2] == compiled.operands[3]
    assert compiled.operand_source_value_indices == [0, 0, 1, 0, 1]


@pytest.mark.parametrize(
    "changes",
    [
        {},
        {"results": ("carry", "dst")},
        {"operands": ("rhs", "lhs")},
        {"result_value_types": ()},
        {"result_value_types": (AsmResultValueType(ScalarTypeKind.F32), None)},
        {"operands": (), "operand_segments": (AsmOperandSegment(AsmOperandSegmentDelimiter.PAREN, ("lhs", "rhs")),)},
        {"immediates": ()},
        {"immediates": (AsmImmediate("i32_value", "constant"),)},
        {"native_assembly_values": ()},
        {"native_assembly_values": (NativeAsmValue(NativeAsmValueKind.MODIFIER_LITERAL, literal="mode:1"),)},
        {"native_assembly_values": (NativeAsmValue(NativeAsmValueKind.OPERAND, field_name="rhs"), NativeAsmValue(NativeAsmValueKind.OPERAND, field_name="lhs"))},
    ],
)
def test_assembly_layout_interning_preserves_every_projection(changes) -> None:
    base = TEST_LOW_ADD_I32_DESCRIPTOR
    form = AsmForm(
        results=("dst", "carry"),
        operands=("lhs", "rhs"),
        result_value_types=(AsmResultValueType(ScalarTypeKind.I32), None),
        immediates=(AsmImmediate("i32_value", "literal"),),
        native_assembly_values=(NativeAsmValue(NativeAsmValueKind.OPERAND, field_name="lhs"), NativeAsmValue(NativeAsmValueKind.OPERAND, field_name="rhs")),
    )
    descriptors = tuple(
        replace(
            base,
            key=f"test.layout.{ordinal}",
            mnemonic=f"test.layout.{ordinal}",
            operands=(base.operands[0], replace(base.operands[0], field_name="carry"), *base.operands[1:]),
            immediates=TEST_LOW_CONST_I32_DESCRIPTOR.immediates,
            asm_forms=(replace(form, native_assembly_mnemonic=f"native.layout.{ordinal}", **(changes if ordinal else {})),),
        )
        for ordinal in range(2)
    )
    compiled = compiler.compile_descriptor_set(replace(TEST_LOW_CORE_DESCRIPTOR_SET, descriptors=descriptors))
    storage = compiled.asm_table_storage
    first, second = compiled.asm_forms
    assert (first.layout_index == second.layout_index) == (not changes)
    assert len(storage.layouts) == (2 if changes else 1)
    for asm_form in compiled.asm_forms:
        layout = storage.layouts[asm_form.layout_index]
        assert tuple(storage.operand_indices[layout.result_operand_index_start : layout.result_operand_index_start + layout.result_operand_index_count]) == asm_form.result_indices
        assert tuple(storage.operand_indices[layout.operand_index_start : layout.operand_index_start + layout.operand_index_count]) == asm_form.operand_indices
        assert tuple(storage.operand_segments[layout.operand_segment_start : layout.operand_segment_start + layout.operand_segment_count]) == asm_form.operand_segments
        assert tuple(storage.immediates[layout.immediate_start : layout.immediate_start + len(asm_form.immediates)]) == asm_form.immediates
        assert layout.explicit_immediate_count == asm_form.explicit_immediate_count
        assert layout.has_named_immediates == asm_form.has_named_immediates
        assert layout.requires_named_immediates == asm_form.requires_named_immediates
        assert layout.native_owns_immediate_syntax == asm_form.native_owns_immediate_syntax
        assert tuple(storage.native_values[layout.native_assembly_value_start : layout.native_assembly_value_start + layout.native_assembly_value_count]) == asm_form.native_assembly_values
        if layout.result_value_type_start is None:
            assert not asm_form.result_value_types
        else:
            assert tuple(storage.result_value_types[layout.result_value_type_start : layout.result_value_type_start + layout.result_operand_index_count]) == asm_form.result_value_types
    before = tuple(storage.layouts)
    storage.append_forms(compiled.asm_forms)
    assert tuple(storage.layouts) == before


@pytest.mark.parametrize("default_mask", range(8))
@pytest.mark.parametrize("named", [False, True])
def test_assembly_immediate_order_retains_complete_descriptor_projection(default_mask: int, named: bool) -> None:
    immediate = TEST_LOW_CONST_I32_DESCRIPTOR.immediates[0]
    fields = ("offset", "stride", "mode")
    immediates = tuple(replace(immediate, field_name=name, flags=(ImmediateFlag.DEFAULT_VALUE,) if default_mask & (1 << index) else ()) for index, name in enumerate(fields))
    orders = [order for count in range(4) for order in permutations(range(3), count)]
    forms = tuple(
        AsmForm(
            mnemonic=f"test.immediate_order.{ordinal}",
            results=("dst",),
            operands=("lhs", "rhs"),
            immediates=tuple(AsmImmediate(fields[index], f"named_{fields[index]}" if named else None) for index in order),
        )
        for ordinal, order in enumerate(orders)
    )
    descriptor = replace(TEST_LOW_ADD_I32_DESCRIPTOR, immediates=immediates, asm_forms=forms)
    compiled = compiler.compile_descriptor_set(replace(TEST_LOW_CORE_DESCRIPTOR_SET, descriptors=(descriptor,)))
    by_mnemonic = {form.mnemonic: form for form in compiled.asm_forms}
    storage = compiled.asm_table_storage
    for authored, explicit in zip(forms, orders, strict=True):
        form = by_mnemonic[authored.mnemonic]
        layout = storage.layouts[form.layout_index]
        rows = storage.immediates[layout.immediate_start : layout.immediate_start + len(fields)]
        expected_order = (*explicit, *(index for index in range(3) if index not in explicit))
        assert tuple(row.immediate_index for row in rows) == expected_order
        assert tuple(row.name for row in rows) == tuple(f"named_{fields[index]}" if named and index in explicit else None for index in expected_order)
        assert layout.explicit_immediate_count == len(explicit)
        assert layout.has_named_immediates == any(named or default_mask & (1 << index) for index in explicit)
        named_fields = expected_order if layout.has_named_immediates else expected_order[len(explicit) :]
        assert layout.requires_named_immediates == any(not default_mask & (1 << index) for index in named_fields)


@pytest.mark.parametrize("explicit", [False, True])
def test_assembly_layout_sharing_includes_descriptor_default_syntax(explicit: bool) -> None:
    base = TEST_LOW_CONST_I32_DESCRIPTOR
    descriptors = tuple(
        replace(
            base,
            key=f"test.default.{ordinal}",
            mnemonic=f"test.default.{ordinal}",
            immediates=(replace(base.immediates[0], flags=(ImmediateFlag.DEFAULT_VALUE,) if ordinal else (), default_value=7 if ordinal == 2 else 0),),
            asm_forms=(AsmForm(results=("dst",), immediates=(AsmImmediate("i32_value"),) if explicit else ()),),
        )
        for ordinal in range(3)
    )
    compiled = compiler.compile_descriptor_set(replace(TEST_LOW_CORE_DESCRIPTOR_SET, descriptors=descriptors))
    first, second, third = compiled.asm_forms
    assert first.layout_index != second.layout_index
    assert second.layout_index == third.layout_index
    assert [form.has_named_immediates for form in compiled.asm_forms] == [False, explicit, explicit]
    assert [form.requires_named_immediates for form in compiled.asm_forms] == [not explicit, False, False]


@pytest.mark.parametrize(
    ("value", "owns_syntax"),
    [
        (None, False),
        (NativeAsmValue(NativeAsmValueKind.LITERAL, literal="exec"), False),
        (NativeAsmValue(NativeAsmValueKind.MODIFIER_LITERAL, literal="mode:1"), False),
        (NativeAsmValue(NativeAsmValueKind.RESULT, field_name="dst"), False),
        (NativeAsmValue(NativeAsmValueKind.OPERAND, field_name="address"), False),
        (NativeAsmValue(NativeAsmValueKind.REGISTER_PART, field_name="dst"), False),
        (NativeAsmValue(NativeAsmValueKind.IMMEDIATE_I64, field_name="i32_value"), True),
        (NativeAsmValue(NativeAsmValueKind.IMMEDIATE_UNSIGNED_HEX, field_name="i32_value", bit_width=32), True),
        (NativeAsmValue(NativeAsmValueKind.IMMEDIATE_TARGET_FORMAT, field_name="i32_value", target_format_id=1), True),
    ],
)
def test_native_assembly_immediate_ownership_is_derived(value, owns_syntax: bool) -> None:
    base = TEST_LOW_WRITE_LOW16_I32_DESCRIPTOR
    form = replace(base.asm_forms[0], native_assembly_values=(value,) if value is not None else ())
    descriptor = replace(base, immediates=TEST_LOW_CONST_I32_DESCRIPTOR.immediates, asm_forms=(form,))
    compiled = compiler.compile_descriptor_set(replace(TEST_LOW_CORE_DESCRIPTOR_SET, descriptors=(descriptor,)))
    assert compiled.asm_forms[0].native_owns_immediate_syntax == owns_syntax
    assert compiled.asm_table_storage.layouts[0].native_owns_immediate_syntax == owns_syntax


def test_physical_view_lookup_preserves_exact_class_and_unit_relations() -> None:
    compiled = compiler.compile_descriptor_set(TEST_LOW_CORE_DESCRIPTOR_SET)
    expected = {(view.physical_register_id, view.reg_class_id): ordinal for ordinal, view in enumerate(compiled.physical_register_views)}
    for physical_id, lookup in enumerate(compiled.physical_register_view_lookups):
        ordinals = compiled.physical_register_view_ordinals[lookup.ordinal_start : lookup.ordinal_start + lookup.class_count]
        for class_id in range(len(compiled.reg_classes)):
            offset = class_id - lookup.class_base
            actual = ordinals[offset] if 0 <= offset < lookup.class_count else 0xFFFFFFFF
            assert actual == expected.get((physical_id, class_id), 0xFFFFFFFF)


@pytest.mark.parametrize("change", ["none", "value", "kind", "source", "map_order"])
def test_operand_form_interns_exact_predicates_and_maps(change: str) -> None:
    base = TEST_LOW_ADD_I32_DESCRIPTOR
    operands = (*base.operands, replace(base.operands[-1], field_name="extra"))
    replacements = []
    sources = []
    for ordinal in range(2):
        changed = ordinal == 1
        matched_index = 2 if changed and change == "source" else 3
        retained = [index for index in range(1, len(operands)) if index != matched_index]
        if changed and change == "map_order":
            retained.reverse()
        replacement = replace(
            base,
            key=f"test.replacement.{ordinal}",
            mnemonic=f"test.replacement.{ordinal}",
            operands=(operands[0], *(operands[index] for index in retained)),
            asm_forms=(),
        )
        match = OperandFormMatch(
            source_operand=operands[matched_index].field_name,
            match_kind=OperandFormMatchKind.ALL_EQUAL_EXACT_I64 if changed and change == "kind" else OperandFormMatchKind.ALL_EQUAL_I64,
            match_i64=17 if changed and change == "value" else 0,
        )
        sources.append(
            replace(
                base,
                key=f"test.source.{ordinal}",
                mnemonic=f"test.source.{ordinal}",
                operands=operands,
                asm_forms=(),
                operand_forms=(OperandForm(replacement_descriptor=replacement.key, matches=(match,)),),
            )
        )
        replacements.append(replacement)
    compiled = compiler.compile_descriptor_set(replace(TEST_LOW_CORE_DESCRIPTOR_SET, descriptors=(*sources, *replacements)))
    first, second = compiled.operand_forms
    same_matches = change in ("none", "map_order")
    same_map = change not in ("source", "map_order")
    assert (first.match_start == second.match_start) == same_matches
    assert (first.operand_map_start == second.operand_map_start) == same_map
    assert len(compiled.operand_form_matches) == (1 if same_matches else 2)
    assert len(compiled.operand_form_operand_indices) == (2 if same_map else 4)
    for source, replacement, form in zip(sources, replacements, compiled.operand_forms, strict=True):
        assert compiled.descriptors[form.replacement_descriptor_ordinal].key == replacement.key
        assert form.match_count == 1
        match = compiled.operand_form_matches[form.match_start]
        authored_match = source.operand_forms[0].matches[0]
        source_index = next(index for index, operand in enumerate(source.operands) if operand.field_name == authored_match.source_operand)
        assert match.source_operand_index == source_index
        assert match.source_packet_operand_index == source_index - 1
        assert match.match_kind == authored_match.match_kind
        assert match.match_i64 == authored_match.match_i64
        expected_map = [next(index for index, operand in enumerate(source.operands[1:]) if operand.field_name == retained.field_name) for retained in replacement.operands[1:]]
        assert compiled.operand_form_operand_indices[form.operand_map_start : form.operand_map_start + form.operand_map_count] == expected_map


@pytest.mark.parametrize("change", ["none", "width", "space", "order"])
def test_operand_form_preserves_effect_identity(change: str) -> None:
    base = TEST_LOW_LOAD_V4I32_DESCRIPTOR
    effects = (base.effects[0], replace(base.effects[0], memory_space=MemorySpace.WORKGROUP))
    replacement_effects = effects
    if change == "width":
        replacement_effects = (replace(effects[0], width_bits=256), effects[1])
    elif change == "space":
        replacement_effects = (replace(effects[0], memory_space=MemorySpace.GLOBAL), effects[1])
    elif change == "order":
        replacement_effects = tuple(reversed(effects))
    replacement = replace(base, key="test.load.variant", mnemonic="test.load.variant", operands=base.operands[:1], asm_forms=(), effects=replacement_effects)
    source = replace(
        base,
        effects=effects,
        operand_forms=(
            OperandForm(
                replacement_descriptor=replacement.key,
                matches=(OperandFormMatch(source_operand="address", match_kind=OperandFormMatchKind.ALL_EQUAL_I64, match_i64=0),),
            ),
        ),
    )
    descriptor_set = replace(TEST_LOW_CORE_DESCRIPTOR_SET, descriptors=(source, replacement))
    if change == "none":
        compiler.compile_descriptor_set(descriptor_set)
    else:
        with pytest.raises(ValueError, match="must preserve semantic effect ordinals"):
            compiler.compile_descriptor_set(descriptor_set)
