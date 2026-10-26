# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Register vocabulary and scheduling contracts in shared descriptor views."""

from __future__ import annotations

from dataclasses import replace

import pytest

from loom.gen.target.low import c_emit, compiler, views
from loom.gen.target.low.low_descriptors import (
    DescriptorAllowlist,
    compile_descriptor_set_family,
    generate_descriptor_set,
    generate_descriptor_set_family,
)
from loom.target.low_descriptors import DescriptorSet, Hazard, HazardKind, PressureDelta, descriptor_stable_id
from loom.target.test.descriptors import (
    TEST_LOW_ADD_I32_DESCRIPTOR,
    TEST_LOW_CONST_I32_DESCRIPTOR,
    TEST_LOW_CORE_DESCRIPTOR_SET,
    TEST_LOW_LOAD_V4I32_DESCRIPTOR,
    TEST_LOW_STATE_ADD_I32_DESCRIPTOR,
    TEST_LOW_STATE_ADD_I32_RHS_ZERO_DESCRIPTOR,
    TEST_LOW_STATE_UPDATE_I32_DESCRIPTOR,
)


def _emitted_initializer(source: str, declaration: str) -> str:
    start = source.index(declaration)
    return source[start : source.index("\n};", start)]


def test_descriptor_layout_pool_preserves_all_compiled_spans() -> None:
    compiled = compiler.compile_descriptor_set(TEST_LOW_CORE_DESCRIPTOR_SET)
    layouts = []
    rows = c_emit._descriptor_row_lines(compiled, compiled.descriptors, compiled.descriptor_rows, layouts, {})
    assert len(layouts) < len(rows)
    for descriptor, source, row in zip(compiled.descriptors, compiled.descriptor_rows, rows, strict=True):
        index = int(next(field for field in row if field.startswith(".layout_index = ")).removeprefix(".layout_index = ").removesuffix(","))
        expected = []
        for span in ("feature_mask_word", "encoding_field_value", "immediate", "effect", "constraint", "storage_lease", "operand_form"):
            count = source[f"{span}_count"]
            expected.extend((f".{span}_start = {source[f'{span}_start'] if count else 0},", f".{span}_count = {count},"))
        expected.extend(
            (f".encoding_format_id = {descriptor.encoding_format_id},", f".minimum_packet_operand_count = {source['minimum_packet_operand_count']},", f".op_kind = {descriptor.op_kind.c_name},")
        )
        assert layouts[index] == tuple(expected)


def test_descriptor_identity_and_encoding_do_not_duplicate_layouts() -> None:
    base = TEST_LOW_ADD_I32_DESCRIPTOR
    descriptors = tuple(replace(base, key=f"test.add.variant{index}", mnemonic=f"variant{index}", semantic_tag=f"test.variant{index}", encoding_id=index) for index in range(3))
    spec = replace(TEST_LOW_CORE_DESCRIPTOR_SET, descriptors=descriptors)
    compiled = compiler.compile_descriptor_set(spec)
    layouts = []
    rows = c_emit._descriptor_row_lines(compiled, compiled.descriptors, compiled.descriptor_rows, layouts, {})
    assert len(layouts) == 1
    assert len({tuple(row) for row in rows}) == 3
    assert all(".layout_index = 0," in row for row in rows)
    source = generate_descriptor_set(spec).source
    assert source.count("static const loom_low_descriptor_layout_t ") == 1
    assert ".descriptor_layouts = kTestLowCoreDescriptorLayouts," in source
    assert ".descriptor_layout_count = 1," in source


def test_descriptor_layouts_preserve_view_local_operand_form_spans() -> None:
    base = TEST_LOW_STATE_ADD_I32_DESCRIPTOR
    zero = TEST_LOW_STATE_ADD_I32_RHS_ZERO_DESCRIPTOR
    sibling_zero = replace(zero, key="test.sibling.rhs_zero", mnemonic="test.sibling.rhs_zero")
    sibling = replace(base, key="test.sibling", mnemonic="test.sibling", operand_forms=(replace(base.operand_forms[0], replacement_descriptor=sibling_zero.key),))
    storage = replace(TEST_LOW_CORE_DESCRIPTOR_SET, descriptors=(base, zero, sibling, sibling_zero))
    view_sibling = replace(sibling, asm_forms=(replace(sibling.asm_forms[0], mnemonic="view.add"),))
    spec = replace(storage, key="test.low.sibling", c_table_prefix="TestSibling", function_name="loom_test_sibling_descriptor_set", descriptors=(view_sibling, sibling_zero))
    compiled, (view,) = compile_descriptor_set_family(storage, (spec,))
    assert not view.uses_storage_descriptor_tables
    assert compiled.descriptor_rows[2]["operand_form_start"] == 1
    assert view.descriptor_rows[0]["operand_form_start"] == 0
    layouts = []
    indices = {}
    storage_rows = c_emit._descriptor_row_lines(compiled, compiled.descriptors, compiled.descriptor_rows, layouts, indices)
    view_rows = c_emit._descriptor_row_lines(compiled, view.descriptors, view.descriptor_rows, layouts, indices)
    storage_index = next(field for field in storage_rows[2] if field.startswith(".layout_index"))
    view_index = next(field for field in view_rows[0] if field.startswith(".layout_index"))
    assert storage_index != view_index
    # The view's remapped span can share the first storage descriptor's layout:
    # its offset is interpreted against the selected view's operand-form table.
    assert view_index in storage_rows[0]
    source = generate_descriptor_set_family(storage, (storage, spec)).source
    assert source.count("static const loom_low_descriptor_layout_t ") == 1
    assert source.count(".descriptor_layouts = kTestLowCoreDescriptorLayouts,") == 2
    assert ".operand_forms = kTestSiblingOperandForms," in source


@pytest.mark.parametrize("selected_ordinals", [(0,), (0, 1), (0, 2), (2, 0)])
def test_descriptor_lookup_uses_the_view_ordinal_space(selected_ordinals) -> None:
    storage = replace(TEST_LOW_CORE_DESCRIPTOR_SET, descriptors=(TEST_LOW_CONST_I32_DESCRIPTOR, TEST_LOW_ADD_I32_DESCRIPTOR, TEST_LOW_LOAD_V4I32_DESCRIPTOR))
    compiled = compiler.compile_descriptor_set(storage)
    spec = replace(storage, descriptors=tuple(storage.descriptors[ordinal] for ordinal in selected_ordinals))
    view = views.descriptor_set_view_for_spec(compiled, spec)
    lookup_descriptors = compiled.descriptors if view.uses_storage_descriptor_tables else view.descriptors
    keys = [lookup_descriptors[ordinal].key for ordinal in view.descriptor_refs]
    assert set(keys) == {descriptor.key for descriptor in spec.descriptors}
    identities = [descriptor_stable_id(key) for key in keys]
    assert identities == sorted(identities)


def _view_spec() -> DescriptorSet:
    storage = TEST_LOW_CORE_DESCRIPTOR_SET
    classes = {item.name: item for item in storage.reg_classes}
    return replace(
        storage,
        key="test.low.register_view.core",
        function_name="loom_test_low_register_view_core_descriptor_set",
        c_table_prefix="TestLowRegisterViewCore",
        c_enum_prefix="TEST_LOW_REGISTER_VIEW_CORE",
        reg_classes=(
            classes["test.i32"],
            replace(
                classes["test.phys"],
                allocatable_count=16,
                fixed_location_base=16,
                fixed_location_count=4,
            ),
        ),
        descriptors=(TEST_LOW_ADD_I32_DESCRIPTOR,),
    )


def test_view_preserves_capacity_fixed_locations_and_absence() -> None:
    compiled = compiler.compile_descriptor_set(TEST_LOW_CORE_DESCRIPTOR_SET)
    spec = _view_spec()
    view = views.descriptor_set_view_for_spec(compiled, spec)
    assert len(view.reg_classes) == len(compiled.reg_classes)
    assert view.reg_classes[compiled.reg_class_ids["test.phys"]] == spec.reg_classes[1]
    assert view.reg_classes[compiled.reg_class_ids["test.f32"]] is None
    assert compiled.reg_classes[compiled.reg_class_ids["test.phys"]].allocatable_count == 32


def test_view_register_ids_follow_storage_not_authored_order() -> None:
    storage = TEST_LOW_CORE_DESCRIPTOR_SET
    spec = _view_spec()
    reordered = replace(spec, reg_classes=tuple(reversed(spec.reg_classes)))
    compiled = compiler.compile_descriptor_set(storage)
    original_view = views.descriptor_set_view_for_spec(compiled, spec)
    reordered_view = views.descriptor_set_view_for_spec(compiled, reordered)
    assert reordered_view.reg_classes == original_view.reg_classes

    generated = tuple(generate_descriptor_set_family(storage, (view,)) for view in (spec, reordered))
    assert generated[0].source == generated[1].source
    for family in generated:
        header = family.view_headers[0]
        for name in ("i32", "phys"):
            storage_id = compiled.reg_class_ids[f"test.{name}"]
            assert f"TEST_LOW_REGISTER_VIEW_CORE_REG_CLASS_ID_TEST_{name.upper()} = {storage_id}u" in header
        assert "REG_CLASS_ID_TEST_F32" not in header


def test_equal_view_register_tables_share_storage() -> None:
    spec = _view_spec()
    alias = replace(
        spec,
        key="test.low.register_alias.core",
        function_name="loom_test_low_register_alias_core_descriptor_set",
        c_table_prefix="TestLowRegisterAliasCore",
    )
    source = generate_descriptor_set_family(TEST_LOW_CORE_DESCRIPTOR_SET, (spec, alias)).source
    assert source.count(".reg_classes = kTestLowRegisterViewCoreRegClasses,") == 2
    assert "kTestLowRegisterAliasCoreRegClasses[]" not in source
    assert "kTestLowCoreRegClasses[]" not in source
    assert ".name_string_ref = LOOM_STRING_REF_NONE," in source
    assert source.count(".operands = kTestLowCoreOperands,") == 2


def test_view_only_schedule_reuses_storage_spans_without_merging_identity() -> None:
    descriptor = TEST_LOW_ADD_I32_DESCRIPTOR
    schedule = next(schedule for schedule in TEST_LOW_CORE_DESCRIPTOR_SET.schedule_classes if schedule.name == descriptor.schedule_class)
    schedule = replace(schedule, hazards=(Hazard(HazardKind.MIN_DISTANCE, target_id=1, distance=2),), pressure_deltas=(PressureDelta("test.i32", 1),))
    view_schedule = replace(schedule, name="test.view.alu", latency_cycles=7)
    storage = replace(TEST_LOW_CORE_DESCRIPTOR_SET, descriptors=(descriptor,), schedule_classes=(schedule, view_schedule))
    view_spec = replace(_view_spec(), descriptors=(replace(descriptor, schedule_class=view_schedule.name),), schedule_classes=(view_schedule,))

    compiled, (view,) = compile_descriptor_set_family(storage, (view_spec,))

    assert compiled.schedule_class_ids == {schedule.name: 0, view_schedule.name: 1}
    assert compiled.schedule_rows[0] == compiled.schedule_rows[1]
    assert compiled.issue_uses == list(schedule.issue_uses)
    assert compiled.hazards == list(schedule.hazards)
    assert compiled.pressure_deltas == list(schedule.pressure_deltas)
    assert compiled.schedule_classes[1].latency_cycles == 7
    assert view.descriptors[0].schedule_class == view_schedule.name


@pytest.mark.parametrize(
    "changes",
    [
        {"alloc_unit_bits": 256},
        {"target_bank_id": 1},
        {"full_register_part_mask": 3},
    ],
)
def test_view_rejects_storage_identity_changes(changes: dict[str, int]) -> None:
    compiled = compiler.compile_descriptor_set(TEST_LOW_CORE_DESCRIPTOR_SET)
    spec = _view_spec()
    spec = replace(
        spec,
        reg_classes=(spec.reg_classes[0], replace(spec.reg_classes[1], **changes)),
    )
    with pytest.raises(ValueError, match="differs from storage outside allocation"):
        views.descriptor_set_view_for_spec(compiled, spec)


def test_view_rejects_descriptor_reference_to_absent_class() -> None:
    compiled = compiler.compile_descriptor_set(TEST_LOW_CORE_DESCRIPTOR_SET)
    spec = _view_spec()
    spec = replace(spec, reg_classes=spec.reg_classes[1:])
    with pytest.raises(ValueError, match=r"references absent register classes: test\.i32"):
        views.descriptor_set_view_for_spec(compiled, spec)


def test_view_rejects_invalid_fixed_location_geometry() -> None:
    compiled = compiler.compile_descriptor_set(TEST_LOW_CORE_DESCRIPTOR_SET)
    spec = _view_spec()
    spec = replace(
        spec,
        reg_classes=(
            spec.reg_classes[0],
            replace(spec.reg_classes[1], fixed_location_base=15),
        ),
    )
    with pytest.raises(ValueError, match="fixed-location range overlaps"):
        views.descriptor_set_view_for_spec(compiled, spec)


def test_view_retains_alias_indices_from_shared_namespace() -> None:
    compiled = compiler.compile_descriptor_set(TEST_LOW_CORE_DESCRIPTOR_SET)
    classes = {item.name: item for item in compiled.reg_classes}
    spec = replace(
        _view_spec(),
        reg_classes=(classes["test.i32"], classes["test.pressure.alias32"]),
    )
    view = views.descriptor_set_view_for_spec(compiled, spec)
    row = view.reg_classes[compiled.reg_class_ids["test.pressure.alias32"]]
    assert row is not None and row.alias_set_id == 2
    assert view.reg_classes[compiled.reg_class_ids["test.alias32"]] is None


@pytest.mark.parametrize(
    ("descriptor_keys", "expected"),
    [
        (("test.add.i32",), False),
        (("test.event.fast.i32", "test.event.consume.late.i32"), False),
        (("test.event.memory.read.i32",), False),
        (("test.event.memory.write.i32",), True),
        (("test.event.memory.read.i32", "test.event.memory.write.i32"), True),
    ],
)
@pytest.mark.parametrize("supports_native_scheduling", [False, True])
def test_effect_timing_summary_uses_selected_effect_endpoints(descriptor_keys: tuple[str, ...], expected: bool, supports_native_scheduling: bool) -> None:
    storage = replace(TEST_LOW_CORE_DESCRIPTOR_SET, supports_native_scheduling=supports_native_scheduling)
    selected = tuple(descriptor for descriptor in storage.descriptors if descriptor.key in descriptor_keys)
    view = replace(storage, descriptors=selected)
    expected_descriptor_flags = "LOOM_LOW_DESCRIPTOR_SET_FLAG_NATIVE_SCHEDULING" if supports_native_scheduling else "0"
    expected_model_flags = "LOOM_LOW_SCHEDULE_MODEL_FLAG_POSITIVE_EFFECT_SEPARATIONS" if expected else "0"
    # A family view retains shared event tables even when its selected
    # descriptors do not use the positive effect pairs in those tables.
    for source in (
        generate_descriptor_set_family(storage, (view,)).source,
        generate_descriptor_set(storage, DescriptorAllowlist(keys=descriptor_keys)).source,
    ):
        descriptor_set = _emitted_initializer(
            source,
            "static const loom_low_descriptor_set_t kTestLowCoreSet = {",
        )
        schedule_model = _emitted_initializer(
            source,
            "static const loom_low_schedule_model_t kTestLowCoreScheduleModel[] = {",
        )
        assert f".flags = {expected_descriptor_flags}," in descriptor_set
        assert f".flags = {expected_model_flags}," in schedule_model


@pytest.mark.parametrize(
    ("descriptor_key", "expected"),
    [
        ("test.add.i32", False),
        (TEST_LOW_STATE_UPDATE_I32_DESCRIPTOR.key, True),
    ],
)
def test_descriptor_set_summarizes_idempotent_state_updates(descriptor_key: str, expected: bool) -> None:
    source = generate_descriptor_set(
        TEST_LOW_CORE_DESCRIPTOR_SET,
        DescriptorAllowlist(keys=(descriptor_key,)),
    ).source
    descriptor_set = _emitted_initializer(
        source,
        "static const loom_low_descriptor_set_t kTestLowCoreSet = {",
    )
    flags = next(line for line in descriptor_set.splitlines() if ".flags = " in line)
    assert ("LOOM_LOW_DESCRIPTOR_SET_FLAG_IDEMPOTENT_STATE_UPDATES" in flags) == expected


@pytest.mark.parametrize("separation_cycles", [-2, 0, 3])
def test_effect_timing_summary_distinguishes_positive_separations(
    separation_cycles: int,
) -> None:
    storage = TEST_LOW_CORE_DESCRIPTOR_SET
    write = next(descriptor for descriptor in storage.descriptors if descriptor.key == "test.event.memory.write.i32")
    event = write.effects[0].producer_event
    separation = next(item for item in storage.event_separations if item.producer_event == event and item.consumer_event == event)
    storage = replace(
        storage,
        event_separations=(replace(separation, minimum_issue_separation_cycles=separation_cycles),),
    )
    view = replace(storage, descriptors=(write,))
    expected = separation_cycles > 0
    source = generate_descriptor_set_family(storage, (view,)).source
    expected_flags = "LOOM_LOW_SCHEDULE_MODEL_FLAG_POSITIVE_EFFECT_SEPARATIONS" if expected else "0"
    schedule_model = _emitted_initializer(
        source,
        "static const loom_low_schedule_model_t kTestLowCoreScheduleModel[] = {",
    )
    assert f".flags = {expected_flags}," in schedule_model
