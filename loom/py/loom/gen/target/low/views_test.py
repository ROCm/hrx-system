# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Register vocabulary and scheduling contracts in shared descriptor views."""

from __future__ import annotations

from dataclasses import replace

import pytest

from loom.gen.target.low import compiler, views
from loom.gen.target.low.low_descriptors import (
    DescriptorAllowlist,
    generate_descriptor_set,
    generate_descriptor_set_family,
)
from loom.target.low_descriptors import DescriptorSet
from loom.target.test.descriptors import (
    TEST_LOW_ADD_I32_DESCRIPTOR,
    TEST_LOW_CORE_DESCRIPTOR_SET,
)


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
    expected_flags = ["LOOM_LOW_DESCRIPTOR_SET_FLAG_NATIVE_SCHEDULING"] if supports_native_scheduling else []
    if expected:
        expected_flags.append("LOOM_LOW_DESCRIPTOR_SET_FLAG_POSITIVE_EFFECT_SEPARATIONS")
    expected_field = f".flags = {' | '.join(expected_flags) or '0'},"
    # A family view retains shared event tables even when its selected
    # descriptors do not use the positive effect pairs in those tables.
    assert expected_field in generate_descriptor_set_family(storage, (view,)).source
    assert expected_field in generate_descriptor_set(storage, DescriptorAllowlist(keys=descriptor_keys)).source


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
    expected_flags = "LOOM_LOW_DESCRIPTOR_SET_FLAG_NATIVE_SCHEDULING"
    if expected:
        expected_flags += " | LOOM_LOW_DESCRIPTOR_SET_FLAG_POSITIVE_EFFECT_SEPARATIONS"
    assert f".flags = {expected_flags}," in source
