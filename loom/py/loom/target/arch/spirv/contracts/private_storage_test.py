# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from loom.target.arch.spirv.contracts.logical_core import (
    SPIRV_LOGICAL_CORE_CONTRACT_FRAGMENT,
)
from loom.target.arch.spirv.contracts.private_storage import private_storage_rules
from loom.target.arch.spirv.descriptors import SPIRV_LOGICAL_CORE_DESCRIPTOR_SET
from loom.target.arch.spirv.features import feature_bits_value
from loom.target.contracts import DescriptorRule, GuardKind


def _typed_memory_rules():
    for rule in private_storage_rules():
        if rule.source_op.name in ("view.load", "view.store"):
            yield rule


def _source_types(rule):
    return next(
        guard.type_pattern.elements
        for guard in rule.guards
        if guard.kind is GuardKind.VALUE_TYPE and guard.field == "view"
    )


def test_private_memory_covers_each_cell_width_and_payload_family() -> None:
    actual = set()
    for rule in _typed_memory_rules():
        rule.validate(SPIRV_LOGICAL_CORE_DESCRIPTOR_SET)
        memory = rule.emit[0].source_memory
        cell_width = memory.minimum_alignment
        representation = next(
            guard.count
            for guard in rule.guards
            if guard.kind is GuardKind.LOW_VALUE_REPRESENTATION
        )
        assert representation == 16 + cell_width.bit_length() - 1
        assert memory.memory_spaces == ("private",)
        transfers = tuple(
            step
            for step in rule.emit
            if step.descriptor.key.startswith(
                ("spirv.op_load.function.", "spirv.op_store.function.")
            )
        )
        assert len(transfers) == memory.element_byte_count // cell_width
        assert all(step.source_memory is memory for step in transfers)
        assert all(
            step.descriptor.key.endswith(f".i{cell_width * 8}") for step in transfers
        )
        software_bfloat = any(
            guard.kind is GuardKind.LOW_VALUE_REGISTER_CLASS for guard in rule.guards
        )
        for element in _source_types(rule):
            key = (
                element,
                cell_width,
                rule.source_op.name,
                memory.allow_dynamic_stride_values,
                software_bfloat,
            )
            assert key not in actual
            actual.add(key)

    assert actual == {
        (element, cell, operation, dynamic, software_bfloat)
        for element, width in (
            ("i8", 1),
            ("f8E4M3", 1),
            ("f8E5M2", 1),
            ("i16", 2),
            ("f16", 2),
            ("bf16", 2),
            ("i32", 4),
            ("f32", 4),
            ("i64", 8),
            ("f64", 8),
        )
        for cell in (1, 2, 4, 8)
        if cell <= width
        for operation in ("view.load", "view.store")
        for dynamic in (False, True)
        for software_bfloat in ((False, True) if element == "bf16" else (False,))
    }


def test_private_cells_require_arithmetic_features_without_global_storage() -> None:
    baseline = feature_bits_value(("vulkan_shader", "int64"))
    for rule in _typed_memory_rules():
        if _source_types(rule) != ("i32",):
            continue
        width = rule.emit[0].source_memory.minimum_alignment
        required = 0
        for guard in rule.guards:
            if guard.kind is GuardKind.DESCRIPTOR_AVAILABLE:
                for mask in guard.descriptor.feature_mask_words:
                    required |= mask
        assert required & ~baseline == feature_bits_value(
            () if width == 4 else ("int16",) if width == 2 else ("int8",)
        )


def test_raw_byte_rules_keep_physical_and_logical_storage_separate() -> None:
    observed = set()
    for rule in SPIRV_LOGICAL_CORE_CONTRACT_FRAGMENT.cases:
        if not isinstance(rule, DescriptorRule) or rule.source_op.name not in (
            "buffer.load.i8.u",
            "buffer.store.i8",
        ):
            continue
        spaces = next(
            guard.memory_spaces
            for guard in rule.guards
            if guard.kind is GuardKind.VALUE_MEMORY_SPACE
        )
        is_load = rule.source_op.name == "buffer.load.i8.u"
        assert "workgroup" not in spaces
        assert ("constant" in spaces) == (is_load and "global" in spaces)
        assert not ("private" in spaces and "global" in spaces)
        observed.add((is_load, "private" in spaces))
    assert observed == {
        (load, private) for load in (False, True) for private in (False, True)
    }
