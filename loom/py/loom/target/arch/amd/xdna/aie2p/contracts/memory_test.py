# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for AMD XDNA AIE2P scalar and vector memory selection."""

from loom.dialect.vector import defs as vector
from loom.dialect.view import defs as view
from loom.target.arch.amd.xdna.aie2p.contracts.memory import AIE2P_MEMORY_RULES
from loom.target.contracts import (
    GuardKind,
    SourceMemoryAddressLayout,
    SourceMemoryOperation,
    SourceMemoryProject,
    SourceMemoryProjectKind,
    SourceMemoryRootKind,
)

_I32_MIN = -(2**31)
_I32_MAX = (2**31) - 1


def _rules_for(*source_ops):
    return [rule for rule in AIE2P_MEMORY_RULES if rule.source_op in source_ops]


def _assert_address_forms(
    rules,
    *,
    immediate_minimum: int,
    immediate_maximum: int,
) -> None:
    assert [len(rule.emit) for rule in rules] == [1, 3, 2, 3, 4]
    assert [
        (
            rule.emit[-1].source_memory.static_byte_offset_minimum,
            rule.emit[-1].source_memory.static_byte_offset_maximum,
            rule.emit[-1].source_memory.dynamic_term_count,
            rule.emit[-1].source_memory.dynamic_term_count_minimum,
            rule.emit[-1].source_memory.allow_dynamic_stride_values,
        )
        for rule in rules
    ] == [
        (immediate_minimum, immediate_maximum, 0, 0, False),
        (_I32_MIN, _I32_MAX, 0, 0, False),
        (0, 0, None, 1, True),
        (-64, 63, None, 1, True),
        (_I32_MIN, _I32_MAX, None, 1, True),
    ]
    assert [
        sum(guard.kind is GuardKind.OPERAND_SEGMENT_COUNT for guard in rule.guards)
        for rule in rules
    ] == [1, 1, 0, 0, 0]
    immediate = rules[0].emit[0].immediates["imm"]
    assert isinstance(immediate, SourceMemoryProject)
    assert immediate.kind is SourceMemoryProjectKind.STATIC_BYTE_OFFSET
    for rule in rules[1:]:
        assert rule.emit[-2].descriptor.key == ("amd.xdna.aie2p.move.to.address-index")


def test_scalar_memory_rules_cover_every_address_form() -> None:
    rules = _rules_for(view.view_load, view.view_store)
    assert len(rules) == 10
    assert [rule.descriptor.key for rule in rules] == [
        "amd.xdna.aie2p.load.scalar.indexed.immediate",
        *("amd.xdna.aie2p.load.scalar.indexed.register",) * 4,
        "amd.xdna.aie2p.store.scalar.indexed.immediate",
        *("amd.xdna.aie2p.store.scalar.indexed.register",) * 4,
    ]
    for operation_rules in (rules[:5], rules[5:]):
        _assert_address_forms(
            operation_rules,
            immediate_minimum=-32,
            immediate_maximum=28,
        )
    for rule in rules:
        memory_emit = rule.emit[-1]
        constraint = memory_emit.source_memory
        assert constraint is not None
        assert constraint.root_kind is SourceMemoryRootKind.BLOCK_ARGUMENT
        assert constraint.address_layout is SourceMemoryAddressLayout.COMPACT_ROW_MAJOR
        assert constraint.memory_spaces == ("unknown", "generic", "workgroup")
        assert constraint.element_byte_count == 4
        assert constraint.vector_lane_count == 1
        assert constraint.vector_lane_byte_stride == 4
        assert constraint.minimum_alignment == 4


def test_vector_memory_rules_cover_every_full_width_address_form() -> None:
    rules = _rules_for(vector.vector_load, vector.vector_store)
    assert len(rules) == 30
    expected_shapes = (
        (1, 64),
        (1, 64),
        (2, 32),
        (2, 32),
        (4, 16),
        (4, 16),
    )
    expected_operations = (
        SourceMemoryOperation.LOAD,
        SourceMemoryOperation.STORE,
    ) * 3
    for family_index, ((element_byte_count, vector_lane_count), operation) in enumerate(
        zip(expected_shapes, expected_operations, strict=True)
    ):
        operation_rules = rules[family_index * 5 : family_index * 5 + 5]
        descriptor_prefix = (
            "amd.xdna.aie2p.load.a.i8x64.indexed"
            if operation is SourceMemoryOperation.LOAD
            else "amd.xdna.aie2p.store.i8x64.indexed"
        )
        assert [rule.descriptor.key for rule in operation_rules] == [
            f"{descriptor_prefix}.immediate",
            *(f"{descriptor_prefix}.register",) * 4,
        ]
        _assert_address_forms(
            operation_rules,
            immediate_minimum=-512,
            immediate_maximum=448,
        )
        for rule in operation_rules:
            memory_emit = rule.emit[-1]
            constraint = memory_emit.source_memory
            assert constraint is not None
            assert constraint.operation is operation
            assert constraint.root_kind is SourceMemoryRootKind.BLOCK_ARGUMENT
            assert (
                constraint.address_layout is SourceMemoryAddressLayout.COMPACT_ROW_MAJOR
            )
            assert constraint.memory_spaces == (
                "unknown",
                "generic",
                "workgroup",
            )
            assert constraint.element_byte_count == element_byte_count
            assert constraint.vector_lane_count == vector_lane_count
            assert constraint.vector_lane_byte_stride == element_byte_count
            assert constraint.minimum_alignment == 64
