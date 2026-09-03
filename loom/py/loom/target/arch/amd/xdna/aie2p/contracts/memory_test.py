# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for AMD XDNA AIE2P scalar and vector memory selection."""

from dataclasses import replace

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


_MEMORY_ROOTS = (
    (
        SourceMemoryRootKind.BLOCK_ARGUMENT,
        ("unknown", "generic", "workgroup"),
    ),
    (
        SourceMemoryRootKind.ALLOCA,
        ("private", "workgroup"),
    ),
)


def _rules_for(root_kind, *source_ops, volatile: bool = False):
    return [
        rule
        for rule in AIE2P_MEMORY_RULES
        if rule.source_op in source_ops
        and rule.emit[-1].source_memory.root_kind is root_kind
        and rule.descriptor.key.endswith(".volatile") is volatile
    ]


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
    for root_kind, memory_spaces in _MEMORY_ROOTS:
        rules = _rules_for(root_kind, view.view_load, view.view_store)
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
            assert constraint.root_kind is root_kind
            assert (
                constraint.address_layout is SourceMemoryAddressLayout.COMPACT_ROW_MAJOR
            )
            assert constraint.memory_spaces == memory_spaces
            assert constraint.element_byte_count == 4
            assert constraint.vector_lane_count == 1
            assert constraint.vector_lane_byte_stride == 4
            assert constraint.minimum_alignment == 4


def test_volatile_memory_rules_mirror_every_ordinary_rule() -> None:
    for root_kind, _ in _MEMORY_ROOTS:
        for source_ops in (
            (view.view_load, view.view_store),
            (vector.vector_load, vector.vector_store),
        ):
            ordinary_rules = _rules_for(root_kind, *source_ops)
            volatile_rules = _rules_for(root_kind, *source_ops, volatile=True)
            assert len(volatile_rules) == len(ordinary_rules)
            for volatile_rule, ordinary_rule in zip(
                volatile_rules, ordinary_rules, strict=True
            ):
                assert volatile_rule.descriptor.key == (
                    f"{ordinary_rule.descriptor.key}.volatile"
                )
                assert volatile_rule.guards[0].kind is (
                    GuardKind.INSTANCE_FLAGS_HAS_ALL
                )
                assert volatile_rule.guards[0].field == "memory_flags"
                assert volatile_rule.guards[0].enum_keyword == "volatile"
                assert volatile_rule.guards[1:] == ordinary_rule.guards
                assert len(volatile_rule.emit) == len(ordinary_rule.emit)
                for emit_index, (volatile_emit, ordinary_emit) in enumerate(
                    zip(volatile_rule.emit, ordinary_rule.emit, strict=True)
                ):
                    expected_descriptor = ordinary_emit.descriptor
                    if emit_index == len(ordinary_rule.emit) - 1:
                        assert volatile_emit.descriptor.key == (
                            f"{ordinary_emit.descriptor.key}.volatile"
                        )
                    else:
                        assert volatile_emit.descriptor == ordinary_emit.descriptor
                    assert (
                        replace(volatile_emit, descriptor=expected_descriptor)
                        == ordinary_emit
                    )


def test_vector_memory_rules_cover_every_native_width_and_address_form() -> None:
    expected_families = tuple(
        (
            width_bits,
            element_type,
            element_byte_count,
            width_bits // (element_byte_count * 8),
            operation,
        )
        for width_bits in (128, 256, 512)
        for element_type, element_byte_count in (
            ("i8", 1),
            ("i16", 2),
            ("bf16", 2),
            ("i32", 4),
            ("f32", 4),
        )
        for operation in (
            SourceMemoryOperation.LOAD,
            SourceMemoryOperation.STORE,
        )
    )
    for root_kind, memory_spaces in _MEMORY_ROOTS:
        rules = _rules_for(root_kind, vector.vector_load, vector.vector_store)
        expected_descriptor_keys = []
        for (
            _,
            element_type,
            _,
            vector_lane_count,
            operation,
        ) in expected_families:
            shape = f"{element_type}x{vector_lane_count}"
            descriptor_prefix = (
                f"amd.xdna.aie2p.load.a.{shape}.indexed"
                if operation is SourceMemoryOperation.LOAD
                else f"amd.xdna.aie2p.store.{shape}.indexed"
            )
            expected_descriptor_keys.extend(
                [
                    f"{descriptor_prefix}.immediate",
                    *(f"{descriptor_prefix}.register",) * 4,
                ]
            )
        assert [rule.descriptor.key for rule in rules] == expected_descriptor_keys
        for family_index, (
            width_bits,
            element_type,
            element_byte_count,
            vector_lane_count,
            operation,
        ) in enumerate(expected_families):
            operation_rules = rules[family_index * 5 : family_index * 5 + 5]
            shape = f"{element_type}x{vector_lane_count}"
            descriptor_prefix = (
                f"amd.xdna.aie2p.load.a.{shape}.indexed"
                if operation is SourceMemoryOperation.LOAD
                else f"amd.xdna.aie2p.store.{shape}.indexed"
            )
            assert [rule.descriptor.key for rule in operation_rules] == [
                f"{descriptor_prefix}.immediate",
                *(f"{descriptor_prefix}.register",) * 4,
            ]
            _assert_address_forms(
                operation_rules,
                immediate_minimum=-width_bits,
                immediate_maximum=width_bits - width_bits // 8,
            )
            for rule in operation_rules:
                memory_emit = rule.emit[-1]
                constraint = memory_emit.source_memory
                assert constraint is not None
                assert constraint.operation is operation
                assert constraint.root_kind is root_kind
                assert (
                    constraint.address_layout
                    is SourceMemoryAddressLayout.COMPACT_ROW_MAJOR
                )
                assert constraint.memory_spaces == memory_spaces
                assert constraint.element_byte_count == element_byte_count
                assert constraint.vector_lane_count == vector_lane_count
                assert constraint.vector_lane_byte_stride == element_byte_count
                assert constraint.minimum_alignment == width_bits // 8
