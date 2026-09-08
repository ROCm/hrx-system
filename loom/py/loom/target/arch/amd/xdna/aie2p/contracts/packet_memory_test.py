# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for AIE2P packet-converting memory selection."""

from loom.dialect.vector import defs as vector
from loom.target.arch.amd.xdna.aie2p.contracts.packet_conversion import (
    INTEGER_PACK_CASES,
    INTEGER_WIDEN_CASES,
)
from loom.target.arch.amd.xdna.aie2p.contracts.packet_memory import (
    AIE2P_PACKET_MEMORY_RULES,
)
from loom.target.contracts import (
    EmitDescriptorOp,
    SourceMemoryAddressLayout,
    SourceMemoryOperation,
    SourceMemoryRootKind,
    SourceNodeRelation,
    ValueRef,
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


def _source_memory_emit(rule) -> EmitDescriptorOp:
    return next(
        emit
        for emit in reversed(rule.emit)
        if isinstance(emit, EmitDescriptorOp) and emit.source_memory is not None
    )


def _fused_address_cases(width_bits: int):
    return (
        ("immediate", -width_bits, width_bits - width_bits // 8, 0, 0, False),
        ("register", _I32_MIN, _I32_MAX, 0, 0, False),
        ("register", 0, 0, None, 1, True),
        ("register", -64, 63, None, 1, True),
        ("register", _I32_MIN, _I32_MAX, None, 1, True),
    )


def _fused_rule_identity(rule):
    source_memory = _source_memory_emit(rule).source_memory
    assert source_memory is not None
    assert len(rule.source_nodes) == 1
    source_node = rule.source_nodes[0]
    return (
        rule.source_op.name,
        source_node.source_op.name,
        source_node.relation,
        rule.report_key,
        rule.descriptor.key,
        source_memory.operation,
        source_memory.root_kind,
        source_memory.memory_spaces,
        source_memory.element_byte_count,
        source_memory.vector_lane_count,
        source_memory.minimum_alignment,
        source_memory.static_byte_offset_minimum,
        source_memory.static_byte_offset_maximum,
        source_memory.dynamic_term_count,
        source_memory.dynamic_term_count_minimum,
        source_memory.allow_dynamic_stride_values,
    )


def test_fused_packet_memory_rules_cover_the_native_shape_matrix() -> None:
    logical_cases = [
        *(
            (
                vector.vector_load,
                vector.vector_extf,
                SourceNodeRelation.ADJACENT_UNIQUE_USER,
                SourceMemoryOperation.LOAD,
                f"native_memory_load_bf16x{lane_count}_to_f32x{lane_count}",
                (f"amd.xdna.aie2p.load.convert.bf16x{lane_count}.to.f32x{lane_count}"),
                2,
                lane_count,
                width_bits,
            )
            for lane_count, width_bits in ((16, 256), (32, 512))
        ),
        *(
            (
                vector.vector_load,
                source_op,
                SourceNodeRelation.ADJACENT_UNIQUE_USER,
                SourceMemoryOperation.LOAD,
                (
                    f"native_memory_load_{signedness}_"
                    f"{widen_case.input_element}x{widen_case.lane_count}_to_"
                    f"{widen_case.result_element}x{widen_case.lane_count}"
                ),
                (
                    f"amd.xdna.aie2p.load.widen.{widen_case.physical_shape}."
                    f"{signedness}.configured"
                ),
                int(widen_case.input_element[1:]) // 8,
                widen_case.lane_count,
                widen_case.memory_width_bits,
            )
            for source_op, signedness in (
                (vector.vector_extui, "unsigned"),
                (vector.vector_extsi, "signed"),
            )
            for widen_case in INTEGER_WIDEN_CASES
        ),
        *(
            (
                vector.vector_store,
                vector.vector_fptrunc,
                SourceNodeRelation.ADJACENT_DEFINITION,
                SourceMemoryOperation.STORE,
                f"native_memory_store_f32x{lane_count}_to_bf16x{lane_count}",
                (f"amd.xdna.aie2p.store.convert.f32x{lane_count}.to.bf16x{lane_count}"),
                2,
                lane_count,
                width_bits,
            )
            for lane_count, width_bits in ((16, 256), (32, 512))
        ),
        *(
            (
                vector.vector_store,
                pack_case.source_op,
                SourceNodeRelation.ADJACENT_DEFINITION,
                SourceMemoryOperation.STORE,
                f"native_memory_store_{pack_case.report_key}",
                (
                    f"amd.xdna.aie2p.store.pack.{pack_case.physical_width}."
                    "trunc.configured"
                ),
                1,
                pack_case.result_lanes,
                pack_case.memory_width_bits,
            )
            for pack_case in INTEGER_PACK_CASES
        ),
    ]
    expected_identities = set()
    for root_kind, memory_spaces in _MEMORY_ROOTS:
        for volatile in (True, False):
            volatile_suffix = ".volatile" if volatile else ""
            for (
                source_op,
                related_op,
                relation,
                operation,
                report_key,
                descriptor_prefix,
                element_byte_count,
                lane_count,
                width_bits,
            ) in logical_cases:
                for (
                    address_family,
                    static_minimum,
                    static_maximum,
                    dynamic_count,
                    dynamic_minimum,
                    dynamic_strides,
                ) in _fused_address_cases(width_bits):
                    expected_identities.add(
                        (
                            source_op.name,
                            related_op.name,
                            relation,
                            report_key,
                            (
                                f"{descriptor_prefix}.indexed.{address_family}"
                                f"{volatile_suffix}"
                            ),
                            operation,
                            root_kind,
                            memory_spaces,
                            element_byte_count,
                            lane_count,
                            width_bits // 8,
                            static_minimum,
                            static_maximum,
                            dynamic_count,
                            dynamic_minimum,
                            dynamic_strides,
                        )
                    )

    actual_identities = [
        _fused_rule_identity(rule) for rule in AIE2P_PACKET_MEMORY_RULES
    ]
    assert len(actual_identities) == len(set(actual_identities))
    assert set(actual_identities) == expected_identities


def test_fused_packet_memory_rules_preserve_graph_and_state_contracts() -> None:
    for rule in AIE2P_PACKET_MEMORY_RULES:
        assert rule.priority == 1
        source_node = rule.source_nodes[0]
        memory_emit = _source_memory_emit(rule)
        assert memory_emit.descriptor == rule.descriptor
        assert (
            memory_emit.source_memory.address_layout
            is SourceMemoryAddressLayout.COMPACT_ROW_MAJOR
        )

        if rule.source_op is vector.vector_load:
            assert source_node.relation is SourceNodeRelation.ADJACENT_UNIQUE_USER
            assert source_node.parent_value == ValueRef.result("result")
            assert source_node.node_value == ValueRef.operand("input")
        else:
            assert rule.source_op is vector.vector_store
            assert source_node.relation is SourceNodeRelation.ADJACENT_DEFINITION
            assert source_node.parent_value == ValueRef.operand("value")
            assert source_node.node_value == ValueRef.result("result")

        descriptor_keys = [
            emit.descriptor.key
            for emit in rule.emit
            if isinstance(emit, EmitDescriptorOp)
        ]
        memory_index = descriptor_keys.index(rule.descriptor.key)
        if ".load.widen." in rule.descriptor.key:
            assert descriptor_keys[memory_index - 3 : memory_index] == [
                "amd.xdna.aie2p.constant.i32.shift",
                "amd.xdna.aie2p.state.saturation.immediate",
                "amd.xdna.aie2p.state.ups-mode.immediate",
            ]
        elif ".store.convert." in rule.descriptor.key:
            assert descriptor_keys[memory_index - 1] == (
                "amd.xdna.aie2p.state.rounding.immediate"
            )
        elif ".store.pack." in rule.descriptor.key:
            assert descriptor_keys[memory_index - 2 : memory_index] == [
                "amd.xdna.aie2p.state.saturation.immediate",
                "amd.xdna.aie2p.state.pack-size.immediate",
            ]
