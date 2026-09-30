# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for AIE2P packet-converting memory selection."""

from loom.dialect.encoding import defs as encoding
from loom.dialect.vector import defs as vector
from loom.dsl import EncodingOperandSummaryDef
from loom.target.arch.amd.xdna.aie2p.contracts.bfp import AIE2P_BFP_ENCODE_RULE
from loom.target.arch.amd.xdna.aie2p.contracts.packet_conversion import (
    I4_UNPACK_SOURCE_LANE_COUNTS,
    INTEGER_PACK_INSTRUCTIONS,
    INTEGER_WIDEN_INSTRUCTIONS,
)
from loom.target.arch.amd.xdna.aie2p.contracts.packet_memory import (
    AIE2P_PACKET_MEMORY_RULES,
)
from loom.target.contracts import (
    EmitDescriptorOp,
    EmitRegisterConcat,
    EmitRegisterSlice,
    Guard,
    SourceMemoryAddressLayout,
    SourceMemoryOperation,
    SourceMemoryProject,
    SourceMemoryProjectKind,
    SourceMemoryRootKind,
    SourceNodeRelation,
    ValueRef,
    Vector,
)

_I32_MIN = -(2**31)
_I32_MAX = (2**31) - 1

_MEMORY_ROOTS = (
    (
        SourceMemoryRootKind.ANY,
        ("unknown", "generic", "private", "workgroup"),
    ),
)


def _expected_memory_spaces(
    operation: SourceMemoryOperation, memory_spaces: tuple[str, ...]
) -> tuple[str, ...]:
    return (
        (*memory_spaces, "constant")
        if operation is SourceMemoryOperation.LOAD
        else memory_spaces
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
                source_op,
                SourceNodeRelation.ADJACENT_UNIQUE_USER,
                SourceMemoryOperation.LOAD,
                (
                    f"native_memory_load_{signedness}_i4x"
                    f"{source_lane_count * 2}_to_i8x{source_lane_count * 2}"
                ),
                (
                    f"amd.xdna.aie2p.load.unpack.{source_kind}4x"
                    f"{source_lane_count * 2}.to.{source_kind}8x"
                    f"{source_lane_count * 2}.configured"
                ),
                1,
                source_lane_count,
                source_lane_count * 8,
            )
            for source_op, source_kind, signedness in (
                (vector.vector_bitunpacku, "u", "unsigned"),
                (vector.vector_bitunpacks, "s", "signed"),
            )
            for source_lane_count in I4_UNPACK_SOURCE_LANE_COUNTS
        ),
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
                    f"{instruction.input_element}x{instruction.native_lane_count}_to_"
                    f"{instruction.result_element}x{instruction.native_lane_count}"
                ),
                (
                    f"amd.xdna.aie2p.load.widen.{instruction.physical_shape}."
                    f"{signedness}.configured"
                ),
                int(instruction.input_element[1:]) // 8,
                instruction.native_lane_count,
                instruction.memory_width_bits,
            )
            for source_op, signedness in (
                (vector.vector_extui, "unsigned"),
                (vector.vector_extsi, "signed"),
            )
            for instruction in INTEGER_WIDEN_INSTRUCTIONS
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
                pack_instruction.source_op,
                SourceNodeRelation.ADJACENT_DEFINITION,
                SourceMemoryOperation.STORE,
                f"native_memory_store_{pack_instruction.report_key}",
                (
                    f"amd.xdna.aie2p.store.pack.{pack_instruction.physical_width}."
                    "trunc.configured"
                ),
                int(pack_instruction.result_element[1:]) // 8,
                pack_instruction.result_lanes,
                pack_instruction.memory_width_bits,
            )
            for pack_instruction in INTEGER_PACK_INSTRUCTIONS
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
                            _expected_memory_spaces(operation, memory_spaces),
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

            # Both 64-byte packets must fit the selected addressing form.
            for (
                address_family,
                static_minimum,
                static_maximum,
                dynamic_count,
                dynamic_minimum,
                dynamic_strides,
            ) in (
                ("immediate", -512, 384, 0, 0, False),
                ("register", _I32_MIN, _I32_MAX - 64, 0, 0, False),
                ("register", 0, 0, None, 1, True),
                ("register", -64, 63, None, 1, True),
                ("register", _I32_MIN, _I32_MAX - 64, None, 1, True),
            ):
                expected_identities.add(
                    (
                        vector.vector_load.name,
                        vector.vector_encode.name,
                        SourceNodeRelation.ADJACENT_UNIQUE_USER,
                        "native_memory_load_bf16x64_to_bfp16ebs8",
                        (
                            "amd.xdna.aie2p.load.convert.bf16x32.to.f32x32."
                            f"indexed.{address_family}{volatile_suffix}"
                        ),
                        SourceMemoryOperation.LOAD,
                        root_kind,
                        _expected_memory_spaces(
                            SourceMemoryOperation.LOAD, memory_spaces
                        ),
                        2,
                        64,
                        64,
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
        assert memory_emit.source_memory.address_layout is SourceMemoryAddressLayout.ANY

        if rule.source_op is vector.vector_load:
            assert source_node.relation is SourceNodeRelation.ADJACENT_UNIQUE_USER
            assert source_node.parent_value == ValueRef.result("result")
            assert source_node.node_value == ValueRef.operand(
                "source"
                if source_node.source_op
                in (
                    vector.vector_bitunpacku,
                    vector.vector_bitunpacks,
                    vector.vector_encode,
                )
                else "input"
            )
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
        if ".load.unpack." in rule.descriptor.key:
            assert descriptor_keys[memory_index - 1] == (
                "amd.xdna.aie2p.state.unpack-size.immediate"
            )
        elif ".load.widen." in rule.descriptor.key:
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


def test_bfp_load_rules_preserve_two_native_chunks() -> None:
    rules = [
        rule
        for rule in AIE2P_PACKET_MEMORY_RULES
        if rule.source_nodes[0].source_op is vector.vector_encode
    ]
    assert len(rules) == len(_MEMORY_ROOTS) * 2 * 5
    for rule in rules:
        memory_emits = [
            emit
            for emit in rule.emit
            if isinstance(emit, EmitDescriptorOp) and emit.descriptor == rule.descriptor
        ]
        assert len(memory_emits) == 2
        source_memory = memory_emits[0].source_memory
        assert source_memory is not None
        assert memory_emits[1].source_memory == source_memory
        assert source_memory.operation is SourceMemoryOperation.LOAD
        assert source_memory.element_byte_count == 2
        assert source_memory.vector_lane_count == 64
        assert source_memory.vector_lane_byte_stride == 2
        assert source_memory.minimum_alignment == 64
        assert all(
            emit.operands["ptr"] == ValueRef.operand("view") for emit in memory_emits
        )
        assert not any(isinstance(emit, EmitRegisterSlice) for emit in rule.emit)

        concat = rule.emit[-3]
        assert isinstance(concat, EmitRegisterConcat)
        assert concat.sources == tuple(emit.results["op"] for emit in memory_emits)
        assert concat.sources[0] != concat.sources[1]

        static_projects = [
            project
            for emit in rule.emit
            if isinstance(emit, EmitDescriptorOp) and isinstance(emit.immediates, dict)
            for project in emit.immediates.values()
            if isinstance(project, SourceMemoryProject)
        ]
        has_folded_static_base = (
            source_memory.dynamic_term_count_minimum == 1
            and source_memory.static_byte_offset_minimum in (0, _I32_MIN)
        )
        assert tuple(project.kind for project in static_projects) == (
            (SourceMemoryProjectKind.STATIC_BYTE_OFFSET_PLUS_LITERAL,)
            if has_folded_static_base
            else (
                SourceMemoryProjectKind.STATIC_BYTE_OFFSET,
                SourceMemoryProjectKind.STATIC_BYTE_OFFSET_PLUS_LITERAL,
            )
        )
        assert tuple(project.literal_i64 for project in static_projects) == (
            (64,) if has_folded_static_base else (0, 64)
        )


def test_bfp_encodes_preserve_schema_and_ordered_conversion_tail() -> None:
    expected_guards = (
        Guard.value_type("source", Vector("bf16", lanes=64)),
        Guard.value_type("result", Vector("i8", lanes=72)),
        Guard.operand_segment_count("auxiliary", 0),
        Guard.value_storage_operand_schema(
            "schema",
            EncodingOperandSummaryDef(
                element_format=encoding.enum_fact(encoding.NumericFormat, "bfp16ebs8"),
                payload_packing=encoding.enum_fact(
                    encoding.PayloadPacking, "target_fragment"
                ),
                rounding_policy=encoding.enum_fact(
                    encoding.RoundingPolicy, "flush_subnormal"
                ),
                payload_register_count=18,
                payload_element_count=64,
            ),
        ),
    )
    rules = [
        AIE2P_BFP_ENCODE_RULE,
        *(
            rule
            for rule in AIE2P_PACKET_MEMORY_RULES
            if rule.source_nodes[0].source_op is vector.vector_encode
        ),
    ]
    for rule in rules:
        if rule.source_op is vector.vector_encode:
            assert rule.guards == expected_guards
            result = ValueRef.result("result")
            source_slices = [
                emit for emit in rule.emit if isinstance(emit, EmitRegisterSlice)
            ]
            assert [(emit.unit_offset, emit.unit_count) for emit in source_slices] == [
                (0, 2),
                (2, 2),
            ]
            assert all(
                emit.source == ValueRef.operand("source") for emit in source_slices
            )
            conversions = [
                emit
                for emit in rule.emit
                if isinstance(emit, EmitDescriptorOp)
                and emit.descriptor.key == "amd.xdna.aie2p.convert.bf16x32.to.f32x32"
            ]
            assert [emit.operands["src"] for emit in conversions] == [
                emit.result for emit in source_slices
            ]
            expected_halves = tuple(emit.results["dst"] for emit in conversions)
        else:
            source_node = rule.source_nodes[0]
            assert source_node.guards == expected_guards
            result = ValueRef.result("result", source_node=source_node.name)
            expected_halves = tuple(
                emit.results["op"]
                for emit in rule.emit
                if isinstance(emit, EmitDescriptorOp)
                and emit.descriptor == rule.descriptor
            )

        concat, rounding, encoded = rule.emit[-3:]
        assert isinstance(concat, EmitRegisterConcat)
        assert concat.sources == expected_halves
        assert concat.result_type == Vector("f32", lanes=64)
        assert isinstance(rounding, EmitDescriptorOp)
        assert rounding.descriptor.key == "amd.xdna.aie2p.state.rounding.immediate"
        assert rounding.immediates == {"i": 12}
        assert isinstance(encoded, EmitDescriptorOp)
        assert encoded.descriptor.key == "amd.xdna.aie2p.convert.f32x64.bfp16ebs8"
        assert encoded.operands == {"src": concat.result}
        assert encoded.results == {"dst": result}
