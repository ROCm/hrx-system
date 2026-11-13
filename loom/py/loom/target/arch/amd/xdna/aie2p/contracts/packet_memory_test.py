# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for AIE2P packet-converting memory selection."""

from collections import Counter, defaultdict

import pytest

from loom.dialect.encoding import defs as encoding
from loom.dialect.vector import defs as vector
from loom.dsl import EncodingOperandSummaryDef
from loom.target.arch.amd.xdna.aie2p.contracts.bfp import AIE2P_BFP_ENCODE_RULES
from loom.target.arch.amd.xdna.aie2p.contracts.packet_conversion import (
    I4_UNPACK_SOURCE_LANE_COUNTS,
    INTEGER_PACK_INSTRUCTIONS,
    INTEGER_WIDEN_INSTRUCTIONS,
)
from loom.target.arch.amd.xdna.aie2p.contracts.packet_memory import (
    AIE2P_BFP_GROUP_MEMORY_RULES,
    AIE2P_PACKET_MEMORY_RULES,
)
from loom.target.contracts import (
    DescriptorEmitForm,
    EmitDescriptorOp,
    EmitRegisterConcat,
    EmitRegisterSlice,
    Guard,
    SourceMemoryAddressLayout,
    SourceMemoryDynamicIndexSource,
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

_FUSED_MEMORY_RULES = tuple(
    rule for rule in AIE2P_PACKET_MEMORY_RULES if rule.source_nodes
)

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


def _fused_address_cases(width_bits: int, *, maximum_chunk_offset: int = 0):
    no_index = SourceMemoryDynamicIndexSource.NONE
    value_index = SourceMemoryDynamicIndexSource.VALUE
    full_static_maximum = _I32_MAX - maximum_chunk_offset
    return (
        (
            "immediate",
            -width_bits,
            width_bits - width_bits // 8 - maximum_chunk_offset,
            0,
            0,
            no_index,
            0,
            False,
        ),
        ("register", _I32_MIN, full_static_maximum, 0, 0, no_index, 0, False),
        ("register", 0, 0, None, 1, no_index, 0, True),
        ("register", -64, 63, None, 1, no_index, 0, True),
        ("register", -128, 127, 1, 0, value_index, 1, False),
        ("register", _I32_MIN, full_static_maximum, 1, 0, value_index, 1, False),
        ("register", _I32_MIN, full_static_maximum, None, 1, no_index, 0, True),
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
        source_memory.dynamic_index_source,
        source_memory.dynamic_byte_stride,
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
    expected_identities = Counter()
    for root_kind, memory_spaces in _MEMORY_ROOTS:
        for volatile in (True, False):
            volatile_suffix = ".volatile" if volatile else ""
            # FIFO groups form a pointer before the first fetch. Static
            # modifiers use either compact or full-range native constants.
            for static_minimum, static_maximum, dynamic_count, dynamic_minimum in (
                (0, 0, 0, 0),
                (-1024, 1023, 0, 0),
                (_I32_MIN, _I32_MAX, 0, 0),
                (_I32_MIN, _I32_MAX, None, 1),
            ):
                expected_identities[
                    (
                        vector.vector_load.name,
                        vector.vector_fragment.name,
                        SourceNodeRelation.EXCLUSIVE_USER,
                        "native_memory_load_bfp16ebs8_group",
                        "amd.xdna.aie2p.load.load-fifo.low512" + volatile_suffix,
                        SourceMemoryOperation.LOAD,
                        root_kind,
                        _expected_memory_spaces(
                            SourceMemoryOperation.LOAD, memory_spaces
                        ),
                        1,
                        None,
                        64,
                        static_minimum,
                        static_maximum,
                        dynamic_count,
                        dynamic_minimum,
                        SourceMemoryDynamicIndexSource.NONE,
                        0,
                        dynamic_count is None,
                    )
                ] += 1
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
                    dynamic_index_source,
                    dynamic_byte_stride,
                    dynamic_strides,
                ) in _fused_address_cases(width_bits):
                    identity = (
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
                        dynamic_index_source,
                        dynamic_byte_stride,
                        dynamic_strides,
                    )
                    expected_identities[identity] += (
                        2
                        if related_op in (vector.vector_extf, vector.vector_fptrunc)
                        and lane_count == 16
                        else 1
                    )

            # Both BF16 packets, or all 72 interleaved BFP bytes, must fit.
            for (
                related_op,
                relation,
                report_key,
                descriptor_pattern,
                element_bytes,
                lanes,
                final_offset,
            ) in (
                (
                    vector.vector_encode,
                    SourceNodeRelation.ADJACENT_UNIQUE_USER,
                    "native_memory_load_bf16x64_to_bfp16ebs8",
                    "amd.xdna.aie2p.load.convert.bf16x32.to.f32x32.indexed.{address}",
                    2,
                    64,
                    64,
                ),
                (
                    vector.vector_fragment,
                    SourceNodeRelation.EXCLUSIVE_USER,
                    "native_memory_load_packed_bfp16ebs8",
                    "amd.xdna.aie2p.load.a.i8x64.indexed.{address}",
                    1,
                    72,
                    68,
                ),
            ):
                for (
                    address_family,
                    static_minimum,
                    static_maximum,
                    dynamic_count,
                    dynamic_minimum,
                    dynamic_index_source,
                    dynamic_byte_stride,
                    dynamic_strides,
                ) in _fused_address_cases(512, maximum_chunk_offset=final_offset):
                    expected_identities[
                        (
                            vector.vector_load.name,
                            related_op.name,
                            relation,
                            report_key,
                            descriptor_pattern.format(
                                address=address_family,
                            )
                            + volatile_suffix,
                            SourceMemoryOperation.LOAD,
                            root_kind,
                            _expected_memory_spaces(
                                SourceMemoryOperation.LOAD, memory_spaces
                            ),
                            element_bytes,
                            lanes,
                            64,
                            static_minimum,
                            static_maximum,
                            dynamic_count,
                            dynamic_minimum,
                            dynamic_index_source,
                            dynamic_byte_stride,
                            dynamic_strides,
                        )
                    ] += 1

    actual_identities = Counter(
        _fused_rule_identity(rule) for rule in _FUSED_MEMORY_RULES
    )
    assert actual_identities == expected_identities


def test_fused_packet_memory_rules_preserve_graph_and_state_contracts() -> None:
    for rule in _FUSED_MEMORY_RULES:
        assert rule.priority == 1
        source_node = rule.source_nodes[0]
        memory_emit = _source_memory_emit(rule)
        assert any(
            isinstance(emit, EmitDescriptorOp) and emit.descriptor == rule.descriptor
            for emit in rule.emit
        )
        assert memory_emit.source_memory.address_layout is SourceMemoryAddressLayout.ANY

        if rule.source_op is vector.vector_load:
            assert source_node.relation is (
                SourceNodeRelation.EXCLUSIVE_USER
                if source_node.source_op is vector.vector_fragment
                else SourceNodeRelation.ADJACENT_UNIQUE_USER
            )
            assert source_node.parent_value == ValueRef.result("result")
            assert source_node.node_value == ValueRef.operand(
                "data"
                if source_node.source_op is vector.vector_fragment
                else "source"
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


def test_bfp_stores_cover_addresses_and_preserve_exact_record_extents() -> None:
    cases = set()
    for rule in AIE2P_PACKET_MEMORY_RULES:
        if rule.report_key != "native_memory_store_bfp16ebs8":
            continue
        memory = _source_memory_emit(rule).source_memory
        assert memory.operation is SourceMemoryOperation.STORE
        assert memory.element_byte_count == 1
        assert memory.vector_lane_count is None
        assert memory.minimum_alignment == 8
        assert Guard.low_value_register_class("value", "aie2p.mexa") in rule.guards
        assert Guard.value_static_dim0_multiple("value", 72) in rule.guards
        emits = [emit for emit in rule.emit if isinstance(emit, EmitDescriptorOp)]
        assert not any(".load." in emit.descriptor.key for emit in emits)
        pointer_move, position, push, flush = emits[-4:]
        assert pointer_move.descriptor.key == "amd.xdna.aie2p.move.store-fifo.pointer"
        assert position.immediates == {"i": 0}
        assert push.descriptor.key.removesuffix(".volatile") == (
            "amd.xdna.aie2p.store.bfp16ebs8.fifo.push.empty"
        )
        assert "fifo_reg" not in push.operands
        assert push.operands["src"] == ValueRef.operand("value")
        assert push.operands["ptr"] == pointer_move.results["dst"]
        assert push.operands["avail"] == position.results["dst"]
        assert flush.descriptor.key.removesuffix(".volatile") == (
            "amd.xdna.aie2p.store.fifo.flush.512"
        )
        assert flush.operands["fifo_reg"] == push.results["fifo_reg_out"]
        assert flush.operands["ptr"] == push.results["ptr_out"]
        assert flush.operands["avail"] == push.results["avail_out"]
        assert rule.sequence_carried == (
            push.results["ptr_out"].field,
            push.results["avail_out"].field,
        )
        assert push.form is flush.form is DescriptorEmitForm.PER_LANE_SEQUENCE
        if memory.dynamic_term_count_minimum:
            address = "dynamic"
        elif memory.static_byte_offset_maximum:
            address = emits[0].descriptor.key.rsplit(".", 1)[1]
        else:
            address = "zero"
        cases.add((address, push.descriptor.key.endswith(".volatile")))
    assert cases == {
        (address, volatile)
        for address in ("zero", "mova", "movxm", "dynamic")
        for volatile in (False, True)
    }


def test_bfp_group_address_setup_preserves_the_fifo_pointer_domain() -> None:
    forms = set()
    for rule in AIE2P_BFP_GROUP_MEMORY_RULES:
        memory = _source_memory_emit(rule).source_memory
        pointer_move = rule.emit[0]
        assert pointer_move.descriptor.key == "amd.xdna.aie2p.move.load-fifo.pointer"
        pointer = pointer_move.results["dst"]
        setup = rule.emit[1]
        if ".constant.modifier." in setup.descriptor.key:
            (immediate,) = setup.descriptor.immediates
            assert memory.static_byte_offset_minimum == immediate.signed_min
            assert memory.static_byte_offset_maximum == immediate.unsigned_max
            assert memory.dynamic_term_count == 0
            forms.add(setup.descriptor.key.rsplit(".", 1)[1])
        elif setup.descriptor.key == "amd.xdna.aie2p.move.er.to.em":
            assert memory.dynamic_term_count_minimum == 1
            forms.add("dynamic")
        else:
            assert memory.static_byte_offset_minimum == 0
            assert memory.static_byte_offset_maximum == 0
            forms.add("zero")
        if memory.static_byte_offset_maximum != 0:
            update = rule.emit[2]
            assert update.operands["ptr"] == pointer
            assert update.result_types == {"ptr_out": pointer}
            pointer = update.results["ptr_out"]
        preload = next(
            emit
            for emit in rule.emit
            if isinstance(emit, EmitDescriptorOp)
            and ".load.load-fifo.low512" in emit.descriptor.key
        )
        first_pop = next(
            emit
            for emit in rule.emit
            if isinstance(emit, EmitDescriptorOp)
            and ".fifo.pop.buffered64" in emit.descriptor.key
        )
        assert preload.operands["ptr"] == pointer
        assert first_pop.operands["ptr"] == pointer
    assert forms == {"zero", "mova", "movxm", "dynamic"}


@pytest.mark.parametrize("record_count", [8, 16, 24, 64, 128])
def test_bfp_group_consumes_each_byte_once_without_reading_padding(
    record_count: int,
) -> None:
    for rule in AIE2P_BFP_GROUP_MEMORY_RULES:
        memory = _source_memory_emit(rule).source_memory
        if memory.static_byte_offset_maximum != 0:
            continue
        # Unique byte identities make duplication, gaps, and record-order
        # mistakes observable independently of exponent/mantissa arithmetic.
        values = {ValueRef.source_memory_root(): 0}
        reads = []

        def fetch(start, *, reads=reads):
            reads.append(start)
            return tuple(range(start, start + 64))

        sequence_start = next(
            index
            for index, emit in enumerate(rule.emit)
            if emit.form is DescriptorEmitForm.PER_LANE_SEQUENCE
        )
        assert memory.vector_lane_count is None
        group_count = record_count // 8
        program = list(rule.emit[:sequence_start])
        for group in range(group_count):
            program.extend(
                emit
                for emit in rule.emit[sequence_start:]
                if not (
                    isinstance(emit, EmitDescriptorOp)
                    and emit.skip_last_lane
                    and group + 1 == group_count
                )
            )
        result = ()
        for emit in program:
            if isinstance(emit, EmitRegisterConcat):
                result += tuple(
                    value for source in emit.sources for value in values[source]
                )
                continue
            assert isinstance(emit, EmitDescriptorOp)
            key = emit.descriptor.key.removesuffix(".volatile")
            operands = {name: values[ref] for name, ref in emit.operands.items()}
            if key.endswith("move.load-fifo.pointer"):
                results = {"dst": operands["src"]}
            elif key.endswith("load.load-fifo.low512"):
                results = {"dst": fetch(operands["ptr"] + emit.immediates["imm"])}
            elif key.endswith("constant.i32.fifo-position"):
                results = {"dst": emit.immediates["i"]}
            elif key.endswith("fifo.fill.512"):
                pointer, fifo, position = (
                    operands[name] for name in ("ptr", "fifo_reg", "pos")
                )
                assert fifo == () and position == 0
                results = {
                    "ptr_out": pointer,
                    "fifo_reg_out": fetch(pointer),
                    "pos_out": 64,
                }
            else:
                assert ".bfp16ebs8.fifo.pop" in key
                pointer, fifo, position = (
                    operands[name] for name in ("ptr", "fifo_reg", "pos")
                )
                assert len(fifo) == position
                assert 8 <= position <= 64
                assert key.endswith(".buffered64") == (position == 64)
                # A native pop fetches at pointer + position, consumes 72
                # bytes, and carries the remaining bytes into the next pop.
                available = fifo + fetch(pointer + position)
                results = {
                    "dst": available[:72],
                    "ptr_out": pointer + 72,
                    "fifo_reg_out": available[72:],
                    "pos_out": position - 8,
                }
            for name, ref in emit.results.items():
                values[ref] = results[name]
        byte_count = record_count * 72
        assert result == tuple(range(byte_count))
        assert reads == list(range(0, byte_count, 64))
        assert results["ptr_out"] == byte_count
        assert results["pos_out"] == 0
        assert results["fifo_reg_out"] == ()


def test_fused_float_memory_rules_publish_subnormal_policy() -> None:
    float_rules = [
        rule
        for rule in _FUSED_MEMORY_RULES
        if rule.source_nodes[0].source_op is vector.vector_extf
        or rule.source_nodes[0].source_op is vector.vector_fptrunc
    ]
    assert len(float_rules) == (
        len(_MEMORY_ROOTS) * len(_fused_address_cases(256)) * 3 * 2 * 2
    )
    representation_variants = defaultdict(set)
    for rule in float_rules:
        source_node = rule.source_nodes[0]
        source_memory = _source_memory_emit(rule).source_memory
        assert source_memory is not None
        lane_count = source_memory.vector_lane_count
        if source_node.source_op is vector.vector_extf:
            base_guards = (
                Guard.value_type("input", Vector("bf16", lanes=lane_count)),
                Guard.value_type("result", Vector("f32", lanes=lane_count)),
                Guard.value_not_subnormal_or_instance_flags_has_all(
                    "input", "subnormal", "daz"
                ),
            )
            register_field = "result"
        else:
            assert source_node.source_op is vector.vector_fptrunc
            base_guards = (
                Guard.value_type("input", Vector("f32", lanes=lane_count)),
                Guard.value_type("result", Vector("bf16", lanes=lane_count)),
                Guard.value_not_subnormal_or_instance_flags_has_all(
                    "input", "subnormal", "daz"
                ),
            )
            register_field = "input"
        if lane_count == 16:
            assert source_node.guards[:3] == base_guards
            assert len(source_node.guards) == 4
            register_guard = source_node.guards[3]
            assert register_guard == Guard.low_value_register_class(
                register_field, register_guard.register_class
            )
            representation_variants[_fused_rule_identity(rule)].add(
                register_guard.register_class
            )
        else:
            assert source_node.guards == base_guards
    assert all(
        variants == {"aie2p.vec256", "aie2p.ebmll"}
        for variants in representation_variants.values()
    )


def test_bfp_load_rules_preserve_two_native_chunks() -> None:
    rules = [
        rule
        for rule in _FUSED_MEMORY_RULES
        if rule.source_nodes[0].source_op is vector.vector_encode
    ]
    assert len(rules) == len(_MEMORY_ROOTS) * 2 * 7
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
            emit.operands["ptr"] == ValueRef.source_memory_root()
            for emit in memory_emits
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
        *AIE2P_BFP_ENCODE_RULES,
        *(
            rule
            for rule in _FUSED_MEMORY_RULES
            if rule.source_nodes[0].source_op is vector.vector_encode
        ),
    ]
    for rule in rules:
        if rule.report_key == "encode.f32.bfp16ebs8":
            assert rule.guards == (
                Guard.value_type("source", Vector("f32", lanes=64)),
                *expected_guards[1:],
            )
            rounding, encoded = rule.emit
            assert rounding.descriptor.key == "amd.xdna.aie2p.state.rounding.immediate"
            assert rounding.immediates == {"i": 12}
            assert encoded.descriptor.key == "amd.xdna.aie2p.convert.f32x64.bfp16ebs8"
            assert encoded.operands == {"src": ValueRef.operand("source")}
            assert encoded.results == {"dst": ValueRef.result("result")}
            continue
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
