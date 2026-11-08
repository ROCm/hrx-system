# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for AMD XDNA AIE2P scalar and vector memory selection."""

from dataclasses import replace

from loom.dialect.buffer import defs as buffer
from loom.dialect.vector import defs as vector
from loom.dialect.view import defs as view
from loom.target.arch.amd.xdna.aie2p.contracts.memory import (
    AIE2P_MEMORY_RULES,
    AIE2P_VIEW_CARRIER_RULES,
    AIE2P_VIEW_ELEMENT_TYPES,
)
from loom.target.arch.amd.xdna.aie2p.contracts.packet_memory import (
    AIE2P_INDEXED_PACKET_MEMORY_RULES,
)
from loom.target.contracts import (
    DescriptorResultType,
    EmitDescriptorOp,
    EmitRegisterConcat,
    EmitRegisterCopy,
    EmitRegisterSlice,
    Guard,
    GuardKind,
    SourceMemoryAddressLayout,
    SourceMemoryDynamicIndexSource,
    SourceMemoryOperation,
    SourceMemoryProject,
    SourceMemoryProjectKind,
    SourceMemoryRootKind,
    SourceValueKind,
    ValueRef,
    View,
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


def _source_memory_emit(rule):
    return next(
        emit
        for emit in reversed(rule.emit)
        if getattr(emit, "source_memory", None) is not None
    )


def _primary_descriptor_emit_count(rule) -> int:
    return sum(
        isinstance(emit, EmitDescriptorOp) and emit.descriptor == rule.descriptor
        for emit in rule.emit
    )


def _rules_for(
    root_kind,
    *source_ops,
    accumulator: bool = False,
    bytewise_scalar: bool | None = None,
    volatile: bool = False,
    scalarized_vector_load: bool | None = None,
    split_vector_load: bool | None = None,
    wide_vector: bool | None = None,
    pair_scalar: bool | None = None,
):
    return [
        rule
        for rule in AIE2P_MEMORY_RULES
        if rule.source_op in source_ops
        and (".accumulator." in rule.descriptor.key) is accumulator
        and _source_memory_emit(rule).source_memory.root_kind is root_kind
        and rule.descriptor.key.endswith(".volatile") is volatile
        and (
            bytewise_scalar is None
            or (
                rule.source_op in (view.view_load, view.view_store)
                and ".scalar.i8.indexed." in rule.descriptor.key
                and _source_memory_emit(rule).source_memory.element_byte_count > 1
            )
            is bytewise_scalar
        )
        and (
            scalarized_vector_load is None
            or (
                rule.source_op is vector.vector_load
                and ".load.scalar." in rule.descriptor.key
            )
            is scalarized_vector_load
        )
        and (
            split_vector_load is None
            or (
                rule.source_op is vector.vector_load
                and _primary_descriptor_emit_count(rule) == 2
            )
            is split_vector_load
        )
        and (
            wide_vector is None
            or (
                rule.source_op in (vector.vector_load, vector.vector_store)
                and _source_memory_emit(rule).source_memory.element_byte_count
                * _source_memory_emit(rule).source_memory.vector_lane_count
                == 128
            )
            is wide_vector
        )
        and (
            pair_scalar is None
            or (
                rule.source_op in (view.view_load, view.view_store)
                and _source_memory_emit(rule).source_memory.element_byte_count == 8
                and ".scalar.i32.indexed." in rule.descriptor.key
            )
            is pair_scalar
        )
    ]


def _assert_address_forms(
    rules,
    *,
    immediate_minimum: int,
    immediate_maximum: int,
) -> None:
    descriptor_emits = [
        [
            emit
            for emit in rule.emit
            if isinstance(emit, EmitDescriptorOp)
            and (
                emit.source_memory is not None
                or emit.descriptor.key == "amd.xdna.aie2p.move.to.address-index"
            )
        ]
        for rule in rules
    ]
    assert [len(emits) for emits in descriptor_emits] == [1, 2, 2, 2, 2, 3, 2]
    assert [
        (
            _source_memory_emit(rule).source_memory.static_byte_offset_minimum,
            _source_memory_emit(rule).source_memory.static_byte_offset_maximum,
            _source_memory_emit(rule).source_memory.dynamic_term_count,
            _source_memory_emit(rule).source_memory.dynamic_term_count_minimum,
            _source_memory_emit(rule).source_memory.allow_dynamic_stride_values,
        )
        for rule in rules
    ] == [
        (immediate_minimum, immediate_maximum, 0, 0, False),
        (_I32_MIN, _I32_MAX, 0, 0, False),
        (0, 0, None, 1, True),
        (-64, 63, None, 1, True),
        (-128, 127, 1, 0, False),
        (_I32_MIN, _I32_MAX, 1, 0, False),
        (_I32_MIN, _I32_MAX, None, 1, True),
    ]
    assert [
        sum(guard.kind is GuardKind.OPERAND_SEGMENT_COUNT for guard in rule.guards)
        for rule in rules
    ] == [0, 0, 0, 0, 0, 0, 0]
    immediate = descriptor_emits[0][0].immediates["imm"]
    assert isinstance(immediate, SourceMemoryProject)
    assert immediate.kind is SourceMemoryProjectKind.STATIC_BYTE_OFFSET
    assert [emits[-2].descriptor.key for emits in descriptor_emits[1:]] == [
        "amd.xdna.aie2p.materialize.static-address-index.i20",
        "amd.xdna.aie2p.move.to.address-index",
        "amd.xdna.aie2p.address.index.add.immediate",
        "amd.xdna.aie2p.address.index.add.immediate",
        "amd.xdna.aie2p.address.index.add.register",
        "amd.xdna.aie2p.move.to.address-index",
    ]
    static_offset_emit = descriptor_emits[1][-2]
    assert static_offset_emit.operands == {}
    assert static_offset_emit.immediates == {"i": immediate}
    assert static_offset_emit.result_types == {"dst": DescriptorResultType()}
    assert descriptor_emits[1][-1].operands["dj"] == (static_offset_emit.results["dst"])
    dynamic_offset_emit = descriptor_emits[2][-2]
    assert dynamic_offset_emit.operands["src"].kind is (
        SourceValueKind.SOURCE_MEMORY_DYNAMIC_BYTE_OFFSET
    )
    complete_offset_emit = descriptor_emits[6][-2]
    assert complete_offset_emit.operands["src"].kind is (
        SourceValueKind.SOURCE_MEMORY_BYTE_OFFSET
    )
    materializer = complete_offset_emit.source_memory_byte_offset_materializer
    assert materializer is not None
    assert materializer.multiply_add is not None
    assert materializer.multiply_add.key == "amd.xdna.aie2p.madd.i32"
    assert materializer.static_bias is not None
    assert materializer.static_bias.key == (
        "amd.xdna.aie2p.materialize.static-byte-offset.i32"
    )


def test_view_carriers_materialize_every_supported_view_as_one_address() -> None:
    assert len(AIE2P_VIEW_CARRIER_RULES) == 2 * 5
    view_type = View(AIE2P_VIEW_ELEMENT_TYPES)
    expected_descriptor_keys = (
        None,
        "amd.xdna.aie2p.address.add.a.immediate",
        "amd.xdna.aie2p.address.add.a.register",
        "amd.xdna.aie2p.address.add.a.register",
        "amd.xdna.aie2p.address.add.a.register",
    )
    expected_shapes = (
        (0, 0, 0, 0, False, False, 0),
        (-512, 448, 0, 0, False, False, 64),
        (-1024, 1023, 0, 0, False, False, 0),
        (_I32_MIN, _I32_MAX, 0, 0, False, False, 0),
        (_I32_MIN, _I32_MAX, None, 1, True, True, 0),
    )
    rule_index = 0
    for source_op in (buffer.buffer_view, view.view_subview):
        rules = AIE2P_VIEW_CARRIER_RULES[rule_index : rule_index + 5]
        rule_index += 5
        assert [rule.source_op for rule in rules] == [source_op] * 5
        assert [
            rule.descriptor.key if rule.descriptor is not None else None
            for rule in rules
        ] == list(expected_descriptor_keys)
        assert [rule.guards for rule in rules] == [
            (Guard.value_type("result", view_type),)
        ] * 5
        constraints = [_source_memory_emit(rule).source_memory for rule in rules]
        assert [
            (
                constraint.static_byte_offset_minimum,
                constraint.static_byte_offset_maximum,
                constraint.dynamic_term_count,
                constraint.dynamic_term_count_minimum,
                constraint.allow_dynamic_stride_values,
                constraint.reuse_view_base,
                constraint.minimum_alignment,
            )
            for constraint in constraints
        ] == list(expected_shapes)
        for constraint in constraints:
            assert constraint.operation is SourceMemoryOperation.VIEW_CARRIER
            assert constraint.root_kind is SourceMemoryRootKind.ANY
            assert constraint.memory_spaces == tuple(
                sorted(
                    (
                        "unknown",
                        "global",
                        "workgroup",
                        "private",
                        "constant",
                        "host",
                        "descriptor",
                        "generic",
                    )
                )
            )
            assert constraint.element_byte_count == 1
            assert constraint.vector_lane_count == 1
            assert constraint.vector_lane_byte_stride == 1
            assert constraint.byte_offset_unsigned_bit_count == 32

        zero, immediate, compact_register, static_register, dynamic_register = rules
        assert isinstance(zero.emit[0], EmitRegisterCopy)
        assert zero.emit[0].source.kind is SourceValueKind.SOURCE_MEMORY_ROOT
        assert zero.emit[0].result == ValueRef.result("result")
        for rule in (immediate, compact_register, static_register, dynamic_register):
            root_copy = rule.emit[0]
            assert isinstance(root_copy, EmitRegisterCopy)
            assert root_copy.source.kind is SourceValueKind.SOURCE_MEMORY_ROOT
            assert root_copy.result.kind is SourceValueKind.TEMPORARY
            assert root_copy.source_memory == _source_memory_emit(rule).source_memory
        assert immediate.emit[1].immediates == {
            "imm": SourceMemoryProject.static_byte_offset()
        }
        for rule, form, minimum, maximum in (
            (compact_register, "mova", -1024, 1023),
            (static_register, "movxm", _I32_MIN, _I32_MAX),
        ):
            assert [
                emit.descriptor.key
                for emit in rule.emit
                if isinstance(emit, EmitDescriptorOp)
            ] == [
                f"amd.xdna.aie2p.constant.modifier.{form}",
                "amd.xdna.aie2p.address.add.a.register",
            ]
            modifier = rule.emit[1]
            assert modifier.immediates == {
                "i": SourceMemoryProject.static_byte_offset()
            }
            (immediate,) = modifier.descriptor.immediates
            assert immediate.signed_min == minimum
            assert immediate.unsigned_max == maximum
            assert rule.emit[-1].result_types == {"ptr_out": rule.emit[0].result}
        assert [
            emit.descriptor.key
            for emit in dynamic_register.emit
            if isinstance(emit, EmitDescriptorOp)
        ] == [
            "amd.xdna.aie2p.move.er.to.em",
            "amd.xdna.aie2p.address.add.a.register",
        ]
        dynamic_offset = dynamic_register.emit[1]
        assert dynamic_offset.operands["src"].kind is (
            SourceValueKind.SOURCE_MEMORY_BYTE_OFFSET
        )
        assert dynamic_offset.source_memory_byte_offset_materializer is not None
        assert static_register.emit[-1].copy_operands == ()
        assert dynamic_register.emit[-1].copy_operands == ()


def test_indexed_memory_families_consume_their_planned_storage_root() -> None:
    for rules in (AIE2P_MEMORY_RULES, AIE2P_INDEXED_PACKET_MEMORY_RULES):
        for rule in rules:
            pointers = {ValueRef.source_memory_root()}
            memory = _source_memory_emit(rule).source_memory
            for emit in rule.emit:
                if (
                    isinstance(emit, EmitRegisterCopy)
                    and emit.source_memory is not None
                ):
                    assert emit.source in pointers
                    assert emit.source_memory == memory
                    pointers.add(emit.result)
                elif isinstance(emit, EmitDescriptorOp) and "ptr" in emit.operands:
                    assert emit.operands["ptr"] in pointers
                    assert emit.source_memory == memory
                    if "ptr_out" in emit.results:
                        pointers.add(emit.results["ptr_out"])


def test_indexed_memory_families_materialize_the_complete_planned_address() -> None:
    no_index = SourceMemoryDynamicIndexSource.NONE
    value_index = SourceMemoryDynamicIndexSource.VALUE
    expected_dynamic_constraints = (
        (0, 0, no_index, 0, False),
        (0, 0, no_index, 0, False),
        (None, 1, no_index, 0, True),
        (None, 1, no_index, 0, True),
        (1, 0, value_index, 1, False),
        (1, 0, value_index, 1, False),
        (None, 1, no_index, 0, True),
    )
    # Both tables include their volatile variants. Raw buffers, fragment stores,
    # scalarized accesses and fused packets use the same addressing contract.
    for rules in (AIE2P_MEMORY_RULES, AIE2P_INDEXED_PACKET_MEMORY_RULES):
        assert len(rules) % len(expected_dynamic_constraints) == 0
        for start in range(0, len(rules), len(expected_dynamic_constraints)):
            family = rules[start : start + len(expected_dynamic_constraints)]
            constraints = [_source_memory_emit(rule).source_memory for rule in family]
            assert [
                (
                    constraint.dynamic_term_count,
                    constraint.dynamic_term_count_minimum,
                    constraint.dynamic_index_source,
                    constraint.dynamic_byte_stride,
                    constraint.allow_dynamic_stride_values,
                )
                for constraint in constraints
            ] == list(expected_dynamic_constraints)
            projects = [
                project
                for emit in family[0].emit
                if isinstance(emit, EmitDescriptorOp)
                for project in dict(emit.immediates).values()
                if isinstance(project, SourceMemoryProject)
                and project.kind
                in (
                    SourceMemoryProjectKind.STATIC_BYTE_OFFSET,
                    SourceMemoryProjectKind.STATIC_BYTE_OFFSET_PLUS_LITERAL,
                )
            ]
            assert projects
            maximum_chunk_offset = max(project.literal_i64 for project in projects)
            assert [
                (
                    constraint.static_byte_offset_minimum,
                    constraint.static_byte_offset_maximum,
                )
                for constraint in constraints[4:6]
            ] == [(-128, 127), (_I32_MIN, _I32_MAX - maximum_chunk_offset)]

            rebases_pointer = any(
                isinstance(emit, EmitRegisterCopy) and emit.source_memory is not None
                for emit in family[6].emit
            )
            complete_offset = next(
                emit
                for emit in family[6].emit
                if isinstance(emit, EmitDescriptorOp)
                and emit.descriptor.key
                == (
                    "amd.xdna.aie2p.move.er.to.em"
                    if rebases_pointer
                    else "amd.xdna.aie2p.move.to.address-index"
                )
            )
            assert complete_offset.operands["src"].kind is (
                SourceValueKind.SOURCE_MEMORY_BYTE_OFFSET
            )
            materializer = complete_offset.source_memory_byte_offset_materializer
            assert materializer is not None
            assert materializer.add.key == "amd.xdna.aie2p.add.no-carry.i32"
            assert materializer.multiply.key == "amd.xdna.aie2p.mul.i32"
            assert materializer.multiply_add is not None
            assert materializer.multiply_add.key == "amd.xdna.aie2p.madd.i32"
            assert [
                (conversion.source_type, conversion.descriptor.key)
                for conversion in materializer.integer_conversions
            ] == [
                (source_type, f"amd.xdna.aie2p.extend.signed.{source_type}")
                for source_type in ("i8", "i16")
            ]

            for address_form, rule in enumerate(family[1:], start=1):
                if rebases_pointer:
                    accesses = [
                        emit
                        for emit in rule.emit
                        if isinstance(emit, EmitDescriptorOp)
                        and emit.descriptor == rule.descriptor
                    ]
                    updates = [
                        emit
                        for emit in rule.emit
                        if isinstance(emit, EmitDescriptorOp)
                        and "ptr_out" in emit.results
                    ]
                    assert len(updates) == 1
                    assert {emit.operands["ptr"] for emit in accesses} == {
                        updates[0].results["ptr_out"]
                    }
                    assert [emit.immediates["imm"] for emit in accesses] == [
                        project.literal_i64 for project in projects
                    ]
                    assert all("dj" not in emit.operands for emit in accesses)
                    continue
                producers = {
                    result: emit
                    for emit in rule.emit
                    if isinstance(emit, EmitDescriptorOp)
                    for result in emit.results.values()
                }
                memory_emits = [
                    emit
                    for emit in rule.emit
                    if isinstance(emit, EmitDescriptorOp)
                    and emit.source_memory is not None
                    and "ptr" in emit.operands
                ]
                for memory_emit, project in zip(memory_emits, projects, strict=True):
                    address = producers[memory_emit.operands["dj"]]
                    if address_form == 1:
                        assert address.descriptor.key == (
                            "amd.xdna.aie2p.materialize.static-address-index.i20"
                        )
                        assert address.operands == {}
                        assert address.immediates == {"i": project}
                        assert address.result_types == {"dst": DescriptorResultType()}
                        continue
                    if address_form in (2, 6) and project.literal_i64 == 0:
                        assert address.descriptor.key == (
                            "amd.xdna.aie2p.move.to.address-index"
                        )
                        continue
                    constraint = memory_emit.source_memory
                    fits_immediate = (
                        constraint.static_byte_offset_minimum + project.literal_i64
                        >= -128
                        and constraint.static_byte_offset_maximum + project.literal_i64
                        <= 127
                    )
                    addressing = "immediate" if fits_immediate else "register"
                    assert address.descriptor.key == (
                        f"amd.xdna.aie2p.address.index.add.{addressing}"
                    )
                    assert address.result_types == {"dst": DescriptorResultType()}
                    assert [
                        alternative.reg_class
                        for operand in address.descriptor.operands
                        if operand.field_name == "dst"
                        for alternative in operand.reg_alts
                    ] == ["aie2p.edj"]
                    assert address.source_memory == constraint
                    assert (
                        address.source_memory_byte_offset_materializer == materializer
                    )
                    assert address.operands["s0"].kind is (
                        SourceValueKind.SOURCE_MEMORY_DYNAMIC_BYTE_OFFSET
                    )
                    if fits_immediate:
                        assert set(address.operands) == {"s0"}
                        assert address.immediates == {"imm": project}
                    else:
                        assert set(address.operands) == {"s0", "s1"}
                        assert not address.immediates
                        static_offset = producers[address.operands["s1"]]
                        assert static_offset.descriptor == materializer.static_bias
                        assert static_offset.immediates == {"i": project}
                        assert static_offset.source_memory == constraint


def test_scalar_memory_rules_cover_every_address_form() -> None:
    expected_shapes = (
        ("i8", 1, -8, 7, ("i1", "i8", "f8E4M3", "f8E5M2")),
        ("i16", 2, -16, 14, ("i16", "f16", "bf16")),
        ("i32", 4, -32, 28, ("i32", "f32", "index", "offset")),
    )
    for root_kind, memory_spaces in _MEMORY_ROOTS:
        rules = _rules_for(
            root_kind,
            view.view_load,
            view.view_store,
            bytewise_scalar=False,
            pair_scalar=False,
        )
        rule_index = 0
        for (
            descriptor_type,
            element_byte_count,
            immediate_minimum,
            immediate_maximum,
            value_types,
        ) in expected_shapes:
            for value_type in value_types:
                for operation, source_op in (
                    (SourceMemoryOperation.LOAD, view.view_load),
                    (SourceMemoryOperation.STORE, view.view_store),
                ):
                    operation_rules = rules[rule_index : rule_index + 7]
                    rule_index += 7
                    descriptor_prefix = (
                        f"amd.xdna.aie2p.{operation.name.lower()}.scalar."
                        f"{descriptor_type}.indexed"
                    )
                    assert [rule.descriptor.key for rule in operation_rules] == [
                        f"{descriptor_prefix}.immediate",
                        *(f"{descriptor_prefix}.register",) * 6,
                    ]
                    _assert_address_forms(
                        operation_rules,
                        immediate_minimum=immediate_minimum,
                        immediate_maximum=immediate_maximum,
                    )
                    for rule in operation_rules:
                        assert rule.source_op is source_op
                        assert rule.guards[0].type_pattern.elements == (value_type,)
                        constraint = _source_memory_emit(rule).source_memory
                        assert constraint is not None
                        assert constraint.operation is operation
                        assert constraint.root_kind is root_kind
                        assert (
                            constraint.address_layout is SourceMemoryAddressLayout.ANY
                        )
                        assert constraint.memory_spaces == _expected_memory_spaces(
                            constraint.operation, memory_spaces
                        )
                        assert constraint.element_byte_count == element_byte_count
                        assert constraint.vector_lane_count == 1
                        assert constraint.vector_lane_byte_stride == element_byte_count
                        assert constraint.minimum_alignment == element_byte_count
        assert rule_index == len(rules)


def test_pair_scalar_memory_rules_use_two_native_32bit_accesses() -> None:
    for root_kind, memory_spaces in _MEMORY_ROOTS:
        rules = _rules_for(
            root_kind,
            view.view_load,
            view.view_store,
            pair_scalar=True,
        )
        for operation_index, operation in enumerate(
            (SourceMemoryOperation.LOAD, SourceMemoryOperation.STORE)
        ):
            operation_rules = rules[operation_index * 7 : operation_index * 7 + 7]
            descriptor_prefix = (
                f"amd.xdna.aie2p.{operation.name.lower()}.scalar.i32.indexed"
            )
            assert [rule.descriptor.key for rule in operation_rules] == [
                f"{descriptor_prefix}.immediate",
                *(f"{descriptor_prefix}.register",) * 6,
            ]
            for rule in operation_rules:
                assert rule.guards[0].type_pattern.elements == ("i64", "f64")
                constraint = _source_memory_emit(rule).source_memory
                assert constraint.operation is operation
                assert constraint.root_kind is root_kind
                assert constraint.memory_spaces == _expected_memory_spaces(
                    constraint.operation, memory_spaces
                )
                assert constraint.element_byte_count == 8
                assert constraint.vector_lane_count == 1
                assert constraint.minimum_alignment == 4
                assert _primary_descriptor_emit_count(rule) == 2
                if operation is SourceMemoryOperation.LOAD:
                    concat = rule.emit[-1]
                    assert isinstance(concat, EmitRegisterConcat)
                    assert tuple(source.field for source in concat.sources) == (
                        "chunk_0",
                        "chunk_1",
                    )
                    assert concat.result.field == "result"
                else:
                    slices = [
                        emit
                        for emit in rule.emit
                        if isinstance(emit, EmitRegisterSlice)
                    ]
                    assert [emit.unit_offset for emit in slices] == [0, 1]

            immediate_projects = [
                emit.immediates["imm"]
                for emit in operation_rules[0].emit
                if isinstance(emit, EmitDescriptorOp)
            ]
            assert tuple(project.kind for project in immediate_projects) == (
                SourceMemoryProjectKind.STATIC_BYTE_OFFSET,
                SourceMemoryProjectKind.STATIC_BYTE_OFFSET_PLUS_LITERAL,
            )
            assert tuple(project.literal_i64 for project in immediate_projects) == (
                0,
                4,
            )


def test_bytewise_scalar_memory_rules_preserve_unknown_alignment() -> None:
    expected_shapes = (
        (2, ("i16", "f16", "bf16")),
        (4, ("i32", "f32", "index", "offset")),
        (8, ("i64", "f64")),
    )
    for root_kind, memory_spaces in _MEMORY_ROOTS:
        rules = _rules_for(
            root_kind,
            view.view_load,
            view.view_store,
            bytewise_scalar=True,
        )
        rule_index = 0
        for element_byte_count, value_types in expected_shapes:
            expected_static_ranges = (
                (-8, 7 - (element_byte_count - 1), 0, 0, False),
                (
                    _I32_MIN,
                    _I32_MAX - (element_byte_count - 1),
                    0,
                    0,
                    False,
                ),
                (0, 0, None, 1, True),
                (-64, 63, None, 1, True),
                (-128, 127, 1, 0, False),
                (
                    _I32_MIN,
                    _I32_MAX - (element_byte_count - 1),
                    1,
                    0,
                    False,
                ),
                (
                    _I32_MIN,
                    _I32_MAX - (element_byte_count - 1),
                    None,
                    1,
                    True,
                ),
            )
            for operation in (
                SourceMemoryOperation.LOAD,
                SourceMemoryOperation.STORE,
            ):
                operation_rules = rules[rule_index : rule_index + 7]
                rule_index += 7
                descriptor_family = operation.name.lower()
                assert [rule.descriptor.key for rule in operation_rules] == [
                    f"amd.xdna.aie2p.{descriptor_family}.scalar.i8.indexed.immediate",
                    *(
                        (
                            f"amd.xdna.aie2p.{descriptor_family}.scalar.i8.indexed.register",
                        )
                        * 6
                    ),
                ]
                for address_index, rule in enumerate(operation_rules):
                    assert rule.guards[0].type_pattern.elements == value_types
                    memory_emits = [
                        emit
                        for emit in rule.emit
                        if isinstance(emit, EmitDescriptorOp)
                        and emit.descriptor == rule.descriptor
                    ]
                    assert len(memory_emits) == element_byte_count
                    for memory_emit in memory_emits:
                        constraint = memory_emit.source_memory
                        assert constraint is not None
                        assert constraint.operation is operation
                        assert constraint.root_kind is root_kind
                        assert (
                            constraint.address_layout is SourceMemoryAddressLayout.ANY
                        )
                        assert constraint.memory_spaces == _expected_memory_spaces(
                            constraint.operation, memory_spaces
                        )
                        assert constraint.element_byte_count == element_byte_count
                        assert constraint.vector_lane_count == 1
                        assert constraint.vector_lane_byte_stride == element_byte_count
                        assert constraint.minimum_alignment == 1
                        assert (
                            constraint.static_byte_offset_minimum,
                            constraint.static_byte_offset_maximum,
                            constraint.dynamic_term_count,
                            constraint.dynamic_term_count_minimum,
                            constraint.allow_dynamic_stride_values,
                        ) == expected_static_ranges[address_index]

                    shift_emits = [
                        emit
                        for emit in rule.emit
                        if isinstance(emit, EmitDescriptorOp)
                        and emit.descriptor.key == "amd.xdna.aie2p.lshl.i32"
                    ]
                    expected_word_count = (element_byte_count + 3) // 4
                    expected_shift_count = element_byte_count - expected_word_count
                    assert len(shift_emits) == expected_shift_count
                    merge_emits = [
                        emit
                        for emit in rule.emit
                        if isinstance(emit, EmitDescriptorOp)
                        and emit.descriptor.key == "amd.xdna.aie2p.or.i32"
                    ]
                    if operation is SourceMemoryOperation.LOAD:
                        assert len(merge_emits) == expected_shift_count
                        if expected_word_count == 1:
                            assert merge_emits[-1].results["d0"].field == "result"
                        else:
                            concat = rule.emit[-1]
                            assert isinstance(concat, EmitRegisterConcat)
                            assert concat.result.field == "result"
                            assert len(concat.sources) == expected_word_count
                    else:
                        assert not merge_emits
                        slices = [
                            emit
                            for emit in rule.emit
                            if isinstance(emit, EmitRegisterSlice)
                        ]
                        assert len(slices) == (
                            expected_word_count if expected_word_count > 1 else 0
                        )

                immediate_offsets = [
                    emit.immediates["imm"].literal_i64
                    for emit in operation_rules[0].emit
                    if isinstance(emit, EmitDescriptorOp)
                    and emit.descriptor == operation_rules[0].descriptor
                ]
                assert immediate_offsets == list(range(element_byte_count))
        assert rule_index == len(rules)


def test_two_lane_16bit_load_rules_preserve_exact_access_bounds() -> None:
    expected_static_ranges = (
        (-16, 12, 0, 0, False),
        (_I32_MIN, _I32_MAX - 2, 0, 0, False),
        (0, 0, None, 1, True),
        (-64, 63, None, 1, True),
        (-128, 127, 1, 0, False),
        (_I32_MIN, _I32_MAX - 2, 1, 0, False),
        (_I32_MIN, _I32_MAX - 2, None, 1, True),
    )
    for root_kind, memory_spaces in _MEMORY_ROOTS:
        rules = _rules_for(
            root_kind,
            vector.vector_load,
            scalarized_vector_load=True,
        )
        assert len(rules) == 21
        for element_index, element_type in enumerate(("i16", "f16", "bf16")):
            operation_rules = rules[element_index * 7 : element_index * 7 + 7]
            assert [rule.descriptor.key for rule in operation_rules] == [
                "amd.xdna.aie2p.load.scalar.i16.indexed.immediate",
                *("amd.xdna.aie2p.load.scalar.i16.indexed.register",) * 6,
            ]
            for address_index, rule in enumerate(operation_rules):
                assert rule.guards[0].type_pattern.elements == (element_type,)
                memory_emits = [
                    emit
                    for emit in rule.emit
                    if isinstance(emit, EmitDescriptorOp)
                    and ".load.scalar.i16." in emit.descriptor.key
                ]
                assert len(memory_emits) == 2
                for memory_emit in memory_emits:
                    constraint = memory_emit.source_memory
                    assert constraint is not None
                    assert constraint.operation is SourceMemoryOperation.LOAD
                    assert constraint.root_kind is root_kind
                    assert constraint.address_layout is SourceMemoryAddressLayout.ANY
                    assert constraint.memory_spaces == _expected_memory_spaces(
                        constraint.operation, memory_spaces
                    )
                    assert constraint.element_byte_count == 2
                    assert constraint.vector_lane_count == 2
                    assert constraint.vector_lane_byte_stride == 2
                    assert constraint.minimum_alignment == 2
                    assert (
                        constraint.static_byte_offset_minimum,
                        constraint.static_byte_offset_maximum,
                        constraint.dynamic_term_count,
                        constraint.dynamic_term_count_minimum,
                        constraint.allow_dynamic_stride_values,
                    ) == expected_static_ranges[address_index]

                descriptor_emits = [
                    emit for emit in rule.emit if isinstance(emit, EmitDescriptorOp)
                ]
                assert [emit.descriptor.key for emit in descriptor_emits[-3:]] == [
                    "amd.xdna.aie2p.splat.i16x32",
                    "amd.xdna.aie2p.constant.i32.short",
                    "amd.xdna.aie2p.insert.i16.register",
                ]
                assert descriptor_emits[-1].copy_operands == ("idx",)

            immediate_projects = [
                emit.immediates["imm"]
                for emit in operation_rules[0].emit
                if isinstance(emit, EmitDescriptorOp)
                and ".load.scalar.i16." in emit.descriptor.key
            ]
            assert [project.kind for project in immediate_projects] == [
                SourceMemoryProjectKind.STATIC_BYTE_OFFSET,
                SourceMemoryProjectKind.STATIC_BYTE_OFFSET_PLUS_LITERAL,
            ]
            assert [project.literal_i64 for project in immediate_projects] == [0, 2]


def test_volatile_memory_rules_mirror_every_ordinary_rule() -> None:
    for root_kind, _ in _MEMORY_ROOTS:
        for source_ops, accumulator in (
            ((view.view_load, view.view_store), False),
            ((vector.vector_load, vector.vector_store), False),
            ((vector.vector_load, vector.vector_store), True),
        ):
            ordinary_rules = _rules_for(root_kind, *source_ops, accumulator=accumulator)
            volatile_rules = _rules_for(
                root_kind,
                *source_ops,
                accumulator=accumulator,
                volatile=True,
            )
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
                for volatile_emit, ordinary_emit in zip(
                    volatile_rule.emit, ordinary_rule.emit, strict=True
                ):
                    assert type(volatile_emit) is type(ordinary_emit)
                    if not isinstance(ordinary_emit, EmitDescriptorOp):
                        assert volatile_emit == ordinary_emit
                        continue
                    expected_descriptor = ordinary_emit.descriptor
                    if ordinary_emit.descriptor == ordinary_rule.descriptor:
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
            descriptor_element_type,
            element_byte_count,
            width_bits // (element_byte_count * 8),
            operation,
        )
        for width_bits in (128, 256, 512)
        for element_type, descriptor_element_type, element_byte_count in (
            ("i8", "i8", 1),
            ("f8E4M3", "i8", 1),
            ("f8E5M2", "i8", 1),
            ("i16", "i16", 2),
            ("f16", "i16", 2),
            ("bf16", "bf16", 2),
            ("i32", "i32", 4),
            ("f32", "f32", 4),
            ("i64", "i64", 8),
            ("f64", "f64", 8),
        )
        for operation in (
            SourceMemoryOperation.LOAD,
            SourceMemoryOperation.STORE,
        )
    )
    for root_kind, memory_spaces in _MEMORY_ROOTS:
        rules = _rules_for(
            root_kind,
            vector.vector_load,
            vector.vector_store,
            scalarized_vector_load=False,
            split_vector_load=False,
            wide_vector=False,
        )
        expected_descriptor_keys = []
        for (
            _,
            _,
            descriptor_element_type,
            _,
            vector_lane_count,
            operation,
        ) in expected_families:
            shape = f"{descriptor_element_type}x{vector_lane_count}"
            descriptor_prefix = (
                f"amd.xdna.aie2p.load.a.{shape}.indexed"
                if operation is SourceMemoryOperation.LOAD
                else f"amd.xdna.aie2p.store.{shape}.indexed"
            )
            expected_descriptor_keys.extend(
                [
                    f"{descriptor_prefix}.immediate",
                    *(f"{descriptor_prefix}.register",) * 6,
                ]
            )
        assert [rule.descriptor.key for rule in rules] == expected_descriptor_keys
        for family_index, (
            width_bits,
            element_type,
            descriptor_element_type,
            element_byte_count,
            vector_lane_count,
            operation,
        ) in enumerate(expected_families):
            operation_rules = rules[family_index * 7 : family_index * 7 + 7]
            shape = f"{descriptor_element_type}x{vector_lane_count}"
            descriptor_prefix = (
                f"amd.xdna.aie2p.load.a.{shape}.indexed"
                if operation is SourceMemoryOperation.LOAD
                else f"amd.xdna.aie2p.store.{shape}.indexed"
            )
            assert [rule.descriptor.key for rule in operation_rules] == [
                f"{descriptor_prefix}.immediate",
                *(f"{descriptor_prefix}.register",) * 6,
            ]
            _assert_address_forms(
                operation_rules,
                immediate_minimum=-width_bits,
                immediate_maximum=width_bits - width_bits // 8,
            )
            for rule in operation_rules:
                assert rule.guards[-1].type_pattern.elements == (element_type,)
                assert rule.guards[-1].type_pattern.lanes == vector_lane_count
                expands_logical_carrier = width_bits < 512
                slices = [
                    emit for emit in rule.emit if isinstance(emit, EmitRegisterSlice)
                ]
                concats = [
                    emit for emit in rule.emit if isinstance(emit, EmitRegisterConcat)
                ]
                if expands_logical_carrier and operation is SourceMemoryOperation.LOAD:
                    assert len(slices) == 1
                    assert len(concats) == 1
                    assert concats[0].sources[0] != concats[0].sources[1]
                    assert concats[0].sources[1].field == "memory_padding"
                elif expands_logical_carrier:
                    assert len(slices) == 1
                    assert slices[0].unit_count == 1
                    assert not concats
                else:
                    assert not slices
                    assert not concats
                memory_emit = _source_memory_emit(rule)
                assert not memory_emit.copy_operands
                assert "storage" not in memory_emit.operands
                constraint = memory_emit.source_memory
                assert constraint is not None
                assert constraint.operation is operation
                assert constraint.root_kind is root_kind
                assert constraint.address_layout is SourceMemoryAddressLayout.ANY
                assert constraint.memory_spaces == _expected_memory_spaces(
                    constraint.operation, memory_spaces
                )
                assert constraint.element_byte_count == element_byte_count
                assert constraint.vector_lane_count == vector_lane_count
                assert constraint.vector_lane_byte_stride == element_byte_count
                assert constraint.minimum_alignment == width_bits // 8


def test_256bit_vector_loads_split_at_16_byte_alignment() -> None:
    expected_shapes = (
        (("i8", "f8E4M3", "f8E5M2"), 1, 32),
        (("i16", "f16", "bf16"), 2, 16),
        (("i32", "f32"), 4, 8),
        (("i64", "f64"), 8, 4),
    )
    expected_static_ranges = (
        (-128, 96, 0, 0, False),
        (_I32_MIN, _I32_MAX - 16, 0, 0, False),
        (0, 0, None, 1, True),
        (-64, 63, None, 1, True),
        (-128, 127, 1, 0, False),
        (_I32_MIN, _I32_MAX - 16, 1, 0, False),
        (_I32_MIN, _I32_MAX - 16, None, 1, True),
    )

    for root_kind, memory_spaces in _MEMORY_ROOTS:
        rules = _rules_for(
            root_kind,
            vector.vector_load,
            scalarized_vector_load=False,
            split_vector_load=True,
            wide_vector=False,
        )
        assert len(rules) == len(expected_shapes) * 7
        for shape_index, (
            element_types,
            element_byte_count,
            vector_lane_count,
        ) in enumerate(expected_shapes):
            shape_rules = rules[shape_index * 7 : shape_index * 7 + 7]
            assert [rule.descriptor.key for rule in shape_rules] == [
                "amd.xdna.aie2p.load.a.i8x16.indexed.immediate",
                *("amd.xdna.aie2p.load.a.i8x16.indexed.register",) * 6,
            ]
            for address_index, rule in enumerate(shape_rules):
                assert rule.guards[-1].type_pattern.elements == element_types
                assert rule.guards[-1].type_pattern.lanes == vector_lane_count
                constraint = _source_memory_emit(rule).source_memory
                assert constraint is not None
                assert constraint.operation is SourceMemoryOperation.LOAD
                assert constraint.root_kind is root_kind
                assert constraint.memory_spaces == _expected_memory_spaces(
                    constraint.operation, memory_spaces
                )
                assert constraint.element_byte_count == element_byte_count
                assert constraint.vector_lane_count == vector_lane_count
                assert constraint.vector_lane_byte_stride == element_byte_count
                assert constraint.minimum_alignment == 16
                assert (
                    constraint.static_byte_offset_minimum,
                    constraint.static_byte_offset_maximum,
                    constraint.dynamic_term_count,
                    constraint.dynamic_term_count_minimum,
                    constraint.allow_dynamic_stride_values,
                ) == expected_static_ranges[address_index]

                memory_emits = [
                    emit
                    for emit in rule.emit
                    if isinstance(emit, EmitDescriptorOp)
                    and emit.descriptor == rule.descriptor
                ]
                assert len(memory_emits) == 2
                assert all(not emit.copy_operands for emit in memory_emits)
                assert all("storage" not in emit.operands for emit in memory_emits)
                concats = [
                    emit for emit in rule.emit if isinstance(emit, EmitRegisterConcat)
                ]
                assert len(concats) == 2
                assert all(
                    concat.sources[1].field == "split_padding" for concat in concats
                )

                descriptor_tail = [
                    emit.descriptor.key
                    for emit in rule.emit[-4:]
                    if isinstance(emit, EmitDescriptorOp)
                ]
                assert descriptor_tail == [
                    "amd.xdna.aie2p.constant.i32.mova",
                    "amd.xdna.aie2p.shift.bytes.x.configured",
                    "amd.xdna.aie2p.constant.i32.select",
                    "amd.xdna.aie2p.select.i32x16",
                ]
                final_select = rule.emit[-1]
                assert isinstance(final_select, EmitDescriptorOp)
                assert final_select.results["d"].field == "result"

            immediate_projects = [
                emit.immediates["imm"]
                for emit in shape_rules[0].emit
                if isinstance(emit, EmitDescriptorOp)
                and emit.descriptor == shape_rules[0].descriptor
            ]
            assert tuple(project.kind for project in immediate_projects) == (
                SourceMemoryProjectKind.STATIC_BYTE_OFFSET,
                SourceMemoryProjectKind.STATIC_BYTE_OFFSET_PLUS_LITERAL,
            )
            assert tuple(project.literal_i64 for project in immediate_projects) == (
                0,
                16,
            )


def test_wide_vector_memory_rules_preserve_two_native_chunks() -> None:
    expected_shapes = (
        ("i8", "i8", 1, 128),
        ("f8E4M3", "i8", 1, 128),
        ("f8E5M2", "i8", 1, 128),
        ("i16", "i16", 2, 64),
        ("f16", "i16", 2, 64),
        ("bf16", "bf16", 2, 64),
        ("i32", "i32", 4, 32),
        ("i64", "i64", 8, 16),
        ("f64", "f64", 8, 16),
    )
    expected_static_ranges = (
        (-512, 384, 0, 0, False),
        (_I32_MIN, _I32_MAX - 64, 0, 0, False),
        (0, 0, None, 1, True),
        (-64, 63, None, 1, True),
        (-128, 127, 1, 0, False),
        (_I32_MIN, _I32_MAX - 64, 1, 0, False),
        (_I32_MIN, _I32_MAX - 64, None, 1, True),
    )
    for root_kind, memory_spaces in _MEMORY_ROOTS:
        rules = _rules_for(
            root_kind, vector.vector_load, vector.vector_store, wide_vector=True
        )
        assert len(rules) == len(expected_shapes) * 2 * 7
        for shape_index, (
            element_type,
            descriptor_type,
            element_byte_count,
            vector_lane_count,
        ) in enumerate(expected_shapes):
            for operation_index, operation in enumerate(
                (SourceMemoryOperation.LOAD, SourceMemoryOperation.STORE)
            ):
                start = (shape_index * 2 + operation_index) * 7
                operation_rules = rules[start : start + 7]
                descriptor_family = (
                    "load.a" if operation is SourceMemoryOperation.LOAD else "store"
                )
                prefix = (
                    f"amd.xdna.aie2p.{descriptor_family}."
                    f"{descriptor_type}x{vector_lane_count // 2}.indexed"
                )
                assert [rule.descriptor.key for rule in operation_rules] == [
                    f"{prefix}.immediate",
                    *(f"{prefix}.register",) * 6,
                ]
                for address_index, rule in enumerate(operation_rules):
                    assert rule.guards[-1].type_pattern.elements == (element_type,)
                    assert rule.guards[-1].type_pattern.lanes == vector_lane_count
                    constraint = _source_memory_emit(rule).source_memory
                    assert constraint.operation is operation
                    assert constraint.root_kind is root_kind
                    assert constraint.address_layout is SourceMemoryAddressLayout.ANY
                    assert constraint.memory_spaces == _expected_memory_spaces(
                        constraint.operation, memory_spaces
                    )
                    assert constraint.element_byte_count == element_byte_count
                    assert constraint.vector_lane_count == vector_lane_count
                    assert constraint.vector_lane_byte_stride == element_byte_count
                    assert constraint.minimum_alignment == 64
                    assert constraint.cache_policy_build_flags is None
                    assert (
                        constraint.static_byte_offset_minimum,
                        constraint.static_byte_offset_maximum,
                        constraint.dynamic_term_count,
                        constraint.dynamic_term_count_minimum,
                        constraint.allow_dynamic_stride_values,
                    ) == expected_static_ranges[address_index]
                    memory_emits = [
                        emit
                        for emit in rule.emit
                        if isinstance(emit, EmitDescriptorOp)
                        and emit.descriptor == rule.descriptor
                    ]
                    assert len(memory_emits) == 2
                    assert all(
                        emit.source_memory == constraint for emit in memory_emits
                    )
                    assert all(not emit.copy_operands for emit in memory_emits)
                    assert all("storage" not in emit.operands for emit in memory_emits)
                    slices = [
                        emit
                        for emit in rule.emit
                        if isinstance(emit, EmitRegisterSlice)
                    ]
                    concats = [
                        emit
                        for emit in rule.emit
                        if isinstance(emit, EmitRegisterConcat)
                    ]
                    if operation is SourceMemoryOperation.LOAD:
                        assert not slices
                        assert len(concats) == 1
                        assert [source.field for source in concats[0].sources] == [
                            "chunk_0",
                            "chunk_1",
                        ]
                        assert concats[0].result.field == "result"
                    else:
                        assert not concats
                        assert [emit.unit_offset for emit in slices] == [0, 2]
                        assert [emit.unit_count for emit in slices] == [2, 2]
                        assert all(emit.result_type is None for emit in slices)

                immediate_projects = [
                    emit.immediates["imm"]
                    for emit in operation_rules[0].emit
                    if isinstance(emit, EmitDescriptorOp)
                ]
                assert tuple(project.kind for project in immediate_projects) == (
                    SourceMemoryProjectKind.STATIC_BYTE_OFFSET,
                    SourceMemoryProjectKind.STATIC_BYTE_OFFSET_PLUS_LITERAL,
                )
                assert tuple(project.literal_i64 for project in immediate_projects) == (
                    0,
                    64,
                )
                for address_index, expected_offsets in (
                    (1, (0, 64)),
                    (2, (64,)),
                    (3, (0, 64)),
                    (4, (0, 64)),
                    (5, (0, 64)),
                    (6, (64,)),
                ):
                    static_projects = [
                        project
                        for emit in operation_rules[address_index].emit
                        if isinstance(emit, EmitDescriptorOp)
                        if isinstance(emit.immediates, dict)
                        for project in emit.immediates.values()
                        if isinstance(project, SourceMemoryProject)
                    ]
                    assert tuple(
                        project.literal_i64 for project in static_projects
                    ) == (expected_offsets)


def test_accumulator_memory_rules_decompose_raw_payloads_into_native_chunks() -> None:
    expected_chunk_offsets = (0, 64, 128, 192)
    expected_project_kinds = (
        SourceMemoryProjectKind.STATIC_BYTE_OFFSET,
        SourceMemoryProjectKind.STATIC_BYTE_OFFSET_PLUS_LITERAL,
        SourceMemoryProjectKind.STATIC_BYTE_OFFSET_PLUS_LITERAL,
        SourceMemoryProjectKind.STATIC_BYTE_OFFSET_PLUS_LITERAL,
    )
    expected_static_ranges = (
        (-512, 256, 0, 0, False),
        (_I32_MIN, _I32_MAX - 192, 0, 0, False),
        (0, 0, None, 1, True),
        (-64, 63, None, 1, True),
        (-128, 127, 1, 0, False),
        (_I32_MIN, _I32_MAX - 192, 1, 0, False),
        (_I32_MIN, _I32_MAX - 192, None, 1, True),
    )

    for root_kind, memory_spaces in _MEMORY_ROOTS:
        root_rules = _rules_for(
            root_kind,
            vector.vector_load,
            vector.vector_store,
            accumulator=True,
        )
        for element_byte_count, vector_lane_count, element_types in (
            (4, 64, ("i32", "f32")),
            (8, 32, ("i64",)),
        ):
            rules = [
                rule
                for rule in root_rules
                if _source_memory_emit(rule).source_memory.element_byte_count
                == element_byte_count
            ]
            assert [rule.descriptor.key for rule in rules] == [
                *("amd.xdna.aie2p.load.accumulator.indexed.immediate",) * 7,
                *("amd.xdna.aie2p.store.accumulator.indexed.immediate",) * 7,
            ]
            assert all(
                rule.guards[-1].type_pattern.elements == element_types for rule in rules
            )

            for operation_index, operation in enumerate(
                (SourceMemoryOperation.LOAD, SourceMemoryOperation.STORE)
            ):
                operation_rules = rules[operation_index * 7 : operation_index * 7 + 7]
                for address_index, rule in enumerate(operation_rules):
                    constraint = _source_memory_emit(rule).source_memory
                    assert constraint is not None
                    assert constraint.operation is operation
                    assert constraint.root_kind is root_kind
                    assert constraint.address_layout is SourceMemoryAddressLayout.ANY
                    assert constraint.memory_spaces == _expected_memory_spaces(
                        constraint.operation, memory_spaces
                    )
                    assert constraint.element_byte_count == element_byte_count
                    assert constraint.vector_lane_count == vector_lane_count
                    assert constraint.vector_lane_byte_stride == element_byte_count
                    assert constraint.minimum_alignment == 64
                    assert (
                        constraint.static_byte_offset_minimum,
                        constraint.static_byte_offset_maximum,
                        constraint.dynamic_term_count,
                        constraint.dynamic_term_count_minimum,
                        constraint.allow_dynamic_stride_values,
                    ) == expected_static_ranges[address_index]

                    memory_emits = [
                        emit
                        for emit in rule.emit
                        if isinstance(emit, EmitDescriptorOp)
                        and emit.descriptor == rule.descriptor
                    ]
                    assert len(memory_emits) == 4

                    if operation is SourceMemoryOperation.LOAD:
                        assert not any(
                            isinstance(emit, EmitRegisterSlice) for emit in rule.emit
                        )
                        concat = rule.emit[-1]
                        assert isinstance(concat, EmitRegisterConcat)
                        assert [source.field for source in concat.sources] == [
                            "chunk_0",
                            "chunk_1",
                            "chunk_2",
                            "chunk_3",
                        ]
                        assert concat.result.field == "result"
                    else:
                        slices = [
                            emit
                            for emit in rule.emit
                            if isinstance(emit, EmitRegisterSlice)
                        ]
                        assert [emit.unit_offset for emit in slices] == [0, 1, 2, 3]
                        assert [emit.unit_count for emit in slices] == [1, 1, 1, 1]
                        assert all(emit.result_type is None for emit in slices)

                immediate_projects = [
                    emit.immediates["imm"]
                    for emit in operation_rules[0].emit
                    if isinstance(emit, EmitDescriptorOp)
                ]
                assert all(
                    isinstance(project, SourceMemoryProject)
                    for project in immediate_projects
                )
                assert tuple(project.kind for project in immediate_projects) == (
                    expected_project_kinds
                )
                assert (
                    tuple(project.literal_i64 for project in immediate_projects)
                    == expected_chunk_offsets
                )

                for rule in operation_rules[1:]:
                    accesses = [
                        emit
                        for emit in rule.emit
                        if isinstance(emit, EmitDescriptorOp)
                        and emit.descriptor == rule.descriptor
                    ]
                    assert tuple(emit.immediates["imm"] for emit in accesses) == (
                        expected_chunk_offsets
                    )
                    assert len({emit.operands["ptr"] for emit in accesses}) == 1
