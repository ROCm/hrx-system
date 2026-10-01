# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for AMD XDNA AIE2P accumulator structural contracts."""

from loom.target.arch.amd.xdna.aie2p.contracts.accumulator_structural import (
    _ACCUMULATOR_CONCAT_RULES,
    _ACCUMULATOR_CONCAT_TYPE_SPECS,
    _ACCUMULATOR_SLICE_RESULT_SHAPES,
    _ACCUMULATOR_VECTOR_SHAPES,
    _ACCUMULATOR_VECTOR_SLICE_RULES,
    _accumulator_concat_left_shapes,
    _accumulator_concat_right_shapes,
    _AccumulatorSliceResultShape,
    _AccumulatorVectorShape,
)
from loom.target.contracts import (
    AttrProject,
    DescriptorRule,
    EmitDescriptorOp,
    EmitRegisterConcat,
    EmitRegisterSlice,
    Guard,
    TypePattern,
    ValueAliasRule,
    ValueRef,
    ValueTypeProject,
)


def _lane_bounds(pattern: TypePattern) -> tuple[int, int]:
    if pattern.lanes is not None:
        return pattern.lanes, pattern.lanes
    assert isinstance(pattern.minimum_lanes, int)
    assert isinstance(pattern.maximum_lanes, int)
    return pattern.minimum_lanes, pattern.maximum_lanes


def _matches_lanes(pattern: TypePattern, element_type: str, lane_count: int) -> bool:
    minimum, maximum = _lane_bounds(pattern)
    return pattern.element == element_type and minimum <= lane_count <= maximum


def _value_type_pattern(
    rule: DescriptorRule | ValueAliasRule,
    field: str,
    *,
    element: int = 0,
) -> TypePattern:
    matches = tuple(
        guard.type_pattern
        for guard in rule.guards
        if guard.field == field
        and guard.element == element
        and guard.type_pattern is not None
    )
    assert len(matches) == 1
    return matches[0]


def _slice_rule(
    source_type: TypePattern,
    result_type: TypePattern,
    offset_minimum: int,
    offset_maximum: int,
) -> DescriptorRule | ValueAliasRule:
    matches = tuple(
        rule
        for rule in _ACCUMULATOR_VECTOR_SLICE_RULES
        if Guard.value_type("source", source_type) in rule.guards
        and Guard.value_type("result", result_type) in rule.guards
        and Guard.i64_array_element_range(
            "static_offsets",
            0,
            offset_minimum,
            offset_maximum,
        )
        in rule.guards
    )
    assert len(matches) == 1
    return matches[0]


def _descriptor_keys(rule: DescriptorRule) -> list[str]:
    return [
        emit.descriptor.key for emit in rule.emit if isinstance(emit, EmitDescriptorOp)
    ]


def _accumulator_source_shape(
    element_type: str,
    minimum_lane_count: int,
) -> _AccumulatorVectorShape:
    return next(
        shape
        for shape in _ACCUMULATOR_VECTOR_SHAPES
        if shape.element_type == element_type
        and shape.minimum_lane_count == minimum_lane_count
    )


def _slice_result_shape(
    element_type: str,
    minimum_lane_count: int,
) -> _AccumulatorSliceResultShape:
    return next(
        shape
        for shape in _ACCUMULATOR_SLICE_RESULT_SHAPES
        if shape.element_type == element_type
        and shape.minimum_lane_count == minimum_lane_count
    )


def _matching_concat_rules(
    element_type: str,
    left_lane_count: int,
    right_lane_count: int,
) -> tuple[DescriptorRule, ...]:
    result_lane_count = left_lane_count + right_lane_count
    return tuple(
        rule
        for rule in _ACCUMULATOR_CONCAT_RULES
        if _matches_lanes(
            _value_type_pattern(rule, "inputs", element=0),
            element_type,
            left_lane_count,
        )
        and _matches_lanes(
            _value_type_pattern(rule, "inputs", element=1),
            element_type,
            right_lane_count,
        )
        and _matches_lanes(
            _value_type_pattern(rule, "result"),
            element_type,
            result_lane_count,
        )
    )


def test_accumulator_shapes_use_only_allocatable_native_views() -> None:
    expected_units = {
        "f32": {32: 2, **{lane_count: 4 for lane_count in range(33, 65)}},
        "i32": {lane_count: 4 for lane_count in range(33, 65)},
        "i64": {lane_count: 4 for lane_count in range(17, 33)},
    }
    actual_units = {element_type: {} for element_type in expected_units}
    for shape in _ACCUMULATOR_VECTOR_SHAPES:
        for lane_count in range(
            shape.minimum_lane_count,
            shape.maximum_lane_count + 1,
        ):
            assert lane_count not in actual_units[shape.element_type]
            actual_units[shape.element_type][lane_count] = shape.register_unit_count
            expected_packet_count = (
                lane_count + shape.packet_lane_count - 1
            ) // shape.packet_lane_count
            assert expected_packet_count == shape.logical_packet_count
    assert actual_units == expected_units
    assert {shape.register_unit_count for shape in _ACCUMULATOR_VECTOR_SHAPES} == {
        2,
        4,
    }


def test_accumulator_slice_results_partition_carrier_boundaries() -> None:
    actual = {
        element_type: [
            (
                shape.minimum_lane_count,
                shape.maximum_lane_count,
                shape.logical_packet_count,
                shape.register_unit_count,
                shape.is_accumulator,
            )
            for shape in _ACCUMULATOR_SLICE_RESULT_SHAPES
            if shape.element_type == element_type
        ]
        for element_type in ("f32", "i32", "i64")
    }
    assert actual == {
        "f32": [
            (1, 16, 1, 2, False),
            (17, 31, 2, 4, False),
            (32, 32, 2, 2, True),
            (33, 48, 3, 4, True),
            (49, 64, 4, 4, True),
        ],
        "i32": [
            (1, 16, 1, 2, False),
            (17, 32, 2, 4, False),
            (33, 48, 3, 4, True),
            (49, 64, 4, 4, True),
        ],
        "i64": [
            (1, 8, 1, 2, False),
            (9, 16, 2, 4, False),
            (17, 24, 3, 4, True),
            (25, 32, 4, 4, True),
        ],
    }


def test_accumulator_slice_rules_cover_carrier_transition_boundaries() -> None:
    expected_rule_count = 0
    for source_shape in _ACCUMULATOR_VECTOR_SHAPES:
        for result_shape in _ACCUMULATOR_SLICE_RESULT_SHAPES:
            if result_shape.element_type != source_shape.element_type:
                continue
            maximum_start_packet = (
                source_shape.logical_packet_count - result_shape.logical_packet_count
            )
            if maximum_start_packet < 0:
                continue
            for source_packet_index in range(maximum_start_packet + 1):
                packet_offset = source_packet_index * source_shape.packet_lane_count
                _slice_rule(
                    source_shape.source_type,
                    result_shape.result_type,
                    packet_offset,
                    packet_offset,
                )
                expected_rule_count += 1
                shifted_offset_minimum = packet_offset + 1
                shifted_offset_maximum = min(
                    packet_offset + source_shape.packet_lane_count - 1,
                    source_shape.maximum_lane_count - result_shape.minimum_lane_count,
                )
                if shifted_offset_minimum > shifted_offset_maximum:
                    continue
                _slice_rule(
                    source_shape.source_type,
                    result_shape.result_type,
                    shifted_offset_minimum,
                    shifted_offset_maximum,
                )
                expected_rule_count += 1
    assert len(_ACCUMULATOR_VECTOR_SLICE_RULES) == expected_rule_count


def test_aligned_accumulator_slices_retain_ordered_mbms_units() -> None:
    source_shape = _accumulator_source_shape("f32", 49)
    partial_result = _slice_result_shape("f32", 33)
    assert isinstance(
        _slice_rule(
            source_shape.source_type,
            partial_result.result_type,
            0,
            0,
        ),
        ValueAliasRule,
    )

    half_result = _slice_result_shape("f32", 32)
    for offset, unit_offset in ((0, 0), (32, 2)):
        rule = _slice_rule(
            source_shape.source_type,
            half_result.result_type,
            offset,
            offset,
        )
        assert isinstance(rule, DescriptorRule)
        assert rule.descriptor is None
        assert rule.emit == (
            EmitRegisterSlice(
                source=ValueRef.operand("source"),
                result=ValueRef.result("result"),
                unit_offset=unit_offset,
            ),
        )

    shifted_packets = _slice_rule(
        source_shape.source_type,
        partial_result.result_type,
        16,
        16,
    )
    assert isinstance(shifted_packets, DescriptorRule)
    assert not _descriptor_keys(shifted_packets)
    slices = tuple(
        emit for emit in shifted_packets.emit if isinstance(emit, EmitRegisterSlice)
    )
    assert tuple(emit.unit_offset for emit in slices) == (1, 2, 3)
    joined = shifted_packets.emit[-1]
    assert isinstance(joined, EmitRegisterConcat)
    assert joined.sources[-1] == joined.sources[-2]


def test_aligned_ordinary_slices_move_each_selected_mbms_packet_to_x() -> None:
    for element_type, source_minimum, result_minimum, offset, unit_offsets in (
        ("i32", 49, 17, 32, (2, 3)),
        ("i64", 25, 9, 16, (2, 3)),
    ):
        source_shape = _accumulator_source_shape(element_type, source_minimum)
        result_shape = _slice_result_shape(element_type, result_minimum)
        rule = _slice_rule(
            source_shape.source_type,
            result_shape.result_type,
            offset,
            offset,
        )
        assert isinstance(rule, DescriptorRule)
        assert _descriptor_keys(rule) == [
            "amd.xdna.aie2p.move.accumulator512.to.vector512",
            "amd.xdna.aie2p.move.accumulator512.to.vector512",
        ]
        slices = tuple(
            emit for emit in rule.emit if isinstance(emit, EmitRegisterSlice)
        )
        assert tuple(emit.unit_offset for emit in slices) == unit_offsets
        assert isinstance(rule.emit[-1], EmitRegisterConcat)


def test_shifted_subpacket_in_the_final_packet_reuses_its_x_move() -> None:
    source_shape = _accumulator_source_shape("f32", 32)
    result_shape = _slice_result_shape("f32", 1)
    rule = _slice_rule(
        source_shape.source_type,
        result_shape.result_type,
        17,
        31,
    )
    assert isinstance(rule, DescriptorRule)
    assert _descriptor_keys(rule) == [
        "amd.xdna.aie2p.move.accumulator512.to.vector512",
        "amd.xdna.aie2p.constant.i32.mova",
        "amd.xdna.aie2p.shift.bytes.x.configured",
    ]
    assert rule.emit[2].immediates == {
        "i": AttrProject.i64_array_lane_byte_offset(
            "static_offsets",
            element=0,
            bytes_per_lane=4,
            base_byte_offset=-64,
        )
    }
    shift = rule.emit[-1]
    assert isinstance(shift, EmitDescriptorOp)
    assert shift.operands["s1"] == shift.operands["s2"]


def test_crossing_subpacket_moves_the_adjacent_mbms_packet() -> None:
    source_shape = _accumulator_source_shape("f32", 32)
    result_shape = _slice_result_shape("f32", 1)
    rule = _slice_rule(
        source_shape.source_type,
        result_shape.result_type,
        1,
        15,
    )
    assert isinstance(rule, DescriptorRule)
    assert _descriptor_keys(rule) == [
        "amd.xdna.aie2p.move.accumulator512.to.vector512",
        "amd.xdna.aie2p.move.accumulator512.to.vector512",
        "amd.xdna.aie2p.constant.i32.mova",
        "amd.xdna.aie2p.shift.bytes.x.configured",
    ]
    shift = rule.emit[-1]
    assert isinstance(shift, EmitDescriptorOp)
    assert shift.operands["s1"] != shift.operands["s2"]


def test_shifted_accumulator_result_returns_each_logical_packet_to_mbms() -> None:
    source_shape = _accumulator_source_shape("f32", 49)
    result_shape = _slice_result_shape("f32", 32)
    rule = _slice_rule(
        source_shape.source_type,
        result_shape.result_type,
        1,
        15,
    )
    assert isinstance(rule, DescriptorRule)
    keys = _descriptor_keys(rule)
    assert keys.count("amd.xdna.aie2p.move.accumulator512.to.vector512") == 3
    assert keys.count("amd.xdna.aie2p.shift.bytes.x.configured") == 2
    assert keys.count("amd.xdna.aie2p.move.vector512.to.accumulator512") == 2
    joined = rule.emit[-1]
    assert isinstance(joined, EmitRegisterConcat)
    assert len(joined.sources) == 2


def test_shifted_partial_accumulator_pads_only_the_physical_carrier() -> None:
    source_shape = _accumulator_source_shape("f32", 49)
    result_shape = _slice_result_shape("f32", 33)
    rule = _slice_rule(
        source_shape.source_type,
        result_shape.result_type,
        1,
        15,
    )
    assert isinstance(rule, DescriptorRule)
    keys = _descriptor_keys(rule)
    assert keys.count("amd.xdna.aie2p.move.accumulator512.to.vector512") == 4
    assert keys.count("amd.xdna.aie2p.shift.bytes.x.configured") == 3
    assert keys.count("amd.xdna.aie2p.move.vector512.to.accumulator512") == 3
    joined = rule.emit[-1]
    assert isinstance(joined, EmitRegisterConcat)
    assert len(joined.sources) == 4
    assert joined.sources[-1] == joined.sources[-2]


def test_concat_operand_shapes_partition_each_logical_domain() -> None:
    for element_type, _, packet_lane_count in _ACCUMULATOR_CONCAT_TYPE_SPECS:
        for shapes in (
            _accumulator_concat_left_shapes(element_type, packet_lane_count),
            _accumulator_concat_right_shapes(element_type, packet_lane_count),
        ):
            for lane_count in range(1, 4 * packet_lane_count + 1):
                matches = tuple(
                    shape
                    for shape in shapes
                    if shape.minimum_lane_count
                    <= lane_count
                    <= shape.maximum_lane_count
                )
                assert len(matches) == 1
                shape = matches[0]
                assert (
                    shape.logical_packet_count
                    == (lane_count + packet_lane_count - 1) // packet_lane_count
                )
                expected_accumulator_units = (
                    2
                    if element_type == "f32" and lane_count == 2 * packet_lane_count
                    else 4
                    if lane_count > 2 * packet_lane_count
                    else None
                )
                assert shape.accumulator_unit_count == expected_accumulator_units


def test_accumulator_concat_rules_cover_every_binary_partition_once() -> None:
    result_domains = {
        "f32": range(32, 65),
        "i32": range(33, 65),
        "i64": range(17, 33),
    }
    for element_type, result_lane_counts in result_domains.items():
        for result_lane_count in result_lane_counts:
            for left_lane_count in range(1, result_lane_count):
                rules = _matching_concat_rules(
                    element_type,
                    left_lane_count,
                    result_lane_count - left_lane_count,
                )
                assert len(rules) == 1


def test_accumulator_concat_rules_fill_only_allocatable_carrier_units() -> None:
    for rule in _ACCUMULATOR_CONCAT_RULES:
        result_pattern = _value_type_pattern(rule, "result")
        result_minimum, result_maximum = _lane_bounds(result_pattern)
        shape = next(
            shape
            for shape in _ACCUMULATOR_VECTOR_SHAPES
            if shape.element_type == result_pattern.element
            and shape.minimum_lane_count <= result_minimum
            and result_maximum <= shape.maximum_lane_count
        )
        joined = rule.emit[-1]
        assert isinstance(joined, EmitRegisterConcat)
        if rule.descriptor is None:
            # The complete f32x32 pair concatenates two two-unit aggregates;
            # its dedicated test below checks that zero-instruction form.
            continue
        assert len(joined.sources) == shape.register_unit_count
        if shape.logical_packet_count < shape.register_unit_count:
            assert joined.sources[shape.logical_packet_count :] == (
                joined.sources[shape.logical_packet_count - 1],
            ) * (shape.register_unit_count - shape.logical_packet_count)


def test_partial_accumulator_concat_pads_only_the_physical_carrier() -> None:
    ordinary_boundary = _matching_concat_rules("i64", 16, 1)[0]
    joined = ordinary_boundary.emit[-1]
    assert isinstance(joined, EmitRegisterConcat)
    assert len(joined.sources) == 4
    assert joined.sources[-1] == joined.sources[-2]
    assert (
        sum(
            isinstance(emit, EmitDescriptorOp)
            and emit.descriptor.key == "amd.xdna.aie2p.move.vector512.to.accumulator512"
            for emit in ordinary_boundary.emit
        )
        == 3
    )

    f32_boundary = _matching_concat_rules("f32", 31, 1)[0]
    joined = f32_boundary.emit[-1]
    assert isinstance(joined, EmitRegisterConcat)
    assert len(joined.sources) == 2
    assert joined.sources[-1] != joined.sources[-2]


def test_partial_left_accumulator_repacks_only_the_logical_tail() -> None:
    rule = _matching_concat_rules("i64", 17, 1)[0]
    assert rule.descriptor is not None
    assert rule.descriptor.key == "amd.xdna.aie2p.shift.bytes.x.configured"
    descriptor_emits = tuple(
        emit for emit in rule.emit if isinstance(emit, EmitDescriptorOp)
    )
    assert [emit.descriptor.key for emit in descriptor_emits[:3]] == [
        "amd.xdna.aie2p.move.accumulator512.to.vector512",
        "amd.xdna.aie2p.constant.i32.mova",
        "amd.xdna.aie2p.shift.bytes.x.configured",
    ]
    assert descriptor_emits[1].immediates == {
        "i": ValueTypeProject.static_dim_scaled(
            ValueRef.operand("inputs", element=0),
            scale=8,
            addend=-128,
        )
    }
    remaining = next(
        emit
        for emit in descriptor_emits
        if emit.immediates
        == {
            "i": ValueTypeProject.literal_minus_static_dim_scaled(
                ValueRef.operand("inputs", element=0),
                scale=8,
                literal=192,
            )
        }
    )
    assert remaining.descriptor.key == "amd.xdna.aie2p.constant.i32.mova"
    joined = rule.emit[-1]
    assert isinstance(joined, EmitRegisterConcat)
    assert joined.sources[-1] == joined.sources[-2]


def test_complete_f32_accumulator_pair_remains_a_register_concat() -> None:
    rule = _matching_concat_rules("f32", 32, 32)[0]
    assert rule.descriptor is None
    assert rule.emit == (
        EmitRegisterConcat(
            sources=(
                ValueRef.operand("inputs", element=0),
                ValueRef.operand("inputs", element=1),
            ),
            result=ValueRef.result("result"),
        ),
    )
