# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for AMD XDNA AIE2P structural vector contracts."""

from loom.dialect.vector import defs as vector
from loom.target.arch.amd.xdna.aie2p.contracts.structural import (
    _I16_F16_BF16_8X8_VECTOR,
    _I16_INTERLEAVE_CONTROL,
    _I16_TRANSPOSE_8X8_CONTROLS,
    _I32_F32_4X4_VECTOR,
    _I32_F32_TRANSPOSE_4X4_CONTROL,
    _PACKED_VECTOR_ELEMENT_TYPES,
    _VECTOR_CARRIER_SPECS,
    _WIDE_VECTOR_BITCAST_TYPES,
    _WIDE_VECTOR_CONCAT_SPECS,
    _WIDE_VECTOR_EXTRACT_SPECS,
    AIE2P_STRUCTURAL_RULES,
)
from loom.target.contracts import (
    AttrProject,
    DescriptorRule,
    EmitDescriptorOp,
    EmitRegisterConcat,
    EmitRegisterSlice,
    Guard,
    ValueAliasRule,
    ValueRef,
    ValueTypeProject,
    Vector,
)


def _wide_extract_rule(
    source_type,
    result_type,
    *index_guards: Guard,
) -> DescriptorRule:
    return next(
        rule
        for rule in AIE2P_STRUCTURAL_RULES
        if isinstance(rule, DescriptorRule)
        and rule.source_op is vector.vector_extract
        and Guard.value_type("source", source_type) in rule.guards
        and Guard.value_type("result", result_type) in rule.guards
        and all(guard in rule.guards for guard in index_guards)
    )


def test_wide_static_extracts_select_and_normalize_each_half() -> None:
    for source_type, result_type, half_lanes, storage, _ in _WIDE_VECTOR_EXTRACT_SPECS:
        low_rule = _wide_extract_rule(
            source_type,
            result_type,
            Guard.operand_segment_count("indices", 0),
            Guard.i64_array_element_range("static_indices", 0, 0, half_lanes - 1),
        )
        high_rule = _wide_extract_rule(
            source_type,
            result_type,
            Guard.operand_segment_count("indices", 0),
            Guard.i64_array_element_range(
                "static_indices", 0, half_lanes, 2 * half_lanes - 1
            ),
        )

        assert low_rule.descriptor.key == (
            f"amd.xdna.aie2p.extract.{storage}.immediate"
        )
        assert isinstance(low_rule.emit[0], EmitRegisterSlice)
        assert low_rule.emit[0].unit_offset == 0
        assert low_rule.emit[0].unit_count == 2
        assert isinstance(low_rule.emit[1], EmitDescriptorOp)
        assert low_rule.emit[1].immediates == {
            "idx": AttrProject.i64_array_element("static_indices", element=0)
        }

        assert isinstance(high_rule.emit[0], EmitRegisterSlice)
        assert high_rule.emit[0].unit_offset == 2
        assert high_rule.emit[0].unit_count == 2
        assert isinstance(high_rule.emit[1], EmitDescriptorOp)
        assert high_rule.emit[1].immediates == {
            "idx": AttrProject.i64_array_element_plus_literal(
                "static_indices", element=0, literal=-half_lanes
            )
        }


def test_wide_dynamic_extract_selects_one_physical_half() -> None:
    source_type, result_type, half_lanes, storage, _ = next(
        spec for spec in _WIDE_VECTOR_EXTRACT_SPECS if spec[1].element == "i32"
    )
    rule = _wide_extract_rule(
        source_type,
        result_type,
        Guard.operand_segment_count("indices", 1),
    )

    assert [
        emit.descriptor.key for emit in rule.emit if isinstance(emit, EmitDescriptorOp)
    ] == [
        "amd.xdna.aie2p.constant.i32.short",
        "amd.xdna.aie2p.constant.i32.short",
        "amd.xdna.aie2p.and.i32",
        "amd.xdna.aie2p.and.i32",
        f"amd.xdna.aie2p.extract.{storage}.register",
        f"amd.xdna.aie2p.extract.{storage}.register",
        "amd.xdna.aie2p.select.nonzero.i32",
    ]
    assert rule.emit[2].immediates == {"i": half_lanes - 1}
    assert rule.emit[3].immediates == {"i": half_lanes}


def test_wide_pair_extract_selects_each_scalar_word() -> None:
    source_type, result_type, _, _, _ = next(
        spec for spec in _WIDE_VECTOR_EXTRACT_SPECS if spec[1].element == "i64"
    )
    rule = _wide_extract_rule(
        source_type,
        result_type,
        Guard.operand_segment_count("indices", 1),
    )

    assert isinstance(rule.emit[-1], EmitRegisterConcat)
    assert tuple(source.field for source in rule.emit[-1].sources) == (
        "selected_0",
        "selected_1",
    )


def _slice_rule(
    source_type,
    result_type,
    offset_minimum: int,
    offset_maximum: int,
):
    return next(
        rule
        for rule in AIE2P_STRUCTURAL_RULES
        if rule.source_op is vector.vector_slice
        and Guard.value_type("source", source_type) in rule.guards
        and Guard.value_type("result", result_type) in rule.guards
        and Guard.i64_array_element_range(
            "static_offsets", 0, offset_minimum, offset_maximum
        )
        in rule.guards
    )


def test_static_slices_project_logical_lanes_into_physical_carriers() -> None:
    slice_rules = [
        rule for rule in AIE2P_STRUCTURAL_RULES if rule.source_op is vector.vector_slice
    ]
    assert len(slice_rules) == 8 * len(_VECTOR_CARRIER_SPECS)

    for element_types, element_byte_count, wide_lane_maximum in _VECTOR_CARRIER_SPECS:
        carrier_lane_count = 64 // element_byte_count
        narrow_type = Vector(
            element_types,
            minimum_lanes=1,
            maximum_lanes=carrier_lane_count,
        )
        wide_type = Vector(
            element_types,
            minimum_lanes=carrier_lane_count + 1,
            maximum_lanes=wide_lane_maximum,
        )

        narrow_low = _slice_rule(narrow_type, narrow_type, 0, 0)
        assert isinstance(narrow_low, ValueAliasRule)
        narrow_shift = _slice_rule(
            narrow_type,
            narrow_type,
            1,
            carrier_lane_count - 1,
        )
        assert narrow_shift.emit[0].immediates == {
            "i": AttrProject.i64_array_lane_byte_offset(
                "static_offsets",
                element=0,
                bytes_per_lane=element_byte_count,
            )
        }

        wide_low = _slice_rule(wide_type, narrow_type, 0, 0)
        assert isinstance(wide_low.emit[0], EmitRegisterSlice)
        assert wide_low.emit[0].unit_offset == 0
        wide_crossing = _slice_rule(
            wide_type,
            narrow_type,
            1,
            carrier_lane_count - 1,
        )
        assert [emit.unit_offset for emit in wide_crossing.emit[:2]] == [0, 2]
        wide_high = _slice_rule(
            wide_type,
            narrow_type,
            carrier_lane_count,
            carrier_lane_count,
        )
        assert wide_high.emit[0].unit_offset == 2
        high_shift = _slice_rule(
            wide_type,
            narrow_type,
            carrier_lane_count + 1,
            wide_lane_maximum - 1,
        )
        assert high_shift.emit[1].immediates == {
            "i": AttrProject.i64_array_lane_byte_offset(
                "static_offsets",
                element=0,
                bytes_per_lane=element_byte_count,
                base_byte_offset=-64,
            )
        }

        wide_alias = _slice_rule(wide_type, wide_type, 0, 0)
        assert isinstance(wide_alias, ValueAliasRule)
        wide_shift = _slice_rule(
            wide_type,
            wide_type,
            1,
            carrier_lane_count - 1,
        )
        assert [type(emit) for emit in wide_shift.emit] == [
            EmitRegisterSlice,
            EmitRegisterSlice,
            EmitDescriptorOp,
            EmitDescriptorOp,
            EmitDescriptorOp,
            EmitRegisterConcat,
        ]
        assert wide_shift.emit[2].immediates == {
            "i": AttrProject.i64_array_lane_byte_offset(
                "static_offsets",
                element=0,
                bytes_per_lane=element_byte_count,
            )
        }


def _concat_rule(
    input_type,
    result_type,
    *,
    right_type=None,
) -> DescriptorRule:
    return next(
        rule
        for rule in AIE2P_STRUCTURAL_RULES
        if isinstance(rule, DescriptorRule)
        and rule.source_op is vector.vector_concat
        and Guard.value_type("inputs", input_type) in rule.guards
        and (
            right_type is None
            or Guard.value_type("inputs", right_type, element=1) in rule.guards
        )
        and Guard.value_type("result", result_type) in rule.guards
    )


def test_partial_concat_projects_one_verified_byte_cut_per_element_width() -> None:
    concat_rules = [
        rule
        for rule in AIE2P_STRUCTURAL_RULES
        if rule.source_op is vector.vector_concat
    ]
    assert len(concat_rules) == (
        2 * len(_PACKED_VECTOR_ELEMENT_TYPES)
        + len(_WIDE_VECTOR_CONCAT_SPECS)
        + 3 * len(_VECTOR_CARRIER_SPECS)
    )

    for element_types, element_byte_count in _PACKED_VECTOR_ELEMENT_TYPES:
        carrier_lane_count = 64 // element_byte_count
        left = ValueRef.operand("inputs", element=0)
        shift_rule = _concat_rule(
            Vector(
                element_types,
                minimum_lanes=1,
                maximum_lanes=carrier_lane_count - 1,
            ),
            Vector(
                element_types,
                minimum_lanes=2,
                maximum_lanes=carrier_lane_count,
            ),
        )
        assert [emit.descriptor.key for emit in shift_rule.emit] == [
            "amd.xdna.aie2p.constant.i32.mova",
            "amd.xdna.aie2p.shift.bytes.x.configured",
            "amd.xdna.aie2p.constant.i32.mova",
            "amd.xdna.aie2p.shift.bytes.x.configured",
        ]
        assert shift_rule.emit[0].immediates == {
            "i": ValueTypeProject.static_dim_scaled(
                left,
                scale=element_byte_count,
            )
        }
        assert shift_rule.emit[2].immediates == {
            "i": ValueTypeProject.literal_minus_static_dim_scaled(
                left,
                scale=element_byte_count,
                literal=64,
            )
        }

        half_lane_count = 32 // element_byte_count
        half_rule = _concat_rule(
            Vector(element_types, lanes=half_lane_count),
            Vector(
                element_types,
                minimum_lanes=half_lane_count + 1,
                maximum_lanes=carrier_lane_count,
            ),
        )
        assert half_rule.priority == 1
        assert [type(emit) for emit in half_rule.emit] == [
            EmitRegisterSlice,
            EmitRegisterSlice,
            EmitRegisterConcat,
        ]
        assert [emit.source.element for emit in half_rule.emit[:2]] == [0, 1]
        assert all(emit.unit_count == 1 for emit in half_rule.emit[:2])

    for input_type, result_type in _WIDE_VECTOR_CONCAT_SPECS:
        wide_rule = _concat_rule(input_type, result_type)
        assert len(wide_rule.emit) == 1
        assert isinstance(wide_rule.emit[0], EmitRegisterConcat)
        assert [source.element for source in wide_rule.emit[0].sources] == [0, 1]


def test_split_carrier_concat_covers_each_binary_input_partition() -> None:
    for element_types, element_byte_count, wide_lane_maximum in _VECTOR_CARRIER_SPECS:
        carrier_lane_count = 64 // element_byte_count
        narrow_left = Vector(
            element_types,
            minimum_lanes=1,
            maximum_lanes=carrier_lane_count - 1,
        )
        narrow_right = Vector(
            element_types,
            minimum_lanes=1,
            maximum_lanes=carrier_lane_count,
        )
        partial_right = Vector(
            element_types,
            minimum_lanes=1,
            maximum_lanes=carrier_lane_count - 1,
        )
        wide_operand = Vector(
            element_types,
            minimum_lanes=carrier_lane_count + 1,
            maximum_lanes=wide_lane_maximum - 1,
        )

        narrow_pair_rule = _concat_rule(
            narrow_left,
            Vector(
                element_types,
                minimum_lanes=carrier_lane_count + 1,
                maximum_lanes=wide_lane_maximum,
            ),
            right_type=narrow_right,
        )
        assert [type(emit) for emit in narrow_pair_rule.emit] == [
            EmitDescriptorOp,
            EmitDescriptorOp,
            EmitDescriptorOp,
            EmitDescriptorOp,
            EmitDescriptorOp,
            EmitRegisterConcat,
        ]
        assert narrow_pair_rule.emit[0].immediates == {
            "i": ValueTypeProject.static_dim_scaled(
                ValueRef.operand("inputs"),
                scale=element_byte_count,
            )
        }
        assert narrow_pair_rule.emit[2].immediates == {
            "i": ValueTypeProject.literal_minus_static_dim_scaled(
                ValueRef.operand("inputs"),
                scale=element_byte_count,
                literal=64,
            )
        }

        narrow_wide_rule = _concat_rule(
            narrow_left,
            Vector(
                element_types,
                minimum_lanes=carrier_lane_count + 2,
                maximum_lanes=wide_lane_maximum,
            ),
            right_type=wide_operand,
        )
        assert [type(emit) for emit in narrow_wide_rule.emit[:2]] == [
            EmitRegisterSlice,
            EmitRegisterSlice,
        ]
        assert [emit.source.element for emit in narrow_wide_rule.emit[:2]] == [
            1,
            1,
        ]
        assert [emit.unit_offset for emit in narrow_wide_rule.emit[:2]] == [0, 2]
        assert isinstance(narrow_wide_rule.emit[-1], EmitRegisterConcat)

        wide_narrow_rule = _concat_rule(
            wide_operand,
            Vector(
                element_types,
                minimum_lanes=carrier_lane_count + 2,
                maximum_lanes=wide_lane_maximum,
            ),
            right_type=partial_right,
        )
        assert [type(emit) for emit in wide_narrow_rule.emit[:2]] == [
            EmitRegisterSlice,
            EmitRegisterSlice,
        ]
        assert [emit.source.element for emit in wide_narrow_rule.emit[:2]] == [
            0,
            0,
        ]
        assert wide_narrow_rule.emit[2].immediates == {
            "i": ValueTypeProject.static_dim_scaled(
                ValueRef.operand("inputs"),
                scale=element_byte_count,
                addend=-64,
            )
        }
        assert wide_narrow_rule.emit[4].immediates == {
            "i": ValueTypeProject.literal_minus_static_dim_scaled(
                ValueRef.operand("inputs"),
                scale=element_byte_count,
                literal=128,
            )
        }
        assert isinstance(wide_narrow_rule.emit[-1], EmitRegisterConcat)


def test_i32_f32_4x4_transpose_uses_native_shuffle_mode() -> None:
    rule = next(
        rule
        for rule in AIE2P_STRUCTURAL_RULES
        if isinstance(rule, DescriptorRule)
        and rule.source_op is vector.vector_transpose
        and Guard.value_type("source", _I32_F32_4X4_VECTOR) in rule.guards
    )

    assert [
        emit.descriptor.key for emit in rule.emit if isinstance(emit, EmitDescriptorOp)
    ] == [
        "amd.xdna.aie2p.constant.i32.mova",
        "amd.xdna.aie2p.shuffle.x.configured",
    ]
    assert rule.emit[0].immediates == {"i": _I32_F32_TRANSPOSE_4X4_CONTROL}


def test_16bit_8x8_transpose_preserves_both_full_carrier_halves() -> None:
    assert _I16_F16_BF16_8X8_VECTOR == Vector(("i16", "f16", "bf16"), dims=(8, 8))
    rule = next(
        rule
        for rule in AIE2P_STRUCTURAL_RULES
        if isinstance(rule, DescriptorRule)
        and rule.source_op is vector.vector_transpose
        and Guard.value_type("source", _I16_F16_BF16_8X8_VECTOR) in rule.guards
    )
    assert rule.guards == (
        Guard.value_type("source", _I16_F16_BF16_8X8_VECTOR),
        Guard.value_type("result", _I16_F16_BF16_8X8_VECTOR),
        Guard.i64_array_count("permutation", 2),
        Guard.i64_array_element_range("permutation", 0, 1, 1),
        Guard.i64_array_element_range("permutation", 1, 0, 0),
    )
    assert len(rule.emit) == 7
    low, high = rule.emit[:2]
    assert isinstance(low, EmitRegisterSlice)
    assert isinstance(high, EmitRegisterSlice)
    assert low.source.field == high.source.field == "source"
    assert (low.unit_offset, low.unit_count) == (0, 2)
    assert (high.unit_offset, high.unit_count) == (2, 2)
    assert _I16_TRANSPOSE_8X8_CONTROLS == (52, 53)
    assert [
        (operand.field_name, operand.unit_count) for operand in rule.descriptor.operands
    ] == [("dst", 2), ("s1", 2), ("s2", 2), ("mod", 1)]
    for index, mode in enumerate(_I16_TRANSPOSE_8X8_CONTROLS):
        constant, shuffle = rule.emit[2 + 2 * index : 4 + 2 * index]
        assert constant.descriptor.key == "amd.xdna.aie2p.constant.i32.mova"
        assert constant.immediates == {"i": mode}
        assert shuffle.descriptor.key == "amd.xdna.aie2p.shuffle.x.configured"
        assert shuffle.operands["s1"] == low.result
        assert shuffle.operands["s2"] == high.result
        assert shuffle.operands["mod"] == constant.results["dst"]
    joined = rule.emit[-1]
    assert isinstance(joined, EmitRegisterConcat)
    assert tuple(joined.sources) == (
        rule.emit[3].results["dst"],
        rule.emit[5].results["dst"],
    )
    assert joined.result.field == "result"


def test_wide_bitcast_aliases_preserve_ordinary_y_carriers() -> None:
    assert _WIDE_VECTOR_BITCAST_TYPES == (
        Vector(
            ("i8", "f8E4M3", "f8E5M2"),
            minimum_static_elements=65,
            maximum_static_elements=128,
        ),
        Vector(
            ("i16", "f16", "bf16"),
            minimum_static_elements=33,
            maximum_static_elements=64,
        ),
        Vector("i32", minimum_static_elements=17, maximum_static_elements=32),
        Vector("f32", minimum_static_elements=17, maximum_static_elements=31),
        Vector(
            ("i64", "f64"),
            minimum_static_elements=9,
            maximum_static_elements=16,
        ),
    )
    rules = tuple(
        rule
        for rule in AIE2P_STRUCTURAL_RULES
        if isinstance(rule, ValueAliasRule) and rule.source_op is vector.vector_bitcast
    )
    assert len(rules) == len(_WIDE_VECTOR_BITCAST_TYPES) ** 2
    assert [rule.guards for rule in rules] == [
        (
            Guard.value_type("input", source_type),
            Guard.value_type("result", result_type),
            Guard.low_value_register_unit_count_eq("input", "result"),
        )
        for source_type in _WIDE_VECTOR_BITCAST_TYPES
        for result_type in _WIDE_VECTOR_BITCAST_TYPES
    ]
    assert all(rule.source.field == "input" for rule in rules)
    assert all(rule.result.field == "result" for rule in rules)


def test_16bit_interleave_uses_alternating_native_shuffle() -> None:
    rule = next(
        rule
        for rule in AIE2P_STRUCTURAL_RULES
        if isinstance(rule, DescriptorRule)
        and rule.source_op is vector.vector_interleave
    )
    assert len(rule.emit) == 2
    assert rule.emit[0].immediates == {"i": _I16_INTERLEAVE_CONTROL}
    assert _I16_INTERLEAVE_CONTROL == 18
    assert rule.emit[1].descriptor.key == "amd.xdna.aie2p.shuffle.x.configured"
    assert rule.emit[1].operands["s1"].field == "even"
    assert rule.emit[1].operands["s2"].field == "odd"
    assert Guard.i64_range("axis", 0, 0) in rule.guards
