# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for AMD XDNA AIE2P structural vector contracts."""

from loom.dialect.vector import defs as vector
from loom.target.arch.amd.xdna.aie2p.contracts.structural import (
    _HALF_CARRIER_SLICE_SPECS,
    _I32_F32_TRANSPOSE_4X4_CONTROL,
    _I32_SLICE_HIGH_BYTE_OFFSET,
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


def test_half_carrier_slices_alias_low_and_shift_high() -> None:
    for source_type, result_type, half_lane_count in _HALF_CARRIER_SLICE_SPECS:
        common_guards = (
            Guard.value_type("source", source_type),
            Guard.value_type("result", result_type),
            Guard.operand_segment_count("offsets", 0),
            Guard.i64_array_count("static_offsets", 1),
        )
        low_rule = next(
            rule
            for rule in AIE2P_STRUCTURAL_RULES
            if isinstance(rule, ValueAliasRule)
            and all(guard in rule.guards for guard in common_guards)
            and Guard.i64_array_element_range("static_offsets", 0, 0, 0) in rule.guards
        )
        high_rule = next(
            rule
            for rule in AIE2P_STRUCTURAL_RULES
            if isinstance(rule, DescriptorRule)
            and rule.source_op is vector.vector_slice
            and all(guard in rule.guards for guard in common_guards)
            and Guard.i64_array_element_range(
                "static_offsets", 0, half_lane_count, half_lane_count
            )
            in rule.guards
        )

        assert low_rule.source.field == "source"
        assert low_rule.result.field == "result"
        assert [emit.descriptor.key for emit in high_rule.emit] == [
            "amd.xdna.aie2p.constant.i32.mova",
            "amd.xdna.aie2p.shift.bytes.x.configured",
        ]
        assert high_rule.emit[0].immediates == {"i": _I32_SLICE_HIGH_BYTE_OFFSET}


def test_i32_f32_4x4_transpose_uses_native_shuffle_mode() -> None:
    rule = next(
        rule
        for rule in AIE2P_STRUCTURAL_RULES
        if isinstance(rule, DescriptorRule)
        and rule.source_op is vector.vector_transpose
    )

    assert [
        emit.descriptor.key for emit in rule.emit if isinstance(emit, EmitDescriptorOp)
    ] == [
        "amd.xdna.aie2p.constant.i32.mova",
        "amd.xdna.aie2p.shuffle.x.configured",
    ]
    assert rule.emit[0].immediates == {"i": _I32_F32_TRANSPOSE_4X4_CONTROL}
