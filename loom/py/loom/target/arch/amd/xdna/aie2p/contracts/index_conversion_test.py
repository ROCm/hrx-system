# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for AMD XDNA AIE2P address-boundary conversion rules."""

from loom.dialect.index import defs as index
from loom.dialect.vector import defs as vector
from loom.target.arch.amd.xdna.aie2p.contracts.index_conversion import (
    AIE2P_INDEX_CONVERSION_RULES,
    AIE2P_VECTOR_INDEX_CONVERSION_RULES,
)
from loom.target.contracts import (
    DescriptorRule,
    EmitDescriptorOp,
    EmitRegisterConcat,
    EmitRegisterSlice,
    GuardKind,
    TypePattern,
    ValueAliasRule,
)


def _cast_pair(rule: DescriptorRule | ValueAliasRule) -> tuple[str, str]:
    type_guards = {
        guard.field: guard.type_pattern.element
        for guard in rule.guards
        if guard.kind is GuardKind.VALUE_TYPE and guard.type_pattern is not None
    }
    return type_guards["input"], type_guards["result"]


def _rule_for(
    input_type: str,
    result_type: str,
) -> DescriptorRule | ValueAliasRule:
    return next(
        rule
        for rule in AIE2P_INDEX_CONVERSION_RULES
        if _cast_pair(rule) == (input_type, result_type)
    )


def _vector_rules_for(
    input_type: str,
    result_type: str,
) -> tuple[DescriptorRule | ValueAliasRule, ...]:
    return tuple(
        rule
        for rule in AIE2P_VECTOR_INDEX_CONVERSION_RULES
        if _cast_pair(rule) == (input_type, result_type)
    )


def _covered_lane_counts(type_pattern: TypePattern) -> set[int]:
    if type_pattern.lanes is not None:
        return {type_pattern.lanes}
    if type_pattern.minimum_lanes is not None:
        assert isinstance(type_pattern.minimum_lanes, int)
        assert isinstance(type_pattern.maximum_lanes, int)
        return set(range(type_pattern.minimum_lanes, type_pattern.maximum_lanes + 1))
    assert isinstance(type_pattern.minimum_static_elements, int)
    assert isinstance(type_pattern.maximum_static_elements, int)
    return set(
        range(
            type_pattern.minimum_static_elements,
            type_pattern.maximum_static_elements + 1,
        )
    )


def test_index_conversion_covers_every_address_boundary_pair_once() -> None:
    fixed_width_types = {"i1", "i8", "i16", "i32", "i64"}
    address_types = {"index", "offset"}
    scalar_types = fixed_width_types | address_types
    expected_pairs = {
        (input_type, result_type)
        for input_type in scalar_types
        for result_type in scalar_types
        if input_type in address_types or result_type in address_types
    }

    actual_pairs = [_cast_pair(rule) for rule in AIE2P_INDEX_CONVERSION_RULES]
    assert len(actual_pairs) == 24
    assert len(set(actual_pairs)) == len(actual_pairs)
    assert set(actual_pairs) == expected_pairs
    assert all(
        rule.source_op is index.index_cast for rule in AIE2P_INDEX_CONVERSION_RULES
    )


def test_i64_address_boundaries_use_structural_register_units() -> None:
    to_index = _rule_for("i64", "index")
    to_offset = _rule_for("i64", "offset")
    from_index = _rule_for("index", "i64")
    from_offset = _rule_for("offset", "i64")

    assert isinstance(to_index, DescriptorRule)
    assert isinstance(to_offset, DescriptorRule)
    assert to_index.descriptor is None
    assert to_offset.descriptor is None
    assert len(to_index.emit) == 1
    assert len(to_offset.emit) == 1
    assert isinstance(to_index.emit[0], EmitRegisterSlice)
    assert isinstance(to_offset.emit[0], EmitRegisterSlice)
    assert to_index.emit[0].unit_offset == 0
    assert to_offset.emit[0].unit_offset == 0
    assert all(guard.kind is GuardKind.VALUE_TYPE for guard in to_index.guards)
    assert any(
        guard.kind is GuardKind.VALUE_UNSIGNED_BIT_COUNT and guard.count == 32
        for guard in to_offset.guards
    )

    assert isinstance(from_index, DescriptorRule)
    assert isinstance(from_offset, DescriptorRule)
    assert from_index.descriptor.key == "amd.xdna.aie2p.ashl.i32"
    assert from_offset.descriptor.key == "amd.xdna.aie2p.constant.i32.short"
    assert isinstance(from_index.emit[-1], EmitRegisterConcat)
    assert isinstance(from_offset.emit[-1], EmitRegisterConcat)
    assert len(from_index.emit[-1].sources) == 2
    assert len(from_offset.emit[-1].sources) == 2


def test_same_carrier_casts_are_aliases_with_exact_domain_guards() -> None:
    alias_pairs = {
        _cast_pair(rule)
        for rule in AIE2P_INDEX_CONVERSION_RULES
        if isinstance(rule, ValueAliasRule)
    }
    assert alias_pairs == {
        ("i1", "index"),
        ("i1", "offset"),
        ("i32", "index"),
        ("i32", "offset"),
        ("index", "i8"),
        ("index", "i16"),
        ("index", "i32"),
        ("offset", "i8"),
        ("offset", "i16"),
        ("offset", "i32"),
        ("index", "index"),
        ("offset", "offset"),
        ("index", "offset"),
        ("offset", "index"),
    }
    for pair in (("index", "offset"), ("offset", "index")):
        rule = _rule_for(*pair)
        assert isinstance(rule, ValueAliasRule)
        range_guards = [
            guard for guard in rule.guards if guard.kind is GuardKind.VALUE_I64_RANGE
        ]
        if pair[1] == "offset":
            assert len(range_guards) == 1
            assert range_guards[0].minimum == 0
            assert range_guards[0].maximum == (2**31) - 1
        else:
            assert not range_guards


def test_vector_index_conversion_covers_every_address_boundary_pair() -> None:
    fixed_width_types = {"i1", "i8", "i16", "i32", "i64"}
    address_types = {"index", "offset"}
    scalar_types = fixed_width_types | address_types
    expected_pairs = {
        (input_type, result_type)
        for input_type in scalar_types
        for result_type in scalar_types
        if input_type in address_types or result_type in address_types
    }

    actual_pairs = [_cast_pair(rule) for rule in AIE2P_VECTOR_INDEX_CONVERSION_RULES]
    assert len(AIE2P_VECTOR_INDEX_CONVERSION_RULES) == 56
    assert set(actual_pairs) == expected_pairs
    assert len(set(actual_pairs)) == 24
    assert all(
        rule.source_op is vector.vector_index_cast
        for rule in AIE2P_VECTOR_INDEX_CONVERSION_RULES
    )
    report_keys = [
        rule.report_key
        for rule in AIE2P_VECTOR_INDEX_CONVERSION_RULES
        if isinstance(rule, DescriptorRule)
    ]
    assert all(report_keys)
    assert set(report_keys) == {
        "native_index_cast_address_to_i64",
        "native_index_cast_address_to_narrow",
        "native_index_cast_address_to_predicate",
        "native_index_cast_i64_to_address",
        "native_index_cast_narrow_to_address",
        "native_index_cast_predicate_to_address",
    }


def test_vector_index_conversion_partitions_each_native_lane_interval() -> None:
    address_types = {"index", "offset"}
    expected_lane_counts: dict[tuple[str, str], set[int]] = {}
    for input_type in ("i1", "i8"):
        for result_type in address_types:
            expected_lane_counts[input_type, result_type] = {
                *range(1, 33),
                64,
            }
    for input_type in ("i16",):
        for result_type in address_types:
            expected_lane_counts[input_type, result_type] = set(range(1, 33))
    for input_type in ("i32",):
        for result_type in address_types:
            expected_lane_counts[input_type, result_type] = set(range(1, 65))
    for input_type in ("i64",):
        for result_type in address_types:
            expected_lane_counts[input_type, result_type] = set(range(1, 17))
    for input_type in address_types:
        for result_type in ("i1", "i8", "i16"):
            expected_lane_counts[input_type, result_type] = set(range(1, 33))
        expected_lane_counts[input_type, "i32"] = set(range(1, 65))
        expected_lane_counts[input_type, "i64"] = set(range(1, 17))
        for result_type in address_types:
            expected_lane_counts[input_type, result_type] = set(range(1, 65))

    actual_lane_counts = {pair: set() for pair in expected_lane_counts}
    for rule in AIE2P_VECTOR_INDEX_CONVERSION_RULES:
        pair = _cast_pair(rule)
        type_guards = {
            guard.field: guard.type_pattern
            for guard in rule.guards
            if guard.kind is GuardKind.VALUE_TYPE and guard.type_pattern is not None
        }
        input_lanes = _covered_lane_counts(type_guards["input"])
        result_lanes = _covered_lane_counts(type_guards["result"])
        assert input_lanes == result_lanes
        assert actual_lane_counts[pair].isdisjoint(input_lanes)
        actual_lane_counts[pair].update(input_lanes)
    assert actual_lane_counts == expected_lane_counts


def test_vector_index_conversion_uses_each_physical_mechanism() -> None:
    for result_type in ("index", "offset"):
        predicate_rules = _vector_rules_for("i1", result_type)
        assert predicate_rules
        for rule in predicate_rules:
            assert isinstance(rule, DescriptorRule)
            descriptor_keys = [
                emit.descriptor.key
                for emit in rule.emit
                if isinstance(emit, EmitDescriptorOp)
            ]
            assert "amd.xdna.aie2p.select.i8x64" in descriptor_keys
            assert rule.descriptor is not None
            assert rule.descriptor.key.startswith("amd.xdna.aie2p.widen.")

        i64_rules = _vector_rules_for("i64", result_type)
        assert i64_rules
        for rule in i64_rules:
            assert isinstance(rule, DescriptorRule)
            assert rule.descriptor is not None
            assert rule.descriptor.key == "amd.xdna.aie2p.shuffle.x.configured"
            if result_type == "index":
                assert all(guard.kind is GuardKind.VALUE_TYPE for guard in rule.guards)
            else:
                assert any(
                    guard.kind is GuardKind.VALUE_UNSIGNED_BIT_COUNT
                    for guard in rule.guards
                )

    for input_type in ("index", "offset"):
        predicate_rules = _vector_rules_for(input_type, "i1")
        assert predicate_rules
        for rule in predicate_rules:
            assert isinstance(rule, DescriptorRule)
            assert rule.descriptor is not None
            assert rule.descriptor.key == "amd.xdna.aie2p.cmp.lt.unsigned.i8x64"
            descriptor_keys = [
                emit.descriptor.key
                for emit in rule.emit
                if isinstance(emit, EmitDescriptorOp)
            ]
            assert "amd.xdna.aie2p.shuffle.x.configured" in descriptor_keys
            assert "amd.xdna.aie2p.and.bits512" in descriptor_keys

        i64_rules = _vector_rules_for(input_type, "i64")
        assert i64_rules
        assert all(
            isinstance(rule, DescriptorRule)
            and rule.descriptor is not None
            and rule.descriptor.key.startswith("amd.xdna.aie2p.widen.")
            for rule in i64_rules
        )


def test_vector_same_carrier_aliases_preserve_address_domain_guards() -> None:
    alias_pairs = {
        _cast_pair(rule)
        for rule in AIE2P_VECTOR_INDEX_CONVERSION_RULES
        if isinstance(rule, ValueAliasRule)
    }
    assert alias_pairs == {
        ("i32", "index"),
        ("i32", "offset"),
        ("index", "i32"),
        ("offset", "i32"),
        ("index", "index"),
        ("offset", "offset"),
        ("index", "offset"),
        ("offset", "index"),
    }
    for pair in (("index", "offset"), ("offset", "index")):
        (rule,) = _vector_rules_for(*pair)
        assert isinstance(rule, ValueAliasRule)
        range_guards = [
            guard for guard in rule.guards if guard.kind is GuardKind.VALUE_I64_RANGE
        ]
        if pair[1] == "offset":
            assert len(range_guards) == 1
            assert range_guards[0].minimum == 0
            assert range_guards[0].maximum == (2**31) - 1
        else:
            assert not range_guards
