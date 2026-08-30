# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for AMD XDNA AIE2P address-boundary conversion rules."""

from loom.dialect.index import defs as index
from loom.target.arch.amd.xdna.aie2p.contracts.index_conversion import (
    AIE2P_INDEX_CONVERSION_RULES,
)
from loom.target.contracts import (
    DescriptorRule,
    EmitRegisterConcat,
    EmitRegisterSlice,
    GuardKind,
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
    assert any(
        guard.kind is GuardKind.VALUE_SIGNED_BIT_COUNT and guard.count == 32
        for guard in to_index.guards
    )
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
        assert len(range_guards) == 1
        assert range_guards[0].minimum == 0
        assert range_guards[0].maximum == (2**31) - 1
