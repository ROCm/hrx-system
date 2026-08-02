# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Structural coverage tests for SPIR-V index conversions."""

from __future__ import annotations

from collections import Counter

from loom.dialect.index import defs as index
from loom.target.arch.spirv.contracts.index import SPIRV_INDEX_CONVERSION_RULES
from loom.target.contracts import ContractCase, GuardKind, OrdinalValueAliasRule


def _value_type_element(rule: ContractCase, field: str) -> str:
    matches = [
        guard.type_pattern.element
        for guard in rule.guards
        if guard.kind == GuardKind.VALUE_TYPE and guard.field == field
    ]
    assert len(matches) == 1, (
        f"expected one type guard for {rule.source_op.name}.{field}, got {matches}"
    )
    assert matches[0] is not None
    return matches[0]


def test_index_conversion_contract_contains_only_boundary_operations() -> None:
    actual = Counter(rule.source_op.name for rule in SPIRV_INDEX_CONVERSION_RULES)
    expected = Counter(
        {
            index.index_constant.name: 2,
            index.index_cast.name: 24,
            index.index_assume.name: 1,
        }
    )

    assert actual == expected


def test_index_cast_contract_covers_every_address_boundary_pair() -> None:
    scalar_types = {"i1", "i8", "i16", "i32", "i64", "index", "offset"}
    address_types = {"index", "offset"}
    expected = {
        (source_type, result_type)
        for source_type in scalar_types
        for result_type in scalar_types
        if source_type in address_types or result_type in address_types
    }
    actual = {
        (_value_type_element(rule, "input"), _value_type_element(rule, "result"))
        for rule in SPIRV_INDEX_CONVERSION_RULES
        if rule.source_op is index.index_cast
    }

    assert actual == expected


def test_index_assume_uses_ordinal_aliasing() -> None:
    assume_rules = [
        rule
        for rule in SPIRV_INDEX_CONVERSION_RULES
        if rule.source_op is index.index_assume
    ]

    assert len(assume_rules) == 1
    assert isinstance(assume_rules[0], OrdinalValueAliasRule)
