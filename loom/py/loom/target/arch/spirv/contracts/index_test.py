# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Structural coverage tests for SPIR-V index contract categories."""

from __future__ import annotations

from collections import Counter

from loom.dialect.index import defs as index
from loom.dialect.scf import defs as scf
from loom.target.arch.spirv.contracts.index import (
    SPIRV_INDEX_CONVERSION_RULES,
    SPIRV_INDEX_NUMERIC_RULES,
)
from loom.target.arch.spirv.contracts.logical_core import (
    SPIRV_LOGICAL_CORE_CONTRACT_FRAGMENT,
)
from loom.target.arch.spirv.scalar_alu import INTEGER_COMPARE_PREDICATES
from loom.target.contracts import (
    ContractCase,
    DescriptorResultType,
    DescriptorRule,
    GuardKind,
    OrdinalValueAliasRule,
    SourceValueKind,
)


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


def _enum_keyword(rule: ContractCase, field: str) -> str:
    matches = [
        guard.enum_keyword
        for guard in rule.guards
        if guard.kind == GuardKind.ENUM_ATTR_EQUALS and guard.field == field
    ]
    assert len(matches) == 1, (
        f"expected one enum guard for {rule.source_op.name}.{field}, got {matches}"
    )
    assert matches[0] is not None
    return matches[0]


def _numeric_rule_key(rule: ContractCase) -> tuple[str, str, str]:
    if rule.source_op is index.index_cmp:
        return (
            rule.source_op.name,
            _value_type_element(rule, "lhs"),
            _enum_keyword(rule, "predicate"),
        )
    if rule.source_op is scf.scf_select:
        return (
            rule.source_op.name,
            _value_type_element(rule, "true_value"),
            "",
        )
    for field in ("lhs", "index", "a", "input"):
        if any(
            guard.kind == GuardKind.VALUE_TYPE and guard.field == field
            for guard in rule.guards
        ):
            return rule.source_op.name, _value_type_element(rule, field), ""
    raise AssertionError(f"numeric rule {rule.source_op.name} has no value type key")


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


def test_index_i32_casts_preserve_the_semantic_type_boundary() -> None:
    rules = [
        rule
        for rule in SPIRV_INDEX_CONVERSION_RULES
        if rule.source_op is index.index_cast
        and {
            _value_type_element(rule, "input"),
            _value_type_element(rule, "result"),
        }
        == {"i32", "index"}
    ]

    assert len(rules) == 2
    assert all(isinstance(rule, DescriptorRule) for rule in rules)
    assert all(rule.descriptor.key == "spirv.op_copy_object.i32" for rule in rules)


def test_spirv_unsigned_temporaries_use_descriptor_result_types() -> None:
    invalid_bindings = []
    for case in SPIRV_LOGICAL_CORE_CONTRACT_FRAGMENT.cases:
        if not isinstance(case, DescriptorRule):
            continue
        for emit in case.emit:
            for descriptor_field, value_ref in emit.results.items():
                if (
                    value_ref.kind != SourceValueKind.TEMPORARY
                    or not value_ref.field.startswith("unsigned_")
                ):
                    continue
                result_types = emit.result_types or {}
                if not isinstance(
                    result_types.get(descriptor_field), DescriptorResultType
                ):
                    invalid_bindings.append(
                        f"{case.source_op.name}:{emit.descriptor.key}:"
                        f"{descriptor_field}={value_ref.field}"
                    )

    assert not invalid_bindings


def test_index_assume_uses_ordinal_aliasing() -> None:
    assume_rules = [
        rule
        for rule in SPIRV_INDEX_CONVERSION_RULES
        if rule.source_op is index.index_assume
    ]

    assert len(assume_rules) == 1
    assert isinstance(assume_rules[0], OrdinalValueAliasRule)


def test_index_numeric_contract_covers_the_complete_address_matrix() -> None:
    expected: Counter[tuple[str, str, str]] = Counter()
    for source_op in (index.index_add, index.index_sub):
        expected[(source_op.name, "index", "")] = 1
        expected[(source_op.name, "offset", "")] = 1
    for source_op in (
        index.index_mul,
        index.index_scale,
        index.index_div,
        index.index_rem,
        index.index_min,
        index.index_max,
        index.index_madd,
        index.index_andi,
        index.index_ori,
        index.index_xori,
        index.index_shli,
        index.index_shrsi,
        index.index_shrui,
        index.index_rotli,
        index.index_rotri,
        index.index_ctlzi,
        index.index_cttzi,
        index.index_ctpopi,
    ):
        expected[(source_op.name, "index", "")] = 1
    for predicate in INTEGER_COMPARE_PREDICATES:
        expected[(index.index_cmp.name, "index", predicate.source_predicate)] = 1
        expected[(index.index_cmp.name, "offset", predicate.source_predicate)] = 1
    expected[(scf.scf_select.name, "index", "")] = 1
    expected[(scf.scf_select.name, "offset", "")] = 1

    actual = Counter(_numeric_rule_key(rule) for rule in SPIRV_INDEX_NUMERIC_RULES)

    assert len(SPIRV_INDEX_NUMERIC_RULES) == 44
    assert actual == expected


def test_index_contract_accounts_for_every_registered_index_operation() -> None:
    covered_ops = {
        rule.source_op
        for rule in (*SPIRV_INDEX_CONVERSION_RULES, *SPIRV_INDEX_NUMERIC_RULES)
        if rule.source_op is not scf.scf_select
    }

    assert covered_ops == set(index.ALL_INDEX_OPS)


def test_index_numeric_range_guards_have_spirv_diagnostics() -> None:
    guarded_ops = {
        index.index_scale,
        index.index_div,
        index.index_rem,
        index.index_shli,
        index.index_shrsi,
        index.index_shrui,
        index.index_rotli,
        index.index_rotri,
    }
    range_guards = [
        guard
        for rule in SPIRV_INDEX_NUMERIC_RULES
        if rule.source_op in guarded_ops
        for guard in rule.guards
        if guard.kind == GuardKind.VALUE_I64_RANGE
    ]

    assert len(range_guards) == 10
    assert all(guard.diagnostic is not None for guard in range_guards)
    assert all(
        guard.diagnostic.ref is not None
        and guard.diagnostic.ref.error.error_id == "ERR_SPIRV_027"
        for guard in range_guards
        if guard.diagnostic is not None
    )
