# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Compilation from authored target contracts to compact fragment rows."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import NamedTuple

from loom.dsl import Dialect, Op
from loom.target.contracts.fragments import ContractFragment
from loom.target.contracts.kinds import ContractSystem
from loom.target.contracts.rules import (
    ContractCase,
    DescriptorMatrixRule,
    DescriptorRule,
    OrdinalValueAliasRule,
    RecipeRule,
    UnsupportedRule,
    ValueAliasRule,
    ValueElideRule,
    contract_case_priority,
)

CONTRACT_ROW_NONE = 0xFFFF


@dataclass(frozen=True, slots=True)
class CompiledOpSpan:
    """Populated source-op case span."""

    op_kind: int
    op_name: str
    case_start: int
    case_count: int


@dataclass(frozen=True, slots=True)
class CompiledCase:
    """Compiled generic case row."""

    system: ContractSystem
    row_index: int = CONTRACT_ROW_NONE
    priority: int = 0


@dataclass(frozen=True, slots=True)
class CompiledDescriptorRule:
    """Compiled fragment-local descriptor-rule row."""

    rule_index: int


@dataclass(frozen=True, slots=True)
class CompiledDescriptorMatrix:
    """Compiled fragment-local descriptor-matrix row."""

    source: str


@dataclass(frozen=True, slots=True)
class CompiledContractFragment:
    """Compact contract fragment ready for C emission."""

    name: str
    target_contract_query: bool
    op_spans: tuple[CompiledOpSpan, ...]
    cases: tuple[CompiledCase, ...]
    descriptor_rules: tuple[CompiledDescriptorRule, ...]
    descriptor_matrices: tuple[CompiledDescriptorMatrix, ...]


class CompiledIndexCase(NamedTuple):
    """Case row retaining its owning fragment's ordinal and local row."""

    system: ContractSystem
    binding_index: int
    row_index: int


class CompiledContractIndex(NamedTuple):
    """Dense dialect/op lookup with cases in binding precedence order."""

    dialect_base_id: int
    dialects: tuple[tuple[tuple[int, int], ...], ...]
    cases: tuple[CompiledIndexCase, ...]


def compile_contract_index(
    fragments: Sequence[CompiledContractFragment],
) -> CompiledContractIndex:
    """Composes immutable fragments once, preserving each op's rule order.

    Dialect and op holes have empty spans. Binding order breaks ties between
    fragments; case order within each fragment is unchanged. Compact C field
    limits are checked here instead of during function lowering.
    """
    if len(fragments) > 0xFF:
        raise ValueError("contract index binding count exceeds uint8_t")
    cases_by_op: dict[int, list[tuple[int, CompiledIndexCase]]] = {}
    for binding_index, fragment in enumerate(fragments):
        for span in fragment.op_spans:
            cases_by_op.setdefault(span.op_kind, []).extend(
                (
                    case.priority,
                    CompiledIndexCase(case.system, binding_index, case.row_index),
                )
                for case in fragment.cases[
                    span.case_start : span.case_start + span.case_count
                ]
            )
    if not cases_by_op:
        return CompiledContractIndex(0, (), ())
    dialect_base_id = min(cases_by_op) >> 8
    dialect_count = (max(cases_by_op) >> 8) - dialect_base_id + 1
    if dialect_count > 0xFF:
        raise ValueError("contract index dialect span exceeds uint8_t")
    dialects: list[list[tuple[int, int]]] = [[] for _ in range(dialect_count)]
    cases: list[CompiledIndexCase] = []
    for op_kind, op_cases in sorted(cases_by_op.items()):
        dialect = dialects[(op_kind >> 8) - dialect_base_id]
        op_index = op_kind & 0xFF
        dialect.extend([(CONTRACT_ROW_NONE, 0)] * (op_index + 1 - len(dialect)))
        op_cases.sort(key=lambda item: -item[0])
        dialect[op_index] = (len(cases), len(op_cases))
        cases.extend(case for _, case in op_cases)
    if len(cases) > 0xFFFF:
        raise ValueError("contract index case count exceeds uint16_t")
    return CompiledContractIndex(
        dialect_base_id, tuple(tuple(dialect) for dialect in dialects), tuple(cases)
    )


def compile_contract_fragment(
    table: ContractFragment,
    *,
    dialect_ops: Mapping[str, Sequence[Op]],
    descriptor_rule_rows: Mapping[int, CompiledDescriptorRule],
    lower_rule_indices: Mapping[int, int],
) -> CompiledContractFragment:
    """Compiles an authored contract fragment into compact target fragment rows."""

    op_indexes = _build_op_indexes(dialect_ops)
    cases_by_op: dict[int, list[tuple[int, ContractCase]]] = {}
    descriptor_rule_ordinals: dict[int, int] = {}
    descriptor_rules: list[CompiledDescriptorRule] = []
    descriptor_matrix_ordinals: dict[str, int] = {}
    descriptor_matrices: list[CompiledDescriptorMatrix] = []
    for authored_case_index, contract_case in enumerate(table.cases):
        _require_op_index(op_indexes, contract_case.source_op)
        if isinstance(contract_case, DescriptorRule):
            descriptor_rule = descriptor_rule_rows.get(authored_case_index)
            if descriptor_rule is None:
                continue
            descriptor_rule_index = len(descriptor_rules)
            descriptor_rule_ordinals[authored_case_index] = descriptor_rule_index
            descriptor_rules.append(descriptor_rule)
        elif isinstance(contract_case, DescriptorMatrixRule):
            if contract_case.source not in descriptor_matrix_ordinals:
                descriptor_matrix_ordinals[contract_case.source] = len(
                    descriptor_matrices
                )
                descriptor_matrices.append(
                    CompiledDescriptorMatrix(source=contract_case.source)
                )
        cases_by_op.setdefault(id(contract_case.source_op), []).append(
            (authored_case_index, contract_case)
        )

    compiled_cases: list[CompiledCase] = []
    op_spans: list[CompiledOpSpan] = []
    sorted_op_cases = sorted(
        cases_by_op.items(),
        key=lambda item: (
            op_indexes[item[0]][0].dialect_id,
            op_indexes[item[0]][1],
        ),
    )
    for op_identity, op_cases in sorted_op_cases:
        dialect, op_index = op_indexes[op_identity]
        source_op = op_cases[0][1].source_op
        op_cases.sort(key=lambda item: (-contract_case_priority(item[1]), item[0]))
        case_start = len(compiled_cases)
        for authored_case_index, contract_case in op_cases:
            compiled_case = _compile_case(
                contract_case,
                descriptor_rule_index=descriptor_rule_ordinals.get(
                    authored_case_index,
                    CONTRACT_ROW_NONE,
                ),
                lower_rule_index=lower_rule_indices.get(
                    authored_case_index,
                    CONTRACT_ROW_NONE,
                ),
                descriptor_matrix_index=(
                    descriptor_matrix_ordinals[contract_case.source]
                    if isinstance(contract_case, DescriptorMatrixRule)
                    else CONTRACT_ROW_NONE
                ),
            )
            compiled_cases.append(compiled_case)
        op_spans.append(
            CompiledOpSpan(
                op_kind=(dialect.dialect_id << 8) | op_index,
                op_name=source_op.name,
                case_start=case_start,
                case_count=len(op_cases),
            )
        )

    return CompiledContractFragment(
        name=table.name,
        target_contract_query=table.target_contract_query,
        op_spans=tuple(op_spans),
        cases=tuple(compiled_cases),
        descriptor_rules=tuple(descriptor_rules),
        descriptor_matrices=tuple(descriptor_matrices),
    )


def _compile_case(
    contract_case: ContractCase,
    *,
    descriptor_rule_index: int,
    lower_rule_index: int,
    descriptor_matrix_index: int,
) -> CompiledCase:
    priority = contract_case_priority(contract_case)
    if isinstance(contract_case, DescriptorRule):
        return CompiledCase(
            system=ContractSystem.DESCRIPTOR_RULE,
            row_index=descriptor_rule_index,
            priority=priority,
        )
    if isinstance(contract_case, (ValueAliasRule, OrdinalValueAliasRule)):
        if lower_rule_index == CONTRACT_ROW_NONE:
            raise ValueError(
                f"{contract_case.source_op.name}: value-alias case has no "
                "compiled lower rule"
            )
        return CompiledCase(
            system=ContractSystem.VALUE_ALIAS,
            row_index=lower_rule_index,
            priority=priority,
        )
    if isinstance(contract_case, ValueElideRule):
        if lower_rule_index == CONTRACT_ROW_NONE:
            raise ValueError(
                f"{contract_case.source_op.name}: value-elide case has no "
                "compiled lower rule"
            )
        return CompiledCase(
            system=ContractSystem.VALUE_ELIDE,
            row_index=lower_rule_index,
            priority=priority,
        )
    if isinstance(contract_case, RecipeRule):
        if lower_rule_index == CONTRACT_ROW_NONE:
            raise ValueError(
                f"{contract_case.source_op.name}: recipe case has no "
                "compiled lower rule"
            )
        return CompiledCase(
            system=ContractSystem.RECIPE_RULE,
            row_index=lower_rule_index,
            priority=priority,
        )
    if isinstance(contract_case, UnsupportedRule):
        return CompiledCase(
            system=ContractSystem.UNSUPPORTED,
            priority=priority,
        )
    if isinstance(contract_case, DescriptorMatrixRule):
        if descriptor_matrix_index == CONTRACT_ROW_NONE:
            raise ValueError(
                f"{contract_case.source_op.name}: descriptor-matrix case has "
                "no compiled row"
            )
        return CompiledCase(
            system=ContractSystem.DESCRIPTOR_MATRIX,
            row_index=descriptor_matrix_index,
            priority=priority,
        )
    raise TypeError(f"unsupported contract case {contract_case!r}")


def _build_op_indexes(
    dialect_ops: Mapping[str, Sequence[Op]],
) -> dict[int, tuple[Dialect, int]]:
    indexes: dict[int, tuple[Dialect, int]] = {}
    for dialect_name, ops in dialect_ops.items():
        dialect = _require_dialect(ops, dialect_name)
        for op_index, op in enumerate(ops):
            op_identity = id(op)
            if op_identity in indexes:
                raise ValueError(f"op '{op.name}' appears in multiple dialect tables")
            if op.group != dialect:
                raise ValueError(
                    f"op '{op.name}' does not belong to dialect '{dialect.name}'"
                )
            indexes[op_identity] = (dialect, op_index)
    return indexes


def _require_op_index(indexes: Mapping[int, tuple[Dialect, int]], op: Op) -> None:
    if id(op) not in indexes:
        raise ValueError(f"op '{op.name}' is not present in dialect_ops")


def _require_dialect(ops: Sequence[Op], dialect_name: str) -> Dialect:
    if not ops:
        raise ValueError(f"dialect_ops['{dialect_name}'] must not be empty")
    dialect = ops[0].group
    if dialect is None:
        raise ValueError(f"dialect_ops['{dialect_name}'] contains ungrouped ops")
    if dialect.name != dialect_name:
        raise ValueError(
            f"dialect_ops key '{dialect_name}' does not match dialect '{dialect.name}'"
        )
    return dialect
