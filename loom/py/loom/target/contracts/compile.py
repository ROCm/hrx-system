# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Compilation from authored target contracts to compact fragment rows."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from loom.dsl import Dialect, Op
from loom.target.contracts.fragments import ContractFragment
from loom.target.contracts.kinds import ContractSystem
from loom.target.contracts.rules import (
    ContractCase,
    DescriptorMatrixRule,
    DescriptorRule,
    OrdinalValueAliasRule,
    RecipeRule,
    ValueAliasRule,
    ValueElideRule,
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
class CompiledOpEntry:
    """Dense dialect-local op table entry."""

    case_start: int = CONTRACT_ROW_NONE
    case_count: int = 0


@dataclass(frozen=True, slots=True)
class CompiledDialectTable:
    """Compiled table for one source dialect."""

    dialect_id: int
    dialect_name: str
    op_entries: tuple[CompiledOpEntry, ...]


@dataclass(frozen=True, slots=True)
class CompiledCase:
    """Compiled generic case row."""

    system: ContractSystem
    row_index: int = CONTRACT_ROW_NONE


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


@dataclass(frozen=True, slots=True)
class CompiledContractSetBinding:
    """One ordered fragment binding in a compiled contract set."""

    rule_set_index: int = CONTRACT_ROW_NONE


@dataclass(frozen=True, slots=True)
class CompiledContractSetCase:
    """One contract case projected into a policy-wide dense index."""

    system: ContractSystem
    binding_index: int
    row_index: int = CONTRACT_ROW_NONE


@dataclass(frozen=True, slots=True)
class CompiledContractSet:
    """Immutable policy contract set ready for C emission."""

    name: str
    dialect_base_id: int
    dialects: tuple[CompiledDialectTable, ...]
    cases: tuple[CompiledContractSetCase, ...]
    bindings: tuple[CompiledContractSetBinding, ...]


def compile_contract_set(
    name: str,
    fragments: Sequence[CompiledContractFragment],
) -> CompiledContractSet:
    """Composes ordered fragments into one immutable policy contract set."""

    if not name:
        raise ValueError("contract set name must not be empty")
    if not fragments:
        raise ValueError(f"contract set '{name}' must contain a fragment")
    if len(fragments) > 0xFF:
        raise ValueError(f"contract set '{name}' binding count exceeds uint8_t")

    bindings: list[CompiledContractSetBinding] = []
    rule_set_count = 0
    lower_rule_systems = {
        ContractSystem.DESCRIPTOR_RULE,
        ContractSystem.VALUE_ALIAS,
        ContractSystem.VALUE_ELIDE,
        ContractSystem.RECIPE_RULE,
    }
    for fragment in fragments:
        rule_set_index = CONTRACT_ROW_NONE
        if any(case.system in lower_rule_systems for case in fragment.cases):
            rule_set_index = rule_set_count
            rule_set_count += 1
        bindings.append(
            CompiledContractSetBinding(
                rule_set_index=rule_set_index,
            )
        )

    dialect_names: dict[int, str] = {}
    dialect_op_counts: dict[int, int] = {}
    cases_by_op: dict[tuple[int, int], list[CompiledContractSetCase]] = {}
    for binding_index, fragment in enumerate(fragments):
        for op_span in fragment.op_spans:
            dialect_id = op_span.op_kind >> 8
            op_index = op_span.op_kind & 0xFF
            dialect_name, separator, _ = op_span.op_name.partition(".")
            if not separator:
                raise ValueError(
                    f"contract set '{name}' op '{op_span.op_name}' has no "
                    "dialect-qualified name"
                )
            existing_name = dialect_names.setdefault(dialect_id, dialect_name)
            if existing_name != dialect_name:
                raise ValueError(
                    f"contract set '{name}' dialect {dialect_id} has "
                    f"conflicting names '{existing_name}' and "
                    f"'{dialect_name}'"
                )
            dialect_op_counts[dialect_id] = max(
                dialect_op_counts.get(dialect_id, 0),
                op_index + 1,
            )
            case_limit = op_span.case_start + op_span.case_count
            if case_limit > len(fragment.cases):
                raise ValueError(
                    f"contract set '{name}' op '{op_span.op_name}' case span "
                    "exceeds fragment case rows"
                )
            op_cases = cases_by_op.setdefault((dialect_id, op_index), [])
            for fragment_case in fragment.cases[op_span.case_start : case_limit]:
                op_cases.append(
                    CompiledContractSetCase(
                        system=fragment_case.system,
                        binding_index=binding_index,
                        row_index=fragment_case.row_index,
                    )
                )

    dialect_base_id = min(dialect_op_counts) if dialect_op_counts else 0
    dialect_limit = max(dialect_op_counts) + 1 if dialect_op_counts else 0
    if dialect_limit - dialect_base_id > 0xFF:
        raise ValueError(f"contract set '{name}' dialect span exceeds uint8_t")

    compiled_cases: list[CompiledContractSetCase] = []
    dialects: list[CompiledDialectTable] = []
    for dialect_id in range(dialect_base_id, dialect_limit):
        op_entries: list[CompiledOpEntry] = []
        for op_index in range(dialect_op_counts.get(dialect_id, 0)):
            op_cases = cases_by_op.get((dialect_id, op_index), ())
            if not op_cases:
                op_entries.append(CompiledOpEntry())
                continue
            if len(op_cases) > 0xFFFF:
                raise ValueError(
                    f"contract set '{name}' dialect {dialect_id} op "
                    f"{op_index} case count exceeds uint16_t"
                )
            case_start = len(compiled_cases)
            if case_start + len(op_cases) > 0xFFFF:
                raise ValueError(f"contract set '{name}' case count exceeds uint16_t")
            compiled_cases.extend(op_cases)
            op_entries.append(
                CompiledOpEntry(
                    case_start=case_start,
                    case_count=len(op_cases),
                )
            )
        dialects.append(
            CompiledDialectTable(
                dialect_id=dialect_id,
                dialect_name=dialect_names.get(dialect_id, ""),
                op_entries=tuple(op_entries),
            )
        )

    return CompiledContractSet(
        name=name,
        dialect_base_id=dialect_base_id,
        dialects=tuple(dialects),
        cases=tuple(compiled_cases),
        bindings=tuple(bindings),
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
    if isinstance(contract_case, DescriptorRule):
        return CompiledCase(
            system=ContractSystem.DESCRIPTOR_RULE,
            row_index=descriptor_rule_index,
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
