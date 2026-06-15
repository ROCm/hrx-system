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
    ValueAliasRule,
    ValueElideRule,
)

CONTRACT_ROW_NONE = 0xFFFF


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
    dialect_base_id: int
    dialects: tuple[CompiledDialectTable, ...]
    cases: tuple[CompiledCase, ...]
    descriptor_rules: tuple[CompiledDescriptorRule, ...]
    descriptor_matrices: tuple[CompiledDescriptorMatrix, ...]


def compile_contract_fragment(
    table: ContractFragment,
    *,
    dialect_ops: Mapping[str, Sequence[Op]],
    descriptor_rule_rows: Mapping[int, CompiledDescriptorRule],
    lower_rule_indices: Mapping[int, int],
) -> CompiledContractFragment:
    """Compiles an authored contract fragment into dense target fragment rows."""

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
    sparse_dialect_tables: dict[int, CompiledDialectTable] = {}

    for dialect_name, ops in sorted(
        dialect_ops.items(),
        key=lambda item: _require_dialect(item[1], item[0]).dialect_id,
    ):
        dialect = _require_dialect(ops, dialect_name)
        op_entries: list[CompiledOpEntry] = []
        for op in ops:
            op_cases = cases_by_op.get(id(op), [])
            if not op_cases:
                op_entries.append(CompiledOpEntry())
                continue
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
            op_entries.append(
                CompiledOpEntry(case_start=case_start, case_count=len(op_cases))
            )
        if any(entry.case_count for entry in op_entries):
            sparse_dialect_tables[dialect.dialect_id] = CompiledDialectTable(
                dialect_id=dialect.dialect_id,
                dialect_name=dialect.name,
                op_entries=tuple(op_entries),
            )

    dialect_base_id = min(sparse_dialect_tables) if sparse_dialect_tables else 0
    dialect_limit = max(sparse_dialect_tables) + 1 if sparse_dialect_tables else 0
    dialect_tables = tuple(
        sparse_dialect_tables.get(
            dialect_id,
            CompiledDialectTable(
                dialect_id=dialect_id,
                dialect_name="",
                op_entries=(),
            ),
        )
        for dialect_id in range(dialect_base_id, dialect_limit)
    )

    return CompiledContractFragment(
        name=table.name,
        target_contract_query=table.target_contract_query,
        dialect_base_id=dialect_base_id,
        dialects=dialect_tables,
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
    if isinstance(contract_case, ValueAliasRule):
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
