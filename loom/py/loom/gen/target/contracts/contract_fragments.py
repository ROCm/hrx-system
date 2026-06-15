# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Generator: Python target contract fragments -> compact C ABI shards."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from loom.dsl import Op
from loom.gen.support.c import c_identifier as _c_identifier
from loom.gen.support.c import c_pascal_identifier as _pascal_identifier
from loom.gen.support.generated_file import line_comment_header
from loom.target.contracts import (
    CONTRACT_ROW_NONE,
    CompiledContractFragment,
    CompiledDescriptorMatrix,
    CompiledDescriptorRule,
    ContractFragment,
    ContractSystem,
    compile_contract_fragment,
    compile_lower_rule_set,
    contract_fragment_public_header,
)

_UINT8_MAX = 0xFF
_UINT16_MAX = 0xFFFF

_CONTRACT_SYSTEM_C_NAMES = {
    ContractSystem.DESCRIPTOR_RULE: "LOOM_TARGET_CONTRACT_SYSTEM_DESCRIPTOR_RULE",
    ContractSystem.VALUE_ALIAS: "LOOM_TARGET_CONTRACT_SYSTEM_VALUE_ALIAS",
    ContractSystem.VALUE_ELIDE: "LOOM_TARGET_CONTRACT_SYSTEM_VALUE_ELIDE",
    ContractSystem.SOURCE_MEMORY: "LOOM_TARGET_CONTRACT_SYSTEM_SOURCE_MEMORY",
    ContractSystem.ENVIRONMENT: "LOOM_TARGET_CONTRACT_SYSTEM_ENVIRONMENT",
    ContractSystem.DESCRIPTOR_MATRIX: "LOOM_TARGET_CONTRACT_SYSTEM_DESCRIPTOR_MATRIX",
}

_DESCRIPTOR_MATRIX_SOURCE_C_NAMES = {
    "vector_mma": "LOOM_TARGET_CONTRACT_DESCRIPTOR_MATRIX_SOURCE_VECTOR_MMA",
}


@dataclass(frozen=True, slots=True)
class GeneratedContractFragment:
    """Generated C/H contents for one target contract fragment."""

    header: str
    source: str


def generate_contract_fragment(
    table: ContractFragment,
    *,
    dialect_ops: Mapping[str, Sequence[Op]],
) -> GeneratedContractFragment:
    """Generates C/H text for a compact target contract fragment."""

    lower_rules = compile_lower_rule_set(table, dialect_ops=dialect_ops)
    lower_rule_indices = {authored_case_index: rule_index for rule_index, authored_case_index in enumerate(lower_rules.authored_case_indices)}
    descriptor_rule_rows = {
        authored_case_index: CompiledDescriptorRule(rule_index=rule_index)
        for authored_case_index, rule_index in lower_rule_indices.items()
        if table.cases[authored_case_index].system == ContractSystem.DESCRIPTOR_RULE
    }
    compiled = compile_contract_fragment(
        table,
        dialect_ops=dialect_ops,
        descriptor_rule_rows=descriptor_rule_rows,
        lower_rule_indices=lower_rule_indices,
    )
    _validate_c_shard_shape(compiled)
    public_header = _generated_public_header(table)
    symbol_name = _generated_symbol_name(table)
    c_table_prefix = _generated_table_prefix(table)
    header_guard = _header_guard_from_public_header(public_header)
    return GeneratedContractFragment(
        header=_generate_header(
            header_guard=header_guard,
            table=compiled,
            symbol_name=symbol_name,
        ),
        source=_generate_source(
            table=compiled,
            public_header=public_header,
            symbol_name=symbol_name,
            c_table_prefix=_c_identifier(c_table_prefix),
        ),
    )


def write_contract_fragment_to_paths(
    table: ContractFragment,
    *,
    dialect_ops: Mapping[str, Sequence[Op]],
    header_path: Path,
    source_path: Path,
) -> None:
    """Writes generated C/H contents for one target contract fragment."""

    generated = generate_contract_fragment(table, dialect_ops=dialect_ops)
    header_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.parent.mkdir(parents=True, exist_ok=True)
    header_path.write_text(generated.header, encoding="utf-8")
    source_path.write_text(generated.source, encoding="utf-8")


def _generate_header(
    *,
    header_guard: str,
    table: CompiledContractFragment,
    symbol_name: str,
) -> str:
    lines: list[str] = []
    lines.extend(
        line_comment_header(
            "//",
            generator="loom/py/loom/gen/target/contracts/contract_fragments.py",
        )
    )
    lines.extend(
        [
            "",
            f"#ifndef {header_guard}",
            f"#define {header_guard}",
            "",
            '#include "loom/target/contract.h"',
            "",
            "#ifdef __cplusplus",
            'extern "C" {',
            "#endif",
        ]
    )
    lines.extend(
        [
            f"extern const loom_target_contract_fragment_t {symbol_name};",
            "",
            "#ifdef __cplusplus",
            '}  // extern "C"',
            "#endif",
            "",
            f"#endif  // {header_guard}",
            "",
        ]
    )
    return "\n".join(lines)


def _generate_source(
    *,
    table: CompiledContractFragment,
    public_header: str,
    symbol_name: str,
    c_table_prefix: str,
) -> str:
    lines: list[str] = []
    lines.extend(
        line_comment_header(
            "//",
            generator="loom/py/loom/gen/target/contracts/contract_fragments.py",
        )
    )
    lines.extend(
        [
            "",
            f'#include "{public_header}"',
            "",
            "#include <stddef.h>",
            "",
        ]
    )

    dialect_table_names: list[str | None] = []
    for dialect in table.dialects:
        if not dialect.op_entries:
            dialect_table_names.append(None)
            continue
        array_name = f"k{c_table_prefix}{_pascal_identifier(dialect.dialect_name)}OpEntries"
        dialect_table_names.append(array_name)
        lines.append(f"static const loom_target_contract_op_entry_t {array_name}[] = {{")
        for entry in dialect.op_entries:
            case_start = "LOOM_TARGET_CONTRACT_ROW_NONE" if entry.case_start == CONTRACT_ROW_NONE else str(entry.case_start)
            lines.append(f"    {{{case_start}, {entry.case_count}}},")
        lines.extend(["};", ""])

    dialects_name = f"k{c_table_prefix}Dialects"
    if table.dialects:
        lines.append(f"static const loom_target_contract_dialect_table_t {dialects_name}[] = {{")
        for dialect, op_entries_name in zip(table.dialects, dialect_table_names, strict=True):
            if op_entries_name is None:
                lines.append("    {0, NULL},")
            else:
                lines.append(f"    {{{len(dialect.op_entries)}, {op_entries_name}}},")
        lines.extend(["};", ""])
        dialects_value = dialects_name
    else:
        dialects_value = "NULL"

    cases_name = f"k{c_table_prefix}Cases"
    if table.cases:
        lines.append(f"static const loom_target_contract_fragment_case_t {cases_name}[] = {{")
        for contract_case in table.cases:
            system_name = _CONTRACT_SYSTEM_C_NAMES[contract_case.system]
            row_index = "LOOM_TARGET_CONTRACT_ROW_NONE" if contract_case.row_index == CONTRACT_ROW_NONE else str(contract_case.row_index)
            lines.append(f"    {{{system_name}, 0, {row_index}}},")
        lines.extend(["};", ""])
        cases_value = cases_name
    else:
        cases_value = "NULL"

    descriptor_rules_name = f"k{c_table_prefix}DescriptorRules"
    if table.descriptor_rules:
        lines.append(f"static const loom_target_contract_descriptor_rule_t {descriptor_rules_name}[] = {{")
        lines.extend(f"    {{{descriptor_rule.rule_index}}}," for descriptor_rule in table.descriptor_rules)
        lines.extend(["};", ""])
        descriptor_rules_value = descriptor_rules_name
    else:
        descriptor_rules_value = "NULL"

    descriptor_matrices_name = f"k{c_table_prefix}DescriptorMatrices"
    if table.descriptor_matrices:
        lines.append(f"static const loom_target_contract_descriptor_matrix_rule_t {descriptor_matrices_name}[] = {{")
        lines.extend(_descriptor_matrix_rule_initializer(descriptor_matrix) for descriptor_matrix in table.descriptor_matrices)
        lines.extend(["};", ""])
        descriptor_matrices_value = descriptor_matrices_name
    else:
        descriptor_matrices_value = "NULL"
    flags_value = "LOOM_TARGET_CONTRACT_FRAGMENT_FLAG_TARGET_QUERY" if table.target_contract_query else "0"

    lines.extend(
        [
            f"const loom_target_contract_fragment_t {symbol_name} = {{",
            f"    {table.dialect_base_id},",
            f"    {len(table.dialects)},",
            f"    {flags_value},",
            f"    {dialects_value},",
            f"    {len(table.cases)},",
            f"    {cases_value},",
            f"    {len(table.descriptor_rules)},",
            f"    {descriptor_rules_value},",
            f"    {len(table.descriptor_matrices)},",
            f"    {descriptor_matrices_value},",
            "};",
            "",
        ]
    )
    return "\n".join(lines)


def _validate_c_shard_shape(table: CompiledContractFragment) -> None:
    if table.dialect_base_id > _UINT8_MAX:
        raise ValueError(f"contract fragment '{table.name}' dialect base exceeds uint8_t")
    if len(table.dialects) > _UINT8_MAX:
        raise ValueError(f"contract fragment '{table.name}' dialect count exceeds uint8_t")
    if len(table.cases) > _UINT16_MAX:
        raise ValueError(f"contract fragment '{table.name}' case count exceeds uint16_t")
    if len(table.descriptor_rules) > _UINT16_MAX:
        raise ValueError(f"contract fragment '{table.name}' descriptor-rule count exceeds uint16_t")
    if len(table.descriptor_matrices) > _UINT16_MAX:
        raise ValueError(f"contract fragment '{table.name}' descriptor-matrix count exceeds uint16_t")
    for dialect in table.dialects:
        if len(dialect.op_entries) > _UINT16_MAX:
            raise ValueError(f"contract fragment '{table.name}' dialect '{dialect.dialect_name}' op count exceeds uint16_t")


def _descriptor_matrix_rule_initializer(
    descriptor_matrix: CompiledDescriptorMatrix,
) -> str:
    source_name = _DESCRIPTOR_MATRIX_SOURCE_C_NAMES.get(descriptor_matrix.source)
    if source_name is None:
        raise ValueError(f"unknown descriptor-matrix source '{descriptor_matrix.source}'")
    return f"    {{{source_name}, 0, 0}},"


def _generated_public_header(table: ContractFragment) -> str:
    return contract_fragment_public_header(table)


def _generated_symbol_name(table: ContractFragment) -> str:
    return f"loom_{_c_identifier(table.name).lower()}_contract_fragment"


def _generated_table_prefix(table: ContractFragment) -> str:
    return f"{_pascal_identifier(table.name)}Contract"


def _header_guard_from_public_header(public_header: str) -> str:
    return _c_identifier(public_header).upper() + "_"
