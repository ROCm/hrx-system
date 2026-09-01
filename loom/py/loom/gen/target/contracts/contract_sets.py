# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Generator: compiled target contract sets -> immutable C tables."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from loom.gen.support.c import CIdentifierCase
from loom.gen.support.c import c_identifier as _c_identifier
from loom.gen.support.c import c_pascal_identifier as _pascal_identifier
from loom.gen.support.generated_file import line_comment_header
from loom.gen.target.contracts import lower_rule_spelling
from loom.gen.target.contracts.contract_fragments import (
    generated_contract_fragment_symbol_name,
)
from loom.target.contracts import (
    CONTRACT_ROW_NONE,
    CompiledContractSet,
    ContractFragment,
    ContractSystem,
    contract_fragment_public_header,
)

_CONTRACT_SYSTEM_C_NAMES = {
    ContractSystem.DESCRIPTOR_RULE: "LOOM_TARGET_CONTRACT_SYSTEM_DESCRIPTOR_RULE",
    ContractSystem.VALUE_ALIAS: "LOOM_TARGET_CONTRACT_SYSTEM_VALUE_ALIAS",
    ContractSystem.VALUE_ELIDE: "LOOM_TARGET_CONTRACT_SYSTEM_VALUE_ELIDE",
    ContractSystem.SOURCE_MEMORY: "LOOM_TARGET_CONTRACT_SYSTEM_SOURCE_MEMORY",
    ContractSystem.ENVIRONMENT: "LOOM_TARGET_CONTRACT_SYSTEM_ENVIRONMENT",
    ContractSystem.DESCRIPTOR_MATRIX: "LOOM_TARGET_CONTRACT_SYSTEM_DESCRIPTOR_MATRIX",
    ContractSystem.RECIPE_RULE: "LOOM_TARGET_CONTRACT_SYSTEM_RECIPE_RULE",
}


@dataclass(frozen=True, slots=True)
class ContractSetGenerationInput:
    """One compiled set and its ordered authored fragments."""

    compiled: CompiledContractSet
    fragments: tuple[ContractFragment, ...]


@dataclass(frozen=True, slots=True)
class GeneratedContractSets:
    """Generated C/H contents for a target's complete contract-set family."""

    header: str
    source: str


def generate_contract_sets(
    sets: Sequence[ContractSetGenerationInput],
    *,
    public_header: str,
) -> GeneratedContractSets:
    """Generates one immutable C table family for ordered policy sets."""

    if not sets:
        raise ValueError("contract-set generation requires at least one set")
    if not public_header:
        raise ValueError("contract-set public header must not be empty")
    for set_input in sets:
        if len(set_input.fragments) != len(set_input.compiled.bindings):
            raise ValueError(f"contract set '{set_input.compiled.name}' fragment bindings do not align")
        rule_set_indices = [binding.rule_set_index for binding in set_input.compiled.bindings if binding.rule_set_index != CONTRACT_ROW_NONE]
        if rule_set_indices != list(range(len(rule_set_indices))):
            raise ValueError(f"contract set '{set_input.compiled.name}' rule-set indices must be dense and ordered")

    return GeneratedContractSets(
        header=_generate_header(sets),
        source=_generate_source(sets, public_header=public_header),
    )


def contract_set_symbol_name(name: str) -> str:
    """Returns the public C symbol for a compiled contract set."""

    return f"loom_{_c_identifier(name).lower()}_contract_set"


def _generate_header(sets: Sequence[ContractSetGenerationInput]) -> str:
    guard_names = "_".join(set_input.compiled.name for set_input in sets)
    header_guard = "LOOM_" + _c_identifier(guard_names, case=CIdentifierCase.UPPER) + "_CONTRACT_SETS_H_"
    lines = line_comment_header(
        "//",
        generator="loom/py/loom/gen/target/contracts/contract_sets.py",
    )
    lines.extend(
        [
            "",
            f"#ifndef {header_guard}",
            f"#define {header_guard}",
            "",
            '#include "loom/codegen/low/lower/lower.h"',
            "",
            "#ifdef __cplusplus",
            'extern "C" {',
            "#endif",
            "",
        ]
    )
    for set_input in sets:
        symbol_name = contract_set_symbol_name(set_input.compiled.name)
        lines.append(f"extern const loom_low_lower_contract_set_t {symbol_name};")
    lines.extend(
        [
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
    sets: Sequence[ContractSetGenerationInput],
    *,
    public_header: str,
) -> str:
    fragment_headers: set[str] = set()
    lower_rule_headers: set[str] = set()
    for set_input in sets:
        for binding, fragment in zip(
            set_input.compiled.bindings,
            set_input.fragments,
            strict=True,
        ):
            fragment_headers.add(contract_fragment_public_header(fragment))
            if binding.rule_set_index != CONTRACT_ROW_NONE:
                lower_rule_headers.add(lower_rule_spelling.generated_public_header(fragment))

    lines = line_comment_header(
        "//",
        generator="loom/py/loom/gen/target/contracts/contract_sets.py",
    )
    lines.extend(["", f'#include "{public_header}"'])
    for header in sorted(fragment_headers | lower_rule_headers):
        lines.append(f'#include "{header}"')
    lines.append("")
    for set_input in sets:
        lines.extend(_generate_set_source(set_input))
    return "\n".join(lines)


def _generate_set_source(set_input: ContractSetGenerationInput) -> list[str]:
    table = set_input.compiled
    prefix = _pascal_identifier(table.name) + "ContractSet"
    lines: list[str] = []

    dialect_table_names: list[str | None] = []
    for dialect in table.dialects:
        if not dialect.op_entries:
            dialect_table_names.append(None)
            continue
        op_entries_name = f"k{prefix}{_pascal_identifier(dialect.dialect_name)}OpEntries"
        dialect_table_names.append(op_entries_name)
        lines.append(f"static const loom_target_contract_op_entry_t {op_entries_name}[] = {{")
        for entry in dialect.op_entries:
            case_start = "LOOM_TARGET_CONTRACT_ROW_NONE" if entry.case_start == CONTRACT_ROW_NONE else str(entry.case_start)
            lines.append(f"    {{{case_start}, {entry.case_count}}},")
        lines.extend(["};", ""])

    dialects_name = f"k{prefix}Dialects"
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

    cases_name = f"k{prefix}Cases"
    if table.cases:
        lines.append(f"static const loom_target_contract_case_t {cases_name}[] = {{")
        for contract_case in table.cases:
            system_name = _CONTRACT_SYSTEM_C_NAMES[contract_case.system]
            lines.append(f"    {{{system_name}, {contract_case.binding_index}, {contract_case.row_index}}},")
        lines.extend(["};", ""])
        cases_value = cases_name
    else:
        cases_value = "NULL"

    bindings_name = f"k{prefix}Bindings"
    if table.bindings:
        lines.append(f"static const loom_target_contract_binding_t {bindings_name}[] = {{")
        for binding_index, binding in enumerate(table.bindings):
            fragment = set_input.fragments[binding_index]
            rule_set_index = "LOOM_TARGET_CONTRACT_ROW_NONE" if binding.rule_set_index == CONTRACT_ROW_NONE else str(binding.rule_set_index)
            fragment_symbol = generated_contract_fragment_symbol_name(fragment)
            lines.append(f"    {{&{fragment_symbol}, {rule_set_index}}},")
        lines.extend(["};", ""])
        bindings_value = bindings_name
    else:
        bindings_value = "NULL"

    rule_sets_name = f"k{prefix}RuleSets"
    rule_set_binding_indices = [binding_index for binding_index, binding in enumerate(table.bindings) if binding.rule_set_index != CONTRACT_ROW_NONE]
    if rule_set_binding_indices:
        lines.append(f"static const loom_low_lower_rule_set_t* const {rule_sets_name}[] = {{")
        for binding_index in rule_set_binding_indices:
            fragment = set_input.fragments[binding_index]
            rule_set_symbol = lower_rule_spelling.generated_symbol_name(fragment)
            lines.append(f"    &{rule_set_symbol},")
        lines.extend(["};", ""])
        rule_sets_value = rule_sets_name
    else:
        rule_sets_value = "NULL"

    index_name = f"k{prefix}Index"
    lines.extend(
        [
            f"static const loom_target_contract_index_t {index_name} = {{",
            f"    {table.dialect_base_id},",
            f"    {len(table.dialects)},",
            f"    {len(table.cases)},",
            f"    {len(table.bindings)},",
            f"    {dialects_value},",
            f"    {cases_value},",
            f"    {bindings_value},",
            "};",
            "",
            f"const loom_low_lower_contract_set_t {contract_set_symbol_name(table.name)} = {{",
            "    {",
            f"        {len(rule_set_binding_indices)},",
            f"        {rule_sets_value},",
            "    },",
            f"    &{index_name},",
            "};",
            "",
        ]
    )
    return lines
