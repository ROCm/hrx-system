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
from loom.gen.support.files import write_text_file
from loom.gen.support.generated_file import line_comment_header
from loom.target.contracts import (
    CompiledContractFragment,
    CompiledDescriptorMatrix,
    CompiledDescriptorRule,
    CompiledLowerRuleSet,
    ContractFragment,
    ContractSystem,
    compile_contract_fragment,
    compile_lower_rule_set,
    contract_fragment_public_header,
)

_UINT16_MAX = 0xFFFF

_DESCRIPTOR_MATRIX_SOURCE_C_NAMES = {
    "vector_mma": "LOOM_TARGET_CONTRACT_DESCRIPTOR_MATRIX_SOURCE_VECTOR_MMA",
}


@dataclass(frozen=True, slots=True)
class GeneratedContractFragment:
    """Compiled fragment and its generated C/H contents."""

    header: str
    source: str
    compiled: CompiledContractFragment


def generate_contract_fragment(
    table: ContractFragment,
    *,
    dialect_ops: Mapping[str, Sequence[Op]],
) -> GeneratedContractFragment:
    """Generates C/H text for a compact target contract fragment."""

    lower_rules = compile_lower_rule_set(table, dialect_ops=dialect_ops)
    return generate_contract_fragment_from_lower_rules(
        table,
        dialect_ops=dialect_ops,
        lower_rules=lower_rules,
    )


def generate_contract_fragment_from_lower_rules(
    table: ContractFragment,
    *,
    dialect_ops: Mapping[str, Sequence[Op]],
    lower_rules: CompiledLowerRuleSet,
) -> GeneratedContractFragment:
    """Generates C/H text using the fragment's compiled lower-rule rows."""

    lower_rule_indices = {authored_case_index: rule_index for rule_index, authored_case_index in enumerate(lower_rules.authored_case_indices)}
    descriptor_rule_rows = {
        authored_case_index: CompiledDescriptorRule(rule_index=rule_index)
        for authored_case_index, rule_index in lower_rule_indices.items()
        if table.cases[authored_case_index].system
        in (
            ContractSystem.DESCRIPTOR_RULE,
            ContractSystem.DESCRIPTOR_RULE_LOWERING_ONLY,
        )
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
        compiled=compiled,
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
    write_text_file(header_path, generated.header)
    write_text_file(source_path, generated.source)


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
    flag_names = []
    if table.target_contract_query:
        flag_names.append("LOOM_TARGET_CONTRACT_FRAGMENT_FLAG_TARGET_QUERY")
    flags_value = " | ".join(flag_names) if flag_names else "0"

    lines.extend(
        [
            f"const loom_target_contract_fragment_t {symbol_name} = {{",
            f"    .flags = {flags_value},",
            f"    .descriptor_rule_count = {len(table.descriptor_rules)},",
            f"    .descriptor_matrix_count = {len(table.descriptor_matrices)},",
            f"    .descriptor_rules = {descriptor_rules_value},",
            f"    .descriptor_matrices = {descriptor_matrices_value},",
            "};",
            "",
        ]
    )
    return "\n".join(lines)


def _validate_c_shard_shape(table: CompiledContractFragment) -> None:
    if len(table.op_spans) > _UINT16_MAX:
        raise ValueError(f"contract fragment '{table.name}' op-span count exceeds uint16_t")
    if len(table.cases) > _UINT16_MAX:
        raise ValueError(f"contract fragment '{table.name}' case count exceeds uint16_t")
    if len(table.descriptor_rules) > _UINT16_MAX:
        raise ValueError(f"contract fragment '{table.name}' descriptor-rule count exceeds uint16_t")
    if len(table.descriptor_matrices) > _UINT16_MAX:
        raise ValueError(f"contract fragment '{table.name}' descriptor-matrix count exceeds uint16_t")
    if len(table.source_graph_op_kinds) > _UINT16_MAX:
        raise ValueError(f"contract fragment '{table.name}' source-graph op-kind count exceeds uint16_t")
    for op_span in table.op_spans:
        if op_span.op_kind > _UINT16_MAX:
            raise ValueError(f"contract fragment '{table.name}' op '{op_span.op_name}' kind exceeds uint16_t")
        if op_span.case_start + op_span.case_count > len(table.cases):
            raise ValueError(f"contract fragment '{table.name}' op '{op_span.op_name}' case span exceeds case rows")


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
