# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Private lowering-policy initializers from ordered contract fragments."""

from __future__ import annotations

from collections.abc import Sequence

from loom.gen.support.c import c_identifier, c_pascal_identifier
from loom.gen.support.generated_file import line_comment_header
from loom.gen.target.contracts.lower_rule_spelling import generated_symbol_name
from loom.target.contracts import CompiledContractFragment, ContractFragment, ContractSystem
from loom.target.contracts.compile import compile_contract_index
from loom.target.contracts.lower_rule_tables import CompiledLowerRuleSet
from loom.target.contracts.selection import (
    compile_contract_selection_index,
    contract_selection_blob_words,
    pack_contract_selection_index,
)


def generate_contract_index(
    name: str,
    fragments: Sequence[
        tuple[
            ContractFragment,
            CompiledContractFragment,
            CompiledLowerRuleSet | None,
        ]
    ],
    selection_blob_symbols: dict[tuple[int, ...], str] | None = None,
) -> str:
    """Emits one policy's lookup and paired rule-pool initializer.

    The build supplies the C initializer name and fragment precedence. A
    fragment with no compiled lower rules contributes only descriptor-matrix
    metadata, with no invented ordinary lowering-rule pool. Identical selection
    blobs may be interned across the generated table family through
    |selection_blob_symbols|.
    """
    if c_identifier(name) != name or not name:
        raise ValueError(f"invalid contract initializer name '{name}'")
    index = compile_contract_index([compiled for _, compiled, _ in fragments])
    selection = pack_contract_selection_index(
        compile_contract_selection_index(
            index,
            [compiled for _, compiled, _ in fragments],
            [lower_rules for _, _, lower_rules in fragments],
        )
    )
    prefix = f"k{c_pascal_identifier(name.lower())}"
    lines = line_comment_header("//", generator="loom/py/loom/gen/target/contracts/contract_index.py")
    rules = []
    bindings = []
    for table, compiled, lower_rules in fragments:
        rule_set_index = 0xFFFF
        if lower_rules is not None:
            rule_set_index = len(rules)
            rules.append(f"&{generated_symbol_name(table)}")
        elif any(case.system not in (ContractSystem.DESCRIPTOR_MATRIX, ContractSystem.UNSUPPORTED) for case in compiled.cases):
            raise ValueError(f"fragment '{table.name}' requires a lower-rule pool")
        fragment_symbol = f"loom_{c_identifier(table.name).lower()}_contract_fragment"
        bindings.append(f"{{&{fragment_symbol}, {rule_set_index}}}")

    def array(c_type: str, suffix: str, rows: Sequence[str]) -> str:
        if not rows:
            return "NULL"
        symbol = prefix + suffix
        lines.extend(["", f"static const {c_type} {symbol}[] = {{"])
        lines.extend(f"    {row}," for row in rows)
        lines.append("};")
        return symbol

    rule_sets = array("loom_low_lower_rule_set_t* const", "Rules", rules)
    binding_table = array("loom_target_contract_binding_t", "Bindings", bindings)
    cases = array(
        "loom_target_contract_case_t",
        "Cases",
        [f"{{LOOM_TARGET_CONTRACT_SYSTEM_{case.system.name}, {case.binding_index}, {case.row_index}}}" for case in index.cases],
    )

    dialects = []
    for ordinal, entries in enumerate(index.dialects):
        dialect_id = index.dialect_base_id + ordinal
        ops = array(
            "loom_target_contract_op_entry_t",
            f"Ops{dialect_id}",
            [f"{{{start}, {count}}}" for start, count in entries],
        )
        dialects.append(f"{{{len(entries)}, {ops}}}")
    dialect_table = array("loom_target_contract_dialect_table_t", "Dialects", dialects)

    selection_data = "NULL"
    if selection.rows:
        selection_words = contract_selection_blob_words(selection)
        selection_blob_symbols = selection_blob_symbols if selection_blob_symbols is not None else {}
        selection_data = selection_blob_symbols.get(selection_words, "")
        if not selection_data:
            selection_data = prefix + "SelectionData"
            selection_blob_symbols[selection_words] = selection_data
            lines.extend(["", f"static const uint32_t {selection_data}[] = {{"])
            for word_start in range(0, len(selection_words), 8):
                words = selection_words[word_start : word_start + 8]
                lines.append("    " + " ".join(f"UINT32_C(0x{word:08X})," for word in words))
            lines.append("};")
    lines.extend(
        [
            "",
            f"static const loom_target_contract_index_t {prefix}Index = {{",
            f"    {index.dialect_base_id}, {len(index.dialects)}, {dialect_table},",
            f"    {len(index.cases)}, {cases}, {len(bindings)}, {binding_table},",
            f"    {selection_data},",
            "};",
            "",
            f"#define {name} {{&{prefix}Index, {{{len(rules)}, {rule_sets}}}}}",
            "",
        ]
    )
    return "\n".join(lines)
