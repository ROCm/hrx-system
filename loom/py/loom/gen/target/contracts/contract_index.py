# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Private lowering-policy initializers from ordered contract fragments."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace

from loom.gen.support.c import c_identifier, c_pascal_identifier
from loom.gen.support.generated_file import line_comment_header
from loom.gen.target.contracts.lower_rule_spelling import generated_symbol_name
from loom.gen.target.contracts.lower_rules import (
    required_primary_descriptor_feature_class,
)
from loom.target.contracts import (
    CompiledContractFragment,
    ContractFragment,
    ContractSystem,
    SourceMemoryByteOffsetMaterializer,
)
from loom.target.contracts.compile import (
    CompiledContractIndex,
    CompiledContractOpEntry,
    CompiledIndexCase,
    compile_contract_index,
)
from loom.target.contracts.lower_rule_tables import CompiledLowerRuleSet
from loom.target.contracts.selection import (
    ContractSelectionIndex,
    compile_contract_selection_index,
    contract_selection_blob_words,
    pack_contract_selection_index,
)

_LOWER_RULE_SYSTEMS = frozenset(
    (
        ContractSystem.DESCRIPTOR_RULE,
        ContractSystem.DESCRIPTOR_RULE_LOWERING_ONLY,
        ContractSystem.VALUE_ALIAS,
        ContractSystem.VALUE_ELIDE,
        ContractSystem.RECIPE_RULE,
    )
)


@dataclass(frozen=True, slots=True)
class PooledContractIndex:
    """One composed index rebased onto a family-owned case pool."""

    dialect_base_id: int
    dialects: tuple[tuple[CompiledContractOpEntry, ...], ...]
    case_index_bound: int
    case_starts: dict[int, int]


class ContractCasePool:
    """Expected-linear exact-span interning across a generated table family."""

    def __init__(self, symbol: str):
        self.symbol = symbol
        self._cases: list[CompiledIndexCase] = []
        self._span_starts: dict[tuple[CompiledIndexCase, ...], int] = {}

    @property
    def cases(self) -> tuple[CompiledIndexCase, ...]:
        return tuple(self._cases)

    def pool_index(
        self,
        index: CompiledContractIndex,
        family_binding_indices: Sequence[int],
    ) -> PooledContractIndex:
        """Rebases local bindings and interns each populated operation span."""

        if any(case.binding_index >= len(family_binding_indices) for case in index.cases):
            raise ValueError("contract case binding is outside the family binding map")
        family_cases = tuple(
            CompiledIndexCase(
                case.system,
                family_binding_indices[case.binding_index],
                case.row_index,
            )
            for case in index.cases
        )
        pooled_dialects = []
        case_starts: dict[int, int] = {}
        for dialect in index.dialects:
            pooled_entries = []
            for entry in dialect:
                if entry.case_count == 0:
                    pooled_entries.append(entry)
                    continue
                case_span = family_cases[entry.case_start : entry.case_start + entry.case_count]
                pooled_start = self._span_starts.get(case_span)
                if pooled_start is None:
                    pooled_start = len(self._cases)
                    if pooled_start + len(case_span) > 0xFFFF:
                        raise ValueError("contract family case pool exceeds uint16_t")
                    self._span_starts[case_span] = pooled_start
                    self._cases.extend(case_span)
                case_starts[entry.case_start] = pooled_start
                pooled_entries.append(
                    CompiledContractOpEntry(
                        pooled_start,
                        entry.case_count,
                        entry.query_rule_count,
                    )
                )
            pooled_dialects.append(tuple(pooled_entries))
        return PooledContractIndex(
            index.dialect_base_id,
            tuple(pooled_dialects),
            len(self._cases) if index.cases else 0,
            case_starts,
        )


def generate_contract_case_pool(case_pool: ContractCasePool) -> str:
    """Emits a table family's shared composed-case storage."""

    cases = case_pool.cases
    if not cases:
        return ""
    lines = line_comment_header("//", generator="loom/py/loom/gen/target/contracts/contract_index.py")
    lines.extend(
        [
            "",
            f"static const loom_target_contract_case_t {case_pool.symbol}[] = {{",
        ]
    )
    lines.extend(f"    {{LOOM_TARGET_CONTRACT_SYSTEM_{case.system.name}, {case.binding_index}, {case.row_index}}}," for case in cases)
    lines.extend(["};", ""])
    return "\n".join(lines)


def validate_source_memory_recurrence_materializers(
    name: str,
    materializers: Iterable[tuple[str, SourceMemoryByteOffsetMaterializer]],
) -> None:
    """Requires one carried-offset recipe per physical carrier in a policy."""

    recipes: dict[
        tuple[str | None, int],
        tuple[tuple[str, str, str, str, str | None], str],
    ] = {}
    for origin, materializer in materializers:
        if materializer.increment is None:
            continue
        result = materializer.constant.operands[0]
        carrier = (result.reg_alts[0].reg_class, result.unit_count)
        recipe = (
            materializer.constant.key,
            materializer.add.key,
            materializer.increment.key,
            materializer.constant_immediate,
            materializer.increment_immediate,
        )
        previous = recipes.setdefault(carrier, (recipe, origin))
        if previous[0] != recipe:
            raise ValueError(f"{name}: source-memory recurrence carrier {carrier[0]}x{carrier[1]} has conflicting recipes in fragments '{previous[1]}' and '{origin}'")


def generate_contract_index(
    name: str,
    fragments: Sequence[
        tuple[
            ContractFragment,
            CompiledContractFragment,
            CompiledLowerRuleSet | None,
        ]
    ],
    family_binding_indices: Sequence[int],
    case_pool: ContractCasePool,
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
    if len(family_binding_indices) != len(fragments):
        raise ValueError("contract fragments and family binding indices must align")
    if len(set(family_binding_indices)) != len(family_binding_indices):
        raise ValueError("contract index contains duplicate family binding indices")
    if any(index < 0 or index >= 0xFF for index in family_binding_indices):
        raise ValueError("contract family binding slot count exceeds uint8_t")
    validate_source_memory_recurrence_materializers(
        name,
        (
            (table.name, source_memory.byte_offset_materializer)
            for table, _, lower_rules in fragments
            if lower_rules is not None
            for source_memory in lower_rules.source_memories
            if source_memory.byte_offset_materializer is not None
        ),
    )
    index = compile_contract_index([compiled for _, compiled, _ in fragments])
    semantic_selection = compile_contract_selection_index(
        index,
        [compiled for _, compiled, _ in fragments],
        [lower_rules for _, _, lower_rules in fragments],
    )
    pooled_index = case_pool.pool_index(index, family_binding_indices)
    selection = pack_contract_selection_index(ContractSelectionIndex(tuple(replace(row, case_start=pooled_index.case_starts[row.case_start]) for row in semantic_selection.rows)))
    prefix = f"k{c_pascal_identifier(name.lower())}"
    lines = line_comment_header("//", generator="loom/py/loom/gen/target/contracts/contract_index.py")
    rules = []
    binding_slot_count = max(family_binding_indices, default=-1) + 1
    binding_rows = ["{NULL, UINT16_MAX, 0}"] * binding_slot_count
    for family_binding_index, (table, compiled, lower_rules) in zip(family_binding_indices, fragments, strict=True):
        rule_set_index = 0xFFFF
        if lower_rules is not None:
            rule_set_index = len(rules)
            rules.append(f"&{generated_symbol_name(table)}")
        elif any(case.system not in (ContractSystem.DESCRIPTOR_MATRIX, ContractSystem.UNSUPPORTED) for case in compiled.cases):
            raise ValueError(f"fragment '{table.name}' requires a lower-rule pool")
        fragment_symbol = f"loom_{c_identifier(table.name).lower()}_contract_fragment"
        required_feature_class = 0
        if lower_rules is not None and len(compiled.cases) == len(lower_rules.rules) and all(case.system in _LOWER_RULE_SYSTEMS for case in compiled.cases):
            required_feature_class = required_primary_descriptor_feature_class(lower_rules)
        binding_rows[family_binding_index] = f"{{&{fragment_symbol}, {rule_set_index}, {required_feature_class}}}"
    source_graph_root_op_kinds = tuple(sorted({op_kind for _, _, lower_rules in fragments if lower_rules is not None for op_kind in lower_rules.source_graph_root_op_kinds}))
    source_graph_op_kinds = tuple(sorted({op_kind for _, compiled, _ in fragments for op_kind in compiled.source_graph_op_kinds}))
    if len(source_graph_root_op_kinds) > 0xFFFF:
        raise ValueError("contract index source-graph root count exceeds uint16_t")
    if len(source_graph_op_kinds) > 0xFFFF:
        raise ValueError("contract index source-graph op count exceeds uint16_t")

    def array(c_type: str, suffix: str, rows: Sequence[str]) -> str:
        if not rows:
            return "NULL"
        symbol = prefix + suffix
        lines.extend(["", f"static const {c_type} {symbol}[] = {{"])
        lines.extend(f"    {row}," for row in rows)
        lines.append("};")
        return symbol

    rule_sets = array("loom_low_lower_rule_set_t* const", "Rules", rules)
    binding_table = array("loom_target_contract_binding_t", "Bindings", binding_rows)
    source_graph_root_op_kind_table = array(
        "loom_op_kind_t",
        "SourceGraphRootOpKinds",
        [f"UINT16_C(0x{op_kind:04X})" for op_kind in source_graph_root_op_kinds],
    )
    source_graph_op_kind_table = array(
        "loom_op_kind_t",
        "SourceGraphOpKinds",
        [f"UINT16_C(0x{op_kind:04X})" for op_kind in source_graph_op_kinds],
    )
    cases = case_pool.symbol if pooled_index.case_index_bound else "NULL"

    dialects = []
    for ordinal, entries in enumerate(pooled_index.dialects):
        dialect_id = pooled_index.dialect_base_id + ordinal
        ops = array(
            "loom_target_contract_op_entry_t",
            f"Ops{dialect_id}",
            [f"{{{entry.case_start}, {entry.case_count}, {entry.query_rule_count}}}" for entry in entries],
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
            f"    {pooled_index.dialect_base_id}, {len(pooled_index.dialects)}, {dialect_table},",
            f"    {pooled_index.case_index_bound}, {cases}, {binding_slot_count}, {len(source_graph_root_op_kinds)}, {len(source_graph_op_kinds)},",
            f"    {binding_table}, {selection_data}, {source_graph_root_op_kind_table}, {source_graph_op_kind_table},",
            "};",
            "",
            f"#define {name} {{&{prefix}Index, {{{len(rules)}, {rule_sets}}}}}",
            "",
        ]
    )
    return "\n".join(lines)
