# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Generator: Python target contracts -> target-low lower-rule C tables."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from loom.dsl import Op
from loom.gen.support.c import c_identifier as _c_identifier
from loom.gen.support.generated_file import line_comment_header
from loom.gen.target.contracts import lower_rule_rows, lower_rule_spelling
from loom.target.contracts import (
    CompiledLowerRuleSet,
    ContractFragment,
    LowerDiagnosticParam,
    compile_lower_rule_set,
)
from loom.target.contracts.diagnostics import MAX_TARGET_DIAGNOSTIC_PARAMS


@dataclass(frozen=True, slots=True)
class GeneratedLowerRuleSet:
    """Generated C/H contents for one lower-rule set."""

    header: str
    source: str


def generate_lower_rule_set(
    table: ContractFragment,
    *,
    dialect_ops: Mapping[str, Sequence[Op]],
) -> GeneratedLowerRuleSet:
    """Generates C/H text for a generated target-low lower-rule set."""

    compiled = compile_lower_rule_set(table, dialect_ops=dialect_ops)
    public_header = lower_rule_spelling.generated_public_header(table)
    symbol_name = lower_rule_spelling.generated_symbol_name(table)
    c_table_prefix = _c_identifier(lower_rule_spelling.generated_table_prefix(table))
    header_guard = lower_rule_spelling.header_guard_from_public_header(public_header)
    return GeneratedLowerRuleSet(
        header=_generate_header(
            header_guard=header_guard,
            symbol_name=symbol_name,
        ),
        source=_generate_source(
            table=compiled,
            source_contract=table,
            public_header=public_header,
            symbol_name=symbol_name,
            c_table_prefix=c_table_prefix,
        ),
    )


def write_lower_rule_set_to_paths(
    table: ContractFragment,
    *,
    dialect_ops: Mapping[str, Sequence[Op]],
    header_path: Path,
    source_path: Path,
) -> None:
    """Writes generated C/H contents for one generated lower-rule set."""

    generated = generate_lower_rule_set(table, dialect_ops=dialect_ops)
    header_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.parent.mkdir(parents=True, exist_ok=True)
    header_path.write_text(generated.header, encoding="utf-8")
    source_path.write_text(generated.source, encoding="utf-8")


def _generate_header(*, header_guard: str, symbol_name: str) -> str:
    lines: list[str] = []
    lines.extend(
        line_comment_header(
            "//",
            generator="loom/py/loom/gen/target/contracts/lower_rules.py",
        )
    )
    lines.extend(
        [
            "",
            f"#ifndef {header_guard}",
            f"#define {header_guard}",
            "",
            '#include "loom/codegen/low/lower/lower_rules.h"',
            "",
            "#ifdef __cplusplus",
            'extern "C" {',
            "#endif",
            "",
            f"extern const loom_low_lower_rule_set_t {symbol_name};",
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
    table: CompiledLowerRuleSet,
    source_contract: ContractFragment,
    public_header: str,
    symbol_name: str,
    c_table_prefix: str,
) -> str:
    lines: list[str] = []
    lines.extend(
        line_comment_header(
            "//",
            generator="loom/py/loom/gen/target/contracts/lower_rules.py",
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
    lines.extend(f'#include "{include}"' for include in source_contract.c_source_includes)
    lines.extend(f'#include "{include}"' for include in _materializer_includes(source_contract))
    lines.extend(f'#include "{include}"' for include in _op_header_includes(table))
    lines.extend(
        [
            "",
            f"#if LOOM_LOW_LOWER_MAX_DIAGNOSTIC_PARAMS != {MAX_TARGET_DIAGNOSTIC_PARAMS}",
            '#error "target diagnostic parameter capacity mismatch"',
            "#endif",
        ]
    )
    lines.append("")

    type_patterns_name = f"k{c_table_prefix}TypePatterns"
    lines.extend(
        lower_rule_rows.emit_optional_array(
            type_patterns_name,
            "loom_low_lower_type_pattern_t",
            [lower_rule_rows.type_pattern_row(row.type_pattern) for row in table.type_patterns],
        )
    )

    value_refs_name = f"k{c_table_prefix}ValueRefs"
    lines.extend(
        lower_rule_rows.emit_optional_array(
            value_refs_name,
            "loom_low_lower_value_ref_t",
            [lower_rule_rows.value_ref_row(row) for row in table.value_refs],
        )
    )

    materializers_name = f"k{c_table_prefix}Materializers"
    lines.extend(
        lower_rule_rows.emit_optional_array(
            materializers_name,
            "loom_low_lower_value_materializer_t",
            [
                [
                    f".can_materialize = {materializer.can_materialize}",
                    f".materialize = {materializer.materialize}",
                ]
                for materializer in source_contract.materializers
            ],
        )
    )

    descriptor_ref_keys = lower_rule_rows.descriptor_ref_keys(table, source_contract)
    descriptor_refs = {key: index for index, key in enumerate(descriptor_ref_keys)}
    descriptor_refs_name = f"k{c_table_prefix}DescriptorRefs"
    lines.extend(
        lower_rule_rows.emit_optional_array(
            descriptor_refs_name,
            "loom_low_lower_rule_descriptor_ref_t",
            [lower_rule_rows.descriptor_ref_row(key) for key in descriptor_ref_keys],
        )
    )

    source_memories_name = f"k{c_table_prefix}SourceMemories"
    lines.extend(
        lower_rule_rows.emit_optional_array(
            source_memories_name,
            "loom_low_lower_source_memory_t",
            [lower_rule_rows.source_memory_row(descriptor_refs, row) for row in table.source_memories],
        )
    )

    diagnostic_param_rows: list[LowerDiagnosticParam] = []
    diagnostic_rows: list[list[str]] = []
    for row in table.diagnostics:
        param_start = len(diagnostic_param_rows)
        diagnostic_param_rows.extend(row.params)
        diagnostic_rows.append(
            [
                f".error_ref = {lower_rule_spelling.error_ref_c_expr(row.error)}",
                f".param_start = {param_start}",
                f".param_count = {len(row.params)}",
            ]
        )

    diagnostic_params_name = f"k{c_table_prefix}DiagnosticParams"
    lines.extend(
        lower_rule_rows.emit_optional_array(
            diagnostic_params_name,
            "loom_low_lower_diagnostic_param_t",
            [lower_rule_rows.diagnostic_param_row(row) for row in diagnostic_param_rows],
        )
    )

    diagnostics_name = f"k{c_table_prefix}Diagnostics"
    lines.extend(
        lower_rule_rows.emit_optional_array(
            diagnostics_name,
            "loom_low_lower_diagnostic_t",
            diagnostic_rows,
        )
    )

    guards_name = f"k{c_table_prefix}Guards"
    lines.extend(
        lower_rule_rows.emit_optional_array(
            guards_name,
            "loom_low_lower_guard_t",
            [lower_rule_rows.guard_row(descriptor_refs, row) for row in table.guards],
        )
    )

    attr_copies_name = f"k{c_table_prefix}AttrCopies"
    lines.extend(
        lower_rule_rows.emit_optional_array(
            attr_copies_name,
            "loom_low_lower_attr_copy_t",
            [lower_rule_rows.attr_copy_row(row) for row in table.attr_copies],
        )
    )

    tied_results_name = f"k{c_table_prefix}TiedResults"
    lines.extend(
        lower_rule_rows.emit_optional_array(
            tied_results_name,
            "loom_tied_result_t",
            [lower_rule_rows.tied_result_row(row) for row in table.tied_results],
        )
    )

    emits_name = f"k{c_table_prefix}Emits"
    lines.extend(
        lower_rule_rows.emit_optional_array(
            emits_name,
            "loom_low_lower_emit_t",
            [lower_rule_rows.emit_row(descriptor_refs, row) for row in table.emits],
        )
    )

    rules_name = f"k{c_table_prefix}Rules"
    lines.extend(
        lower_rule_rows.emit_optional_array(
            rules_name,
            "loom_low_lower_rule_t",
            [lower_rule_rows.rule_row(row) for row in table.rules],
        )
    )

    spans_name = f"k{c_table_prefix}Spans"
    lines.extend(
        lower_rule_rows.emit_optional_array(
            spans_name,
            "loom_low_lower_rule_span_t",
            [lower_rule_rows.span_row(row) for row in table.spans],
        )
    )

    lines.append(f"const loom_low_lower_rule_set_t {symbol_name} = {{")
    lines.extend(
        f"    {field},"
        for field in lower_rule_rows.rule_set_row(
            table=table,
            source_contract=source_contract,
            spans_name=spans_name,
            rules_name=rules_name,
            type_patterns_name=type_patterns_name,
            value_refs_name=value_refs_name,
            materializers_name=materializers_name,
            source_memories_name=source_memories_name,
            descriptor_ref_keys=descriptor_ref_keys,
            descriptor_refs_name=descriptor_refs_name,
            diagnostic_params_name=diagnostic_params_name,
            guards_name=guards_name,
            attr_copies_name=attr_copies_name,
            tied_results_name=tied_results_name,
            emits_name=emits_name,
            diagnostics_name=diagnostics_name,
        )
    )
    lines.extend(["};", ""])
    return "\n".join(lines)


def _op_header_includes(table: CompiledLowerRuleSet) -> tuple[str, ...]:
    dialect_names = {rule.source_op.group.name for rule in table.rules if rule.source_op.group is not None}
    return tuple(f"loom/ops/{name}/ops.h" for name in sorted(dialect_names))


def _materializer_includes(table: ContractFragment) -> tuple[str, ...]:
    return tuple(sorted({materializer.header for materializer in table.materializers}))
