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
    LOWER_EMIT_FLAG_BIND_RESULTS_TO_REFS,
    LOWER_EMIT_FLAG_RESULT_DESCRIPTOR_TYPE,
    LOWER_EMIT_FLAG_RESULT_TYPE_PATTERN,
    LOWER_SOURCE_MEMORY_NONE,
    CompiledLowerRuleSet,
    ContractFragment,
    GuardKind,
    LowerDiagnosticParam,
    LowerEmitKind,
    TypePattern,
    compile_lower_rule_set,
)
from loom.target.contracts.diagnostics import (
    MAX_TARGET_DIAGNOSTIC_PARAMS,
    DiagnosticParamKind,
)

_I64_MIN = -(2**63)
_I64_MAX = 2**63 - 1
_U8_MAX = 0xFF
_U16_MAX = 0xFFFF
_U32_MAX = 0xFFFF_FFFF
_U64_MAX = 0xFFFF_FFFF_FFFF_FFFF


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
    return generate_lower_rule_set_from_compiled(table, compiled=compiled)


def generate_lower_rule_set_from_compiled(
    table: ContractFragment,
    *,
    compiled: CompiledLowerRuleSet,
) -> GeneratedLowerRuleSet:
    """Generates C/H text from compiled target-low lower-rule rows."""

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
    report_keys = _collect_report_keys(table)
    _validate_c_table_shape(table, source_contract, descriptor_ref_keys)
    descriptor_refs = {key: index for index, key in enumerate(descriptor_ref_keys)}
    report_key_ordinals = {key: index + 1 for index, key in enumerate(report_keys)}
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

    report_keys_name = f"k{c_table_prefix}ReportKeys"
    lines.extend(
        lower_rule_rows.emit_optional_array(
            report_keys_name,
            "iree_string_view_t",
            [lower_rule_rows.report_key_row(key) for key in report_keys],
        )
    )

    rules_name = f"k{c_table_prefix}Rules"
    lines.extend(
        lower_rule_rows.emit_optional_array(
            rules_name,
            "loom_low_lower_rule_t",
            [lower_rule_rows.rule_row(row, report_key_ordinals) for row in table.rules],
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
            report_keys=report_keys,
            report_keys_name=report_keys_name,
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


def _validate_c_table_shape(
    table: CompiledLowerRuleSet,
    source_contract: ContractFragment,
    descriptor_ref_keys: tuple[str, ...],
) -> None:
    """Validates that compiled rows fit the compact C table ABI."""

    subject = f"lower-rule set '{table.name}'"
    _require_u16(len(table.spans), f"{subject} span count")
    _require_u16(len(table.rules), f"{subject} rule count")
    _require_u16(len(_collect_report_keys(table)), f"{subject} report-key count")
    _require_u16(len(table.type_patterns), f"{subject} type-pattern count")
    _require_u16(len(table.value_refs), f"{subject} value-ref count")
    _require_u16(
        len(source_contract.materializers),
        f"{subject} materializer count",
    )
    _require_u16(len(table.source_memories), f"{subject} source-memory count")
    _require_u16(len(descriptor_ref_keys), f"{subject} descriptor-ref count")
    _require_u16(len(table.guards), f"{subject} guard count")
    _require_u16(len(table.attr_copies), f"{subject} attr-copy count")
    _require_u16(len(table.tied_results), f"{subject} tied-result count")
    _require_u16(len(table.emits), f"{subject} emit count")
    _require_u16(len(table.diagnostics), f"{subject} diagnostic count")

    diagnostic_param_count = 0
    for index, row in enumerate(table.diagnostics):
        diagnostic_subject = f"{subject} diagnostic {index}"
        _require_u16(
            diagnostic_param_count,
            f"{diagnostic_subject} param start",
        )
        param_count = len(row.params)
        _require_target_diagnostic_param_count(
            param_count,
            f"{diagnostic_subject} param count",
        )
        for param_index, param in enumerate(row.params):
            param_subject = f"{diagnostic_subject} param {param_index}"
            _require_u16(param.value_ref_index, f"{param_subject} value-ref index")
            if param.kind == DiagnosticParamKind.VALUE_TYPE:
                _require_table_index(
                    param.value_ref_index,
                    len(table.value_refs),
                    f"{param_subject} value-ref index",
                    "value-ref",
                )
            _require_i64(param.i64_value, f"{param_subject} i64 value")
            _require_u32(param.u32_value, f"{param_subject} u32 value")
            _require_u64(param.u64_value, f"{param_subject} u64 value")
        diagnostic_param_count += param_count
    _require_u16(diagnostic_param_count, f"{subject} diagnostic-param count")

    for index, row in enumerate(table.type_patterns):
        _validate_type_pattern_c_shape(
            f"{subject} type-pattern {index}",
            row.type_pattern,
        )

    for index, row in enumerate(table.value_refs):
        row_subject = f"{subject} value-ref {index}"
        _require_u16(row.index, f"{row_subject} index")
        _require_u16(row.element_index, f"{row_subject} element index")
        _require_u16(row.materializer_index, f"{row_subject} materializer index")
        if row.materializer_index:
            _require_one_based_table_index(
                row.materializer_index,
                len(source_contract.materializers),
                f"{row_subject} materializer index",
                "materializer",
            )

    for index, row in enumerate(table.source_memories):
        row_subject = f"{subject} source-memory {index}"
        constraint = row.constraint
        _require_u32(
            constraint.element_byte_count,
            f"{row_subject} element byte count",
        )
        _require_u32(
            constraint.vector_lane_count,
            f"{row_subject} vector lane count",
        )
        _require_i64(
            constraint.vector_lane_byte_stride,
            f"{row_subject} vector lane byte stride",
        )
        _require_i64(
            constraint.static_byte_offset_minimum,
            f"{row_subject} static byte offset minimum",
        )
        _require_i64(
            constraint.static_byte_offset_maximum,
            f"{row_subject} static byte offset maximum",
        )
        _require_u32(
            constraint.minimum_alignment,
            f"{row_subject} minimum alignment",
        )
        dynamic_term_count = constraint.dynamic_term_count
        if dynamic_term_count is not None:
            _require_u8_not_reserved_any(
                dynamic_term_count,
                f"{row_subject} dynamic term count",
            )
        _require_u8_not_reserved_any(
            constraint.dynamic_term_count_minimum,
            f"{row_subject} minimum dynamic term count",
        )
        if constraint.dynamic_view_base_term_count is not None:
            _require_u8_not_reserved_any(
                constraint.dynamic_view_base_term_count,
                f"{row_subject} dynamic view-base term count",
            )
        if constraint.dynamic_byte_stride is not None:
            _require_i64(
                constraint.dynamic_byte_stride,
                f"{row_subject} dynamic byte stride",
            )
        _require_u8(
            constraint.dynamic_offset_unsigned_bit_count,
            f"{row_subject} dynamic offset unsigned bit count",
        )
        _require_u16(
            row.dynamic_offset_diagnostic_index,
            f"{row_subject} dynamic-offset diagnostic index",
        )
        _require_optional_table_index(
            row.dynamic_offset_diagnostic_index,
            len(table.diagnostics),
            f"{row_subject} dynamic-offset diagnostic index",
            "diagnostic",
        )
        _require_u32(
            constraint.cache_policy_build_flags,
            f"{row_subject} cache policy build flags",
        )
        _require_u16(row.diagnostic_index, f"{row_subject} diagnostic index")
        _require_optional_table_index(
            row.diagnostic_index,
            len(table.diagnostics),
            f"{row_subject} diagnostic index",
            "diagnostic",
        )

    for index, row in enumerate(table.guards):
        row_subject = f"{subject} guard {index}"
        _require_u16(row.value_ref_index, f"{row_subject} value-ref index")
        _require_u16(
            row.other_value_ref_index,
            f"{row_subject} other value-ref index",
        )
        if lower_rule_rows.guard_uses_value_ref(row.kind):
            _require_table_index(
                row.value_ref_index,
                len(table.value_refs),
                f"{row_subject} value-ref index",
                "value-ref",
            )
        if lower_rule_rows.guard_uses_other_value_ref(row.kind):
            _require_table_index(
                row.other_value_ref_index,
                len(table.value_refs),
                f"{row_subject} other value-ref index",
                "value-ref",
            )
        _require_u16(row.attr_index, f"{row_subject} attr index")
        _require_u16(row.type_pattern_index, f"{row_subject} type-pattern index")
        if row.kind == GuardKind.VALUE_TYPE:
            _require_table_index(
                row.type_pattern_index,
                len(table.type_patterns),
                f"{row_subject} type-pattern index",
                "type-pattern",
            )
        _require_u16(row.diagnostic_index, f"{row_subject} diagnostic index")
        _require_optional_table_index(
            row.diagnostic_index,
            len(table.diagnostics),
            f"{row_subject} diagnostic index",
            "diagnostic",
        )
        _require_u64(row.u64, f"{row_subject} u64 payload")
        _require_u16(row.register_class_id, f"{row_subject} register class id")
        _require_i64(row.minimum_i64, f"{row_subject} minimum i64")
        _require_i64(row.maximum_i64, f"{row_subject} maximum i64")

    for index, row in enumerate(table.attr_copies):
        row_subject = f"{subject} attr-copy {index}"
        _require_u16(row.source_attr_index, f"{row_subject} source attr index")
        _require_u16(
            row.other_source_attr_index,
            f"{row_subject} other source attr index",
        )
        _require_u16(
            row.source_element_index,
            f"{row_subject} source element index",
        )
        _require_u16(
            row.source_element_count,
            f"{row_subject} source element count",
        )
        _require_u8(
            row.source_element_bit_width,
            f"{row_subject} source element bit width",
        )
        _require_u8(row.target_bit_offset, f"{row_subject} target bit offset")
        _require_u16(row.value_ref_index, f"{row_subject} value-ref index")
        if lower_rule_rows.attr_copy_uses_value_ref(row.kind):
            _require_table_index(
                row.value_ref_index,
                len(table.value_refs),
                f"{row_subject} value-ref index",
                "value-ref",
            )
        _require_u8(row.dynamic_term_index, f"{row_subject} dynamic term index")
        _require_i64(row.literal_i64, f"{row_subject} literal i64")

    for index, row in enumerate(table.tied_results):
        row_subject = f"{subject} tied-result {index}"
        _require_u16(row.result_index, f"{row_subject} result index")
        _require_u16(row.operand_index, f"{row_subject} operand index")

    for index, row in enumerate(table.emits):
        row_subject = f"{subject} emit {index}"
        _require_u16(row.flags, f"{row_subject} flags")
        _require_u16(row.operand_ref_start, f"{row_subject} operand-ref start")
        _require_u16(row.operand_ref_count, f"{row_subject} operand-ref count")
        _require_table_range(
            row.operand_ref_start,
            row.operand_ref_count,
            len(table.value_refs),
            f"{row_subject} operand-ref range",
            "value-ref",
        )
        _require_u16(row.copy_operand_mask, f"{row_subject} copy operand mask")
        if row.operand_ref_count < 16:
            allowed_operand_mask = (1 << row.operand_ref_count) - 1
            if row.copy_operand_mask & ~allowed_operand_mask:
                raise ValueError(f"{row_subject} copy operand mask references an operand outside operand-ref range: {row.copy_operand_mask}")
        _require_u16(
            row.accumulator_operand_index,
            f"{row_subject} accumulator operand index",
        )
        if row.kind == LowerEmitKind.DESCRIPTOR_OP_ACCUMULATE_LANES and row.accumulator_operand_index >= row.operand_ref_count:
            raise ValueError(f"{row_subject} accumulator operand index is outside operand-ref range: {row.accumulator_operand_index}")
        _require_u16(row.result_ref_start, f"{row_subject} result-ref start")
        _require_u16(
            row.result_type_pattern_start,
            f"{row_subject} result type-pattern start",
        )
        _require_u16(row.result_ref_count, f"{row_subject} result-ref count")
        if row.flags & LOWER_EMIT_FLAG_RESULT_TYPE_PATTERN and row.flags & LOWER_EMIT_FLAG_RESULT_DESCRIPTOR_TYPE:
            raise ValueError(f"{row_subject} cannot use both result type-pattern and descriptor result-type flags")
        if row.flags & LOWER_EMIT_FLAG_RESULT_TYPE_PATTERN:
            _require_table_range(
                row.result_type_pattern_start,
                row.result_ref_count,
                len(table.type_patterns),
                f"{row_subject} result type-pattern range",
                "type-pattern",
            )
        else:
            _require_table_range(
                row.result_ref_start,
                row.result_ref_count,
                len(table.value_refs),
                f"{row_subject} result-ref range",
                "value-ref",
            )
        _require_u16(
            row.result_bind_ref_start,
            f"{row_subject} result bind-ref start",
        )
        if row.flags & LOWER_EMIT_FLAG_BIND_RESULTS_TO_REFS:
            _require_table_range(
                row.result_bind_ref_start,
                row.result_ref_count,
                len(table.value_refs),
                f"{row_subject} result bind-ref range",
                "value-ref",
            )
        _require_u16(row.attr_copy_start, f"{row_subject} attr-copy start")
        _require_u16(row.attr_copy_count, f"{row_subject} attr-copy count")
        _require_table_range(
            row.attr_copy_start,
            row.attr_copy_count,
            len(table.attr_copies),
            f"{row_subject} attr-copy range",
            "attr-copy",
        )
        _require_u16(row.tied_result_start, f"{row_subject} tied-result start")
        _require_u16(row.tied_result_count, f"{row_subject} tied-result count")
        _require_table_range(
            row.tied_result_start,
            row.tied_result_count,
            len(table.tied_results),
            f"{row_subject} tied-result range",
            "tied-result",
        )
        _require_u16(
            row.source_memory_ordinal,
            f"{row_subject} source-memory ordinal",
        )
        if row.source_memory_ordinal != LOWER_SOURCE_MEMORY_NONE:
            _require_one_based_table_index(
                row.source_memory_ordinal,
                len(table.source_memories),
                f"{row_subject} source-memory ordinal",
                "source-memory",
            )

    for index, row in enumerate(table.rules):
        row_subject = f"{subject} rule {index}"
        if row.report_key:
            _require_report_key(row.report_key, f"{row_subject} report key")
        _require_u16(row.temporary_count, f"{row_subject} temporary count")
        _require_u16(row.guard_start, f"{row_subject} guard start")
        _require_u16(row.guard_count, f"{row_subject} guard count")
        _require_table_range(
            row.guard_start,
            row.guard_count,
            len(table.guards),
            f"{row_subject} guard range",
            "guard",
        )
        _require_u16(row.emit_start, f"{row_subject} emit start")
        _require_u16(row.emit_count, f"{row_subject} emit count")
        _require_table_range(
            row.emit_start,
            row.emit_count,
            len(table.emits),
            f"{row_subject} emit range",
            "emit",
        )
        _require_u16(row.alias_ref_start, f"{row_subject} alias-ref start")
        _require_u16(row.alias_ref_count, f"{row_subject} alias-ref count")
        _require_table_range(
            row.alias_ref_start,
            row.alias_ref_count * 2,
            len(table.value_refs),
            f"{row_subject} alias-ref range",
            "value-ref",
        )
        _require_u16(row.elide_ref_start, f"{row_subject} elide-ref start")
        _require_u16(row.elide_ref_count, f"{row_subject} elide-ref count")
        _require_table_range(
            row.elide_ref_start,
            row.elide_ref_count,
            len(table.value_refs),
            f"{row_subject} elide-ref range",
            "value-ref",
        )

    for index, row in enumerate(table.spans):
        row_subject = f"{subject} span {index}"
        _require_u16(row.rule_start, f"{row_subject} rule start")
        _require_u16(row.rule_count, f"{row_subject} rule count")
        _require_table_range(
            row.rule_start,
            row.rule_count,
            len(table.rules),
            f"{row_subject} rule range",
            "rule",
        )
        for rule_index in range(row.rule_start, row.rule_start + row.rule_count):
            rule = table.rules[rule_index]
            if rule.source_op is not row.source_op:
                raise ValueError(f"{row_subject} rule range contains rule {rule_index} for source op '{rule.source_op.name}', expected '{row.source_op.name}'")


def _validate_type_pattern_c_shape(subject: str, type_pattern: TypePattern) -> None:
    if type_pattern.kind not in {"vector", "view"}:
        return
    _require_u8(len(type_pattern.dims), f"{subject} rank")
    for index, dim in enumerate(type_pattern.dims):
        _require_i64(dim, f"{subject} static dim {index}")
    if type_pattern.kind == "view":
        return
    if type_pattern.lanes is not None:
        _require_i64(type_pattern.lanes, f"{subject} static lanes")
    if isinstance(type_pattern.minimum_lanes, int):
        _require_i64(type_pattern.minimum_lanes, f"{subject} minimum lanes")
    if isinstance(type_pattern.maximum_lanes, int):
        _require_i64(type_pattern.maximum_lanes, f"{subject} maximum lanes")


def _require_u8(value: int, subject: str) -> None:
    if not 0 <= value <= _U8_MAX:
        raise ValueError(f"{subject} exceeds uint8_t: {value}")


def _require_u8_not_reserved_any(value: int, subject: str) -> None:
    if not 0 <= value < _U8_MAX:
        raise ValueError(f"{subject} exceeds uint8_t reserved-any range: {value}")


def _require_target_diagnostic_param_count(value: int, subject: str) -> None:
    if value > MAX_TARGET_DIAGNOSTIC_PARAMS:
        raise ValueError(f"{subject} exceeds target diagnostic param capacity: {value}")
    _require_u8(value, subject)


def _require_u16(value: int, subject: str) -> None:
    if not 0 <= value <= _U16_MAX:
        raise ValueError(f"{subject} exceeds uint16_t: {value}")


def _require_u32(value: int, subject: str) -> None:
    if not 0 <= value <= _U32_MAX:
        raise ValueError(f"{subject} exceeds uint32_t: {value}")


def _require_u64(value: int, subject: str) -> None:
    if not 0 <= value <= _U64_MAX:
        raise ValueError(f"{subject} exceeds uint64_t: {value}")


def _require_i64(value: int, subject: str) -> None:
    if not _I64_MIN <= value <= _I64_MAX:
        raise ValueError(f"{subject} exceeds int64_t: {value}")


def _require_report_key(value: str, subject: str) -> None:
    if any(char.isspace() for char in value):
        raise ValueError(f"{subject} must not contain whitespace: {value!r}")
    if value != value.strip(".") or ".." in value:
        raise ValueError(f"{subject} must not have empty segments: {value!r}")


def _require_table_index(
    index: int,
    row_count: int,
    subject: str,
    table_name: str,
) -> None:
    if 0 <= index < row_count:
        return
    raise ValueError(f"{subject} references missing {table_name} row: index={index} rows={row_count}")


def _require_one_based_table_index(
    ordinal: int,
    row_count: int,
    subject: str,
    table_name: str,
) -> None:
    if 1 <= ordinal <= row_count:
        return
    raise ValueError(f"{subject} references missing {table_name} row: ordinal={ordinal} rows={row_count}")


def _require_optional_table_index(
    index: int,
    row_count: int,
    subject: str,
    table_name: str,
) -> None:
    if index == 0xFFFF:
        return
    _require_table_index(index, row_count, subject, table_name)


def _require_table_range(
    start: int,
    count: int,
    row_count: int,
    subject: str,
    table_name: str,
) -> None:
    if start < 0 or count < 0:
        raise ValueError(f"{subject} has negative {table_name} range: start={start} count={count}")
    if start > row_count or start + count > row_count:
        raise ValueError(f"{subject} exceeds {table_name} table: start={start} count={count} rows={row_count}")


def _op_header_includes(table: CompiledLowerRuleSet) -> tuple[str, ...]:
    dialect_names = {rule.source_op.group.name for rule in table.rules if rule.source_op.group is not None}
    return tuple(f"loom/ops/{name}/ops.h" for name in sorted(dialect_names))


def _materializer_includes(table: ContractFragment) -> tuple[str, ...]:
    return tuple(sorted({materializer.header for materializer in table.materializers}))


def _collect_report_keys(table: CompiledLowerRuleSet) -> tuple[str, ...]:
    report_keys: list[str] = []
    seen: set[str] = set()
    for rule in table.rules:
        if not rule.report_key or rule.report_key in seen:
            continue
        report_keys.append(rule.report_key)
        seen.add(rule.report_key)
    return tuple(report_keys)
