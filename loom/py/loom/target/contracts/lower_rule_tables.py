# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Compact target-Low rule-table representation and packing."""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum, unique

from loom.dsl import Op
from loom.errors import ErrorDef
from loom.target.contracts.diagnostics import (
    DiagnosticParamKind,
)
from loom.target.contracts.guards import GuardKind
from loom.target.contracts.kinds import SourceValueKind
from loom.target.contracts.patterns import TypePattern
from loom.target.contracts.rules import (
    SourceNodeRelation,
)
from loom.target.contracts.source_memory import (
    SourceMemoryAddressMaterializer,
    SourceMemoryByteOffsetMaterializer,
    SourceMemoryConstraint,
)
from loom.target.low_descriptors import (
    Descriptor,
)


@unique
class LowerEmitKind(Enum):
    """Interpreter emit operation used by a compiled lower-rule row."""

    DESCRIPTOR_OP = "descriptor_op"
    DESCRIPTOR_CONST = "descriptor_const"
    DESCRIPTOR_OP_FIRST_LANE = "descriptor_op_first_lane"
    DESCRIPTOR_OP_PER_LANE = "descriptor_op_per_lane"
    DESCRIPTOR_OP_PER_LANE_SEQUENCE = "descriptor_op_per_lane_sequence"
    DESCRIPTOR_OP_ACCUMULATE_LANES = "descriptor_op_accumulate_lanes"
    REGISTER_SLICE = "register_slice"
    REGISTER_CONCAT = "register_concat"
    REGISTER_COPY = "register_copy"


@unique
class LowerAttrCopyKind(Enum):
    """Interpreter attribute-copy operation used by a compiled emit row."""

    DIRECT = "direct"
    ENUM_ORDINAL = "enum_ordinal"
    I64_ARRAY_ELEMENT = "i64_array_element"
    I64_ARRAY_ELEMENT_PLUS_LITERAL = "i64_array_element_plus_literal"
    I64_ARRAY_PACK_ELEMENTS = "i64_array_pack_elements"
    I64_ATTRS_PACK_CONSECUTIVE = "i64_attrs_pack_consecutive"
    I64_LITERAL = "i64_literal"
    VALUE_EXACT_I64 = "value_exact_i64"
    VALUE_EXACT_I64_I32_WORD = "value_exact_i64_i32_word"
    VALUE_EXACT_I64_NEGATE = "value_exact_i64_negate"
    VALUE_EXACT_I64_LOG2 = "value_exact_i64_log2"
    VALUE_EXACT_I64_MINUS_ONE = "value_exact_i64_minus_one"
    VALUE_U32_DIVISOR_MAGIC_MULTIPLIER = "value_u32_divisor_magic_multiplier"
    VALUE_U32_DIVISOR_MAGIC_SHIFT = "value_u32_divisor_magic_shift"
    VALUE_I32_AS_U32_BITS = "value_i32_as_u32_bits"
    VALUE_FLOAT_AS_F16_BITS = "value_float_as_f16_bits"
    VALUE_FLOAT_AS_BF16_BITS = "value_float_as_bf16_bits"
    VALUE_FLOAT_AS_F32_BITS = "value_float_as_f32_bits"
    VALUE_FLOAT_AS_F32_I32 = "value_float_as_f32_i32"
    VALUE_FLOAT_AS_F64_BITS = "value_float_as_f64_bits"
    VALUE_FLOAT_AS_F64_I32_WORD = "value_float_as_f64_i32_word"
    I64_ARRAY_LANE_BYTE = "i64_array_lane_byte"
    SOURCE_MEMORY_STATIC_BYTE_OFFSET = "source_memory_static_byte_offset"
    SOURCE_MEMORY_STATIC_BYTE_OFFSET_PLUS_LITERAL = (
        "source_memory_static_byte_offset_plus_literal"
    )
    SOURCE_MEMORY_STATIC_BYTE_OFFSET_QUOTIENT = (
        "source_memory_static_byte_offset_quotient"
    )
    SOURCE_MEMORY_STATIC_BYTE_OFFSET_REMAINDER = (
        "source_memory_static_byte_offset_remainder"
    )
    SOURCE_MEMORY_DYNAMIC_BYTE_STRIDE = "source_memory_dynamic_byte_stride"
    SOURCE_OP_INSTANCE_FLAGS = "source_op_instance_flags"
    I64_LOW_BIT_MASK = "i64_low_bit_mask"
    I64_SHIFTED_LOW_BIT_MASK = "i64_shifted_low_bit_mask"
    I64_SHIFTED_LOW_BIT_CLEAR_MASK = "i64_shifted_low_bit_clear_mask"
    I64_LITERAL_MINUS_ATTR = "i64_literal_minus_attr"
    I64_LITERAL_MINUS_ATTRS = "i64_literal_minus_attrs"
    I64_ATTR_MINUS_LITERAL = "i64_attr_minus_literal"


LOWER_EMIT_FLAG_SWAP_OPERANDS_0_1 = 1 << 0
LOWER_EMIT_FLAG_BIND_RESULTS_TO_REFS = 1 << 1
LOWER_EMIT_FLAG_RESULT_TYPE_PATTERN = 1 << 2
LOWER_EMIT_FLAG_ACCUMULATE_SEED_FIRST_LANE = 1 << 3
LOWER_EMIT_FLAG_ACCUMULATE_TREE_BALANCED = 1 << 4
LOWER_EMIT_FLAG_ACCUMULATE_SKIP_FIRST_LANE = 1 << 5
LOWER_EMIT_FLAG_RESULT_DESCRIPTOR_TYPE = 1 << 6
LOWER_SOURCE_MEMORY_NONE = 0
LOWER_RULE_FLAG_CONTRACT_ONLY = 1 << 0
LOWER_RULE_FLAG_ORDINAL_VALUE_ALIAS = 1 << 1

_LOW_VALUE_GUARD_KINDS = (
    GuardKind.LOW_VALUE_REGISTER_CLASS,
    GuardKind.LOW_VALUE_REGISTER_UNIT_COUNT,
    GuardKind.VALUE_STATIC_DIM0_MULTIPLE,
    GuardKind.LOW_VALUE_REGISTER_UNIT_COUNT_EQ,
)


@dataclass(frozen=True, slots=True)
class LowerTypePattern:
    """Compiled type-pattern row."""

    type_pattern: TypePattern


@dataclass(frozen=True, slots=True)
class LowerValueRef:
    """Compiled source value-reference row."""

    kind: SourceValueKind
    index: int
    source_node_index: int = 0
    element_index: int = 0
    materializer_index: int = 0


@dataclass(frozen=True, slots=True)
class LowerSourceNode:
    """Compiled source-op relation row."""

    relation: SourceNodeRelation
    source_op: Op
    parent_node_index: int
    parent_value_ref_index: int
    node_value_ref_index: int
    guard_start: int
    guard_count: int


@dataclass(frozen=True, slots=True)
class LowerDiagnosticParam:
    """Compiled parameter projection for a rejection diagnostic."""

    name: str
    kind: DiagnosticParamKind
    string_value: str = ""
    value_ref_index: int = 0
    i64_value: int = 0
    u32_value: int = 0
    u64_value: int = 0
    bool_value: bool = False


@dataclass(frozen=True, slots=True)
class LowerDiagnostic:
    """Compiled rejection diagnostic row."""

    error: ErrorDef
    params: tuple[LowerDiagnosticParam, ...]
    target_context_param_count: int = 0


@dataclass(frozen=True, slots=True)
class LowerSourceMemory:
    """Compiled source-memory constraint row."""

    constraint: SourceMemoryConstraint
    diagnostic_index: int
    dynamic_offset_diagnostic_index: int
    address_layout_diagnostic_index: int = 0xFFFF
    address_diagnostic_index: int = 0xFFFF
    byte_offset_materializer: SourceMemoryByteOffsetMaterializer | None = None
    address_materializer: SourceMemoryAddressMaterializer | None = None


@dataclass(frozen=True, slots=True)
class LowerGuard:
    """Compiled selection guard row."""

    kind: GuardKind
    value_ref_index: int = 0
    other_value_ref_index: int = 0
    attr_index: int = 0
    type_pattern_index: int = 0
    diagnostic_index: int = 0xFFFF
    attr_kind: str | None = None
    u64: int = 0
    u64_c_expression: str | None = None
    memory_spaces: tuple[str, ...] = ()
    descriptor: Descriptor | None = None
    register_class_id: int = 0
    minimum_i64: int = 0
    maximum_i64: int = 0


@dataclass(frozen=True, slots=True)
class LowerAttrCopy:
    """Compiled source-attribute projection row."""

    kind: LowerAttrCopyKind
    target_name: str
    source_attr_index: int = 0
    other_source_attr_index: int = 0
    source_element_index: int = 0
    source_element_count: int = 0
    source_element_bit_width: int = 0
    target_bit_offset: int = 0
    value_ref_index: int = 0
    literal_i64: int = 0
    dynamic_term_index: int = 0


@dataclass(frozen=True, slots=True)
class LowerTiedResult:
    """Compiled result-to-operand tie row."""

    result_index: int
    operand_index: int
    has_type_change: bool = False


@dataclass(frozen=True, slots=True)
class LowerEmit:
    """Compiled emit-program row."""

    kind: LowerEmitKind
    descriptor: Descriptor | None = None
    flags: int = 0
    operand_ref_start: int = 0
    operand_ref_count: int = 0
    copy_operand_mask: int = 0
    accumulator_operand_index: int = 0
    result_ref_start: int = 0
    result_type_pattern_start: int = 0
    result_ref_count: int = 0
    result_bind_ref_start: int = 0
    attr_copy_start: int = 0
    attr_copy_count: int = 0
    tied_result_start: int = 0
    tied_result_count: int = 0
    source_memory_ordinal: int = LOWER_SOURCE_MEMORY_NONE
    structural_offset: int = 0
    structural_unit_count: int = 0


@dataclass(frozen=True, slots=True)
class LowerRule:
    """Compiled lowering rule row."""

    source_op: Op
    temporary_count: int
    guard_start: int
    guard_count: int
    emit_start: int
    emit_count: int
    source_node_start: int = 0
    source_node_count: int = 0
    flags: int = 0
    alias_ref_start: int = 0
    alias_ref_count: int = 0
    elide_ref_start: int = 0
    elide_ref_count: int = 0
    report_key: str = ""


@dataclass(frozen=True, slots=True)
class LowerRuleSpan:
    """Compiled op-kind to rule-range row."""

    source_op: Op
    rule_start: int
    rule_count: int


@dataclass(frozen=True, slots=True)
class CompiledLowerRuleSet:
    """Lower-rule set rows generated from a contract fragment."""

    name: str
    authored_case_indices: tuple[int, ...]
    rules: tuple[LowerRule, ...]
    spans: tuple[LowerRuleSpan, ...]
    type_patterns: tuple[LowerTypePattern, ...]
    value_refs: tuple[LowerValueRef, ...]
    source_nodes: tuple[LowerSourceNode, ...]
    source_memories: tuple[LowerSourceMemory, ...]
    guards: tuple[LowerGuard, ...]
    attr_copies: tuple[LowerAttrCopy, ...]
    tied_results: tuple[LowerTiedResult, ...]
    emits: tuple[LowerEmit, ...]
    diagnostics: tuple[LowerDiagnostic, ...]


def _intern_program_rows[ProgramRowT: Hashable](
    rows: Sequence[ProgramRowT],
    ranges: Iterable[tuple[int, int]],
) -> tuple[tuple[ProgramRowT, ...], tuple[int, ...]]:
    """Interns exact immutable row programs and returns their new starts."""

    interned_rows: list[ProgramRowT] = []
    program_starts: dict[tuple[ProgramRowT, ...], int] = {(): 0}
    rewritten_starts: list[int] = []
    for start, count in ranges:
        program = tuple(rows[start : start + count])
        interned_start = program_starts.get(program)
        if interned_start is None:
            interned_start = len(interned_rows)
            program_starts[program] = interned_start
            interned_rows.extend(program)
        rewritten_starts.append(interned_start)
    return tuple(interned_rows), tuple(rewritten_starts)


def _append_interned_row_sequence[RowT](
    rows: list[RowT], sequence: tuple[RowT, ...]
) -> int:
    """Appends a row sequence unless an identical span already exists."""

    if not sequence:
        return 0
    sequence_count = len(sequence)
    for start in range(len(rows) - sequence_count + 1):
        if tuple(rows[start : start + sequence_count]) == sequence:
            return start
    ordinal = len(rows)
    rows.extend(sequence)
    return ordinal


def _build_spans(
    rules: list[LowerRule],
    op_ordinals: dict[int, int],
) -> tuple[LowerRuleSpan, ...]:
    spans: list[LowerRuleSpan] = []
    i = 0
    while i < len(rules):
        rule_start = i
        rule = rules[i]
        rule_count = 1
        while (
            i + rule_count < len(rules)
            and rules[i + rule_count].source_op is rule.source_op
        ):
            rule_count += 1
        spans.append(
            LowerRuleSpan(
                source_op=rule.source_op,
                rule_start=rule_start,
                rule_count=rule_count,
            )
        )
        i += rule_count
    return tuple(
        sorted(spans, key=lambda span: _op_kind_key(span.source_op, op_ordinals))
    )


def _build_op_ordinals(dialect_ops: Mapping[str, Sequence[Op]]) -> dict[int, int]:
    op_ordinals: dict[int, int] = {}
    for ops in dialect_ops.values():
        for op_index, op in enumerate(ops):
            op_ordinals[id(op)] = op_index
    return op_ordinals


def _op_kind_key(op: Op, op_ordinals: dict[int, int]) -> tuple[int, int]:
    if op.group is None:
        raise ValueError(f"op '{op.name}' is not assigned to a dialect")
    try:
        op_index = op_ordinals[id(op)]
    except KeyError as exc:
        raise ValueError(f"op '{op.name}' is not present in dialect_ops") from exc
    return (op.group.dialect_id, op_index)
