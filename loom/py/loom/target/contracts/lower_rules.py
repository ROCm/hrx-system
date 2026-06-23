# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Compilation from descriptor-rule contracts to target-low lowering rows."""

from __future__ import annotations

import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum, unique

from loom.dsl import ATTR_TYPE_ENUM, Op
from loom.error.target import (
    ERR_TARGET_002,
    ERR_TARGET_003,
    ERR_TARGET_004,
    ERR_TARGET_005,
    ERR_TARGET_006,
    ERR_TARGET_007,
    ERR_TARGET_008,
)
from loom.errors import ErrorDef
from loom.target.contracts.diagnostics import (
    DiagnosticParam,
    DiagnosticParamKind,
    DiagnosticRef,
    i64_param,
    string_param,
    target_diagnostic,
    u32_param,
    value_type_param,
)
from loom.target.contracts.emits import (
    DescriptorAccumulatorSeed,
    DescriptorAccumulatorTree,
    DescriptorEmitForm,
    DescriptorResultType,
    EmitDescriptorOp,
)
from loom.target.contracts.fragments import ContractFragment
from loom.target.contracts.guards import Guard, GuardKind
from loom.target.contracts.immediates import (
    AttrProject,
    AttrProjectKind,
    SourceMemoryProject,
    SourceMemoryProjectKind,
    SourceOpProject,
    SourceOpProjectKind,
    ValueProject,
    ValueProjectKind,
)
from loom.target.contracts.kinds import SourceValueKind
from loom.target.contracts.patterns import TypePattern
from loom.target.contracts.rules import (
    DescriptorRule,
    RecipeRule,
    ValueAliasRule,
    ValueElideRule,
)
from loom.target.contracts.source import ValueRef
from loom.target.contracts.source_memory import (
    SourceMemoryByteOffsetMaterializer,
    SourceMemoryConstraint,
)
from loom.target.low_descriptors import ConstraintKind, Descriptor, OperandRole


@unique
class LowerEmitKind(Enum):
    """Interpreter emit operation used by a compiled lower-rule row."""

    DESCRIPTOR_OP = "descriptor_op"
    DESCRIPTOR_CONST = "descriptor_const"
    DESCRIPTOR_OP_FIRST_LANE = "descriptor_op_first_lane"
    DESCRIPTOR_OP_PER_LANE = "descriptor_op_per_lane"
    DESCRIPTOR_OP_PER_LANE_SEQUENCE = "descriptor_op_per_lane_sequence"
    DESCRIPTOR_OP_ACCUMULATE_LANES = "descriptor_op_accumulate_lanes"


@unique
class LowerAttrCopyKind(Enum):
    """Interpreter attribute-copy operation used by a compiled emit row."""

    DIRECT = "direct"
    ENUM_ORDINAL = "enum_ordinal"
    I64_ARRAY_ELEMENT = "i64_array_element"
    I64_ARRAY_PACK_ELEMENTS = "i64_array_pack_elements"
    I64_ATTRS_PACK_CONSECUTIVE = "i64_attrs_pack_consecutive"
    I64_LITERAL = "i64_literal"
    VALUE_EXACT_I64 = "value_exact_i64"
    VALUE_EXACT_I64_NEGATE = "value_exact_i64_negate"
    VALUE_EXACT_I64_LOG2 = "value_exact_i64_log2"
    VALUE_EXACT_I64_MINUS_ONE = "value_exact_i64_minus_one"
    VALUE_U32_DIVISOR_MAGIC_MULTIPLIER = "value_u32_divisor_magic_multiplier"
    VALUE_U32_DIVISOR_MAGIC_SHIFT = "value_u32_divisor_magic_shift"
    VALUE_I32_AS_U32_BITS = "value_i32_as_u32_bits"
    VALUE_F64_AS_F16_BITS = "value_f64_as_f16_bits"
    VALUE_F64_AS_BF16_BITS = "value_f64_as_bf16_bits"
    VALUE_F64_AS_F32_BITS = "value_f64_as_f32_bits"
    VALUE_F64_AS_F64_BITS = "value_f64_as_f64_bits"
    I64_ARRAY_LANE_BYTE = "i64_array_lane_byte"
    SOURCE_MEMORY_STATIC_BYTE_OFFSET = "source_memory_static_byte_offset"
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


LOWER_EMIT_FLAG_SWAP_OPERANDS_0_1 = 1 << 0
LOWER_EMIT_FLAG_BIND_RESULTS_TO_REFS = 1 << 1
LOWER_EMIT_FLAG_RESULT_TYPE_PATTERN = 1 << 2
LOWER_EMIT_FLAG_ACCUMULATE_SEED_FIRST_LANE = 1 << 3
LOWER_EMIT_FLAG_ACCUMULATE_TREE_BALANCED = 1 << 4
LOWER_EMIT_FLAG_ACCUMULATE_SKIP_FIRST_LANE = 1 << 5
LOWER_EMIT_FLAG_RESULT_DESCRIPTOR_TYPE = 1 << 6
LOWER_SOURCE_MEMORY_NONE = 0
LOWER_RULE_FLAG_CONTRACT_ONLY = 1 << 0

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
    materializer_index: int = 0


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


@dataclass(frozen=True, slots=True)
class LowerSourceMemory:
    """Compiled source-memory constraint row."""

    constraint: SourceMemoryConstraint
    diagnostic_index: int
    dynamic_offset_diagnostic_index: int
    byte_offset_materializer: SourceMemoryByteOffsetMaterializer | None = None


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
    descriptor: Descriptor
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


@dataclass(frozen=True, slots=True)
class LowerRule:
    """Compiled lowering rule row."""

    source_op: Op
    temporary_count: int
    guard_start: int
    guard_count: int
    emit_start: int
    emit_count: int
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
    source_memories: tuple[LowerSourceMemory, ...]
    guards: tuple[LowerGuard, ...]
    attr_copies: tuple[LowerAttrCopy, ...]
    tied_results: tuple[LowerTiedResult, ...]
    emits: tuple[LowerEmit, ...]
    diagnostics: tuple[LowerDiagnostic, ...]


def compile_lower_rule_set(
    table: ContractFragment,
    *,
    dialect_ops: Mapping[str, Sequence[Op]],
) -> CompiledLowerRuleSet:
    """Compiles table-authored descriptor rules to target-low rule rows."""

    compiler = _LowerRuleSetCompiler(table, dialect_ops)
    return compiler.compile()


class _LowerRuleSetCompiler:
    def __init__(
        self,
        table: ContractFragment,
        dialect_ops: Mapping[str, Sequence[Op]],
    ) -> None:
        self._table = table
        self._op_ordinals = _build_op_ordinals(dialect_ops)
        self._rules: list[LowerRule] = []
        self._type_patterns: list[LowerTypePattern] = []
        self._value_refs: list[LowerValueRef] = []
        self._source_memories: list[LowerSourceMemory] = []
        self._guards: list[LowerGuard] = []
        self._attr_copies: list[LowerAttrCopy] = []
        self._tied_results: list[LowerTiedResult] = []
        self._emits: list[LowerEmit] = []
        self._diagnostics: list[LowerDiagnostic] = []
        self._authored_case_indices: list[int] = []
        self._type_pattern_ordinals: dict[TypePattern, int] = {}
        self._diagnostic_ordinals: dict[LowerDiagnostic, int] = {}
        self._register_class_ordinals = {
            reg_class.name: index
            for index, reg_class in enumerate(table.descriptor_set.reg_classes)
        }
        self._materializer_ordinals = {
            materializer.name: index + 1
            for index, materializer in enumerate(table.materializers)
        }

    def compile(self) -> CompiledLowerRuleSet:
        for authored_case_index, contract_case in enumerate(self._table.cases):
            if isinstance(contract_case, DescriptorRule):
                self._append_descriptor_rule(authored_case_index, contract_case)
            elif isinstance(contract_case, ValueAliasRule):
                self._append_alias_rule(authored_case_index, contract_case)
            elif isinstance(contract_case, ValueElideRule):
                self._append_elide_rule(authored_case_index, contract_case)
            elif isinstance(contract_case, RecipeRule):
                self._append_recipe_rule(authored_case_index, contract_case)

        spans = _build_spans(self._rules, self._op_ordinals)
        return CompiledLowerRuleSet(
            name=self._table.name,
            authored_case_indices=tuple(self._authored_case_indices),
            rules=tuple(self._rules),
            spans=spans,
            type_patterns=tuple(self._type_patterns),
            value_refs=tuple(self._value_refs),
            source_memories=tuple(self._source_memories),
            guards=tuple(self._guards),
            attr_copies=tuple(self._attr_copies),
            tied_results=tuple(self._tied_results),
            emits=tuple(self._emits),
            diagnostics=tuple(self._diagnostics),
        )

    def _append_descriptor_rule(
        self,
        authored_case_index: int,
        rule: DescriptorRule,
    ) -> None:
        if not rule.emit:
            raise ValueError(
                f"{rule.source_op.name}: descriptor-rule contracts must "
                "author their emit program in Python"
            )
        guard_start = len(self._guards)
        type_patterns_by_field: dict[str, TypePattern] = {}
        for guard in rule.guards:
            self._append_guard(rule.source_op, guard, type_patterns_by_field)

        emit_start = len(self._emits)
        temporary_ordinals: dict[str, int] = {}
        for emit in rule.emit:
            self._append_emit(
                rule.source_op,
                emit,
                type_patterns_by_field,
                temporary_ordinals,
            )
        self._rules.append(
            LowerRule(
                source_op=rule.source_op,
                temporary_count=len(temporary_ordinals),
                guard_start=guard_start,
                guard_count=len(self._guards) - guard_start,
                emit_start=emit_start,
                emit_count=len(self._emits) - emit_start,
                report_key=rule.report_key,
            )
        )
        self._authored_case_indices.append(authored_case_index)

    def _append_alias_rule(
        self,
        authored_case_index: int,
        rule: ValueAliasRule,
    ) -> None:
        guard_start = len(self._guards)
        type_patterns_by_field: dict[str, TypePattern] = {}
        for guard in rule.guards:
            self._append_guard(rule.source_op, guard, type_patterns_by_field)
        alias_ref_start = self._append_value_ref_sequence(
            (
                self._lower_value_ref(rule.source_op, rule.source, {}),
                self._lower_value_ref(rule.source_op, rule.result, {}),
            )
        )
        self._rules.append(
            LowerRule(
                source_op=rule.source_op,
                temporary_count=0,
                guard_start=guard_start,
                guard_count=len(self._guards) - guard_start,
                emit_start=0,
                emit_count=0,
                alias_ref_start=alias_ref_start,
                alias_ref_count=1,
            )
        )
        self._authored_case_indices.append(authored_case_index)

    def _append_elide_rule(
        self,
        authored_case_index: int,
        rule: ValueElideRule,
    ) -> None:
        guard_start = len(self._guards)
        type_patterns_by_field: dict[str, TypePattern] = {}
        for guard in rule.guards:
            self._append_guard(rule.source_op, guard, type_patterns_by_field)
        elide_ref_start = self._append_value_ref_sequence(
            tuple(
                self._lower_value_ref(rule.source_op, value, {})
                for value in rule.values
            )
        )
        self._rules.append(
            LowerRule(
                source_op=rule.source_op,
                temporary_count=0,
                guard_start=guard_start,
                guard_count=len(self._guards) - guard_start,
                emit_start=0,
                emit_count=0,
                elide_ref_start=elide_ref_start,
                elide_ref_count=len(rule.values),
            )
        )
        self._authored_case_indices.append(authored_case_index)

    def _append_recipe_rule(
        self,
        authored_case_index: int,
        rule: RecipeRule,
    ) -> None:
        guard_start = len(self._guards)
        type_patterns_by_field: dict[str, TypePattern] = {}
        for guard in rule.guards:
            self._append_guard(rule.source_op, guard, type_patterns_by_field)
        self._rules.append(
            LowerRule(
                source_op=rule.source_op,
                temporary_count=0,
                guard_start=guard_start,
                guard_count=len(self._guards) - guard_start,
                emit_start=0,
                emit_count=0,
                flags=LOWER_RULE_FLAG_CONTRACT_ONLY,
            )
        )
        self._authored_case_indices.append(authored_case_index)

    def _append_guard(
        self,
        source_op: Op,
        guard: Guard,
        type_patterns_by_field: dict[str, TypePattern],
    ) -> None:
        if guard.kind == GuardKind.VALUE_TYPE:
            if guard.type_pattern is None:
                raise ValueError(f"{source_op.name}: value_type guard needs a type")
            type_patterns_by_field[guard.field] = guard.type_pattern
            self._guards.append(
                LowerGuard(
                    kind=guard.kind,
                    value_ref_index=self._append_value_ref(
                        source_op,
                        _value_ref_for_source_field(source_op, guard.field),
                    ),
                    type_pattern_index=self._append_type_pattern(guard.type_pattern),
                    diagnostic_index=self._append_diagnostic_ref(
                        source_op,
                        _guard_diagnostic(
                            guard,
                            _value_type_diagnostic(guard.field, guard.type_pattern),
                        ),
                    ),
                )
            )
            return

        if guard.kind == GuardKind.ENUM_ATTR_EQUALS:
            attr_index = _source_attr_index(source_op, guard.field)
            attr = source_op.attrs[attr_index]
            if attr.attr_type != ATTR_TYPE_ENUM or attr.enum_def is None:
                raise ValueError(
                    f"{source_op.name}: enum guard field '{guard.field}' "
                    "must name an enum attr"
                )
            enum_keyword = guard.enum_keyword
            if enum_keyword is None:
                raise ValueError(f"{source_op.name}: enum guard needs a keyword")
            enum_value = next(
                enum_case.value
                for enum_case in attr.enum_def.cases
                if enum_case.keyword == enum_keyword
            )
            self._guards.append(
                LowerGuard(
                    kind=guard.kind,
                    attr_index=attr_index,
                    diagnostic_index=self._append_diagnostic_ref(
                        source_op,
                        _guard_diagnostic(
                            guard,
                            _enum_attr_diagnostic(guard.field, enum_keyword),
                        ),
                    ),
                    u64=enum_value,
                )
            )
            return

        if guard.kind == GuardKind.ATTR_KIND:
            if guard.attr_type is None:
                raise ValueError(f"{source_op.name}: attr_kind guard needs a kind")
            self._guards.append(
                LowerGuard(
                    kind=guard.kind,
                    attr_index=_source_attr_index(source_op, guard.field),
                    diagnostic_index=self._append_diagnostic_ref(
                        source_op,
                        _guard_diagnostic(
                            guard,
                            _attr_diagnostic(guard.field, guard.attr_type),
                        ),
                    ),
                    attr_kind=guard.attr_type,
                )
            )
            return

        if guard.kind == GuardKind.I64_RANGE:
            if guard.minimum is None or guard.maximum is None:
                raise ValueError(f"{source_op.name}: i64_range guard needs bounds")
            self._guards.append(
                LowerGuard(
                    kind=guard.kind,
                    attr_index=_source_attr_index(source_op, guard.field),
                    diagnostic_index=self._append_diagnostic_ref(
                        source_op,
                        _guard_diagnostic(
                            guard,
                            _i64_attr_range_diagnostic(
                                guard.field,
                                guard.minimum,
                                guard.maximum,
                            ),
                        ),
                    ),
                    minimum_i64=guard.minimum,
                    maximum_i64=guard.maximum,
                )
            )
            return

        if guard.kind == GuardKind.DESCRIPTOR_AVAILABLE:
            if guard.descriptor is None:
                raise ValueError(
                    f"{source_op.name}: descriptor guard needs a descriptor"
                )
            self._guards.append(
                LowerGuard(
                    kind=guard.kind,
                    descriptor=guard.descriptor,
                    diagnostic_index=self._append_diagnostic_ref(
                        source_op,
                        _guard_diagnostic(
                            guard,
                            _descriptor_available_diagnostic(guard.descriptor),
                        ),
                    ),
                )
            )
            return

        if guard.kind == GuardKind.VALUE_MATERIALIZABLE:
            if guard.materializer is None:
                raise ValueError(f"{source_op.name}: materializer guard needs a name")
            self._guards.append(
                LowerGuard(
                    kind=guard.kind,
                    value_ref_index=self._append_value_ref(
                        source_op,
                        ValueRef.operand(
                            guard.field,
                            materializer=guard.materializer,
                        ),
                    ),
                    diagnostic_index=self._append_diagnostic_ref(
                        source_op,
                        _guard_diagnostic(
                            guard,
                            _materializer_diagnostic(guard.field, guard.materializer),
                        ),
                    ),
                )
            )
            return

        if guard.kind in _LOW_VALUE_GUARD_KINDS:
            self._append_low_value_guard(source_op, guard)
            return

        if guard.kind == GuardKind.OPERAND_SEGMENT_COUNT:
            if guard.count is None:
                raise ValueError(
                    f"{source_op.name}: operand-segment-count guard needs a count"
                )
            self._guards.append(
                LowerGuard(
                    kind=guard.kind,
                    attr_index=_source_operand_index(source_op, guard.field),
                    diagnostic_index=self._append_diagnostic_ref(
                        source_op,
                        _guard_diagnostic(
                            guard,
                            _operand_segment_count_diagnostic(
                                guard.field,
                                guard.count,
                            ),
                        ),
                    ),
                    u64=guard.count,
                )
            )
            return

        if guard.kind == GuardKind.I64_ARRAY_COUNT:
            if guard.count is None:
                raise ValueError(f"{source_op.name}: array-count guard needs a count")
            self._guards.append(
                LowerGuard(
                    kind=guard.kind,
                    attr_index=_source_attr_index(source_op, guard.field),
                    diagnostic_index=self._append_diagnostic_ref(
                        source_op,
                        _guard_diagnostic(
                            guard,
                            _i64_array_count_diagnostic(guard.field, guard.count),
                        ),
                    ),
                    u64=guard.count,
                )
            )
            return

        if guard.kind == GuardKind.I64_ARRAY_ELEMENT_RANGE:
            if guard.element is None or guard.minimum is None or guard.maximum is None:
                raise ValueError(
                    f"{source_op.name}: array element range guard needs payload"
                )
            self._guards.append(
                LowerGuard(
                    kind=guard.kind,
                    attr_index=_source_attr_index(source_op, guard.field),
                    diagnostic_index=self._append_diagnostic_ref(
                        source_op,
                        _guard_diagnostic(
                            guard,
                            _i64_array_element_range_diagnostic(
                                guard.field,
                                guard.element,
                                guard.minimum,
                                guard.maximum,
                            ),
                        ),
                    ),
                    u64=guard.element,
                    minimum_i64=guard.minimum,
                    maximum_i64=guard.maximum,
                )
            )
            return

        if guard.kind == GuardKind.I64_ARRAY_ELEMENTS_RANGE:
            if guard.minimum is None or guard.maximum is None:
                raise ValueError(
                    f"{source_op.name}: array elements range guard needs bounds"
                )
            self._guards.append(
                LowerGuard(
                    kind=guard.kind,
                    attr_index=_source_attr_index(source_op, guard.field),
                    diagnostic_index=self._append_diagnostic_ref(
                        source_op,
                        _guard_diagnostic(
                            guard,
                            _i64_array_elements_range_diagnostic(
                                guard.field,
                                guard.minimum,
                                guard.maximum,
                            ),
                        ),
                    ),
                    minimum_i64=guard.minimum,
                    maximum_i64=guard.maximum,
                )
            )
            return

        if guard.kind in (
            GuardKind.VALUE_SIGNED_BIT_COUNT,
            GuardKind.VALUE_UNSIGNED_BIT_COUNT,
            GuardKind.VALUE_EXACT_I64,
            GuardKind.VALUE_EXACT_POWER_OF_TWO_I64,
            GuardKind.VALUE_U32_DIVISOR_MAGIC_IS_ADD,
            GuardKind.VALUE_EXACT_F64,
            GuardKind.VALUE_I64_RANGE,
            GuardKind.VALUE_I64_RANGE_LE,
            GuardKind.VALUE_I64_RANGE_GE,
            GuardKind.VALUE_F64_EQUALS,
            GuardKind.VALUE_STORAGE_ELEMENT_FORMAT,
            GuardKind.VALUE_PACKED_INTEGER_PAYLOAD_FROM_LANES,
            GuardKind.VALUE_PACKED_INTEGER_LANES_FROM_PAYLOAD,
            GuardKind.VALUE_STATIC_ELEMENT_COUNT_EQ,
        ):
            self._append_value_fact_guard(source_op, guard)
            return

        if guard.kind == GuardKind.VALUE_NO_USES:
            self._guards.append(
                LowerGuard(
                    kind=guard.kind,
                    value_ref_index=self._append_value_ref(
                        source_op,
                        _value_ref_for_source_field(source_op, guard.field),
                    ),
                    diagnostic_index=self._append_diagnostic_ref(
                        source_op,
                        _guard_diagnostic(
                            guard,
                            _value_no_uses_diagnostic(guard.field),
                        ),
                    ),
                )
            )
            return

        if guard.kind == GuardKind.VECTOR_EXTRACT_SHAPE:
            if guard.other_field is None or guard.attr_field is None:
                raise ValueError(
                    f"{source_op.name}: vector-extract-shape guard needs source, "
                    "result, and static_indices fields"
                )
            self._guards.append(
                LowerGuard(
                    kind=guard.kind,
                    value_ref_index=self._append_value_ref(
                        source_op,
                        _value_ref_for_source_field(source_op, guard.field),
                    ),
                    other_value_ref_index=self._append_value_ref(
                        source_op,
                        _value_ref_for_source_field(source_op, guard.other_field),
                    ),
                    attr_index=_source_attr_index(source_op, guard.attr_field),
                    diagnostic_index=self._append_diagnostic_ref(
                        source_op,
                        _guard_diagnostic(
                            guard,
                            _named_constraint_diagnostic(
                                "shape",
                                source_op.name,
                                guard.kind.value,
                            ),
                        ),
                    ),
                )
            )
            return

        if guard.kind == GuardKind.INSTANCE_FLAGS_HAS_ALL:
            attr_index = _source_attr_index(source_op, guard.field)
            attr = source_op.attrs[attr_index]
            enum_keyword = guard.enum_keyword
            if attr.enum_def is None or enum_keyword is None:
                raise ValueError(
                    f"{source_op.name}: instance-flags guard needs an enum keyword"
                )
            enum_value = next(
                enum_case.value
                for enum_case in attr.enum_def.cases
                if enum_case.keyword == enum_keyword
            )
            self._guards.append(
                LowerGuard(
                    kind=guard.kind,
                    diagnostic_index=self._append_diagnostic_ref(
                        source_op,
                        _guard_diagnostic(
                            guard,
                            _instance_flags_diagnostic(guard.field, enum_keyword),
                        ),
                    ),
                    u64=enum_value,
                )
            )
            return

        raise ValueError(
            f"{source_op.name}: guard kind '{guard.kind.value}' is not "
            "representable by generated descriptor rules yet"
        )

    def _append_low_value_guard(self, source_op: Op, guard: Guard) -> None:
        value_ref_index = self._append_value_ref(
            source_op,
            _value_ref_for_source_field(source_op, guard.field),
        )
        if guard.kind == GuardKind.LOW_VALUE_REGISTER_CLASS:
            if guard.register_class is None:
                raise ValueError(
                    f"{source_op.name}: register-class guard needs a class"
                )
            register_class_id = self._register_class_ordinals.get(guard.register_class)
            if register_class_id is None:
                raise ValueError(
                    f"{source_op.name}: descriptor set has no register class "
                    f"'{guard.register_class}'"
                )
            self._guards.append(
                LowerGuard(
                    kind=guard.kind,
                    value_ref_index=value_ref_index,
                    register_class_id=register_class_id,
                    diagnostic_index=self._append_diagnostic_ref(
                        source_op,
                        _guard_diagnostic(
                            guard,
                            _register_class_diagnostic(
                                guard.field,
                                guard.register_class,
                            ),
                        ),
                    ),
                )
            )
            return

        if guard.kind == GuardKind.LOW_VALUE_REGISTER_UNIT_COUNT:
            if guard.count is None or guard.count <= 0:
                raise ValueError(
                    f"{source_op.name}: register-unit-count guard needs a "
                    "positive count"
                )
            self._guards.append(
                LowerGuard(
                    kind=guard.kind,
                    value_ref_index=value_ref_index,
                    diagnostic_index=self._append_diagnostic_ref(
                        source_op,
                        _guard_diagnostic(
                            guard,
                            _register_unit_count_exact_diagnostic(
                                guard.field,
                                guard.count,
                            ),
                        ),
                    ),
                    u64=guard.count,
                )
            )
            return

        if guard.kind == GuardKind.VALUE_STATIC_DIM0_MULTIPLE:
            if guard.count is None or guard.count <= 0:
                raise ValueError(
                    f"{source_op.name}: static-dim multiple guard needs a divisor"
                )
            self._guards.append(
                LowerGuard(
                    kind=guard.kind,
                    value_ref_index=value_ref_index,
                    diagnostic_index=self._append_diagnostic_ref(
                        source_op,
                        _guard_diagnostic(
                            guard,
                            _static_dim0_multiple_diagnostic(
                                guard.field,
                                guard.count,
                            ),
                        ),
                    ),
                    u64=guard.count,
                )
            )
            return

        if guard.other_field is None:
            raise ValueError(f"{source_op.name}: unit-count guard needs another value")
        self._guards.append(
            LowerGuard(
                kind=guard.kind,
                value_ref_index=value_ref_index,
                other_value_ref_index=self._append_value_ref(
                    source_op,
                    _value_ref_for_source_field(source_op, guard.other_field),
                ),
                diagnostic_index=self._append_diagnostic_ref(
                    source_op,
                    _guard_diagnostic(
                        guard,
                        _register_unit_count_diagnostic(
                            guard.field,
                            guard.other_field,
                        ),
                    ),
                ),
            )
        )

    def _append_value_fact_guard(self, source_op: Op, guard: Guard) -> None:
        value_ref_index = self._append_value_ref(
            source_op,
            _value_ref_for_source_field(source_op, guard.field),
        )
        if guard.kind in (
            GuardKind.VALUE_SIGNED_BIT_COUNT,
            GuardKind.VALUE_UNSIGNED_BIT_COUNT,
        ):
            if guard.count is None or guard.count <= 0:
                raise ValueError(f"{source_op.name}: bit-count guard needs a count")
            self._guards.append(
                LowerGuard(
                    kind=guard.kind,
                    value_ref_index=value_ref_index,
                    diagnostic_index=self._append_diagnostic_ref(
                        source_op,
                        _guard_diagnostic(
                            guard,
                            _bounded_integer_diagnostic(guard.field, guard),
                        ),
                    ),
                    u64=guard.count,
                )
            )
            return
        if guard.kind == GuardKind.VALUE_EXACT_I64:
            self._guards.append(
                LowerGuard(
                    kind=guard.kind,
                    value_ref_index=value_ref_index,
                    diagnostic_index=self._append_diagnostic_ref(
                        source_op,
                        _guard_diagnostic(
                            guard,
                            _exact_integer_diagnostic(guard.field),
                        ),
                    ),
                )
            )
            return
        if guard.kind == GuardKind.VALUE_EXACT_POWER_OF_TWO_I64:
            self._guards.append(
                LowerGuard(
                    kind=guard.kind,
                    value_ref_index=value_ref_index,
                    diagnostic_index=self._append_diagnostic_ref(
                        source_op,
                        _guard_diagnostic(
                            guard,
                            _exact_power_of_two_integer_diagnostic(guard.field),
                        ),
                    ),
                )
            )
            return
        if guard.kind == GuardKind.VALUE_U32_DIVISOR_MAGIC_IS_ADD:
            if guard.count not in (0, 1):
                raise ValueError(
                    f"{source_op.name}: divisor-magic add guard needs 0 or 1"
                )
            self._guards.append(
                LowerGuard(
                    kind=guard.kind,
                    value_ref_index=value_ref_index,
                    diagnostic_index=self._append_diagnostic_ref(
                        source_op,
                        _guard_diagnostic(
                            guard,
                            _u32_divisor_magic_is_add_diagnostic(
                                guard.field,
                                is_add=bool(guard.count),
                            ),
                        ),
                    ),
                    u64=guard.count,
                )
            )
            return
        if guard.kind == GuardKind.VALUE_EXACT_F64:
            self._guards.append(
                LowerGuard(
                    kind=guard.kind,
                    value_ref_index=value_ref_index,
                    diagnostic_index=self._append_diagnostic_ref(
                        source_op,
                        _guard_diagnostic(
                            guard,
                            _exact_float_diagnostic(guard.field),
                        ),
                    ),
                )
            )
            return
        if guard.kind == GuardKind.VALUE_I64_RANGE:
            if guard.minimum is None or guard.maximum is None:
                raise ValueError(f"{source_op.name}: value range guard needs bounds")
            self._guards.append(
                LowerGuard(
                    kind=guard.kind,
                    value_ref_index=value_ref_index,
                    diagnostic_index=self._append_diagnostic_ref(
                        source_op,
                        _guard_diagnostic(
                            guard,
                            _integer_range_diagnostic(
                                guard.field,
                                guard.minimum,
                                guard.maximum,
                            ),
                        ),
                    ),
                    minimum_i64=guard.minimum,
                    maximum_i64=guard.maximum,
                )
            )
            return
        if guard.kind in (
            GuardKind.VALUE_I64_RANGE_LE,
            GuardKind.VALUE_I64_RANGE_GE,
            GuardKind.VALUE_STATIC_ELEMENT_COUNT_EQ,
        ):
            if guard.other_field is None:
                relation_name = (
                    "static-element-count relation"
                    if guard.kind == GuardKind.VALUE_STATIC_ELEMENT_COUNT_EQ
                    else "value range relation"
                )
                raise ValueError(
                    f"{source_op.name}: {relation_name} guard needs another value"
                )
            other_value_ref_index = self._append_value_ref(
                source_op,
                _value_ref_for_source_field(source_op, guard.other_field),
            )
            if guard.kind == GuardKind.VALUE_STATIC_ELEMENT_COUNT_EQ:
                diagnostic = _static_element_count_relation_diagnostic(
                    guard.field,
                    guard.other_field,
                )
            else:
                relation = "le" if guard.kind == GuardKind.VALUE_I64_RANGE_LE else "ge"
                diagnostic = _integer_range_relation_diagnostic(
                    guard.field,
                    guard.other_field,
                    relation,
                )
            self._guards.append(
                LowerGuard(
                    kind=guard.kind,
                    value_ref_index=value_ref_index,
                    other_value_ref_index=other_value_ref_index,
                    diagnostic_index=self._append_diagnostic_ref(
                        source_op,
                        _guard_diagnostic(
                            guard,
                            diagnostic,
                        ),
                    ),
                )
            )
            return
        if guard.kind == GuardKind.VALUE_F64_EQUALS:
            if guard.f64_value is None:
                raise ValueError(f"{source_op.name}: f64-equals guard needs a value")
            self._guards.append(
                LowerGuard(
                    kind=guard.kind,
                    value_ref_index=value_ref_index,
                    diagnostic_index=self._append_diagnostic_ref(
                        source_op,
                        _guard_diagnostic(
                            guard,
                            _float_equals_diagnostic(guard.field, guard.f64_value),
                        ),
                    ),
                    u64=_f64_bits(guard.f64_value),
                )
            )
            return
        if guard.kind == GuardKind.VALUE_STORAGE_ELEMENT_FORMAT:
            if guard.numeric_format_c_expression is None:
                raise ValueError(
                    f"{source_op.name}: storage element-format guard needs "
                    "a numeric format C expression"
                )
            self._guards.append(
                LowerGuard(
                    kind=guard.kind,
                    value_ref_index=value_ref_index,
                    diagnostic_index=self._append_diagnostic_ref(
                        source_op,
                        _guard_diagnostic(
                            guard,
                            _storage_element_format_diagnostic(guard.field),
                        ),
                    ),
                    u64_c_expression=guard.numeric_format_c_expression,
                )
            )
            return
        if guard.kind in (
            GuardKind.VALUE_PACKED_INTEGER_PAYLOAD_FROM_LANES,
            GuardKind.VALUE_PACKED_INTEGER_LANES_FROM_PAYLOAD,
        ):
            if guard.other_field is None or guard.attr_field is None:
                raise ValueError(
                    f"{source_op.name}: {guard.kind.value} guard needs "
                    "two values and one attr field"
                )
            if guard.minimum is None or guard.minimum <= 0:
                raise ValueError(
                    f"{source_op.name}: {guard.kind.value} guard needs "
                    "a positive storage unit bit count"
                )
            if guard.count is None or guard.count <= 0:
                raise ValueError(
                    f"{source_op.name}: {guard.kind.value} guard needs "
                    "a positive storage payload value"
                )
            if guard.kind == GuardKind.VALUE_PACKED_INTEGER_LANES_FROM_PAYLOAD and (
                guard.maximum is None or guard.maximum <= 0
            ):
                raise ValueError(
                    f"{source_op.name}: {guard.kind.value} guard needs "
                    "a positive maximum lane count"
                )
            self._guards.append(
                LowerGuard(
                    kind=guard.kind,
                    value_ref_index=value_ref_index,
                    other_value_ref_index=self._append_value_ref(
                        source_op,
                        _value_ref_for_source_field(source_op, guard.other_field),
                    ),
                    attr_index=_source_attr_index(source_op, guard.attr_field),
                    diagnostic_index=self._append_diagnostic_ref(
                        source_op,
                        _guard_diagnostic(
                            guard,
                            _named_constraint_diagnostic(
                                "value",
                                guard.field,
                                guard.kind.value,
                            ),
                        ),
                    ),
                    u64=guard.count,
                    minimum_i64=guard.minimum,
                    maximum_i64=guard.maximum or 0,
                )
            )
            return
        raise ValueError(
            f"{source_op.name}: guard kind '{guard.kind.value}' is not "
            "a value-fact guard"
        )

    def _append_emit(
        self,
        source_op: Op,
        emit: EmitDescriptorOp,
        type_patterns_by_field: dict[str, TypePattern],
        temporary_ordinals: dict[str, int],
    ) -> None:
        emit_kind = _lower_emit_kind(source_op, emit, type_patterns_by_field)
        operand_bindings = emit.operands if emit.operands is not None else {}
        result_bindings = emit.results if emit.results is not None else {}
        operand_refs: list[LowerValueRef] = []
        operand_ordinals_by_descriptor_field: dict[str, int] = {}
        for descriptor_operand in emit.descriptor.operands:
            if not _descriptor_operand_is_input(descriptor_operand.role):
                continue
            value_ref = operand_bindings.get(descriptor_operand.field_name)
            if value_ref is not None:
                operand_ordinals_by_descriptor_field[descriptor_operand.field_name] = (
                    len(operand_refs)
                )
                operand_refs.append(
                    self._lower_value_ref(source_op, value_ref, temporary_ordinals)
                )
        operand_ref_start = self._append_value_ref_sequence(tuple(operand_refs))
        operand_ref_count = len(operand_refs)

        result_bind_refs: list[LowerValueRef] = []
        result_type_refs: list[LowerValueRef] = []
        result_type_patterns: list[TypePattern] = []
        result_descriptor_type_count = 0
        result_type_bindings = (
            emit.result_types if emit.result_types is not None else result_bindings
        )
        for descriptor_operand in emit.descriptor.operands:
            if not _descriptor_operand_is_output(descriptor_operand.role):
                continue
            value_ref = result_bindings.get(descriptor_operand.field_name)
            if value_ref is not None:
                if value_ref.kind == SourceValueKind.TEMPORARY:
                    temporary_ordinals.setdefault(
                        value_ref.field,
                        len(temporary_ordinals),
                    )
                result_bind_refs.append(
                    self._lower_value_ref(source_op, value_ref, temporary_ordinals)
                )
                type_binding = result_type_bindings.get(descriptor_operand.field_name)
                if type_binding is None:
                    type_binding = value_ref
                if isinstance(type_binding, TypePattern):
                    _require_exact_result_type_pattern(
                        source_op,
                        descriptor_operand.field_name,
                        type_binding,
                    )
                    result_type_patterns.append(type_binding)
                elif isinstance(type_binding, DescriptorResultType):
                    result_descriptor_type_count += 1
                else:
                    result_type_refs.append(
                        self._lower_value_ref(
                            source_op,
                            type_binding,
                            temporary_ordinals,
                        )
                    )
        result_ref_count = len(result_bind_refs)
        if result_type_patterns and result_type_refs:
            raise ValueError(
                f"{source_op.name}: descriptor emit cannot mix value-ref and "
                "type-pattern result type bindings"
            )
        if result_descriptor_type_count and (result_type_patterns or result_type_refs):
            raise ValueError(
                f"{source_op.name}: descriptor emit cannot mix descriptor and "
                "source result type bindings"
            )
        if result_type_patterns and len(result_type_patterns) != result_ref_count:
            raise ValueError(
                f"{source_op.name}: descriptor emit result type patterns must "
                "cover every result"
            )
        if result_type_refs and len(result_type_refs) != result_ref_count:
            raise ValueError(
                f"{source_op.name}: descriptor emit result type refs must "
                "cover every result"
            )
        if (
            result_descriptor_type_count
            and result_descriptor_type_count != result_ref_count
        ):
            raise ValueError(
                f"{source_op.name}: descriptor emit descriptor result types must "
                "cover every result"
            )

        flags = 0
        if emit.swap_first_two_operands:
            flags |= LOWER_EMIT_FLAG_SWAP_OPERANDS_0_1
        if emit.accumulator_seed == DescriptorAccumulatorSeed.FIRST_LANE:
            flags |= LOWER_EMIT_FLAG_ACCUMULATE_SEED_FIRST_LANE
        if emit.accumulator_tree == DescriptorAccumulatorTree.BALANCED:
            flags |= LOWER_EMIT_FLAG_ACCUMULATE_TREE_BALANCED
        if emit.skip_first_lane:
            flags |= LOWER_EMIT_FLAG_ACCUMULATE_SKIP_FIRST_LANE

        if result_type_patterns:
            result_type_pattern_start = self._append_type_pattern_sequence(
                tuple(result_type_patterns)
            )
            result_ref_start = self._append_value_ref_sequence(tuple(result_bind_refs))
            flags |= LOWER_EMIT_FLAG_RESULT_TYPE_PATTERN
        elif result_descriptor_type_count:
            result_type_pattern_start = 0
            result_ref_start = self._append_value_ref_sequence(tuple(result_bind_refs))
            flags |= LOWER_EMIT_FLAG_RESULT_DESCRIPTOR_TYPE
        else:
            result_type_pattern_start = 0
            result_ref_start = self._append_value_ref_sequence(tuple(result_type_refs))

        result_bind_ref_start = 0
        if tuple(result_bind_refs) != tuple(result_type_refs):
            result_bind_ref_start = self._append_value_ref_sequence(
                tuple(result_bind_refs)
            )
            flags |= LOWER_EMIT_FLAG_BIND_RESULTS_TO_REFS

        attr_copies = self._lower_attr_copies(source_op, emit)
        attr_copy_start = self._append_attr_copy_sequence(tuple(attr_copies))

        tied_results, copy_operand_mask = _lower_descriptor_ties(
            emit.descriptor,
            operand_ordinals_by_descriptor_field,
        )
        copy_operand_mask |= _lower_explicit_copy_operand_mask(
            source_op,
            emit,
            operand_ordinals_by_descriptor_field,
        )
        tied_result_start = self._append_tied_result_sequence(tuple(tied_results))

        accumulator_operand_index = 0
        if emit.accumulator is not None:
            try:
                accumulator_operand_index = operand_ordinals_by_descriptor_field[
                    emit.accumulator
                ]
            except KeyError as exc:
                raise ValueError(
                    f"{source_op.name}: accumulator '{emit.accumulator}' "
                    "is not an emitted operand"
                ) from exc

        source_memory_ordinal = LOWER_SOURCE_MEMORY_NONE
        if emit.source_memory is not None:
            if emit_kind != LowerEmitKind.DESCRIPTOR_OP:
                raise ValueError(
                    f"{source_op.name}: source-memory emits must use descriptor-op form"
                )
            source_memory_ordinal = self._append_source_memory(
                source_op,
                emit.source_memory,
                emit.source_memory_byte_offset_materializer,
            )

        self._emits.append(
            LowerEmit(
                kind=emit_kind,
                descriptor=emit.descriptor,
                flags=flags,
                operand_ref_start=operand_ref_start,
                operand_ref_count=operand_ref_count,
                copy_operand_mask=copy_operand_mask,
                accumulator_operand_index=accumulator_operand_index,
                result_ref_start=result_ref_start,
                result_type_pattern_start=result_type_pattern_start,
                result_ref_count=result_ref_count,
                result_bind_ref_start=result_bind_ref_start,
                attr_copy_start=attr_copy_start,
                attr_copy_count=len(attr_copies),
                tied_result_start=tied_result_start,
                tied_result_count=len(tied_results),
                source_memory_ordinal=source_memory_ordinal,
            )
        )

    def _append_source_memory(
        self,
        source_op: Op,
        constraint: SourceMemoryConstraint,
        byte_offset_materializer: SourceMemoryByteOffsetMaterializer | None,
    ) -> int:
        row = LowerSourceMemory(
            constraint,
            diagnostic_index=self._append_diagnostic_ref(
                source_op,
                _source_memory_diagnostic(constraint),
            ),
            dynamic_offset_diagnostic_index=self._append_diagnostic_ref(
                source_op,
                _source_memory_dynamic_offset_diagnostic(constraint),
            ),
            byte_offset_materializer=byte_offset_materializer,
        )
        for index, existing in enumerate(self._source_memories):
            if existing == row:
                return index + 1
        self._source_memories.append(row)
        return len(self._source_memories)

    def _append_type_pattern(self, type_pattern: TypePattern) -> int:
        ordinal = self._type_pattern_ordinals.get(type_pattern)
        if ordinal is not None:
            return ordinal
        ordinal = len(self._type_patterns)
        self._type_pattern_ordinals[type_pattern] = ordinal
        self._type_patterns.append(LowerTypePattern(type_pattern))
        return ordinal

    def _append_type_pattern_sequence(self, sequence: tuple[TypePattern, ...]) -> int:
        if not sequence:
            return 0
        lowered_sequence = tuple(LowerTypePattern(pattern) for pattern in sequence)
        sequence_count = len(lowered_sequence)
        for start in range(len(self._type_patterns) - sequence_count + 1):
            if (
                tuple(self._type_patterns[start : start + sequence_count])
                == lowered_sequence
            ):
                return start
        ordinal = len(self._type_patterns)
        for pattern in sequence:
            self._type_pattern_ordinals.setdefault(pattern, len(self._type_patterns))
            self._type_patterns.append(LowerTypePattern(pattern))
        return ordinal

    def _append_value_ref(self, source_op: Op, value_ref: ValueRef) -> int:
        return self._append_value_ref_sequence(
            (self._lower_value_ref(source_op, value_ref, {}),)
        )

    def _lower_value_ref(
        self,
        source_op: Op,
        value_ref: ValueRef,
        temporary_ordinals: Mapping[str, int],
    ) -> LowerValueRef:
        return _lower_value_ref(
            source_op,
            value_ref,
            temporary_ordinals,
            materializer_ordinals=self._materializer_ordinals,
        )

    def _append_value_ref_sequence(self, sequence: tuple[LowerValueRef, ...]) -> int:
        if not sequence:
            return 0
        sequence_count = len(sequence)
        for start in range(len(self._value_refs) - sequence_count + 1):
            if tuple(self._value_refs[start : start + sequence_count]) == sequence:
                return start
        ordinal = len(self._value_refs)
        self._value_refs.extend(sequence)
        return ordinal

    def _lower_attr_copies(
        self,
        source_op: Op,
        emit: EmitDescriptorOp,
    ) -> tuple[LowerAttrCopy, ...]:
        if not emit.immediates:
            return ()
        attr_copies: list[LowerAttrCopy] = []
        if not isinstance(emit.immediates, Mapping):
            for projection in emit.immediates:
                attr_copies.extend(self._lower_attr_expansion(source_op, projection))
            return tuple(attr_copies)
        for target_name, binding in emit.immediates.items():
            if isinstance(binding, int):
                attr_copies.append(
                    LowerAttrCopy(
                        kind=LowerAttrCopyKind.I64_LITERAL,
                        target_name=target_name,
                        literal_i64=binding,
                    )
                )
                continue
            if isinstance(binding, AttrProject):
                attr_copies.append(
                    self._lower_attr_project(source_op, target_name, binding)
                )
                continue
            if isinstance(binding, SourceOpProject):
                attr_copies.append(
                    self._lower_source_op_project(source_op, target_name, binding)
                )
                continue
            if isinstance(binding, SourceMemoryProject):
                attr_copies.append(
                    self._lower_source_memory_project(target_name, binding)
                )
                continue
            attr_copies.append(
                self._lower_value_project(source_op, target_name, binding)
            )
        return tuple(attr_copies)

    def _lower_attr_project(
        self,
        source_op: Op,
        target_name: str,
        project: AttrProject,
    ) -> LowerAttrCopy:
        source_attr_index = _source_attr_index(source_op, project.source_attr)
        if project.kind == AttrProjectKind.DIRECT:
            return LowerAttrCopy(
                kind=LowerAttrCopyKind.DIRECT,
                target_name=target_name,
                source_attr_index=source_attr_index,
            )
        if project.kind == AttrProjectKind.ENUM_ORDINAL:
            return LowerAttrCopy(
                kind=LowerAttrCopyKind.ENUM_ORDINAL,
                target_name=target_name,
                source_attr_index=source_attr_index,
            )
        if project.kind == AttrProjectKind.I64_ARRAY_ELEMENT:
            if project.element is None:
                raise ValueError(
                    f"{source_op.name}: i64-array element projection needs an element"
                )
            return LowerAttrCopy(
                kind=LowerAttrCopyKind.I64_ARRAY_ELEMENT,
                target_name=target_name,
                source_attr_index=source_attr_index,
                source_element_index=project.element,
                target_bit_offset=project.target_bit_offset,
            )
        if project.kind == AttrProjectKind.I64_ARRAY_PACK_ELEMENTS:
            if (
                project.element is None
                or project.count is None
                or project.bit_width is None
            ):
                raise ValueError(
                    f"{source_op.name}: i64-array pack projection needs payload"
                )
            return LowerAttrCopy(
                kind=LowerAttrCopyKind.I64_ARRAY_PACK_ELEMENTS,
                target_name=target_name,
                source_attr_index=source_attr_index,
                source_element_index=project.element,
                source_element_count=project.count,
                source_element_bit_width=project.bit_width,
                target_bit_offset=project.target_bit_offset,
            )
        if project.kind == AttrProjectKind.I64_ATTRS_PACK_CONSECUTIVE:
            if project.count is None or project.bit_width is None:
                raise ValueError(
                    f"{source_op.name}: i64-attrs pack projection needs payload"
                )
            return LowerAttrCopy(
                kind=LowerAttrCopyKind.I64_ATTRS_PACK_CONSECUTIVE,
                target_name=target_name,
                source_attr_index=source_attr_index,
                source_element_count=project.count,
                source_element_bit_width=project.bit_width,
                target_bit_offset=project.target_bit_offset,
            )
        if project.kind == AttrProjectKind.I64_LOW_BIT_MASK:
            return LowerAttrCopy(
                kind=LowerAttrCopyKind.I64_LOW_BIT_MASK,
                target_name=target_name,
                source_attr_index=source_attr_index,
            )
        if project.kind == AttrProjectKind.I64_SHIFTED_LOW_BIT_MASK:
            return LowerAttrCopy(
                kind=LowerAttrCopyKind.I64_SHIFTED_LOW_BIT_MASK,
                target_name=target_name,
                source_attr_index=source_attr_index,
                other_source_attr_index=_source_attr_index(
                    source_op,
                    project.other_source_attr,
                ),
            )
        if project.kind == AttrProjectKind.I64_SHIFTED_LOW_BIT_CLEAR_MASK:
            return LowerAttrCopy(
                kind=LowerAttrCopyKind.I64_SHIFTED_LOW_BIT_CLEAR_MASK,
                target_name=target_name,
                source_attr_index=source_attr_index,
                other_source_attr_index=_source_attr_index(
                    source_op,
                    project.other_source_attr,
                ),
            )
        if project.kind == AttrProjectKind.I64_LITERAL_MINUS_ATTR:
            return LowerAttrCopy(
                kind=LowerAttrCopyKind.I64_LITERAL_MINUS_ATTR,
                target_name=target_name,
                source_attr_index=source_attr_index,
                literal_i64=project.literal_i64,
            )
        if project.kind == AttrProjectKind.I64_LITERAL_MINUS_ATTRS:
            return LowerAttrCopy(
                kind=LowerAttrCopyKind.I64_LITERAL_MINUS_ATTRS,
                target_name=target_name,
                source_attr_index=source_attr_index,
                other_source_attr_index=_source_attr_index(
                    source_op,
                    project.other_source_attr,
                ),
                literal_i64=project.literal_i64,
            )
        raise ValueError(
            f"{source_op.name}: immediate projection '{project.kind.value}' is "
            "not representable by generated lower rules yet"
        )

    def _lower_attr_expansion(
        self,
        source_op: Op,
        project: AttrProject,
    ) -> tuple[LowerAttrCopy, ...]:
        if project.kind != AttrProjectKind.EXPAND_LANE_I64_ARRAY_TO_BYTE_LANES:
            raise ValueError(
                f"{source_op.name}: immediate projection '{project.kind.value}' is "
                "not representable by generated lower rules yet"
            )
        if project.source_lane_count is None or project.bytes_per_lane is None:
            raise ValueError(
                f"{source_op.name}: lane-byte projection needs source lane count "
                "and bytes per lane"
            )
        source_attr_index = _source_attr_index(source_op, project.source_attr)
        return tuple(
            LowerAttrCopy(
                kind=LowerAttrCopyKind.I64_ARRAY_LANE_BYTE,
                target_name=target_name,
                source_attr_index=source_attr_index,
                source_element_index=index // project.bytes_per_lane,
                source_element_count=project.bytes_per_lane,
                literal_i64=index % project.bytes_per_lane,
            )
            for index, target_name in enumerate(project.target_names)
        )

    def _lower_value_project(
        self,
        source_op: Op,
        target_name: str,
        project: ValueProject,
    ) -> LowerAttrCopy:
        if project.kind == ValueProjectKind.EXACT_I64:
            kind = LowerAttrCopyKind.VALUE_EXACT_I64
        elif project.kind == ValueProjectKind.EXACT_I64_NEGATE:
            kind = LowerAttrCopyKind.VALUE_EXACT_I64_NEGATE
        elif project.kind == ValueProjectKind.EXACT_I64_LOG2:
            kind = LowerAttrCopyKind.VALUE_EXACT_I64_LOG2
        elif project.kind == ValueProjectKind.EXACT_I64_MINUS_ONE:
            kind = LowerAttrCopyKind.VALUE_EXACT_I64_MINUS_ONE
        elif project.kind == ValueProjectKind.U32_DIVISOR_MAGIC_MULTIPLIER:
            kind = LowerAttrCopyKind.VALUE_U32_DIVISOR_MAGIC_MULTIPLIER
        elif project.kind == ValueProjectKind.U32_DIVISOR_MAGIC_SHIFT:
            kind = LowerAttrCopyKind.VALUE_U32_DIVISOR_MAGIC_SHIFT
        elif project.kind == ValueProjectKind.I32_AS_U32_BITS:
            kind = LowerAttrCopyKind.VALUE_I32_AS_U32_BITS
        elif project.kind == ValueProjectKind.F64_AS_F16_BITS:
            kind = LowerAttrCopyKind.VALUE_F64_AS_F16_BITS
        elif project.kind == ValueProjectKind.F64_AS_BF16_BITS:
            kind = LowerAttrCopyKind.VALUE_F64_AS_BF16_BITS
        elif project.kind == ValueProjectKind.F64_AS_F32_BITS:
            kind = LowerAttrCopyKind.VALUE_F64_AS_F32_BITS
        elif project.kind == ValueProjectKind.F64_AS_F64_BITS:
            kind = LowerAttrCopyKind.VALUE_F64_AS_F64_BITS
        else:
            raise ValueError(
                f"{source_op.name}: immediate projection '{project.kind.value}' is "
                "not representable by generated lower rules yet"
            )
        return LowerAttrCopy(
            kind=kind,
            target_name=target_name,
            value_ref_index=self._append_value_ref(
                source_op,
                _value_ref_for_source_field(source_op, project.source_value),
            ),
            target_bit_offset=project.target_bit_offset,
        )

    def _lower_source_memory_project(
        self,
        target_name: str,
        project: SourceMemoryProject,
    ) -> LowerAttrCopy:
        if project.kind == SourceMemoryProjectKind.STATIC_BYTE_OFFSET:
            kind = LowerAttrCopyKind.SOURCE_MEMORY_STATIC_BYTE_OFFSET
        elif project.kind == SourceMemoryProjectKind.STATIC_BYTE_OFFSET_QUOTIENT:
            kind = LowerAttrCopyKind.SOURCE_MEMORY_STATIC_BYTE_OFFSET_QUOTIENT
        elif project.kind == SourceMemoryProjectKind.STATIC_BYTE_OFFSET_REMAINDER:
            kind = LowerAttrCopyKind.SOURCE_MEMORY_STATIC_BYTE_OFFSET_REMAINDER
        elif project.kind == SourceMemoryProjectKind.DYNAMIC_BYTE_STRIDE:
            kind = LowerAttrCopyKind.SOURCE_MEMORY_DYNAMIC_BYTE_STRIDE
        else:
            raise ValueError(
                "source-memory immediate projection "
                f"'{project.kind.value}' is not representable by generated "
                "lower rules yet"
            )
        return LowerAttrCopy(
            kind=kind,
            target_name=target_name,
            dynamic_term_index=project.dynamic_term_index,
            literal_i64=project.divisor,
        )

    def _lower_source_op_project(
        self,
        source_op: Op,
        target_name: str,
        project: SourceOpProject,
    ) -> LowerAttrCopy:
        if project.kind != SourceOpProjectKind.INSTANCE_FLAGS:
            raise ValueError(
                f"{source_op.name}: immediate projection '{project.kind.value}' is "
                "not representable by generated lower rules yet"
            )
        return LowerAttrCopy(
            kind=LowerAttrCopyKind.SOURCE_OP_INSTANCE_FLAGS,
            target_name=target_name,
        )

    def _append_attr_copy_sequence(self, sequence: tuple[LowerAttrCopy, ...]) -> int:
        if not sequence:
            return 0
        sequence_count = len(sequence)
        for start in range(len(self._attr_copies) - sequence_count + 1):
            if tuple(self._attr_copies[start : start + sequence_count]) == sequence:
                return start
        ordinal = len(self._attr_copies)
        self._attr_copies.extend(sequence)
        return ordinal

    def _append_tied_result_sequence(
        self,
        sequence: tuple[LowerTiedResult, ...],
    ) -> int:
        if not sequence:
            return 0
        sequence_count = len(sequence)
        for start in range(len(self._tied_results) - sequence_count + 1):
            if tuple(self._tied_results[start : start + sequence_count]) == sequence:
                return start
        ordinal = len(self._tied_results)
        self._tied_results.extend(sequence)
        return ordinal

    def _append_diagnostic(self, diagnostic: LowerDiagnostic) -> int:
        ordinal = self._diagnostic_ordinals.get(diagnostic)
        if ordinal is not None:
            return ordinal
        ordinal = len(self._diagnostics)
        self._diagnostic_ordinals[diagnostic] = ordinal
        self._diagnostics.append(diagnostic)
        return ordinal

    def _append_diagnostic_ref(self, source_op: Op, ref: DiagnosticRef) -> int:
        return self._append_diagnostic(self._lower_diagnostic_ref(source_op, ref))

    def _lower_diagnostic_ref(
        self, source_op: Op, ref: DiagnosticRef
    ) -> LowerDiagnostic:
        params = [
            self._lower_diagnostic_param(source_op, param) for param in ref.params
        ]
        return LowerDiagnostic(ref.error, tuple(params))

    def _lower_diagnostic_param(
        self,
        source_op: Op,
        param: DiagnosticParam,
    ) -> LowerDiagnosticParam:
        if param.kind == DiagnosticParamKind.VALUE_TYPE:
            return LowerDiagnosticParam(
                name=param.name,
                kind=param.kind,
                value_ref_index=self._append_value_ref(
                    source_op,
                    _value_ref_for_source_field(source_op, param.field),
                ),
            )
        return LowerDiagnosticParam(
            name=param.name,
            kind=param.kind,
            string_value=param.string_value,
            i64_value=param.i64_value,
            u32_value=param.u32_value,
            u64_value=param.u64_value,
            bool_value=param.bool_value,
        )


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


def _f64_bits(value: float) -> int:
    return int.from_bytes(struct.pack("<d", value), byteorder="little", signed=False)


def _lower_emit_kind(
    source_op: Op,
    emit: EmitDescriptorOp,
    type_patterns_by_field: dict[str, TypePattern],
) -> LowerEmitKind:
    if emit.form == DescriptorEmitForm.OP:
        return LowerEmitKind.DESCRIPTOR_OP
    if emit.form == DescriptorEmitForm.CONST:
        return LowerEmitKind.DESCRIPTOR_CONST
    if emit.form == DescriptorEmitForm.FIRST_LANE:
        return LowerEmitKind.DESCRIPTOR_OP_FIRST_LANE
    if emit.form == DescriptorEmitForm.PER_LANE:
        return LowerEmitKind.DESCRIPTOR_OP_PER_LANE
    if emit.form == DescriptorEmitForm.PER_LANE_SEQUENCE:
        return LowerEmitKind.DESCRIPTOR_OP_PER_LANE_SEQUENCE
    if emit.form == DescriptorEmitForm.ACCUMULATE_LANES:
        return LowerEmitKind.DESCRIPTOR_OP_ACCUMULATE_LANES

    if all(
        not _descriptor_operand_is_input(descriptor_operand.role)
        for descriptor_operand in emit.descriptor.operands
    ):
        return LowerEmitKind.DESCRIPTOR_CONST

    result_bindings = (
        emit.result_types
        if emit.result_types is not None
        else (emit.results if emit.results is not None else {})
    )
    vector_result_lanes: int | None = None
    for descriptor_operand in emit.descriptor.operands:
        if not _descriptor_operand_is_output(descriptor_operand.role):
            continue
        result_type_binding = result_bindings.get(descriptor_operand.field_name)
        if result_type_binding is None:
            continue
        if isinstance(result_type_binding, TypePattern):
            result_type = result_type_binding
        elif isinstance(result_type_binding, DescriptorResultType):
            return LowerEmitKind.DESCRIPTOR_OP
        else:
            result_type = _require_type_pattern(
                source_op,
                result_type_binding.field,
                type_patterns_by_field,
            )
        if result_type.kind != "vector":
            return LowerEmitKind.DESCRIPTOR_OP
        vector_result_lanes = result_type.lanes
        if vector_result_lanes is None:
            raise ValueError(
                f"{source_op.name}: per-lane descriptor emits require a "
                "static vector lane count"
            )
        if descriptor_operand.unit_count == 1:
            return LowerEmitKind.DESCRIPTOR_OP_PER_LANE
    return LowerEmitKind.DESCRIPTOR_OP


def _require_type_pattern(
    source_op: Op,
    field: str,
    type_patterns_by_field: dict[str, TypePattern],
) -> TypePattern:
    type_pattern = type_patterns_by_field.get(field)
    if type_pattern is None:
        raise ValueError(
            f"{source_op.name}: descriptor emit field '{field}' needs a "
            "value_type guard"
        )
    return type_pattern


def _require_exact_result_type_pattern(
    source_op: Op,
    descriptor_field: str,
    type_pattern: TypePattern,
) -> None:
    if type_pattern.kind == "view":
        raise ValueError(
            f"{source_op.name}: descriptor emit result type pattern for "
            f"'{descriptor_field}' cannot synthesize view types"
        )
    if len(type_pattern.elements) != 1:
        raise ValueError(
            f"{source_op.name}: descriptor emit result type pattern for "
            f"'{descriptor_field}' must select exactly one scalar element"
        )
    if type_pattern.kind == "vector" and type_pattern.lanes is None:
        raise ValueError(
            f"{source_op.name}: descriptor emit result type pattern for "
            f"'{descriptor_field}' must have an exact vector lane count"
        )


def _value_ref_for_source_field(source_op: Op, field: str) -> ValueRef:
    if source_op.operand(field) is not None:
        return ValueRef.operand(field)
    if source_op.result(field) is not None:
        return ValueRef.result(field)
    raise ValueError(f"{source_op.name}: source field '{field}' is not a value")


def _lower_value_ref(
    source_op: Op,
    value_ref: ValueRef,
    temporary_ordinals: Mapping[str, int],
    *,
    materializer_ordinals: Mapping[str, int],
) -> LowerValueRef:
    materializer_index = 0
    if value_ref.materializer is not None:
        materializer_index = materializer_ordinals.get(value_ref.materializer, 0)
        if materializer_index == 0:
            raise ValueError(
                f"{source_op.name}: source value field '{value_ref.field}' "
                f"references unknown materializer '{value_ref.materializer}'"
            )
    return LowerValueRef(
        kind=value_ref.kind,
        index=_source_value_index(source_op, value_ref, temporary_ordinals),
        materializer_index=materializer_index,
    )


def _source_value_index(
    source_op: Op,
    value_ref: ValueRef,
    temporary_ordinals: Mapping[str, int],
) -> int:
    if value_ref.kind == SourceValueKind.OPERAND:
        operand = source_op.operand(value_ref.field)
        if operand is not None:
            return source_op.operands.index(operand) + value_ref.element
    if value_ref.kind == SourceValueKind.RESULT:
        result = source_op.result(value_ref.field)
        if result is not None:
            return source_op.results.index(result)
    if value_ref.kind == SourceValueKind.TEMPORARY:
        ordinal = temporary_ordinals.get(value_ref.field)
        if ordinal is not None:
            return ordinal
    if value_ref.kind == SourceValueKind.SOURCE_MEMORY_DYNAMIC_TERM:
        return value_ref.element
    if value_ref.kind == SourceValueKind.SOURCE_MEMORY_DYNAMIC_BYTE_OFFSET:
        return 0
    raise ValueError(f"source value field '{value_ref.field}' is not declared")


def _source_attr_index(source_op: Op, field: str) -> int:
    attr = source_op.attr(field)
    if attr is None:
        raise ValueError(f"{source_op.name}: source field '{field}' is not an attr")
    return source_op.attrs.index(attr)


def _source_operand_index(source_op: Op, field: str) -> int:
    operand = source_op.operand(field)
    if operand is None:
        raise ValueError(f"{source_op.name}: source field '{field}' is not an operand")
    return source_op.operands.index(operand)


def _type_pattern_text(type_pattern: TypePattern) -> str:
    element_text = _type_pattern_element_text(type_pattern)
    if type_pattern.kind == "scalar":
        return element_text
    if type_pattern.kind == "view":
        return f"view<{element_text}>"
    if type_pattern.dims:
        dims_text = "x".join(str(dim) for dim in type_pattern.dims)
        return f"vector<{dims_text}x{element_text}>"
    if type_pattern.lanes is not None:
        return f"vector<{type_pattern.lanes}x{element_text}>"
    return f"vector<{element_text}>"


def _type_pattern_element_text(type_pattern: TypePattern) -> str:
    if len(type_pattern.elements) == 1:
        return type_pattern.elements[0]
    return "{" + ", ".join(type_pattern.elements) + "}"


def _value_type_diagnostic(field: str, type_pattern: TypePattern) -> DiagnosticRef:
    return target_diagnostic(
        ERR_TARGET_002,
        string_param("field_name", field),
        value_type_param("actual_type", field),
        string_param("expected_type", _type_pattern_text(type_pattern)),
    )


def _enum_attr_diagnostic(field: str, enum_keyword: str) -> DiagnosticRef:
    return _named_constraint_diagnostic(
        "field",
        field,
        f"enum_case.{enum_keyword}",
    )


def _i64_attr_range_diagnostic(
    field: str,
    minimum: int,
    maximum: int,
) -> DiagnosticRef:
    return _range_constraint_diagnostic(
        "field", field, "i64_attr_range", minimum, maximum
    )


def _descriptor_available_diagnostic(descriptor: Descriptor) -> DiagnosticRef:
    return _named_constraint_diagnostic(
        "descriptor",
        descriptor.key,
        "descriptor_available",
    )


def _materializer_diagnostic(field: str, materializer: str) -> DiagnosticRef:
    return _named_constraint_diagnostic(
        "field",
        field,
        f"materializer.{materializer}",
    )


def _register_class_diagnostic(field: str, register_class: str) -> DiagnosticRef:
    return _named_constraint_diagnostic(
        "field",
        field,
        f"low_register_class.{register_class}",
    )


def _static_dim0_multiple_diagnostic(field: str, multiple: int) -> DiagnosticRef:
    return _count_constraint_diagnostic(
        "field",
        field,
        "static_dim0_multiple",
        multiple,
    )


def _register_unit_count_diagnostic(
    field: str,
    other_field: str,
) -> DiagnosticRef:
    return _relation_constraint_diagnostic(
        "field",
        field,
        other_field,
        "low_register_unit_count_eq",
    )


def _static_element_count_relation_diagnostic(
    field: str,
    other_field: str,
) -> DiagnosticRef:
    return _relation_constraint_diagnostic(
        "field",
        field,
        other_field,
        "static_element_count_eq",
    )


def _register_unit_count_exact_diagnostic(field: str, count: int) -> DiagnosticRef:
    return _count_constraint_diagnostic(
        "field",
        field,
        "low_register_unit_count",
        count,
    )


def _operand_segment_count_diagnostic(field: str, count: int) -> DiagnosticRef:
    return _count_constraint_diagnostic(
        "operand_segment",
        field,
        "operand_segment_count",
        count,
    )


def _i64_array_count_diagnostic(field: str, count: int) -> DiagnosticRef:
    return _count_constraint_diagnostic("i64_array", field, "array_count", count)


def _i64_array_element_range_diagnostic(
    field: str,
    element: int,
    minimum: int,
    maximum: int,
) -> DiagnosticRef:
    return _element_range_constraint_diagnostic(
        "i64_array",
        field,
        "element_range",
        element,
        minimum,
        maximum,
    )


def _i64_array_elements_range_diagnostic(
    field: str,
    minimum: int,
    maximum: int,
) -> DiagnosticRef:
    return _range_constraint_diagnostic(
        "i64_array",
        field,
        "elements_range",
        minimum,
        maximum,
    )


def _bounded_integer_diagnostic(field: str, guard: Guard) -> DiagnosticRef:
    signedness = (
        "signed" if guard.kind == GuardKind.VALUE_SIGNED_BIT_COUNT else "unsigned"
    )
    return _count_constraint_diagnostic(
        "value_fact",
        field,
        f"{signedness}_bit_count",
        guard.count or 0,
    )


def _exact_integer_diagnostic(field: str) -> DiagnosticRef:
    return _named_constraint_diagnostic("value_fact", field, "exact_i64")


def _exact_power_of_two_integer_diagnostic(field: str) -> DiagnosticRef:
    return _named_constraint_diagnostic(
        "value_fact",
        field,
        "exact_power_of_two_i64",
    )


def _u32_divisor_magic_is_add_diagnostic(field: str, *, is_add: bool) -> DiagnosticRef:
    suffix = "add" if is_add else "no_add"
    return _named_constraint_diagnostic(
        "value_fact",
        field,
        f"u32_divisor_magic.{suffix}",
    )


def _exact_float_diagnostic(field: str) -> DiagnosticRef:
    return _named_constraint_diagnostic("value_fact", field, "exact_f64")


def _integer_range_diagnostic(
    field: str,
    minimum: int,
    maximum: int,
) -> DiagnosticRef:
    return _range_constraint_diagnostic(
        "value_fact", field, "i64_range", minimum, maximum
    )


def _integer_range_relation_diagnostic(
    field: str,
    other_field: str,
    relation: str,
) -> DiagnosticRef:
    return _relation_constraint_diagnostic(
        "value_fact",
        field,
        other_field,
        f"i64_range_{relation}",
    )


def _float_equals_diagnostic(field: str, value: float) -> DiagnosticRef:
    return _named_constraint_diagnostic(
        "value_fact", field, f"f64_equals.0x{_f64_bits(value):016x}"
    )


def _storage_element_format_diagnostic(field: str) -> DiagnosticRef:
    return _named_constraint_diagnostic("value", field, "storage_schema.element_format")


def _value_no_uses_diagnostic(field: str) -> DiagnosticRef:
    return _named_constraint_diagnostic("value", field, "no_ordinary_uses")


def _instance_flags_diagnostic(field: str, enum_keyword: str) -> DiagnosticRef:
    return _named_constraint_diagnostic("flags", field, f"has_all.{enum_keyword}")


def _source_memory_diagnostic(
    constraint: SourceMemoryConstraint,
) -> DiagnosticRef:
    if constraint.diagnostic is not None:
        ref = constraint.diagnostic.ref
        if ref is None:
            raise ValueError("source-memory diagnostic is missing an error ref")
        return ref
    return target_diagnostic(
        ERR_TARGET_008,
        string_param("operation_kind", constraint.operation.value),
    )


def _source_memory_dynamic_offset_diagnostic(
    constraint: SourceMemoryConstraint,
) -> DiagnosticRef:
    if constraint.dynamic_offset_diagnostic is not None:
        ref = constraint.dynamic_offset_diagnostic.ref
        if ref is None:
            raise ValueError(
                "source-memory dynamic-offset diagnostic is missing an error ref"
            )
        return ref
    return _source_memory_diagnostic(constraint)


def _attr_diagnostic(field: str, attr_type: str) -> DiagnosticRef:
    return _named_constraint_diagnostic("field", field, f"attr_kind.{attr_type}")


def _named_constraint_diagnostic(
    subject_role: str,
    subject_name: str,
    constraint_key: str,
) -> DiagnosticRef:
    return target_diagnostic(
        ERR_TARGET_003,
        string_param("subject_role", subject_role),
        string_param("subject_name", subject_name),
        string_param("constraint_key", constraint_key),
    )


def _count_constraint_diagnostic(
    subject_role: str,
    subject_name: str,
    constraint_key: str,
    expected_count: int,
) -> DiagnosticRef:
    return target_diagnostic(
        ERR_TARGET_004,
        string_param("subject_role", subject_role),
        string_param("subject_name", subject_name),
        string_param("constraint_key", constraint_key),
        u32_param("expected_count", expected_count),
    )


def _range_constraint_diagnostic(
    subject_role: str,
    subject_name: str,
    constraint_key: str,
    minimum: int,
    maximum: int,
) -> DiagnosticRef:
    return target_diagnostic(
        ERR_TARGET_005,
        string_param("subject_role", subject_role),
        string_param("subject_name", subject_name),
        string_param("constraint_key", constraint_key),
        i64_param("minimum", minimum),
        i64_param("maximum", maximum),
    )


def _element_range_constraint_diagnostic(
    subject_role: str,
    subject_name: str,
    constraint_key: str,
    element: int,
    minimum: int,
    maximum: int,
) -> DiagnosticRef:
    return target_diagnostic(
        ERR_TARGET_006,
        string_param("subject_role", subject_role),
        string_param("subject_name", subject_name),
        u32_param("element", element),
        string_param("constraint_key", constraint_key),
        i64_param("minimum", minimum),
        i64_param("maximum", maximum),
    )


def _relation_constraint_diagnostic(
    subject_role: str,
    subject_name: str,
    other_subject_name: str,
    constraint_key: str,
) -> DiagnosticRef:
    return target_diagnostic(
        ERR_TARGET_007,
        string_param("subject_role", subject_role),
        string_param("subject_name", subject_name),
        string_param("other_subject_name", other_subject_name),
        string_param("constraint_key", constraint_key),
    )


def _guard_diagnostic(
    guard: Guard,
    default_diagnostic: DiagnosticRef,
) -> DiagnosticRef:
    if guard.diagnostic is None:
        return default_diagnostic
    ref = guard.diagnostic.ref
    if ref is None:
        raise ValueError("guard diagnostic is missing an error ref")
    return ref


def _descriptor_operand_is_input(role: OperandRole) -> bool:
    return role in (
        OperandRole.OPERAND,
        OperandRole.OPERAND_RESULT,
        OperandRole.PREDICATE,
        OperandRole.RESOURCE,
    )


def _descriptor_operand_is_output(role: OperandRole) -> bool:
    return role in (OperandRole.RESULT, OperandRole.OPERAND_RESULT)


def _lower_descriptor_ties(
    descriptor: Descriptor,
    operand_ordinals_by_descriptor_field: Mapping[str, int],
) -> tuple[tuple[LowerTiedResult, ...], int]:
    result_ordinals_by_descriptor_index: dict[int, int] = {}
    operand_ordinals_by_descriptor_index: dict[int, int] = {}
    result_ordinal = 0
    for descriptor_index, descriptor_operand in enumerate(descriptor.operands):
        if _descriptor_operand_is_output(descriptor_operand.role):
            result_ordinals_by_descriptor_index[descriptor_index] = result_ordinal
            result_ordinal += 1
        if _descriptor_operand_is_input(descriptor_operand.role):
            operand_ordinal = operand_ordinals_by_descriptor_field.get(
                descriptor_operand.field_name
            )
            if operand_ordinal is not None:
                operand_ordinals_by_descriptor_index[descriptor_index] = operand_ordinal

    tied_results: list[LowerTiedResult] = []
    copy_operand_mask = 0
    for constraint in descriptor.constraints:
        if constraint.kind not in (ConstraintKind.TIED, ConstraintKind.DESTRUCTIVE):
            continue
        if constraint.rhs_operand_index is None:
            raise ValueError(
                f"descriptor '{descriptor.key}' constraint needs a rhs operand"
            )
        try:
            result_index = result_ordinals_by_descriptor_index[
                constraint.lhs_operand_index
            ]
            operand_index = operand_ordinals_by_descriptor_index[
                constraint.rhs_operand_index
            ]
        except KeyError as exc:
            raise ValueError(
                f"descriptor '{descriptor.key}' constraint references an "
                "unbound result or operand"
            ) from exc
        if constraint.kind is ConstraintKind.TIED:
            tied_results.append(
                LowerTiedResult(
                    result_index=result_index,
                    operand_index=operand_index,
                )
            )
        else:
            copy_operand_mask |= 1 << operand_index
    return tuple(tied_results), copy_operand_mask


def _lower_explicit_copy_operand_mask(
    source_op: Op,
    emit: EmitDescriptorOp,
    operand_ordinals_by_descriptor_field: Mapping[str, int],
) -> int:
    copy_operand_mask = 0
    for descriptor_field in emit.copy_operands:
        try:
            operand_index = operand_ordinals_by_descriptor_field[descriptor_field]
        except KeyError as exc:
            raise ValueError(
                f"{source_op.name}: copied descriptor operand "
                f"'{descriptor_field}' is not emitted"
            ) from exc
        copy_operand_mask |= 1 << operand_index
    return copy_operand_mask


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
