# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Compilation from descriptor-rule contracts to target-low lowering rows."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace

from loom.dsl import ATTR_TYPE_ENUM, Op
from loom.target.contracts.diagnostics import (
    DiagnosticParam,
    DiagnosticParamKind,
    DiagnosticRef,
)
from loom.target.contracts.emits import (
    ContractEmit,
    DescriptorAccumulatorSeed,
    DescriptorAccumulatorTree,
    DescriptorResultType,
    EmitDescriptorOp,
    EmitRegisterConcat,
    EmitRegisterCopy,
    EmitRegisterSlice,
    ResultTypeBinding,
)
from loom.target.contracts.fragments import ContractFragment
from loom.target.contracts.guards import Guard, GuardKind
from loom.target.contracts.immediates import (
    AttrProject,
    AttrProjectKind,
    SourceMemoryProject,
    SourceOpProject,
    ValueProject,
    ValueProjectKind,
)
from loom.target.contracts.kinds import SourceValueKind
from loom.target.contracts.lower_rule_bindings import (
    _descriptor_operand_is_input,
    _descriptor_operand_is_output,
    _f64_bits,
    _lower_descriptor_ties,
    _lower_emit_kind,
    _lower_explicit_copy_operand_mask,
    _lower_source_memory_project,
    _lower_source_op_project,
    _lower_value_ref,
    _operand_segment_counts,
    _order_operand_segment_guards,
    _require_exact_result_type_pattern,
    _source_attr_index,
    _source_operand_index,
    _value_ref_for_source_field,
)
from loom.target.contracts.lower_rule_diagnostics import (
    _attr_diagnostic,
    _bounded_integer_diagnostic,
    _descriptor_available_diagnostic,
    _enum_attr_diagnostic,
    _exact_float_diagnostic,
    _exact_integer_diagnostic,
    _exact_power_of_two_integer_diagnostic,
    _float_equals_diagnostic,
    _guard_diagnostic,
    _i64_array_count_diagnostic,
    _i64_array_element_range_diagnostic,
    _i64_array_elements_range_diagnostic,
    _i64_attr_range_diagnostic,
    _instance_flags_diagnostic,
    _integer_range_diagnostic,
    _integer_range_relation_diagnostic,
    _materializer_diagnostic,
    _named_constraint_diagnostic,
    _operand_segment_count_diagnostic,
    _register_class_diagnostic,
    _register_unit_count_diagnostic,
    _register_unit_count_exact_diagnostic,
    _source_memory_address_diagnostic,
    _source_memory_address_layout_diagnostic,
    _source_memory_diagnostic,
    _source_memory_dynamic_offset_diagnostic,
    _static_dim0_multiple_diagnostic,
    _static_element_count_relation_diagnostic,
    _storage_element_format_diagnostic,
    _u32_divisor_magic_is_add_diagnostic,
    _value_no_uses_diagnostic,
    _value_type_diagnostic,
)
from loom.target.contracts.lower_rule_tables import (
    _LOW_VALUE_GUARD_KINDS,
    LOWER_EMIT_FLAG_ACCUMULATE_SEED_FIRST_LANE,
    LOWER_EMIT_FLAG_ACCUMULATE_SKIP_FIRST_LANE,
    LOWER_EMIT_FLAG_ACCUMULATE_TREE_BALANCED,
    LOWER_EMIT_FLAG_BIND_RESULTS_TO_REFS,
    LOWER_EMIT_FLAG_RESULT_DESCRIPTOR_TYPE,
    LOWER_EMIT_FLAG_RESULT_TYPE_PATTERN,
    LOWER_EMIT_FLAG_SWAP_OPERANDS_0_1,
    LOWER_RULE_FLAG_CONTRACT_ONLY,
    LOWER_RULE_FLAG_ORDINAL_VALUE_ALIAS,
    LOWER_RULE_PRIMARY_EMIT_NONE,
    LOWER_SOURCE_MEMORY_NONE,
    CompiledLowerRuleSet,
    LowerAttrCopy,
    LowerAttrCopyKind,
    LowerDiagnostic,
    LowerDiagnosticParam,
    LowerEmit,
    LowerEmitKind,
    LowerGuard,
    LowerRule,
    LowerSourceMemory,
    LowerSourceNode,
    LowerTiedResult,
    LowerTypePattern,
    LowerValueRef,
    _append_interned_row_sequence,
    _build_op_ordinals,
    _build_spans,
    _intern_program_rows,
    _op_kind_key,
)
from loom.target.contracts.patterns import TypePattern
from loom.target.contracts.rules import (
    DescriptorRule,
    OrdinalValueAliasRule,
    RecipeRule,
    ValueAliasRule,
    ValueElideRule,
    contract_case_priority,
)
from loom.target.contracts.source import ValueRef
from loom.target.contracts.source_memory import (
    SourceMemoryAddressLayout,
    SourceMemoryAddressMaterializer,
    SourceMemoryByteOffsetMaterializer,
    SourceMemoryConstraint,
)


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
        self._source_nodes: list[LowerSourceNode] = []
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
        self._source_ops: dict[str, Op] = {}
        self._source_node_ordinals: dict[str, int] = {}
        self._operand_segment_counts: dict[int, dict[str, int]] = {}

    def compile(self) -> CompiledLowerRuleSet:
        ordered_cases = sorted(
            enumerate(self._table.cases),
            key=lambda item: (
                _op_kind_key(item[1].source_op, self._op_ordinals),
                -contract_case_priority(item[1]),
                item[0],
            ),
        )
        for authored_case_index, contract_case in ordered_cases:
            if isinstance(contract_case, DescriptorRule):
                self._append_descriptor_rule(authored_case_index, contract_case)
            elif isinstance(contract_case, ValueAliasRule):
                self._append_alias_rule(
                    authored_case_index,
                    contract_case.source_op,
                    contract_case.source,
                    contract_case.result,
                    contract_case.guards,
                )
            elif isinstance(contract_case, OrdinalValueAliasRule):
                self._append_alias_rule(
                    authored_case_index,
                    contract_case.source_op,
                    contract_case.source,
                    contract_case.result,
                    contract_case.guards,
                    flags=LOWER_RULE_FLAG_ORDINAL_VALUE_ALIAS,
                )
            elif isinstance(contract_case, ValueElideRule):
                self._append_elide_rule(authored_case_index, contract_case)
            elif isinstance(contract_case, RecipeRule):
                self._append_recipe_rule(authored_case_index, contract_case)

        guard_ranges = [(rule.guard_start, rule.guard_count) for rule in self._rules]
        guard_ranges.extend(
            (source_node.guard_start, source_node.guard_count)
            for source_node in self._source_nodes
        )
        guards, guard_starts = _intern_program_rows(
            self._guards,
            guard_ranges,
        )
        emits, emit_starts = _intern_program_rows(
            self._emits,
            ((rule.emit_start, rule.emit_count) for rule in self._rules),
        )
        rules = tuple(
            replace(
                rule,
                guard_start=guard_starts[index],
                emit_start=emit_starts[index],
            )
            for index, rule in enumerate(self._rules)
        )
        source_nodes = tuple(
            replace(
                source_node,
                guard_start=guard_starts[len(self._rules) + index],
            )
            for index, source_node in enumerate(self._source_nodes)
        )
        spans = _build_spans(rules, self._op_ordinals)
        return CompiledLowerRuleSet(
            name=self._table.name,
            authored_case_indices=tuple(self._authored_case_indices),
            rules=rules,
            spans=spans,
            type_patterns=tuple(self._type_patterns),
            value_refs=tuple(self._value_refs),
            source_nodes=source_nodes,
            source_memories=tuple(self._source_memories),
            guards=guards,
            attr_copies=tuple(self._attr_copies),
            tied_results=tuple(self._tied_results),
            emits=emits,
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
        if rule.descriptor is not None and not any(
            isinstance(emit, EmitDescriptorOp) and emit.descriptor == rule.descriptor
            for emit in rule.emit
        ):
            raise ValueError(
                f"{rule.source_op.name}: primary descriptor "
                f"'{rule.descriptor.key}' is not emitted by the rule"
            )
        self._source_ops = {"": rule.source_op}
        self._source_node_ordinals = {"": 0}
        for source_node_index, source_node in enumerate(rule.source_nodes, start=1):
            self._source_ops[source_node.name] = source_node.source_op
            self._source_node_ordinals[source_node.name] = source_node_index
        type_patterns_by_source_node: dict[int, dict[str, TypePattern]] = {}
        source_node_start = len(self._source_nodes)
        for source_node_index, source_node in enumerate(rule.source_nodes, start=1):
            source_node_guard_start = len(self._guards)
            source_node_type_patterns: dict[str, TypePattern] = {}
            type_patterns_by_source_node[source_node_index] = source_node_type_patterns
            self._append_guards(
                source_node.source_op,
                source_node.guards,
                source_node_type_patterns,
                source_node_index=source_node_index,
            )
            self._source_nodes.append(
                LowerSourceNode(
                    relation=source_node.relation,
                    source_op=source_node.source_op,
                    parent_node_index=self._source_node_ordinals[source_node.parent],
                    parent_value_ref_index=self._append_value_ref(
                        rule.source_op,
                        replace(
                            source_node.parent_value,
                            source_node=source_node.parent,
                        ),
                    ),
                    node_value_ref_index=self._append_value_ref(
                        rule.source_op,
                        replace(
                            source_node.node_value,
                            source_node=source_node.name,
                        ),
                    ),
                    guard_start=source_node_guard_start,
                    guard_count=len(self._guards) - source_node_guard_start,
                )
            )
        guard_start = len(self._guards)
        type_patterns_by_field: dict[str, TypePattern] = {}
        type_patterns_by_source_node[0] = type_patterns_by_field
        self._append_guards(rule.source_op, rule.guards, type_patterns_by_field)

        emit_start = len(self._emits)
        temporary_ordinals: dict[str, int] = {}
        primary_emit_ordinal = LOWER_RULE_PRIMARY_EMIT_NONE
        for emit in rule.emit:
            if (
                primary_emit_ordinal == LOWER_RULE_PRIMARY_EMIT_NONE
                and isinstance(emit, EmitDescriptorOp)
                and (rule.descriptor is None or emit.descriptor == rule.descriptor)
            ):
                primary_emit_ordinal = len(self._emits) - emit_start
            self._append_emit(
                rule.source_op,
                emit,
                type_patterns_by_source_node,
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
                primary_emit_ordinal=primary_emit_ordinal,
                source_node_start=(source_node_start if rule.source_nodes else 0),
                source_node_count=len(rule.source_nodes),
                report_key=rule.report_key,
            )
        )
        self._authored_case_indices.append(authored_case_index)

    def _append_alias_rule(
        self,
        authored_case_index: int,
        source_op: Op,
        source: ValueRef,
        result: ValueRef,
        guards: tuple[Guard, ...],
        *,
        flags: int = 0,
    ) -> None:
        self._source_ops = {"": source_op}
        self._source_node_ordinals = {"": 0}
        guard_start = len(self._guards)
        type_patterns_by_field: dict[str, TypePattern] = {}
        self._append_guards(source_op, guards, type_patterns_by_field)
        is_ordinal_alias = bool(flags & LOWER_RULE_FLAG_ORDINAL_VALUE_ALIAS)
        alias_ref_start = self._append_value_ref_sequence(
            (
                self._lower_value_ref(
                    source_op,
                    source,
                    {},
                    allow_variadic_span=is_ordinal_alias,
                ),
                self._lower_value_ref(
                    source_op,
                    result,
                    {},
                    allow_variadic_span=is_ordinal_alias,
                ),
            )
        )
        self._rules.append(
            LowerRule(
                source_op=source_op,
                temporary_count=0,
                guard_start=guard_start,
                guard_count=len(self._guards) - guard_start,
                emit_start=0,
                emit_count=0,
                flags=flags,
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
        self._source_ops = {"": rule.source_op}
        self._source_node_ordinals = {"": 0}
        guard_start = len(self._guards)
        type_patterns_by_field: dict[str, TypePattern] = {}
        self._append_guards(rule.source_op, rule.guards, type_patterns_by_field)
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
        self._source_ops = {"": rule.source_op}
        self._source_node_ordinals = {"": 0}
        guard_start = len(self._guards)
        type_patterns_by_field: dict[str, TypePattern] = {}
        self._append_guards(rule.source_op, rule.guards, type_patterns_by_field)
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

    def _append_guards(
        self,
        source_op: Op,
        guards: Sequence[Guard],
        type_patterns_by_field: dict[str, TypePattern],
        *,
        source_node_index: int = 0,
    ) -> None:
        self._operand_segment_counts[source_node_index] = _operand_segment_counts(
            source_op, guards
        )
        for guard in _order_operand_segment_guards(source_op, guards):
            self._append_guard(source_op, guard, type_patterns_by_field)

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
            GuardKind.VALUE_EXACT_FLOAT,
            GuardKind.VALUE_I64_RANGE,
            GuardKind.VALUE_I64_RANGE_LE,
            GuardKind.VALUE_I64_RANGE_GE,
            GuardKind.VALUE_FLOAT_EQUALS,
            GuardKind.VALUE_STORAGE_ELEMENT_FORMAT,
            GuardKind.VALUE_MEMORY_SPACE,
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
        if guard.kind == GuardKind.VALUE_EXACT_FLOAT:
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
        if guard.kind == GuardKind.VALUE_FLOAT_EQUALS:
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
        if guard.kind == GuardKind.VALUE_MEMORY_SPACE:
            self._guards.append(
                LowerGuard(
                    kind=guard.kind,
                    value_ref_index=value_ref_index,
                    diagnostic_index=self._append_diagnostic_ref(
                        source_op,
                        _guard_diagnostic(
                            guard,
                            _named_constraint_diagnostic(
                                "value_fact",
                                guard.field,
                                "memory_space",
                            ),
                        ),
                    ),
                    memory_spaces=guard.memory_spaces,
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
        emit: ContractEmit,
        type_patterns_by_source_node: dict[int, dict[str, TypePattern]],
        temporary_ordinals: dict[str, int],
    ) -> None:
        if isinstance(emit, EmitDescriptorOp):
            self._append_descriptor_emit(
                source_op,
                emit,
                type_patterns_by_source_node,
                temporary_ordinals,
            )
            return
        if isinstance(emit, EmitRegisterSlice):
            self._append_structural_emit(
                source_op,
                LowerEmitKind.REGISTER_SLICE,
                (emit.source,),
                emit.result,
                emit.result_type,
                temporary_ordinals,
                structural_offset=emit.unit_offset,
                structural_unit_count=emit.unit_count or 0,
            )
            return
        if isinstance(emit, EmitRegisterConcat):
            self._append_structural_emit(
                source_op,
                LowerEmitKind.REGISTER_CONCAT,
                emit.sources,
                emit.result,
                emit.result_type,
                temporary_ordinals,
            )
            return
        if isinstance(emit, EmitRegisterCopy):
            self._append_structural_emit(
                source_op,
                LowerEmitKind.REGISTER_COPY,
                (emit.source,),
                emit.result,
                emit.result_type,
                temporary_ordinals,
            )
            return
        raise TypeError(f"unsupported contract emit type: {type(emit).__name__}")

    def _append_descriptor_emit(
        self,
        source_op: Op,
        emit: EmitDescriptorOp,
        type_patterns_by_source_node: dict[int, dict[str, TypePattern]],
        temporary_ordinals: dict[str, int],
    ) -> None:
        emit_kind = _lower_emit_kind(
            source_op,
            emit,
            type_patterns_by_source_node,
            self._source_node_ordinals,
            self._source_ops,
        )
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
                emit.source_memory_address_materializer,
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

    def _append_structural_emit(
        self,
        source_op: Op,
        emit_kind: LowerEmitKind,
        sources: Sequence[ValueRef],
        result: ValueRef,
        result_type: ResultTypeBinding | None,
        temporary_ordinals: dict[str, int],
        *,
        structural_offset: int = 0,
        structural_unit_count: int = 0,
    ) -> None:
        operand_refs = tuple(
            self._lower_value_ref(source_op, source, temporary_ordinals)
            for source in sources
        )
        operand_ref_start = self._append_value_ref_sequence(operand_refs)

        if result.kind == SourceValueKind.TEMPORARY:
            temporary_ordinals.setdefault(result.field, len(temporary_ordinals))
        result_bind_ref = self._lower_value_ref(
            source_op,
            result,
            temporary_ordinals,
        )
        result_bind_ref_start = self._append_value_ref_sequence((result_bind_ref,))

        flags = 0
        result_type_pattern_start = 0
        type_binding = result if result_type is None else result_type
        if structural_unit_count != 0:
            result_ref_start = result_bind_ref_start
        elif isinstance(type_binding, TypePattern):
            _require_exact_result_type_pattern(
                source_op,
                "result",
                type_binding,
            )
            result_type_pattern_start = self._append_type_pattern(type_binding)
            result_ref_start = result_bind_ref_start
            flags |= (
                LOWER_EMIT_FLAG_BIND_RESULTS_TO_REFS
                | LOWER_EMIT_FLAG_RESULT_TYPE_PATTERN
            )
        elif isinstance(type_binding, DescriptorResultType):
            raise ValueError(
                f"{source_op.name}: structural emits have no descriptor result type"
            )
        else:
            result_type_ref = self._lower_value_ref(
                source_op,
                type_binding,
                temporary_ordinals,
            )
            result_ref_start = self._append_value_ref_sequence((result_type_ref,))
            if result_type_ref != result_bind_ref:
                flags |= LOWER_EMIT_FLAG_BIND_RESULTS_TO_REFS

        self._emits.append(
            LowerEmit(
                kind=emit_kind,
                descriptor=None,
                flags=flags,
                operand_ref_start=operand_ref_start,
                operand_ref_count=len(operand_refs),
                result_ref_start=result_ref_start,
                result_type_pattern_start=result_type_pattern_start,
                result_ref_count=1,
                result_bind_ref_start=result_bind_ref_start,
                structural_offset=structural_offset,
                structural_unit_count=structural_unit_count,
            )
        )

    def _append_source_memory(
        self,
        source_op: Op,
        constraint: SourceMemoryConstraint,
        byte_offset_materializer: SourceMemoryByteOffsetMaterializer | None,
        address_materializer: SourceMemoryAddressMaterializer | None,
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
            address_layout_diagnostic_index=(
                0xFFFF
                if constraint.address_layout == SourceMemoryAddressLayout.ANY
                else self._append_diagnostic_ref(
                    source_op,
                    _source_memory_address_layout_diagnostic(constraint),
                )
            ),
            address_diagnostic_index=self._append_diagnostic_ref(
                source_op,
                _source_memory_address_diagnostic(
                    constraint,
                    address_materializer,
                ),
            ),
            byte_offset_materializer=byte_offset_materializer,
            address_materializer=address_materializer,
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
        *,
        allow_variadic_span: bool = False,
    ) -> LowerValueRef:
        source_node_index = 0
        referenced_op = source_op
        if value_ref.source_node:
            referenced_op = self._source_ops.get(value_ref.source_node)
            source_node_index = self._source_node_ordinals.get(
                value_ref.source_node, -1
            )
            if referenced_op is None or source_node_index < 0:
                raise ValueError(
                    f"{source_op.name}: source value field '{value_ref.field}' "
                    f"references unknown source node '{value_ref.source_node}'"
                )
        if value_ref.kind == SourceValueKind.OPERAND and not allow_variadic_span:
            operand = referenced_op.operand(value_ref.field)
            if operand is not None and operand.variadic:
                expected_count = self._operand_segment_counts.get(
                    source_node_index, {}
                ).get(value_ref.field)
                if expected_count is None:
                    raise ValueError(
                        f"{referenced_op.name}: variadic operand reference "
                        f"'{value_ref.field}[{value_ref.element}]' needs an "
                        "operand_segment_count guard"
                    )
                if value_ref.element >= expected_count:
                    raise ValueError(
                        f"{referenced_op.name}: variadic operand reference "
                        f"'{value_ref.field}[{value_ref.element}]' exceeds guarded "
                        f"segment count {expected_count}"
                    )
        return _lower_value_ref(
            referenced_op,
            value_ref,
            temporary_ordinals,
            source_node_index=source_node_index,
            materializer_ordinals=self._materializer_ordinals,
        )

    def _append_value_ref_sequence(self, sequence: tuple[LowerValueRef, ...]) -> int:
        return _append_interned_row_sequence(self._value_refs, sequence)

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
                    _lower_source_op_project(source_op, target_name, binding)
                )
                continue
            if isinstance(binding, SourceMemoryProject):
                attr_copies.append(_lower_source_memory_project(target_name, binding))
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
        if project.kind == AttrProjectKind.I64_ARRAY_ELEMENT_PLUS_LITERAL:
            if project.element is None:
                raise ValueError(
                    f"{source_op.name}: i64-array element-plus-literal "
                    "projection needs an element"
                )
            return LowerAttrCopy(
                kind=LowerAttrCopyKind.I64_ARRAY_ELEMENT_PLUS_LITERAL,
                target_name=target_name,
                source_attr_index=source_attr_index,
                source_element_index=project.element,
                literal_i64=project.literal_i64,
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
        if project.kind == AttrProjectKind.I64_ATTR_MINUS_LITERAL:
            return LowerAttrCopy(
                kind=LowerAttrCopyKind.I64_ATTR_MINUS_LITERAL,
                target_name=target_name,
                source_attr_index=source_attr_index,
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
        elif project.kind == ValueProjectKind.EXACT_I64_I32_WORD:
            kind = LowerAttrCopyKind.VALUE_EXACT_I64_I32_WORD
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
        elif project.kind == ValueProjectKind.FLOAT_AS_F16_BITS:
            kind = LowerAttrCopyKind.VALUE_FLOAT_AS_F16_BITS
        elif project.kind == ValueProjectKind.FLOAT_AS_BF16_BITS:
            kind = LowerAttrCopyKind.VALUE_FLOAT_AS_BF16_BITS
        elif project.kind == ValueProjectKind.FLOAT_AS_F32_BITS:
            kind = LowerAttrCopyKind.VALUE_FLOAT_AS_F32_BITS
        elif project.kind == ValueProjectKind.FLOAT_AS_F32_I32:
            kind = LowerAttrCopyKind.VALUE_FLOAT_AS_F32_I32
        elif project.kind == ValueProjectKind.FLOAT_AS_F64_BITS:
            kind = LowerAttrCopyKind.VALUE_FLOAT_AS_F64_BITS
        elif project.kind == ValueProjectKind.FLOAT_AS_F64_I32_WORD:
            kind = LowerAttrCopyKind.VALUE_FLOAT_AS_F64_I32_WORD
        else:
            raise ValueError(
                f"{source_op.name}: immediate projection '{project.kind.value}' is "
                "not representable by generated lower rules yet"
            )
        referenced_op = source_op
        if project.source_node:
            referenced_op = self._source_ops.get(project.source_node)
            if referenced_op is None:
                raise ValueError(
                    f"{source_op.name}: immediate projection references unknown "
                    f"source node '{project.source_node}'"
                )
        value_ref = _value_ref_for_source_field(
            referenced_op,
            project.source_value,
        )
        if project.source_node:
            value_ref = replace(value_ref, source_node=project.source_node)
        return LowerAttrCopy(
            kind=kind,
            target_name=target_name,
            value_ref_index=self._append_value_ref(
                source_op,
                value_ref,
            ),
            target_bit_offset=project.target_bit_offset,
            source_element_index=project.word_index,
        )

    def _append_attr_copy_sequence(self, sequence: tuple[LowerAttrCopy, ...]) -> int:
        return _append_interned_row_sequence(self._attr_copies, sequence)

    def _append_tied_result_sequence(
        self,
        sequence: tuple[LowerTiedResult, ...],
    ) -> int:
        return _append_interned_row_sequence(self._tied_results, sequence)

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
        return LowerDiagnostic(
            ref.error,
            tuple(params),
            target_context_param_count=ref.target_context_param_count,
        )

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
