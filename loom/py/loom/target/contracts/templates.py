# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Templates for regular target contract descriptor-rule families."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from loom.dsl import EnumCase, Op
from loom.target.contracts.emits import DescriptorEmitForm, EmitDescriptorOp
from loom.target.contracts.guards import Guard, GuardDiagnostic
from loom.target.contracts.patterns import TypePattern
from loom.target.contracts.rules import DescriptorRule
from loom.target.contracts.source import ValueRef
from loom.target.low_descriptors import Descriptor

type DirectTypePatterns = TypePattern | Mapping[str, TypePattern]


@dataclass(frozen=True, slots=True)
class DirectDescriptorCase:
    """One direct source-op to descriptor mapping for a typed op family."""

    source_op: Op
    descriptor: Descriptor
    type_patterns: DirectTypePatterns
    semantic_tag: str | None = None
    priority: int = 0


@dataclass(frozen=True, slots=True)
class SelectDescriptorCase:
    """One select-like source-op to descriptor mapping."""

    source_op: Op
    descriptor: Descriptor
    condition_type: TypePattern
    value_type: TypePattern
    semantic_tag: str | None = None
    priority: int = 0


@dataclass(frozen=True, slots=True)
class PredicateDescriptorCase:
    """One enum-predicate case for a compare-like op family."""

    predicate: str | EnumCase
    descriptor: Descriptor
    semantic_tag: str | None = None
    priority: int = 0


@dataclass(frozen=True, slots=True)
class DotDescriptorCase:
    """One dot-like source op to descriptor mapping."""

    source_op: Op
    descriptor: Descriptor
    lhs_type: TypePattern
    rhs_type: TypePattern
    accumulator_type: TypePattern
    result_type: TypePattern
    kind: str | EnumCase | None = None
    semantic_tag: str | None = None
    priority: int = 0


@dataclass(frozen=True, slots=True)
class ReductionDescriptorCase:
    """One vector reduction kind lowered through extract and combine descriptors."""

    kind: str | EnumCase
    input_type: TypePattern
    accumulator_type: TypePattern
    extract_descriptor: Descriptor
    combine_descriptor: Descriptor
    extract_semantic_tag: str | None = None
    combine_semantic_tag: str | None = None
    priority: int = 0


def unary_descriptor_rules(
    cases: Sequence[DirectDescriptorCase],
    *,
    source_input: str = "input",
    source_result: str = "result",
    descriptor_input: str = "input",
    descriptor_result: str = "dst",
) -> tuple[DescriptorRule, ...]:
    """Expands regular same-typed unary source ops to descriptor rules."""

    rules: list[DescriptorRule] = []
    for case in cases:
        _require_semantic_tag(
            case.descriptor,
            case.semantic_tag,
            f"{case.source_op.name} unary descriptor case",
        )
        rules.append(
            DescriptorRule(
                source_op=case.source_op,
                descriptor=case.descriptor,
                guards=_value_type_guards(
                    case.type_patterns,
                    fields=(source_input, source_result),
                    subject=f"{case.source_op.name} unary descriptor case",
                ),
                emit=(
                    _emit_descriptor_op(
                        descriptor=case.descriptor,
                        operands=((descriptor_input, source_input),),
                        results=((descriptor_result, source_result),),
                    ),
                ),
                priority=case.priority,
            )
        )
    return tuple(rules)


def binary_descriptor_rules(
    cases: Sequence[DirectDescriptorCase],
    *,
    source_lhs: str = "lhs",
    source_rhs: str = "rhs",
    source_result: str = "result",
    descriptor_lhs: str = "lhs",
    descriptor_rhs: str = "rhs",
    descriptor_result: str = "dst",
) -> tuple[DescriptorRule, ...]:
    """Expands regular same-typed binary source ops to descriptor rules."""

    rules: list[DescriptorRule] = []
    for case in cases:
        _require_semantic_tag(
            case.descriptor,
            case.semantic_tag,
            f"{case.source_op.name} binary descriptor case",
        )
        rules.append(
            DescriptorRule(
                source_op=case.source_op,
                descriptor=case.descriptor,
                guards=_value_type_guards(
                    case.type_patterns,
                    fields=(source_lhs, source_rhs, source_result),
                    subject=f"{case.source_op.name} binary descriptor case",
                ),
                emit=(
                    _emit_descriptor_op(
                        descriptor=case.descriptor,
                        operands=(
                            (descriptor_lhs, source_lhs),
                            (descriptor_rhs, source_rhs),
                        ),
                        results=((descriptor_result, source_result),),
                    ),
                ),
                priority=case.priority,
            )
        )
    return tuple(rules)


def ternary_descriptor_rules(
    cases: Sequence[DirectDescriptorCase],
    *,
    source_a: str = "a",
    source_b: str = "b",
    source_c: str = "c",
    source_result: str = "result",
    descriptor_a: str = "a",
    descriptor_b: str = "b",
    descriptor_c: str = "c",
    descriptor_result: str = "dst",
) -> tuple[DescriptorRule, ...]:
    """Expands regular same-typed ternary source ops to descriptor rules."""

    rules: list[DescriptorRule] = []
    for case in cases:
        _require_semantic_tag(
            case.descriptor,
            case.semantic_tag,
            f"{case.source_op.name} ternary descriptor case",
        )
        rules.append(
            DescriptorRule(
                source_op=case.source_op,
                descriptor=case.descriptor,
                guards=_value_type_guards(
                    case.type_patterns,
                    fields=(source_a, source_b, source_c, source_result),
                    subject=f"{case.source_op.name} ternary descriptor case",
                ),
                emit=(
                    _emit_descriptor_op(
                        descriptor=case.descriptor,
                        operands=(
                            (descriptor_a, source_a),
                            (descriptor_b, source_b),
                            (descriptor_c, source_c),
                        ),
                        results=((descriptor_result, source_result),),
                    ),
                ),
                priority=case.priority,
            )
        )
    return tuple(rules)


def select_descriptor_rules(
    cases: Sequence[SelectDescriptorCase],
    *,
    source_condition: str = "condition",
    source_true_value: str = "true_value",
    source_false_value: str = "false_value",
    source_result: str = "result",
    descriptor_condition: str = "condition",
    descriptor_true_value: str = "true_value",
    descriptor_false_value: str = "false_value",
    descriptor_result: str = "dst",
) -> tuple[DescriptorRule, ...]:
    """Expands select-like source ops to descriptor rules."""

    rules: list[DescriptorRule] = []
    for case in cases:
        _require_semantic_tag(
            case.descriptor,
            case.semantic_tag,
            f"{case.source_op.name} select descriptor case",
        )
        rules.append(
            DescriptorRule(
                source_op=case.source_op,
                descriptor=case.descriptor,
                guards=(
                    Guard.value_type(source_condition, case.condition_type),
                    Guard.value_type(source_true_value, case.value_type),
                    Guard.value_type(source_false_value, case.value_type),
                    Guard.value_type(source_result, case.value_type),
                ),
                emit=(
                    _emit_descriptor_op(
                        descriptor=case.descriptor,
                        operands=(
                            (descriptor_true_value, source_true_value),
                            (descriptor_false_value, source_false_value),
                            (descriptor_condition, source_condition),
                        ),
                        results=((descriptor_result, source_result),),
                    ),
                ),
                priority=case.priority,
            )
        )
    return tuple(rules)


def compare_descriptor_rules(
    source_op: Op,
    cases: Sequence[PredicateDescriptorCase],
    *,
    operand_type: TypePattern,
    result_type: TypePattern,
    predicate_attr: str = "predicate",
    source_lhs: str = "lhs",
    source_rhs: str = "rhs",
    source_result: str = "result",
    descriptor_lhs: str = "lhs",
    descriptor_rhs: str = "rhs",
    descriptor_result: str = "dst",
) -> tuple[DescriptorRule, ...]:
    """Expands enum-predicate compare ops to one descriptor rule per predicate."""

    rules: list[DescriptorRule] = []
    for case in cases:
        _require_semantic_tag(
            case.descriptor,
            case.semantic_tag,
            f"{source_op.name} compare descriptor case",
        )
        rules.append(
            DescriptorRule(
                source_op=source_op,
                descriptor=case.descriptor,
                guards=(
                    Guard.enum_attr_equals(predicate_attr, case.predicate),
                    Guard.value_type(source_lhs, operand_type),
                    Guard.value_type(source_rhs, operand_type),
                    Guard.value_type(source_result, result_type),
                ),
                emit=(
                    _emit_descriptor_op(
                        descriptor=case.descriptor,
                        operands=(
                            (descriptor_lhs, source_lhs),
                            (descriptor_rhs, source_rhs),
                        ),
                        results=((descriptor_result, source_result),),
                    ),
                ),
                priority=case.priority,
            )
        )
    return tuple(rules)


def dot_descriptor_rules(
    cases: Sequence[DotDescriptorCase],
    *,
    kind_attr: str = "kind",
    source_lhs: str = "lhs",
    source_rhs: str = "rhs",
    source_accumulator: str = "acc",
    source_result: str = "result",
    descriptor_lhs: str = "lhs",
    descriptor_rhs: str = "rhs",
    descriptor_accumulator: str = "acc",
    descriptor_result: str = "dst",
    kind_diagnostic: GuardDiagnostic | None = None,
    type_diagnostics: Mapping[str, GuardDiagnostic] | None = None,
    descriptor_diagnostic: GuardDiagnostic | None = None,
    emit_form: DescriptorEmitForm = DescriptorEmitForm.OP,
) -> tuple[DescriptorRule, ...]:
    """Expands dot-like accumulator ops to descriptor rules."""

    rules: list[DescriptorRule] = []
    for case in cases:
        _require_semantic_tag(
            case.descriptor,
            case.semantic_tag,
            f"{case.source_op.name} dot descriptor case",
        )
        guards: list[Guard] = []
        if case.kind is not None:
            guards.append(
                Guard.enum_attr_equals(
                    kind_attr,
                    case.kind,
                    diagnostic=kind_diagnostic,
                )
            )
        guards.extend(
            (
                _value_type_guard(
                    source_lhs, case.lhs_type, diagnostics=type_diagnostics
                ),
                _value_type_guard(
                    source_rhs, case.rhs_type, diagnostics=type_diagnostics
                ),
                _value_type_guard(
                    source_accumulator,
                    case.accumulator_type,
                    diagnostics=type_diagnostics,
                ),
                _value_type_guard(
                    source_result, case.result_type, diagnostics=type_diagnostics
                ),
                Guard.descriptor_available(
                    case.descriptor,
                    diagnostic=descriptor_diagnostic,
                ),
            )
        )
        rules.append(
            DescriptorRule(
                source_op=case.source_op,
                descriptor=case.descriptor,
                guards=tuple(guards),
                emit=(
                    _emit_descriptor_op(
                        descriptor=case.descriptor,
                        operands=(
                            (descriptor_accumulator, source_accumulator),
                            (descriptor_lhs, source_lhs),
                            (descriptor_rhs, source_rhs),
                        ),
                        results=((descriptor_result, source_result),),
                        form=emit_form,
                    ),
                ),
                priority=case.priority,
            )
        )
    return tuple(rules)


def reduction_descriptor_rules(
    source_op: Op,
    cases: Sequence[ReductionDescriptorCase],
    *,
    lane_count: int,
    kind_attr: str = "kind",
    source_input: str = "input",
    source_init: str = "init",
    source_result: str = "result",
    extract_source: str = "source",
    extract_result: str = "dst",
    extract_lane: str = "lane",
    combine_lhs: str = "lhs",
    combine_rhs: str = "rhs",
    combine_result: str = "dst",
) -> tuple[DescriptorRule, ...]:
    """Expands fixed-lane reductions into extract/combine descriptor chains."""

    if lane_count <= 0:
        raise ValueError("reduction lane count must be positive")
    rules: list[DescriptorRule] = []
    for case in cases:
        _require_semantic_tag(
            case.extract_descriptor,
            case.extract_semantic_tag,
            f"{source_op.name} reduction extract descriptor case",
        )
        _require_semantic_tag(
            case.combine_descriptor,
            case.combine_semantic_tag,
            f"{source_op.name} reduction combine descriptor case",
        )
        rules.append(
            DescriptorRule(
                source_op=source_op,
                descriptor=case.combine_descriptor,
                guards=(
                    Guard.enum_attr_equals(kind_attr, case.kind),
                    Guard.value_type(source_input, case.input_type),
                    Guard.value_type(source_init, case.accumulator_type),
                    Guard.value_type(source_result, case.accumulator_type),
                ),
                emit=_reduction_emit_chain(
                    lane_count=lane_count,
                    input_field=source_input,
                    init_field=source_init,
                    result_field=source_result,
                    accumulator_type=case.accumulator_type,
                    extract_descriptor=case.extract_descriptor,
                    combine_descriptor=case.combine_descriptor,
                    extract_source=extract_source,
                    extract_result=extract_result,
                    extract_lane=extract_lane,
                    combine_lhs=combine_lhs,
                    combine_rhs=combine_rhs,
                    combine_result=combine_result,
                ),
                priority=case.priority,
            )
        )
    return tuple(rules)


def _value_type_guards(
    patterns: DirectTypePatterns,
    *,
    fields: Sequence[str],
    subject: str,
) -> tuple[Guard, ...]:
    return tuple(
        Guard.value_type(field, _type_pattern_for_field(patterns, field, subject))
        for field in fields
    )


def _value_type_guard(
    field: str,
    type_pattern: TypePattern,
    *,
    diagnostics: Mapping[str, GuardDiagnostic] | None = None,
) -> Guard:
    diagnostic = None if diagnostics is None else diagnostics.get(field)
    return Guard.value_type(field, type_pattern, diagnostic=diagnostic)


def _type_pattern_for_field(
    patterns: DirectTypePatterns,
    field: str,
    subject: str,
) -> TypePattern:
    if isinstance(patterns, TypePattern):
        return patterns
    type_pattern = patterns.get(field)
    if type_pattern is None:
        raise ValueError(f"{subject}: source field '{field}' has no type pattern")
    return type_pattern


def _reduction_emit_chain(
    *,
    lane_count: int,
    input_field: str,
    init_field: str,
    result_field: str,
    accumulator_type: TypePattern,
    extract_descriptor: Descriptor,
    combine_descriptor: Descriptor,
    extract_source: str,
    extract_result: str,
    extract_lane: str,
    combine_lhs: str,
    combine_rhs: str,
    combine_result: str,
) -> tuple[EmitDescriptorOp, ...]:
    emits: list[EmitDescriptorOp] = []
    accumulator = ValueRef.operand(init_field)
    for lane in range(lane_count):
        lane_value = ValueRef.temporary(f"lane{lane}")
        emits.append(
            EmitDescriptorOp(
                descriptor=extract_descriptor,
                operands={extract_source: ValueRef.operand(input_field)},
                results={extract_result: lane_value},
                result_types={extract_result: accumulator_type},
                immediates={extract_lane: lane},
            )
        )
        next_accumulator = (
            ValueRef.result(result_field)
            if lane == lane_count - 1
            else ValueRef.temporary(f"acc{lane}")
        )
        emits.append(
            EmitDescriptorOp(
                descriptor=combine_descriptor,
                operands={
                    combine_lhs: accumulator,
                    combine_rhs: lane_value,
                },
                results={combine_result: next_accumulator},
                result_types={combine_result: accumulator_type},
            )
        )
        accumulator = next_accumulator
    return tuple(emits)


def _emit_descriptor_op(
    *,
    descriptor: Descriptor,
    operands: Sequence[tuple[str, str]],
    results: Sequence[tuple[str, str]],
    form: DescriptorEmitForm = DescriptorEmitForm.AUTO,
) -> EmitDescriptorOp:
    return EmitDescriptorOp(
        descriptor=descriptor,
        operands={
            descriptor_field: ValueRef.operand(source_field)
            for descriptor_field, source_field in operands
        },
        results={
            descriptor_field: ValueRef.result(source_field)
            for descriptor_field, source_field in results
        },
        form=form,
    )


def _require_semantic_tag(
    descriptor: Descriptor,
    semantic_tag: str | None,
    subject: str,
) -> None:
    if semantic_tag is None:
        return
    if descriptor.semantic_tag != semantic_tag:
        raise ValueError(
            f"{subject}: descriptor '{descriptor.key}' has semantic tag "
            f"'{descriptor.semantic_tag}', expected '{semantic_tag}'"
        )
