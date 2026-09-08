# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Source, type, and descriptor bindings for target-Low rules."""

from __future__ import annotations

import struct
from collections.abc import Mapping, Sequence

from loom.dsl import Op
from loom.target.contracts.emits import (
    DescriptorEmitForm,
    DescriptorResultType,
    EmitDescriptorOp,
)
from loom.target.contracts.guards import Guard, GuardKind
from loom.target.contracts.immediates import (
    SourceMemoryProject,
    SourceMemoryProjectKind,
    SourceOpProject,
    SourceOpProjectKind,
)
from loom.target.contracts.kinds import SourceValueKind
from loom.target.contracts.lower_rule_tables import (
    LowerAttrCopy,
    LowerAttrCopyKind,
    LowerEmitKind,
    LowerTiedResult,
    LowerValueRef,
)
from loom.target.contracts.patterns import TypePattern
from loom.target.contracts.source import ValueRef
from loom.target.low_descriptors import (
    ConstraintKind,
    Descriptor,
    DescriptorOpKind,
    OperandRole,
)


def _f64_bits(value: float) -> int:
    return int.from_bytes(struct.pack("<d", value), byteorder="little", signed=False)


def _lower_source_memory_project(
    target_name: str,
    project: SourceMemoryProject,
) -> LowerAttrCopy:
    if project.kind == SourceMemoryProjectKind.STATIC_BYTE_OFFSET:
        kind = LowerAttrCopyKind.SOURCE_MEMORY_STATIC_BYTE_OFFSET
    elif project.kind == SourceMemoryProjectKind.STATIC_BYTE_OFFSET_PLUS_LITERAL:
        kind = LowerAttrCopyKind.SOURCE_MEMORY_STATIC_BYTE_OFFSET_PLUS_LITERAL
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
        literal_i64=(
            project.literal_i64
            if project.kind == SourceMemoryProjectKind.STATIC_BYTE_OFFSET_PLUS_LITERAL
            else project.divisor
        ),
    )


def _lower_source_op_project(
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


def _lower_emit_kind(
    source_op: Op,
    emit: EmitDescriptorOp,
    type_patterns_by_source_node: dict[int, dict[str, TypePattern]],
    source_node_ordinals: Mapping[str, int],
    source_ops: Mapping[str, Op],
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

    if emit.descriptor.op_kind is DescriptorOpKind.CONST:
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
            source_node_index = source_node_ordinals[result_type_binding.source_node]
            referenced_op = source_ops[result_type_binding.source_node]
            result_type = _require_type_pattern(
                referenced_op,
                result_type_binding.field,
                type_patterns_by_source_node[source_node_index],
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
    source_node_index: int = 0,
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
        source_node_index=source_node_index,
        element_index=(
            value_ref.element
            if value_ref.kind in (SourceValueKind.OPERAND, SourceValueKind.RESULT)
            else 0
        ),
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
            return source_op.operands.index(operand)
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
    if value_ref.kind == SourceValueKind.SOURCE_MEMORY_ADDRESS:
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


def _operand_segment_counts(
    source_op: Op,
    guards: Sequence[Guard],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for guard in guards:
        if guard.kind != GuardKind.OPERAND_SEGMENT_COUNT:
            continue
        if guard.count is None:
            raise ValueError(
                f"{source_op.name}: operand-segment-count guard needs a count"
            )
        previous_count = counts.get(guard.field)
        if previous_count is not None:
            raise ValueError(
                f"{source_op.name}: variadic operand '{guard.field}' has "
                "multiple operand_segment_count guards"
            )
        counts[guard.field] = guard.count
    return counts


def _order_operand_segment_guards(
    source_op: Op,
    guards: Sequence[Guard],
) -> tuple[Guard, ...]:
    """Orders each variadic arity guard before its first dereference."""
    ordered_guards = list(guards)
    for segment_guard in guards:
        if segment_guard.kind != GuardKind.OPERAND_SEGMENT_COUNT:
            continue
        segment_guard_index = next(
            index
            for index, candidate in enumerate(ordered_guards)
            if candidate is segment_guard
        )
        first_reference_index = next(
            (
                index
                for index, candidate in enumerate(ordered_guards)
                if candidate.kind != GuardKind.OPERAND_SEGMENT_COUNT
                and any(
                    field == segment_guard.field
                    for field in (candidate.field, candidate.other_field)
                )
            ),
            None,
        )
        if (
            first_reference_index is not None
            and first_reference_index < segment_guard_index
        ):
            ordered_guards.pop(segment_guard_index)
            ordered_guards.insert(first_reference_index, segment_guard)
    return tuple(ordered_guards)


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
