# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""SPIR-V ordinary-vector ALU and lanewise selection contract rows."""

from __future__ import annotations

from loom.dialect.vector import defs as vector
from loom.dsl import Op
from loom.target.arch.spirv.contracts.descriptor_rule import (
    descriptor_feature_guards as _feature_guards,
)
from loom.target.arch.spirv.contracts.descriptor_rule import (
    emit_descriptor_op as _descriptor_emit,
)
from loom.target.arch.spirv.contracts.descriptor_rule import (
    logical_core_descriptor as _descriptor,
)
from loom.target.arch.spirv.ordinary_vector import (
    BOOLEAN_ORDINARY_VECTOR_COMPONENT_TYPE,
    NATIVE_ORDINARY_VECTOR_LANE_COUNTS,
    ORDINARY_VECTOR_COMPONENT_TYPES,
    OrdinaryVectorComponentKind,
    OrdinaryVectorComponentType,
)
from loom.target.arch.spirv.ordinary_vector_integer import (
    ORDINARY_VECTOR_INTEGER_COMPONENT_TYPE_PAIRS,
    OrdinaryVectorIntegerComponentTypePair,
)
from loom.target.arch.spirv.scalar_alu import (
    BOOLEAN_BINARY_OPERATIONS,
    FLOAT_COMPARE_PREDICATES,
    FLOAT_COMPARE_TYPES,
    INTEGER_BITWISE_BINARY_OPERATIONS,
    SIGNED_INTEGER_BINARY_OPERATIONS,
    SIGNED_INTEGER_COMPARE_PREDICATES,
    UNSIGNED_INTEGER_BINARY_OPERATIONS,
    UNSIGNED_ORDERED_INTEGER_COMPARE_PREDICATES,
    ComparePredicate,
    ScalarBinaryOperation,
)
from loom.target.contracts import (
    ContractCase,
    DescriptorResultType,
    DescriptorRule,
    EmitDescriptorOp,
    Guard,
    SourceValueKind,
    TypePattern,
    ValueRef,
    Vector,
)

_VECTOR_BINARY_SOURCE_OPS: dict[str, Op] = {
    "addi": vector.vector_addi,
    "subi": vector.vector_subi,
    "muli": vector.vector_muli,
    "divui": vector.vector_divui,
    "divsi": vector.vector_divsi,
    "remui": vector.vector_remui,
    "remsi": vector.vector_remsi,
    "andi": vector.vector_andi,
    "ori": vector.vector_ori,
    "xori": vector.vector_xori,
    "shli": vector.vector_shli,
    "shrsi": vector.vector_shrsi,
    "shrui": vector.vector_shrui,
}

_INTEGER_COMPARE_PREDICATE_BY_SOURCE = {
    predicate.source_predicate: predicate
    for predicate in (
        *SIGNED_INTEGER_COMPARE_PREDICATES,
        *UNSIGNED_ORDERED_INTEGER_COMPARE_PREDICATES,
    )
}

_INTEGER_EXTREMA_OPERATIONS = (
    (vector.vector_minsi, _INTEGER_COMPARE_PREDICATE_BY_SOURCE["slt"], False),
    (vector.vector_maxsi, _INTEGER_COMPARE_PREDICATE_BY_SOURCE["sgt"], False),
    (vector.vector_minui, _INTEGER_COMPARE_PREDICATE_BY_SOURCE["ult"], True),
    (vector.vector_maxui, _INTEGER_COMPARE_PREDICATE_BY_SOURCE["ugt"], True),
)

_FLOAT_COMPONENT_BY_SUFFIX = {
    component.suffix: component
    for component in ORDINARY_VECTOR_COMPONENT_TYPES
    if component.kind == OrdinaryVectorComponentKind.FLOAT
}

_LANE_COUNTS = (1, *NATIVE_ORDINARY_VECTOR_LANE_COUNTS)


def _suffix(component: OrdinaryVectorComponentType, lane_count: int) -> str:
    return component.suffix if lane_count == 1 else f"v{lane_count}{component.suffix}"


def _source_vector_pattern(
    source_type: str | tuple[str, ...], lane_count: int
) -> TypePattern:
    return Vector(source_type, lanes=lane_count)


def _typed_guards(fields: tuple[str, ...], pattern: TypePattern) -> tuple[Guard, ...]:
    return tuple(Guard.value_type(field, pattern) for field in fields)


def _direct_binary_rule(
    source_op: Op,
    source_type: str,
    component: OrdinaryVectorComponentType,
    lane_count: int,
    operation: ScalarBinaryOperation,
) -> DescriptorRule:
    descriptor = _descriptor(
        f"spirv.op_{operation.descriptor_suffix}.{_suffix(component, lane_count)}"
    )
    pattern = _source_vector_pattern(source_type, lane_count)
    return DescriptorRule(
        source_op=source_op,
        descriptor=descriptor,
        guards=(
            *_typed_guards(("lhs", "rhs", "result"), pattern),
            *_feature_guards(descriptor),
        ),
        emit=(
            _descriptor_emit(
                descriptor=descriptor,
                operands={
                    "lhs": ValueRef.operand("lhs"),
                    "rhs": ValueRef.operand("rhs"),
                },
                results={"dst": ValueRef.result("result")},
            ),
        ),
    )


def _integer_view_emit(
    source_component: OrdinaryVectorComponentType,
    result_component: OrdinaryVectorComponentType,
    lane_count: int,
    *,
    input_ref: ValueRef,
    output_ref: ValueRef,
) -> EmitDescriptorOp:
    descriptor = _descriptor(
        f"spirv.op_bitcast.{_suffix(source_component, lane_count)}."
        f"{_suffix(result_component, lane_count)}"
    )
    result_types = (
        {"dst": DescriptorResultType()}
        if output_ref.kind == SourceValueKind.TEMPORARY
        else None
    )
    return _descriptor_emit(
        descriptor=descriptor,
        operands={"input": input_ref},
        results={"dst": output_ref},
        result_types=result_types,
    )


def _unsigned_binary_rule(
    component_pair: OrdinaryVectorIntegerComponentTypePair,
    lane_count: int,
    operation: ScalarBinaryOperation,
) -> DescriptorRule:
    signed_suffix = _suffix(component_pair.signed, lane_count)
    unsigned_suffix = _suffix(component_pair.unsigned, lane_count)
    to_unsigned = _descriptor(f"spirv.op_bitcast.{signed_suffix}.{unsigned_suffix}")
    descriptor = _descriptor(
        f"spirv.op_{operation.descriptor_suffix}.{unsigned_suffix}"
    )
    to_signed = _descriptor(f"spirv.op_bitcast.{unsigned_suffix}.{signed_suffix}")
    pattern = _source_vector_pattern(component_pair.source_type, lane_count)
    return DescriptorRule(
        source_op=_VECTOR_BINARY_SOURCE_OPS[operation.source_op_key],
        descriptor=descriptor,
        guards=(
            *_typed_guards(("lhs", "rhs", "result"), pattern),
            *_feature_guards(to_unsigned, descriptor, to_signed),
        ),
        emit=(
            _integer_view_emit(
                component_pair.signed,
                component_pair.unsigned,
                lane_count,
                input_ref=ValueRef.operand("lhs"),
                output_ref=ValueRef.temporary("unsigned_lhs"),
            ),
            _integer_view_emit(
                component_pair.signed,
                component_pair.unsigned,
                lane_count,
                input_ref=ValueRef.operand("rhs"),
                output_ref=ValueRef.temporary("unsigned_rhs"),
            ),
            _descriptor_emit(
                descriptor=descriptor,
                operands={
                    "lhs": ValueRef.temporary("unsigned_lhs"),
                    "rhs": ValueRef.temporary("unsigned_rhs"),
                },
                results={"dst": ValueRef.temporary("unsigned_result")},
                result_types={"dst": DescriptorResultType()},
            ),
            _integer_view_emit(
                component_pair.unsigned,
                component_pair.signed,
                lane_count,
                input_ref=ValueRef.temporary("unsigned_result"),
                output_ref=ValueRef.result("result"),
            ),
        ),
    )


def _direct_compare_rule(
    source_op: Op,
    predicate: ComparePredicate,
    source_type: str,
    operand_component: OrdinaryVectorComponentType,
    lane_count: int,
) -> DescriptorRule:
    descriptor = _descriptor(
        f"spirv.op_{predicate.descriptor_suffix}."
        f"{_suffix(operand_component, lane_count)}"
    )
    operand_pattern = _source_vector_pattern(source_type, lane_count)
    result_pattern = _source_vector_pattern("i1", lane_count)
    return DescriptorRule(
        source_op=source_op,
        descriptor=descriptor,
        guards=(
            Guard.enum_attr_equals("predicate", predicate.source_predicate),
            *_typed_guards(("lhs", "rhs"), operand_pattern),
            Guard.value_type("result", result_pattern),
            *_feature_guards(descriptor),
        ),
        emit=(
            _descriptor_emit(
                descriptor=descriptor,
                operands={
                    "lhs": ValueRef.operand("lhs"),
                    "rhs": ValueRef.operand("rhs"),
                },
                results={"dst": ValueRef.result("result")},
            ),
        ),
    )


def _unsigned_compare_rule(
    component_pair: OrdinaryVectorIntegerComponentTypePair,
    lane_count: int,
    predicate: ComparePredicate,
) -> DescriptorRule:
    signed_suffix = _suffix(component_pair.signed, lane_count)
    unsigned_suffix = _suffix(component_pair.unsigned, lane_count)
    to_unsigned = _descriptor(f"spirv.op_bitcast.{signed_suffix}.{unsigned_suffix}")
    descriptor = _descriptor(
        f"spirv.op_{predicate.descriptor_suffix}.{unsigned_suffix}"
    )
    operand_pattern = _source_vector_pattern(component_pair.source_type, lane_count)
    return DescriptorRule(
        source_op=vector.vector_cmpi,
        descriptor=descriptor,
        guards=(
            Guard.enum_attr_equals("predicate", predicate.source_predicate),
            *_typed_guards(("lhs", "rhs"), operand_pattern),
            Guard.value_type("result", _source_vector_pattern("i1", lane_count)),
            *_feature_guards(to_unsigned, descriptor),
        ),
        emit=(
            _integer_view_emit(
                component_pair.signed,
                component_pair.unsigned,
                lane_count,
                input_ref=ValueRef.operand("lhs"),
                output_ref=ValueRef.temporary("unsigned_lhs"),
            ),
            _integer_view_emit(
                component_pair.signed,
                component_pair.unsigned,
                lane_count,
                input_ref=ValueRef.operand("rhs"),
                output_ref=ValueRef.temporary("unsigned_rhs"),
            ),
            _descriptor_emit(
                descriptor=descriptor,
                operands={
                    "lhs": ValueRef.temporary("unsigned_lhs"),
                    "rhs": ValueRef.temporary("unsigned_rhs"),
                },
                results={"dst": ValueRef.result("result")},
            ),
        ),
    )


def _integer_binary_rules() -> tuple[DescriptorRule, ...]:
    rules: list[DescriptorRule] = []
    for component_pair in ORDINARY_VECTOR_INTEGER_COMPONENT_TYPE_PAIRS:
        for lane_count in _LANE_COUNTS:
            rules.extend(
                _direct_binary_rule(
                    _VECTOR_BINARY_SOURCE_OPS[operation.source_op_key],
                    component_pair.source_type,
                    component_pair.signed,
                    lane_count,
                    operation,
                )
                for operation in (
                    *SIGNED_INTEGER_BINARY_OPERATIONS,
                    *INTEGER_BITWISE_BINARY_OPERATIONS,
                )
            )
            rules.extend(
                _unsigned_binary_rule(component_pair, lane_count, operation)
                for operation in UNSIGNED_INTEGER_BINARY_OPERATIONS
            )
    for lane_count in _LANE_COUNTS:
        rules.extend(
            _direct_binary_rule(
                _VECTOR_BINARY_SOURCE_OPS[operation.source_op_key],
                "i1",
                BOOLEAN_ORDINARY_VECTOR_COMPONENT_TYPE,
                lane_count,
                operation,
            )
            for operation in BOOLEAN_BINARY_OPERATIONS
        )
    return tuple(rules)


def _integer_compare_rules() -> tuple[DescriptorRule, ...]:
    rules: list[DescriptorRule] = []
    for component_pair in ORDINARY_VECTOR_INTEGER_COMPONENT_TYPE_PAIRS:
        for lane_count in _LANE_COUNTS:
            rules.extend(
                _direct_compare_rule(
                    vector.vector_cmpi,
                    predicate,
                    component_pair.source_type,
                    component_pair.signed,
                    lane_count,
                )
                for predicate in SIGNED_INTEGER_COMPARE_PREDICATES
            )
            rules.extend(
                _unsigned_compare_rule(component_pair, lane_count, predicate)
                for predicate in UNSIGNED_ORDERED_INTEGER_COMPARE_PREDICATES
            )
    return tuple(rules)


def _integer_extrema_rule(
    source_op: Op,
    component_pair: OrdinaryVectorIntegerComponentTypePair,
    lane_count: int,
    predicate: ComparePredicate,
    uses_unsigned_ordering: bool,
) -> DescriptorRule:
    signed_suffix = _suffix(component_pair.signed, lane_count)
    compare_component = (
        component_pair.unsigned if uses_unsigned_ordering else component_pair.signed
    )
    compare = _descriptor(
        f"spirv.op_{predicate.descriptor_suffix}."
        f"{_suffix(compare_component, lane_count)}"
    )
    select = _descriptor(f"spirv.op_select.{signed_suffix}")
    pattern = _source_vector_pattern(component_pair.source_type, lane_count)
    condition_pattern = _source_vector_pattern("i1", lane_count)

    lhs = ValueRef.operand("lhs")
    rhs = ValueRef.operand("rhs")
    emits: list[EmitDescriptorOp] = []
    if uses_unsigned_ordering:
        unsigned_lhs = ValueRef.temporary("unsigned_lhs")
        unsigned_rhs = ValueRef.temporary("unsigned_rhs")
        emits.extend(
            (
                _integer_view_emit(
                    component_pair.signed,
                    component_pair.unsigned,
                    lane_count,
                    input_ref=lhs,
                    output_ref=unsigned_lhs,
                ),
                _integer_view_emit(
                    component_pair.signed,
                    component_pair.unsigned,
                    lane_count,
                    input_ref=rhs,
                    output_ref=unsigned_rhs,
                ),
            )
        )
        lhs = unsigned_lhs
        rhs = unsigned_rhs

    condition = ValueRef.temporary("condition")
    emits.extend(
        (
            _descriptor_emit(
                descriptor=compare,
                operands={"lhs": lhs, "rhs": rhs},
                results={"dst": condition},
                result_types={"dst": condition_pattern},
            ),
            _descriptor_emit(
                descriptor=select,
                operands={
                    "condition": condition,
                    "true_value": ValueRef.operand("lhs"),
                    "false_value": ValueRef.operand("rhs"),
                },
                results={"dst": ValueRef.result("result")},
            ),
        )
    )
    return DescriptorRule(
        source_op=source_op,
        descriptor=select,
        guards=(
            *_typed_guards(("lhs", "rhs", "result"), pattern),
            *_feature_guards(*(emit.descriptor for emit in emits)),
        ),
        emit=tuple(emits),
    )


def _integer_extrema_rules() -> tuple[DescriptorRule, ...]:
    return tuple(
        _integer_extrema_rule(
            source_op,
            component_pair,
            lane_count,
            predicate,
            uses_unsigned_ordering,
        )
        for component_pair in ORDINARY_VECTOR_INTEGER_COMPONENT_TYPE_PAIRS
        for lane_count in _LANE_COUNTS
        for source_op, predicate, uses_unsigned_ordering in _INTEGER_EXTREMA_OPERATIONS
    )


def _float_compare_rules() -> tuple[DescriptorRule, ...]:
    return tuple(
        _direct_compare_rule(
            vector.vector_cmpf,
            predicate,
            scalar.source_type,
            _FLOAT_COMPONENT_BY_SUFFIX[scalar.suffix],
            lane_count,
        )
        for scalar in FLOAT_COMPARE_TYPES
        for lane_count in _LANE_COUNTS
        for predicate in FLOAT_COMPARE_PREDICATES
    )


def _select_rule(
    component: OrdinaryVectorComponentType, lane_count: int
) -> DescriptorRule:
    descriptor = _descriptor(f"spirv.op_select.{_suffix(component, lane_count)}")
    value_pattern = _source_vector_pattern(component.source_types, lane_count)
    return DescriptorRule(
        source_op=vector.vector_select,
        descriptor=descriptor,
        guards=(
            Guard.value_type("condition", _source_vector_pattern("i1", lane_count)),
            *_typed_guards(("true_value", "false_value", "result"), value_pattern),
            *_feature_guards(descriptor),
        ),
        emit=(
            _descriptor_emit(
                descriptor=descriptor,
                operands={
                    "condition": ValueRef.operand("condition"),
                    "true_value": ValueRef.operand("true_value"),
                    "false_value": ValueRef.operand("false_value"),
                },
                results={"dst": ValueRef.result("result")},
            ),
        ),
    )


def _select_rules() -> tuple[DescriptorRule, ...]:
    return tuple(
        _select_rule(component, lane_count)
        for component in ORDINARY_VECTOR_COMPONENT_TYPES
        for lane_count in _LANE_COUNTS
    )


SPIRV_ORDINARY_VECTOR_ALU_CONTRACT_CASES: tuple[ContractCase, ...] = (
    *_integer_binary_rules(),
    *_integer_extrema_rules(),
    *_integer_compare_rules(),
    *_float_compare_rules(),
    *_select_rules(),
)
