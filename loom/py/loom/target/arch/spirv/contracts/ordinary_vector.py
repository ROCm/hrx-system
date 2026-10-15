# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""SPIR-V ordinary-vector source-to-low contract rows."""

from __future__ import annotations

from loom.dialect.scf import defs as scf
from loom.dialect.vector import defs as vector
from loom.target.arch.spirv.descriptors import SPIRV_LOGICAL_CORE_DESCRIPTOR_SET
from loom.target.arch.spirv.ordinary_vector import (
    BOOLEAN_ORDINARY_VECTOR_COMPONENT_TYPE,
    NATIVE_ORDINARY_VECTOR_LANE_COUNTS,
    ORDINARY_VECTOR_COMPONENT_TYPES,
    ORDINARY_VECTOR_SHUFFLE_INSTRUCTIONS,
    ORDINARY_VECTOR_TYPES,
    OrdinaryVectorComponentKind,
    OrdinaryVectorComponentType,
    OrdinaryVectorShuffleInstruction,
    OrdinaryVectorType,
)
from loom.target.arch.spirv.ordinary_vector_bit_layout import (
    ORDINARY_VECTOR_BIT_LAYOUT_ALIAS_CASES,
    ORDINARY_VECTOR_BIT_LAYOUT_DIRECT_CASES,
    OrdinaryVectorBitLayoutCase,
    OrdinaryVectorBitLayoutType,
)
from loom.target.contracts import (
    AttrProject,
    ContractCase,
    DescriptorEmitForm,
    DescriptorResultType,
    DescriptorRule,
    EmitDescriptorOp,
    Guard,
    ResultTypeBinding,
    Scalar,
    TypePattern,
    ValueAliasRule,
    ValueProject,
    ValueRef,
    Vector,
    descriptor_by_key,
)
from loom.target.low_descriptors import Descriptor


def _descriptor(key: str) -> Descriptor:
    return descriptor_by_key(SPIRV_LOGICAL_CORE_DESCRIPTOR_SET, key)


def _feature_guards(*descriptors: Descriptor) -> tuple[Guard, ...]:
    return tuple(
        Guard.descriptor_available(descriptor)
        for descriptor in descriptors
        if descriptor.feature_mask_words
    )


def _emit(
    descriptor: Descriptor,
    *,
    operands: dict[str, ValueRef] | None = None,
    results: dict[str, ValueRef] | None = None,
    result_types: dict[str, ResultTypeBinding] | None = None,
    immediates: dict[str, AttrProject | ValueProject] | None = None,
    form: DescriptorEmitForm = DescriptorEmitForm.OP,
) -> EmitDescriptorOp:
    return EmitDescriptorOp(
        descriptor=descriptor,
        operands={} if operands is None else operands,
        results={} if results is None else results,
        result_types=result_types,
        immediates={} if immediates is None else immediates,
        form=form,
    )


_SOURCE_COMPONENT_TYPES = tuple(
    source_type
    for component_type in ORDINARY_VECTOR_COMPONENT_TYPES
    for source_type in component_type.source_types
)
_SOURCE_SCALAR_PATTERN = Scalar(_SOURCE_COMPONENT_TYPES)
_SOURCE_V1_PATTERN = Vector(_SOURCE_COMPONENT_TYPES, lanes=1)


def _singleton_rules() -> tuple[ValueAliasRule, ...]:
    static_index_guards = (
        Guard.operand_segment_count("indices", 0),
        Guard.i64_array_count("static_indices", 1),
        Guard.i64_array_element_range(
            "static_indices", element=0, minimum=0, maximum=0
        ),
    )
    return (
        ValueAliasRule(
            source_op=vector.vector_splat,
            source=ValueRef.operand("scalar"),
            result=ValueRef.result("result"),
            guards=(
                Guard.value_type("scalar", _SOURCE_SCALAR_PATTERN),
                Guard.value_type("result", _SOURCE_V1_PATTERN),
            ),
        ),
        ValueAliasRule(
            source_op=vector.vector_from_elements,
            source=ValueRef.operand("elements", element=0),
            result=ValueRef.result("result"),
            guards=(
                Guard.operand_segment_count("elements", 1),
                Guard.value_type("elements", _SOURCE_SCALAR_PATTERN),
                Guard.value_type("result", _SOURCE_V1_PATTERN),
            ),
        ),
        ValueAliasRule(
            source_op=vector.vector_extract,
            source=ValueRef.operand("source"),
            result=ValueRef.result("result"),
            guards=(
                Guard.value_type("source", _SOURCE_V1_PATTERN),
                Guard.value_type("result", _SOURCE_SCALAR_PATTERN),
                *static_index_guards,
            ),
        ),
        ValueAliasRule(
            source_op=vector.vector_insert,
            source=ValueRef.operand("value"),
            result=ValueRef.result("result"),
            guards=(
                Guard.value_type("value", _SOURCE_SCALAR_PATTERN),
                Guard.value_type("dest", _SOURCE_V1_PATTERN),
                Guard.value_type("result", _SOURCE_V1_PATTERN),
                *static_index_guards,
            ),
        ),
    )


def _select_descriptor_for_component(
    component_type: OrdinaryVectorComponentType,
) -> Descriptor:
    return _descriptor(f"spirv.op_select.{component_type.suffix}")


def _singleton_select_rule(
    component_type: OrdinaryVectorComponentType,
) -> DescriptorRule:
    descriptor = _select_descriptor_for_component(component_type)
    vector_pattern = Vector(component_type.source_types, lanes=1)
    return DescriptorRule(
        source_op=scf.scf_select,
        descriptor=descriptor,
        guards=(
            Guard.value_type("condition", Scalar("i1")),
            Guard.value_type("true_value", vector_pattern),
            Guard.value_type("false_value", vector_pattern),
            Guard.value_type("result", vector_pattern),
            *_feature_guards(descriptor),
        ),
        emit=(
            _emit(
                descriptor,
                operands={
                    "condition": ValueRef.operand("condition"),
                    "true_value": ValueRef.operand("true_value"),
                    "false_value": ValueRef.operand("false_value"),
                },
                results={"dst": ValueRef.result("result")},
            ),
        ),
    )


def _component_pattern(component_type: OrdinaryVectorComponentType) -> TypePattern:
    return Scalar(component_type.source_types)


def _vector_pattern(vector_type: OrdinaryVectorType) -> TypePattern:
    return Vector(
        vector_type.component_type.source_types,
        lanes=vector_type.lane_count,
    )


def _bit_layout_pattern(layout_type: OrdinaryVectorBitLayoutType) -> TypePattern:
    return Vector(layout_type.element_type, lanes=layout_type.lane_count)


def _bitcast_alias_rule(case: OrdinaryVectorBitLayoutCase) -> ValueAliasRule:
    return ValueAliasRule(
        source_op=vector.vector_bitcast,
        source=ValueRef.operand("input"),
        result=ValueRef.result("result"),
        guards=(
            Guard.value_type("input", _bit_layout_pattern(case.source)),
            Guard.value_type("result", _bit_layout_pattern(case.result)),
        ),
    )


def _bitcast_rule(case: OrdinaryVectorBitLayoutCase) -> DescriptorRule:
    descriptor = _descriptor(case.key)
    return DescriptorRule(
        source_op=vector.vector_bitcast,
        descriptor=descriptor,
        guards=(
            Guard.value_type("input", _bit_layout_pattern(case.source)),
            Guard.value_type("result", _bit_layout_pattern(case.result)),
            *_feature_guards(descriptor),
        ),
        emit=(
            _emit(
                descriptor,
                operands={"input": ValueRef.operand("input")},
                results={"dst": ValueRef.result("result")},
            ),
        ),
    )


def _construct_descriptor(vector_type: OrdinaryVectorType) -> Descriptor:
    return _descriptor(f"spirv.op_composite_construct.{vector_type.suffix}")


def _extract_descriptor(vector_type: OrdinaryVectorType) -> Descriptor:
    return _descriptor(
        f"spirv.op_composite_extract.{vector_type.suffix}."
        f"{vector_type.component_type.suffix}"
    )


def _insert_descriptor(vector_type: OrdinaryVectorType) -> Descriptor:
    return _descriptor(
        f"spirv.op_composite_insert.{vector_type.component_type.suffix}."
        f"{vector_type.suffix}"
    )


_SOURCE_SHUFFLE_BY_TYPE = {
    instruction.source_type: instruction
    for instruction in ORDINARY_VECTOR_SHUFFLE_INSTRUCTIONS
    if instruction.source_type == instruction.result_type
}
if _SOURCE_SHUFFLE_BY_TYPE.keys() != set(ORDINARY_VECTOR_TYPES):
    raise ValueError("source vector shuffle instructions must cover native types")


def _splat_rule(vector_type: OrdinaryVectorType) -> DescriptorRule:
    descriptor = _construct_descriptor(vector_type)
    scalar = ValueRef.operand("scalar")
    return DescriptorRule(
        source_op=vector.vector_splat,
        descriptor=descriptor,
        guards=(
            Guard.value_type("scalar", _component_pattern(vector_type.component_type)),
            Guard.value_type("result", _vector_pattern(vector_type)),
            *_feature_guards(descriptor),
        ),
        emit=(
            _emit(
                descriptor,
                operands={
                    f"component{lane_index}": scalar
                    for lane_index in range(vector_type.lane_count)
                },
                results={"dst": ValueRef.result("result")},
            ),
        ),
    )


def _from_elements_rule(vector_type: OrdinaryVectorType) -> DescriptorRule:
    descriptor = _construct_descriptor(vector_type)
    return DescriptorRule(
        source_op=vector.vector_from_elements,
        descriptor=descriptor,
        guards=(
            Guard.operand_segment_count("elements", vector_type.lane_count),
            Guard.value_type(
                "elements", _component_pattern(vector_type.component_type)
            ),
            Guard.value_type("result", _vector_pattern(vector_type)),
            *_feature_guards(descriptor),
        ),
        emit=(
            _emit(
                descriptor,
                operands={
                    f"component{lane_index}": ValueRef.operand(
                        "elements", element=lane_index
                    )
                    for lane_index in range(vector_type.lane_count)
                },
                results={"dst": ValueRef.result("result")},
            ),
        ),
    )


def _extract_rule(vector_type: OrdinaryVectorType) -> DescriptorRule:
    descriptor = _extract_descriptor(vector_type)
    return DescriptorRule(
        source_op=vector.vector_extract,
        descriptor=descriptor,
        guards=(
            Guard.value_type("source", _vector_pattern(vector_type)),
            Guard.value_type("result", _component_pattern(vector_type.component_type)),
            Guard.operand_segment_count("indices", 0),
            Guard.i64_array_count("static_indices", 1),
            Guard.i64_array_element_range(
                "static_indices",
                element=0,
                minimum=0,
                maximum=vector_type.lane_count - 1,
            ),
            *_feature_guards(descriptor),
        ),
        emit=(
            _emit(
                descriptor,
                operands={"composite": ValueRef.operand("source")},
                results={"dst": ValueRef.result("result")},
                immediates={
                    "component_index": AttrProject.i64_array_element(
                        "static_indices", element=0
                    )
                },
            ),
        ),
    )


def _insert_rule(vector_type: OrdinaryVectorType) -> DescriptorRule:
    descriptor = _insert_descriptor(vector_type)
    vector_pattern = _vector_pattern(vector_type)
    return DescriptorRule(
        source_op=vector.vector_insert,
        descriptor=descriptor,
        guards=(
            Guard.value_type("value", _component_pattern(vector_type.component_type)),
            Guard.value_type("dest", vector_pattern),
            Guard.value_type("result", vector_pattern),
            Guard.operand_segment_count("indices", 0),
            Guard.i64_array_count("static_indices", 1),
            Guard.i64_array_element_range(
                "static_indices",
                element=0,
                minimum=0,
                maximum=vector_type.lane_count - 1,
            ),
            *_feature_guards(descriptor),
        ),
        emit=(
            _emit(
                descriptor,
                operands={
                    "component": ValueRef.operand("value"),
                    "composite": ValueRef.operand("dest"),
                },
                results={"dst": ValueRef.result("result")},
                immediates={
                    "component_index": AttrProject.i64_array_element(
                        "static_indices", element=0
                    )
                },
            ),
        ),
    )


def _shuffle_rule(
    instruction: OrdinaryVectorShuffleInstruction,
) -> DescriptorRule:
    descriptor = _descriptor(instruction.key)
    source_pattern = _vector_pattern(instruction.source_type)
    result_pattern = _vector_pattern(instruction.result_type)
    return DescriptorRule(
        source_op=vector.vector_shuffle,
        descriptor=descriptor,
        guards=(
            Guard.value_type("source", source_pattern),
            Guard.value_type("result", result_pattern),
            Guard.i64_array_count("source_lanes", instruction.result_type.lane_count),
            Guard.i64_array_elements_range(
                "source_lanes", 0, instruction.source_type.lane_count - 1
            ),
            *_feature_guards(descriptor),
        ),
        emit=(
            _emit(
                descriptor,
                operands={
                    "source0": ValueRef.operand("source"),
                    "source1": ValueRef.operand("source"),
                },
                results={"dst": ValueRef.result("result")},
                immediates={
                    name: AttrProject.i64_array_element(
                        "source_lanes", element=lane_index
                    )
                    for lane_index, name in enumerate(instruction.selector_names)
                },
            ),
        ),
    )


def _select_rule(vector_type: OrdinaryVectorType) -> DescriptorRule:
    descriptor = _descriptor(f"spirv.op_select.{vector_type.suffix}")
    condition_vector_type = OrdinaryVectorType(
        BOOLEAN_ORDINARY_VECTOR_COMPONENT_TYPE,
        vector_type.lane_count,
    )
    condition_construct_descriptor = _construct_descriptor(condition_vector_type)
    condition = ValueRef.operand("condition")
    vector_condition = ValueRef.temporary("vector_condition")
    vector_pattern = _vector_pattern(vector_type)
    return DescriptorRule(
        source_op=scf.scf_select,
        descriptor=descriptor,
        guards=(
            Guard.value_type("condition", Scalar("i1")),
            Guard.value_type("true_value", vector_pattern),
            Guard.value_type("false_value", vector_pattern),
            Guard.value_type("result", vector_pattern),
            *_feature_guards(condition_construct_descriptor, descriptor),
        ),
        emit=(
            _emit(
                condition_construct_descriptor,
                operands={
                    f"component{lane_index}": condition
                    for lane_index in range(vector_type.lane_count)
                },
                results={"dst": vector_condition},
                result_types={"dst": _vector_pattern(condition_vector_type)},
            ),
            _emit(
                descriptor,
                operands={
                    "condition": vector_condition,
                    "true_value": ValueRef.operand("true_value"),
                    "false_value": ValueRef.operand("false_value"),
                },
                results={"dst": ValueRef.result("result")},
            ),
        ),
    )


def _constant_descriptor(
    component_type: OrdinaryVectorComponentType,
    *,
    boolean_value: int | None = None,
) -> Descriptor:
    if component_type.kind == OrdinaryVectorComponentKind.BOOLEAN:
        if boolean_value is None:
            raise ValueError("boolean constant requires an exact value")
        suffix = "false" if boolean_value == 0 else "true"
        return _descriptor(f"spirv.op_constant_{suffix}.bool")
    if component_type.kind == OrdinaryVectorComponentKind.OFFSET:
        return _descriptor("spirv.op_constant.offset64")
    return _descriptor(f"spirv.op_constant.{component_type.suffix}")


def _constant_guards(
    component_type: OrdinaryVectorComponentType,
    result_pattern: TypePattern,
    descriptor: Descriptor,
    *,
    boolean_value: int | None = None,
) -> tuple[Guard, ...]:
    guards = [Guard.value_type("result", result_pattern)]
    if component_type.kind == OrdinaryVectorComponentKind.BOOLEAN:
        if boolean_value is None:
            raise ValueError("boolean constant requires an exact value")
        guards.append(Guard.value_i64_range("result", boolean_value, boolean_value))
    elif component_type.kind == OrdinaryVectorComponentKind.FLOAT:
        guards.extend(
            (
                Guard.attr_kind("value", "f64"),
                Guard.value_exact_float("result"),
            )
        )
    else:
        guards.append(Guard.attr_kind("value", "i64"))
        if component_type.kind == OrdinaryVectorComponentKind.SIGNED_INTEGER:
            guards.append(
                Guard.i64_range(
                    "value",
                    component_type.signed_minimum,
                    component_type.signed_maximum,
                )
            )
    guards.extend(_feature_guards(descriptor))
    return tuple(guards)


def _constant_immediates(
    component_type: OrdinaryVectorComponentType,
) -> dict[str, AttrProject | ValueProject]:
    if component_type.kind == OrdinaryVectorComponentKind.BOOLEAN:
        return {}
    if component_type.kind == OrdinaryVectorComponentKind.FLOAT:
        return {
            f"{component_type.source_types[0]}_bits": ValueProject.float_bits("result")
        }
    if component_type.kind == OrdinaryVectorComponentKind.OFFSET:
        return {"offset64_value": AttrProject.direct("value")}
    return {f"{component_type.suffix}_value": AttrProject.direct("value")}


def _constant_rule(
    component_type: OrdinaryVectorComponentType,
    lane_count: int | None,
    *,
    boolean_value: int | None = None,
) -> DescriptorRule:
    scalar_descriptor = _constant_descriptor(
        component_type, boolean_value=boolean_value
    )
    result_pattern = (
        Vector(component_type.source_types, lanes=lane_count)
        if lane_count is not None
        else Vector(
            component_type.source_types,
            minimum_lanes=5,
            maximum_lanes="LOOM_DIM_MAX_STATIC_SIZE",
        )
    )
    if lane_count is None or lane_count == 1:
        return DescriptorRule(
            source_op=vector.vector_constant,
            descriptor=scalar_descriptor,
            guards=_constant_guards(
                component_type,
                result_pattern,
                scalar_descriptor,
                boolean_value=boolean_value,
            ),
            emit=(
                _emit(
                    scalar_descriptor,
                    results={"dst": ValueRef.result("result")},
                    result_types={"dst": DescriptorResultType()},
                    immediates=_constant_immediates(component_type),
                    form=DescriptorEmitForm.CONST,
                ),
            ),
        )

    vector_type = OrdinaryVectorType(component_type, lane_count)
    construct_descriptor = _construct_descriptor(vector_type)
    temporary = ValueRef.temporary("component")
    return DescriptorRule(
        source_op=vector.vector_constant,
        descriptor=construct_descriptor,
        guards=(
            *_constant_guards(
                component_type,
                result_pattern,
                scalar_descriptor,
                boolean_value=boolean_value,
            ),
            *_feature_guards(construct_descriptor),
        ),
        emit=(
            _emit(
                scalar_descriptor,
                results={"dst": temporary},
                result_types={"dst": Scalar(component_type.source_types[0])},
                immediates=_constant_immediates(component_type),
                form=DescriptorEmitForm.CONST,
            ),
            _emit(
                construct_descriptor,
                operands={
                    f"component{lane_index}": temporary
                    for lane_index in range(lane_count)
                },
                results={"dst": ValueRef.result("result")},
            ),
        ),
    )


def _constant_rules() -> tuple[DescriptorRule, ...]:
    rules: list[DescriptorRule] = []
    for component_type in ORDINARY_VECTOR_COMPONENT_TYPES:
        boolean_values = (
            (0, 1)
            if component_type.kind == OrdinaryVectorComponentKind.BOOLEAN
            else (None,)
        )
        for lane_count in (1, *NATIVE_ORDINARY_VECTOR_LANE_COUNTS, None):
            rules.extend(
                _constant_rule(
                    component_type,
                    lane_count,
                    boolean_value=boolean_value,
                )
                for boolean_value in boolean_values
            )
    return tuple(rules)


def _native_rules() -> tuple[DescriptorRule, ...]:
    rules: list[DescriptorRule] = []
    for component_type in ORDINARY_VECTOR_COMPONENT_TYPES:
        for lane_count in NATIVE_ORDINARY_VECTOR_LANE_COUNTS:
            vector_type = OrdinaryVectorType(component_type, lane_count)
            rules.extend(
                (
                    _splat_rule(vector_type),
                    _from_elements_rule(vector_type),
                    _extract_rule(vector_type),
                    _insert_rule(vector_type),
                    _shuffle_rule(_SOURCE_SHUFFLE_BY_TYPE[vector_type]),
                    _select_rule(vector_type),
                )
            )
    return tuple(rules)


SPIRV_ORDINARY_VECTOR_CONTRACT_CASES: tuple[ContractCase, ...] = (
    *_singleton_rules(),
    # Wider ordinary operations are legalized before selection. A surviving
    # uniform producer carries its one component for fact-based consumers.
    ValueAliasRule(
        source_op=vector.vector_splat,
        source=ValueRef.operand("scalar"),
        result=ValueRef.result("result"),
        guards=(
            Guard.value_type("scalar", _SOURCE_SCALAR_PATTERN),
            Guard.value_type(
                "result",
                Vector(
                    _SOURCE_COMPONENT_TYPES,
                    minimum_lanes=5,
                    maximum_lanes="LOOM_DIM_MAX_STATIC_SIZE",
                ),
            ),
        ),
    ),
    *(_bitcast_alias_rule(case) for case in ORDINARY_VECTOR_BIT_LAYOUT_ALIAS_CASES),
    *(_bitcast_rule(case) for case in ORDINARY_VECTOR_BIT_LAYOUT_DIRECT_CASES),
    *(
        _singleton_select_rule(component_type)
        for component_type in ORDINARY_VECTOR_COMPONENT_TYPES
    ),
    *_constant_rules(),
    *_native_rules(),
)
