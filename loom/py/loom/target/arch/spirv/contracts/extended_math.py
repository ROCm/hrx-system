# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""SPIR-V source-to-low rules for GLSL.std.450 math instructions."""

from __future__ import annotations

from loom.dialect.scalar import math as scalar_math
from loom.dialect.vector import defs as vector
from loom.dsl import Op
from loom.target.arch.spirv.contracts.descriptor_rule import (
    descriptor_feature_guards,
    emit_descriptor_op,
    logical_core_descriptor,
)
from loom.target.arch.spirv.extended_math import (
    EXTENDED_MATH_INSTRUCTIONS,
    EXTENDED_MATH_OPERATIONS,
    ExtendedMathInstruction,
)
from loom.target.arch.spirv.ordinary_vector import (
    OrdinaryVectorComponentType,
    OrdinaryVectorType,
)
from loom.target.contracts import (
    ContractCase,
    DescriptorRule,
    Guard,
    Scalar,
    TypePattern,
    ValueRef,
    Vector,
)

_SCALAR_SOURCE_OPS = {
    "sinf": scalar_math.scalar_sinf,
    "cosf": scalar_math.scalar_cosf,
    "expf": scalar_math.scalar_expf,
    "logf": scalar_math.scalar_logf,
    "exp2f": scalar_math.scalar_exp2f,
    "log2f": scalar_math.scalar_log2f,
    "sqrtf": scalar_math.scalar_sqrtf,
    "rsqrtf": scalar_math.scalar_rsqrtf,
}

_VECTOR_SOURCE_OPS = {
    "sinf": vector.vector_sinf,
    "cosf": vector.vector_cosf,
    "expf": vector.vector_expf,
    "logf": vector.vector_logf,
    "exp2f": vector.vector_exp2f,
    "log2f": vector.vector_log2f,
    "sqrtf": vector.vector_sqrtf,
    "rsqrtf": vector.vector_rsqrtf,
}


def _extended_math_rule(
    instruction: ExtendedMathInstruction,
    source_op: Op,
    source_type: TypePattern,
) -> DescriptorRule:
    descriptor = logical_core_descriptor(instruction.descriptor_key)
    return DescriptorRule(
        source_op=source_op,
        descriptor=descriptor,
        guards=(
            *(
                Guard.value_type(operand_name, source_type)
                for operand_name in instruction.operation.operand_names
            ),
            Guard.value_type("result", source_type),
            *(
                Guard.instance_flags_has_all(source_op.attrs[0].name, fastmath_flag)
                for fastmath_flag in instruction.operation.required_fastmath_flags
            ),
            *descriptor_feature_guards(descriptor),
        ),
        emit=(
            emit_descriptor_op(
                descriptor=descriptor,
                operands={
                    operand_name: ValueRef.operand(operand_name)
                    for operand_name in instruction.operation.operand_names
                },
                results={"dst": ValueRef.result("result")},
            ),
        ),
    )


def _extended_math_cases() -> tuple[ContractCase, ...]:
    cases: list[ContractCase] = []
    for instruction in EXTENDED_MATH_INSTRUCTIONS:
        operation_key = instruction.operation.source_op_key
        if instruction.operation not in EXTENDED_MATH_OPERATIONS:
            continue
        component_type = (
            instruction.value_type.component_type
            if isinstance(instruction.value_type, OrdinaryVectorType)
            else instruction.value_type
        )
        source_type = component_type.source_types[0]
        if isinstance(instruction.value_type, OrdinaryVectorComponentType):
            cases.append(
                _extended_math_rule(
                    instruction,
                    _SCALAR_SOURCE_OPS[operation_key],
                    Scalar(source_type),
                )
            )
            cases.append(
                _extended_math_rule(
                    instruction,
                    _VECTOR_SOURCE_OPS[operation_key],
                    Vector(source_type, lanes=1),
                )
            )
            continue
        assert isinstance(instruction.value_type, OrdinaryVectorType)
        cases.append(
            _extended_math_rule(
                instruction,
                _VECTOR_SOURCE_OPS[operation_key],
                Vector(source_type, lanes=instruction.value_type.lane_count),
            )
        )
    return tuple(cases)


SPIRV_EXTENDED_MATH_CONTRACT_CASES = _extended_math_cases()
