# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Capability-qualified, single-rounding scalar and vector FMA contracts."""

from loom.dialect.scalar import math as scalar_math
from loom.dialect.vector import defs as vector
from loom.target.arch.spirv.contracts.descriptor_rule import (
    descriptor_feature_guards,
    emit_descriptor_op,
    logical_core_descriptor,
)
from loom.target.arch.spirv.ordinary_vector import OrdinaryVectorType
from loom.target.arch.spirv.ordinary_vector_float import FUSED_MULTIPLY_ADD_INSTRUCTIONS
from loom.target.contracts import DescriptorRule, Guard, Scalar, ValueRef, Vector


def fused_multiply_add_rules() -> tuple[DescriptorRule, ...]:
    rules = []
    for instruction in FUSED_MULTIPLY_ADD_INSTRUCTIONS:
        value_type = instruction.result_type
        if isinstance(value_type, OrdinaryVectorType):
            source_shapes = (
                (
                    vector.vector_fmaf,
                    Vector(
                        value_type.component_type.source_types[0],
                        lanes=value_type.lane_count,
                    ),
                ),
            )
        else:
            source_shapes = (
                (scalar_math.scalar_fmaf, Scalar(value_type.source_types[0])),
                (vector.vector_fmaf, Vector(value_type.source_types[0], lanes=1)),
            )
        descriptor = logical_core_descriptor(instruction.key)
        for source_op, source_type in source_shapes:
            rules.append(
                DescriptorRule(
                    source_op=source_op,
                    descriptor=descriptor,
                    # Prefer the least restrictive environment allowed by the source.
                    priority=len(instruction.required_fastmath_flags),
                    guards=(
                        # Identify the type family before considering permissions;
                        # then reject inapplicable policies before matching operands.
                        Guard.value_type("a", source_type),
                        *(
                            Guard.instance_flags_has_all("fastmath", flag)
                            for flag in instruction.required_fastmath_flags
                        ),
                        *(
                            Guard.value_type(name, source_type)
                            for name in ("b", "c", "result")
                        ),
                        *descriptor_feature_guards(descriptor),
                    ),
                    emit=(
                        emit_descriptor_op(
                            descriptor=descriptor,
                            operands={
                                name: ValueRef.operand(name) for name in ("a", "b", "c")
                            },
                            results={"dst": ValueRef.result("result")},
                        ),
                    ),
                )
            )
    return tuple(rules)
