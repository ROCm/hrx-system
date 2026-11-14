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
from loom.target.arch.spirv.ordinary_vector_float import (
    FUSED_MULTIPLY_ADD_INSTRUCTIONS,
    FusedMultiplyAddInstruction,
)
from loom.target.contracts import (
    DescriptorEmitForm,
    DescriptorResultType,
    DescriptorRule,
    EmitDescriptorOp,
    Guard,
    Scalar,
    ValueRef,
    Vector,
)


def _fused_multiply_add_emits(
    instruction: FusedMultiplyAddInstruction,
) -> tuple[EmitDescriptorOp, ...]:
    # Vulkan permits signaling NaNs to survive floating-point instructions.
    # Preserve the native rounded result, but make arithmetic NaNs quiet.
    quiet_nan = "nnan" not in instruction.required_fastmath_flags
    result = ValueRef.temporary("fused") if quiet_nan else ValueRef.result("result")
    emits = [
        emit_descriptor_op(
            descriptor=logical_core_descriptor(instruction.key),
            operands={name: ValueRef.operand(name) for name in ("a", "b", "c")},
            results={"dst": result},
            result_types={"dst": DescriptorResultType()} if quiet_nan else None,
        )
    ]
    if not quiet_nan:
        return tuple(emits)
    value_type = instruction.result_type
    scalar = (
        value_type.component_type
        if isinstance(value_type, OrdinaryVectorType)
        else value_type
    )
    nan = ValueRef.temporary("nan")
    quiet = ValueRef.temporary("quiet_nan")
    emits.extend(
        (
            emit_descriptor_op(
                descriptor=logical_core_descriptor(
                    f"spirv.op_f_unord_not_equal.{value_type.suffix}"
                ),
                operands={"lhs": result, "rhs": result},
                results={"dst": nan},
                result_types={"dst": DescriptorResultType()},
            ),
            EmitDescriptorOp(
                descriptor=logical_core_descriptor(
                    f"spirv.op_constant.{scalar.suffix}"
                ),
                results={"dst": quiet},
                result_types={"dst": DescriptorResultType()},
                immediates={
                    f"{scalar.suffix}_bits": {
                        "f16": 0x7E00,
                        "f32": 0x7FC00000,
                        "f64": 0x7FF8000000000000,
                    }[scalar.suffix]
                },
                form=DescriptorEmitForm.CONST,
            ),
        )
    )
    if isinstance(value_type, OrdinaryVectorType):
        lanes = ValueRef.temporary("quiet_nan_lanes")
        emits.append(
            emit_descriptor_op(
                descriptor=logical_core_descriptor(
                    f"spirv.op_composite_construct.{value_type.suffix}"
                ),
                operands={
                    f"component{lane}": quiet for lane in range(value_type.lane_count)
                },
                results={"dst": lanes},
                result_types={"dst": DescriptorResultType()},
            )
        )
        quiet = lanes
    emits.append(
        emit_descriptor_op(
            descriptor=logical_core_descriptor(f"spirv.op_select.{value_type.suffix}"),
            operands={"condition": nan, "true_value": quiet, "false_value": result},
            results={"dst": ValueRef.result("result")},
        )
    )
    return tuple(emits)


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
                    emit=_fused_multiply_add_emits(instruction),
                )
            )
    return tuple(rules)
