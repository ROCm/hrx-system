# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Shared x86 scalar and uniform-vector constant lowering rules."""

from __future__ import annotations

from loom.dialect.scalar import conversion as scalar_conversion
from loom.dialect.vector import defs as vector
from loom.target.contracts import (
    DescriptorEmitForm,
    DescriptorRule,
    EmitDescriptorOp,
    Guard,
    Scalar,
    TypePattern,
    ValueProject,
    ValueRef,
)
from loom.target.low_descriptors import Descriptor

_I32 = Scalar("i32")
_I64 = Scalar("i64")
_F32 = Scalar("f32")
_F64 = Scalar("f64")


def integer_vector_splat_zero_rule(
    scalar_type: TypePattern,
    result_type: TypePattern,
    descriptor: Descriptor,
    *,
    result_register_class: str | None = None,
) -> DescriptorRule:
    """Constructs an exact integer-zero splat with a dependency-breaking op."""
    return DescriptorRule(
        source_op=vector.vector_splat,
        descriptor=descriptor,
        guards=(
            Guard.value_type("scalar", scalar_type),
            Guard.value_i64_range("scalar", 0, 0),
            Guard.value_type("result", result_type),
            *(
                (Guard.low_value_register_class("result", result_register_class),)
                if result_register_class is not None
                else ()
            ),
        ),
        emit=(
            EmitDescriptorOp(
                descriptor=descriptor,
                results={"dst": ValueRef.result("result")},
                form=DescriptorEmitForm.OP,
            ),
        ),
        priority=2,
    )


def integer_vector_splat_rule(
    scalar_type: TypePattern,
    result_type: TypePattern,
    lane_move_descriptor: Descriptor | None,
    broadcast_descriptor: Descriptor,
    *,
    broadcast_operand: str,
    broadcast_immediates: dict[str, int] | None = None,
    lane_type: TypePattern | None = None,
    priority: int = 0,
    result_register_class: str | None = None,
) -> DescriptorRule:
    """Broadcasts an integer scalar directly or through an XMM transfer."""
    if lane_move_descriptor is None:
        emits = (
            EmitDescriptorOp(
                descriptor=broadcast_descriptor,
                operands={broadcast_operand: ValueRef.operand("scalar")},
                results={"dst": ValueRef.result("result")},
                immediates=(
                    {} if broadcast_immediates is None else broadcast_immediates
                ),
                form=DescriptorEmitForm.OP,
            ),
        )
    else:
        scalar = ValueRef.temporary("scalar")
        emits = (
            EmitDescriptorOp(
                descriptor=lane_move_descriptor,
                operands={"input": ValueRef.operand("scalar")},
                results={"dst": scalar},
                result_types={"dst": result_type if lane_type is None else lane_type},
                form=DescriptorEmitForm.OP,
            ),
            EmitDescriptorOp(
                descriptor=broadcast_descriptor,
                operands={broadcast_operand: scalar},
                results={"dst": ValueRef.result("result")},
                immediates=(
                    {} if broadcast_immediates is None else broadcast_immediates
                ),
                form=DescriptorEmitForm.OP,
            ),
        )
    return DescriptorRule(
        source_op=vector.vector_splat,
        descriptor=broadcast_descriptor,
        guards=(
            Guard.value_type("scalar", scalar_type),
            Guard.value_type("result", result_type),
            *(
                (Guard.low_value_register_class("result", result_register_class),)
                if result_register_class is not None
                else ()
            ),
        ),
        emit=emits,
        priority=priority,
    )


def floating_scalar_zero_rule(
    result_type: TypePattern,
    descriptor: Descriptor,
) -> DescriptorRule:
    """Constructs exact positive floating zero with a dependency-breaking op."""
    return DescriptorRule(
        source_op=scalar_conversion.scalar_constant,
        descriptor=descriptor,
        guards=(
            Guard.attr_kind("value", "f64"),
            Guard.value_type("result", result_type),
            Guard.value_float_equals("result", 0.0),
        ),
        emit=(
            EmitDescriptorOp(
                descriptor=descriptor,
                results={"dst": ValueRef.result("result")},
                form=DescriptorEmitForm.OP,
            ),
        ),
        priority=1,
    )


def floating_scalar_constant_bits_rule(
    result_type: TypePattern,
    integer_type: TypePattern,
    immediate_name: str,
    immediate: ValueProject,
    move_descriptor: Descriptor,
    bitcast_descriptor: Descriptor,
    *,
    priority: int = 0,
) -> DescriptorRule:
    """Constructs a floating scalar from its exact integer bit pattern."""
    return DescriptorRule(
        source_op=scalar_conversion.scalar_constant,
        descriptor=bitcast_descriptor,
        guards=(
            Guard.attr_kind("value", "f64"),
            Guard.value_type("result", result_type),
            Guard.value_exact_float("result"),
        ),
        emit=(
            EmitDescriptorOp(
                descriptor=move_descriptor,
                results={"dst": ValueRef.temporary("bits")},
                result_types={"dst": integer_type},
                immediates={immediate_name: immediate},
                form=DescriptorEmitForm.CONST,
            ),
            EmitDescriptorOp(
                descriptor=bitcast_descriptor,
                operands={"input": ValueRef.temporary("bits")},
                results={"dst": ValueRef.result("result")},
                form=DescriptorEmitForm.OP,
            ),
        ),
        priority=priority,
    )


def floating_storage_scalar_constant_rule(
    result_type: TypePattern,
    move_descriptor: Descriptor,
) -> DescriptorRule:
    """Constructs a storage float in its integer carrier from exact bits."""
    return DescriptorRule(
        source_op=scalar_conversion.scalar_constant,
        descriptor=move_descriptor,
        guards=(
            Guard.attr_kind("value", "f64"),
            Guard.value_type("result", result_type),
            Guard.value_exact_float("result"),
        ),
        emit=(
            EmitDescriptorOp(
                descriptor=move_descriptor,
                results={"dst": ValueRef.result("result")},
                immediates={"imm32": ValueProject.float_bits("result")},
                form=DescriptorEmitForm.CONST,
            ),
        ),
    )


def f32_scalar_constant_rule(
    move_descriptor: Descriptor,
    bitcast_descriptor: Descriptor,
) -> DescriptorRule:
    """Constructs an f32 scalar from its exact bit pattern."""
    return floating_scalar_constant_bits_rule(
        _F32,
        _I32,
        "imm32",
        ValueProject.float_as_f32_i32("result"),
        move_descriptor,
        bitcast_descriptor,
    )


def f64_scalar_constant_rule(
    move_descriptor: Descriptor,
    bitcast_descriptor: Descriptor,
) -> DescriptorRule:
    """Constructs an f64 scalar from its exact bit pattern."""
    return floating_scalar_constant_bits_rule(
        _F64,
        _I64,
        "imm64",
        ValueProject.float_as_f64_i64("result"),
        move_descriptor,
        bitcast_descriptor,
    )


def floating_vector_splat_zero_rule(
    result_type: TypePattern,
    descriptor: Descriptor,
    *,
    result_register_class: str | None = None,
) -> DescriptorRule:
    """Constructs an exact positive-zero splat with a dependency-breaking op."""
    return DescriptorRule(
        source_op=vector.vector_splat,
        descriptor=descriptor,
        guards=(
            Guard.value_type(
                "scalar", Scalar(("f8E4M3", "f8E5M2", "f16", "bf16", "f32", "f64"))
            ),
            Guard.value_type("result", result_type),
            Guard.value_float_equals("scalar", 0.0),
            *(
                (Guard.low_value_register_class("result", result_register_class),)
                if result_register_class is not None
                else ()
            ),
        ),
        emit=(
            EmitDescriptorOp(
                descriptor=descriptor,
                results={"dst": ValueRef.result("result")},
                form=DescriptorEmitForm.OP,
            ),
        ),
        priority=2,
    )


def f64_vector_splat_rule(
    result_type: TypePattern,
    descriptor: Descriptor,
) -> DescriptorRule:
    """Broadcasts the low f64 lane across an XMM vector."""
    return DescriptorRule(
        source_op=vector.vector_splat,
        descriptor=descriptor,
        guards=(
            Guard.value_type("scalar", _F64),
            Guard.value_type("result", result_type),
        ),
        emit=(
            EmitDescriptorOp(
                descriptor=descriptor,
                operands={"source": ValueRef.operand("scalar")},
                results={"dst": ValueRef.result("result")},
                immediates={"control": 0},
                form=DescriptorEmitForm.OP,
            ),
        ),
    )
