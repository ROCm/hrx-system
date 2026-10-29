# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Shared x86 scalar and uniform-vector constant lowering rules."""

from __future__ import annotations

from loom.dialect.scalar import conversion as scalar_conversion
from loom.dialect.vector import defs as vector
from loom.dsl import Op
from loom.target.contracts import (
    AttrProject,
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


def integer_vector_zero_rule(
    result_type: TypePattern,
    descriptor: Descriptor,
    *,
    result_register_class: str | None = None,
) -> DescriptorRule:
    """Constructs an integer zero vector with a dependency-breaking op."""
    return DescriptorRule(
        source_op=vector.vector_splati,
        descriptor=descriptor,
        guards=(
            Guard.value_type("result", result_type),
            Guard.i64_range("value", 0, 0),
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
        priority=1,
    )


def _uniform_vector_constant_rule(
    source_op: Op,
    result_type: TypePattern,
    immediate_type: TypePattern,
    broadcast_type: TypePattern,
    immediate_name: str,
    immediate: AttrProject | ValueProject,
    move_descriptor: Descriptor,
    broadcast_descriptor: Descriptor,
    broadcast_operand: str,
    broadcast_immediates: dict[str, int] | None,
    lane_move_descriptor: Descriptor | None,
    guards: tuple[Guard, ...],
    priority: int,
    result_register_class: str | None = None,
) -> DescriptorRule:
    """Constructs a uniform vector from one immediate scalar payload."""
    bits = ValueRef.temporary("bits")
    emits = [
        EmitDescriptorOp(
            descriptor=move_descriptor,
            results={"dst": bits},
            result_types={"dst": immediate_type},
            immediates={immediate_name: immediate},
            form=DescriptorEmitForm.CONST,
        )
    ]
    broadcast_source = bits
    if lane_move_descriptor is not None:
        broadcast_source = ValueRef.temporary("scalar")
        emits.append(
            EmitDescriptorOp(
                descriptor=lane_move_descriptor,
                operands={"input": bits},
                results={"dst": broadcast_source},
                result_types={"dst": broadcast_type},
                form=DescriptorEmitForm.OP,
            )
        )
    emits.append(
        EmitDescriptorOp(
            descriptor=broadcast_descriptor,
            operands={broadcast_operand: broadcast_source},
            results={"dst": ValueRef.result("result")},
            immediates=({} if broadcast_immediates is None else broadcast_immediates),
            form=DescriptorEmitForm.OP,
        )
    )
    return DescriptorRule(
        source_op=source_op,
        descriptor=broadcast_descriptor,
        guards=(
            Guard.value_type("result", result_type),
            *guards,
            *(
                (Guard.low_value_register_class("result", result_register_class),)
                if result_register_class is not None
                else ()
            ),
        ),
        emit=tuple(emits),
        priority=priority,
    )


def i32_vector_constant_rule(
    result_type: TypePattern,
    move_descriptor: Descriptor,
    broadcast_descriptor: Descriptor,
    *,
    broadcast_operand: str,
    broadcast_immediates: dict[str, int] | None = None,
    lane_move_descriptor: Descriptor | None = None,
    lane_type: TypePattern | None = None,
    priority: int = 0,
) -> DescriptorRule:
    """Constructs a uniform i32 vector from its exact element value."""
    return _uniform_vector_constant_rule(
        vector.vector_splati,
        result_type,
        _I32,
        result_type if lane_type is None else lane_type,
        "imm32",
        AttrProject.direct("value"),
        move_descriptor,
        broadcast_descriptor,
        broadcast_operand,
        broadcast_immediates,
        lane_move_descriptor,
        (Guard.i64_range("value", -(2**31), (2**31) - 1),),
        priority,
    )


def integer_vector_constant_rule(
    result_type: TypePattern,
    lane_type: TypePattern,
    element_bit_width: int,
    move_descriptor: Descriptor,
    lane_move_descriptor: Descriptor | None,
    broadcast_descriptor: Descriptor,
    *,
    priority: int = 0,
    result_register_class: str | None = None,
) -> DescriptorRule:
    """Constructs a uniform integer vector from one exact element value."""
    if element_bit_width not in (8, 16, 32, 64):
        raise ValueError(f"unsupported x86 integer element width {element_bit_width}")
    immediate_type = _I64 if element_bit_width == 64 else _I32
    return _uniform_vector_constant_rule(
        vector.vector_splati,
        result_type,
        immediate_type,
        lane_type,
        "imm64" if element_bit_width == 64 else "imm32",
        AttrProject.direct("value"),
        move_descriptor,
        broadcast_descriptor,
        "value",
        None,
        lane_move_descriptor,
        (
            Guard.i64_range(
                "value",
                -(2 ** (element_bit_width - 1)),
                (2 ** (element_bit_width - 1)) - 1,
            ),
        ),
        priority,
        result_register_class,
    )


def i64_vector_constant_rule(
    result_type: TypePattern,
    move_descriptor: Descriptor,
    broadcast_descriptor: Descriptor,
    *,
    broadcast_operand: str,
    broadcast_immediates: dict[str, int] | None = None,
    lane_move_descriptor: Descriptor | None = None,
    priority: int = 0,
) -> DescriptorRule:
    """Constructs a uniform i64 vector from its exact element value."""
    return _uniform_vector_constant_rule(
        vector.vector_splati,
        result_type,
        _I64,
        result_type,
        "imm64",
        AttrProject.direct("value"),
        move_descriptor,
        broadcast_descriptor,
        broadcast_operand,
        broadcast_immediates,
        lane_move_descriptor,
        (Guard.i64_range("value", -(2**63), (2**63) - 1),),
        priority,
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


def floating_vector_zero_rule(
    result_type: TypePattern,
    descriptor: Descriptor,
    *,
    result_register_class: str | None = None,
) -> DescriptorRule:
    """Constructs an exact positive-zero vector with a dependency-breaking op."""
    return DescriptorRule(
        source_op=vector.vector_splatf,
        descriptor=descriptor,
        guards=(
            Guard.value_type("result", result_type),
            Guard.value_float_equals("result", 0.0),
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
        priority=1,
    )


def _floating_vector_constant_rule(
    result_type: TypePattern,
    scalar_type: TypePattern,
    integer_type: TypePattern,
    immediate_name: str,
    immediate: ValueProject,
    move_descriptor: Descriptor,
    bitcast_descriptor: Descriptor,
    broadcast_descriptor: Descriptor,
    broadcast_operand: str,
    broadcast_immediates: dict[str, int] | None = None,
) -> DescriptorRule:
    """Constructs a uniform float vector from its exact element bit pattern."""
    return _uniform_vector_constant_rule(
        vector.vector_splatf,
        result_type,
        integer_type,
        scalar_type,
        immediate_name,
        immediate,
        move_descriptor,
        broadcast_descriptor,
        broadcast_operand,
        broadcast_immediates,
        bitcast_descriptor,
        (Guard.value_exact_float("result"),),
        0,
    )


def f32_vector_constant_rule(
    result_type: TypePattern,
    move_descriptor: Descriptor,
    bitcast_descriptor: Descriptor,
    broadcast_descriptor: Descriptor,
) -> DescriptorRule:
    """Constructs a uniform f32 vector from its exact element bit pattern."""
    return _floating_vector_constant_rule(
        result_type,
        _F32,
        _I32,
        "imm32",
        ValueProject.float_as_f32_i32("result"),
        move_descriptor,
        bitcast_descriptor,
        broadcast_descriptor,
        "value",
    )


def f64_vector_constant_rule(
    result_type: TypePattern,
    move_descriptor: Descriptor,
    bitcast_descriptor: Descriptor,
    broadcast_descriptor: Descriptor,
) -> DescriptorRule:
    """Constructs a uniform f64 vector from its exact element bit pattern."""
    return _floating_vector_constant_rule(
        result_type,
        _F64,
        _I64,
        "imm64",
        ValueProject.float_as_f64_i64("result"),
        move_descriptor,
        bitcast_descriptor,
        broadcast_descriptor,
        "source",
        {"control": 0},
    )


def floating_vector_constant_bits_rule(
    result_type: TypePattern,
    lane_type: TypePattern,
    element_type: str,
    move_descriptor: Descriptor,
    lane_move_descriptor: Descriptor | None,
    broadcast_descriptor: Descriptor,
    *,
    priority: int = 0,
    result_register_class: str | None = None,
) -> DescriptorRule:
    """Constructs a uniform float vector from exact element-format bits."""
    if element_type in ("f8E4M3", "f8E5M2", "f16", "bf16"):
        immediate_type = _I32
        immediate_name = "imm32"
        immediate = ValueProject.float_bits("result")
    elif element_type == "f32":
        immediate_type = _I32
        immediate_name = "imm32"
        immediate = ValueProject.float_as_f32_i32("result")
    elif element_type == "f64":
        immediate_type = _I64
        immediate_name = "imm64"
        immediate = ValueProject.float_as_f64_i64("result")
    else:
        raise ValueError(f"unsupported x86 floating element type {element_type}")
    return _uniform_vector_constant_rule(
        vector.vector_splatf,
        result_type,
        immediate_type,
        lane_type,
        immediate_name,
        immediate,
        move_descriptor,
        broadcast_descriptor,
        "value",
        None,
        lane_move_descriptor,
        (Guard.value_exact_float("result"),),
        priority,
        result_register_class,
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
