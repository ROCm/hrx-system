# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


# ruff: noqa: F403, F405

"""Scalar floating-point arithmetic descriptor overlays."""

from __future__ import annotations

from .common import *

_SCALAR_FLOAT_BINARY_OPERATIONS = (
    ("add", "add"),
    ("sub", "sub"),
    ("mul", "mul"),
    ("min", "minnum"),
    ("max", "maxnum"),
)

_SCALAR_FLOAT_COMMUTATIVE_BINARY_OPERATIONS = frozenset(("add", "mul", "min", "max"))

_SCALAR_FLOAT_UNARY_OPERATIONS = (
    ("ceil", "ceil"),
    ("floor", "floor"),
    ("rndne", "round_even"),
    ("trunc", "trunc"),
)

_SCALAR_FLOAT_BIT_WIDTHS = (16, 32)


def _scalar_float_register_part(bit_width: int) -> str | None:
    if bit_width == 16:
        return _REG_PART_SGPR_LOW16
    if bit_width == 32:
        return None
    raise ValueError(f"unsupported scalar floating-point bit width {bit_width}")


def _s_float_binary_overlay(
    operation: str, semantic: str, bit_width: int
) -> AmdgpuDescriptorOverlay:
    register_part = _scalar_float_register_part(bit_width)
    constraints = (Constraint(ConstraintKind.REMATERIALIZABLE, 0),)
    if operation in _SCALAR_FLOAT_COMMUTATIVE_BINARY_OPERATIONS:
        constraints += (Constraint(ConstraintKind.COMMUTABLE, 1, 2),)
    return AmdgpuDescriptorOverlay(
        descriptor_key=f"amdgpu.s_{operation}_f{bit_width}",
        instruction_name=f"S_{operation.upper()}_F{bit_width}",
        mnemonic=f"s_{operation}_f{bit_width}",
        encoding_name="ENC_SOP2",
        semantic_tag=f"float.{semantic}.f{bit_width}",
        schedule_class=_SCHEDULE_SALU,
        operands=(
            AmdgpuOperandOverlay("SDST", _sgpr_result(register_part=register_part)),
            AmdgpuOperandOverlay(
                "SSRC0", _sgpr_operand("lhs", register_part=register_part)
            ),
            AmdgpuOperandOverlay(
                "SSRC1", _sgpr_operand("rhs", register_part=register_part)
            ),
        ),
        constraints=constraints,
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _s_float_unary_overlay(
    operation: str, semantic: str, bit_width: int
) -> AmdgpuDescriptorOverlay:
    register_part = _scalar_float_register_part(bit_width)
    return AmdgpuDescriptorOverlay(
        descriptor_key=f"amdgpu.s_{operation}_f{bit_width}",
        instruction_name=f"S_{operation.upper()}_F{bit_width}",
        mnemonic=f"s_{operation}_f{bit_width}",
        encoding_name="ENC_SOP1",
        encoding_condition="Nothas_lit_0_Nothas_lit_1",
        semantic_tag=f"float.{semantic}.f{bit_width}",
        schedule_class=_SCHEDULE_SALU,
        operands=(
            AmdgpuOperandOverlay("SDST", _sgpr_result(register_part=register_part)),
            AmdgpuOperandOverlay(
                "SSRC0", _sgpr_operand("input", register_part=register_part)
            ),
        ),
        constraints=(Constraint(ConstraintKind.REMATERIALIZABLE, 0),),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _s_float_arithmetic_overlays() -> tuple[AmdgpuDescriptorOverlay, ...]:
    return (
        *(
            _s_float_binary_overlay(operation, semantic, bit_width)
            for bit_width in _SCALAR_FLOAT_BIT_WIDTHS
            for operation, semantic in _SCALAR_FLOAT_BINARY_OPERATIONS
        ),
        *(
            _s_float_unary_overlay(operation, semantic, bit_width)
            for bit_width in _SCALAR_FLOAT_BIT_WIDTHS
            for operation, semantic in _SCALAR_FLOAT_UNARY_OPERATIONS
        ),
    )


__all__ = ("_s_float_arithmetic_overlays",)
