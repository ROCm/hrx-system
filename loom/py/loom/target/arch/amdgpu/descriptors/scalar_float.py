# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


# ruff: noqa: F403, F405

"""Scalar floating-point descriptor overlays."""

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

_SCALAR_FLOAT_CONVERSIONS = (
    ("f32_i32", "convert.signed.i32.f32", None, None),
    ("f32_u32", "convert.unsigned.u32.f32", None, None),
    ("i32_f32", "convert.float.f32.signed.i32", None, None),
    ("u32_f32", "convert.float.f32.unsigned.u32", None, None),
    ("f16_f32", "convert.float.f32.f16", _REG_PART_SGPR_LOW16, None),
    ("f32_f16", "convert.float.f16.f32", None, _REG_PART_SGPR_LOW16),
    ("hi_f32_f16", "convert.float.f16.high.f32", None, _REG_PART_SGPR_HIGH16),
)

_SCALAR_FLOAT_COMPARE_PREDICATES = (
    ("oeq", "EQ"),
    ("ogt", "GT"),
    ("oge", "GE"),
    ("olt", "LT"),
    ("ole", "LE"),
    ("one", "LG"),
    ("ord", "O"),
    ("ueq", "NLG"),
    ("ugt", "NLE"),
    ("uge", "NLT"),
    ("ult", "NGE"),
    ("ule", "NGT"),
    ("une", "NEQ"),
    ("uno", "U"),
)


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


def _s_float_conversion_overlay(
    operation: str,
    semantic_tag: str,
    result_register_part: str | None,
    input_register_part: str | None,
) -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key=f"amdgpu.s_cvt_{operation}",
        instruction_name=f"S_CVT_{operation.upper()}",
        mnemonic=f"s_cvt_{operation}",
        encoding_name="ENC_SOP1",
        encoding_condition="Nothas_lit_0_Nothas_lit_1",
        semantic_tag=semantic_tag,
        schedule_class=_SCHEDULE_SALU,
        operands=(
            AmdgpuOperandOverlay(
                "SDST", _sgpr_result(register_part=result_register_part)
            ),
            AmdgpuOperandOverlay(
                "SSRC0", _sgpr_operand("input", register_part=input_register_part)
            ),
        ),
        constraints=(Constraint(ConstraintKind.REMATERIALIZABLE, 0),),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _s_float_conversion_overlays() -> tuple[AmdgpuDescriptorOverlay, ...]:
    return (
        *(
            _s_float_conversion_overlay(*conversion)
            for conversion in _SCALAR_FLOAT_CONVERSIONS
        ),
        AmdgpuDescriptorOverlay(
            descriptor_key="amdgpu.s_cvt_pk_rtz_f16_f32",
            instruction_name="S_CVT_PK_RTZ_F16_F32",
            mnemonic="s_cvt_pk_rtz_f16_f32",
            encoding_name="ENC_SOP2",
            semantic_tag="convert.float.f32x2.f16x2.rtz",
            schedule_class=_SCHEDULE_SALU,
            operands=(
                AmdgpuOperandOverlay("SDST", _sgpr_result()),
                AmdgpuOperandOverlay("SSRC0", _sgpr_operand("lhs")),
                AmdgpuOperandOverlay("SSRC1", _sgpr_operand("rhs")),
            ),
            constraints=(Constraint(ConstraintKind.REMATERIALIZABLE, 0),),
            flags=(DescriptorFlag.DEAD_REMOVABLE,),
        ),
    )


def _s_float_compare_overlay(
    predicate: str, instruction_suffix: str, bit_width: int
) -> AmdgpuDescriptorOverlay:
    register_part = _scalar_float_register_part(bit_width)
    return AmdgpuDescriptorOverlay(
        descriptor_key=f"amdgpu.s_cmp_{predicate}_f{bit_width}",
        instruction_name=f"S_CMP_{instruction_suffix}_F{bit_width}",
        mnemonic=f"s_cmp_{instruction_suffix.lower()}_f{bit_width}",
        encoding_name="ENC_SOPC",
        semantic_tag=f"cmp.f{bit_width}.{predicate}",
        schedule_class=_SCHEDULE_SALU_COMPARE,
        operands=(
            AmdgpuOperandOverlay(
                "SSRC0", _sgpr_operand("lhs", register_part=register_part)
            ),
            AmdgpuOperandOverlay(
                "SSRC1", _sgpr_operand("rhs", register_part=register_part)
            ),
        ),
        implicit_operands=(_scc_output(_scc_result()),),
        asm_forms=_asm(results=("scc",), operands=("lhs", "rhs")),
        constraints=(Constraint(ConstraintKind.REMATERIALIZABLE, 0),),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _s_float_compare_overlays() -> tuple[AmdgpuDescriptorOverlay, ...]:
    return tuple(
        _s_float_compare_overlay(predicate, instruction_suffix, bit_width)
        for bit_width in _SCALAR_FLOAT_BIT_WIDTHS
        for predicate, instruction_suffix in _SCALAR_FLOAT_COMPARE_PREDICATES
    )


__all__ = (
    "_s_float_arithmetic_overlays",
    "_s_float_compare_overlays",
    "_s_float_conversion_overlays",
)
