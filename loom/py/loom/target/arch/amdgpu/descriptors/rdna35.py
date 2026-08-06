# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


# ruff: noqa: F403, F405

"""AMDGPU RDNA 3.5 descriptor overlays."""

from __future__ import annotations

from .common import *

_RDNA35_DPP16_F32_ROWS = (
    ("add", "V_ADD_F32", "v_add_f32", "add"),
    ("sub", "V_SUB_F32", "v_sub_f32", "sub"),
    ("mul", "V_MUL_F32", "v_mul_f32", "mul"),
    ("min", "V_MIN_F32", "v_min_f32", "minnum"),
    ("max", "V_MAX_F32", "v_max_f32", "maxnum"),
)


def _v_binary_f32_dpp16_uniform_rhs_overlays(
    *,
    operation: str,
    instruction_name: str,
    mnemonic: str,
    semantic: str,
) -> tuple[AmdgpuDescriptorOverlay, ...]:
    return (
        AmdgpuDescriptorOverlay(
            descriptor_key=f"amdgpu.v_{operation}_f32.dpp16_sgpr",
            instruction_name=instruction_name,
            mnemonic=mnemonic,
            encoding_name="VOP3_VOP_DPP16",
            encoding_condition="has_dpp16",
            semantic_tag=f"float.{semantic}.f32",
            schedule_class=_SCHEDULE_VALU,
            operands=(
                AmdgpuOperandOverlay("VDST", _vgpr_result()),
                AmdgpuOperandOverlay("VSRC0", _vgpr_operand("crosslane")),
                AmdgpuOperandOverlay("SRC1", _sgpr_operand("rhs")),
            ),
            asm_forms=_asm(
                mnemonic=f"v_{operation}_f32_dpp16_sgpr",
                native_assembly_mnemonic=f"{mnemonic}_e64_dpp",
                results=("dst",),
                operands=("crosslane", "rhs"),
                immediates=("dpp_ctrl",),
                named_immediates=True,
                native_assembly_values=(
                    _native_result("dst"),
                    _native_operand("crosslane"),
                    _native_operand("rhs"),
                    _native_amdgpu_dpp_ctrl_immediate("dpp_ctrl"),
                    _native_literal("row_mask:0xf"),
                    _native_literal("bank_mask:0xf"),
                    _native_literal("bound_ctrl:1"),
                ),
            ),
            immediate_fields=("DPP_CTRL",),
            immediates=(_DPP_CTRL_IMMEDIATE,),
            fixed_encoding_fields=(
                ("SRC0", 250),
                ("ROW_MASK", 0xF),
                ("BANK_MASK", 0xF),
                ("BOUND_CTRL", 1),
            ),
            effects=(_CONVERGENT_EFFECT,),
        ),
        AmdgpuDescriptorOverlay(
            descriptor_key=f"amdgpu.v_{operation}_f32.dpp16_src1_inline",
            instruction_name=instruction_name,
            mnemonic=mnemonic,
            encoding_name="VOP3_VOP_DPP16",
            encoding_condition="has_dpp16",
            semantic_tag=f"float.{semantic}.f32",
            schedule_class=_SCHEDULE_VALU,
            operands=(
                AmdgpuOperandOverlay("VDST", _vgpr_result()),
                AmdgpuOperandOverlay("VSRC0", _vgpr_operand("crosslane")),
            ),
            asm_forms=_asm(
                mnemonic=f"v_{operation}_f32_dpp16_src1_inline",
                native_assembly_mnemonic=f"{mnemonic}_e64_dpp",
                results=("dst",),
                operands=("crosslane",),
                immediates=("rhs", "dpp_ctrl"),
                named_immediates=True,
                native_assembly_values=(
                    _native_result("dst"),
                    _native_operand("crosslane"),
                    _native_i64_immediate("rhs"),
                    _native_amdgpu_dpp_ctrl_immediate("dpp_ctrl"),
                    _native_literal("row_mask:0xf"),
                    _native_literal("bank_mask:0xf"),
                    _native_literal("bound_ctrl:1"),
                ),
            ),
            immediate_fields=("SRC1", "DPP_CTRL"),
            immediates=(
                replace(_SOURCE_INLINE_F32_IMMEDIATE, field_name="rhs"),
                _DPP_CTRL_IMMEDIATE,
            ),
            fixed_encoding_fields=(
                ("SRC0", 250),
                ("ROW_MASK", 0xF),
                ("BANK_MASK", 0xF),
                ("BOUND_CTRL", 1),
            ),
            effects=(_CONVERGENT_EFFECT,),
        ),
    )


def _rdna35_dpp16_f32_uniform_rhs_overlays() -> tuple[AmdgpuDescriptorOverlay, ...]:
    return tuple(
        overlay
        for operation, instruction_name, mnemonic, semantic in (_RDNA35_DPP16_F32_ROWS)
        for overlay in _v_binary_f32_dpp16_uniform_rhs_overlays(
            operation=operation,
            instruction_name=instruction_name,
            mnemonic=mnemonic,
            semantic=semantic,
        )
    )


__all__ = (
    "_RDNA35_DPP16_F32_ROWS",
    "_rdna35_dpp16_f32_uniform_rhs_overlays",
    "_v_binary_f32_dpp16_uniform_rhs_overlays",
)
