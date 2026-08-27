# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


# ruff: noqa: F403, F405

"""AMDGPU RDNA 3.5 descriptor overlays."""

from __future__ import annotations

from .alu import _v_dpp_uniform_rhs_integer_compare_candidates
from .common import *

_RDNA35_DPP_F32_ROWS = (
    ("add", "V_ADD_F32", "v_add_f32", "add"),
    ("sub", "V_SUB_F32", "v_sub_f32", "sub"),
    ("mul", "V_MUL_F32", "v_mul_f32", "mul"),
    ("min", "V_MIN_F32", "v_min_f32", "minnum"),
    ("max", "V_MAX_F32", "v_max_f32", "maxnum"),
)

_DPP16_SELECTOR_IMMEDIATE = replace(
    _DPP_CTRL_IMMEDIATE,
    encoding_field_id=amdgpu_encoding_field_id("DPP_CTRL"),
)


def _v_dpp_uniform_rhs_variant_overlays(
    *,
    base_overlay: AmdgpuDescriptorOverlay,
    rhs_inline_immediate: Immediate,
    descriptor_suffix: str,
    encoding_name: str,
    sdst_encoding_name: str,
    encoding_condition: str,
    selector_immediate: Immediate,
    selector_native_value: NativeAsmValue,
    native_modifier_values: tuple[NativeAsmValue, ...],
    fixed_encoding_fields: tuple[tuple[str, AmdgpuFixedEncodingValue], ...],
) -> tuple[AmdgpuDescriptorOverlay, ...]:
    source0_indices = tuple(
        i
        for i, operand in enumerate(base_overlay.operands)
        if operand.xml_field_name in ("SRC0", "VSRC0")
    )
    source1_indices = tuple(
        i
        for i, operand in enumerate(base_overlay.operands)
        if operand.xml_field_name in ("SRC1", "VSRC1")
    )
    if len(source0_indices) != 1 or len(source1_indices) != 1:
        raise ValueError(
            f"DPP uniform-RHS base '{base_overlay.descriptor_key}' must have "
            "exactly one source0 and source1 operand"
        )
    source0_index = source0_indices[0]
    source1_index = source1_indices[0]
    if base_overlay.immediates or base_overlay.implicit_operands:
        raise ValueError(
            f"DPP uniform-RHS base '{base_overlay.descriptor_key}' must expose "
            "only explicit register operands"
        )

    source1_overlay = base_overlay.operands[source1_index]
    source1_name = source1_overlay.descriptor_operand.field_name
    dpp_encoding_name = (
        sdst_encoding_name
        if any(operand.xml_field_name == "SDST" for operand in base_overlay.operands)
        else encoding_name
    )
    sgpr_operands = tuple(
        replace(
            operand,
            xml_field_name="VSRC0",
            descriptor_operand=replace(
                operand.descriptor_operand,
                reg_alts=_VGPR_ALT,
            ),
        )
        if i == source0_index
        else replace(
            operand,
            xml_field_name="SRC1",
            descriptor_operand=replace(
                operand.descriptor_operand,
                reg_alts=_SGPR_ALT,
            ),
        )
        if i == source1_index
        else operand
        for i, operand in enumerate(base_overlay.operands)
    )
    inline_operands = tuple(
        operand for i, operand in enumerate(sgpr_operands) if i != source1_index
    )
    result_names = tuple(
        operand.descriptor_operand.field_name
        for operand in base_overlay.operands
        if operand.descriptor_operand.role is OperandRole.RESULT
    )
    sgpr_operand_names = tuple(
        operand.descriptor_operand.field_name
        for operand in sgpr_operands
        if operand.descriptor_operand.role is not OperandRole.RESULT
    )
    inline_operand_names = tuple(
        operand.descriptor_operand.field_name
        for operand in inline_operands
        if operand.descriptor_operand.role is not OperandRole.RESULT
    )
    sgpr_native_values = tuple(
        _native_result(operand.descriptor_operand.field_name)
        if operand.descriptor_operand.role is OperandRole.RESULT
        else _native_operand(operand.descriptor_operand.field_name)
        for operand in sgpr_operands
    )
    inline_native_values = tuple(
        _native_result(operand.descriptor_operand.field_name)
        if operand.descriptor_operand.role is OperandRole.RESULT
        else _native_i64_immediate(source1_name)
        if i == source1_index
        else _native_operand(operand.descriptor_operand.field_name)
        for i, operand in enumerate(sgpr_operands)
    )
    inline_rhs_immediate = replace(
        rhs_inline_immediate,
        field_name=source1_name,
        encoding_field_id=amdgpu_encoding_field_id("SRC1"),
    )
    mnemonic = base_overlay.mnemonic or base_overlay.instruction_name.lower()
    native_mnemonic = base_overlay.instruction_name.lower()
    return (
        AmdgpuDescriptorOverlay(
            descriptor_key=f"{base_overlay.descriptor_key}.{descriptor_suffix}_sgpr",
            instruction_name=base_overlay.instruction_name,
            mnemonic=mnemonic,
            encoding_name=dpp_encoding_name,
            encoding_condition=encoding_condition,
            semantic_tag=base_overlay.semantic_tag,
            schedule_class=base_overlay.schedule_class,
            operands=sgpr_operands,
            asm_forms=_asm(
                mnemonic=f"{mnemonic}_{descriptor_suffix}_sgpr",
                native_assembly_mnemonic=f"{native_mnemonic}_e64_dpp",
                results=result_names,
                operands=sgpr_operand_names,
                immediates=(selector_immediate.field_name,),
                named_immediates=True,
                native_assembly_values=(
                    *sgpr_native_values,
                    selector_native_value,
                    *native_modifier_values,
                ),
            ),
            immediates=(selector_immediate,),
            fixed_encoding_fields=fixed_encoding_fields,
            effects=(*base_overlay.effects, _CONVERGENT_EFFECT),
            flags=base_overlay.flags,
        ),
        AmdgpuDescriptorOverlay(
            descriptor_key=f"{base_overlay.descriptor_key}.{descriptor_suffix}_src1_inline",
            instruction_name=base_overlay.instruction_name,
            mnemonic=mnemonic,
            encoding_name=dpp_encoding_name,
            encoding_condition=encoding_condition,
            semantic_tag=base_overlay.semantic_tag,
            schedule_class=base_overlay.schedule_class,
            operands=inline_operands,
            asm_forms=_asm(
                mnemonic=f"{mnemonic}_{descriptor_suffix}_src1_inline",
                native_assembly_mnemonic=f"{native_mnemonic}_e64_dpp",
                results=result_names,
                operands=inline_operand_names,
                immediates=(source1_name, selector_immediate.field_name),
                named_immediates=True,
                native_assembly_values=(
                    *inline_native_values,
                    selector_native_value,
                    *native_modifier_values,
                ),
            ),
            immediates=(
                inline_rhs_immediate,
                selector_immediate,
            ),
            fixed_encoding_fields=fixed_encoding_fields,
            effects=(*base_overlay.effects, _CONVERGENT_EFFECT),
            flags=base_overlay.flags,
        ),
    )


def _v_binary_f32_uniform_rhs_overlays(
    *,
    operation: str,
    instruction_name: str,
    low_mnemonic: str,
    semantic: str,
    descriptor_suffix: str,
    encoding_name: str,
    encoding_condition: str,
    selector_immediate: Immediate,
    selector_native_value: NativeAsmValue,
    native_modifier_values: tuple[NativeAsmValue, ...],
    fixed_encoding_fields: tuple[tuple[str, AmdgpuFixedEncodingValue], ...],
) -> tuple[AmdgpuDescriptorOverlay, ...]:
    base_overlay = AmdgpuDescriptorOverlay(
        descriptor_key=f"amdgpu.v_{operation}_f32",
        instruction_name=instruction_name,
        mnemonic=low_mnemonic,
        encoding_name="ENC_VOP3",
        semantic_tag=f"float.{semantic}.f32",
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            AmdgpuOperandOverlay("SRC0", _vgpr_operand("crosslane")),
            AmdgpuOperandOverlay("SRC1", _sgpr_operand("rhs")),
        ),
    )
    return _v_dpp_uniform_rhs_variant_overlays(
        base_overlay=base_overlay,
        rhs_inline_immediate=_SOURCE_INLINE_F32_IMMEDIATE,
        descriptor_suffix=descriptor_suffix,
        encoding_name=encoding_name,
        sdst_encoding_name=encoding_name,
        encoding_condition=encoding_condition,
        selector_immediate=selector_immediate,
        selector_native_value=selector_native_value,
        native_modifier_values=native_modifier_values,
        fixed_encoding_fields=fixed_encoding_fields,
    )


def _v_binary_f32_dpp16_uniform_rhs_overlays(
    *,
    operation: str,
    instruction_name: str,
    low_mnemonic: str,
    semantic: str,
) -> tuple[AmdgpuDescriptorOverlay, ...]:
    selector_immediate = replace(_DPP16_SELECTOR_IMMEDIATE, field_name="dpp_ctrl")
    return _v_binary_f32_uniform_rhs_overlays(
        operation=operation,
        instruction_name=instruction_name,
        low_mnemonic=low_mnemonic,
        semantic=semantic,
        descriptor_suffix="dpp16",
        encoding_name="VOP3_VOP_DPP16",
        encoding_condition="has_dpp16",
        selector_immediate=selector_immediate,
        selector_native_value=_native_amdgpu_dpp_ctrl_immediate("dpp_ctrl"),
        native_modifier_values=(
            _native_literal("row_mask:0xf"),
            _native_literal("bank_mask:0xf"),
            _native_literal("bound_ctrl:1"),
        ),
        fixed_encoding_fields=(
            ("SRC0", 250),
            ("ROW_MASK", 0xF),
            ("BANK_MASK", 0xF),
            ("BOUND_CTRL", 1),
        ),
    )


def _v_binary_f32_dpp8_uniform_rhs_overlays(
    *,
    operation: str,
    instruction_name: str,
    low_mnemonic: str,
    semantic: str,
) -> tuple[AmdgpuDescriptorOverlay, ...]:
    return _v_binary_f32_uniform_rhs_overlays(
        operation=operation,
        instruction_name=instruction_name,
        low_mnemonic=low_mnemonic,
        semantic=semantic,
        descriptor_suffix="dpp8",
        encoding_name="VOP3_VOP_DPP8",
        encoding_condition="has_dpp8",
        selector_immediate=_DPP8_IMMEDIATE,
        selector_native_value=_native_amdgpu_dpp8_immediate("dpp8"),
        native_modifier_values=(),
        fixed_encoding_fields=(("SRC0", 233),),
    )


def _rdna35_dpp_integer_compare_uniform_rhs_overlays() -> tuple[
    AmdgpuDescriptorOverlay, ...
]:
    return tuple(
        overlay
        for base_overlay, rhs_inline_immediate in (
            _v_dpp_uniform_rhs_integer_compare_candidates()
        )
        for overlay in (
            *_v_dpp_uniform_rhs_variant_overlays(
                base_overlay=base_overlay,
                rhs_inline_immediate=rhs_inline_immediate,
                descriptor_suffix="dpp16",
                encoding_name="VOP3_VOP_DPP16",
                sdst_encoding_name="VOP3_SDST_ENC_VOP_DPP16",
                encoding_condition="has_dpp16",
                selector_immediate=replace(
                    _DPP16_SELECTOR_IMMEDIATE,
                    field_name="dpp_ctrl",
                ),
                selector_native_value=_native_amdgpu_dpp_ctrl_immediate("dpp_ctrl"),
                native_modifier_values=(
                    _native_literal("row_mask:0xf"),
                    _native_literal("bank_mask:0xf"),
                    _native_literal("bound_ctrl:1"),
                ),
                fixed_encoding_fields=(
                    ("SRC0", 250),
                    ("ROW_MASK", 0xF),
                    ("BANK_MASK", 0xF),
                    ("BOUND_CTRL", 1),
                ),
            ),
            *_v_dpp_uniform_rhs_variant_overlays(
                base_overlay=base_overlay,
                rhs_inline_immediate=rhs_inline_immediate,
                descriptor_suffix="dpp8",
                encoding_name="VOP3_VOP_DPP8",
                sdst_encoding_name="VOP3_SDST_ENC_VOP_DPP8",
                encoding_condition="has_dpp8",
                selector_immediate=_DPP8_IMMEDIATE,
                selector_native_value=_native_amdgpu_dpp8_immediate("dpp8"),
                native_modifier_values=(),
                fixed_encoding_fields=(("SRC0", 233),),
            ),
        )
    )


def _rdna35_dpp16_f32_uniform_rhs_overlays() -> tuple[AmdgpuDescriptorOverlay, ...]:
    return tuple(
        overlay
        for operation, instruction_name, low_mnemonic, semantic in _RDNA35_DPP_F32_ROWS
        for overlay in _v_binary_f32_dpp16_uniform_rhs_overlays(
            operation=operation,
            instruction_name=instruction_name,
            low_mnemonic=low_mnemonic,
            semantic=semantic,
        )
    )


def _rdna35_dpp8_f32_uniform_rhs_overlays() -> tuple[AmdgpuDescriptorOverlay, ...]:
    return tuple(
        overlay
        for operation, instruction_name, low_mnemonic, semantic in _RDNA35_DPP_F32_ROWS
        for overlay in _v_binary_f32_dpp8_uniform_rhs_overlays(
            operation=operation,
            instruction_name=instruction_name,
            low_mnemonic=low_mnemonic,
            semantic=semantic,
        )
    )


__all__ = (
    "_RDNA35_DPP_F32_ROWS",
    "_rdna35_dpp_integer_compare_uniform_rhs_overlays",
    "_rdna35_dpp16_f32_uniform_rhs_overlays",
    "_rdna35_dpp8_f32_uniform_rhs_overlays",
    "_v_binary_f32_dpp16_uniform_rhs_overlays",
    "_v_binary_f32_dpp8_uniform_rhs_overlays",
)
