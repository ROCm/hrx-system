# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


# ruff: noqa: F403, F405

"""Matrix, dot, WMMA, MFMA, and SWMMAC descriptor overlays."""

from __future__ import annotations

from ..matrix_formats import (
    AMDGPU_CDNA4_MATRIX_FORMAT_ENUM_DOMAIN_NAMES,
    AMDGPU_F8F6F4_MATRIX_PHYSICAL_FORMATS,
)
from .common import *

_WMMA_GFX11_WAVE64_ACCUMULATOR_SIZE_REASON = "gfx11-wave64-wmma-half-width-accumulator"
_GFX12_WAVE64_MATRIX_OPERAND_SIZE_REASON = "gfx12-wave64-matrix-half-width-operands"
_CDNA4_F8F6F4_VARIABLE_OPERAND_WIDTH_REASON = (
    "cdna4-f8f6f4-selector-controls-operand-width"
)


def _cdna4_f8f6f4_operand_width_exception_reason(
    units: int,
) -> str | None:
    return None if units == 8 else _CDNA4_F8F6F4_VARIABLE_OPERAND_WIDTH_REASON


def _v_wmma_16x16x16_overlay(
    *,
    descriptor_key: str,
    instruction_name: str,
    mnemonic: str,
    low_mnemonic_suffix: str = "",
    semantic_tag: str,
    input_units: int,
    accumulator_units: int,
    input_size_exception_reason: str | None = None,
    accumulator_size_exception_reason: str | None = None,
    op_sel_hi_field: str = "OP_SEL_HI",
) -> AmdgpuDescriptorOverlay:
    low_mnemonic = f"{mnemonic}{low_mnemonic_suffix}"
    asm_forms = None
    if low_mnemonic_suffix:
        asm_forms = _asm(
            native_assembly_mnemonic=mnemonic,
            results=("dst",),
            operands=("lhs", "rhs", "acc"),
        )
    return AmdgpuDescriptorOverlay(
        descriptor_key=descriptor_key,
        instruction_name=instruction_name,
        mnemonic=low_mnemonic,
        encoding_name="ENC_VOP3P",
        semantic_tag=semantic_tag,
        schedule_class=_SCHEDULE_WMMA,
        operands=(
            AmdgpuOperandOverlay(
                "VDST",
                _vgpr_result(units=accumulator_units),
                size_exception_reason=accumulator_size_exception_reason,
            ),
            AmdgpuOperandOverlay(
                "SRC0",
                _vgpr_operand("lhs", units=input_units),
                size_exception_reason=input_size_exception_reason,
            ),
            AmdgpuOperandOverlay(
                "SRC1",
                _vgpr_operand("rhs", units=input_units),
                size_exception_reason=input_size_exception_reason,
            ),
            AmdgpuOperandOverlay(
                "SRC2",
                _vgpr_const_operand("acc", units=accumulator_units),
                size_exception_reason=accumulator_size_exception_reason,
            ),
        ),
        fixed_encoding_fields=((op_sel_hi_field, 0x7),),
        constraints=_destructive_accumulator_constraints(3),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
        asm_forms=asm_forms,
    )


def _with_zero_accumulator_form(
    overlay: AmdgpuDescriptorOverlay,
) -> tuple[AmdgpuDescriptorOverlay, AmdgpuDescriptorOverlay]:
    if overlay.mnemonic is None:
        raise ValueError(
            f"WMMA descriptor '{overlay.descriptor_key}' needs a mnemonic for "
            "its zero-accumulator asm form"
        )
    zero_descriptor_key = f"{overlay.descriptor_key}.acc_zero"
    source_overlay = replace(
        overlay,
        operand_forms=(
            *overlay.operand_forms,
            OperandForm(
                replacement_descriptor=zero_descriptor_key,
                matches=(
                    OperandFormMatch(
                        source_operand="acc",
                        match_kind=OperandFormMatchKind.ALL_EQUAL_I64,
                        match_i64=0,
                    ),
                ),
            ),
        ),
    )

    zero_operands = tuple(
        operand_overlay
        for operand_overlay in overlay.operands
        if operand_overlay.descriptor_operand.field_name != "acc"
    )
    zero_result_names = tuple(
        operand_overlay.descriptor_operand.field_name
        for operand_overlay in zero_operands
        if operand_overlay.descriptor_operand.role is OperandRole.RESULT
    )
    zero_operand_names = tuple(
        operand_overlay.descriptor_operand.field_name
        for operand_overlay in zero_operands
        if operand_overlay.descriptor_operand.role
        in (OperandRole.OPERAND, OperandRole.PREDICATE, OperandRole.RESOURCE)
        and OperandFlag.IMPLICIT not in operand_overlay.descriptor_operand.flags
    )
    base_native_mnemonic = overlay.mnemonic
    if overlay.asm_forms is not None:
        if len(overlay.asm_forms) != 1:
            raise ValueError(
                f"WMMA descriptor '{overlay.descriptor_key}' must have exactly "
                "one base asm form for its zero-accumulator asm form"
            )
        base_native_mnemonic = (
            overlay.asm_forms[0].native_assembly_mnemonic or overlay.mnemonic
        )
    zero_native_values = tuple(
        _native_result(field_name) for field_name in zero_result_names
    ) + tuple(_native_operand(field_name) for field_name in zero_operand_names)
    zero_native_modifiers = ()
    if overlay.asm_forms is not None:
        zero_native_modifiers = tuple(
            value
            for value in overlay.asm_forms[0].native_assembly_values
            if value.kind
            in (
                NativeAsmValueKind.IMMEDIATE_I64,
                NativeAsmValueKind.IMMEDIATE_UNSIGNED_HEX,
                NativeAsmValueKind.IMMEDIATE_TARGET_FORMAT,
            )
        )
    zero_overlay = replace(
        overlay,
        descriptor_key=zero_descriptor_key,
        operands=zero_operands,
        fixed_encoding_fields=(
            *overlay.fixed_encoding_fields,
            ("SRC2", _predefined("0")),
        ),
        constraints=_EARLY_CLOBBER_RESULT_CONSTRAINTS,
        operand_forms=(),
        asm_forms=_asm(
            mnemonic=f"{overlay.mnemonic}_acc_zero",
            native_assembly_mnemonic=base_native_mnemonic,
            results=zero_result_names,
            operands=zero_operand_names,
            native_assembly_values=(
                *zero_native_values,
                _native_literal("0"),
                *zero_native_modifiers,
            ),
        ),
    )
    return source_overlay, zero_overlay


def _v_wmma_f32_16x16x16_f16_overlay(
    *,
    descriptor_key_suffix: str = "",
    low_mnemonic_suffix: str = "",
    input_units: int = 4,
    accumulator_units: int = 8,
    input_size_exception_reason: str | None = None,
    accumulator_size_exception_reason: str | None = None,
    op_sel_hi_field: str = "OP_SEL_HI",
) -> AmdgpuDescriptorOverlay:
    return _v_wmma_16x16x16_overlay(
        descriptor_key=f"amdgpu.v_wmma_f32_16x16x16_f16{descriptor_key_suffix}",
        instruction_name="V_WMMA_F32_16X16X16_F16",
        mnemonic="v_wmma_f32_16x16x16_f16",
        low_mnemonic_suffix=low_mnemonic_suffix,
        semantic_tag="matrix.wmma.f32.16x16x16.f16",
        input_units=input_units,
        accumulator_units=accumulator_units,
        input_size_exception_reason=input_size_exception_reason,
        accumulator_size_exception_reason=accumulator_size_exception_reason,
        op_sel_hi_field=op_sel_hi_field,
    )


def _v_wmma_f32_16x16x16_bf16_overlay(
    *,
    descriptor_key_suffix: str = "",
    low_mnemonic_suffix: str = "",
    input_units: int = 4,
    accumulator_units: int = 8,
    input_size_exception_reason: str | None = None,
    accumulator_size_exception_reason: str | None = None,
    op_sel_hi_field: str = "OP_SEL_HI",
) -> AmdgpuDescriptorOverlay:
    return _v_wmma_16x16x16_overlay(
        descriptor_key=f"amdgpu.v_wmma_f32_16x16x16_bf16{descriptor_key_suffix}",
        instruction_name="V_WMMA_F32_16X16X16_BF16",
        mnemonic="v_wmma_f32_16x16x16_bf16",
        low_mnemonic_suffix=low_mnemonic_suffix,
        semantic_tag="matrix.wmma.f32.16x16x16.bf16",
        input_units=input_units,
        accumulator_units=accumulator_units,
        input_size_exception_reason=input_size_exception_reason,
        accumulator_size_exception_reason=accumulator_size_exception_reason,
        op_sel_hi_field=op_sel_hi_field,
    )


def _v_wmma_f16_16x16x16_f16_overlay(
    *,
    descriptor_key_suffix: str = "",
    low_mnemonic_suffix: str = "",
    input_units: int = 4,
    accumulator_units: int = 4,
    input_size_exception_reason: str | None = None,
    accumulator_size_exception_reason: str | None = None,
    op_sel_hi_field: str = "OP_SEL_HI",
) -> AmdgpuDescriptorOverlay:
    return _v_wmma_16x16x16_overlay(
        descriptor_key=f"amdgpu.v_wmma_f16_16x16x16_f16{descriptor_key_suffix}",
        instruction_name="V_WMMA_F16_16X16X16_F16",
        mnemonic="v_wmma_f16_16x16x16_f16",
        low_mnemonic_suffix=low_mnemonic_suffix,
        semantic_tag="matrix.wmma.f16.16x16x16.f16",
        input_units=input_units,
        accumulator_units=accumulator_units,
        input_size_exception_reason=input_size_exception_reason,
        accumulator_size_exception_reason=accumulator_size_exception_reason,
        op_sel_hi_field=op_sel_hi_field,
    )


def _v_wmma_bf16_16x16x16_bf16_overlay(
    *,
    descriptor_key_suffix: str = "",
    low_mnemonic_suffix: str = "",
    input_units: int = 4,
    accumulator_units: int = 4,
    input_size_exception_reason: str | None = None,
    accumulator_size_exception_reason: str | None = None,
    op_sel_hi_field: str = "OP_SEL_HI",
) -> AmdgpuDescriptorOverlay:
    return _v_wmma_16x16x16_overlay(
        descriptor_key=f"amdgpu.v_wmma_bf16_16x16x16_bf16{descriptor_key_suffix}",
        instruction_name="V_WMMA_BF16_16X16X16_BF16",
        mnemonic="v_wmma_bf16_16x16x16_bf16",
        low_mnemonic_suffix=low_mnemonic_suffix,
        semantic_tag="matrix.wmma.bf16.16x16x16.bf16",
        input_units=input_units,
        accumulator_units=accumulator_units,
        input_size_exception_reason=input_size_exception_reason,
        accumulator_size_exception_reason=accumulator_size_exception_reason,
        op_sel_hi_field=op_sel_hi_field,
    )


def _v_wmma_f32_16x16x16_packed8_overlay(
    *,
    lhs_type: str,
    rhs_type: str,
    descriptor_key_suffix: str = "",
    low_mnemonic_suffix: str = "",
    input_units: int = 2,
    accumulator_units: int = 8,
    input_size_exception_reason: str | None = None,
    accumulator_size_exception_reason: str | None = None,
    op_sel_hi_field: str = "OP_SEL_HI",
) -> AmdgpuDescriptorOverlay:
    lhs_type_upper = lhs_type.upper()
    rhs_type_upper = rhs_type.upper()
    return _v_wmma_16x16x16_overlay(
        descriptor_key=(
            f"amdgpu.v_wmma_f32_16x16x16_{lhs_type}_{rhs_type}{descriptor_key_suffix}"
        ),
        instruction_name=f"V_WMMA_F32_16X16X16_{lhs_type_upper}_{rhs_type_upper}",
        mnemonic=f"v_wmma_f32_16x16x16_{lhs_type}_{rhs_type}",
        low_mnemonic_suffix=low_mnemonic_suffix,
        semantic_tag=f"matrix.wmma.f32.16x16x16.{lhs_type}.{rhs_type}",
        input_units=input_units,
        accumulator_units=accumulator_units,
        input_size_exception_reason=input_size_exception_reason,
        accumulator_size_exception_reason=accumulator_size_exception_reason,
        op_sel_hi_field=op_sel_hi_field,
    )


def _v_wmma_i32_16x16x16_overlay(
    *,
    descriptor_key: str,
    descriptor_key_suffix: str = "",
    instruction_name: str,
    mnemonic: str,
    low_mnemonic_suffix: str = "",
    semantic_tag: str,
    operand_units: int,
    accumulator_units: int = 8,
    operand_size_exception_reason: str | None = None,
    accumulator_size_exception_reason: str | None = None,
    mirrors_sign_select_to_high_halves: bool = False,
    op_sel_hi_field: str = "OP_SEL_HI",
) -> AmdgpuDescriptorOverlay:
    low_mnemonic = f"{mnemonic}{low_mnemonic_suffix}"
    immediate_fields = (
        ("NEG", "NEG_HI", "CLAMP")
        if mirrors_sign_select_to_high_halves
        else ("NEG", "CLAMP")
    )
    immediates = (
        (
            _MATRIX_SIGN_SELECT_IMMEDIATE,
            _MATRIX_SIGN_SELECT_HIGH_IMMEDIATE,
            _MATRIX_CLAMP_IMMEDIATE,
        )
        if mirrors_sign_select_to_high_halves
        else (_MATRIX_SIGN_SELECT_IMMEDIATE, _MATRIX_CLAMP_IMMEDIATE)
    )
    native_modifiers = (
        _native_amdgpu_named_bit_list_immediate("neg_lo", 3),
        *(
            (_native_amdgpu_named_bit_list_immediate("neg_hi", 3),)
            if mirrors_sign_select_to_high_halves
            else ()
        ),
        _native_amdgpu_named_flag_immediate("clamp"),
    )
    asm_forms = _asm(
        native_assembly_mnemonic=mnemonic if low_mnemonic_suffix else None,
        results=("dst",),
        operands=("lhs", "rhs", "acc"),
        immediates=tuple(immediate.field_name for immediate in immediates),
        named_immediates=True,
        native_assembly_values=(
            _native_result("dst"),
            _native_operand("lhs"),
            _native_operand("rhs"),
            _native_operand("acc"),
            *native_modifiers,
        ),
    )
    return AmdgpuDescriptorOverlay(
        descriptor_key=f"{descriptor_key}{descriptor_key_suffix}",
        instruction_name=instruction_name,
        mnemonic=low_mnemonic,
        encoding_name="ENC_VOP3P",
        semantic_tag=semantic_tag,
        schedule_class=_SCHEDULE_WMMA,
        operands=(
            AmdgpuOperandOverlay(
                "VDST",
                _vgpr_result(units=accumulator_units),
                size_exception_reason=accumulator_size_exception_reason,
            ),
            AmdgpuOperandOverlay(
                "SRC0",
                _vgpr_operand("lhs", units=operand_units),
                size_exception_reason=operand_size_exception_reason,
            ),
            AmdgpuOperandOverlay(
                "SRC1",
                _vgpr_operand("rhs", units=operand_units),
                size_exception_reason=operand_size_exception_reason,
            ),
            AmdgpuOperandOverlay(
                "SRC2",
                _vgpr_const_operand("acc", units=accumulator_units),
                size_exception_reason=accumulator_size_exception_reason,
            ),
        ),
        immediate_fields=immediate_fields,
        immediates=immediates,
        fixed_encoding_fields=((op_sel_hi_field, 0x7),),
        constraints=_destructive_accumulator_constraints(3),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
        asm_forms=asm_forms,
    )


def _v_wmma_i32_16x16x16_iu8_overlay(
    *,
    descriptor_key_suffix: str = "",
    low_mnemonic_suffix: str = "",
    operand_units: int = 2,
    accumulator_units: int = 8,
    operand_size_exception_reason: str | None = None,
    accumulator_size_exception_reason: str | None = None,
    mirrors_sign_select_to_high_halves: bool = False,
    op_sel_hi_field: str = "OP_SEL_HI",
) -> AmdgpuDescriptorOverlay:
    return _v_wmma_i32_16x16x16_overlay(
        descriptor_key="amdgpu.v_wmma_i32_16x16x16_iu8",
        descriptor_key_suffix=descriptor_key_suffix,
        instruction_name="V_WMMA_I32_16X16X16_IU8",
        mnemonic="v_wmma_i32_16x16x16_iu8",
        low_mnemonic_suffix=low_mnemonic_suffix,
        semantic_tag="matrix.wmma.i32.16x16x16.iu8",
        operand_units=operand_units,
        accumulator_units=accumulator_units,
        operand_size_exception_reason=operand_size_exception_reason,
        accumulator_size_exception_reason=accumulator_size_exception_reason,
        mirrors_sign_select_to_high_halves=mirrors_sign_select_to_high_halves,
        op_sel_hi_field=op_sel_hi_field,
    )


def _v_wmma_i32_16x16x16_iu4_overlay(
    *,
    descriptor_key_suffix: str = "",
    low_mnemonic_suffix: str = "",
    operand_units: int = 1,
    accumulator_units: int = 8,
    operand_size_exception_reason: str | None = None,
    accumulator_size_exception_reason: str | None = None,
    mirrors_sign_select_to_high_halves: bool = False,
    op_sel_hi_field: str = "OP_SEL_HI",
) -> AmdgpuDescriptorOverlay:
    return _v_wmma_i32_16x16x16_overlay(
        descriptor_key="amdgpu.v_wmma_i32_16x16x16_iu4",
        descriptor_key_suffix=descriptor_key_suffix,
        instruction_name="V_WMMA_I32_16X16X16_IU4",
        mnemonic="v_wmma_i32_16x16x16_iu4",
        low_mnemonic_suffix=low_mnemonic_suffix,
        semantic_tag="matrix.wmma.i32.16x16x16.iu4",
        operand_units=operand_units,
        accumulator_units=accumulator_units,
        operand_size_exception_reason=operand_size_exception_reason,
        accumulator_size_exception_reason=accumulator_size_exception_reason,
        mirrors_sign_select_to_high_halves=mirrors_sign_select_to_high_halves,
        op_sel_hi_field=op_sel_hi_field,
    )


def _v_wmma_i32_16x16x32_iu4_overlay(
    *,
    descriptor_key_suffix: str = "",
    low_mnemonic_suffix: str = "",
    operand_units: int = 2,
    accumulator_units: int = 8,
    operand_size_exception_reason: str | None = None,
    accumulator_size_exception_reason: str | None = None,
    op_sel_hi_field: str = "OP_SEL_HI",
) -> AmdgpuDescriptorOverlay:
    return _v_wmma_i32_16x16x16_overlay(
        descriptor_key="amdgpu.v_wmma_i32_16x16x32_iu4",
        descriptor_key_suffix=descriptor_key_suffix,
        instruction_name="V_WMMA_I32_16X16X32_IU4",
        mnemonic="v_wmma_i32_16x16x32_iu4",
        low_mnemonic_suffix=low_mnemonic_suffix,
        semantic_tag="matrix.wmma.i32.16x16x32.iu4",
        operand_units=operand_units,
        accumulator_units=accumulator_units,
        operand_size_exception_reason=operand_size_exception_reason,
        accumulator_size_exception_reason=accumulator_size_exception_reason,
        op_sel_hi_field=op_sel_hi_field,
    )


def _matrix_schedule_class(descriptor_key: str) -> str:
    if _amdgpu_mfma_has_qualified_timing(descriptor_key):
        return _amdgpu_mfma_schedule_class_name(descriptor_key)
    return _SCHEDULE_MFMA


def _cdna4_f8f6f4_matrix_format_immediate(
    field_name: str, physical_format: str
) -> Immediate:
    return Immediate(
        field_name,
        ImmediateKind.ENUM,
        bit_width=3,
        enum_domain=AMDGPU_CDNA4_MATRIX_FORMAT_ENUM_DOMAIN_NAMES[physical_format],
    )


def _cdna4_f8f6f4_matrix_format_immediates(
    matrix_physical_formats: tuple[str, str],
) -> tuple[Immediate, ...]:
    lhs_physical_format, rhs_physical_format = matrix_physical_formats
    return (
        _cdna4_f8f6f4_matrix_format_immediate("matrix_a_fmt", lhs_physical_format),
        _cdna4_f8f6f4_matrix_format_immediate("matrix_b_fmt", rhs_physical_format),
    )


def _cdna4_f8f6f4_matrix_asm_forms(
    mnemonic: str, *, has_scale_operands: bool
) -> tuple[AsmForm, ...]:
    operands = (
        ("lhs", "rhs", "acc", "lhs_scale", "rhs_scale")
        if has_scale_operands
        else ("lhs", "rhs", "acc")
    )
    return _asm(
        native_assembly_mnemonic=mnemonic,
        results=("dst",),
        operands=operands,
        immediates=("matrix_a_fmt", "matrix_b_fmt"),
        named_immediates=True,
        native_assembly_values=(
            _native_result("dst"),
            *(_native_operand(operand) for operand in operands),
            _native_amdgpu_required_named_i64_immediate("matrix_a_fmt", name="cbsz"),
            _native_amdgpu_required_named_i64_immediate("matrix_b_fmt", name="blgp"),
        ),
    )


def _v_mfma_overlay(
    *,
    instruction_name: str,
    semantic_tag: str,
    accumulator_units: int,
    lhs_units: int,
    rhs_units: int,
    encoding_name: str = "VOP3P_MFMA",
    low_mnemonic_suffix: str = "",
    matrix_physical_formats: tuple[str, str] | None = None,
) -> AmdgpuDescriptorOverlay:
    native_mnemonic = instruction_name.lower()
    mnemonic = f"{native_mnemonic}{low_mnemonic_suffix}"
    descriptor_key = f"amdgpu.{mnemonic}"
    matrix_format_immediates = (
        _cdna4_f8f6f4_matrix_format_immediates(matrix_physical_formats)
        if matrix_physical_formats is not None
        else ()
    )
    return AmdgpuDescriptorOverlay(
        descriptor_key=descriptor_key,
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        encoding_name=encoding_name,
        semantic_tag=semantic_tag,
        schedule_class=_matrix_schedule_class(descriptor_key),
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_agpr_result(units=accumulator_units)),
            AmdgpuOperandOverlay(
                "SRC0",
                _vgpr_agpr_operand("lhs", units=lhs_units),
                size_exception_reason=(
                    _cdna4_f8f6f4_operand_width_exception_reason(lhs_units)
                    if matrix_physical_formats is not None
                    else None
                ),
            ),
            AmdgpuOperandOverlay(
                "SRC1",
                _vgpr_agpr_operand("rhs", units=rhs_units),
                size_exception_reason=(
                    _cdna4_f8f6f4_operand_width_exception_reason(rhs_units)
                    if matrix_physical_formats is not None
                    else None
                ),
            ),
            AmdgpuOperandOverlay(
                "SRC2", _vgpr_agpr_const_operand("acc", units=accumulator_units)
            ),
        ),
        immediate_fields=(
            ("CBSZ", "BLGP") if matrix_physical_formats is not None else ()
        ),
        immediates=matrix_format_immediates,
        constraints=_destructive_accumulator_constraints(3),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
        asm_forms=(
            _cdna4_f8f6f4_matrix_asm_forms(native_mnemonic, has_scale_operands=False)
            if matrix_physical_formats is not None
            else None
        ),
    )


def _v_mfma_scale_overlay(
    *,
    instruction_name: str,
    semantic_tag: str,
    accumulator_units: int,
    lhs_units: int,
    rhs_units: int,
    low_mnemonic_suffix: str,
    matrix_physical_formats: tuple[str, str],
) -> AmdgpuDescriptorOverlay:
    native_mnemonic = instruction_name.lower()
    mnemonic = f"{native_mnemonic}{low_mnemonic_suffix}"
    descriptor_key = f"amdgpu.{mnemonic}"
    return AmdgpuDescriptorOverlay(
        descriptor_key=descriptor_key,
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        encoding_name="ENC_VOP3PX2",
        semantic_tag=semantic_tag,
        schedule_class=_matrix_schedule_class(descriptor_key),
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_agpr_result(units=accumulator_units)),
            AmdgpuOperandOverlay(
                "SRC0",
                _vgpr_agpr_operand("lhs", units=lhs_units),
                size_exception_reason=(
                    _cdna4_f8f6f4_operand_width_exception_reason(lhs_units)
                ),
            ),
            AmdgpuOperandOverlay(
                "SRC1",
                _vgpr_agpr_operand("rhs", units=rhs_units),
                size_exception_reason=(
                    _cdna4_f8f6f4_operand_width_exception_reason(rhs_units)
                ),
            ),
            AmdgpuOperandOverlay(
                "SRC2", _vgpr_agpr_const_operand("acc", units=accumulator_units)
            ),
            AmdgpuOperandOverlay("SCALE_SRC0", _sgpr_vgpr_operand("lhs_scale")),
            AmdgpuOperandOverlay("SCALE_SRC1", _sgpr_vgpr_operand("rhs_scale")),
        ),
        immediate_fields=("CBSZ", "BLGP"),
        immediates=_cdna4_f8f6f4_matrix_format_immediates(matrix_physical_formats),
        fixed_encoding_fields=(
            ("ABID", 1),
            ("ENCODING", 0x1A7),
            ("X2ENCODING", 0xD3AC),
        ),
        constraints=_destructive_accumulator_constraints(3),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
        asm_forms=_cdna4_f8f6f4_matrix_asm_forms(
            native_mnemonic, has_scale_operands=True
        ),
    )


def _v_mfma_f32_16x16x16_f16_overlay() -> AmdgpuDescriptorOverlay:
    return _v_mfma_overlay(
        instruction_name="V_MFMA_F32_16X16X16_F16",
        semantic_tag="matrix.mfma.f32.16x16x16.f16",
        accumulator_units=4,
        lhs_units=2,
        rhs_units=2,
    )


def _v_mfma_f32_16x16x16_bf16_overlay() -> AmdgpuDescriptorOverlay:
    return _v_mfma_overlay(
        instruction_name="V_MFMA_F32_16X16X16_BF16",
        semantic_tag="matrix.mfma.f32.16x16x16.bf16.1k",
        accumulator_units=4,
        lhs_units=2,
        rhs_units=2,
    )


def _v_mfma_f32_16x16x4_f32_overlay() -> AmdgpuDescriptorOverlay:
    return _v_mfma_overlay(
        instruction_name="V_MFMA_F32_16X16X4_F32",
        semantic_tag="matrix.mfma.f32.16x16x4.f32",
        accumulator_units=4,
        lhs_units=1,
        rhs_units=1,
    )


def _cdna3_mfma_overlays() -> tuple[AmdgpuDescriptorOverlay, ...]:
    return (
        _v_mfma_overlay(
            instruction_name="V_MFMA_F32_16X16X1_4B_F32",
            semantic_tag="matrix.mfma.f32.16x16x1.f32",
            accumulator_units=16,
            lhs_units=1,
            rhs_units=1,
        ),
        _v_mfma_overlay(
            instruction_name="V_MFMA_F32_16X16X4_4B_F16",
            semantic_tag="matrix.mfma.f32.16x16x4.f16",
            accumulator_units=16,
            lhs_units=2,
            rhs_units=2,
        ),
        _v_mfma_f32_16x16x4_f32_overlay(),
        _v_mfma_overlay(
            instruction_name="V_MFMA_F32_16X16X4_4B_BF16",
            semantic_tag="matrix.mfma.f32.16x16x4.bf16.1k",
            accumulator_units=16,
            lhs_units=2,
            rhs_units=2,
        ),
        _v_mfma_overlay(
            instruction_name="V_MFMA_F32_16X16X8_XF32",
            semantic_tag="matrix.mfma.f32.16x16x8.xf32",
            accumulator_units=4,
            lhs_units=2,
            rhs_units=2,
        ),
        _v_mfma_f32_16x16x16_f16_overlay(),
        _v_mfma_f32_16x16x16_bf16_overlay(),
        _v_mfma_overlay(
            instruction_name="V_MFMA_F32_16X16X32_BF8_BF8",
            semantic_tag="matrix.mfma.f32.16x16x32.bf8.bf8",
            accumulator_units=4,
            lhs_units=2,
            rhs_units=2,
        ),
        _v_mfma_overlay(
            instruction_name="V_MFMA_F32_16X16X32_BF8_FP8",
            semantic_tag="matrix.mfma.f32.16x16x32.bf8.fp8",
            accumulator_units=4,
            lhs_units=2,
            rhs_units=2,
        ),
        _v_mfma_overlay(
            instruction_name="V_MFMA_F32_16X16X32_FP8_BF8",
            semantic_tag="matrix.mfma.f32.16x16x32.fp8.bf8",
            accumulator_units=4,
            lhs_units=2,
            rhs_units=2,
        ),
        _v_mfma_overlay(
            instruction_name="V_MFMA_F32_16X16X32_FP8_FP8",
            semantic_tag="matrix.mfma.f32.16x16x32.fp8.fp8",
            accumulator_units=4,
            lhs_units=2,
            rhs_units=2,
        ),
        _v_mfma_overlay(
            instruction_name="V_MFMA_F32_32X32X1_2B_F32",
            semantic_tag="matrix.mfma.f32.32x32x1.f32",
            accumulator_units=32,
            lhs_units=1,
            rhs_units=1,
        ),
        _v_mfma_overlay(
            instruction_name="V_MFMA_F32_32X32X2_F32",
            semantic_tag="matrix.mfma.f32.32x32x2.f32",
            accumulator_units=16,
            lhs_units=1,
            rhs_units=1,
        ),
        _v_mfma_overlay(
            instruction_name="V_MFMA_F32_32X32X4_XF32",
            semantic_tag="matrix.mfma.f32.32x32x4.xf32",
            accumulator_units=16,
            lhs_units=2,
            rhs_units=2,
        ),
        _v_mfma_overlay(
            instruction_name="V_MFMA_F32_32X32X4_2B_F16",
            semantic_tag="matrix.mfma.f32.32x32x4.f16",
            accumulator_units=32,
            lhs_units=2,
            rhs_units=2,
        ),
        _v_mfma_overlay(
            instruction_name="V_MFMA_F32_32X32X4_2B_BF16",
            semantic_tag="matrix.mfma.f32.32x32x4.bf16.1k",
            accumulator_units=32,
            lhs_units=2,
            rhs_units=2,
        ),
        _v_mfma_overlay(
            instruction_name="V_MFMA_F32_32X32X8_F16",
            semantic_tag="matrix.mfma.f32.32x32x8.f16",
            accumulator_units=16,
            lhs_units=2,
            rhs_units=2,
        ),
        _v_mfma_overlay(
            instruction_name="V_MFMA_F32_32X32X8_BF16",
            semantic_tag="matrix.mfma.f32.32x32x8.bf16.1k",
            accumulator_units=16,
            lhs_units=2,
            rhs_units=2,
        ),
        _v_mfma_overlay(
            instruction_name="V_MFMA_F32_32X32X16_BF8_BF8",
            semantic_tag="matrix.mfma.f32.32x32x16.bf8.bf8",
            accumulator_units=16,
            lhs_units=2,
            rhs_units=2,
        ),
        _v_mfma_overlay(
            instruction_name="V_MFMA_F32_32X32X16_BF8_FP8",
            semantic_tag="matrix.mfma.f32.32x32x16.bf8.fp8",
            accumulator_units=16,
            lhs_units=2,
            rhs_units=2,
        ),
        _v_mfma_overlay(
            instruction_name="V_MFMA_F32_32X32X16_FP8_BF8",
            semantic_tag="matrix.mfma.f32.32x32x16.fp8.bf8",
            accumulator_units=16,
            lhs_units=2,
            rhs_units=2,
        ),
        _v_mfma_overlay(
            instruction_name="V_MFMA_F32_32X32X16_FP8_FP8",
            semantic_tag="matrix.mfma.f32.32x32x16.fp8.fp8",
            accumulator_units=16,
            lhs_units=2,
            rhs_units=2,
        ),
        _v_mfma_overlay(
            instruction_name="V_MFMA_F32_4X4X1_16B_F32",
            semantic_tag="matrix.mfma.f32.4x4x1.f32",
            accumulator_units=4,
            lhs_units=1,
            rhs_units=1,
        ),
        _v_mfma_overlay(
            instruction_name="V_MFMA_F32_4X4X4_16B_F16",
            semantic_tag="matrix.mfma.f32.4x4x4.f16",
            accumulator_units=4,
            lhs_units=2,
            rhs_units=2,
        ),
        _v_mfma_overlay(
            instruction_name="V_MFMA_F32_4X4X4_16B_BF16",
            semantic_tag="matrix.mfma.f32.4x4x4.bf16.1k",
            accumulator_units=4,
            lhs_units=2,
            rhs_units=2,
        ),
        _v_mfma_overlay(
            instruction_name="V_MFMA_F64_16X16X4_F64",
            semantic_tag="matrix.mfma.f64.16x16x4.f64",
            accumulator_units=8,
            lhs_units=2,
            rhs_units=2,
        ),
        _v_mfma_overlay(
            instruction_name="V_MFMA_F64_4X4X4_4B_F64",
            semantic_tag="matrix.mfma.f64.4x4x4.f64",
            accumulator_units=2,
            lhs_units=2,
            rhs_units=2,
        ),
        _v_mfma_overlay(
            instruction_name="V_MFMA_I32_16X16X4_4B_I8",
            semantic_tag="matrix.mfma.i32.16x16x4.i8",
            accumulator_units=16,
            lhs_units=1,
            rhs_units=1,
        ),
        _v_mfma_overlay(
            instruction_name="V_MFMA_I32_16X16X32_I8",
            semantic_tag="matrix.mfma.i32.16x16x32.i8",
            accumulator_units=4,
            lhs_units=2,
            rhs_units=2,
        ),
        _v_mfma_overlay(
            instruction_name="V_MFMA_I32_32X32X4_2B_I8",
            semantic_tag="matrix.mfma.i32.32x32x4.i8",
            accumulator_units=32,
            lhs_units=1,
            rhs_units=1,
        ),
        _v_mfma_overlay(
            instruction_name="V_MFMA_I32_32X32X16_I8",
            semantic_tag="matrix.mfma.i32.32x32x16.i8",
            accumulator_units=16,
            lhs_units=2,
            rhs_units=2,
        ),
        _v_mfma_overlay(
            instruction_name="V_MFMA_I32_4X4X4_16B_I8",
            semantic_tag="matrix.mfma.i32.4x4x4.i8",
            accumulator_units=4,
            lhs_units=1,
            rhs_units=1,
        ),
    )


def _cdna4_f8f6f4_mfma_overlays(
    *,
    instruction_name: str,
    semantic_tag: str,
    accumulator_units: int,
    operand_element_count: int,
    has_scale_operands: bool,
) -> tuple[AmdgpuDescriptorOverlay, ...]:
    return tuple(
        (
            _v_mfma_scale_overlay(
                instruction_name=instruction_name,
                semantic_tag=(f"{semantic_tag}.{lhs_format.token}.{rhs_format.token}"),
                accumulator_units=accumulator_units,
                lhs_units=lhs_format.register_count_for(operand_element_count),
                rhs_units=rhs_format.register_count_for(operand_element_count),
                low_mnemonic_suffix=(f"_{lhs_format.token}_{rhs_format.token}"),
                matrix_physical_formats=(
                    lhs_format.token,
                    rhs_format.token,
                ),
            )
            if has_scale_operands
            else _v_mfma_overlay(
                instruction_name=instruction_name,
                semantic_tag=(f"{semantic_tag}.{lhs_format.token}.{rhs_format.token}"),
                accumulator_units=accumulator_units,
                lhs_units=lhs_format.register_count_for(operand_element_count),
                rhs_units=rhs_format.register_count_for(operand_element_count),
                low_mnemonic_suffix=(f"_{lhs_format.token}_{rhs_format.token}"),
                matrix_physical_formats=(
                    lhs_format.token,
                    rhs_format.token,
                ),
            )
        )
        for lhs_format in AMDGPU_F8F6F4_MATRIX_PHYSICAL_FORMATS
        for rhs_format in AMDGPU_F8F6F4_MATRIX_PHYSICAL_FORMATS
    )


def _cdna4_mfma_overlays() -> tuple[AmdgpuDescriptorOverlay, ...]:
    return (
        *(
            overlay
            for overlay in _cdna3_mfma_overlays()
            if overlay.instruction_name
            not in (
                "V_MFMA_F32_16X16X8_XF32",
                "V_MFMA_F32_32X32X4_XF32",
            )
        ),
        _v_mfma_overlay(
            instruction_name="V_MFMA_F32_16X16X32_F16",
            semantic_tag="matrix.mfma.f32.16x16x32.f16",
            accumulator_units=4,
            lhs_units=4,
            rhs_units=4,
        ),
        _v_mfma_overlay(
            instruction_name="V_MFMA_F32_16X16X32_BF16",
            semantic_tag="matrix.mfma.f32.16x16x32.bf16",
            accumulator_units=4,
            lhs_units=4,
            rhs_units=4,
        ),
        _v_mfma_overlay(
            instruction_name="V_MFMA_F32_32X32X16_F16",
            semantic_tag="matrix.mfma.f32.32x32x16.f16",
            accumulator_units=16,
            lhs_units=4,
            rhs_units=4,
        ),
        _v_mfma_overlay(
            instruction_name="V_MFMA_F32_32X32X16_BF16",
            semantic_tag="matrix.mfma.f32.32x32x16.bf16",
            accumulator_units=16,
            lhs_units=4,
            rhs_units=4,
        ),
        _v_mfma_overlay(
            instruction_name="V_MFMA_I32_16X16X64_I8",
            semantic_tag="matrix.mfma.i32.16x16x64.i8",
            accumulator_units=4,
            lhs_units=4,
            rhs_units=4,
        ),
        _v_mfma_overlay(
            instruction_name="V_MFMA_I32_32X32X32_I8",
            semantic_tag="matrix.mfma.i32.32x32x32.i8",
            accumulator_units=16,
            lhs_units=4,
            rhs_units=4,
        ),
        *_cdna4_f8f6f4_mfma_overlays(
            instruction_name="V_MFMA_F32_16X16X128_F8F6F4",
            semantic_tag="matrix.mfma.f32.16x16x128.f8f6f4",
            accumulator_units=4,
            operand_element_count=32,
            has_scale_operands=False,
        ),
        *_cdna4_f8f6f4_mfma_overlays(
            instruction_name="V_MFMA_F32_32X32X64_F8F6F4",
            semantic_tag="matrix.mfma.f32.32x32x64.f8f6f4",
            accumulator_units=16,
            operand_element_count=32,
            has_scale_operands=False,
        ),
        *_cdna4_f8f6f4_mfma_overlays(
            instruction_name="V_MFMA_SCALE_F32_16X16X128_F8F6F4",
            semantic_tag="matrix.mfma.scale.f32.16x16x128.f8f6f4",
            accumulator_units=4,
            operand_element_count=32,
            has_scale_operands=True,
        ),
        *_cdna4_f8f6f4_mfma_overlays(
            instruction_name="V_MFMA_SCALE_F32_32X32X64_F8F6F4",
            semantic_tag="matrix.mfma.scale.f32.32x32x64.f8f6f4",
            accumulator_units=16,
            operand_element_count=32,
            has_scale_operands=True,
        ),
    )


def _v_smfmac_f32_16x16x32_bf16_overlay() -> AmdgpuDescriptorOverlay:
    return _v_smfmac_f32_overlay(
        descriptor_key="amdgpu.v_smfmac_f32_16x16x32_bf16",
        instruction_name="V_SMFMAC_F32_16X16X32_BF16",
        mnemonic="v_smfmac_f32_16x16x32_bf16",
        semantic_tag="matrix.smfmac.f32.16x16x32.bf16",
        accumulator_units=4,
        lhs_units=2,
        rhs_units=4,
    )


def _v_smfmac_f32_16x16x32_f16_overlay() -> AmdgpuDescriptorOverlay:
    return _v_smfmac_f32_overlay(
        descriptor_key="amdgpu.v_smfmac_f32_16x16x32_f16",
        instruction_name="V_SMFMAC_F32_16X16X32_F16",
        mnemonic="v_smfmac_f32_16x16x32_f16",
        semantic_tag="matrix.smfmac.f32.16x16x32.f16",
        accumulator_units=4,
        lhs_units=2,
        rhs_units=4,
    )


def _v_smfmac_f32_32x32x16_bf16_overlay() -> AmdgpuDescriptorOverlay:
    return _v_smfmac_f32_overlay(
        descriptor_key="amdgpu.v_smfmac_f32_32x32x16_bf16",
        instruction_name="V_SMFMAC_F32_32X32X16_BF16",
        mnemonic="v_smfmac_f32_32x32x16_bf16",
        semantic_tag="matrix.smfmac.f32.32x32x16.bf16",
        accumulator_units=16,
        lhs_units=2,
        rhs_units=4,
    )


def _v_smfmac_f32_32x32x16_f16_overlay() -> AmdgpuDescriptorOverlay:
    return _v_smfmac_f32_overlay(
        descriptor_key="amdgpu.v_smfmac_f32_32x32x16_f16",
        instruction_name="V_SMFMAC_F32_32X32X16_F16",
        mnemonic="v_smfmac_f32_32x32x16_f16",
        semantic_tag="matrix.smfmac.f32.32x32x16.f16",
        accumulator_units=16,
        lhs_units=2,
        rhs_units=4,
    )


def _v_smfmac_f32_overlay(
    *,
    descriptor_key: str,
    instruction_name: str,
    mnemonic: str,
    semantic_tag: str,
    accumulator_units: int,
    lhs_units: int,
    rhs_units: int,
) -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key=descriptor_key,
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        encoding_name="VOP3P_MFMA",
        semantic_tag=semantic_tag,
        schedule_class=_matrix_schedule_class(descriptor_key),
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_agpr_result(units=accumulator_units)),
            AmdgpuOperandOverlay(
                "VDST",
                _vgpr_agpr_operand("acc", units=accumulator_units),
                role_exception_reason=_SMFMAC_VDST_ACCUMULATOR_REASON,
            ),
            AmdgpuOperandOverlay("SRC0", _vgpr_agpr_operand("lhs", units=lhs_units)),
            AmdgpuOperandOverlay("SRC1", _vgpr_agpr_operand("rhs", units=rhs_units)),
            AmdgpuOperandOverlay("SRC2", _vgpr_operand("sparse_metadata")),
        ),
        constraints=_DESTRUCTIVE_ACCUMULATOR_CONSTRAINTS,
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
        asm_forms=_asm(
            results=("dst",),
            operands=("acc", "lhs", "rhs", "sparse_metadata"),
            native_assembly_values=(
                _native_result("dst"),
                _native_operand("lhs"),
                _native_operand("rhs"),
                _native_operand("sparse_metadata"),
            ),
        ),
    )


def _cdna_smfmac_overlay(
    *,
    instruction_name: str,
    semantic_tag: str,
    accumulator_units: int,
    lhs_units: int,
    rhs_units: int,
) -> AmdgpuDescriptorOverlay:
    mnemonic = instruction_name.lower()
    return _v_smfmac_f32_overlay(
        descriptor_key=f"amdgpu.{mnemonic}",
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        semantic_tag=semantic_tag,
        accumulator_units=accumulator_units,
        lhs_units=lhs_units,
        rhs_units=rhs_units,
    )


def _cdna3_smfmac_overlays() -> tuple[AmdgpuDescriptorOverlay, ...]:
    return (
        _v_smfmac_f32_16x16x32_f16_overlay(),
        _v_smfmac_f32_16x16x32_bf16_overlay(),
        _cdna_smfmac_overlay(
            instruction_name="V_SMFMAC_F32_16X16X64_BF8_BF8",
            semantic_tag="matrix.smfmac.f32.16x16x64.bf8.bf8",
            accumulator_units=4,
            lhs_units=2,
            rhs_units=4,
        ),
        _cdna_smfmac_overlay(
            instruction_name="V_SMFMAC_F32_16X16X64_BF8_FP8",
            semantic_tag="matrix.smfmac.f32.16x16x64.bf8.fp8",
            accumulator_units=4,
            lhs_units=2,
            rhs_units=4,
        ),
        _cdna_smfmac_overlay(
            instruction_name="V_SMFMAC_F32_16X16X64_FP8_BF8",
            semantic_tag="matrix.smfmac.f32.16x16x64.fp8.bf8",
            accumulator_units=4,
            lhs_units=2,
            rhs_units=4,
        ),
        _cdna_smfmac_overlay(
            instruction_name="V_SMFMAC_F32_16X16X64_FP8_FP8",
            semantic_tag="matrix.smfmac.f32.16x16x64.fp8.fp8",
            accumulator_units=4,
            lhs_units=2,
            rhs_units=4,
        ),
        _cdna_smfmac_overlay(
            instruction_name="V_SMFMAC_I32_16X16X64_I8",
            semantic_tag="matrix.smfmac.i32.16x16x64.i8",
            accumulator_units=4,
            lhs_units=2,
            rhs_units=4,
        ),
        _v_smfmac_f32_32x32x16_f16_overlay(),
        _v_smfmac_f32_32x32x16_bf16_overlay(),
        _cdna_smfmac_overlay(
            instruction_name="V_SMFMAC_F32_32X32X32_BF8_BF8",
            semantic_tag="matrix.smfmac.f32.32x32x32.bf8.bf8",
            accumulator_units=16,
            lhs_units=2,
            rhs_units=4,
        ),
        _cdna_smfmac_overlay(
            instruction_name="V_SMFMAC_F32_32X32X32_BF8_FP8",
            semantic_tag="matrix.smfmac.f32.32x32x32.bf8.fp8",
            accumulator_units=16,
            lhs_units=2,
            rhs_units=4,
        ),
        _cdna_smfmac_overlay(
            instruction_name="V_SMFMAC_F32_32X32X32_FP8_BF8",
            semantic_tag="matrix.smfmac.f32.32x32x32.fp8.bf8",
            accumulator_units=16,
            lhs_units=2,
            rhs_units=4,
        ),
        _cdna_smfmac_overlay(
            instruction_name="V_SMFMAC_F32_32X32X32_FP8_FP8",
            semantic_tag="matrix.smfmac.f32.32x32x32.fp8.fp8",
            accumulator_units=16,
            lhs_units=2,
            rhs_units=4,
        ),
        _cdna_smfmac_overlay(
            instruction_name="V_SMFMAC_I32_32X32X32_I8",
            semantic_tag="matrix.smfmac.i32.32x32x32.i8",
            accumulator_units=16,
            lhs_units=2,
            rhs_units=4,
        ),
    )


def _cdna4_smfmac_overlays() -> tuple[AmdgpuDescriptorOverlay, ...]:
    return (
        *_cdna3_smfmac_overlays(),
        _cdna_smfmac_overlay(
            instruction_name="V_SMFMAC_F32_16X16X64_F16",
            semantic_tag="matrix.smfmac.f32.16x16x64.f16",
            accumulator_units=4,
            lhs_units=4,
            rhs_units=8,
        ),
        _cdna_smfmac_overlay(
            instruction_name="V_SMFMAC_F32_16X16X64_BF16",
            semantic_tag="matrix.smfmac.f32.16x16x64.bf16",
            accumulator_units=4,
            lhs_units=4,
            rhs_units=8,
        ),
        _cdna_smfmac_overlay(
            instruction_name="V_SMFMAC_F32_16X16X128_BF8_BF8",
            semantic_tag="matrix.smfmac.f32.16x16x128.bf8.bf8",
            accumulator_units=4,
            lhs_units=4,
            rhs_units=8,
        ),
        _cdna_smfmac_overlay(
            instruction_name="V_SMFMAC_F32_16X16X128_BF8_FP8",
            semantic_tag="matrix.smfmac.f32.16x16x128.bf8.fp8",
            accumulator_units=4,
            lhs_units=4,
            rhs_units=8,
        ),
        _cdna_smfmac_overlay(
            instruction_name="V_SMFMAC_F32_16X16X128_FP8_BF8",
            semantic_tag="matrix.smfmac.f32.16x16x128.fp8.bf8",
            accumulator_units=4,
            lhs_units=4,
            rhs_units=8,
        ),
        _cdna_smfmac_overlay(
            instruction_name="V_SMFMAC_F32_16X16X128_FP8_FP8",
            semantic_tag="matrix.smfmac.f32.16x16x128.fp8.fp8",
            accumulator_units=4,
            lhs_units=4,
            rhs_units=8,
        ),
        _cdna_smfmac_overlay(
            instruction_name="V_SMFMAC_I32_16X16X128_I8",
            semantic_tag="matrix.smfmac.i32.16x16x128.i8",
            accumulator_units=4,
            lhs_units=4,
            rhs_units=8,
        ),
        _cdna_smfmac_overlay(
            instruction_name="V_SMFMAC_F32_32X32X32_F16",
            semantic_tag="matrix.smfmac.f32.32x32x32.f16",
            accumulator_units=16,
            lhs_units=4,
            rhs_units=8,
        ),
        _cdna_smfmac_overlay(
            instruction_name="V_SMFMAC_F32_32X32X32_BF16",
            semantic_tag="matrix.smfmac.f32.32x32x32.bf16",
            accumulator_units=16,
            lhs_units=4,
            rhs_units=8,
        ),
        _cdna_smfmac_overlay(
            instruction_name="V_SMFMAC_F32_32X32X64_BF8_BF8",
            semantic_tag="matrix.smfmac.f32.32x32x64.bf8.bf8",
            accumulator_units=16,
            lhs_units=4,
            rhs_units=8,
        ),
        _cdna_smfmac_overlay(
            instruction_name="V_SMFMAC_F32_32X32X64_BF8_FP8",
            semantic_tag="matrix.smfmac.f32.32x32x64.bf8.fp8",
            accumulator_units=16,
            lhs_units=4,
            rhs_units=8,
        ),
        _cdna_smfmac_overlay(
            instruction_name="V_SMFMAC_F32_32X32X64_FP8_BF8",
            semantic_tag="matrix.smfmac.f32.32x32x64.fp8.bf8",
            accumulator_units=16,
            lhs_units=4,
            rhs_units=8,
        ),
        _cdna_smfmac_overlay(
            instruction_name="V_SMFMAC_F32_32X32X64_FP8_FP8",
            semantic_tag="matrix.smfmac.f32.32x32x64.fp8.fp8",
            accumulator_units=16,
            lhs_units=4,
            rhs_units=8,
        ),
        _cdna_smfmac_overlay(
            instruction_name="V_SMFMAC_I32_32X32X64_I8",
            semantic_tag="matrix.smfmac.i32.32x32x64.i8",
            accumulator_units=16,
            lhs_units=4,
            rhs_units=8,
        ),
    )


def _v_swmmac_overlay(
    *,
    instruction_name: str,
    semantic_tag: str,
    accumulator_units: int,
    lhs_units: int,
    rhs_units: int,
    descriptor_key_suffix: str = "",
    low_mnemonic_suffix: str = "",
    accumulator_size_exception_reason: str | None = None,
    lhs_size_exception_reason: str | None = None,
    rhs_size_exception_reason: str | None = None,
    has_integer_sign_select: bool = False,
    op_sel_hi_field: str = "OP_SEL_HI",
) -> AmdgpuDescriptorOverlay:
    mnemonic = instruction_name.lower()
    low_mnemonic = f"{mnemonic}{low_mnemonic_suffix}"
    immediate_fields = ("NEG", "CLAMP") if has_integer_sign_select else ()
    immediates = (
        (_MATRIX_SIGN_SELECT_IMMEDIATE, _MATRIX_CLAMP_IMMEDIATE)
        if has_integer_sign_select
        else ()
    )
    native_modifiers = (
        (
            _native_amdgpu_named_bit_list_immediate("neg_lo", 3),
            _native_amdgpu_named_flag_immediate("clamp"),
        )
        if has_integer_sign_select
        else ()
    )
    return AmdgpuDescriptorOverlay(
        descriptor_key=f"amdgpu.{mnemonic}{descriptor_key_suffix}",
        instruction_name=instruction_name,
        mnemonic=low_mnemonic,
        encoding_name="ENC_VOP3P",
        semantic_tag=semantic_tag,
        schedule_class=_SCHEDULE_SWMMAC,
        operands=(
            AmdgpuOperandOverlay(
                "VDST",
                _vgpr_result(units=accumulator_units),
                size_exception_reason=accumulator_size_exception_reason,
            ),
            AmdgpuOperandOverlay(
                "VDST",
                _vgpr_operand("acc", units=accumulator_units),
                role_exception_reason=_SMFMAC_VDST_ACCUMULATOR_REASON,
                size_exception_reason=accumulator_size_exception_reason,
            ),
            AmdgpuOperandOverlay(
                "SRC0",
                _vgpr_operand("lhs", units=lhs_units),
                size_exception_reason=lhs_size_exception_reason,
            ),
            AmdgpuOperandOverlay(
                "SRC1",
                _vgpr_operand("rhs", units=rhs_units),
                size_exception_reason=rhs_size_exception_reason,
            ),
            AmdgpuOperandOverlay("SRC2", _vgpr_operand("sparse_metadata")),
        ),
        immediate_fields=immediate_fields,
        immediates=immediates,
        fixed_encoding_fields=((op_sel_hi_field, 0x7),),
        constraints=_DESTRUCTIVE_ACCUMULATOR_CONSTRAINTS,
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
        asm_forms=_asm(
            native_assembly_mnemonic=mnemonic if low_mnemonic_suffix else None,
            results=("dst",),
            operands=("acc", "lhs", "rhs", "sparse_metadata"),
            immediates=tuple(immediate.field_name for immediate in immediates),
            named_immediates=True,
            native_assembly_values=(
                _native_result("dst"),
                _native_operand("lhs"),
                _native_operand("rhs"),
                _native_operand("sparse_metadata"),
                *native_modifiers,
            ),
        ),
    )


def _v_swmmac_16x16_overlays(
    *, wave_size: int = 32, op_sel_hi_field: str = "OP_SEL_HI"
) -> tuple[AmdgpuDescriptorOverlay, ...]:
    if wave_size not in (32, 64):
        raise ValueError(f"unsupported SWMMAC wave size {wave_size}")
    unit_divisor = wave_size // 32
    descriptor_key_suffix = ".w64" if wave_size == 64 else ""
    low_mnemonic_suffix = "_w64" if wave_size == 64 else ""

    def units(wave32_units: int) -> int:
        return max(1, wave32_units // unit_divisor)

    def size_exception_reason(wave32_units: int) -> str | None:
        return (
            _GFX12_WAVE64_MATRIX_OPERAND_SIZE_REASON
            if units(wave32_units) != wave32_units
            else None
        )

    return (
        _v_swmmac_overlay(
            instruction_name="V_SWMMAC_F32_16X16X32_F16",
            semantic_tag="matrix.swmmac.f32.16x16x32.f16",
            accumulator_units=units(8),
            lhs_units=units(4),
            rhs_units=units(8),
            descriptor_key_suffix=descriptor_key_suffix,
            low_mnemonic_suffix=low_mnemonic_suffix,
            accumulator_size_exception_reason=size_exception_reason(8),
            lhs_size_exception_reason=size_exception_reason(4),
            rhs_size_exception_reason=size_exception_reason(8),
            op_sel_hi_field=op_sel_hi_field,
        ),
        _v_swmmac_overlay(
            instruction_name="V_SWMMAC_F32_16X16X32_BF16",
            semantic_tag="matrix.swmmac.f32.16x16x32.bf16",
            accumulator_units=units(8),
            lhs_units=units(4),
            rhs_units=units(8),
            descriptor_key_suffix=descriptor_key_suffix,
            low_mnemonic_suffix=low_mnemonic_suffix,
            accumulator_size_exception_reason=size_exception_reason(8),
            lhs_size_exception_reason=size_exception_reason(4),
            rhs_size_exception_reason=size_exception_reason(8),
            op_sel_hi_field=op_sel_hi_field,
        ),
        _v_swmmac_overlay(
            instruction_name="V_SWMMAC_F16_16X16X32_F16",
            semantic_tag="matrix.swmmac.f16.16x16x32.f16",
            accumulator_units=units(4),
            lhs_units=units(4),
            rhs_units=units(8),
            descriptor_key_suffix=descriptor_key_suffix,
            low_mnemonic_suffix=low_mnemonic_suffix,
            accumulator_size_exception_reason=size_exception_reason(4),
            lhs_size_exception_reason=size_exception_reason(4),
            rhs_size_exception_reason=size_exception_reason(8),
            op_sel_hi_field=op_sel_hi_field,
        ),
        _v_swmmac_overlay(
            instruction_name="V_SWMMAC_BF16_16X16X32_BF16",
            semantic_tag="matrix.swmmac.bf16.16x16x32.bf16",
            accumulator_units=units(4),
            lhs_units=units(4),
            rhs_units=units(8),
            descriptor_key_suffix=descriptor_key_suffix,
            low_mnemonic_suffix=low_mnemonic_suffix,
            accumulator_size_exception_reason=size_exception_reason(4),
            lhs_size_exception_reason=size_exception_reason(4),
            rhs_size_exception_reason=size_exception_reason(8),
            op_sel_hi_field=op_sel_hi_field,
        ),
        _v_swmmac_overlay(
            instruction_name="V_SWMMAC_I32_16X16X32_IU8",
            semantic_tag="matrix.swmmac.i32.16x16x32.iu8",
            accumulator_units=units(8),
            lhs_units=units(2),
            rhs_units=units(4),
            descriptor_key_suffix=descriptor_key_suffix,
            low_mnemonic_suffix=low_mnemonic_suffix,
            accumulator_size_exception_reason=size_exception_reason(8),
            lhs_size_exception_reason=size_exception_reason(2),
            rhs_size_exception_reason=size_exception_reason(4),
            has_integer_sign_select=True,
            op_sel_hi_field=op_sel_hi_field,
        ),
        _v_swmmac_overlay(
            instruction_name="V_SWMMAC_I32_16X16X32_IU4",
            semantic_tag="matrix.swmmac.i32.16x16x32.iu4",
            accumulator_units=units(8),
            lhs_units=units(1),
            rhs_units=units(2),
            descriptor_key_suffix=descriptor_key_suffix,
            low_mnemonic_suffix=low_mnemonic_suffix,
            accumulator_size_exception_reason=size_exception_reason(8),
            lhs_size_exception_reason=size_exception_reason(1),
            rhs_size_exception_reason=size_exception_reason(2),
            has_integer_sign_select=True,
            op_sel_hi_field=op_sel_hi_field,
        ),
        _v_swmmac_overlay(
            instruction_name="V_SWMMAC_I32_16X16X64_IU4",
            semantic_tag="matrix.swmmac.i32.16x16x64.iu4",
            accumulator_units=units(8),
            lhs_units=units(2),
            rhs_units=units(4),
            descriptor_key_suffix=descriptor_key_suffix,
            low_mnemonic_suffix=low_mnemonic_suffix,
            accumulator_size_exception_reason=size_exception_reason(8),
            lhs_size_exception_reason=size_exception_reason(2),
            rhs_size_exception_reason=size_exception_reason(4),
            has_integer_sign_select=True,
            op_sel_hi_field=op_sel_hi_field,
        ),
        _v_swmmac_overlay(
            instruction_name="V_SWMMAC_F32_16X16X32_FP8_FP8",
            semantic_tag="matrix.swmmac.f32.16x16x32.fp8.fp8",
            accumulator_units=units(8),
            lhs_units=units(2),
            rhs_units=units(4),
            descriptor_key_suffix=descriptor_key_suffix,
            low_mnemonic_suffix=low_mnemonic_suffix,
            accumulator_size_exception_reason=size_exception_reason(8),
            lhs_size_exception_reason=size_exception_reason(2),
            rhs_size_exception_reason=size_exception_reason(4),
            op_sel_hi_field=op_sel_hi_field,
        ),
        _v_swmmac_overlay(
            instruction_name="V_SWMMAC_F32_16X16X32_FP8_BF8",
            semantic_tag="matrix.swmmac.f32.16x16x32.fp8.bf8",
            accumulator_units=units(8),
            lhs_units=units(2),
            rhs_units=units(4),
            descriptor_key_suffix=descriptor_key_suffix,
            low_mnemonic_suffix=low_mnemonic_suffix,
            accumulator_size_exception_reason=size_exception_reason(8),
            lhs_size_exception_reason=size_exception_reason(2),
            rhs_size_exception_reason=size_exception_reason(4),
            op_sel_hi_field=op_sel_hi_field,
        ),
        _v_swmmac_overlay(
            instruction_name="V_SWMMAC_F32_16X16X32_BF8_FP8",
            semantic_tag="matrix.swmmac.f32.16x16x32.bf8.fp8",
            accumulator_units=units(8),
            lhs_units=units(2),
            rhs_units=units(4),
            descriptor_key_suffix=descriptor_key_suffix,
            low_mnemonic_suffix=low_mnemonic_suffix,
            accumulator_size_exception_reason=size_exception_reason(8),
            lhs_size_exception_reason=size_exception_reason(2),
            rhs_size_exception_reason=size_exception_reason(4),
            op_sel_hi_field=op_sel_hi_field,
        ),
        _v_swmmac_overlay(
            instruction_name="V_SWMMAC_F32_16X16X32_BF8_BF8",
            semantic_tag="matrix.swmmac.f32.16x16x32.bf8.bf8",
            accumulator_units=units(8),
            lhs_units=units(2),
            rhs_units=units(4),
            descriptor_key_suffix=descriptor_key_suffix,
            low_mnemonic_suffix=low_mnemonic_suffix,
            accumulator_size_exception_reason=size_exception_reason(8),
            lhs_size_exception_reason=size_exception_reason(2),
            rhs_size_exception_reason=size_exception_reason(4),
            op_sel_hi_field=op_sel_hi_field,
        ),
    )


def _vop3p_packed_dot_fixed_fields(
    *,
    op_sel_hi_field: str = "OP_SEL_HI",
    lhs_signed: bool = False,
    rhs_signed: bool = False,
) -> tuple[tuple[str, int], ...]:
    # Packed VOP3P dot instructions use high-half source selection as the
    # canonical no-op modifier spelling. IU dot instructions additionally use
    # NEG low bits as integer signedness selectors for src0/src1.
    neg = (1 if lhs_signed else 0) | (2 if rhs_signed else 0)
    fields = [(op_sel_hi_field, 0x7)]
    if neg != 0:
        fields.append(("NEG", neg))
    return tuple(fields)


def _v_dot2_f32_packed_float_overlay(
    *,
    descriptor_key: str,
    instruction_name: str,
    mnemonic: str,
    semantic_tag: str,
    op_sel_hi_field: str = "OP_SEL_HI",
) -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key=descriptor_key,
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        encoding_name="ENC_VOP3P",
        semantic_tag=semantic_tag,
        schedule_class=_SCHEDULE_PACKED_DOT,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            AmdgpuOperandOverlay("SRC0", _vgpr_operand("lhs")),
            AmdgpuOperandOverlay("SRC1", _vgpr_operand("rhs")),
            AmdgpuOperandOverlay("SRC2", _vgpr_const_operand("acc")),
        ),
        fixed_encoding_fields=_vop3p_packed_dot_fixed_fields(
            op_sel_hi_field=op_sel_hi_field
        ),
        constraints=(
            Constraint(ConstraintKind.TIED, 0, 3),
            Constraint(ConstraintKind.DESTRUCTIVE, 0, 3),
            Constraint(ConstraintKind.COMMUTABLE, 1, 2),
        ),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_dot2_f32_f16_overlay(
    *, op_sel_hi_field: str = "OP_SEL_HI"
) -> AmdgpuDescriptorOverlay:
    return _v_dot2_f32_packed_float_overlay(
        descriptor_key="amdgpu.v_dot2_f32_f16",
        instruction_name="V_DOT2_F32_F16",
        mnemonic="v_dot2_f32_f16",
        semantic_tag="dot.f16f16.f32x1",
        op_sel_hi_field=op_sel_hi_field,
    )


def _v_dot2_f32_bf16_overlay(
    *, op_sel_hi_field: str = "OP_SEL_HI"
) -> AmdgpuDescriptorOverlay:
    return _v_dot2_f32_packed_float_overlay(
        descriptor_key="amdgpu.v_dot2_f32_bf16",
        instruction_name="V_DOT2_F32_BF16",
        mnemonic="v_dot2_f32_bf16",
        semantic_tag="dot.bf16bf16.f32x1",
        op_sel_hi_field=op_sel_hi_field,
    )


def _v_dot4_i32_i8_overlay(
    *, op_sel_hi_field: str = "OP_SEL_HI", signedness_modifiers: bool
) -> AmdgpuDescriptorOverlay:
    instruction_name = "V_DOT4_I32_IU8" if signedness_modifiers else "V_DOT4_I32_I8"
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.v_dot4_i32_i8",
        instruction_name=instruction_name,
        mnemonic="v_dot4_i32_i8",
        encoding_name="ENC_VOP3P",
        semantic_tag="dot.s8s8.i32x1",
        schedule_class=_SCHEDULE_PACKED_DOT,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            AmdgpuOperandOverlay("SRC0", _vgpr_operand("lhs")),
            AmdgpuOperandOverlay("SRC1", _vgpr_operand("rhs")),
            AmdgpuOperandOverlay("SRC2", _vgpr_const_operand("acc")),
        ),
        fixed_encoding_fields=_vop3p_packed_dot_fixed_fields(
            op_sel_hi_field=op_sel_hi_field,
            lhs_signed=signedness_modifiers,
            rhs_signed=signedness_modifiers,
        ),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_dot4_i32_iu8_overlay(
    *, op_sel_hi_field: str = "OP_SEL_HI", lhs_signed: bool, rhs_signed: bool
) -> AmdgpuDescriptorOverlay:
    lhs_tag = "s8" if lhs_signed else "u8"
    rhs_tag = "s8" if rhs_signed else "u8"
    return AmdgpuDescriptorOverlay(
        descriptor_key=f"amdgpu.v_dot4_i32_iu8.{lhs_tag}{rhs_tag}",
        instruction_name="V_DOT4_I32_IU8",
        mnemonic="v_dot4_i32_iu8",
        encoding_name="ENC_VOP3P",
        semantic_tag=f"dot.{lhs_tag}{rhs_tag}.i32x1",
        schedule_class=_SCHEDULE_PACKED_DOT,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            AmdgpuOperandOverlay("SRC0", _vgpr_operand("lhs")),
            AmdgpuOperandOverlay("SRC1", _vgpr_operand("rhs")),
            AmdgpuOperandOverlay("SRC2", _vgpr_const_operand("acc")),
        ),
        fixed_encoding_fields=_vop3p_packed_dot_fixed_fields(
            op_sel_hi_field=op_sel_hi_field,
            lhs_signed=lhs_signed,
            rhs_signed=rhs_signed,
        ),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
        asm_forms=_asm(
            mnemonic=f"v_dot4_i32_iu8_{lhs_tag}{rhs_tag}",
            native_assembly_mnemonic="v_dot4_i32_iu8",
            results=("dst",),
            operands=("lhs", "rhs", "acc"),
        ),
    )


def _v_dot4_u32_u8_overlay(
    *,
    op_sel_hi_field: str = "OP_SEL_HI",
) -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.v_dot4_u32_u8",
        instruction_name="V_DOT4_U32_U8",
        mnemonic="v_dot4_u32_u8",
        encoding_name="ENC_VOP3P",
        semantic_tag="dot.u8u8.i32x1",
        schedule_class=_SCHEDULE_PACKED_DOT,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            AmdgpuOperandOverlay("SRC0", _vgpr_operand("lhs")),
            AmdgpuOperandOverlay("SRC1", _vgpr_operand("rhs")),
            AmdgpuOperandOverlay("SRC2", _vgpr_const_operand("acc")),
        ),
        fixed_encoding_fields=_vop3p_packed_dot_fixed_fields(
            op_sel_hi_field=op_sel_hi_field
        ),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_dot4_f32_packed8_overlay(
    *,
    lhs_type: str,
    rhs_type: str,
    op_sel_hi_field: str = "OP_SEL_HI",
) -> AmdgpuDescriptorOverlay:
    lhs_type_upper = lhs_type.upper()
    rhs_type_upper = rhs_type.upper()
    return AmdgpuDescriptorOverlay(
        descriptor_key=f"amdgpu.v_dot4_f32_{lhs_type}_{rhs_type}",
        instruction_name=f"V_DOT4_F32_{lhs_type_upper}_{rhs_type_upper}",
        mnemonic=f"v_dot4_f32_{lhs_type}_{rhs_type}",
        encoding_name="ENC_VOP3P",
        semantic_tag=f"dot.{lhs_type}{rhs_type}.f32x1",
        schedule_class=_SCHEDULE_PACKED_DOT,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            AmdgpuOperandOverlay("SRC0", _vgpr_operand("lhs")),
            AmdgpuOperandOverlay("SRC1", _vgpr_operand("rhs")),
            AmdgpuOperandOverlay("SRC2", _vgpr_const_operand("acc")),
        ),
        fixed_encoding_fields=_vop3p_packed_dot_fixed_fields(
            op_sel_hi_field=op_sel_hi_field
        ),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_dot8_i32_i4_overlay(
    *, op_sel_hi_field: str = "OP_SEL_HI", signedness_modifiers: bool
) -> AmdgpuDescriptorOverlay:
    instruction_name = "V_DOT8_I32_IU4" if signedness_modifiers else "V_DOT8_I32_I4"
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.v_dot8_i32_i4",
        instruction_name=instruction_name,
        mnemonic="v_dot8_i32_i4",
        encoding_name="ENC_VOP3P",
        semantic_tag="dot.s4s4.i32x1",
        schedule_class=_SCHEDULE_PACKED_DOT,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            AmdgpuOperandOverlay("SRC0", _vgpr_operand("lhs")),
            AmdgpuOperandOverlay("SRC1", _vgpr_operand("rhs")),
            AmdgpuOperandOverlay("SRC2", _vgpr_const_operand("acc")),
        ),
        fixed_encoding_fields=_vop3p_packed_dot_fixed_fields(
            op_sel_hi_field=op_sel_hi_field,
            lhs_signed=signedness_modifiers,
            rhs_signed=signedness_modifiers,
        ),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_dot8_i32_iu4_overlay(
    *, op_sel_hi_field: str = "OP_SEL_HI", lhs_signed: bool, rhs_signed: bool
) -> AmdgpuDescriptorOverlay:
    lhs_tag = "s4" if lhs_signed else "u4"
    rhs_tag = "s4" if rhs_signed else "u4"
    return AmdgpuDescriptorOverlay(
        descriptor_key=f"amdgpu.v_dot8_i32_iu4.{lhs_tag}{rhs_tag}",
        instruction_name="V_DOT8_I32_IU4",
        mnemonic="v_dot8_i32_iu4",
        encoding_name="ENC_VOP3P",
        semantic_tag=f"dot.{lhs_tag}{rhs_tag}.i32x1",
        schedule_class=_SCHEDULE_PACKED_DOT,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            AmdgpuOperandOverlay("SRC0", _vgpr_operand("lhs")),
            AmdgpuOperandOverlay("SRC1", _vgpr_operand("rhs")),
            AmdgpuOperandOverlay("SRC2", _vgpr_const_operand("acc")),
        ),
        fixed_encoding_fields=_vop3p_packed_dot_fixed_fields(
            op_sel_hi_field=op_sel_hi_field,
            lhs_signed=lhs_signed,
            rhs_signed=rhs_signed,
        ),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
        asm_forms=_asm(
            mnemonic=f"v_dot8_i32_iu4_{lhs_tag}{rhs_tag}",
            native_assembly_mnemonic="v_dot8_i32_iu4",
            results=("dst",),
            operands=("lhs", "rhs", "acc"),
        ),
    )


def _v_dot8_u32_u4_overlay(
    *,
    op_sel_hi_field: str = "OP_SEL_HI",
) -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.v_dot8_u32_u4",
        instruction_name="V_DOT8_U32_U4",
        mnemonic="v_dot8_u32_u4",
        encoding_name="ENC_VOP3P",
        semantic_tag="dot.u4u4.i32x1",
        schedule_class=_SCHEDULE_PACKED_DOT,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            AmdgpuOperandOverlay("SRC0", _vgpr_operand("lhs")),
            AmdgpuOperandOverlay("SRC1", _vgpr_operand("rhs")),
            AmdgpuOperandOverlay("SRC2", _vgpr_const_operand("acc")),
        ),
        fixed_encoding_fields=_vop3p_packed_dot_fixed_fields(
            op_sel_hi_field=op_sel_hi_field
        ),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


__all__ = (
    "_GFX12_WAVE64_MATRIX_OPERAND_SIZE_REASON",
    "_WMMA_GFX11_WAVE64_ACCUMULATOR_SIZE_REASON",
    "_cdna3_mfma_overlays",
    "_cdna3_smfmac_overlays",
    "_cdna4_mfma_overlays",
    "_cdna4_smfmac_overlays",
    "_v_swmmac_16x16_overlays",
    "_v_dot2_f32_bf16_overlay",
    "_v_dot2_f32_f16_overlay",
    "_v_dot2_f32_packed_float_overlay",
    "_v_dot4_f32_packed8_overlay",
    "_v_dot4_i32_i8_overlay",
    "_v_dot4_i32_iu8_overlay",
    "_v_dot4_u32_u8_overlay",
    "_v_dot8_i32_i4_overlay",
    "_v_dot8_i32_iu4_overlay",
    "_v_dot8_u32_u4_overlay",
    "_v_mfma_f32_16x16x16_bf16_overlay",
    "_v_mfma_f32_16x16x16_f16_overlay",
    "_v_mfma_f32_16x16x4_f32_overlay",
    "_v_smfmac_f32_16x16x32_bf16_overlay",
    "_v_smfmac_f32_16x16x32_f16_overlay",
    "_v_smfmac_f32_32x32x16_bf16_overlay",
    "_v_smfmac_f32_32x32x16_f16_overlay",
    "_v_smfmac_f32_overlay",
    "_v_wmma_16x16x16_overlay",
    "_v_wmma_bf16_16x16x16_bf16_overlay",
    "_v_wmma_f16_16x16x16_f16_overlay",
    "_v_wmma_f32_16x16x16_bf16_overlay",
    "_v_wmma_f32_16x16x16_f16_overlay",
    "_v_wmma_f32_16x16x16_packed8_overlay",
    "_v_wmma_i32_16x16x16_iu4_overlay",
    "_v_wmma_i32_16x16x16_iu8_overlay",
    "_v_wmma_i32_16x16x16_overlay",
    "_v_wmma_i32_16x16x32_iu4_overlay",
    "_vop3p_packed_dot_fixed_fields",
    "_with_zero_accumulator_form",
)
