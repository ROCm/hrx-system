# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


# ruff: noqa: F403, F405

"""Scalar, vector, conversion, compare, and select descriptor overlays."""

from __future__ import annotations

from .common import *

_DPP_CTRL_IMMEDIATE = Immediate(
    "dpp_ctrl",
    ImmediateKind.UNSIGNED,
    bit_width=9,
    unsigned_max=0x1FF,
)

_SDWA_DST_UNUSED_IMMEDIATE = Immediate(
    "dst_unused",
    ImmediateKind.UNSIGNED,
    bit_width=2,
    unsigned_max=2,
)

_SDWA_SOURCE_SEXT_IMMEDIATE = Immediate(
    "src0_sext",
    ImmediateKind.UNSIGNED,
    bit_width=1,
    unsigned_max=1,
)

_SOPK_I16_IMMEDIATE = Immediate(
    "imm16",
    ImmediateKind.SIGNED,
    bit_width=16,
    signed_min=-(2**15),
    unsigned_max=(2**15) - 1,
)

_SYMBOL_BYTE_OFFSET_IMMEDIATE = Immediate(
    "byte_offset",
    ImmediateKind.UNSIGNED,
    flags=(ImmediateFlag.DEFAULT_VALUE,),
    bit_width=64,
    unsigned_max=(2**63) - 1,
    default_value=0,
)


def _symbol_rel32_immediate(field_name: str = "symbol") -> Immediate:
    return Immediate(
        field_name,
        ImmediateKind.ORDINAL,
        flags=(ImmediateFlag.SYMBOLIC, ImmediateFlag.RELATIVE),
        bit_width=32,
        unsigned_max=(2**32) - 1,
        encoding_field_id=amdgpu_encoding_field_id("LITERAL"),
    )


def _sdwa_selector_immediate(field_name: str) -> Immediate:
    return Immediate(
        field_name,
        ImmediateKind.UNSIGNED,
        bit_width=3,
        unsigned_max=6,
    )


def _s_add_u32_overlay() -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.s_add_u32",
        instruction_name="S_ADD_U32",
        mnemonic="s_add_u32",
        encoding_name="ENC_SOP2",
        semantic_tag="integer.add.u32",
        schedule_class=_SCHEDULE_SALU,
        operands=(
            AmdgpuOperandOverlay("SDST", _sgpr_result()),
            AmdgpuOperandOverlay("SSRC0", _sgpr_operand("lhs")),
            AmdgpuOperandOverlay("SSRC1", _sgpr_operand("rhs")),
        ),
        implicit_operands=(_SCC_CLOBBER_OUTPUT,),
        operand_forms=(
            _literal_operand_form(
                replacement_descriptor="amdgpu.s_addk_i32",
                source_operand="rhs",
                immediate_field="imm16",
            ),
            _literal_operand_form(
                replacement_descriptor="amdgpu.s_add_u32.rhs_inline",
                source_operand="rhs",
            ),
        ),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _s_add_u32_rhs_inline_overlay() -> AmdgpuDescriptorOverlay:
    return _s_binary_u32_rhs_inline_overlay(
        descriptor_key="amdgpu.s_add_u32.rhs_inline",
        instruction_name="S_ADD_U32",
        mnemonic="s_add_u32",
        semantic_tag="integer.add.u32",
    )


def _s_addk_i32_overlay(
    *,
    instruction_name: str = "S_ADDK_I32",
    mnemonic: str = "s_addk_i32",
) -> AmdgpuDescriptorOverlay:
    return _s_tied_sopk_i32_overlay(
        descriptor_key="amdgpu.s_addk_i32",
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        semantic_tag="integer.add.u32",
        implicit_operands=(_SCC_CLOBBER_OUTPUT,),
    )


def _s_add_u32_rhs_symbol_rel32_lo_overlay() -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.s_add_u32.rhs_symbol_rel32_lo",
        instruction_name="S_ADD_U32",
        mnemonic="s_add_u32",
        encoding_name="ENC_SOP2",
        encoding_format_id=AMDGPU_ENCODING_FORMAT_SOP2_LITERAL,
        semantic_tag="address.add.pc_relative.lo.u32",
        schedule_class=_SCHEDULE_SALU,
        operands=(
            AmdgpuOperandOverlay("SDST", _sgpr_result()),
            AmdgpuOperandOverlay("SSRC0", _sgpr_operand("lhs")),
        ),
        implicit_operands=(_SCC_CLOBBER_OUTPUT,),
        immediates=(
            _symbol_rel32_immediate(),
            _SYMBOL_BYTE_OFFSET_IMMEDIATE,
        ),
        fixed_encoding_fields=(("SSRC1", _predefined("SRC_LITERAL", "OPR_SSRC")),),
        asm_forms=_asm(
            mnemonic="s_add_u32_rhs_symbol_rel32_lo",
            results=("dst",),
            operands=("lhs",),
            immediates=("symbol", "byte_offset"),
            named_immediates=True,
        ),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _s_addc_u32_overlay() -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.s_addc_u32",
        instruction_name="S_ADDC_U32",
        mnemonic="s_addc_u32",
        encoding_name="ENC_SOP2",
        semantic_tag="integer.add.carry_in_out.u32",
        schedule_class=_SCHEDULE_SALU,
        operands=(
            AmdgpuOperandOverlay("SDST", _sgpr_result("sum")),
            AmdgpuOperandOverlay("SSRC0", _sgpr_operand("lhs")),
            AmdgpuOperandOverlay("SSRC1", _sgpr_operand("rhs")),
        ),
        implicit_operands=(
            _scc_output(_scc_clobber("carry")),
            _scc_input(_scc_state_read("carry_in")),
        ),
        asm_forms=_asm(results=("sum",), operands=("lhs", "rhs")),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _s_addc_u32_rhs_symbol_rel32_hi_overlay() -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.s_addc_u32.rhs_symbol_rel32_hi",
        instruction_name="S_ADDC_U32",
        mnemonic="s_addc_u32",
        encoding_name="ENC_SOP2",
        encoding_format_id=AMDGPU_ENCODING_FORMAT_SOP2_LITERAL,
        semantic_tag="address.add.pc_relative.hi.u32",
        schedule_class=_SCHEDULE_SALU,
        operands=(
            AmdgpuOperandOverlay("SDST", _sgpr_result("sum")),
            AmdgpuOperandOverlay("SSRC0", _sgpr_operand("lhs")),
        ),
        implicit_operands=(
            _scc_output(_scc_clobber("carry")),
            _scc_input(_scc_state_read("carry_in")),
        ),
        immediates=(
            _symbol_rel32_immediate(),
            _SYMBOL_BYTE_OFFSET_IMMEDIATE,
        ),
        fixed_encoding_fields=(("SSRC1", _predefined("SRC_LITERAL", "OPR_SSRC")),),
        asm_forms=_asm(
            mnemonic="s_addc_u32_rhs_symbol_rel32_hi",
            results=("sum",),
            operands=("lhs",),
            immediates=("symbol", "byte_offset"),
            named_immediates=True,
        ),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _s_sub_u32_overlay() -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.s_sub_u32",
        instruction_name="S_SUB_U32",
        mnemonic="s_sub_u32",
        encoding_name="ENC_SOP2",
        semantic_tag="integer.sub.u32",
        schedule_class=_SCHEDULE_SALU,
        operands=(
            AmdgpuOperandOverlay("SDST", _sgpr_result()),
            AmdgpuOperandOverlay("SSRC0", _sgpr_operand("lhs")),
            AmdgpuOperandOverlay("SSRC1", _sgpr_operand("rhs")),
        ),
        implicit_operands=(_SCC_CLOBBER_OUTPUT,),
        operand_forms=(
            _literal_operand_form(
                replacement_descriptor="amdgpu.s_sub_u32.rhs_inline",
                source_operand="rhs",
            ),
        ),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _s_sub_u32_rhs_inline_overlay() -> AmdgpuDescriptorOverlay:
    return _s_binary_u32_rhs_inline_overlay(
        descriptor_key="amdgpu.s_sub_u32.rhs_inline",
        instruction_name="S_SUB_U32",
        mnemonic="s_sub_u32",
        semantic_tag="integer.sub.u32",
    )


def _s_mul_i32_overlay() -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.s_mul_i32",
        instruction_name="S_MUL_I32",
        mnemonic="s_mul_i32",
        encoding_name="ENC_SOP2",
        semantic_tag="integer.mul.lo.i32",
        schedule_class=_SCHEDULE_SALU,
        operands=(
            AmdgpuOperandOverlay("SDST", _sgpr_result()),
            AmdgpuOperandOverlay("SSRC0", _sgpr_operand("lhs")),
            AmdgpuOperandOverlay("SSRC1", _sgpr_operand("rhs")),
        ),
        operand_forms=(
            _literal_operand_form(
                replacement_descriptor="amdgpu.s_mulk_i32",
                source_operand="rhs",
                immediate_field="imm16",
            ),
            _literal_operand_form(
                replacement_descriptor="amdgpu.s_mul_i32.rhs_inline",
                source_operand="rhs",
            ),
        ),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _s_mul_i32_rhs_inline_overlay() -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.s_mul_i32.rhs_inline",
        instruction_name="S_MUL_I32",
        mnemonic="s_mul_i32",
        encoding_name="ENC_SOP2",
        semantic_tag="integer.mul.lo.i32",
        schedule_class=_SCHEDULE_SALU,
        operands=(
            AmdgpuOperandOverlay("SDST", _sgpr_result()),
            AmdgpuOperandOverlay("SSRC0", _sgpr_operand("lhs")),
        ),
        asm_forms=_asm(
            mnemonic="s_mul_i32_rhs_inline",
            results=("dst",),
            operands=("lhs",),
            immediates=("imm32",),
        ),
        immediate_fields=("SSRC1",),
        immediates=(_SOURCE_INLINE_U32_IMMEDIATE,),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _s_mulk_i32_overlay() -> AmdgpuDescriptorOverlay:
    return _s_tied_sopk_i32_overlay(
        descriptor_key="amdgpu.s_mulk_i32",
        instruction_name="S_MULK_I32",
        mnemonic="s_mulk_i32",
        semantic_tag="integer.mul.lo.i32",
    )


def _s_tied_sopk_i32_overlay(
    *,
    descriptor_key: str,
    instruction_name: str,
    mnemonic: str,
    semantic_tag: str,
    implicit_operands: tuple[AmdgpuImplicitOperandOverlay, ...] = (),
) -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key=descriptor_key,
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        encoding_name="ENC_SOPK",
        semantic_tag=semantic_tag,
        schedule_class=_SCHEDULE_SALU,
        operands=(
            AmdgpuOperandOverlay("SDST", _sgpr_result()),
            AmdgpuOperandOverlay(
                "SDST",
                _sgpr_operand("lhs"),
                role_exception_reason=(
                    "the encoded scalar destination register is also the tied lhs input"
                ),
            ),
        ),
        implicit_operands=implicit_operands,
        immediate_fields=("SIMM16",),
        immediates=(_SOPK_I16_IMMEDIATE,),
        asm_forms=_asm(
            mnemonic=mnemonic,
            results=("dst",),
            operands=("lhs",),
            immediates=("imm16",),
        ),
        constraints=(
            Constraint(ConstraintKind.TIED, 0, 1),
            Constraint(ConstraintKind.DESTRUCTIVE, 0, 1),
        ),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _s_mul_hi_u32_overlay() -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.s_mul_hi_u32",
        instruction_name="S_MUL_HI_U32",
        mnemonic="s_mul_hi_u32",
        encoding_name="ENC_SOP2",
        semantic_tag="integer.mul.hi.u32",
        schedule_class=_SCHEDULE_SALU,
        operands=(
            AmdgpuOperandOverlay("SDST", _sgpr_result()),
            AmdgpuOperandOverlay("SSRC0", _sgpr_operand("lhs")),
            AmdgpuOperandOverlay("SSRC1", _sgpr_operand("rhs")),
        ),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _s_binary_u32_overlay(
    *,
    descriptor_key: str,
    instruction_name: str,
    mnemonic: str,
    semantic_tag: str,
    rhs_inline_descriptor_key: str | None = None,
) -> AmdgpuDescriptorOverlay:
    operand_forms: tuple[OperandForm, ...] = ()
    if rhs_inline_descriptor_key is not None:
        operand_forms = (
            _literal_operand_form(
                replacement_descriptor=rhs_inline_descriptor_key,
                source_operand="rhs",
            ),
        )
    return AmdgpuDescriptorOverlay(
        descriptor_key=descriptor_key,
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        encoding_name="ENC_SOP2",
        semantic_tag=semantic_tag,
        schedule_class=_SCHEDULE_SALU,
        operands=(
            AmdgpuOperandOverlay("SDST", _sgpr_result()),
            AmdgpuOperandOverlay("SSRC0", _sgpr_operand("lhs")),
            AmdgpuOperandOverlay("SSRC1", _sgpr_operand("rhs")),
        ),
        implicit_operands=(_SCC_CLOBBER_OUTPUT,),
        operand_forms=operand_forms,
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _s_binary_u32_rhs_inline_overlay(
    *,
    descriptor_key: str,
    instruction_name: str,
    mnemonic: str,
    semantic_tag: str,
) -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key=descriptor_key,
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        encoding_name="ENC_SOP2",
        semantic_tag=semantic_tag,
        schedule_class=_SCHEDULE_SALU,
        operands=(
            AmdgpuOperandOverlay("SDST", _sgpr_result()),
            AmdgpuOperandOverlay("SSRC0", _sgpr_operand("lhs")),
        ),
        implicit_operands=(_SCC_CLOBBER_OUTPUT,),
        asm_forms=_asm(
            mnemonic=f"{mnemonic}_rhs_inline",
            results=("dst",),
            operands=("lhs",),
            immediates=("imm32",),
        ),
        immediate_fields=("SSRC1",),
        immediates=(_SOURCE_INLINE_U32_IMMEDIATE,),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _s_binary_u64_overlay(
    *,
    descriptor_key: str,
    instruction_name: str,
    mnemonic: str,
    semantic_tag: str,
) -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key=descriptor_key,
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        encoding_name="ENC_SOP2",
        semantic_tag=semantic_tag,
        schedule_class=_SCHEDULE_SALU,
        operands=(
            AmdgpuOperandOverlay("SDST", _sgpr_result(units=2)),
            AmdgpuOperandOverlay("SSRC0", _sgpr_operand("lhs", units=2)),
            AmdgpuOperandOverlay("SSRC1", _sgpr_operand("rhs", units=2)),
        ),
        implicit_operands=(_SCC_CLOBBER_OUTPUT,),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _s_min_i32_overlay() -> AmdgpuDescriptorOverlay:
    return _s_binary_u32_overlay(
        descriptor_key="amdgpu.s_min_i32",
        instruction_name="S_MIN_I32",
        mnemonic="s_min_i32",
        semantic_tag="integer.min.i32",
    )


def _s_max_i32_overlay() -> AmdgpuDescriptorOverlay:
    return _s_binary_u32_overlay(
        descriptor_key="amdgpu.s_max_i32",
        instruction_name="S_MAX_I32",
        mnemonic="s_max_i32",
        semantic_tag="integer.max.i32",
    )


def _s_min_u32_overlay() -> AmdgpuDescriptorOverlay:
    return _s_binary_u32_overlay(
        descriptor_key="amdgpu.s_min_u32",
        instruction_name="S_MIN_U32",
        mnemonic="s_min_u32",
        semantic_tag="integer.min.u32",
    )


def _s_max_u32_overlay() -> AmdgpuDescriptorOverlay:
    return _s_binary_u32_overlay(
        descriptor_key="amdgpu.s_max_u32",
        instruction_name="S_MAX_U32",
        mnemonic="s_max_u32",
        semantic_tag="integer.max.u32",
    )


def _s_cselect_b32_overlay() -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.s_cselect_b32",
        instruction_name="S_CSELECT_B32",
        mnemonic="s_cselect_b32",
        encoding_name="ENC_SOP2",
        semantic_tag="control.select.b32",
        schedule_class=_SCHEDULE_SALU,
        operands=(
            AmdgpuOperandOverlay("SDST", _sgpr_result()),
            AmdgpuOperandOverlay("SSRC0", _sgpr_operand("true_value")),
            AmdgpuOperandOverlay("SSRC1", _sgpr_operand("false_value")),
        ),
        implicit_operands=(_scc_input(_scc_predicate("condition")),),
        asm_forms=_asm(
            results=("dst",),
            operands=("true_value", "false_value", "condition"),
        ),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _s_shift_u64_overlay(
    *,
    descriptor_key: str,
    instruction_name: str,
    mnemonic: str,
    semantic_tag: str,
) -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key=descriptor_key,
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        encoding_name="ENC_SOP2",
        semantic_tag=semantic_tag,
        schedule_class=_SCHEDULE_SALU,
        operands=(
            AmdgpuOperandOverlay("SDST", _sgpr_result(units=2)),
            AmdgpuOperandOverlay("SSRC0", _sgpr_operand("value", units=2)),
            AmdgpuOperandOverlay("SSRC1", _sgpr_operand("shift")),
        ),
        implicit_operands=(_SCC_CLOBBER_OUTPUT,),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _s_cmp_i32_overlay(
    *,
    descriptor_key: str,
    instruction_name: str,
    mnemonic: str,
    semantic_tag: str,
) -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key=descriptor_key,
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        encoding_name="ENC_SOPC",
        semantic_tag=semantic_tag,
        schedule_class=_SCHEDULE_SALU,
        operands=(
            AmdgpuOperandOverlay("SSRC0", _sgpr_operand("lhs")),
            AmdgpuOperandOverlay("SSRC1", _sgpr_operand("rhs")),
        ),
        implicit_operands=(_scc_output(_scc_result()),),
        asm_forms=_asm(results=("scc",), operands=("lhs", "rhs")),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _s_cmp_u64_overlay(
    *,
    descriptor_key: str,
    instruction_name: str,
    mnemonic: str,
    semantic_tag: str,
) -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key=descriptor_key,
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        encoding_name="ENC_SOPC",
        semantic_tag=semantic_tag,
        schedule_class=_SCHEDULE_SALU,
        operands=(
            AmdgpuOperandOverlay("SSRC0", _sgpr_operand("lhs", units=2)),
            AmdgpuOperandOverlay("SSRC1", _sgpr_operand("rhs", units=2)),
        ),
        implicit_operands=(_scc_output(_scc_result()),),
        asm_forms=_asm(results=("scc",), operands=("lhs", "rhs")),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_add_u32_overlay(instruction_name: str) -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.v_add_u32",
        instruction_name=instruction_name,
        mnemonic="v_add_u32",
        encoding_name="ENC_VOP2",
        semantic_tag="integer.add.u32",
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            AmdgpuOperandOverlay("SRC0", _sgpr_vgpr_operand("lhs")),
            AmdgpuOperandOverlay("VSRC1", _vgpr_operand("rhs")),
        ),
        operand_forms=(
            _literal_operand_form(
                replacement_descriptor="amdgpu.v_add_u32.src0_inline",
                source_operand="lhs",
            ),
            _literal_operand_form(
                replacement_descriptor="amdgpu.v_add_u32.lit",
                source_operand="lhs",
            ),
        ),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_add_u32_src0_inline_overlay(instruction_name: str) -> AmdgpuDescriptorOverlay:
    return _v_binary_src0_inline_overlay(
        descriptor_key="amdgpu.v_add_u32.src0_inline",
        instruction_name=instruction_name,
        mnemonic="v_add_u32",
        semantic_tag="integer.add.u32",
    )


def _v_binary_literal_overlay(
    *,
    descriptor_key: str,
    instruction_name: str,
    mnemonic: str,
    semantic_tag: str,
    rhs_name: str = "rhs",
) -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key=descriptor_key,
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        encoding_name="VOP2_INST_LITERAL",
        encoding_condition="has_lit",
        semantic_tag=semantic_tag,
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            AmdgpuOperandOverlay("VSRC1", _vgpr_operand(rhs_name)),
        ),
        asm_forms=_asm(
            mnemonic=f"{mnemonic}_lit",
            results=("dst",),
            operands=(rhs_name,),
            immediates=("imm32",),
        ),
        immediate_fields=("LITERAL",),
        immediates=(_U32_IMMEDIATE,),
        fixed_encoding_fields=(("SRC0", _predefined("SRC_LITERAL", "OPR_SRC")),),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_binary_src0_inline_overlay(
    *,
    descriptor_key: str,
    instruction_name: str,
    mnemonic: str,
    semantic_tag: str,
    rhs_name: str = "rhs",
) -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key=descriptor_key,
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        encoding_name="ENC_VOP2",
        semantic_tag=semantic_tag,
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            AmdgpuOperandOverlay("VSRC1", _vgpr_operand(rhs_name)),
        ),
        asm_forms=_asm(
            mnemonic=f"{mnemonic}_src0_inline",
            results=("dst",),
            operands=(rhs_name,),
            immediates=("imm32",),
        ),
        immediate_fields=("SRC0",),
        immediates=(_SOURCE_INLINE_U32_IMMEDIATE,),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_binary_src0_inline_f32_overlay(
    *,
    descriptor_key: str,
    instruction_name: str,
    mnemonic: str,
    semantic_tag: str,
    rhs_name: str = "rhs",
) -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key=descriptor_key,
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        encoding_name="ENC_VOP2",
        semantic_tag=semantic_tag,
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            AmdgpuOperandOverlay("VSRC1", _vgpr_operand(rhs_name)),
        ),
        asm_forms=_asm(
            mnemonic=f"{mnemonic}_src0_inline",
            results=("dst",),
            operands=(rhs_name,),
            immediates=("imm32",),
        ),
        immediate_fields=("SRC0",),
        immediates=(_SOURCE_INLINE_F32_IMMEDIATE,),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_binary_f32_operand_forms(
    descriptor: AmdgpuDescriptorOverlay,
    *,
    source_operand: str = "lhs",
) -> tuple[OperandForm, ...]:
    descriptor_key = descriptor.descriptor_key
    return (
        _literal_operand_form(
            replacement_descriptor=f"{descriptor_key}.src0_inline",
            source_operand=source_operand,
        ),
        _literal_operand_form(
            replacement_descriptor=f"{descriptor_key}.lit",
            source_operand=source_operand,
        ),
    )


def _v_add_u32_literal_overlay(instruction_name: str) -> AmdgpuDescriptorOverlay:
    return _v_binary_literal_overlay(
        descriptor_key="amdgpu.v_add_u32.lit",
        instruction_name=instruction_name,
        mnemonic="v_add_u32",
        semantic_tag="integer.add.u32",
    )


def _v_add_co_u32_overlay() -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.v_add_co_u32",
        instruction_name="V_ADD_CO_U32",
        mnemonic="v_add_co_u32",
        encoding_name="VOP3_SDST_ENC",
        semantic_tag="integer.add.carry_out.u32",
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result("sum")),
            AmdgpuOperandOverlay("SDST", _sgpr_result("carry", units=2)),
            AmdgpuOperandOverlay("SRC0", _sgpr_vgpr_operand("lhs")),
            AmdgpuOperandOverlay("SRC1", _sgpr_vgpr_operand("rhs")),
        ),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_add_co_ci_u32_overlay(
    *, instruction_name: str = "V_ADD_CO_CI_U32", mnemonic: str = "v_add_co_ci_u32"
) -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.v_add_co_ci_u32",
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        encoding_name="VOP3_SDST_ENC",
        semantic_tag="integer.add.carry_in_out.u32",
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result("sum")),
            AmdgpuOperandOverlay("SDST", _sgpr_result("carry", units=2)),
            AmdgpuOperandOverlay("SRC0", _sgpr_vgpr_operand("lhs")),
            AmdgpuOperandOverlay("SRC1", _sgpr_vgpr_operand("rhs")),
            AmdgpuOperandOverlay("SRC2", _sgpr_operand("carry_in", units=2)),
        ),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_sub_co_u32_overlay() -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.v_sub_co_u32",
        instruction_name="V_SUB_CO_U32",
        mnemonic="v_sub_co_u32",
        encoding_name="VOP3_SDST_ENC",
        semantic_tag="integer.sub.borrow_out.u32",
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result("difference")),
            AmdgpuOperandOverlay("SDST", _sgpr_result("borrow", units=2)),
            AmdgpuOperandOverlay("SRC0", _sgpr_vgpr_operand("lhs")),
            AmdgpuOperandOverlay("SRC1", _sgpr_vgpr_operand("rhs")),
        ),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_sub_co_ci_u32_overlay(
    *, instruction_name: str = "V_SUB_CO_CI_U32", mnemonic: str = "v_sub_co_ci_u32"
) -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.v_sub_co_ci_u32",
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        encoding_name="VOP3_SDST_ENC",
        semantic_tag="integer.sub.borrow_in_out.u32",
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result("difference")),
            AmdgpuOperandOverlay("SDST", _sgpr_result("borrow", units=2)),
            AmdgpuOperandOverlay("SRC0", _sgpr_vgpr_operand("lhs")),
            AmdgpuOperandOverlay("SRC1", _sgpr_vgpr_operand("rhs")),
            AmdgpuOperandOverlay("SRC2", _sgpr_operand("borrow_in", units=2)),
        ),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_sub_u32_overlay(instruction_name: str, mnemonic: str) -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.v_sub_u32",
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        encoding_name="ENC_VOP2",
        semantic_tag="integer.sub.u32",
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            AmdgpuOperandOverlay("SRC0", _sgpr_vgpr_operand("lhs")),
            AmdgpuOperandOverlay("VSRC1", _vgpr_operand("rhs")),
        ),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_binary_u32_overlay(
    *,
    descriptor_key: str,
    instruction_name: str,
    mnemonic: str,
    semantic_tag: str,
    lhs_name: str = "lhs",
    rhs_name: str = "rhs",
    src0_inline_descriptor_key: str | None = None,
    literal_descriptor_key: str | None = None,
) -> AmdgpuDescriptorOverlay:
    operand_forms = []
    if src0_inline_descriptor_key is not None:
        operand_forms.append(
            _literal_operand_form(
                replacement_descriptor=src0_inline_descriptor_key,
                source_operand=lhs_name,
            )
        )
    if literal_descriptor_key is not None:
        operand_forms.append(
            _literal_operand_form(
                replacement_descriptor=literal_descriptor_key,
                source_operand=lhs_name,
            )
        )
    return AmdgpuDescriptorOverlay(
        descriptor_key=descriptor_key,
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        encoding_name="ENC_VOP2",
        semantic_tag=semantic_tag,
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            AmdgpuOperandOverlay("SRC0", _vgpr_operand(lhs_name)),
            AmdgpuOperandOverlay("VSRC1", _vgpr_operand(rhs_name)),
        ),
        operand_forms=tuple(operand_forms),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_mul_lo_u32_overlay() -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.v_mul_lo_u32",
        instruction_name="V_MUL_LO_U32",
        mnemonic="v_mul_lo_u32",
        encoding_name="ENC_VOP3",
        semantic_tag="integer.mul.lo.u32",
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            AmdgpuOperandOverlay("SRC0", _vgpr_operand("lhs")),
            AmdgpuOperandOverlay("SRC1", _vgpr_operand("rhs")),
        ),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_mul_hi_u32_overlay() -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.v_mul_hi_u32",
        instruction_name="V_MUL_HI_U32",
        mnemonic="v_mul_hi_u32",
        encoding_name="ENC_VOP3",
        semantic_tag="integer.mul.hi.u32",
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            AmdgpuOperandOverlay("SRC0", _vgpr_operand("lhs")),
            AmdgpuOperandOverlay("SRC1", _vgpr_operand("rhs")),
        ),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_mul_u32_u24_overlay() -> AmdgpuDescriptorOverlay:
    return _v_binary_u32_overlay(
        descriptor_key="amdgpu.v_mul_u32_u24",
        instruction_name="V_MUL_U32_U24",
        mnemonic="v_mul_u32_u24",
        semantic_tag="integer.mul.lo.u24.u32",
        src0_inline_descriptor_key="amdgpu.v_mul_u32_u24.src0_inline",
        literal_descriptor_key="amdgpu.v_mul_u32_u24.lit",
    )


def _v_mul_u32_u24_src0_inline_overlay() -> AmdgpuDescriptorOverlay:
    return _v_binary_src0_inline_overlay(
        descriptor_key="amdgpu.v_mul_u32_u24.src0_inline",
        instruction_name="V_MUL_U32_U24",
        mnemonic="v_mul_u32_u24",
        semantic_tag="integer.mul.lo.u24.u32",
    )


def _v_mul_u32_u24_literal_overlay() -> AmdgpuDescriptorOverlay:
    return _v_binary_literal_overlay(
        descriptor_key="amdgpu.v_mul_u32_u24.lit",
        instruction_name="V_MUL_U32_U24",
        mnemonic="v_mul_u32_u24",
        semantic_tag="integer.mul.lo.u24.u32",
    )


def _v_mad_u32_u24_overlay(
    *,
    include_literal_forms: bool = True,
) -> AmdgpuDescriptorOverlay:
    operand_forms: tuple[OperandForm, ...] = ()
    if include_literal_forms:
        operand_forms = (
            _literal_operand_form(
                replacement_descriptor="amdgpu.v_mad_u32_u24.src0_lit",
                source_operand="a",
            ),
            _literal_operand_form(
                replacement_descriptor="amdgpu.v_mad_u32_u24.src1_lit",
                source_operand="b",
            ),
            _literal_operand_form(
                replacement_descriptor="amdgpu.v_mad_u32_u24.src2_lit",
                source_operand="addend",
            ),
        )
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.v_mad_u32_u24",
        instruction_name="V_MAD_U32_U24",
        mnemonic="v_mad_u32_u24",
        encoding_name="ENC_VOP3",
        semantic_tag="integer.mad.lo.u24.u32",
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            AmdgpuOperandOverlay(
                "SRC0",
                _sgpr_vgpr_operand("a"),
                size_exception_reason=_U24_SOURCE_SIZE_REASON,
            ),
            AmdgpuOperandOverlay(
                "SRC1",
                _sgpr_vgpr_operand("b"),
                size_exception_reason=_U24_SOURCE_SIZE_REASON,
            ),
            AmdgpuOperandOverlay("SRC2", _sgpr_vgpr_operand("addend")),
        ),
        operand_forms=operand_forms,
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_mad_u32_u24_literal_overlay(literal_source: str) -> AmdgpuDescriptorOverlay:
    source_fields = {
        "src0": ("SRC0", "a", _sgpr_vgpr_operand("a"), _U24_SOURCE_SIZE_REASON),
        "src1": ("SRC1", "b", _sgpr_vgpr_operand("b"), _U24_SOURCE_SIZE_REASON),
        "src2": ("SRC2", "addend", _sgpr_vgpr_operand("addend"), None),
    }
    literal_field = source_fields[literal_source][0]
    operands = [AmdgpuOperandOverlay("VDST", _vgpr_result())]
    asm_operands = []
    for source_name, (
        xml_field,
        field_name,
        operand,
        size_reason,
    ) in source_fields.items():
        if source_name == literal_source:
            continue
        asm_operands.append(field_name)
        operands.append(
            AmdgpuOperandOverlay(
                xml_field,
                operand,
                size_exception_reason=size_reason,
            )
        )
    return AmdgpuDescriptorOverlay(
        descriptor_key=f"amdgpu.v_mad_u32_u24.{literal_source}_lit",
        instruction_name="V_MAD_U32_U24",
        mnemonic=f"v_mad_u32_u24_{literal_source}_lit",
        encoding_name="ENC_VOP3",
        encoding_format_id=AMDGPU_ENCODING_FORMAT_VOP3_LITERAL,
        semantic_tag="integer.mad.lo.u24.u32",
        schedule_class=_SCHEDULE_VALU,
        operands=tuple(operands),
        asm_forms=_asm(
            results=("dst",),
            operands=tuple(asm_operands),
            immediates=("imm32",),
        ),
        immediates=(_LITERAL_U32_IMMEDIATE,),
        fixed_encoding_fields=((literal_field, _predefined("SRC_LITERAL", "OPR_SRC")),),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_minmax_i32_overlay(
    *,
    descriptor_key: str,
    instruction_name: str,
    mnemonic: str,
    semantic_tag: str,
) -> AmdgpuDescriptorOverlay:
    return _v_binary_u32_overlay(
        descriptor_key=descriptor_key,
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        semantic_tag=semantic_tag,
    )


def _v_min_i32_overlay() -> AmdgpuDescriptorOverlay:
    return _v_minmax_i32_overlay(
        descriptor_key="amdgpu.v_min_i32",
        instruction_name="V_MIN_I32",
        mnemonic="v_min_i32",
        semantic_tag="integer.min.i32",
    )


def _v_max_i32_overlay() -> AmdgpuDescriptorOverlay:
    return _v_minmax_i32_overlay(
        descriptor_key="amdgpu.v_max_i32",
        instruction_name="V_MAX_I32",
        mnemonic="v_max_i32",
        semantic_tag="integer.max.i32",
    )


def _v_min_u32_overlay() -> AmdgpuDescriptorOverlay:
    return _v_minmax_i32_overlay(
        descriptor_key="amdgpu.v_min_u32",
        instruction_name="V_MIN_U32",
        mnemonic="v_min_u32",
        semantic_tag="integer.min.u32",
    )


def _v_max_u32_overlay() -> AmdgpuDescriptorOverlay:
    return _v_minmax_i32_overlay(
        descriptor_key="amdgpu.v_max_u32",
        instruction_name="V_MAX_U32",
        mnemonic="v_max_u32",
        semantic_tag="integer.max.u32",
    )


def _v_readfirstlane_b32_overlay() -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.v_readfirstlane_b32",
        instruction_name="V_READFIRSTLANE_B32",
        mnemonic="v_readfirstlane_b32",
        encoding_name="ENC_VOP1",
        semantic_tag="lane.readfirst.b32",
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _sgpr_result()),
            AmdgpuOperandOverlay("SRC0", _vgpr_operand("value")),
        ),
        effects=(_CONVERGENT_EFFECT,),
    )


def _s_and_b32_overlay() -> AmdgpuDescriptorOverlay:
    return _s_binary_u32_overlay(
        descriptor_key="amdgpu.s_and_b32",
        instruction_name="S_AND_B32",
        mnemonic="s_and_b32",
        semantic_tag="integer.and.u32",
    )


def _s_or_b32_overlay() -> AmdgpuDescriptorOverlay:
    return _s_binary_u32_overlay(
        descriptor_key="amdgpu.s_or_b32",
        instruction_name="S_OR_B32",
        mnemonic="s_or_b32",
        semantic_tag="integer.or.u32",
    )


def _s_xor_b32_overlay() -> AmdgpuDescriptorOverlay:
    return _s_binary_u32_overlay(
        descriptor_key="amdgpu.s_xor_b32",
        instruction_name="S_XOR_B32",
        mnemonic="s_xor_b32",
        semantic_tag="integer.xor.u32",
    )


def _s_and_b64_overlay() -> AmdgpuDescriptorOverlay:
    return _s_binary_u64_overlay(
        descriptor_key="amdgpu.s_and_b64",
        instruction_name="S_AND_B64",
        mnemonic="s_and_b64",
        semantic_tag="integer.and.u64",
    )


def _s_or_b64_overlay() -> AmdgpuDescriptorOverlay:
    return _s_binary_u64_overlay(
        descriptor_key="amdgpu.s_or_b64",
        instruction_name="S_OR_B64",
        mnemonic="s_or_b64",
        semantic_tag="integer.or.u64",
    )


def _s_xor_b64_overlay() -> AmdgpuDescriptorOverlay:
    return _s_binary_u64_overlay(
        descriptor_key="amdgpu.s_xor_b64",
        instruction_name="S_XOR_B64",
        mnemonic="s_xor_b64",
        semantic_tag="integer.xor.u64",
    )


def _s_cmp_i32_overlays() -> tuple[AmdgpuDescriptorOverlay, ...]:
    return (
        _s_cmp_i32_overlay(
            descriptor_key="amdgpu.s_cmp_eq_i32",
            instruction_name="S_CMP_EQ_I32",
            mnemonic="s_cmp_eq_i32",
            semantic_tag="integer.compare.eq.i32",
        ),
        _s_cmp_i32_overlay(
            descriptor_key="amdgpu.s_cmp_lg_i32",
            instruction_name="S_CMP_LG_I32",
            mnemonic="s_cmp_lg_i32",
            semantic_tag="integer.compare.ne.i32",
        ),
        _s_cmp_i32_overlay(
            descriptor_key="amdgpu.s_cmp_lt_i32",
            instruction_name="S_CMP_LT_I32",
            mnemonic="s_cmp_lt_i32",
            semantic_tag="integer.compare.slt.i32",
        ),
        _s_cmp_i32_overlay(
            descriptor_key="amdgpu.s_cmp_le_i32",
            instruction_name="S_CMP_LE_I32",
            mnemonic="s_cmp_le_i32",
            semantic_tag="integer.compare.sle.i32",
        ),
        _s_cmp_i32_overlay(
            descriptor_key="amdgpu.s_cmp_gt_i32",
            instruction_name="S_CMP_GT_I32",
            mnemonic="s_cmp_gt_i32",
            semantic_tag="integer.compare.sgt.i32",
        ),
        _s_cmp_i32_overlay(
            descriptor_key="amdgpu.s_cmp_ge_i32",
            instruction_name="S_CMP_GE_I32",
            mnemonic="s_cmp_ge_i32",
            semantic_tag="integer.compare.sge.i32",
        ),
        _s_cmp_i32_overlay(
            descriptor_key="amdgpu.s_cmp_lt_u32",
            instruction_name="S_CMP_LT_U32",
            mnemonic="s_cmp_lt_u32",
            semantic_tag="integer.compare.ult.i32",
        ),
        _s_cmp_i32_overlay(
            descriptor_key="amdgpu.s_cmp_le_u32",
            instruction_name="S_CMP_LE_U32",
            mnemonic="s_cmp_le_u32",
            semantic_tag="integer.compare.ule.i32",
        ),
        _s_cmp_i32_overlay(
            descriptor_key="amdgpu.s_cmp_gt_u32",
            instruction_name="S_CMP_GT_U32",
            mnemonic="s_cmp_gt_u32",
            semantic_tag="integer.compare.ugt.i32",
        ),
        _s_cmp_i32_overlay(
            descriptor_key="amdgpu.s_cmp_ge_u32",
            instruction_name="S_CMP_GE_U32",
            mnemonic="s_cmp_ge_u32",
            semantic_tag="integer.compare.uge.i32",
        ),
    )


def _s_cmp_u64_overlays() -> tuple[AmdgpuDescriptorOverlay, ...]:
    return (
        _s_cmp_u64_overlay(
            descriptor_key="amdgpu.s_cmp_eq_u64",
            instruction_name="S_CMP_EQ_U64",
            mnemonic="s_cmp_eq_u64",
            semantic_tag="integer.compare.eq.u64",
        ),
        _s_cmp_u64_overlay(
            descriptor_key="amdgpu.s_cmp_lg_u64",
            instruction_name="S_CMP_LG_U64",
            mnemonic="s_cmp_lg_u64",
            semantic_tag="integer.compare.ne.u64",
        ),
    )


def _s_and_saveexec_b64_overlay(
    encoding_condition: str,
) -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.s_and_saveexec_b64",
        instruction_name="S_AND_SAVEEXEC_B64",
        mnemonic="s_and_saveexec_b64",
        encoding_name="ENC_SOP1",
        encoding_condition=encoding_condition,
        semantic_tag="control.exec.and_save",
        schedule_class=_SCHEDULE_SALU,
        operands=(
            AmdgpuOperandOverlay("SDST", _sgpr_result("saved_exec", units=2)),
            AmdgpuOperandOverlay("SSRC0", _sgpr_operand("mask", units=2)),
        ),
        implicit_operands=(
            AmdgpuImplicitOperandOverlay(
                "OPR_SDST_EXEC",
                descriptor_operand=_exec_clobber("exec_out"),
                data_format_name="FMT_NUM_M64",
                size_bits=64,
                is_input=False,
                is_output=True,
            ),
            AmdgpuImplicitOperandOverlay(
                "OPR_SDST_EXEC",
                descriptor_operand=_exec_state_read(),
                data_format_name="FMT_NUM_M64",
                size_bits=64,
                is_input=True,
                is_output=False,
            ),
            _scc_output(_scc_result("active")),
        ),
        asm_forms=_asm(results=("saved_exec", "active"), operands=("mask",)),
        effects=(_CONVERGENT_EFFECT,),
        flags=(DescriptorFlag.SIDE_EFFECTING,),
    )


def _s_lshl_b32_overlay() -> AmdgpuDescriptorOverlay:
    return _s_binary_u32_overlay(
        descriptor_key="amdgpu.s_lshl_b32",
        instruction_name="S_LSHL_B32",
        mnemonic="s_lshl_b32",
        semantic_tag="integer.shl.u32",
        rhs_inline_descriptor_key="amdgpu.s_lshl_b32.rhs_inline",
    )


def _s_lshl_b32_rhs_inline_overlay() -> AmdgpuDescriptorOverlay:
    return _s_binary_u32_rhs_inline_overlay(
        descriptor_key="amdgpu.s_lshl_b32.rhs_inline",
        instruction_name="S_LSHL_B32",
        mnemonic="s_lshl_b32",
        semantic_tag="integer.shl.u32",
    )


def _s_lshl_b64_overlay() -> AmdgpuDescriptorOverlay:
    return _s_shift_u64_overlay(
        descriptor_key="amdgpu.s_lshl_b64",
        instruction_name="S_LSHL_B64",
        mnemonic="s_lshl_b64",
        semantic_tag="integer.shl.u64",
    )


def _s_lshr_b32_overlay() -> AmdgpuDescriptorOverlay:
    return _s_binary_u32_overlay(
        descriptor_key="amdgpu.s_lshr_b32",
        instruction_name="S_LSHR_B32",
        mnemonic="s_lshr_b32",
        semantic_tag="integer.shr.u32",
        rhs_inline_descriptor_key="amdgpu.s_lshr_b32.rhs_inline",
    )


def _s_lshr_b32_rhs_inline_overlay() -> AmdgpuDescriptorOverlay:
    return _s_binary_u32_rhs_inline_overlay(
        descriptor_key="amdgpu.s_lshr_b32.rhs_inline",
        instruction_name="S_LSHR_B32",
        mnemonic="s_lshr_b32",
        semantic_tag="integer.shr.u32",
    )


def _s_lshr_b64_overlay() -> AmdgpuDescriptorOverlay:
    return _s_shift_u64_overlay(
        descriptor_key="amdgpu.s_lshr_b64",
        instruction_name="S_LSHR_B64",
        mnemonic="s_lshr_b64",
        semantic_tag="integer.shr.u64",
    )


def _s_ashr_i32_overlay() -> AmdgpuDescriptorOverlay:
    return _s_binary_u32_overlay(
        descriptor_key="amdgpu.s_ashr_i32",
        instruction_name="S_ASHR_I32",
        mnemonic="s_ashr_i32",
        semantic_tag="integer.shr.i32",
        rhs_inline_descriptor_key="amdgpu.s_ashr_i32.rhs_inline",
    )


def _s_ashr_i32_rhs_inline_overlay() -> AmdgpuDescriptorOverlay:
    return _s_binary_u32_rhs_inline_overlay(
        descriptor_key="amdgpu.s_ashr_i32.rhs_inline",
        instruction_name="S_ASHR_I32",
        mnemonic="s_ashr_i32",
        semantic_tag="integer.shr.i32",
    )


def _s_bfe_b32_overlay(*, is_signed: bool) -> AmdgpuDescriptorOverlay:
    type_suffix = "i32" if is_signed else "u32"
    return AmdgpuDescriptorOverlay(
        descriptor_key=f"amdgpu.s_bfe_{type_suffix}",
        instruction_name=f"S_BFE_{type_suffix.upper()}",
        mnemonic=f"s_bfe_{type_suffix}",
        encoding_name="ENC_SOP2",
        semantic_tag=f"integer.bitfield.extract.{type_suffix}",
        schedule_class=_SCHEDULE_SALU,
        operands=(
            AmdgpuOperandOverlay("SDST", _sgpr_result()),
            AmdgpuOperandOverlay("SSRC0", _sgpr_operand("value")),
            AmdgpuOperandOverlay("SSRC1", _sgpr_operand("control")),
        ),
        implicit_operands=(_SCC_CLOBBER_OUTPUT,),
        operand_forms=(
            _literal_operand_form(
                replacement_descriptor=f"amdgpu.s_bfe_{type_suffix}.lit",
                source_operand="control",
            ),
        ),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _s_bfe_b32_literal_overlay(*, is_signed: bool) -> AmdgpuDescriptorOverlay:
    type_suffix = "i32" if is_signed else "u32"
    return AmdgpuDescriptorOverlay(
        descriptor_key=f"amdgpu.s_bfe_{type_suffix}.lit",
        instruction_name=f"S_BFE_{type_suffix.upper()}",
        mnemonic=f"s_bfe_{type_suffix}",
        encoding_name="ENC_SOP2",
        encoding_format_id=AMDGPU_ENCODING_FORMAT_SOP2_LITERAL,
        semantic_tag=f"integer.bitfield.extract.{type_suffix}",
        schedule_class=_SCHEDULE_SALU,
        operands=(
            AmdgpuOperandOverlay("SDST", _sgpr_result()),
            AmdgpuOperandOverlay("SSRC0", _sgpr_operand("value")),
        ),
        implicit_operands=(_SCC_CLOBBER_OUTPUT,),
        asm_forms=_asm(
            mnemonic=f"s_bfe_{type_suffix}_lit",
            results=("dst",),
            operands=("value",),
            immediates=("imm32",),
        ),
        immediate_fields=("LITERAL",),
        immediates=(_LITERAL_U32_IMMEDIATE,),
        fixed_encoding_fields=(("SSRC1", _predefined("SRC_LITERAL", "OPR_SSRC")),),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_and_b32_overlay() -> AmdgpuDescriptorOverlay:
    return _v_binary_u32_overlay(
        descriptor_key="amdgpu.v_and_b32",
        instruction_name="V_AND_B32",
        mnemonic="v_and_b32",
        semantic_tag="integer.and.u32",
        src0_inline_descriptor_key="amdgpu.v_and_b32.src0_inline",
        literal_descriptor_key="amdgpu.v_and_b32.lit",
    )


def _v_and_b32_src0_inline_overlay() -> AmdgpuDescriptorOverlay:
    return _v_binary_src0_inline_overlay(
        descriptor_key="amdgpu.v_and_b32.src0_inline",
        instruction_name="V_AND_B32",
        mnemonic="v_and_b32",
        semantic_tag="integer.and.u32",
        rhs_name="rhs",
    )


def _v_and_b32_literal_overlay() -> AmdgpuDescriptorOverlay:
    return _v_binary_literal_overlay(
        descriptor_key="amdgpu.v_and_b32.lit",
        instruction_name="V_AND_B32",
        mnemonic="v_and_b32",
        semantic_tag="integer.and.u32",
        rhs_name="rhs",
    )


def _v_or_b32_overlay() -> AmdgpuDescriptorOverlay:
    return _v_binary_u32_overlay(
        descriptor_key="amdgpu.v_or_b32",
        instruction_name="V_OR_B32",
        mnemonic="v_or_b32",
        semantic_tag="integer.or.u32",
        src0_inline_descriptor_key="amdgpu.v_or_b32.src0_inline",
        literal_descriptor_key="amdgpu.v_or_b32.lit",
    )


def _v_or_b32_src0_inline_overlay() -> AmdgpuDescriptorOverlay:
    return _v_binary_src0_inline_overlay(
        descriptor_key="amdgpu.v_or_b32.src0_inline",
        instruction_name="V_OR_B32",
        mnemonic="v_or_b32",
        semantic_tag="integer.or.u32",
        rhs_name="rhs",
    )


def _v_or_b32_literal_overlay() -> AmdgpuDescriptorOverlay:
    return _v_binary_literal_overlay(
        descriptor_key="amdgpu.v_or_b32.lit",
        instruction_name="V_OR_B32",
        mnemonic="v_or_b32",
        semantic_tag="integer.or.u32",
        rhs_name="rhs",
    )


def _v_xor_b32_overlay() -> AmdgpuDescriptorOverlay:
    return _v_binary_u32_overlay(
        descriptor_key="amdgpu.v_xor_b32",
        instruction_name="V_XOR_B32",
        mnemonic="v_xor_b32",
        semantic_tag="integer.xor.u32",
        src0_inline_descriptor_key="amdgpu.v_xor_b32.src0_inline",
        literal_descriptor_key="amdgpu.v_xor_b32.lit",
    )


def _v_xor_b32_src0_inline_overlay() -> AmdgpuDescriptorOverlay:
    return _v_binary_src0_inline_overlay(
        descriptor_key="amdgpu.v_xor_b32.src0_inline",
        instruction_name="V_XOR_B32",
        mnemonic="v_xor_b32",
        semantic_tag="integer.xor.u32",
        rhs_name="rhs",
    )


def _v_xor_b32_literal_overlay() -> AmdgpuDescriptorOverlay:
    return _v_binary_literal_overlay(
        descriptor_key="amdgpu.v_xor_b32.lit",
        instruction_name="V_XOR_B32",
        mnemonic="v_xor_b32",
        semantic_tag="integer.xor.u32",
        rhs_name="rhs",
    )


def _v_lshlrev_b32_overlay() -> AmdgpuDescriptorOverlay:
    return _v_binary_u32_overlay(
        descriptor_key="amdgpu.v_lshlrev_b32",
        instruction_name="V_LSHLREV_B32",
        mnemonic="v_lshlrev_b32",
        semantic_tag="integer.shl.u32",
        lhs_name="shift",
        rhs_name="value",
        src0_inline_descriptor_key="amdgpu.v_lshlrev_b32.src0_inline",
        literal_descriptor_key="amdgpu.v_lshlrev_b32.lit",
    )


def _v_lshlrev_b32_src0_inline_overlay() -> AmdgpuDescriptorOverlay:
    return _v_binary_src0_inline_overlay(
        descriptor_key="amdgpu.v_lshlrev_b32.src0_inline",
        instruction_name="V_LSHLREV_B32",
        mnemonic="v_lshlrev_b32",
        semantic_tag="integer.shl.u32",
        rhs_name="value",
    )


def _v_lshlrev_b32_src0_16_low16_overlay() -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.v_lshlrev_b32.src0_16_low16",
        instruction_name="V_LSHLREV_B32",
        mnemonic="v_lshlrev_b32",
        encoding_name="ENC_VOP2",
        semantic_tag="integer.shl.u32.low16_to_high16",
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            AmdgpuOperandOverlay(
                "VSRC1",
                _vgpr_operand("value", register_part=_REG_PART_VGPR_LOW16),
                size_exception_reason=_D16_PARTIAL_REGISTER_SIZE_REASON,
            ),
        ),
        asm_forms=_asm(
            mnemonic="v_lshlrev_b32_src0_16_low16",
            results=("dst",),
            operands=("value",),
            immediates=("imm32",),
        ),
        immediate_fields=("SRC0",),
        immediates=(_SOURCE_INLINE_U32_16_IMMEDIATE,),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_lshlrev_b32_literal_overlay() -> AmdgpuDescriptorOverlay:
    return _v_binary_literal_overlay(
        descriptor_key="amdgpu.v_lshlrev_b32.lit",
        instruction_name="V_LSHLREV_B32",
        mnemonic="v_lshlrev_b32",
        semantic_tag="integer.shl.u32",
        rhs_name="value",
    )


def _v_lshlrev_b32_vop3_immediate_overlay() -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.v_lshlrev_b32.vop3_imm",
        instruction_name="V_LSHLREV_B32",
        mnemonic="v_lshlrev_b32",
        encoding_name="ENC_VOP3",
        semantic_tag="integer.shl.u32",
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            AmdgpuOperandOverlay("SRC1", _sgpr_vgpr_operand("value")),
        ),
        asm_forms=_asm(
            mnemonic="v_lshlrev_b32_vop3_imm",
            results=("dst",),
            operands=("value",),
            immediates=("imm32",),
        ),
        immediate_fields=("SRC0",),
        immediates=(_SOURCE_INLINE_U32_IMMEDIATE,),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_lshlrev_b64_overlay() -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.v_lshlrev_b64",
        instruction_name="V_LSHLREV_B64",
        mnemonic="v_lshlrev_b64",
        encoding_name="ENC_VOP3",
        semantic_tag="integer.shl.u64",
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result(units=2)),
            AmdgpuOperandOverlay("SRC0", _vgpr_operand("shift")),
            AmdgpuOperandOverlay("SRC1", _vgpr_operand("value", units=2)),
        ),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_lshl_add_u32_shift_immediate_overlay() -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.v_lshl_add_u32.shift_imm",
        instruction_name="V_LSHL_ADD_U32",
        mnemonic="v_lshl_add_u32",
        encoding_name="ENC_VOP3",
        semantic_tag="integer.shl.add.u32",
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            AmdgpuOperandOverlay("SRC0", _sgpr_vgpr_operand("value")),
            AmdgpuOperandOverlay("SRC2", _sgpr_vgpr_operand("addend")),
        ),
        asm_forms=_asm(
            mnemonic="v_lshl_add_u32_shift_imm",
            results=("dst",),
            operands=("value", "addend"),
            immediates=("shift",),
        ),
        immediate_fields=("SRC1",),
        immediates=(_source_inline_u32_immediate("shift"),),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_bfe_offset_immediate() -> Immediate:
    return replace(_source_inline_u32_immediate("offset"), unsigned_max=31)


def _v_bfe_width_immediate() -> Immediate:
    return replace(_source_inline_u32_immediate("width"), unsigned_max=32)


def _v_bfe_low16_offset_immediate() -> Immediate:
    return replace(
        _v_bfe_offset_immediate(),
        flags=(ImmediateFlag.DEFAULT_VALUE,),
        default_value=0,
    )


def _v_bfe_offset_width_inline_overlay(*, is_signed: bool) -> AmdgpuDescriptorOverlay:
    type_suffix = "i32" if is_signed else "u32"
    return AmdgpuDescriptorOverlay(
        descriptor_key=f"amdgpu.v_bfe_{type_suffix}.offset_width_inline",
        instruction_name=f"V_BFE_{type_suffix.upper()}",
        mnemonic=f"v_bfe_{type_suffix}",
        encoding_name="ENC_VOP3",
        semantic_tag=f"integer.bitfield.extract.{type_suffix}",
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            AmdgpuOperandOverlay("SRC0", _sgpr_vgpr_operand("value")),
        ),
        asm_forms=_asm(
            mnemonic=f"v_bfe_{type_suffix}_offset_width_inline",
            native_assembly_mnemonic=f"v_bfe_{type_suffix}",
            results=("dst",),
            operands=("value",),
            immediates=("offset", "width"),
            named_immediates=True,
            native_assembly_values=(
                _native_result("dst"),
                _native_operand("value"),
                _native_i64_immediate("offset"),
                _native_i64_immediate("width"),
            ),
        ),
        immediate_fields=("SRC1", "SRC2"),
        immediates=(
            _v_bfe_offset_immediate(),
            _v_bfe_width_immediate(),
        ),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_bfe_u32_offset_0_width_16_low16_overlay() -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.v_bfe_u32.offset_0_width_16_low16",
        instruction_name="V_BFE_U32",
        mnemonic="v_bfe_u32",
        encoding_name="ENC_VOP3",
        semantic_tag="integer.bitfield.extract.u32.low16_to_full32",
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            AmdgpuOperandOverlay(
                "SRC0",
                _vgpr_operand("value", register_part=_REG_PART_VGPR_LOW16),
                size_exception_reason=_D16_PARTIAL_REGISTER_SIZE_REASON,
            ),
        ),
        asm_forms=_asm(
            mnemonic="v_bfe_u32_offset_0_width_16_low16",
            native_assembly_mnemonic="v_bfe_u32",
            results=("dst",),
            operands=("value",),
            immediates=("offset", "width"),
            named_immediates=True,
            native_assembly_values=(
                _native_result("dst"),
                _native_operand("value"),
                _native_i64_immediate("offset"),
                _native_i64_immediate("width"),
            ),
        ),
        immediate_fields=("SRC1", "SRC2"),
        immediates=(
            _v_bfe_low16_offset_immediate(),
            _source_inline_u32_16_immediate("width"),
        ),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_bfi_b32_src0_literal_overlay() -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.v_bfi_b32.src0_lit",
        instruction_name="V_BFI_B32",
        mnemonic="v_bfi_b32",
        encoding_name="ENC_VOP3",
        encoding_format_id=AMDGPU_ENCODING_FORMAT_VOP3_LITERAL,
        semantic_tag="integer.bitfield.insert.u32",
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            AmdgpuOperandOverlay("SRC1", _sgpr_vgpr_operand("insert")),
            AmdgpuOperandOverlay("SRC2", _sgpr_vgpr_operand("base")),
        ),
        asm_forms=_asm(
            mnemonic="v_bfi_b32_src0_lit",
            results=("dst",),
            operands=("insert", "base"),
            immediates=("imm32",),
        ),
        immediates=(_LITERAL_U32_IMMEDIATE,),
        fixed_encoding_fields=(("SRC0", _predefined("SRC_LITERAL", "OPR_SRC")),),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_perm_b32_overlay() -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.v_perm_b32",
        instruction_name="V_PERM_B32",
        mnemonic="v_perm_b32",
        encoding_name="ENC_VOP3",
        semantic_tag="integer.byte.permute.u32",
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            AmdgpuOperandOverlay("SRC0", _sgpr_vgpr_operand("src0")),
            AmdgpuOperandOverlay("SRC1", _sgpr_vgpr_operand("src1")),
            AmdgpuOperandOverlay("SRC2", _sgpr_vgpr_operand("selectors")),
        ),
        asm_forms=_asm(
            results=("dst",),
            operands=("src0", "src1", "selectors"),
        ),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_lshrrev_b32_overlay() -> AmdgpuDescriptorOverlay:
    return _v_binary_u32_overlay(
        descriptor_key="amdgpu.v_lshrrev_b32",
        instruction_name="V_LSHRREV_B32",
        mnemonic="v_lshrrev_b32",
        semantic_tag="integer.shr.u32",
        lhs_name="shift",
        rhs_name="value",
        src0_inline_descriptor_key="amdgpu.v_lshrrev_b32.src0_inline",
        literal_descriptor_key="amdgpu.v_lshrrev_b32.lit",
    )


def _v_lshrrev_b32_src0_inline_overlay() -> AmdgpuDescriptorOverlay:
    return _v_binary_src0_inline_overlay(
        descriptor_key="amdgpu.v_lshrrev_b32.src0_inline",
        instruction_name="V_LSHRREV_B32",
        mnemonic="v_lshrrev_b32",
        semantic_tag="integer.shr.u32",
        rhs_name="value",
    )


def _v_lshrrev_b32_literal_overlay() -> AmdgpuDescriptorOverlay:
    return _v_binary_literal_overlay(
        descriptor_key="amdgpu.v_lshrrev_b32.lit",
        instruction_name="V_LSHRREV_B32",
        mnemonic="v_lshrrev_b32",
        semantic_tag="integer.shr.u32",
        rhs_name="value",
    )


def _v_ashrrev_i32_overlay() -> AmdgpuDescriptorOverlay:
    return _v_binary_u32_overlay(
        descriptor_key="amdgpu.v_ashrrev_i32",
        instruction_name="V_ASHRREV_I32",
        mnemonic="v_ashrrev_i32",
        semantic_tag="integer.shr.i32",
        lhs_name="shift",
        rhs_name="value",
        src0_inline_descriptor_key="amdgpu.v_ashrrev_i32.src0_inline",
        literal_descriptor_key="amdgpu.v_ashrrev_i32.lit",
    )


def _v_ashrrev_i32_src0_inline_overlay() -> AmdgpuDescriptorOverlay:
    return _v_binary_src0_inline_overlay(
        descriptor_key="amdgpu.v_ashrrev_i32.src0_inline",
        instruction_name="V_ASHRREV_I32",
        mnemonic="v_ashrrev_i32",
        semantic_tag="integer.shr.i32",
        rhs_name="value",
    )


def _v_ashrrev_i32_literal_overlay() -> AmdgpuDescriptorOverlay:
    return _v_binary_literal_overlay(
        descriptor_key="amdgpu.v_ashrrev_i32.lit",
        instruction_name="V_ASHRREV_I32",
        mnemonic="v_ashrrev_i32",
        semantic_tag="integer.shr.i32",
        rhs_name="value",
    )


def _integer_bitwise_shift_overlays() -> tuple[AmdgpuDescriptorOverlay, ...]:
    return (
        _s_and_b32_overlay(),
        _s_or_b32_overlay(),
        _s_xor_b32_overlay(),
        _s_and_b64_overlay(),
        _s_or_b64_overlay(),
        _s_xor_b64_overlay(),
        _s_lshl_b32_overlay(),
        _s_lshl_b32_rhs_inline_overlay(),
        _s_lshl_b64_overlay(),
        _s_lshr_b32_overlay(),
        _s_lshr_b32_rhs_inline_overlay(),
        _s_lshr_b64_overlay(),
        _s_ashr_i32_overlay(),
        _s_ashr_i32_rhs_inline_overlay(),
        _s_bfe_b32_overlay(is_signed=False),
        _s_bfe_b32_literal_overlay(is_signed=False),
        _s_bfe_b32_overlay(is_signed=True),
        _s_bfe_b32_literal_overlay(is_signed=True),
        _v_and_b32_overlay(),
        _v_and_b32_src0_inline_overlay(),
        _v_and_b32_literal_overlay(),
        _v_or_b32_overlay(),
        _v_or_b32_src0_inline_overlay(),
        _v_or_b32_literal_overlay(),
        _v_xor_b32_overlay(),
        _v_xor_b32_src0_inline_overlay(),
        _v_xor_b32_literal_overlay(),
        _v_lshlrev_b32_overlay(),
        _v_lshlrev_b32_src0_inline_overlay(),
        _v_lshlrev_b32_src0_16_low16_overlay(),
        _v_lshlrev_b32_literal_overlay(),
        _v_lshlrev_b32_vop3_immediate_overlay(),
        _v_lshlrev_b64_overlay(),
        _v_lshl_add_u32_shift_immediate_overlay(),
        _v_bfe_offset_width_inline_overlay(is_signed=False),
        _v_bfe_offset_width_inline_overlay(is_signed=True),
        _v_bfe_u32_offset_0_width_16_low16_overlay(),
        _v_bfi_b32_src0_literal_overlay(),
        _v_lshrrev_b32_overlay(),
        _v_lshrrev_b32_src0_inline_overlay(),
        _v_lshrrev_b32_literal_overlay(),
        _v_ashrrev_i32_overlay(),
        _v_ashrrev_i32_src0_inline_overlay(),
        _v_ashrrev_i32_literal_overlay(),
    )


def _integer_bitwise_permute_overlays() -> tuple[AmdgpuDescriptorOverlay, ...]:
    return (_v_perm_b32_overlay(),)


def _v_binary_f32_overlay(
    *,
    descriptor_key: str,
    instruction_name: str,
    mnemonic: str,
    semantic_tag: str,
    src0_name: str = "lhs",
    vsrc1_name: str = "rhs",
) -> AmdgpuDescriptorOverlay:
    descriptor = AmdgpuDescriptorOverlay(
        descriptor_key=descriptor_key,
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        encoding_name="ENC_VOP2",
        semantic_tag=semantic_tag,
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            AmdgpuOperandOverlay("SRC0", _sgpr_vgpr_operand(src0_name)),
            AmdgpuOperandOverlay("VSRC1", _vgpr_operand(vsrc1_name)),
        ),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )
    return replace(
        descriptor,
        operand_forms=_v_binary_f32_operand_forms(descriptor, source_operand=src0_name),
    )


def _v_add_f32_overlay() -> AmdgpuDescriptorOverlay:
    return _v_binary_f32_overlay(
        descriptor_key="amdgpu.v_add_f32",
        instruction_name="V_ADD_F32",
        mnemonic="v_add_f32",
        semantic_tag="float.add.f32",
    )


def _v_add_f32_literal_overlay() -> AmdgpuDescriptorOverlay:
    return _v_binary_literal_overlay(
        descriptor_key="amdgpu.v_add_f32.lit",
        instruction_name="V_ADD_F32",
        mnemonic="v_add_f32",
        semantic_tag="float.add.f32",
    )


def _v_add_f32_src0_inline_overlay() -> AmdgpuDescriptorOverlay:
    return _v_binary_src0_inline_f32_overlay(
        descriptor_key="amdgpu.v_add_f32.src0_inline",
        instruction_name="V_ADD_F32",
        mnemonic="v_add_f32",
        semantic_tag="float.add.f32",
    )


def _v_sub_f32_overlay() -> AmdgpuDescriptorOverlay:
    return _v_binary_f32_overlay(
        descriptor_key="amdgpu.v_sub_f32",
        instruction_name="V_SUB_F32",
        mnemonic="v_sub_f32",
        semantic_tag="float.sub.f32",
    )


def _v_sub_f32_literal_overlay() -> AmdgpuDescriptorOverlay:
    return _v_binary_literal_overlay(
        descriptor_key="amdgpu.v_sub_f32.lit",
        instruction_name="V_SUB_F32",
        mnemonic="v_sub_f32",
        semantic_tag="float.sub.f32",
    )


def _v_sub_f32_src0_inline_overlay() -> AmdgpuDescriptorOverlay:
    return _v_binary_src0_inline_f32_overlay(
        descriptor_key="amdgpu.v_sub_f32.src0_inline",
        instruction_name="V_SUB_F32",
        mnemonic="v_sub_f32",
        semantic_tag="float.sub.f32",
    )


def _v_subrev_f32_overlay() -> AmdgpuDescriptorOverlay:
    return _v_binary_f32_overlay(
        descriptor_key="amdgpu.v_subrev_f32",
        instruction_name="V_SUBREV_F32",
        mnemonic="v_subrev_f32",
        semantic_tag="float.sub.f32",
        src0_name="rhs",
        vsrc1_name="lhs",
    )


def _v_subrev_f32_literal_overlay() -> AmdgpuDescriptorOverlay:
    return _v_binary_literal_overlay(
        descriptor_key="amdgpu.v_subrev_f32.lit",
        instruction_name="V_SUBREV_F32",
        mnemonic="v_subrev_f32",
        semantic_tag="float.sub.f32",
        rhs_name="lhs",
    )


def _v_subrev_f32_src0_inline_overlay() -> AmdgpuDescriptorOverlay:
    return _v_binary_src0_inline_f32_overlay(
        descriptor_key="amdgpu.v_subrev_f32.src0_inline",
        instruction_name="V_SUBREV_F32",
        mnemonic="v_subrev_f32",
        semantic_tag="float.sub.f32",
        rhs_name="lhs",
    )


def _v_mul_f32_overlay() -> AmdgpuDescriptorOverlay:
    return _v_binary_f32_overlay(
        descriptor_key="amdgpu.v_mul_f32",
        instruction_name="V_MUL_F32",
        mnemonic="v_mul_f32",
        semantic_tag="float.mul.f32",
    )


def _v_mul_f32_literal_overlay() -> AmdgpuDescriptorOverlay:
    return _v_binary_literal_overlay(
        descriptor_key="amdgpu.v_mul_f32.lit",
        instruction_name="V_MUL_F32",
        mnemonic="v_mul_f32",
        semantic_tag="float.mul.f32",
    )


def _v_mul_f32_src0_inline_overlay() -> AmdgpuDescriptorOverlay:
    return _v_binary_src0_inline_f32_overlay(
        descriptor_key="amdgpu.v_mul_f32.src0_inline",
        instruction_name="V_MUL_F32",
        mnemonic="v_mul_f32",
        semantic_tag="float.mul.f32",
    )


def _v_min_f32_overlay() -> AmdgpuDescriptorOverlay:
    return _v_binary_f32_overlay(
        descriptor_key="amdgpu.v_min_f32",
        instruction_name="V_MIN_F32",
        mnemonic="v_min_f32",
        semantic_tag="float.minnum.f32",
    )


def _v_min_f32_literal_overlay() -> AmdgpuDescriptorOverlay:
    return _v_binary_literal_overlay(
        descriptor_key="amdgpu.v_min_f32.lit",
        instruction_name="V_MIN_F32",
        mnemonic="v_min_f32",
        semantic_tag="float.minnum.f32",
    )


def _v_min_f32_src0_inline_overlay() -> AmdgpuDescriptorOverlay:
    return _v_binary_src0_inline_f32_overlay(
        descriptor_key="amdgpu.v_min_f32.src0_inline",
        instruction_name="V_MIN_F32",
        mnemonic="v_min_f32",
        semantic_tag="float.minnum.f32",
    )


def _v_max_f32_overlay() -> AmdgpuDescriptorOverlay:
    return _v_binary_f32_overlay(
        descriptor_key="amdgpu.v_max_f32",
        instruction_name="V_MAX_F32",
        mnemonic="v_max_f32",
        semantic_tag="float.maxnum.f32",
    )


def _v_max_f32_literal_overlay() -> AmdgpuDescriptorOverlay:
    return _v_binary_literal_overlay(
        descriptor_key="amdgpu.v_max_f32.lit",
        instruction_name="V_MAX_F32",
        mnemonic="v_max_f32",
        semantic_tag="float.maxnum.f32",
    )


def _v_max_f32_src0_inline_overlay() -> AmdgpuDescriptorOverlay:
    return _v_binary_src0_inline_f32_overlay(
        descriptor_key="amdgpu.v_max_f32.src0_inline",
        instruction_name="V_MAX_F32",
        mnemonic="v_max_f32",
        semantic_tag="float.maxnum.f32",
    )


def _v_binary_f32_overlays() -> tuple[AmdgpuDescriptorOverlay, ...]:
    return (
        _v_add_f32_overlay(),
        _v_add_f32_literal_overlay(),
        _v_add_f32_src0_inline_overlay(),
        _v_sub_f32_overlay(),
        _v_sub_f32_literal_overlay(),
        _v_sub_f32_src0_inline_overlay(),
        _v_mul_f32_overlay(),
        _v_mul_f32_literal_overlay(),
        _v_mul_f32_src0_inline_overlay(),
        _v_min_f32_overlay(),
        _v_min_f32_literal_overlay(),
        _v_min_f32_src0_inline_overlay(),
        _v_max_f32_overlay(),
        _v_max_f32_literal_overlay(),
        _v_max_f32_src0_inline_overlay(),
    )


def _v_subrev_f32_overlays() -> tuple[AmdgpuDescriptorOverlay, ...]:
    return (
        _v_subrev_f32_overlay(),
        _v_subrev_f32_literal_overlay(),
        _v_subrev_f32_src0_inline_overlay(),
    )


def _v_fma_f32_overlay() -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.v_fma_f32",
        instruction_name="V_FMA_F32",
        mnemonic="v_fma_f32",
        encoding_name="ENC_VOP3",
        semantic_tag="float.fma.f32",
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            AmdgpuOperandOverlay("SRC0", _sgpr_vgpr_operand("a")),
            AmdgpuOperandOverlay("SRC1", _sgpr_vgpr_operand("b")),
            AmdgpuOperandOverlay("SRC2", _sgpr_vgpr_operand("c")),
        ),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


_V_MIX_SOURCE_PARTS = ("f32", "f16lo", "f16hi")

_V_MIX_HALF_RESULT_ACC_SIZE_REASON = "half-result-mix-ties-full-vgpr-accumulator"


def _v_mix_source_operand(field_name: str, source_part: str) -> Operand:
    if source_part == "f32":
        return _sgpr_vgpr_operand(field_name)
    register_part = {
        "lo": _REG_PART_VGPR_LOW16,
        "hi": _REG_PART_VGPR_HIGH16,
    }[source_part.removeprefix("f16")]
    return _vgpr_operand(field_name, register_part=register_part)


def _v_mix_source_size_reason(source_part: str) -> str | None:
    return _D16_PARTIAL_REGISTER_SIZE_REASON if source_part != "f32" else None


def _v_mix_source_selectors(source_parts: tuple[str, str, str]) -> tuple[int, int]:
    op_sel = 0
    op_sel_hi = 0
    for source_index, source_part in enumerate(source_parts):
        if source_part == "f16hi":
            op_sel |= 1 << source_index
        if source_part != "f32":
            op_sel_hi |= 1 << source_index
    return op_sel, op_sel_hi


def _v_mix_ternary_overlay(
    source_parts: tuple[str, str, str],
    *,
    descriptor_key_prefix: str,
    instruction_name: str,
    mnemonic_prefix: str,
    result_operand: Operand,
    semantic_tag_prefix: str,
    op_sel_field: str = "OPSEL",
    op_sel_hi_field: str = "OPSEL_HI",
    require_half_source: bool = False,
    operand_forms: tuple[OperandForm, ...] = (),
) -> AmdgpuDescriptorOverlay:
    op_sel, op_sel_hi = _v_mix_source_selectors(source_parts)
    if require_half_source and op_sel_hi == 0:
        raise ValueError(
            f"{instruction_name} descriptors require at least one f16 source"
        )
    suffix = "_".join(source_parts)
    return AmdgpuDescriptorOverlay(
        descriptor_key=f"{descriptor_key_prefix}.{suffix}",
        instruction_name=instruction_name,
        mnemonic=f"{mnemonic_prefix}_{suffix}",
        encoding_name="ENC_VOP3P",
        semantic_tag=f"{semantic_tag_prefix}.{'.'.join(source_parts)}",
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", result_operand),
            AmdgpuOperandOverlay(
                "SRC0",
                _v_mix_source_operand("a", source_parts[0]),
                size_exception_reason=_v_mix_source_size_reason(source_parts[0]),
            ),
            AmdgpuOperandOverlay(
                "SRC1",
                _v_mix_source_operand("b", source_parts[1]),
                size_exception_reason=_v_mix_source_size_reason(source_parts[1]),
            ),
            AmdgpuOperandOverlay(
                "SRC2",
                _v_mix_source_operand("c", source_parts[2]),
                size_exception_reason=_v_mix_source_size_reason(source_parts[2]),
            ),
        ),
        fixed_encoding_fields=(
            (op_sel_field, op_sel),
            (op_sel_hi_field, op_sel_hi),
        ),
        operand_forms=operand_forms,
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_mix_source_combinations(
    *, include_all_f32: bool
) -> tuple[tuple[str, str, str], ...]:
    return tuple(
        (source0_part, source1_part, source2_part)
        for source0_part in _V_MIX_SOURCE_PARTS
        for source1_part in _V_MIX_SOURCE_PARTS
        for source2_part in _V_MIX_SOURCE_PARTS
        if include_all_f32
        or (source0_part, source1_part, source2_part) != ("f32", "f32", "f32")
    )


def _v_fma_mix_f32_overlay(
    source_parts: tuple[str, str, str],
    *,
    op_sel_field: str = "OPSEL",
    op_sel_hi_field: str = "OPSEL_HI",
) -> AmdgpuDescriptorOverlay:
    suffix = "_".join(source_parts)
    operand_forms: tuple[OperandForm, ...] = ()
    if source_parts[2] == "f32":
        operand_forms = (
            _literal_operand_form(
                replacement_descriptor=f"amdgpu.v_fma_mix_f32.{suffix}.src2_lit",
                source_operand="c",
            ),
        )
    return _v_mix_ternary_overlay(
        source_parts,
        descriptor_key_prefix="amdgpu.v_fma_mix_f32",
        instruction_name="V_FMA_MIX_F32",
        mnemonic_prefix="v_fma_mix_f32",
        result_operand=_vgpr_result(),
        semantic_tag_prefix="float.fma.mix",
        op_sel_field=op_sel_field,
        op_sel_hi_field=op_sel_hi_field,
        require_half_source=True,
        operand_forms=operand_forms,
    )


def _v_fma_mix_f32_src2_literal_overlay(
    source_parts: tuple[str, str, str],
    *,
    op_sel_field: str = "OPSEL",
    op_sel_hi_field: str = "OPSEL_HI",
) -> AmdgpuDescriptorOverlay:
    if source_parts[2] != "f32":
        raise ValueError("V_FMA_MIX_F32 source-2 literal forms require f32 c")
    op_sel, op_sel_hi = _v_mix_source_selectors(source_parts)
    suffix = "_".join(source_parts)
    return AmdgpuDescriptorOverlay(
        descriptor_key=f"amdgpu.v_fma_mix_f32.{suffix}.src2_lit",
        instruction_name="V_FMA_MIX_F32",
        mnemonic=f"v_fma_mix_f32_{suffix}_src2_lit",
        encoding_name="ENC_VOP3P",
        encoding_format_id=AMDGPU_ENCODING_FORMAT_VOP3P_LITERAL,
        semantic_tag=f"float.fma.mix.{'.'.join(source_parts)}",
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            AmdgpuOperandOverlay(
                "SRC0",
                _v_mix_source_operand("a", source_parts[0]),
                size_exception_reason=_v_mix_source_size_reason(source_parts[0]),
            ),
            AmdgpuOperandOverlay(
                "SRC1",
                _v_mix_source_operand("b", source_parts[1]),
                size_exception_reason=_v_mix_source_size_reason(source_parts[1]),
            ),
        ),
        asm_forms=_asm(
            results=("dst",),
            operands=("a", "b"),
            immediates=("imm32",),
        ),
        immediates=(_LITERAL_U32_IMMEDIATE,),
        fixed_encoding_fields=(
            (op_sel_field, op_sel),
            (op_sel_hi_field, op_sel_hi),
            ("SRC2", _predefined("SRC_LITERAL", "OPR_SRC")),
        ),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_fma_mix_f32_overlays(
    *,
    op_sel_field: str = "OPSEL",
    op_sel_hi_field: str = "OPSEL_HI",
) -> tuple[AmdgpuDescriptorOverlay, ...]:
    overlays = []
    for source_parts in _v_mix_source_combinations(include_all_f32=False):
        overlays.append(
            _v_fma_mix_f32_overlay(
                source_parts,
                op_sel_field=op_sel_field,
                op_sel_hi_field=op_sel_hi_field,
            )
        )
        if source_parts[2] == "f32":
            overlays.append(
                _v_fma_mix_f32_src2_literal_overlay(
                    source_parts,
                    op_sel_field=op_sel_field,
                    op_sel_hi_field=op_sel_hi_field,
                )
            )
    return tuple(overlays)


def _v_mix_half_result_overlay(
    source_parts: tuple[str, str, str],
    *,
    descriptor_key_prefix: str,
    instruction_name: str,
    mnemonic_prefix: str,
    result_part: str,
    semantic_tag_prefix: str,
    op_sel_field: str = "OPSEL",
    op_sel_hi_field: str = "OPSEL_HI",
) -> AmdgpuDescriptorOverlay:
    result_register_part = {
        "lo": _REG_PART_VGPR_LOW16,
        "hi": _REG_PART_VGPR_HIGH16,
    }[result_part]
    op_sel, op_sel_hi = _v_mix_source_selectors(source_parts)
    suffix = "_".join(source_parts)
    return AmdgpuDescriptorOverlay(
        descriptor_key=f"{descriptor_key_prefix}.{suffix}",
        instruction_name=instruction_name,
        mnemonic=f"{mnemonic_prefix}_{suffix}",
        encoding_name="ENC_VOP3P",
        semantic_tag=f"{semantic_tag_prefix}.{'.'.join(source_parts)}",
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay(
                "VDST", _vgpr_result(register_part=result_register_part)
            ),
            AmdgpuOperandOverlay(
                "VDST",
                Operand(
                    "acc",
                    OperandRole.OPERAND,
                    _VGPR_ALT,
                    flags=(OperandFlag.IMPLICIT,),
                ),
                role_exception_reason=(
                    "the encoded destination register carries the untouched "
                    "half for the tied partial result"
                ),
                size_exception_reason=_V_MIX_HALF_RESULT_ACC_SIZE_REASON,
            ),
            AmdgpuOperandOverlay(
                "SRC0",
                _v_mix_source_operand("a", source_parts[0]),
                size_exception_reason=_v_mix_source_size_reason(source_parts[0]),
            ),
            AmdgpuOperandOverlay(
                "SRC1",
                _v_mix_source_operand("b", source_parts[1]),
                size_exception_reason=_v_mix_source_size_reason(source_parts[1]),
            ),
            AmdgpuOperandOverlay(
                "SRC2",
                _v_mix_source_operand("c", source_parts[2]),
                size_exception_reason=_v_mix_source_size_reason(source_parts[2]),
            ),
        ),
        constraints=(
            Constraint(ConstraintKind.TIED, 0, 1),
            Constraint(ConstraintKind.DESTRUCTIVE, 0, 1),
        ),
        asm_forms=_asm(results=("dst",), operands=("acc", "a", "b", "c")),
        fixed_encoding_fields=(
            (op_sel_field, op_sel),
            (op_sel_hi_field, op_sel_hi),
        ),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_fma_mixlo_f16_overlays(
    *,
    op_sel_field: str = "OPSEL",
    op_sel_hi_field: str = "OPSEL_HI",
) -> tuple[AmdgpuDescriptorOverlay, ...]:
    return tuple(
        _v_mix_half_result_overlay(
            source_parts,
            descriptor_key_prefix="amdgpu.v_fma_mixlo_f16",
            instruction_name="V_FMA_MIXLO_F16",
            mnemonic_prefix="v_fma_mixlo_f16",
            result_part="lo",
            semantic_tag_prefix="float.fma.mixlo.f16",
            op_sel_field=op_sel_field,
            op_sel_hi_field=op_sel_hi_field,
        )
        for source_parts in _v_mix_source_combinations(include_all_f32=True)
    )


def _v_fma_mixhi_f16_overlays(
    *,
    op_sel_field: str = "OPSEL",
    op_sel_hi_field: str = "OPSEL_HI",
) -> tuple[AmdgpuDescriptorOverlay, ...]:
    return tuple(
        _v_mix_half_result_overlay(
            source_parts,
            descriptor_key_prefix="amdgpu.v_fma_mixhi_f16",
            instruction_name="V_FMA_MIXHI_F16",
            mnemonic_prefix="v_fma_mixhi_f16",
            result_part="hi",
            semantic_tag_prefix="float.fma.mixhi.f16",
            op_sel_field=op_sel_field,
            op_sel_hi_field=op_sel_hi_field,
        )
        for source_parts in _v_mix_source_combinations(include_all_f32=True)
    )


def _v_mad_mix_f32_overlay(
    source_parts: tuple[str, str, str],
    *,
    op_sel_field: str = "OPSEL",
    op_sel_hi_field: str = "OPSEL_HI",
) -> AmdgpuDescriptorOverlay:
    return _v_mix_ternary_overlay(
        source_parts,
        descriptor_key_prefix="amdgpu.v_mad_mix_f32",
        instruction_name="V_MAD_MIX_F32",
        mnemonic_prefix="v_mad_mix_f32",
        result_operand=_vgpr_result(),
        semantic_tag_prefix="float.mad.mix",
        op_sel_field=op_sel_field,
        op_sel_hi_field=op_sel_hi_field,
        require_half_source=True,
    )


def _v_mad_mix_f32_overlays(
    *,
    op_sel_field: str = "OPSEL",
    op_sel_hi_field: str = "OPSEL_HI",
) -> tuple[AmdgpuDescriptorOverlay, ...]:
    return tuple(
        _v_mad_mix_f32_overlay(
            source_parts,
            op_sel_field=op_sel_field,
            op_sel_hi_field=op_sel_hi_field,
        )
        for source_parts in _v_mix_source_combinations(include_all_f32=False)
    )


def _v_mad_mixlo_f16_overlays(
    *,
    op_sel_field: str = "OPSEL",
    op_sel_hi_field: str = "OPSEL_HI",
) -> tuple[AmdgpuDescriptorOverlay, ...]:
    return tuple(
        _v_mix_half_result_overlay(
            source_parts,
            descriptor_key_prefix="amdgpu.v_mad_mixlo_f16",
            instruction_name="V_MAD_MIXLO_F16",
            mnemonic_prefix="v_mad_mixlo_f16",
            result_part="lo",
            semantic_tag_prefix="float.mad.mixlo.f16",
            op_sel_field=op_sel_field,
            op_sel_hi_field=op_sel_hi_field,
        )
        for source_parts in _v_mix_source_combinations(include_all_f32=True)
    )


def _v_mad_mixhi_f16_overlays(
    *,
    op_sel_field: str = "OPSEL",
    op_sel_hi_field: str = "OPSEL_HI",
) -> tuple[AmdgpuDescriptorOverlay, ...]:
    return tuple(
        _v_mix_half_result_overlay(
            source_parts,
            descriptor_key_prefix="amdgpu.v_mad_mixhi_f16",
            instruction_name="V_MAD_MIXHI_F16",
            mnemonic_prefix="v_mad_mixhi_f16",
            result_part="hi",
            semantic_tag_prefix="float.mad.mixhi.f16",
            op_sel_field=op_sel_field,
            op_sel_hi_field=op_sel_hi_field,
        )
        for source_parts in _v_mix_source_combinations(include_all_f32=True)
    )


def _v_fmac_f32_overlay() -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.v_fmac_f32",
        instruction_name="V_FMAC_F32",
        mnemonic="v_fmac_f32",
        encoding_name="ENC_VOP2",
        semantic_tag="float.fmac.f32",
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            AmdgpuOperandOverlay(
                "VDST",
                Operand(
                    "acc",
                    OperandRole.OPERAND,
                    _VGPR_ALT,
                    flags=(OperandFlag.IMPLICIT,),
                ),
                role_exception_reason=(
                    "the encoded destination register is also the tied "
                    "accumulator input"
                ),
            ),
            AmdgpuOperandOverlay("SRC0", _sgpr_vgpr_operand("a")),
            AmdgpuOperandOverlay("VSRC1", _vgpr_operand("b")),
        ),
        constraints=(
            Constraint(ConstraintKind.TIED, 0, 1),
            Constraint(ConstraintKind.DESTRUCTIVE, 0, 1),
        ),
        asm_forms=_asm(results=("dst",), operands=("acc", "a", "b")),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_fmaak_f32_overlay() -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.v_fmaak_f32",
        instruction_name="V_FMAAK_F32",
        mnemonic="v_fmaak_f32",
        encoding_name="VOP2_INST_LITERAL",
        encoding_condition="default",
        semantic_tag="float.fmaak.f32",
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            AmdgpuOperandOverlay("SRC0", _sgpr_vgpr_operand("a")),
            AmdgpuOperandOverlay("VSRC1", _vgpr_operand("b")),
        ),
        asm_forms=_asm(
            results=("dst",),
            operands=("a", "b"),
            immediates=("imm32",),
            native_assembly_values=(
                _native_result("dst"),
                _native_operand("a"),
                _native_operand("b"),
                _native_unsigned_hex_immediate("imm32", 32),
            ),
        ),
        immediate_fields=("LITERAL",),
        immediates=(_LITERAL_U32_IMMEDIATE,),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_fmamk_f32_overlay() -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.v_fmamk_f32",
        instruction_name="V_FMAMK_F32",
        mnemonic="v_fmamk_f32",
        encoding_name="VOP2_INST_LITERAL",
        encoding_condition="default",
        semantic_tag="float.fmamk.f32",
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            AmdgpuOperandOverlay("SRC0", _sgpr_vgpr_operand("a")),
            AmdgpuOperandOverlay("VSRC1", _vgpr_operand("c")),
        ),
        asm_forms=_asm(
            results=("dst",),
            operands=("a", "c"),
            immediates=("imm32",),
            native_assembly_values=(
                _native_result("dst"),
                _native_operand("a"),
                _native_unsigned_hex_immediate("imm32", 32),
                _native_operand("c"),
            ),
        ),
        immediate_fields=("LITERAL",),
        immediates=(_LITERAL_U32_IMMEDIATE,),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _s_fmaak_f32_overlay() -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.s_fmaak_f32",
        instruction_name="S_FMAAK_F32",
        mnemonic="s_fmaak_f32",
        encoding_name="SOP2_INST_LITERAL",
        encoding_condition="default",
        semantic_tag="float.fmaak.f32",
        schedule_class=_SCHEDULE_SALU,
        operands=(
            AmdgpuOperandOverlay("SDST", _sgpr_result()),
            AmdgpuOperandOverlay("SSRC0", _sgpr_operand("a")),
            AmdgpuOperandOverlay("SSRC1", _sgpr_operand("b")),
        ),
        asm_forms=_asm(
            results=("dst",),
            operands=("a", "b"),
            immediates=("imm32",),
            native_assembly_values=(
                _native_result("dst"),
                _native_operand("a"),
                _native_operand("b"),
                _native_unsigned_hex_immediate("imm32", 32),
            ),
        ),
        immediate_fields=("LITERAL",),
        immediates=(_LITERAL_U32_IMMEDIATE,),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _s_fmamk_f32_overlay() -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.s_fmamk_f32",
        instruction_name="S_FMAMK_F32",
        mnemonic="s_fmamk_f32",
        encoding_name="SOP2_INST_LITERAL",
        encoding_condition="default",
        semantic_tag="float.fmamk.f32",
        schedule_class=_SCHEDULE_SALU,
        operands=(
            AmdgpuOperandOverlay("SDST", _sgpr_result()),
            AmdgpuOperandOverlay("SSRC0", _sgpr_operand("a")),
            AmdgpuOperandOverlay("SSRC1", _sgpr_operand("c")),
        ),
        asm_forms=_asm(
            results=("dst",),
            operands=("a", "c"),
            immediates=("imm32",),
            native_assembly_values=(
                _native_result("dst"),
                _native_operand("a"),
                _native_unsigned_hex_immediate("imm32", 32),
                _native_operand("c"),
            ),
        ),
        immediate_fields=("LITERAL",),
        immediates=(_LITERAL_U32_IMMEDIATE,),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _s_fmac_f32_overlay() -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.s_fmac_f32",
        instruction_name="S_FMAC_F32",
        mnemonic="s_fmac_f32",
        encoding_name="ENC_SOP2",
        semantic_tag="float.fmac.f32",
        schedule_class=_SCHEDULE_SALU,
        operands=(
            AmdgpuOperandOverlay("SDST", _sgpr_result()),
            AmdgpuOperandOverlay(
                "SDST",
                Operand(
                    "acc",
                    OperandRole.OPERAND,
                    _SGPR_ALT,
                    flags=(OperandFlag.IMPLICIT,),
                ),
                role_exception_reason=(
                    "the encoded scalar destination register is also the tied "
                    "accumulator input"
                ),
            ),
            AmdgpuOperandOverlay("SSRC0", _sgpr_operand("a")),
            AmdgpuOperandOverlay("SSRC1", _sgpr_operand("b")),
        ),
        constraints=(
            Constraint(ConstraintKind.TIED, 0, 1),
            Constraint(ConstraintKind.DESTRUCTIVE, 0, 1),
        ),
        asm_forms=_asm(results=("dst",), operands=("acc", "a", "b")),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _f16_literal_immediate(field_name: str = "imm16") -> Immediate:
    return Immediate(
        field_name,
        ImmediateKind.UNSIGNED,
        bit_width=16,
        unsigned_max=0xFFFF,
        encoding_field_id=amdgpu_encoding_field_id("LITERAL"),
    )


def _f16_vgpr_result(field_name: str = "dst") -> Operand:
    return _vgpr_result(field_name, register_part=_REG_PART_VGPR_LOW16)


def _f16_vgpr_operand(field_name: str) -> Operand:
    return _vgpr_operand(field_name, register_part=_REG_PART_VGPR_LOW16)


def _f16_sgpr_result(field_name: str = "dst") -> Operand:
    return _sgpr_result(field_name, register_part=_REG_PART_SGPR_LOW16)


def _f16_sgpr_operand(field_name: str) -> Operand:
    return _sgpr_operand(field_name, register_part=_REG_PART_SGPR_LOW16)


def _s_fmac_f16_overlay() -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.s_fmac_f16",
        instruction_name="S_FMAC_F16",
        mnemonic="s_fmac_f16",
        encoding_name="ENC_SOP2",
        semantic_tag="float.fmac.f16",
        schedule_class=_SCHEDULE_SALU,
        operands=(
            AmdgpuOperandOverlay("SDST", _f16_sgpr_result()),
            AmdgpuOperandOverlay(
                "SDST",
                Operand(
                    "acc",
                    OperandRole.OPERAND,
                    _SGPR_ALT,
                    flags=(OperandFlag.IMPLICIT,),
                    register_part=_REG_PART_SGPR_LOW16,
                ),
                role_exception_reason=(
                    "the encoded low half of the scalar destination register "
                    "is also the tied accumulator input"
                ),
            ),
            AmdgpuOperandOverlay("SSRC0", _f16_sgpr_operand("a")),
            AmdgpuOperandOverlay("SSRC1", _f16_sgpr_operand("b")),
        ),
        constraints=(
            Constraint(ConstraintKind.TIED, 0, 1),
            Constraint(ConstraintKind.DESTRUCTIVE, 0, 1),
        ),
        asm_forms=_asm(results=("dst",), operands=("acc", "a", "b")),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_fma_f16_overlay() -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.v_fma_f16",
        instruction_name="V_FMA_F16",
        mnemonic="v_fma_f16",
        encoding_name="ENC_VOP3",
        semantic_tag="float.fma.f16",
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _f16_vgpr_result()),
            AmdgpuOperandOverlay("SRC0", _f16_vgpr_operand("a")),
            AmdgpuOperandOverlay("SRC1", _f16_vgpr_operand("b")),
            AmdgpuOperandOverlay("SRC2", _f16_vgpr_operand("c")),
        ),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_fmac_f16_overlay() -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.v_fmac_f16",
        instruction_name="V_FMAC_F16",
        mnemonic="v_fmac_f16",
        encoding_name="ENC_VOP2",
        semantic_tag="float.fmac.f16",
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _f16_vgpr_result()),
            AmdgpuOperandOverlay(
                "VDST",
                Operand(
                    "acc",
                    OperandRole.OPERAND,
                    _VGPR_ALT,
                    flags=(OperandFlag.IMPLICIT,),
                    register_part=_REG_PART_VGPR_LOW16,
                ),
                role_exception_reason=(
                    "the encoded destination half-register is also the tied "
                    "accumulator input"
                ),
            ),
            AmdgpuOperandOverlay("SRC0", _f16_vgpr_operand("a")),
            AmdgpuOperandOverlay("VSRC1", _f16_vgpr_operand("b")),
        ),
        constraints=(
            Constraint(ConstraintKind.TIED, 0, 1),
            Constraint(ConstraintKind.DESTRUCTIVE, 0, 1),
        ),
        asm_forms=_asm(results=("dst",), operands=("acc", "a", "b")),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_fmaak_f16_overlay() -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.v_fmaak_f16",
        instruction_name="V_FMAAK_F16",
        mnemonic="v_fmaak_f16",
        encoding_name="VOP2_INST_LITERAL",
        encoding_condition="default",
        semantic_tag="float.fmaak.f16",
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _f16_vgpr_result()),
            AmdgpuOperandOverlay("SRC0", _f16_vgpr_operand("a")),
            AmdgpuOperandOverlay("VSRC1", _f16_vgpr_operand("b")),
        ),
        asm_forms=_asm(
            results=("dst",),
            operands=("a", "b"),
            immediates=("imm16",),
            native_assembly_values=(
                _native_result("dst"),
                _native_operand("a"),
                _native_operand("b"),
                _native_unsigned_hex_immediate("imm16", 16),
            ),
        ),
        immediate_fields=("LITERAL",),
        immediates=(_f16_literal_immediate(),),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_fmamk_f16_overlay() -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.v_fmamk_f16",
        instruction_name="V_FMAMK_F16",
        mnemonic="v_fmamk_f16",
        encoding_name="VOP2_INST_LITERAL",
        encoding_condition="default",
        semantic_tag="float.fmamk.f16",
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _f16_vgpr_result()),
            AmdgpuOperandOverlay("SRC0", _f16_vgpr_operand("a")),
            AmdgpuOperandOverlay("VSRC1", _f16_vgpr_operand("c")),
        ),
        asm_forms=_asm(
            results=("dst",),
            operands=("a", "c"),
            immediates=("imm16",),
            native_assembly_values=(
                _native_result("dst"),
                _native_operand("a"),
                _native_unsigned_hex_immediate("imm16", 16),
                _native_operand("c"),
            ),
        ),
        immediate_fields=("LITERAL",),
        immediates=(_f16_literal_immediate(),),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_mad_f16_overlay() -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.v_mad_f16",
        instruction_name="V_MAD_F16",
        mnemonic="v_mad_f16",
        encoding_name="ENC_VOP3",
        semantic_tag="float.mad.f16",
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _f16_vgpr_result()),
            AmdgpuOperandOverlay("SRC0", _f16_vgpr_operand("a")),
            AmdgpuOperandOverlay("SRC1", _f16_vgpr_operand("b")),
            AmdgpuOperandOverlay("SRC2", _f16_vgpr_operand("c")),
        ),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_mac_f16_overlay() -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.v_mac_f16",
        instruction_name="V_MAC_F16",
        mnemonic="v_mac_f16",
        encoding_name="ENC_VOP2",
        semantic_tag="float.mac.f16",
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _f16_vgpr_result()),
            AmdgpuOperandOverlay(
                "VDST",
                Operand(
                    "acc",
                    OperandRole.OPERAND,
                    _VGPR_ALT,
                    flags=(OperandFlag.IMPLICIT,),
                    register_part=_REG_PART_VGPR_LOW16,
                ),
                role_exception_reason=(
                    "the encoded destination half-register is also the tied "
                    "accumulator input"
                ),
            ),
            AmdgpuOperandOverlay("SRC0", _f16_vgpr_operand("a")),
            AmdgpuOperandOverlay("VSRC1", _f16_vgpr_operand("b")),
        ),
        constraints=(
            Constraint(ConstraintKind.TIED, 0, 1),
            Constraint(ConstraintKind.DESTRUCTIVE, 0, 1),
        ),
        asm_forms=_asm(results=("dst",), operands=("acc", "a", "b")),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_madak_f16_overlay() -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.v_madak_f16",
        instruction_name="V_MADAK_F16",
        mnemonic="v_madak_f16",
        encoding_name="VOP2_INST_LITERAL",
        encoding_condition="default",
        semantic_tag="float.madak.f16",
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _f16_vgpr_result()),
            AmdgpuOperandOverlay("SRC0", _f16_vgpr_operand("a")),
            AmdgpuOperandOverlay("VSRC1", _f16_vgpr_operand("b")),
        ),
        asm_forms=_asm(
            results=("dst",),
            operands=("a", "b"),
            immediates=("imm16",),
            native_assembly_values=(
                _native_result("dst"),
                _native_operand("a"),
                _native_operand("b"),
                _native_unsigned_hex_immediate("imm16", 16),
            ),
        ),
        immediate_fields=("LITERAL",),
        immediates=(_f16_literal_immediate(),),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_madmk_f16_overlay() -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.v_madmk_f16",
        instruction_name="V_MADMK_F16",
        mnemonic="v_madmk_f16",
        encoding_name="VOP2_INST_LITERAL",
        encoding_condition="default",
        semantic_tag="float.madmk.f16",
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _f16_vgpr_result()),
            AmdgpuOperandOverlay("SRC0", _f16_vgpr_operand("a")),
            AmdgpuOperandOverlay("VSRC1", _f16_vgpr_operand("c")),
        ),
        asm_forms=_asm(
            results=("dst",),
            operands=("a", "c"),
            immediates=("imm16",),
            native_assembly_values=(
                _native_result("dst"),
                _native_operand("a"),
                _native_unsigned_hex_immediate("imm16", 16),
                _native_operand("c"),
            ),
        ),
        immediate_fields=("LITERAL",),
        immediates=(_f16_literal_immediate(),),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_fma_f64_overlay() -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.v_fma_f64",
        instruction_name="V_FMA_F64",
        mnemonic="v_fma_f64",
        encoding_name="ENC_VOP3",
        semantic_tag="float.fma.f64",
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result(units=2)),
            AmdgpuOperandOverlay("SRC0", _sgpr_vgpr_operand("a", units=2)),
            AmdgpuOperandOverlay("SRC1", _sgpr_vgpr_operand("b", units=2)),
            AmdgpuOperandOverlay("SRC2", _sgpr_vgpr_operand("c", units=2)),
        ),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_fmac_f64_overlay() -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.v_fmac_f64",
        instruction_name="V_FMAC_F64",
        mnemonic="v_fmac_f64",
        encoding_name="ENC_VOP2",
        semantic_tag="float.fmac.f64",
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result(units=2)),
            AmdgpuOperandOverlay(
                "VDST",
                Operand(
                    "acc",
                    OperandRole.OPERAND,
                    _VGPR_ALT,
                    flags=(OperandFlag.IMPLICIT,),
                    unit_count=2,
                ),
                role_exception_reason=(
                    "the encoded destination register pair is also the tied "
                    "accumulator input"
                ),
            ),
            AmdgpuOperandOverlay("SRC0", _sgpr_vgpr_operand("a", units=2)),
            AmdgpuOperandOverlay("VSRC1", _vgpr_operand("b", units=2)),
        ),
        constraints=(
            Constraint(ConstraintKind.TIED, 0, 1),
            Constraint(ConstraintKind.DESTRUCTIVE, 0, 1),
        ),
        asm_forms=_asm(results=("dst",), operands=("acc", "a", "b")),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_pk_ternary_overlay(
    *,
    descriptor_key: str,
    instruction_name: str,
    mnemonic: str,
    semantic_tag: str,
    units: int = 1,
    include_literal_forms: bool = False,
) -> AmdgpuDescriptorOverlay:
    operand_forms: tuple[OperandForm, ...] = ()
    if include_literal_forms:
        operand_forms = tuple(
            _literal_operand_form(
                replacement_descriptor=f"{descriptor_key}.{source_name}_lit",
                source_operand=source_operand,
            )
            for source_name, _, source_operand, _ in _V_PK_TERNARY_SOURCES
        )
    return AmdgpuDescriptorOverlay(
        descriptor_key=descriptor_key,
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        encoding_name="ENC_VOP3P",
        semantic_tag=semantic_tag,
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result(units=units)),
            AmdgpuOperandOverlay("SRC0", _sgpr_vgpr_operand("a", units=units)),
            AmdgpuOperandOverlay("SRC1", _sgpr_vgpr_operand("b", units=units)),
            AmdgpuOperandOverlay("SRC2", _sgpr_vgpr_operand("c", units=units)),
        ),
        operand_forms=operand_forms,
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


_V_PK_TERNARY_SOURCES = (
    ("src0", "SRC0", "a", _sgpr_vgpr_operand),
    ("src1", "SRC1", "b", _sgpr_vgpr_operand),
    ("src2", "SRC2", "c", _sgpr_vgpr_operand),
)


def _v_pk_ternary_literal_overlay(
    *,
    descriptor_key: str,
    instruction_name: str,
    mnemonic: str,
    semantic_tag: str,
    literal_source: str,
    units: int = 1,
) -> AmdgpuDescriptorOverlay:
    literal_field = ""
    operands = [AmdgpuOperandOverlay("VDST", _vgpr_result(units=units))]
    asm_operands = []
    for (
        source_name,
        xml_field_name,
        operand_name,
        operand_builder,
    ) in _V_PK_TERNARY_SOURCES:
        if source_name == literal_source:
            literal_field = xml_field_name
            continue
        asm_operands.append(operand_name)
        operands.append(
            AmdgpuOperandOverlay(
                xml_field_name, operand_builder(operand_name, units=units)
            )
        )
    if not literal_field:
        raise ValueError(f"unknown packed VOP3P literal source '{literal_source}'")
    return AmdgpuDescriptorOverlay(
        descriptor_key=f"{descriptor_key}.{literal_source}_lit",
        instruction_name=instruction_name,
        mnemonic=f"{mnemonic}_{literal_source}_lit",
        encoding_name="ENC_VOP3P",
        encoding_format_id=AMDGPU_ENCODING_FORMAT_VOP3P_LITERAL,
        semantic_tag=semantic_tag,
        schedule_class=_SCHEDULE_VALU,
        operands=tuple(operands),
        asm_forms=_asm(
            results=("dst",),
            operands=tuple(asm_operands),
            immediates=("imm32",),
        ),
        immediates=(_LITERAL_U32_IMMEDIATE,),
        fixed_encoding_fields=((literal_field, _predefined("SRC_LITERAL", "OPR_SRC")),),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_pk_ternary_literal_overlays(
    *,
    descriptor_key: str,
    instruction_name: str,
    mnemonic: str,
    semantic_tag: str,
    units: int = 1,
) -> tuple[AmdgpuDescriptorOverlay, ...]:
    return tuple(
        _v_pk_ternary_literal_overlay(
            descriptor_key=descriptor_key,
            instruction_name=instruction_name,
            mnemonic=mnemonic,
            semantic_tag=semantic_tag,
            literal_source=source_name,
            units=units,
        )
        for source_name, _, _, _ in _V_PK_TERNARY_SOURCES
    )


def _v_pk_fma_f16_overlay(
    *,
    include_literal_forms: bool = False,
) -> AmdgpuDescriptorOverlay:
    return _v_pk_ternary_overlay(
        descriptor_key="amdgpu.v_pk_fma_f16",
        instruction_name="V_PK_FMA_F16",
        mnemonic="v_pk_fma_f16",
        semantic_tag="float.fma.pk2.f16",
        include_literal_forms=include_literal_forms,
    )


def _v_pk_fma_f16_literal_overlays() -> tuple[AmdgpuDescriptorOverlay, ...]:
    return _v_pk_ternary_literal_overlays(
        descriptor_key="amdgpu.v_pk_fma_f16",
        instruction_name="V_PK_FMA_F16",
        mnemonic="v_pk_fma_f16",
        semantic_tag="float.fma.pk2.f16",
    )


def _v_pk_fmac_f16_overlay() -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.v_pk_fmac_f16",
        instruction_name="V_PK_FMAC_F16",
        mnemonic="v_pk_fmac_f16",
        encoding_name="ENC_VOP2",
        semantic_tag="float.fmac.pk2.f16",
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            AmdgpuOperandOverlay(
                "VDST",
                Operand(
                    "acc",
                    OperandRole.OPERAND,
                    _VGPR_ALT,
                    flags=(OperandFlag.IMPLICIT,),
                ),
                role_exception_reason=(
                    "the encoded destination register is also the tied packed "
                    "accumulator input"
                ),
            ),
            AmdgpuOperandOverlay("SRC0", _sgpr_vgpr_operand("a")),
            AmdgpuOperandOverlay("VSRC1", _vgpr_operand("b")),
        ),
        constraints=(
            Constraint(ConstraintKind.TIED, 0, 1),
            Constraint(ConstraintKind.DESTRUCTIVE, 0, 1),
        ),
        asm_forms=_asm(results=("dst",), operands=("acc", "a", "b")),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_pk_mad_i16_overlay(
    *,
    include_literal_forms: bool = False,
) -> AmdgpuDescriptorOverlay:
    return _v_pk_ternary_overlay(
        descriptor_key="amdgpu.v_pk_mad_i16",
        instruction_name="V_PK_MAD_I16",
        mnemonic="v_pk_mad_i16",
        semantic_tag="integer.mad.pk2.i16",
        include_literal_forms=include_literal_forms,
    )


def _v_pk_mad_i16_literal_overlays() -> tuple[AmdgpuDescriptorOverlay, ...]:
    return _v_pk_ternary_literal_overlays(
        descriptor_key="amdgpu.v_pk_mad_i16",
        instruction_name="V_PK_MAD_I16",
        mnemonic="v_pk_mad_i16",
        semantic_tag="integer.mad.pk2.i16",
    )


def _v_pk_mad_u16_overlay(
    *,
    include_literal_forms: bool = False,
) -> AmdgpuDescriptorOverlay:
    return _v_pk_ternary_overlay(
        descriptor_key="amdgpu.v_pk_mad_u16",
        instruction_name="V_PK_MAD_U16",
        mnemonic="v_pk_mad_u16",
        semantic_tag="integer.mad.pk2.u16",
        include_literal_forms=include_literal_forms,
    )


def _v_pk_mad_u16_literal_overlays() -> tuple[AmdgpuDescriptorOverlay, ...]:
    return _v_pk_ternary_literal_overlays(
        descriptor_key="amdgpu.v_pk_mad_u16",
        instruction_name="V_PK_MAD_U16",
        mnemonic="v_pk_mad_u16",
        semantic_tag="integer.mad.pk2.u16",
    )


def _v_pk_fma_f32_overlay() -> AmdgpuDescriptorOverlay:
    return _v_pk_ternary_overlay(
        descriptor_key="amdgpu.v_pk_fma_f32",
        instruction_name="V_PK_FMA_F32",
        mnemonic="v_pk_fma_f32",
        semantic_tag="float.fma.pk2.f32",
        units=2,
    )


def _v_unary_f32_overlay(
    *,
    descriptor_key: str,
    instruction_name: str,
    mnemonic: str,
    semantic_tag: str,
    schedule_class: str = _SCHEDULE_VALU,
) -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key=descriptor_key,
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        encoding_name="ENC_VOP1",
        semantic_tag=semantic_tag,
        schedule_class=schedule_class,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            AmdgpuOperandOverlay("SRC0", _sgpr_vgpr_operand("input")),
        ),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_trans_unary_f32_overlay(
    *,
    descriptor_key: str,
    instruction_name: str,
    mnemonic: str,
    semantic_tag: str,
) -> AmdgpuDescriptorOverlay:
    return _v_unary_f32_overlay(
        descriptor_key=descriptor_key,
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        semantic_tag=semantic_tag,
        schedule_class=_amdgpu_trans_schedule_class_name(descriptor_key),
    )


def _v_exp_f32_overlay() -> AmdgpuDescriptorOverlay:
    return _v_trans_unary_f32_overlay(
        descriptor_key="amdgpu.v_exp_f32",
        instruction_name="V_EXP_F32",
        mnemonic="v_exp_f32",
        semantic_tag="float.exp2.f32",
    )


def _v_log_f32_overlay() -> AmdgpuDescriptorOverlay:
    return _v_trans_unary_f32_overlay(
        descriptor_key="amdgpu.v_log_f32",
        instruction_name="V_LOG_F32",
        mnemonic="v_log_f32",
        semantic_tag="float.log2.f32",
    )


def _v_sin_f32_overlay() -> AmdgpuDescriptorOverlay:
    return _v_trans_unary_f32_overlay(
        descriptor_key="amdgpu.v_sin_f32",
        instruction_name="V_SIN_F32",
        mnemonic="v_sin_f32",
        semantic_tag="float.sin_turns.f32",
    )


def _v_cos_f32_overlay() -> AmdgpuDescriptorOverlay:
    return _v_trans_unary_f32_overlay(
        descriptor_key="amdgpu.v_cos_f32",
        instruction_name="V_COS_F32",
        mnemonic="v_cos_f32",
        semantic_tag="float.cos_turns.f32",
    )


def _v_floor_f32_overlay() -> AmdgpuDescriptorOverlay:
    return _v_unary_f32_overlay(
        descriptor_key="amdgpu.v_floor_f32",
        instruction_name="V_FLOOR_F32",
        mnemonic="v_floor_f32",
        semantic_tag="float.floor.f32",
    )


def _v_ceil_f32_overlay() -> AmdgpuDescriptorOverlay:
    return _v_unary_f32_overlay(
        descriptor_key="amdgpu.v_ceil_f32",
        instruction_name="V_CEIL_F32",
        mnemonic="v_ceil_f32",
        semantic_tag="float.ceil.f32",
    )


def _v_rndne_f32_overlay() -> AmdgpuDescriptorOverlay:
    return _v_unary_f32_overlay(
        descriptor_key="amdgpu.v_rndne_f32",
        instruction_name="V_RNDNE_F32",
        mnemonic="v_rndne_f32",
        semantic_tag="float.round_even.f32",
    )


def _v_trunc_f32_overlay() -> AmdgpuDescriptorOverlay:
    return _v_unary_f32_overlay(
        descriptor_key="amdgpu.v_trunc_f32",
        instruction_name="V_TRUNC_F32",
        mnemonic="v_trunc_f32",
        semantic_tag="float.trunc.f32",
    )


def _v_native_f32_math_overlays() -> tuple[AmdgpuDescriptorOverlay, ...]:
    return (
        _v_exp_f32_overlay(),
        _v_log_f32_overlay(),
        _v_sin_f32_overlay(),
        _v_cos_f32_overlay(),
        _v_floor_f32_overlay(),
        _v_ceil_f32_overlay(),
        _v_rndne_f32_overlay(),
        _v_trunc_f32_overlay(),
    )


def _v_sqrt_f32_overlay() -> AmdgpuDescriptorOverlay:
    return _v_trans_unary_f32_overlay(
        descriptor_key="amdgpu.v_sqrt_f32",
        instruction_name="V_SQRT_F32",
        mnemonic="v_sqrt_f32",
        semantic_tag="float.sqrt.f32",
    )


def _v_rsq_f32_overlay() -> AmdgpuDescriptorOverlay:
    return _v_trans_unary_f32_overlay(
        descriptor_key="amdgpu.v_rsq_f32",
        instruction_name="V_RSQ_F32",
        mnemonic="v_rsq_f32",
        semantic_tag="float.rsqrt.f32",
    )


def _v_rcp_f32_overlay() -> AmdgpuDescriptorOverlay:
    return _v_trans_unary_f32_overlay(
        descriptor_key="amdgpu.v_rcp_f32",
        instruction_name="V_RCP_F32",
        mnemonic="v_rcp_f32",
        semantic_tag="float.reciprocal.f32",
    )


def _v_div_scale_f32_overlay() -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.v_div_scale_f32",
        instruction_name="V_DIV_SCALE_F32",
        mnemonic="v_div_scale_f32",
        encoding_name="VOP3_SDST_ENC",
        semantic_tag="float.div.scale.f32",
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            AmdgpuOperandOverlay("SDST", _sgpr_result("mask", units=2)),
            AmdgpuOperandOverlay("SRC0", _sgpr_vgpr_operand("value")),
            AmdgpuOperandOverlay("SRC1", _sgpr_vgpr_operand("denominator")),
            AmdgpuOperandOverlay("SRC2", _sgpr_vgpr_operand("numerator")),
        ),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_div_fmas_f32_overlay() -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.v_div_fmas_f32",
        instruction_name="V_DIV_FMAS_F32",
        mnemonic="v_div_fmas_f32",
        encoding_name="ENC_VOP3",
        semantic_tag="float.div.fmas.f32",
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            AmdgpuOperandOverlay("SRC0", _sgpr_vgpr_operand("a")),
            AmdgpuOperandOverlay("SRC1", _sgpr_vgpr_operand("b")),
            AmdgpuOperandOverlay("SRC2", _sgpr_vgpr_operand("c")),
        ),
        implicit_operands=(_vcc_input(_vcc_predicate("scale_mask")),),
        asm_forms=_asm(
            results=("dst",),
            operands=("a", "b", "c", "scale_mask"),
        ),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_div_fixup_f32_overlay() -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.v_div_fixup_f32",
        instruction_name="V_DIV_FIXUP_F32",
        mnemonic="v_div_fixup_f32",
        encoding_name="ENC_VOP3",
        semantic_tag="float.div.fixup.f32",
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            AmdgpuOperandOverlay("SRC0", _sgpr_vgpr_operand("quotient")),
            AmdgpuOperandOverlay("SRC1", _sgpr_vgpr_operand("denominator")),
            AmdgpuOperandOverlay("SRC2", _sgpr_vgpr_operand("numerator")),
        ),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_cvt_f32_i32_overlay() -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.v_cvt_f32_i32",
        instruction_name="V_CVT_F32_I32",
        mnemonic="v_cvt_f32_i32",
        encoding_name="ENC_VOP1",
        semantic_tag="convert.signed.i32.f32",
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            AmdgpuOperandOverlay("SRC0", _sgpr_vgpr_operand("input")),
        ),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_cvt_i32_f32_overlay() -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.v_cvt_i32_f32",
        instruction_name="V_CVT_I32_F32",
        mnemonic="v_cvt_i32_f32",
        encoding_name="ENC_VOP1",
        semantic_tag="convert.float.f32.signed.i32",
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            AmdgpuOperandOverlay("SRC0", _sgpr_vgpr_operand("input")),
        ),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_cvt_f32_f16_overlay() -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.v_cvt_f32_f16",
        instruction_name="V_CVT_F32_F16",
        mnemonic="v_cvt_f32_f16",
        encoding_name="ENC_VOP1",
        semantic_tag="convert.float.f16.f32",
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            AmdgpuOperandOverlay(
                "SRC0", _vgpr_operand("input", register_part=_REG_PART_VGPR_LOW16)
            ),
        ),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_cvt_f16_f32_overlay() -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.v_cvt_f16_f32",
        instruction_name="V_CVT_F16_F32",
        mnemonic="v_cvt_f16_f32",
        encoding_name="ENC_VOP1",
        semantic_tag="convert.float.f32.f16",
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay(
                "VDST", _vgpr_result(register_part=_REG_PART_VGPR_LOW16)
            ),
            AmdgpuOperandOverlay("SRC0", _sgpr_vgpr_operand("input")),
        ),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_cvt_pk_u16_u32_overlay() -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.v_cvt_pk_u16_u32",
        instruction_name="V_CVT_PK_U16_U32",
        mnemonic="v_cvt_pk_u16_u32",
        encoding_name="ENC_VOP3",
        semantic_tag="convert.pack.u32.u16x2",
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            AmdgpuOperandOverlay("SRC0", _sgpr_vgpr_operand("low")),
            AmdgpuOperandOverlay("SRC1", _sgpr_vgpr_operand("high")),
        ),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_cvt_pk_bf16_f32_overlay() -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.v_cvt_pk_bf16_f32",
        instruction_name="V_CVT_PK_BF16_F32",
        mnemonic="v_cvt_pk_bf16_f32",
        encoding_name="ENC_VOP3",
        semantic_tag="convert.float.f32.bf16x2",
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            AmdgpuOperandOverlay("SRC0", _sgpr_vgpr_operand("low")),
            AmdgpuOperandOverlay("SRC1", _sgpr_vgpr_operand("high")),
        ),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_cvt_f32_u32_overlay() -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.v_cvt_f32_u32",
        instruction_name="V_CVT_F32_U32",
        mnemonic="v_cvt_f32_u32",
        encoding_name="ENC_VOP1",
        semantic_tag="convert.unsigned.u32.f32",
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            AmdgpuOperandOverlay("SRC0", _sgpr_vgpr_operand("input")),
        ),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_cvt_u32_f32_overlay() -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.v_cvt_u32_f32",
        instruction_name="V_CVT_U32_F32",
        mnemonic="v_cvt_u32_f32",
        encoding_name="ENC_VOP1",
        semantic_tag="convert.float.f32.unsigned.u32",
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            AmdgpuOperandOverlay("SRC0", _sgpr_vgpr_operand("input")),
        ),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_cmp_i32_overlay(
    *, predicate: str, instruction_suffix: str, semantic_suffix: str
) -> AmdgpuDescriptorOverlay:
    instruction_predicate = instruction_suffix.lower()
    descriptor_key = f"amdgpu.v_cmp_{predicate}_i32"
    return AmdgpuDescriptorOverlay(
        descriptor_key=descriptor_key,
        instruction_name=f"V_CMP_{instruction_suffix}_I32",
        mnemonic=f"v_cmp_{instruction_predicate}_i32",
        encoding_name="ENC_VOP3",
        semantic_tag=f"cmp.i32.{semantic_suffix}",
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _sgpr_result("mask", units=2)),
            AmdgpuOperandOverlay("SRC0", _vgpr_const_operand("lhs")),
            AmdgpuOperandOverlay("SRC1", _vgpr_const_operand("rhs")),
        ),
        operand_forms=_v_cmp_inline_operand_forms(descriptor_key),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_cmp_u32_overlay(
    *, predicate: str, instruction_suffix: str, semantic_suffix: str
) -> AmdgpuDescriptorOverlay:
    instruction_predicate = instruction_suffix.lower()
    descriptor_key = f"amdgpu.v_cmp_{predicate}_u32"
    return AmdgpuDescriptorOverlay(
        descriptor_key=descriptor_key,
        instruction_name=f"V_CMP_{instruction_suffix}_U32",
        mnemonic=f"v_cmp_{instruction_predicate}_u32",
        encoding_name="ENC_VOP3",
        semantic_tag=f"cmp.u32.{semantic_suffix}",
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _sgpr_result("mask", units=2)),
            AmdgpuOperandOverlay("SRC0", _vgpr_const_operand("lhs")),
            AmdgpuOperandOverlay("SRC1", _vgpr_const_operand("rhs")),
        ),
        operand_forms=_v_cmp_inline_operand_forms(descriptor_key),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_cmp_inline_operand_forms(descriptor_key: str) -> tuple[OperandForm, ...]:
    return (
        _literal_operand_form(
            replacement_descriptor=f"{descriptor_key}.src0_inline",
            source_operand="lhs",
            immediate_field="lhs",
        ),
        _literal_operand_form(
            replacement_descriptor=f"{descriptor_key}.src1_inline",
            source_operand="rhs",
            immediate_field="rhs",
        ),
    )


def _v_cmp_source_inline_overlay(
    *,
    predicate: str,
    instruction_suffix: str,
    semantic_suffix: str,
    type_suffix: str,
    literal_source: str,
    immediate: Immediate,
) -> AmdgpuDescriptorOverlay:
    source_fields = {
        "src0": ("SRC0", "lhs", _vgpr_const_operand("lhs")),
        "src1": ("SRC1", "rhs", _vgpr_const_operand("rhs")),
    }
    literal_field, literal_operand, _ = source_fields[literal_source]
    remaining_operands = [
        (xml_field, field_name, operand)
        for source_name, (xml_field, field_name, operand) in source_fields.items()
        if source_name != literal_source
    ]
    descriptor_key = f"amdgpu.v_cmp_{predicate}_{type_suffix}.{literal_source}_inline"
    instruction_predicate = instruction_suffix.lower()
    return AmdgpuDescriptorOverlay(
        descriptor_key=descriptor_key,
        instruction_name=f"V_CMP_{instruction_suffix}_{type_suffix.upper()}",
        mnemonic=f"v_cmp_{instruction_predicate}_{type_suffix}",
        encoding_name="ENC_VOP3",
        semantic_tag=f"cmp.{type_suffix}.{semantic_suffix}",
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _sgpr_result("mask", units=2)),
            *(
                AmdgpuOperandOverlay(xml_field, operand)
                for xml_field, _, operand in remaining_operands
            ),
        ),
        asm_forms=_asm(
            mnemonic=f"v_cmp_{instruction_predicate}_{type_suffix}_{literal_source}_inline",
            results=("mask",),
            operands=tuple(field_name for _, field_name, _ in remaining_operands),
            immediates=(literal_operand,),
            named_immediates=True,
        ),
        immediate_fields=(literal_field,),
        immediates=(replace(immediate, field_name=literal_operand),),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_cmp_i32_source_overlays(
    *, predicate: str, instruction_suffix: str, semantic_suffix: str
) -> tuple[AmdgpuDescriptorOverlay, ...]:
    return tuple(
        _v_cmp_source_inline_overlay(
            predicate=predicate,
            instruction_suffix=instruction_suffix,
            semantic_suffix=semantic_suffix,
            type_suffix="i32",
            literal_source=literal_source,
            immediate=_SOURCE_INLINE_U32_IMMEDIATE,
        )
        for literal_source in ("src0", "src1")
    )


def _v_cmp_u32_source_overlays(
    *, predicate: str, instruction_suffix: str, semantic_suffix: str
) -> tuple[AmdgpuDescriptorOverlay, ...]:
    return tuple(
        _v_cmp_source_inline_overlay(
            predicate=predicate,
            instruction_suffix=instruction_suffix,
            semantic_suffix=semantic_suffix,
            type_suffix="u32",
            literal_source=literal_source,
            immediate=_SOURCE_INLINE_U32_IMMEDIATE,
        )
        for literal_source in ("src0", "src1")
    )


def _v_cmp_f32_overlay(
    *, predicate: str, instruction_suffix: str, semantic_suffix: str
) -> AmdgpuDescriptorOverlay:
    instruction_predicate = instruction_suffix.lower()
    return AmdgpuDescriptorOverlay(
        descriptor_key=f"amdgpu.v_cmp_{predicate}_f32",
        instruction_name=f"V_CMP_{instruction_suffix}_F32",
        mnemonic=f"v_cmp_{instruction_predicate}_f32",
        encoding_name="ENC_VOP3",
        semantic_tag=f"cmp.f32.{semantic_suffix}",
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _sgpr_result("mask", units=2)),
            AmdgpuOperandOverlay("SRC0", _vgpr_const_operand("lhs")),
            AmdgpuOperandOverlay("SRC1", _vgpr_const_operand("rhs")),
        ),
        operand_forms=_v_cmp_inline_operand_forms(f"amdgpu.v_cmp_{predicate}_f32"),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_cmp_f32_source_overlays(
    *, predicate: str, instruction_suffix: str, semantic_suffix: str
) -> tuple[AmdgpuDescriptorOverlay, ...]:
    return tuple(
        _v_cmp_source_inline_overlay(
            predicate=predicate,
            instruction_suffix=instruction_suffix,
            semantic_suffix=semantic_suffix,
            type_suffix="f32",
            literal_source=literal_source,
            immediate=_SOURCE_INLINE_F32_IMMEDIATE,
        )
        for literal_source in ("src0", "src1")
    )


def _v_cmp_overlays() -> tuple[AmdgpuDescriptorOverlay, ...]:
    return (
        _v_cmp_i32_overlay(
            predicate="eq", instruction_suffix="EQ", semantic_suffix="eq"
        ),
        *_v_cmp_i32_source_overlays(
            predicate="eq", instruction_suffix="EQ", semantic_suffix="eq"
        ),
        _v_cmp_i32_overlay(
            predicate="ne", instruction_suffix="NE", semantic_suffix="ne"
        ),
        *_v_cmp_i32_source_overlays(
            predicate="ne", instruction_suffix="NE", semantic_suffix="ne"
        ),
        _v_cmp_i32_overlay(
            predicate="slt", instruction_suffix="LT", semantic_suffix="slt"
        ),
        *_v_cmp_i32_source_overlays(
            predicate="slt", instruction_suffix="LT", semantic_suffix="slt"
        ),
        _v_cmp_i32_overlay(
            predicate="sle", instruction_suffix="LE", semantic_suffix="sle"
        ),
        *_v_cmp_i32_source_overlays(
            predicate="sle", instruction_suffix="LE", semantic_suffix="sle"
        ),
        _v_cmp_i32_overlay(
            predicate="sgt", instruction_suffix="GT", semantic_suffix="sgt"
        ),
        *_v_cmp_i32_source_overlays(
            predicate="sgt", instruction_suffix="GT", semantic_suffix="sgt"
        ),
        _v_cmp_i32_overlay(
            predicate="sge", instruction_suffix="GE", semantic_suffix="sge"
        ),
        *_v_cmp_i32_source_overlays(
            predicate="sge", instruction_suffix="GE", semantic_suffix="sge"
        ),
        _v_cmp_u32_overlay(
            predicate="ult", instruction_suffix="LT", semantic_suffix="ult"
        ),
        *_v_cmp_u32_source_overlays(
            predicate="ult", instruction_suffix="LT", semantic_suffix="ult"
        ),
        _v_cmp_u32_overlay(
            predicate="ule", instruction_suffix="LE", semantic_suffix="ule"
        ),
        *_v_cmp_u32_source_overlays(
            predicate="ule", instruction_suffix="LE", semantic_suffix="ule"
        ),
        _v_cmp_u32_overlay(
            predicate="ugt", instruction_suffix="GT", semantic_suffix="ugt"
        ),
        *_v_cmp_u32_source_overlays(
            predicate="ugt", instruction_suffix="GT", semantic_suffix="ugt"
        ),
        _v_cmp_u32_overlay(
            predicate="uge", instruction_suffix="GE", semantic_suffix="uge"
        ),
        *_v_cmp_u32_source_overlays(
            predicate="uge", instruction_suffix="GE", semantic_suffix="uge"
        ),
        _v_cmp_f32_overlay(
            predicate="oeq", instruction_suffix="EQ", semantic_suffix="oeq"
        ),
        *_v_cmp_f32_source_overlays(
            predicate="oeq", instruction_suffix="EQ", semantic_suffix="oeq"
        ),
        _v_cmp_f32_overlay(
            predicate="ogt", instruction_suffix="GT", semantic_suffix="ogt"
        ),
        *_v_cmp_f32_source_overlays(
            predicate="ogt", instruction_suffix="GT", semantic_suffix="ogt"
        ),
        _v_cmp_f32_overlay(
            predicate="oge", instruction_suffix="GE", semantic_suffix="oge"
        ),
        *_v_cmp_f32_source_overlays(
            predicate="oge", instruction_suffix="GE", semantic_suffix="oge"
        ),
        _v_cmp_f32_overlay(
            predicate="olt", instruction_suffix="LT", semantic_suffix="olt"
        ),
        *_v_cmp_f32_source_overlays(
            predicate="olt", instruction_suffix="LT", semantic_suffix="olt"
        ),
        _v_cmp_f32_overlay(
            predicate="ole", instruction_suffix="LE", semantic_suffix="ole"
        ),
        *_v_cmp_f32_source_overlays(
            predicate="ole", instruction_suffix="LE", semantic_suffix="ole"
        ),
        _v_cmp_f32_overlay(
            predicate="one", instruction_suffix="LG", semantic_suffix="one"
        ),
        *_v_cmp_f32_source_overlays(
            predicate="one", instruction_suffix="LG", semantic_suffix="one"
        ),
        _v_cmp_f32_overlay(
            predicate="ord", instruction_suffix="O", semantic_suffix="ord"
        ),
        *_v_cmp_f32_source_overlays(
            predicate="ord", instruction_suffix="O", semantic_suffix="ord"
        ),
        _v_cmp_f32_overlay(
            predicate="ueq", instruction_suffix="NLG", semantic_suffix="ueq"
        ),
        *_v_cmp_f32_source_overlays(
            predicate="ueq", instruction_suffix="NLG", semantic_suffix="ueq"
        ),
        _v_cmp_f32_overlay(
            predicate="ugt", instruction_suffix="NLE", semantic_suffix="ugt"
        ),
        *_v_cmp_f32_source_overlays(
            predicate="ugt", instruction_suffix="NLE", semantic_suffix="ugt"
        ),
        _v_cmp_f32_overlay(
            predicate="uge", instruction_suffix="NLT", semantic_suffix="uge"
        ),
        *_v_cmp_f32_source_overlays(
            predicate="uge", instruction_suffix="NLT", semantic_suffix="uge"
        ),
        _v_cmp_f32_overlay(
            predicate="ult", instruction_suffix="NGE", semantic_suffix="ult"
        ),
        *_v_cmp_f32_source_overlays(
            predicate="ult", instruction_suffix="NGE", semantic_suffix="ult"
        ),
        _v_cmp_f32_overlay(
            predicate="ule", instruction_suffix="NGT", semantic_suffix="ule"
        ),
        *_v_cmp_f32_source_overlays(
            predicate="ule", instruction_suffix="NGT", semantic_suffix="ule"
        ),
        _v_cmp_f32_overlay(
            predicate="une", instruction_suffix="NEQ", semantic_suffix="une"
        ),
        *_v_cmp_f32_source_overlays(
            predicate="une", instruction_suffix="NEQ", semantic_suffix="une"
        ),
        _v_cmp_f32_overlay(
            predicate="uno", instruction_suffix="U", semantic_suffix="uno"
        ),
        *_v_cmp_f32_source_overlays(
            predicate="uno", instruction_suffix="U", semantic_suffix="uno"
        ),
    )


def _v_cndmask_b32_overlay(
    *,
    include_literal_forms: bool = True,
) -> AmdgpuDescriptorOverlay:
    operand_forms = [
        _literal_operand_form(
            replacement_descriptor="amdgpu.v_cndmask_b32.src0_inline",
            source_operand="false_value",
            immediate_field="false_value",
        ),
        _literal_operand_form(
            replacement_descriptor="amdgpu.v_cndmask_b32.src1_inline",
            source_operand="true_value",
            immediate_field="true_value",
        ),
    ]
    if include_literal_forms:
        operand_forms.extend(
            (
                _literal_operand_form(
                    replacement_descriptor="amdgpu.v_cndmask_b32.src0_lit",
                    source_operand="false_value",
                ),
                _literal_operand_form(
                    replacement_descriptor="amdgpu.v_cndmask_b32.src1_lit",
                    source_operand="true_value",
                ),
            )
        )
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.v_cndmask_b32",
        instruction_name="V_CNDMASK_B32",
        mnemonic="v_cndmask_b32",
        encoding_name="ENC_VOP3",
        semantic_tag="select.mask.b32",
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            AmdgpuOperandOverlay("SRC0", _vgpr_const_operand("false_value")),
            AmdgpuOperandOverlay("SRC1", _vgpr_const_operand("true_value")),
            AmdgpuOperandOverlay("SRC2", _sgpr_predicate("mask", units=2)),
        ),
        operand_forms=tuple(operand_forms),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_cndmask_b32_source_inline_overlay(
    literal_source: str,
) -> AmdgpuDescriptorOverlay:
    source_fields = {
        "src0": ("SRC0", "false_value", _vgpr_const_operand("false_value")),
        "src1": ("SRC1", "true_value", _vgpr_const_operand("true_value")),
    }
    literal_field, literal_operand, _ = source_fields[literal_source]
    remaining_operands = [
        (xml_field, field_name, operand)
        for source_name, (xml_field, field_name, operand) in source_fields.items()
        if source_name != literal_source
    ]
    return AmdgpuDescriptorOverlay(
        descriptor_key=f"amdgpu.v_cndmask_b32.{literal_source}_inline",
        instruction_name="V_CNDMASK_B32",
        mnemonic="v_cndmask_b32",
        encoding_name="ENC_VOP3",
        semantic_tag="select.mask.b32",
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            *(
                AmdgpuOperandOverlay(xml_field, operand)
                for xml_field, _, operand in remaining_operands
            ),
            AmdgpuOperandOverlay("SRC2", _sgpr_predicate("mask", units=2)),
        ),
        asm_forms=_asm(
            mnemonic=f"v_cndmask_b32_{literal_source}_inline",
            results=("dst",),
            operands=(
                *(field_name for _, field_name, _ in remaining_operands),
                "mask",
            ),
            immediates=(literal_operand,),
            named_immediates=True,
        ),
        immediate_fields=(literal_field,),
        immediates=(_source_inline_u32_immediate(literal_operand),),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_cndmask_b32_source_literal_overlay(
    literal_source: str,
) -> AmdgpuDescriptorOverlay:
    source_fields = {
        "src0": ("SRC0", "false_value", _vgpr_const_operand("false_value")),
        "src1": ("SRC1", "true_value", _vgpr_const_operand("true_value")),
    }
    literal_field, _, _ = source_fields[literal_source]
    remaining_operands = [
        (xml_field, field_name, operand)
        for source_name, (xml_field, field_name, operand) in source_fields.items()
        if source_name != literal_source
    ]
    return AmdgpuDescriptorOverlay(
        descriptor_key=f"amdgpu.v_cndmask_b32.{literal_source}_lit",
        instruction_name="V_CNDMASK_B32",
        mnemonic=f"v_cndmask_b32_{literal_source}_lit",
        encoding_name="ENC_VOP3",
        encoding_format_id=AMDGPU_ENCODING_FORMAT_VOP3_LITERAL,
        semantic_tag="select.mask.b32",
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            *(
                AmdgpuOperandOverlay(xml_field, operand)
                for xml_field, _, operand in remaining_operands
            ),
            AmdgpuOperandOverlay("SRC2", _sgpr_predicate("mask", units=2)),
        ),
        asm_forms=_asm(
            results=("dst",),
            operands=(
                *(field_name for _, field_name, _ in remaining_operands),
                "mask",
            ),
            immediates=("imm32",),
        ),
        immediates=(_LITERAL_U32_IMMEDIATE,),
        fixed_encoding_fields=((literal_field, _predefined("SRC_LITERAL", "OPR_SRC")),),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_cndmask_b32_literal_inline_overlay(
    literal_source: str,
) -> AmdgpuDescriptorOverlay:
    source_fields = {
        "src0": ("SRC0", "false_value"),
        "src1": ("SRC1", "true_value"),
    }
    inline_source = "src1" if literal_source == "src0" else "src0"
    literal_field, _ = source_fields[literal_source]
    inline_field, inline_operand = source_fields[inline_source]
    return AmdgpuDescriptorOverlay(
        descriptor_key=(
            f"amdgpu.v_cndmask_b32.{literal_source}_lit_{inline_source}_inline"
        ),
        instruction_name="V_CNDMASK_B32",
        mnemonic=f"v_cndmask_b32_{literal_source}_lit_{inline_source}_inline",
        encoding_name="ENC_VOP3",
        encoding_format_id=AMDGPU_ENCODING_FORMAT_VOP3_LITERAL,
        semantic_tag="select.mask.b32",
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            AmdgpuOperandOverlay("SRC2", _sgpr_predicate("mask", units=2)),
        ),
        asm_forms=_asm(
            results=("dst",),
            operands=("mask",),
            immediates=("imm32", inline_operand),
            named_immediates=True,
        ),
        immediate_fields=(inline_field,),
        immediates=(
            _LITERAL_U32_IMMEDIATE,
            replace(
                _source_inline_u32_immediate(inline_operand),
                encoding_field_id=amdgpu_encoding_field_id(inline_field),
            ),
        ),
        fixed_encoding_fields=((literal_field, _predefined("SRC_LITERAL", "OPR_SRC")),),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_cndmask_b32_overlays(
    *,
    include_literal_forms: bool = True,
) -> tuple[AmdgpuDescriptorOverlay, ...]:
    overlays = (
        _v_cndmask_b32_overlay(include_literal_forms=include_literal_forms),
        _v_cndmask_b32_source_inline_overlay("src0"),
        _v_cndmask_b32_source_inline_overlay("src1"),
    )
    if not include_literal_forms:
        return overlays
    return (
        *overlays,
        _v_cndmask_b32_source_literal_overlay("src0"),
        _v_cndmask_b32_source_literal_overlay("src1"),
        _v_cndmask_b32_literal_inline_overlay("src0"),
        _v_cndmask_b32_literal_inline_overlay("src1"),
    )


def _v_mov_b32_literal_overlay() -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.v_mov_b32",
        instruction_name="V_MOV_B32",
        mnemonic="v_mov_b32",
        encoding_name="VOP1_INST_LITERAL",
        encoding_condition="has_lit",
        semantic_tag="integer.const.u32",
        schedule_class=_SCHEDULE_VALU,
        operands=(AmdgpuOperandOverlay("VDST", _vgpr_result()),),
        asm_forms=_asm(results=("dst",), immediates=("imm32",)),
        immediate_fields=("LITERAL",),
        immediates=(_U32_IMMEDIATE,),
        fixed_encoding_fields=(("SRC0", _predefined("SRC_LITERAL", "OPR_SRC")),),
        constraints=(Constraint(ConstraintKind.REMATERIALIZABLE, 0),),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_mov_b32_copy_overlay() -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.v_mov_b32_copy",
        instruction_name="V_MOV_B32",
        mnemonic="v_mov_b32",
        encoding_name="ENC_VOP1",
        semantic_tag="register.copy.b32",
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            AmdgpuOperandOverlay("SRC0", _sgpr_vgpr_operand("src")),
        ),
        asm_forms=_asm(mnemonic="v_mov_b32_copy", results=("dst",), operands=("src",)),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_mov_b32_dpp_overlay(
    *,
    descriptor_key: str,
    encoding_name: str,
    encoding_condition: str,
) -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key=descriptor_key,
        instruction_name="V_MOV_B32",
        mnemonic="v_mov_b32",
        encoding_name=encoding_name,
        encoding_condition=encoding_condition,
        semantic_tag="lane.dpp.b32",
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            AmdgpuOperandOverlay("VSRC0", _vgpr_operand("src")),
        ),
        asm_forms=_asm(
            mnemonic=descriptor_key.removeprefix("amdgpu."),
            results=("dst",),
            operands=("src",),
            immediates=("dpp_ctrl",),
            named_immediates=True,
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
    )


def _v_mov_b32_dpp_legacy_overlay() -> AmdgpuDescriptorOverlay:
    return _v_mov_b32_dpp_overlay(
        descriptor_key="amdgpu.v_mov_b32_dpp",
        encoding_name="VOP1_VOP_DPP",
        encoding_condition="has_dpp",
    )


def _v_mov_b32_dpp16_overlay() -> AmdgpuDescriptorOverlay:
    return _v_mov_b32_dpp_overlay(
        descriptor_key="amdgpu.v_mov_b32_dpp16",
        encoding_name="VOP1_VOP_DPP16",
        encoding_condition="has_dpp16",
    )


def _v_mov_b32_sdwa_overlay() -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.v_mov_b32_sdwa",
        instruction_name="V_MOV_B32",
        mnemonic="v_mov_b32",
        encoding_name="VOP1_VOP_SDWA",
        encoding_condition="has_sdwa",
        semantic_tag="subword.extract.b32",
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            AmdgpuOperandOverlay("VSRC0", _vgpr_operand("src")),
        ),
        asm_forms=_asm(
            mnemonic="v_mov_b32_sdwa",
            results=("dst",),
            operands=("src",),
            immediates=("dst_sel", "dst_unused", "src0_sel", "src0_sext"),
            named_immediates=True,
        ),
        immediate_fields=("DST_SEL", "DST_UNUSED", "SRC0_SEL", "SRC0_SEXT"),
        immediates=(
            _sdwa_selector_immediate("dst_sel"),
            _SDWA_DST_UNUSED_IMMEDIATE,
            _sdwa_selector_immediate("src0_sel"),
            _SDWA_SOURCE_SEXT_IMMEDIATE,
        ),
        fixed_encoding_fields=(
            ("SRC0", 249),
            ("CLAMP", 0),
            ("OMOD", 0),
            ("S0", 0),
            ("S1", 0),
            ("SRC0_ABS", 0),
            ("SRC0_NEG", 0),
            ("SRC1_ABS", 0),
            ("SRC1_NEG", 0),
            ("SRC1_SEL", 0),
            ("SRC1_SEXT", 0),
        ),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


__all__ = (
    "_integer_bitwise_permute_overlays",
    "_integer_bitwise_shift_overlays",
    "_s_add_u32_overlay",
    "_s_add_u32_rhs_inline_overlay",
    "_s_addk_i32_overlay",
    "_s_add_u32_rhs_symbol_rel32_lo_overlay",
    "_s_addc_u32_rhs_symbol_rel32_hi_overlay",
    "_s_addc_u32_overlay",
    "_s_and_b32_overlay",
    "_s_and_b64_overlay",
    "_s_and_saveexec_b64_overlay",
    "_s_ashr_i32_overlay",
    "_s_ashr_i32_rhs_inline_overlay",
    "_s_bfe_b32_literal_overlay",
    "_s_bfe_b32_overlay",
    "_s_binary_u32_overlay",
    "_s_binary_u32_rhs_inline_overlay",
    "_s_binary_u64_overlay",
    "_s_cmp_i32_overlay",
    "_s_cmp_i32_overlays",
    "_s_cmp_u64_overlay",
    "_s_cmp_u64_overlays",
    "_s_cselect_b32_overlay",
    "_s_lshl_b32_overlay",
    "_s_lshl_b32_rhs_inline_overlay",
    "_s_lshl_b64_overlay",
    "_s_lshr_b32_overlay",
    "_s_lshr_b32_rhs_inline_overlay",
    "_s_lshr_b64_overlay",
    "_s_max_i32_overlay",
    "_s_max_u32_overlay",
    "_s_min_i32_overlay",
    "_s_min_u32_overlay",
    "_s_mul_hi_u32_overlay",
    "_s_mul_i32_overlay",
    "_s_mul_i32_rhs_inline_overlay",
    "_s_mulk_i32_overlay",
    "_s_or_b32_overlay",
    "_s_or_b64_overlay",
    "_s_shift_u64_overlay",
    "_s_sub_u32_overlay",
    "_s_sub_u32_rhs_inline_overlay",
    "_s_xor_b32_overlay",
    "_s_xor_b64_overlay",
    "_v_add_co_ci_u32_overlay",
    "_v_add_co_u32_overlay",
    "_v_add_f32_literal_overlay",
    "_v_add_f32_overlay",
    "_v_add_f32_src0_inline_overlay",
    "_v_add_u32_literal_overlay",
    "_v_add_u32_overlay",
    "_v_add_u32_src0_inline_overlay",
    "_v_and_b32_literal_overlay",
    "_v_and_b32_overlay",
    "_v_and_b32_src0_inline_overlay",
    "_v_ashrrev_i32_literal_overlay",
    "_v_ashrrev_i32_overlay",
    "_v_ashrrev_i32_src0_inline_overlay",
    "_v_bfe_offset_immediate",
    "_v_bfe_offset_width_inline_overlay",
    "_v_bfe_u32_offset_0_width_16_low16_overlay",
    "_v_bfe_width_immediate",
    "_v_bfi_b32_src0_literal_overlay",
    "_v_binary_f32_overlay",
    "_v_binary_f32_operand_forms",
    "_v_binary_f32_overlays",
    "_v_binary_literal_overlay",
    "_v_binary_src0_inline_f32_overlay",
    "_v_binary_src0_inline_overlay",
    "_v_binary_u32_overlay",
    "_v_cmp_f32_overlay",
    "_v_cmp_f32_source_overlays",
    "_v_cmp_i32_overlay",
    "_v_cmp_i32_source_overlays",
    "_v_cmp_inline_operand_forms",
    "_v_cmp_overlays",
    "_v_cmp_source_inline_overlay",
    "_v_cmp_u32_overlay",
    "_v_cmp_u32_source_overlays",
    "_v_cndmask_b32_literal_inline_overlay",
    "_v_cndmask_b32_overlay",
    "_v_cndmask_b32_overlays",
    "_v_cndmask_b32_source_inline_overlay",
    "_v_cndmask_b32_source_literal_overlay",
    "_v_cvt_f16_f32_overlay",
    "_v_cvt_f32_f16_overlay",
    "_v_cvt_f32_i32_overlay",
    "_v_cvt_f32_u32_overlay",
    "_v_cvt_i32_f32_overlay",
    "_v_cvt_pk_bf16_f32_overlay",
    "_v_cvt_pk_u16_u32_overlay",
    "_v_cvt_u32_f32_overlay",
    "_v_ceil_f32_overlay",
    "_v_div_fixup_f32_overlay",
    "_v_div_fmas_f32_overlay",
    "_v_div_scale_f32_overlay",
    "_v_cos_f32_overlay",
    "_v_exp_f32_overlay",
    "_v_floor_f32_overlay",
    "_s_fmaak_f32_overlay",
    "_s_fmac_f16_overlay",
    "_s_fmac_f32_overlay",
    "_s_fmamk_f32_overlay",
    "_v_fma_f16_overlay",
    "_v_fmaak_f32_overlay",
    "_v_fmaak_f16_overlay",
    "_v_fma_f32_overlay",
    "_v_fma_f64_overlay",
    "_v_fma_mix_f32_overlay",
    "_v_fma_mix_f32_overlays",
    "_v_fma_mix_f32_src2_literal_overlay",
    "_v_fma_mixhi_f16_overlays",
    "_v_fma_mixlo_f16_overlays",
    "_v_fmac_f16_overlay",
    "_v_fmac_f32_overlay",
    "_v_fmac_f64_overlay",
    "_v_fmamk_f16_overlay",
    "_v_fmamk_f32_overlay",
    "_v_pk_fma_f16_overlay",
    "_v_pk_fma_f16_literal_overlays",
    "_v_pk_fma_f32_overlay",
    "_v_pk_fmac_f16_overlay",
    "_v_pk_mad_i16_overlay",
    "_v_pk_mad_i16_literal_overlays",
    "_v_pk_mad_u16_overlay",
    "_v_pk_mad_u16_literal_overlays",
    "_v_pk_ternary_overlay",
    "_v_perm_b32_overlay",
    "_v_lshl_add_u32_shift_immediate_overlay",
    "_v_lshlrev_b32_literal_overlay",
    "_v_lshlrev_b32_overlay",
    "_v_lshlrev_b32_src0_16_low16_overlay",
    "_v_lshlrev_b32_src0_inline_overlay",
    "_v_lshlrev_b32_vop3_immediate_overlay",
    "_v_lshlrev_b64_overlay",
    "_v_lshrrev_b32_literal_overlay",
    "_v_lshrrev_b32_overlay",
    "_v_lshrrev_b32_src0_inline_overlay",
    "_v_mac_f16_overlay",
    "_v_mad_f16_overlay",
    "_v_madak_f16_overlay",
    "_v_mad_mix_f32_overlay",
    "_v_mad_mix_f32_overlays",
    "_v_mad_mixhi_f16_overlays",
    "_v_mad_mixlo_f16_overlays",
    "_v_madmk_f16_overlay",
    "_v_mad_u32_u24_literal_overlay",
    "_v_mad_u32_u24_overlay",
    "_v_log_f32_overlay",
    "_v_max_f32_literal_overlay",
    "_v_max_f32_overlay",
    "_v_max_f32_src0_inline_overlay",
    "_v_max_i32_overlay",
    "_v_max_u32_overlay",
    "_v_min_f32_literal_overlay",
    "_v_min_f32_overlay",
    "_v_min_f32_src0_inline_overlay",
    "_v_min_i32_overlay",
    "_v_min_u32_overlay",
    "_v_minmax_i32_overlay",
    "_v_mov_b32_copy_overlay",
    "_v_mov_b32_dpp16_overlay",
    "_v_mov_b32_dpp_legacy_overlay",
    "_v_mov_b32_dpp_overlay",
    "_v_mov_b32_literal_overlay",
    "_v_mov_b32_sdwa_overlay",
    "_v_native_f32_math_overlays",
    "_v_mul_f32_literal_overlay",
    "_v_mul_f32_overlay",
    "_v_mul_f32_src0_inline_overlay",
    "_v_mul_hi_u32_overlay",
    "_v_mul_lo_u32_overlay",
    "_v_mul_u32_u24_literal_overlay",
    "_v_mul_u32_u24_overlay",
    "_v_mul_u32_u24_src0_inline_overlay",
    "_v_or_b32_literal_overlay",
    "_v_or_b32_overlay",
    "_v_or_b32_src0_inline_overlay",
    "_v_rcp_f32_overlay",
    "_v_readfirstlane_b32_overlay",
    "_v_rndne_f32_overlay",
    "_v_rsq_f32_overlay",
    "_v_sin_f32_overlay",
    "_v_sqrt_f32_overlay",
    "_v_trunc_f32_overlay",
    "_v_sub_f32_literal_overlay",
    "_v_sub_f32_overlay",
    "_v_sub_f32_src0_inline_overlay",
    "_v_sub_co_ci_u32_overlay",
    "_v_sub_co_u32_overlay",
    "_v_subrev_f32_literal_overlay",
    "_v_subrev_f32_overlay",
    "_v_subrev_f32_overlays",
    "_v_subrev_f32_src0_inline_overlay",
    "_v_sub_u32_overlay",
    "_v_unary_f32_overlay",
    "_v_xor_b32_literal_overlay",
    "_v_xor_b32_overlay",
    "_v_xor_b32_src0_inline_overlay",
)
