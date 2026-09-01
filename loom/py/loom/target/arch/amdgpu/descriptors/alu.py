# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


# ruff: noqa: F403, F405

"""Scalar, vector, conversion, compare, and select descriptor overlays."""

from __future__ import annotations

from .common import *

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

_REMATERIALIZABLE_RESULT_CONSTRAINTS = (Constraint(ConstraintKind.REMATERIALIZABLE, 0),)
_REMATERIALIZABLE_COMMUTABLE_BINARY_CONSTRAINTS = (
    *_REMATERIALIZABLE_RESULT_CONSTRAINTS,
    Constraint(ConstraintKind.COMMUTABLE, 1, 2),
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
        effects=(_PC_RELATIVE_EFFECT,),
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
        effects=(_PC_RELATIVE_EFFECT,),
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
        constraints=_REMATERIALIZABLE_RESULT_CONSTRAINTS,
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
            native_assembly_mnemonic="s_mul_i32",
            results=("dst",),
            operands=("lhs",),
            immediates=("imm32",),
        ),
        immediate_fields=("SSRC1",),
        immediates=(_SOURCE_INLINE_U32_IMMEDIATE,),
        constraints=_REMATERIALIZABLE_RESULT_CONSTRAINTS,
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
        constraints=_REMATERIALIZABLE_RESULT_CONSTRAINTS,
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
            native_assembly_mnemonic=mnemonic,
            results=("dst",),
            operands=("lhs",),
            immediates=("imm32",),
        ),
        immediate_fields=("SSRC1",),
        immediates=(_SOURCE_INLINE_U32_IMMEDIATE,),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _s_binary_u32_literal_overlay(
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
        encoding_format_id=AMDGPU_ENCODING_FORMAT_SOP2_LITERAL,
        semantic_tag=semantic_tag,
        schedule_class=_SCHEDULE_SALU,
        operands=(
            AmdgpuOperandOverlay("SDST", _sgpr_result()),
            AmdgpuOperandOverlay("SSRC0", _sgpr_operand("lhs")),
        ),
        implicit_operands=(_SCC_CLOBBER_OUTPUT,),
        asm_forms=_asm(
            mnemonic=f"{mnemonic}_lit",
            results=("dst",),
            operands=("lhs",),
            immediates=("imm32",),
        ),
        immediate_fields=("LITERAL",),
        immediates=(_LITERAL_U32_IMMEDIATE,),
        fixed_encoding_fields=(("SSRC1", _predefined("SRC_LITERAL", "OPR_SSRC")),),
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


def _s_bcnt1_i32_overlay(
    source_bit_count: int, encoding_condition: str
) -> AmdgpuDescriptorOverlay:
    if source_bit_count not in (32, 64):
        raise ValueError("S_BCNT1_I32 source width must be 32 or 64")
    return AmdgpuDescriptorOverlay(
        descriptor_key=f"amdgpu.s_bcnt1_i32_b{source_bit_count}",
        instruction_name=f"S_BCNT1_I32_B{source_bit_count}",
        mnemonic=f"s_bcnt1_i32_b{source_bit_count}",
        encoding_name="ENC_SOP1",
        encoding_condition=encoding_condition,
        semantic_tag=f"integer.ctpop.u{source_bit_count}",
        schedule_class=_SCHEDULE_SALU,
        operands=(
            AmdgpuOperandOverlay("SDST", _sgpr_result()),
            AmdgpuOperandOverlay(
                "SSRC0", _sgpr_operand("input", units=source_bit_count // 32)
            ),
        ),
        implicit_operands=(_SCC_CLOBBER_OUTPUT,),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _s_ctz_i32_overlay(
    source_bit_count: int,
    encoding_condition: str,
    instruction_name: str,
) -> AmdgpuDescriptorOverlay:
    if source_bit_count not in (32, 64):
        raise ValueError("scalar CTZ source width must be 32 or 64")
    return AmdgpuDescriptorOverlay(
        descriptor_key=f"amdgpu.s_ctz_i32_b{source_bit_count}",
        instruction_name=instruction_name,
        mnemonic=instruction_name.lower(),
        encoding_name="ENC_SOP1",
        encoding_condition=encoding_condition,
        semantic_tag=f"integer.cttz.u{source_bit_count}.native_zero_minus_one",
        schedule_class=_SCHEDULE_SALU,
        operands=(
            AmdgpuOperandOverlay("SDST", _sgpr_result()),
            AmdgpuOperandOverlay(
                "SSRC0", _sgpr_operand("input", units=source_bit_count // 32)
            ),
        ),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_bcnt_u32_b32_overlay() -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.v_bcnt_u32_b32",
        instruction_name="V_BCNT_U32_B32",
        mnemonic="v_bcnt_u32_b32",
        encoding_name="ENC_VOP3",
        semantic_tag="integer.ctpop.accumulate.u32",
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            AmdgpuOperandOverlay("SRC0", _sgpr_vgpr_operand("input")),
            AmdgpuOperandOverlay("SRC1", _sgpr_vgpr_operand("addend")),
        ),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_ctz_i32_b32_overlay(instruction_name: str) -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.v_ctz_i32_b32",
        instruction_name=instruction_name,
        mnemonic=instruction_name.lower(),
        encoding_name="ENC_VOP1",
        semantic_tag="integer.cttz.u32.native_zero_minus_one",
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            AmdgpuOperandOverlay("SRC0", _sgpr_vgpr_operand("input")),
        ),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_bcnt_u32_b32_src1_zero_overlay() -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.v_bcnt_u32_b32.src1_zero",
        instruction_name="V_BCNT_U32_B32",
        mnemonic="v_bcnt_u32_b32_src1_zero",
        encoding_name="ENC_VOP3",
        semantic_tag="integer.ctpop.u32",
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            AmdgpuOperandOverlay("SRC0", _sgpr_vgpr_operand("input")),
        ),
        fixed_encoding_fields=(("SRC1", _predefined("0", "OPR_SRC")),),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _integer_bit_count_overlays(
    sop1_encoding_condition: str = "default",
    *,
    scalar_ctz_instruction_names: tuple[str, str] = (
        "S_FF1_I32_B32",
        "S_FF1_I32_B64",
    ),
    vector_ctz_instruction_name: str = "V_FFBL_B32",
) -> tuple[AmdgpuDescriptorOverlay, ...]:
    return (
        _s_bcnt1_i32_overlay(32, sop1_encoding_condition),
        _s_bcnt1_i32_overlay(64, sop1_encoding_condition),
        _s_ctz_i32_overlay(
            32, sop1_encoding_condition, scalar_ctz_instruction_names[0]
        ),
        _s_ctz_i32_overlay(
            64, sop1_encoding_condition, scalar_ctz_instruction_names[1]
        ),
        _v_bcnt_u32_b32_overlay(),
        _v_bcnt_u32_b32_src1_zero_overlay(),
        _v_ctz_i32_b32_overlay(vector_ctz_instruction_name),
    )


def _rdna_integer_bit_count_overlays(
    sop1_encoding_condition: str,
) -> tuple[AmdgpuDescriptorOverlay, ...]:
    return _integer_bit_count_overlays(
        sop1_encoding_condition,
        scalar_ctz_instruction_names=("S_CTZ_I32_B32", "S_CTZ_I32_B64"),
        vector_ctz_instruction_name="V_CTZ_I32_B32",
    )


def _s_cmp_i32_overlay(
    *,
    descriptor_key: str,
    instruction_name: str,
    mnemonic: str,
    semantic_tag: str,
    src1_inline_descriptor_key: str | None = None,
) -> AmdgpuDescriptorOverlay:
    operand_forms: tuple[OperandForm, ...] = ()
    if src1_inline_descriptor_key is not None:
        operand_forms = (
            _literal_operand_form(
                replacement_descriptor=src1_inline_descriptor_key,
                source_operand="rhs",
                immediate_field="rhs",
            ),
        )
    return AmdgpuDescriptorOverlay(
        descriptor_key=descriptor_key,
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        encoding_name="ENC_SOPC",
        semantic_tag=semantic_tag,
        schedule_class=_SCHEDULE_SALU_COMPARE,
        operands=(
            AmdgpuOperandOverlay("SSRC0", _sgpr_operand("lhs")),
            AmdgpuOperandOverlay("SSRC1", _sgpr_operand("rhs")),
        ),
        implicit_operands=(_scc_output(_scc_result()),),
        asm_forms=_asm(results=("scc",), operands=("lhs", "rhs")),
        operand_forms=operand_forms,
        constraints=(Constraint(ConstraintKind.REMATERIALIZABLE, 0),),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _s_cmp_i32_src1_inline_overlay(
    *,
    descriptor_key: str,
    instruction_name: str,
    mnemonic: str,
    semantic_tag: str,
) -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key=f"{descriptor_key}.src1_inline",
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        encoding_name="ENC_SOPC",
        semantic_tag=semantic_tag,
        schedule_class=_SCHEDULE_SALU_COMPARE,
        operands=(AmdgpuOperandOverlay("SSRC0", _sgpr_operand("lhs")),),
        implicit_operands=(_scc_output(_scc_result()),),
        asm_forms=_asm(
            mnemonic=f"{mnemonic}_src1_inline",
            results=("scc",),
            operands=("lhs",),
            immediates=("rhs",),
            named_immediates=True,
        ),
        immediate_fields=("SSRC1",),
        immediates=(replace(_SOURCE_INLINE_U32_IMMEDIATE, field_name="rhs"),),
        constraints=(Constraint(ConstraintKind.REMATERIALIZABLE, 0),),
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
        schedule_class=_SCHEDULE_SALU_COMPARE,
        operands=(
            AmdgpuOperandOverlay("SSRC0", _sgpr_operand("lhs", units=2)),
            AmdgpuOperandOverlay("SSRC1", _sgpr_operand("rhs", units=2)),
        ),
        implicit_operands=(_scc_output(_scc_result()),),
        asm_forms=_asm(results=("scc",), operands=("lhs", "rhs")),
        operand_forms=(
            _literal_operand_form(
                replacement_descriptor=f"{descriptor_key}.src1_inline",
                source_operand="rhs",
                immediate_field="rhs",
            ),
        ),
        constraints=(Constraint(ConstraintKind.REMATERIALIZABLE, 0),),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _s_cmp_u64_src1_inline_overlay(
    *,
    descriptor_key: str,
    instruction_name: str,
    mnemonic: str,
    semantic_tag: str,
) -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key=f"{descriptor_key}.src1_inline",
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        encoding_name="ENC_SOPC",
        semantic_tag=semantic_tag,
        schedule_class=_SCHEDULE_SALU_COMPARE,
        operands=(AmdgpuOperandOverlay("SSRC0", _sgpr_operand("lhs", units=2)),),
        implicit_operands=(_scc_output(_scc_result()),),
        asm_forms=_asm(
            mnemonic=f"{mnemonic}_src1_inline",
            results=("scc",),
            operands=("lhs",),
            immediates=("rhs",),
            named_immediates=True,
        ),
        immediate_fields=("SSRC1",),
        immediates=(replace(_SOURCE_INLINE_U32_IMMEDIATE, field_name="rhs"),),
        constraints=(Constraint(ConstraintKind.REMATERIALIZABLE, 0),),
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
        constraints=_REMATERIALIZABLE_COMMUTABLE_BINARY_CONSTRAINTS,
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_add_u32_rhs_tied_overlay(
    instruction_name: str,
) -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.v_add_u32.rhs_tied",
        instruction_name=instruction_name,
        mnemonic="v_add_u32_rhs_tied",
        encoding_name="ENC_VOP2",
        semantic_tag="integer.add.u32",
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            AmdgpuOperandOverlay("SRC0", _sgpr_vgpr_operand("lhs")),
            AmdgpuOperandOverlay("VSRC1", _vgpr_operand("rhs")),
        ),
        asm_forms=_asm(
            native_assembly_mnemonic=instruction_name.lower(),
            results=("dst",),
            operands=("lhs", "rhs"),
        ),
        constraints=_destructive_accumulator_constraints(2),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_add_u32_src0_inline_overlay(instruction_name: str) -> AmdgpuDescriptorOverlay:
    return _v_binary_src0_inline_overlay(
        descriptor_key="amdgpu.v_add_u32.src0_inline",
        instruction_name=instruction_name,
        mnemonic="v_add_u32",
        semantic_tag="integer.add.u32",
        constraints=_REMATERIALIZABLE_RESULT_CONSTRAINTS,
    )


def _v_binary_literal_overlay(
    *,
    descriptor_key: str,
    instruction_name: str,
    mnemonic: str,
    semantic_tag: str,
    rhs_name: str = "rhs",
    constraints: tuple[Constraint, ...] = (),
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
        constraints=constraints,
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_binary_src0_inline_overlay(
    *,
    descriptor_key: str,
    instruction_name: str,
    mnemonic: str,
    semantic_tag: str,
    rhs_name: str = "rhs",
    constraints: tuple[Constraint, ...] = (),
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
        constraints=constraints,
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_binary_src0_inline_f32_overlay(
    *,
    descriptor_key: str,
    instruction_name: str,
    mnemonic: str,
    semantic_tag: str,
    rhs_name: str = "rhs",
    constraints: tuple[Constraint, ...] = (),
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
        constraints=constraints,
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


def _v_med3_num_f32_overlay(
    *,
    instruction_name: str = "V_MED3_F32",
    mnemonic: str = "v_med3_f32",
) -> AmdgpuDescriptorOverlay:
    return _v_ternary_float_overlay(
        descriptor_key="amdgpu.v_med3_num_f32",
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        semantic_tag="float.med3_num.f32",
        element_bit_width=32,
        constraints=_REMATERIALIZABLE_RESULT_CONSTRAINTS,
    )


def _v_add_u32_literal_overlay(instruction_name: str) -> AmdgpuDescriptorOverlay:
    return _v_binary_literal_overlay(
        descriptor_key="amdgpu.v_add_u32.lit",
        instruction_name=instruction_name,
        mnemonic="v_add_u32",
        semantic_tag="integer.add.u32",
        constraints=_REMATERIALIZABLE_RESULT_CONSTRAINTS,
    )


def _v_add3_u32_overlay(
    *,
    include_literal_forms: bool = True,
) -> AmdgpuDescriptorOverlay:
    operand_forms: tuple[OperandForm, ...] = ()
    if include_literal_forms:
        operand_forms = (
            _literal_operand_form(
                replacement_descriptor="amdgpu.v_add3_u32.src0_lit",
                source_operand="a",
            ),
            _literal_operand_form(
                replacement_descriptor="amdgpu.v_add3_u32.src1_lit",
                source_operand="b",
            ),
            _literal_operand_form(
                replacement_descriptor="amdgpu.v_add3_u32.src2_lit",
                source_operand="c",
            ),
        )
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.v_add3_u32",
        instruction_name="V_ADD3_U32",
        mnemonic="v_add3_u32",
        encoding_name="ENC_VOP3",
        semantic_tag="integer.add3.u32",
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            AmdgpuOperandOverlay("SRC0", _sgpr_vgpr_operand("a")),
            AmdgpuOperandOverlay("SRC1", _sgpr_vgpr_operand("b")),
            AmdgpuOperandOverlay("SRC2", _sgpr_vgpr_operand("c")),
        ),
        operand_forms=operand_forms,
        constraints=_REMATERIALIZABLE_RESULT_CONSTRAINTS,
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_add3_u32_literal_overlay(literal_source: str) -> AmdgpuDescriptorOverlay:
    source_fields = {
        "src0": ("SRC0", "a", _sgpr_vgpr_operand("a")),
        "src1": ("SRC1", "b", _sgpr_vgpr_operand("b")),
        "src2": ("SRC2", "c", _sgpr_vgpr_operand("c")),
    }
    literal_field = source_fields[literal_source][0]
    operands = [AmdgpuOperandOverlay("VDST", _vgpr_result())]
    asm_operands = []
    for source_name, (xml_field, field_name, operand) in source_fields.items():
        if source_name == literal_source:
            continue
        asm_operands.append(field_name)
        operands.append(AmdgpuOperandOverlay(xml_field, operand))
    return AmdgpuDescriptorOverlay(
        descriptor_key=f"amdgpu.v_add3_u32.{literal_source}_lit",
        instruction_name="V_ADD3_U32",
        mnemonic=f"v_add3_u32_{literal_source}_lit",
        encoding_name="ENC_VOP3",
        encoding_format_id=AMDGPU_ENCODING_FORMAT_VOP3_LITERAL,
        semantic_tag="integer.add3.u32",
        schedule_class=_SCHEDULE_VALU,
        operands=tuple(operands),
        asm_forms=_asm(
            results=("dst",),
            operands=tuple(asm_operands),
            immediates=("imm32",),
        ),
        immediates=(_LITERAL_U32_IMMEDIATE,),
        fixed_encoding_fields=((literal_field, _predefined("SRC_LITERAL", "OPR_SRC")),),
        constraints=_REMATERIALIZABLE_RESULT_CONSTRAINTS,
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
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
        constraints=_REMATERIALIZABLE_RESULT_CONSTRAINTS,
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_sub_u32_lhs_tied_overlay(
    instruction_name: str, mnemonic: str
) -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.v_sub_u32.lhs_tied",
        instruction_name=instruction_name,
        mnemonic=f"{mnemonic}_lhs_tied",
        encoding_name="ENC_VOP2",
        semantic_tag="integer.sub.u32",
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            AmdgpuOperandOverlay("SRC0", _vgpr_operand("lhs")),
            AmdgpuOperandOverlay("VSRC1", _vgpr_operand("rhs")),
        ),
        asm_forms=_asm(
            native_assembly_mnemonic=mnemonic,
            results=("dst",),
            operands=("lhs", "rhs"),
        ),
        constraints=_destructive_accumulator_constraints(1),
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
    constraints: tuple[Constraint, ...] = (),
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
            AmdgpuOperandOverlay("SRC0", _sgpr_vgpr_operand(lhs_name)),
            AmdgpuOperandOverlay("VSRC1", _vgpr_operand(rhs_name)),
        ),
        operand_forms=tuple(operand_forms),
        constraints=constraints,
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
            AmdgpuOperandOverlay("SRC0", _sgpr_vgpr_operand("lhs")),
            AmdgpuOperandOverlay("SRC1", _vgpr_operand("rhs")),
        ),
        constraints=_REMATERIALIZABLE_RESULT_CONSTRAINTS,
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
            AmdgpuOperandOverlay("SRC0", _sgpr_vgpr_operand("lhs")),
            AmdgpuOperandOverlay("SRC1", _vgpr_operand("rhs")),
        ),
        constraints=_REMATERIALIZABLE_RESULT_CONSTRAINTS,
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
        constraints=_REMATERIALIZABLE_RESULT_CONSTRAINTS,
    )


def _v_mul_u32_u24_src0_inline_overlay() -> AmdgpuDescriptorOverlay:
    return _v_binary_src0_inline_overlay(
        descriptor_key="amdgpu.v_mul_u32_u24.src0_inline",
        instruction_name="V_MUL_U32_U24",
        mnemonic="v_mul_u32_u24",
        semantic_tag="integer.mul.lo.u24.u32",
        constraints=_REMATERIALIZABLE_RESULT_CONSTRAINTS,
    )


def _v_mul_u32_u24_literal_overlay() -> AmdgpuDescriptorOverlay:
    return _v_binary_literal_overlay(
        descriptor_key="amdgpu.v_mul_u32_u24.lit",
        instruction_name="V_MUL_U32_U24",
        mnemonic="v_mul_u32_u24",
        semantic_tag="integer.mul.lo.u24.u32",
        constraints=_REMATERIALIZABLE_RESULT_CONSTRAINTS,
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
        constraints=_REMATERIALIZABLE_RESULT_CONSTRAINTS,
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
        constraints=_REMATERIALIZABLE_RESULT_CONSTRAINTS,
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
        constraints=_REMATERIALIZABLE_COMMUTABLE_BINARY_CONSTRAINTS,
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


def _v_readlane_b32_src1_inline_overlay() -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.v_readlane_b32.src1_inline",
        instruction_name="V_READLANE_B32",
        mnemonic="v_readlane_b32_src1_inline",
        encoding_name="ENC_VOP3",
        semantic_tag="lane.read.b32",
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _sgpr_result()),
            AmdgpuOperandOverlay("SRC0", _vgpr_operand("value")),
        ),
        asm_forms=_asm(
            results=("dst",),
            operands=("value",),
            immediates=("lane",),
            native_assembly_values=(
                _native_result("dst"),
                _native_operand("value"),
                _native_i64_immediate("lane"),
            ),
        ),
        immediate_fields=("SRC1",),
        immediates=(replace(_source_inline_u32_immediate("lane"), unsigned_max=63),),
        fixed_encoding_fields=(("SRC2", 0),),
        effects=(_CONVERGENT_EFFECT,),
    )


def _v_readlane_b32_src1_sgpr_overlay() -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.v_readlane_b32.src1_sgpr",
        instruction_name="V_READLANE_B32",
        mnemonic="v_readlane_b32_src1_sgpr",
        encoding_name="ENC_VOP3",
        semantic_tag="lane.read.b32",
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _sgpr_result()),
            AmdgpuOperandOverlay("SRC0", _vgpr_operand("value")),
            AmdgpuOperandOverlay("SRC1", _sgpr_operand("lane")),
        ),
        fixed_encoding_fields=(("SRC2", 0),),
        effects=(_CONVERGENT_EFFECT,),
    )


def _s_and_b32_overlay() -> AmdgpuDescriptorOverlay:
    return _s_binary_u32_overlay(
        descriptor_key="amdgpu.s_and_b32",
        instruction_name="S_AND_B32",
        mnemonic="s_and_b32",
        semantic_tag="integer.and.u32",
    )


def _s_and_b32_rhs_inline_overlay() -> AmdgpuDescriptorOverlay:
    return _s_binary_u32_rhs_inline_overlay(
        descriptor_key="amdgpu.s_and_b32.rhs_inline",
        instruction_name="S_AND_B32",
        mnemonic="s_and_b32",
        semantic_tag="integer.and.u32",
    )


def _s_and_b32_literal_overlay() -> AmdgpuDescriptorOverlay:
    return _s_binary_u32_literal_overlay(
        descriptor_key="amdgpu.s_and_b32.lit",
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
            src1_inline_descriptor_key="amdgpu.s_cmp_lg_i32.src1_inline",
        ),
        _s_cmp_i32_src1_inline_overlay(
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
    cases = (
        (
            "amdgpu.s_cmp_eq_u64",
            "S_CMP_EQ_U64",
            "s_cmp_eq_u64",
            "integer.compare.eq.u64",
        ),
        (
            "amdgpu.s_cmp_lg_u64",
            "S_CMP_LG_U64",
            "s_cmp_lg_u64",
            "integer.compare.ne.u64",
        ),
    )
    return tuple(
        overlay
        for descriptor_key, instruction_name, mnemonic, semantic_tag in cases
        for overlay in (
            _s_cmp_u64_overlay(
                descriptor_key=descriptor_key,
                instruction_name=instruction_name,
                mnemonic=mnemonic,
                semantic_tag=semantic_tag,
            ),
            _s_cmp_u64_src1_inline_overlay(
                descriptor_key=descriptor_key,
                instruction_name=instruction_name,
                mnemonic=mnemonic,
                semantic_tag=semantic_tag,
            ),
        )
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


def _s_lshl_add_u32_overlay(shift: int) -> AmdgpuDescriptorOverlay:
    if shift < 1 or shift > 4:
        raise ValueError("S_LSHL_ADD_U32 shift must be in [1, 4]")
    return AmdgpuDescriptorOverlay(
        descriptor_key=f"amdgpu.s_lshl{shift}_add_u32",
        instruction_name=f"S_LSHL{shift}_ADD_U32",
        mnemonic=f"s_lshl{shift}_add_u32",
        encoding_name="ENC_SOP2",
        semantic_tag=f"integer.lshl{shift}_add.u32",
        schedule_class=_SCHEDULE_SALU,
        operands=(
            AmdgpuOperandOverlay("SDST", _sgpr_result()),
            AmdgpuOperandOverlay("SSRC0", _sgpr_operand("value")),
            AmdgpuOperandOverlay("SSRC1", _sgpr_operand("addend")),
        ),
        implicit_operands=(_SCC_CLOBBER_OUTPUT,),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
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
        constraints=_REMATERIALIZABLE_COMMUTABLE_BINARY_CONSTRAINTS,
    )


def _v_and_b32_src0_inline_overlay() -> AmdgpuDescriptorOverlay:
    return _v_binary_src0_inline_overlay(
        descriptor_key="amdgpu.v_and_b32.src0_inline",
        instruction_name="V_AND_B32",
        mnemonic="v_and_b32",
        semantic_tag="integer.and.u32",
        rhs_name="rhs",
        constraints=_REMATERIALIZABLE_RESULT_CONSTRAINTS,
    )


def _v_and_b32_literal_overlay() -> AmdgpuDescriptorOverlay:
    return _v_binary_literal_overlay(
        descriptor_key="amdgpu.v_and_b32.lit",
        instruction_name="V_AND_B32",
        mnemonic="v_and_b32",
        semantic_tag="integer.and.u32",
        rhs_name="rhs",
        constraints=_REMATERIALIZABLE_RESULT_CONSTRAINTS,
    )


def _v_or_b32_overlay() -> AmdgpuDescriptorOverlay:
    return _v_binary_u32_overlay(
        descriptor_key="amdgpu.v_or_b32",
        instruction_name="V_OR_B32",
        mnemonic="v_or_b32",
        semantic_tag="integer.or.u32",
        src0_inline_descriptor_key="amdgpu.v_or_b32.src0_inline",
        literal_descriptor_key="amdgpu.v_or_b32.lit",
        constraints=_REMATERIALIZABLE_RESULT_CONSTRAINTS,
    )


def _v_or_b32_src0_inline_overlay() -> AmdgpuDescriptorOverlay:
    return _v_binary_src0_inline_overlay(
        descriptor_key="amdgpu.v_or_b32.src0_inline",
        instruction_name="V_OR_B32",
        mnemonic="v_or_b32",
        semantic_tag="integer.or.u32",
        rhs_name="rhs",
        constraints=_REMATERIALIZABLE_RESULT_CONSTRAINTS,
    )


def _v_or_b32_literal_overlay() -> AmdgpuDescriptorOverlay:
    return _v_binary_literal_overlay(
        descriptor_key="amdgpu.v_or_b32.lit",
        instruction_name="V_OR_B32",
        mnemonic="v_or_b32",
        semantic_tag="integer.or.u32",
        rhs_name="rhs",
        constraints=_REMATERIALIZABLE_RESULT_CONSTRAINTS,
    )


def _v_xor_b32_overlay() -> AmdgpuDescriptorOverlay:
    return _v_binary_u32_overlay(
        descriptor_key="amdgpu.v_xor_b32",
        instruction_name="V_XOR_B32",
        mnemonic="v_xor_b32",
        semantic_tag="integer.xor.u32",
        src0_inline_descriptor_key="amdgpu.v_xor_b32.src0_inline",
        literal_descriptor_key="amdgpu.v_xor_b32.lit",
        constraints=_REMATERIALIZABLE_RESULT_CONSTRAINTS,
    )


def _v_xor_b32_src0_inline_overlay() -> AmdgpuDescriptorOverlay:
    return _v_binary_src0_inline_overlay(
        descriptor_key="amdgpu.v_xor_b32.src0_inline",
        instruction_name="V_XOR_B32",
        mnemonic="v_xor_b32",
        semantic_tag="integer.xor.u32",
        rhs_name="rhs",
        constraints=_REMATERIALIZABLE_RESULT_CONSTRAINTS,
    )


def _v_xor_b32_literal_overlay() -> AmdgpuDescriptorOverlay:
    return _v_binary_literal_overlay(
        descriptor_key="amdgpu.v_xor_b32.lit",
        instruction_name="V_XOR_B32",
        mnemonic="v_xor_b32",
        semantic_tag="integer.xor.u32",
        rhs_name="rhs",
        constraints=_REMATERIALIZABLE_RESULT_CONSTRAINTS,
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
        constraints=_REMATERIALIZABLE_RESULT_CONSTRAINTS,
    )


def _v_lshlrev_b32_src0_inline_overlay() -> AmdgpuDescriptorOverlay:
    return _v_binary_src0_inline_overlay(
        descriptor_key="amdgpu.v_lshlrev_b32.src0_inline",
        instruction_name="V_LSHLREV_B32",
        mnemonic="v_lshlrev_b32",
        semantic_tag="integer.shl.u32",
        rhs_name="value",
        constraints=_REMATERIALIZABLE_RESULT_CONSTRAINTS,
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
        constraints=_REMATERIALIZABLE_RESULT_CONSTRAINTS,
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
        constraints=_REMATERIALIZABLE_RESULT_CONSTRAINTS,
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


def _v_lshl_add_u32_shift_immediate_overlay(
    *, include_literal_operand_form: bool = True
) -> AmdgpuDescriptorOverlay:
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
        operand_forms=(
            (
                _literal_operand_form(
                    replacement_descriptor="amdgpu.v_lshl_add_u32.shift_imm.src2_lit",
                    source_operand="addend",
                ),
            )
            if include_literal_operand_form
            else ()
        ),
        immediate_fields=("SRC1",),
        immediates=(_source_inline_u32_immediate("shift"),),
        constraints=_REMATERIALIZABLE_RESULT_CONSTRAINTS,
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_lshl_add_u32_shift_immediate_src2_literal_overlay() -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.v_lshl_add_u32.shift_imm.src2_lit",
        instruction_name="V_LSHL_ADD_U32",
        mnemonic="v_lshl_add_u32_shift_imm_src2_lit",
        encoding_name="ENC_VOP3",
        encoding_format_id=AMDGPU_ENCODING_FORMAT_VOP3_LITERAL,
        semantic_tag="integer.shl.add.u32",
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            AmdgpuOperandOverlay("SRC0", _sgpr_vgpr_operand("value")),
        ),
        asm_forms=_asm(
            mnemonic="v_lshl_add_u32_shift_imm_src2_lit",
            native_assembly_mnemonic="v_lshl_add_u32",
            results=("dst",),
            operands=("value",),
            immediates=("shift", "imm32"),
            named_immediates=True,
            native_assembly_values=(
                _native_result("dst"),
                _native_operand("value"),
                _native_i64_immediate("shift"),
                _native_unsigned_hex_immediate("imm32", 32),
            ),
        ),
        immediate_fields=("SRC1", "LITERAL"),
        immediates=(
            _source_inline_u32_immediate("shift"),
            _LITERAL_U32_IMMEDIATE,
        ),
        fixed_encoding_fields=(("SRC2", _predefined("SRC_LITERAL", "OPR_SRC")),),
        constraints=_REMATERIALIZABLE_RESULT_CONSTRAINTS,
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
        constraints=_REMATERIALIZABLE_RESULT_CONSTRAINTS,
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
        constraints=_REMATERIALIZABLE_RESULT_CONSTRAINTS,
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_bfi_b32_overlay() -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.v_bfi_b32",
        instruction_name="V_BFI_B32",
        mnemonic="v_bfi_b32",
        encoding_name="ENC_VOP3",
        semantic_tag="integer.bitfield.insert.u32",
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            AmdgpuOperandOverlay("SRC0", _sgpr_vgpr_operand("mask")),
            AmdgpuOperandOverlay("SRC1", _sgpr_vgpr_operand("insert")),
            AmdgpuOperandOverlay("SRC2", _sgpr_vgpr_operand("base")),
        ),
        asm_forms=_asm(
            mnemonic="v_bfi_b32",
            results=("dst",),
            operands=("mask", "insert", "base"),
        ),
        constraints=_REMATERIALIZABLE_RESULT_CONSTRAINTS,
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_perm_b32_overlay(
    *, include_literal_operand_form: bool = True
) -> AmdgpuDescriptorOverlay:
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
        operand_forms=(
            (
                _literal_operand_form(
                    replacement_descriptor="amdgpu.v_perm_b32.src2_lit",
                    source_operand="selectors",
                ),
            )
            if include_literal_operand_form
            else ()
        ),
        asm_forms=_asm(
            results=("dst",),
            operands=("src0", "src1", "selectors"),
        ),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_perm_b32_src2_literal_overlay() -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.v_perm_b32.src2_lit",
        instruction_name="V_PERM_B32",
        mnemonic="v_perm_b32_src2_lit",
        encoding_name="ENC_VOP3",
        encoding_format_id=AMDGPU_ENCODING_FORMAT_VOP3_LITERAL,
        semantic_tag="integer.byte.permute.u32",
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            AmdgpuOperandOverlay("SRC0", _sgpr_vgpr_operand("src0")),
            AmdgpuOperandOverlay("SRC1", _sgpr_vgpr_operand("src1")),
        ),
        asm_forms=_asm(
            results=("dst",),
            operands=("src0", "src1"),
            immediates=("imm32",),
        ),
        immediates=(_LITERAL_U32_IMMEDIATE,),
        fixed_encoding_fields=(("SRC2", _predefined("SRC_LITERAL", "OPR_SRC")),),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_perm_b32_src1_zero_src2_literal_overlay() -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.v_perm_b32.src1_zero_src2_lit",
        instruction_name="V_PERM_B32",
        mnemonic="v_perm_b32_src1_zero_src2_lit",
        encoding_name="ENC_VOP3",
        encoding_format_id=AMDGPU_ENCODING_FORMAT_VOP3_LITERAL,
        semantic_tag="integer.byte.permute.u32",
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            AmdgpuOperandOverlay("SRC0", _sgpr_vgpr_operand("src0")),
        ),
        asm_forms=_asm(
            results=("dst",),
            operands=("src0",),
            immediates=("imm32",),
        ),
        immediates=(_LITERAL_U32_IMMEDIATE,),
        fixed_encoding_fields=(
            ("SRC1", _predefined("0", "OPR_SRC")),
            ("SRC2", _predefined("SRC_LITERAL", "OPR_SRC")),
        ),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_permlanex16_b32_src12_inline_overlay() -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.v_permlanex16_b32.src12_inline",
        instruction_name="V_PERMLANEX16_B32",
        mnemonic="v_permlanex16_b32_src12_inline",
        encoding_name="ENC_VOP3",
        semantic_tag="lane.permlanex16.b32",
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            AmdgpuOperandOverlay("SRC0", _vgpr_operand("src")),
        ),
        asm_forms=_asm(
            results=("dst",),
            operands=("src",),
            immediates=("selector_low", "selector_high"),
            native_assembly_values=(
                _native_result("dst"),
                _native_operand("src"),
                _native_i64_immediate("selector_low"),
                _native_i64_immediate("selector_high"),
            ),
        ),
        immediate_fields=("SRC1", "SRC2"),
        immediates=(
            _source_inline_u32_immediate("selector_low"),
            _source_inline_u32_immediate("selector_high"),
        ),
        effects=(_CONVERGENT_EFFECT,),
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
        constraints=_REMATERIALIZABLE_RESULT_CONSTRAINTS,
    )


def _v_lshrrev_b32_src0_inline_overlay() -> AmdgpuDescriptorOverlay:
    return _v_binary_src0_inline_overlay(
        descriptor_key="amdgpu.v_lshrrev_b32.src0_inline",
        instruction_name="V_LSHRREV_B32",
        mnemonic="v_lshrrev_b32",
        semantic_tag="integer.shr.u32",
        rhs_name="value",
        constraints=_REMATERIALIZABLE_RESULT_CONSTRAINTS,
    )


def _v_lshrrev_b32_literal_overlay() -> AmdgpuDescriptorOverlay:
    return _v_binary_literal_overlay(
        descriptor_key="amdgpu.v_lshrrev_b32.lit",
        instruction_name="V_LSHRREV_B32",
        mnemonic="v_lshrrev_b32",
        semantic_tag="integer.shr.u32",
        rhs_name="value",
        constraints=_REMATERIALIZABLE_RESULT_CONSTRAINTS,
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
        constraints=_REMATERIALIZABLE_RESULT_CONSTRAINTS,
    )


def _v_ashrrev_i32_src0_inline_overlay() -> AmdgpuDescriptorOverlay:
    return _v_binary_src0_inline_overlay(
        descriptor_key="amdgpu.v_ashrrev_i32.src0_inline",
        instruction_name="V_ASHRREV_I32",
        mnemonic="v_ashrrev_i32",
        semantic_tag="integer.shr.i32",
        rhs_name="value",
        constraints=_REMATERIALIZABLE_RESULT_CONSTRAINTS,
    )


def _v_ashrrev_i32_literal_overlay() -> AmdgpuDescriptorOverlay:
    return _v_binary_literal_overlay(
        descriptor_key="amdgpu.v_ashrrev_i32.lit",
        instruction_name="V_ASHRREV_I32",
        mnemonic="v_ashrrev_i32",
        semantic_tag="integer.shr.i32",
        rhs_name="value",
        constraints=_REMATERIALIZABLE_RESULT_CONSTRAINTS,
    )


def _integer_bitwise_shift_overlays(
    *, include_vop3_literal_forms: bool = True
) -> tuple[AmdgpuDescriptorOverlay, ...]:
    return (
        _s_and_b32_overlay(),
        _s_and_b32_rhs_inline_overlay(),
        _s_and_b32_literal_overlay(),
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
        *(_s_lshl_add_u32_overlay(shift) for shift in range(1, 5)),
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
        _v_lshl_add_u32_shift_immediate_overlay(
            include_literal_operand_form=include_vop3_literal_forms
        ),
        *(
            (_v_lshl_add_u32_shift_immediate_src2_literal_overlay(),)
            if include_vop3_literal_forms
            else ()
        ),
        _v_bfe_offset_width_inline_overlay(is_signed=False),
        _v_bfe_offset_width_inline_overlay(is_signed=True),
        _v_bfe_u32_offset_0_width_16_low16_overlay(),
        _v_bfi_b32_overlay(),
        *((_v_bfi_b32_src0_literal_overlay(),) if include_vop3_literal_forms else ()),
        _v_lshrrev_b32_overlay(),
        _v_lshrrev_b32_src0_inline_overlay(),
        _v_lshrrev_b32_literal_overlay(),
        _v_ashrrev_i32_overlay(),
        _v_ashrrev_i32_src0_inline_overlay(),
        _v_ashrrev_i32_literal_overlay(),
    )


def _integer_bitwise_permute_overlays(
    *, include_vop3_literal_forms: bool = True
) -> tuple[AmdgpuDescriptorOverlay, ...]:
    return (
        _v_perm_b32_overlay(include_literal_operand_form=include_vop3_literal_forms),
        *(
            (
                _v_perm_b32_src2_literal_overlay(),
                _v_perm_b32_src1_zero_src2_literal_overlay(),
            )
            if include_vop3_literal_forms
            else ()
        ),
    )


def _v_binary_f32_overlay(
    *,
    descriptor_key: str,
    instruction_name: str,
    mnemonic: str,
    semantic_tag: str,
    src0_name: str = "lhs",
    vsrc1_name: str = "rhs",
    constraints: tuple[Constraint, ...] = _REMATERIALIZABLE_RESULT_CONSTRAINTS,
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
        constraints=constraints,
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )
    return replace(
        descriptor,
        operand_forms=_v_binary_f32_operand_forms(descriptor, source_operand=src0_name),
    )


def _v_binary_f16_overlay(
    *,
    descriptor_key: str,
    instruction_name: str,
    mnemonic: str,
    semantic_tag: str,
    constraints: tuple[Constraint, ...] = (),
) -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key=descriptor_key,
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        encoding_name="ENC_VOP2",
        semantic_tag=semantic_tag,
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _f16_vgpr_result()),
            AmdgpuOperandOverlay("SRC0", _f16_vgpr_operand("lhs")),
            AmdgpuOperandOverlay("VSRC1", _f16_vgpr_operand("rhs")),
        ),
        constraints=constraints,
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_commutative_binary_f16_overlay(
    *,
    descriptor_key: str,
    instruction_name: str,
    mnemonic: str,
    semantic_tag: str,
) -> AmdgpuDescriptorOverlay:
    return _v_binary_f16_overlay(
        descriptor_key=descriptor_key,
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        semantic_tag=semantic_tag,
        constraints=_REMATERIALIZABLE_COMMUTABLE_BINARY_CONSTRAINTS,
    )


def _v_binary_f32_dpp_overlay(
    *,
    descriptor_key: str,
    instruction_name: str,
    mnemonic: str,
    semantic_tag: str,
    encoding_name: str,
    encoding_condition: str,
) -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key=descriptor_key,
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        encoding_name=encoding_name,
        encoding_condition=encoding_condition,
        semantic_tag=semantic_tag,
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            AmdgpuOperandOverlay("VSRC0", _vgpr_operand("lhs")),
            AmdgpuOperandOverlay("VSRC1", _vgpr_operand("rhs")),
        ),
        asm_forms=_asm(
            mnemonic=f"{mnemonic}_{descriptor_key.rsplit('.', 1)[1]}",
            native_assembly_mnemonic=(
                None if descriptor_key.endswith(".dpp") else f"{mnemonic}_dpp"
            ),
            results=("dst",),
            operands=("lhs", "rhs"),
            immediates=("dpp_ctrl",),
            named_immediates=True,
            native_assembly_values=(
                _native_result("dst"),
                _native_operand("lhs"),
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
    )


def _v_binary_f32_dpp_variant_overlays(
    *,
    descriptor_suffix: str,
    encoding_name: str,
    encoding_condition: str,
) -> tuple[AmdgpuDescriptorOverlay, ...]:
    return tuple(
        _v_binary_f32_dpp_overlay(
            descriptor_key=f"amdgpu.v_{operation}_f32.{descriptor_suffix}",
            instruction_name=f"V_{operation.upper()}_F32",
            mnemonic=f"v_{operation}_f32",
            semantic_tag=f"float.{semantic}.f32",
            encoding_name=encoding_name,
            encoding_condition=encoding_condition,
        )
        for operation, semantic in (
            ("add", "add"),
            ("mul", "mul"),
            ("min", "minnum"),
            ("max", "maxnum"),
        )
    )


def _v_binary_f32_dpp_legacy_overlays() -> tuple[AmdgpuDescriptorOverlay, ...]:
    return _v_binary_f32_dpp_variant_overlays(
        descriptor_suffix="dpp",
        encoding_name="VOP2_VOP_DPP",
        encoding_condition="has_dpp",
    )


def _v_binary_f32_dpp16_overlays() -> tuple[AmdgpuDescriptorOverlay, ...]:
    return _v_binary_f32_dpp_variant_overlays(
        descriptor_suffix="dpp16",
        encoding_name="VOP2_VOP_DPP16",
        encoding_condition="has_dpp16",
    )


def _v_add_f32_overlay() -> AmdgpuDescriptorOverlay:
    return _v_binary_f32_overlay(
        descriptor_key="amdgpu.v_add_f32",
        instruction_name="V_ADD_F32",
        mnemonic="v_add_f32",
        semantic_tag="float.add.f32",
        constraints=_REMATERIALIZABLE_COMMUTABLE_BINARY_CONSTRAINTS,
    )


def _v_add_f32_literal_overlay() -> AmdgpuDescriptorOverlay:
    return _v_binary_literal_overlay(
        descriptor_key="amdgpu.v_add_f32.lit",
        instruction_name="V_ADD_F32",
        mnemonic="v_add_f32",
        semantic_tag="float.add.f32",
        constraints=_REMATERIALIZABLE_RESULT_CONSTRAINTS,
    )


def _v_add_f32_src0_inline_overlay() -> AmdgpuDescriptorOverlay:
    return _v_binary_src0_inline_f32_overlay(
        descriptor_key="amdgpu.v_add_f32.src0_inline",
        instruction_name="V_ADD_F32",
        mnemonic="v_add_f32",
        semantic_tag="float.add.f32",
        constraints=_REMATERIALIZABLE_RESULT_CONSTRAINTS,
    )


def _v_add_f16_overlay() -> AmdgpuDescriptorOverlay:
    return _v_binary_f16_overlay(
        descriptor_key="amdgpu.v_add_f16",
        instruction_name="V_ADD_F16",
        mnemonic="v_add_f16",
        semantic_tag="float.add.f16",
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
        constraints=_REMATERIALIZABLE_RESULT_CONSTRAINTS,
    )


def _v_sub_f32_src0_inline_overlay() -> AmdgpuDescriptorOverlay:
    return _v_binary_src0_inline_f32_overlay(
        descriptor_key="amdgpu.v_sub_f32.src0_inline",
        instruction_name="V_SUB_F32",
        mnemonic="v_sub_f32",
        semantic_tag="float.sub.f32",
        constraints=_REMATERIALIZABLE_RESULT_CONSTRAINTS,
    )


def _v_sub_f16_overlay() -> AmdgpuDescriptorOverlay:
    return _v_binary_f16_overlay(
        descriptor_key="amdgpu.v_sub_f16",
        instruction_name="V_SUB_F16",
        mnemonic="v_sub_f16",
        semantic_tag="float.sub.f16",
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
        constraints=_REMATERIALIZABLE_RESULT_CONSTRAINTS,
    )


def _v_subrev_f32_src0_inline_overlay() -> AmdgpuDescriptorOverlay:
    return _v_binary_src0_inline_f32_overlay(
        descriptor_key="amdgpu.v_subrev_f32.src0_inline",
        instruction_name="V_SUBREV_F32",
        mnemonic="v_subrev_f32",
        semantic_tag="float.sub.f32",
        rhs_name="lhs",
        constraints=_REMATERIALIZABLE_RESULT_CONSTRAINTS,
    )


def _v_mul_f32_overlay() -> AmdgpuDescriptorOverlay:
    return _v_binary_f32_overlay(
        descriptor_key="amdgpu.v_mul_f32",
        instruction_name="V_MUL_F32",
        mnemonic="v_mul_f32",
        semantic_tag="float.mul.f32",
        constraints=_REMATERIALIZABLE_COMMUTABLE_BINARY_CONSTRAINTS,
    )


def _v_mul_f32_literal_overlay() -> AmdgpuDescriptorOverlay:
    return _v_binary_literal_overlay(
        descriptor_key="amdgpu.v_mul_f32.lit",
        instruction_name="V_MUL_F32",
        mnemonic="v_mul_f32",
        semantic_tag="float.mul.f32",
        constraints=_REMATERIALIZABLE_RESULT_CONSTRAINTS,
    )


def _v_mul_f32_src0_inline_overlay() -> AmdgpuDescriptorOverlay:
    return _v_binary_src0_inline_f32_overlay(
        descriptor_key="amdgpu.v_mul_f32.src0_inline",
        instruction_name="V_MUL_F32",
        mnemonic="v_mul_f32",
        semantic_tag="float.mul.f32",
        constraints=_REMATERIALIZABLE_RESULT_CONSTRAINTS,
    )


def _v_mul_f16_overlay() -> AmdgpuDescriptorOverlay:
    return _v_binary_f16_overlay(
        descriptor_key="amdgpu.v_mul_f16",
        instruction_name="V_MUL_F16",
        mnemonic="v_mul_f16",
        semantic_tag="float.mul.f16",
    )


def _v_min_f32_overlay(
    *,
    instruction_name: str = "V_MIN_F32",
    mnemonic: str = "v_min_f32",
) -> AmdgpuDescriptorOverlay:
    return _v_binary_f32_overlay(
        descriptor_key="amdgpu.v_min_f32",
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        semantic_tag="float.minnum.f32",
        constraints=_REMATERIALIZABLE_COMMUTABLE_BINARY_CONSTRAINTS,
    )


def _v_min_f32_literal_overlay(
    *,
    instruction_name: str = "V_MIN_F32",
    mnemonic: str = "v_min_f32",
) -> AmdgpuDescriptorOverlay:
    return _v_binary_literal_overlay(
        descriptor_key="amdgpu.v_min_f32.lit",
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        semantic_tag="float.minnum.f32",
        constraints=_REMATERIALIZABLE_RESULT_CONSTRAINTS,
    )


def _v_min_f32_src0_inline_overlay(
    *,
    instruction_name: str = "V_MIN_F32",
    mnemonic: str = "v_min_f32",
) -> AmdgpuDescriptorOverlay:
    return _v_binary_src0_inline_f32_overlay(
        descriptor_key="amdgpu.v_min_f32.src0_inline",
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        semantic_tag="float.minnum.f32",
        constraints=_REMATERIALIZABLE_RESULT_CONSTRAINTS,
    )


def _v_max_f32_overlay(
    *,
    instruction_name: str = "V_MAX_F32",
    mnemonic: str = "v_max_f32",
) -> AmdgpuDescriptorOverlay:
    return _v_binary_f32_overlay(
        descriptor_key="amdgpu.v_max_f32",
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        semantic_tag="float.maxnum.f32",
        constraints=_REMATERIALIZABLE_COMMUTABLE_BINARY_CONSTRAINTS,
    )


def _v_max_f32_literal_overlay(
    *,
    instruction_name: str = "V_MAX_F32",
    mnemonic: str = "v_max_f32",
) -> AmdgpuDescriptorOverlay:
    return _v_binary_literal_overlay(
        descriptor_key="amdgpu.v_max_f32.lit",
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        semantic_tag="float.maxnum.f32",
        constraints=_REMATERIALIZABLE_RESULT_CONSTRAINTS,
    )


def _v_max_f32_src0_inline_overlay(
    *,
    instruction_name: str = "V_MAX_F32",
    mnemonic: str = "v_max_f32",
) -> AmdgpuDescriptorOverlay:
    return _v_binary_src0_inline_f32_overlay(
        descriptor_key="amdgpu.v_max_f32.src0_inline",
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        semantic_tag="float.maxnum.f32",
        constraints=_REMATERIALIZABLE_RESULT_CONSTRAINTS,
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


def _v_binary_f16_overlays() -> tuple[AmdgpuDescriptorOverlay, ...]:
    return (
        _v_add_f16_overlay(),
        _v_sub_f16_overlay(),
        _v_mul_f16_overlay(),
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


_V_INTERP_HALF_REGISTER_PARTS = (
    ("lo", _REG_PART_VGPR_LOW16, 0),
    ("hi", _REG_PART_VGPR_HIGH16, 1),
)


def _v_interp_f32_overlays(
    op_sel_field: str,
) -> tuple[AmdgpuDescriptorOverlay, ...]:
    return tuple(
        AmdgpuDescriptorOverlay(
            descriptor_key=f"amdgpu.v_interp_{phase}_f32",
            instruction_name=f"V_INTERP_{phase.upper()}_F32",
            mnemonic=f"v_interp_{phase}_f32",
            encoding_name="ENC_VINTERP",
            semantic_tag=f"float.interpolation.{phase}.f32",
            schedule_class=_SCHEDULE_VALU,
            operands=(
                AmdgpuOperandOverlay("VDST", _vgpr_result()),
                AmdgpuOperandOverlay("SRC0", _vgpr_operand(first_source_name)),
                AmdgpuOperandOverlay("SRC1", _vgpr_operand(coordinate_name)),
                AmdgpuOperandOverlay("SRC2", _vgpr_operand(third_source_name)),
            ),
            implicit_operands=(_implicit_m0_input(xml_operand_required=False),),
            asm_forms=_asm(
                results=("dst",),
                operands=(
                    first_source_name,
                    coordinate_name,
                    third_source_name,
                    "m0",
                ),
                native_assembly_values=(
                    _native_result("dst"),
                    _native_operand(first_source_name),
                    _native_operand(coordinate_name),
                    _native_operand(third_source_name),
                    _native_modifier_literal("wait_exp:7"),
                ),
            ),
            fixed_encoding_fields=(
                ("WAIT_EXP", 7),
                (op_sel_field, 0),
                ("CLAMP", 0),
                ("NEG", 0),
            ),
            flags=(DescriptorFlag.DEAD_REMOVABLE,),
        )
        for phase, first_source_name, coordinate_name, third_source_name in (
            ("p10", "p10", "i", "p0"),
            ("p2", "p20", "j", "p10_result"),
        )
    )


def _v_interp_p10_mixed_overlays(
    instruction_name: str,
    mnemonic: str,
    semantic_tag: str,
    op_sel_field: str,
) -> tuple[AmdgpuDescriptorOverlay, ...]:
    return tuple(
        AmdgpuDescriptorOverlay(
            descriptor_key=(f"amdgpu.{mnemonic}.p10_{p10_part}.p0_{p0_part}"),
            instruction_name=instruction_name,
            mnemonic=f"{mnemonic}_p10_{p10_part}_p0_{p0_part}",
            encoding_name="ENC_VINTERP",
            semantic_tag=semantic_tag,
            schedule_class=_SCHEDULE_VALU,
            operands=(
                AmdgpuOperandOverlay("VDST", _vgpr_result()),
                AmdgpuOperandOverlay(
                    "SRC0", _vgpr_operand("p10", register_part=p10_register_part)
                ),
                AmdgpuOperandOverlay("SRC1", _vgpr_operand("i")),
                AmdgpuOperandOverlay(
                    "SRC2", _vgpr_operand("p0", register_part=p0_register_part)
                ),
            ),
            implicit_operands=(_implicit_m0_input(xml_operand_required=False),),
            asm_forms=_asm(
                native_assembly_mnemonic=mnemonic,
                results=("dst",),
                operands=("p10", "i", "p0", "m0"),
                native_assembly_values=(
                    _native_result("dst"),
                    _native_register_part("p10"),
                    _native_operand("i"),
                    _native_register_part("p0"),
                    _native_modifier_literal("wait_exp:7"),
                ),
            ),
            fixed_encoding_fields=(
                ("WAIT_EXP", 7),
                (op_sel_field, p10_op_sel | (p0_op_sel << 2)),
                ("CLAMP", 0),
                ("NEG", 0),
            ),
            flags=(DescriptorFlag.DEAD_REMOVABLE,),
        )
        for p10_part, p10_register_part, p10_op_sel in _V_INTERP_HALF_REGISTER_PARTS
        for p0_part, p0_register_part, p0_op_sel in _V_INTERP_HALF_REGISTER_PARTS
    )


def _v_interp_p2_mixed_overlays(
    instruction_name: str,
    mnemonic: str,
    semantic_tag: str,
    op_sel_field: str,
) -> tuple[AmdgpuDescriptorOverlay, ...]:
    return tuple(
        AmdgpuDescriptorOverlay(
            descriptor_key=(f"amdgpu.{mnemonic}.p20_{p20_part}.result_{result_part}"),
            instruction_name=instruction_name,
            mnemonic=f"{mnemonic}_p20_{p20_part}_result_{result_part}",
            encoding_name="ENC_VINTERP",
            semantic_tag=semantic_tag,
            schedule_class=_SCHEDULE_VALU,
            operands=(
                AmdgpuOperandOverlay(
                    "VDST", _vgpr_result(register_part=result_register_part)
                ),
                AmdgpuOperandOverlay(
                    "SRC0", _vgpr_operand("p20", register_part=p20_register_part)
                ),
                AmdgpuOperandOverlay("SRC1", _vgpr_operand("j")),
                AmdgpuOperandOverlay("SRC2", _vgpr_operand("p10_result")),
            ),
            implicit_operands=(_implicit_m0_input(xml_operand_required=False),),
            asm_forms=_asm(
                native_assembly_mnemonic=mnemonic,
                results=("dst",),
                operands=("p20", "j", "p10_result", "m0"),
                native_assembly_values=(
                    _native_register_part("dst"),
                    _native_register_part("p20"),
                    _native_operand("j"),
                    _native_operand("p10_result"),
                    _native_modifier_literal("wait_exp:7"),
                ),
            ),
            fixed_encoding_fields=(
                ("WAIT_EXP", 7),
                (op_sel_field, p20_op_sel | (result_op_sel << 3)),
                ("CLAMP", 0),
                ("NEG", 0),
            ),
            flags=(DescriptorFlag.DEAD_REMOVABLE,),
        )
        for p20_part, p20_register_part, p20_op_sel in _V_INTERP_HALF_REGISTER_PARTS
        for (
            result_part,
            result_register_part,
            result_op_sel,
        ) in _V_INTERP_HALF_REGISTER_PARTS
    )


def _v_interp_overlays(
    *, op_sel_field: str = "OP_SEL"
) -> tuple[AmdgpuDescriptorOverlay, ...]:
    return (
        *_v_interp_f32_overlays(op_sel_field),
        *_v_interp_p10_mixed_overlays(
            "V_INTERP_P10_F16_F32",
            "v_interp_p10_f16_f32",
            "float.interpolation.p10.f16_f32",
            op_sel_field,
        ),
        *_v_interp_p2_mixed_overlays(
            "V_INTERP_P2_F16_F32",
            "v_interp_p2_f16_f32",
            "float.interpolation.p2.f16_f32",
            op_sel_field,
        ),
        *_v_interp_p10_mixed_overlays(
            "V_INTERP_P10_RTZ_F16_F32",
            "v_interp_p10_rtz_f16_f32",
            "float.interpolation.p10.rtz.f16_f32",
            op_sel_field,
        ),
        *_v_interp_p2_mixed_overlays(
            "V_INTERP_P2_RTZ_F16_F32",
            "v_interp_p2_rtz_f16_f32",
            "float.interpolation.p2.rtz.f16_f32",
            op_sel_field,
        ),
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
    operand_forms: tuple[OperandForm, ...] = (),
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
        operand_forms=operand_forms,
        fixed_encoding_fields=(
            (op_sel_field, op_sel),
            (op_sel_hi_field, op_sel_hi),
        ),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_mix_half_result_src2_literal_overlay(
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
    if source_parts[2] != "f32":
        raise ValueError("half-result mixed-FMA source-2 literal forms require f32 c")
    result_register_part = {
        "lo": _REG_PART_VGPR_LOW16,
        "hi": _REG_PART_VGPR_HIGH16,
    }[result_part]
    op_sel, op_sel_hi = _v_mix_source_selectors(source_parts)
    suffix = "_".join(source_parts)
    return AmdgpuDescriptorOverlay(
        descriptor_key=f"{descriptor_key_prefix}.{suffix}.src2_lit",
        instruction_name=instruction_name,
        mnemonic=f"{mnemonic_prefix}_{suffix}_src2_lit",
        encoding_name="ENC_VOP3P",
        encoding_format_id=AMDGPU_ENCODING_FORMAT_VOP3P_LITERAL,
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
        ),
        constraints=(
            Constraint(ConstraintKind.TIED, 0, 1),
            Constraint(ConstraintKind.DESTRUCTIVE, 0, 1),
        ),
        asm_forms=_asm(
            results=("dst",),
            operands=("acc", "a", "b"),
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


def _v_fma_mix_half_result_overlays(
    *,
    descriptor_key_prefix: str,
    instruction_name: str,
    mnemonic_prefix: str,
    result_part: str,
    semantic_tag_prefix: str,
    op_sel_field: str,
    op_sel_hi_field: str,
) -> tuple[AmdgpuDescriptorOverlay, ...]:
    overlays = []
    for source_parts in _v_mix_source_combinations(include_all_f32=True):
        suffix = "_".join(source_parts)
        operand_forms: tuple[OperandForm, ...] = ()
        if source_parts[2] == "f32":
            operand_forms = (
                _literal_operand_form(
                    replacement_descriptor=f"{descriptor_key_prefix}.{suffix}.src2_lit",
                    source_operand="c",
                ),
            )
        overlays.append(
            _v_mix_half_result_overlay(
                source_parts,
                descriptor_key_prefix=descriptor_key_prefix,
                instruction_name=instruction_name,
                mnemonic_prefix=mnemonic_prefix,
                result_part=result_part,
                semantic_tag_prefix=semantic_tag_prefix,
                op_sel_field=op_sel_field,
                op_sel_hi_field=op_sel_hi_field,
                operand_forms=operand_forms,
            )
        )
        if source_parts[2] == "f32":
            overlays.append(
                _v_mix_half_result_src2_literal_overlay(
                    source_parts,
                    descriptor_key_prefix=descriptor_key_prefix,
                    instruction_name=instruction_name,
                    mnemonic_prefix=mnemonic_prefix,
                    result_part=result_part,
                    semantic_tag_prefix=semantic_tag_prefix,
                    op_sel_field=op_sel_field,
                    op_sel_hi_field=op_sel_hi_field,
                )
            )
    return tuple(overlays)


def _v_fma_mixlo_f16_overlays(
    *,
    op_sel_field: str = "OPSEL",
    op_sel_hi_field: str = "OPSEL_HI",
) -> tuple[AmdgpuDescriptorOverlay, ...]:
    return _v_fma_mix_half_result_overlays(
        descriptor_key_prefix="amdgpu.v_fma_mixlo_f16",
        instruction_name="V_FMA_MIXLO_F16",
        mnemonic_prefix="v_fma_mixlo_f16",
        result_part="lo",
        semantic_tag_prefix="float.fma.mixlo.f16",
        op_sel_field=op_sel_field,
        op_sel_hi_field=op_sel_hi_field,
    )


def _v_fma_mixhi_f16_overlays(
    *,
    op_sel_field: str = "OPSEL",
    op_sel_hi_field: str = "OPSEL_HI",
) -> tuple[AmdgpuDescriptorOverlay, ...]:
    return _v_fma_mix_half_result_overlays(
        descriptor_key_prefix="amdgpu.v_fma_mixhi_f16",
        instruction_name="V_FMA_MIXHI_F16",
        mnemonic_prefix="v_fma_mixhi_f16",
        result_part="hi",
        semantic_tag_prefix="float.fma.mixhi.f16",
        op_sel_field=op_sel_field,
        op_sel_hi_field=op_sel_hi_field,
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
            Constraint(ConstraintKind.COMMUTABLE, 2, 3),
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
        constraints=(Constraint(ConstraintKind.COMMUTABLE, 1, 2),),
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


def _v_float_vop3_result(element_bit_width: int) -> Operand:
    if element_bit_width == 16:
        return _f16_vgpr_result()
    if element_bit_width in (32, 64):
        return _vgpr_result(units=element_bit_width // 32)
    raise ValueError(f"unsupported VOP3 float width {element_bit_width}")


def _v_float_vop3_operand(field_name: str, element_bit_width: int) -> Operand:
    if element_bit_width == 16:
        return _f16_vgpr_operand(field_name)
    if element_bit_width in (32, 64):
        return _sgpr_vgpr_operand(field_name, units=element_bit_width // 32)
    raise ValueError(f"unsupported VOP3 float width {element_bit_width}")


def _v_binary_vop3_float_overlay(
    *,
    descriptor_key: str,
    instruction_name: str,
    mnemonic: str,
    semantic_tag: str,
    element_bit_width: int,
    constraints: tuple[Constraint, ...] = (),
) -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key=descriptor_key,
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        encoding_name="ENC_VOP3",
        semantic_tag=semantic_tag,
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _v_float_vop3_result(element_bit_width)),
            AmdgpuOperandOverlay(
                "SRC0", _v_float_vop3_operand("lhs", element_bit_width)
            ),
            AmdgpuOperandOverlay(
                "SRC1", _v_float_vop3_operand("rhs", element_bit_width)
            ),
        ),
        constraints=constraints,
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_commutative_binary_vop3_float_overlay(
    *,
    descriptor_key: str,
    instruction_name: str,
    mnemonic: str,
    semantic_tag: str,
    element_bit_width: int,
) -> AmdgpuDescriptorOverlay:
    return _v_binary_vop3_float_overlay(
        descriptor_key=descriptor_key,
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        semantic_tag=semantic_tag,
        element_bit_width=element_bit_width,
        constraints=_REMATERIALIZABLE_COMMUTABLE_BINARY_CONSTRAINTS,
    )


def _v_ternary_float_overlay(
    *,
    descriptor_key: str,
    instruction_name: str,
    mnemonic: str,
    semantic_tag: str,
    element_bit_width: int,
    constraints: tuple[Constraint, ...] = (),
) -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key=descriptor_key,
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        encoding_name="ENC_VOP3",
        semantic_tag=semantic_tag,
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _v_float_vop3_result(element_bit_width)),
            AmdgpuOperandOverlay("SRC0", _v_float_vop3_operand("a", element_bit_width)),
            AmdgpuOperandOverlay("SRC1", _v_float_vop3_operand("b", element_bit_width)),
            AmdgpuOperandOverlay("SRC2", _v_float_vop3_operand("c", element_bit_width)),
        ),
        constraints=constraints,
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


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
    return _v_ternary_float_overlay(
        descriptor_key="amdgpu.v_fma_f16",
        instruction_name="V_FMA_F16",
        mnemonic="v_fma_f16",
        semantic_tag="float.fma.f16",
        element_bit_width=16,
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
        fixed_encoding_fields=(("OP_SEL_HI", 0x7),),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_pk_binary_overlay(
    *,
    descriptor_key: str,
    instruction_name: str,
    mnemonic: str,
    semantic_tag: str,
    lhs_name: str = "lhs",
    rhs_name: str = "rhs",
    units: int = 1,
) -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key=descriptor_key,
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        encoding_name="ENC_VOP3P",
        semantic_tag=semantic_tag,
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result(units=units)),
            AmdgpuOperandOverlay("SRC0", _sgpr_vgpr_operand(lhs_name, units=units)),
            AmdgpuOperandOverlay("SRC1", _sgpr_vgpr_operand(rhs_name, units=units)),
        ),
        fixed_encoding_fields=(("OP_SEL_HI", 0x7),),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_pk_with_op_sel_hi_field(
    overlays: tuple[AmdgpuDescriptorOverlay, ...], op_sel_hi_field: str
) -> tuple[AmdgpuDescriptorOverlay, ...]:
    rewritten_overlays = []
    for overlay in overlays:
        replaced_field = False
        fixed_encoding_fields = []
        for field, value in overlay.fixed_encoding_fields:
            if field == "OP_SEL_HI":
                fixed_encoding_fields.append((op_sel_hi_field, value))
                replaced_field = True
            else:
                fixed_encoding_fields.append((field, value))
        if not replaced_field:
            raise ValueError(
                f"packed overlay '{overlay.descriptor_key}' has no OP_SEL_HI field"
            )
        rewritten_overlays.append(
            replace(overlay, fixed_encoding_fields=tuple(fixed_encoding_fields))
        )
    return tuple(rewritten_overlays)


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
        fixed_encoding_fields=(
            ("OP_SEL_HI", 0x7),
            (literal_field, _predefined("SRC_LITERAL", "OPR_SRC")),
        ),
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


def _v_pk_add_f16_overlay() -> AmdgpuDescriptorOverlay:
    return _v_pk_binary_overlay(
        descriptor_key="amdgpu.v_pk_add_f16",
        instruction_name="V_PK_ADD_F16",
        mnemonic="v_pk_add_f16",
        semantic_tag="float.add.pk2.f16",
    )


def _v_pk_minnum_f16_overlay(
    *,
    instruction_name: str = "V_PK_MIN_F16",
    mnemonic: str = "v_pk_min_f16",
) -> AmdgpuDescriptorOverlay:
    return _v_pk_binary_overlay(
        descriptor_key="amdgpu.v_pk_minnum_f16",
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        semantic_tag="float.minnum.pk2.f16",
    )


def _v_pk_maxnum_f16_overlay(
    *,
    instruction_name: str = "V_PK_MAX_F16",
    mnemonic: str = "v_pk_max_f16",
) -> AmdgpuDescriptorOverlay:
    return _v_pk_binary_overlay(
        descriptor_key="amdgpu.v_pk_maxnum_f16",
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        semantic_tag="float.maxnum.pk2.f16",
    )


def _v_pk_minimum_f16_overlay() -> AmdgpuDescriptorOverlay:
    return _v_pk_binary_overlay(
        descriptor_key="amdgpu.v_pk_minimum_f16",
        instruction_name="V_PK_MINIMUM_F16",
        mnemonic="v_pk_minimum_f16",
        semantic_tag="float.minimum.pk2.f16",
    )


def _v_pk_maximum_f16_overlay() -> AmdgpuDescriptorOverlay:
    return _v_pk_binary_overlay(
        descriptor_key="amdgpu.v_pk_maximum_f16",
        instruction_name="V_PK_MAXIMUM_F16",
        mnemonic="v_pk_maximum_f16",
        semantic_tag="float.maximum.pk2.f16",
    )


def _v_pk_mul_f16_overlay() -> AmdgpuDescriptorOverlay:
    return _v_pk_binary_overlay(
        descriptor_key="amdgpu.v_pk_mul_f16",
        instruction_name="V_PK_MUL_F16",
        mnemonic="v_pk_mul_f16",
        semantic_tag="float.mul.pk2.f16",
    )


def _v_pk_mul_bf16_overlay() -> AmdgpuDescriptorOverlay:
    return _v_pk_binary_overlay(
        descriptor_key="amdgpu.v_pk_mul_bf16",
        instruction_name="V_PK_MUL_BF16",
        mnemonic="v_pk_mul_bf16",
        semantic_tag="float.mul.pk2.bf16",
    )


def _v_pk_add_bf16_overlay() -> AmdgpuDescriptorOverlay:
    return _v_pk_binary_overlay(
        descriptor_key="amdgpu.v_pk_add_bf16",
        instruction_name="V_PK_ADD_BF16",
        mnemonic="v_pk_add_bf16",
        semantic_tag="float.add.pk2.bf16",
    )


def _v_pk_fma_bf16_overlay() -> AmdgpuDescriptorOverlay:
    return _v_pk_ternary_overlay(
        descriptor_key="amdgpu.v_pk_fma_bf16",
        instruction_name="V_PK_FMA_BF16",
        mnemonic="v_pk_fma_bf16",
        semantic_tag="float.fma.pk2.bf16",
    )


def _v_pk_add_u16_overlay() -> AmdgpuDescriptorOverlay:
    return _v_pk_binary_overlay(
        descriptor_key="amdgpu.v_pk_add_u16",
        instruction_name="V_PK_ADD_U16",
        mnemonic="v_pk_add_u16",
        semantic_tag="integer.add.pk2.u16",
    )


def _v_pk_sub_i16_overlay() -> AmdgpuDescriptorOverlay:
    return _v_pk_binary_overlay(
        descriptor_key="amdgpu.v_pk_sub_i16",
        instruction_name="V_PK_SUB_I16",
        mnemonic="v_pk_sub_i16",
        semantic_tag="integer.sub.pk2.i16",
    )


def _v_pk_mul_lo_u16_overlay() -> AmdgpuDescriptorOverlay:
    return _v_pk_binary_overlay(
        descriptor_key="amdgpu.v_pk_mul_lo_u16",
        instruction_name="V_PK_MUL_LO_U16",
        mnemonic="v_pk_mul_lo_u16",
        semantic_tag="integer.mul.lo.pk2.u16",
    )


def _v_pk_min_i16_overlay() -> AmdgpuDescriptorOverlay:
    return _v_pk_binary_overlay(
        descriptor_key="amdgpu.v_pk_min_i16",
        instruction_name="V_PK_MIN_I16",
        mnemonic="v_pk_min_i16",
        semantic_tag="integer.min.pk2.i16",
    )


def _v_pk_max_i16_overlay() -> AmdgpuDescriptorOverlay:
    return _v_pk_binary_overlay(
        descriptor_key="amdgpu.v_pk_max_i16",
        instruction_name="V_PK_MAX_I16",
        mnemonic="v_pk_max_i16",
        semantic_tag="integer.max.pk2.i16",
    )


def _v_pk_min_u16_overlay() -> AmdgpuDescriptorOverlay:
    return _v_pk_binary_overlay(
        descriptor_key="amdgpu.v_pk_min_u16",
        instruction_name="V_PK_MIN_U16",
        mnemonic="v_pk_min_u16",
        semantic_tag="integer.min.pk2.u16",
    )


def _v_pk_max_u16_overlay() -> AmdgpuDescriptorOverlay:
    return _v_pk_binary_overlay(
        descriptor_key="amdgpu.v_pk_max_u16",
        instruction_name="V_PK_MAX_U16",
        mnemonic="v_pk_max_u16",
        semantic_tag="integer.max.pk2.u16",
    )


def _v_pk_lshlrev_b16_overlay() -> AmdgpuDescriptorOverlay:
    return _v_pk_binary_overlay(
        descriptor_key="amdgpu.v_pk_lshlrev_b16",
        instruction_name="V_PK_LSHLREV_B16",
        mnemonic="v_pk_lshlrev_b16",
        semantic_tag="integer.shl.pk2.u16",
        lhs_name="shift",
        rhs_name="value",
    )


def _v_pk_lshrrev_b16_overlay() -> AmdgpuDescriptorOverlay:
    return _v_pk_binary_overlay(
        descriptor_key="amdgpu.v_pk_lshrrev_b16",
        instruction_name="V_PK_LSHRREV_B16",
        mnemonic="v_pk_lshrrev_b16",
        semantic_tag="integer.shr.pk2.u16",
        lhs_name="shift",
        rhs_name="value",
    )


def _v_pk_ashrrev_i16_overlay() -> AmdgpuDescriptorOverlay:
    return _v_pk_binary_overlay(
        descriptor_key="amdgpu.v_pk_ashrrev_i16",
        instruction_name="V_PK_ASHRREV_I16",
        mnemonic="v_pk_ashrrev_i16",
        semantic_tag="integer.shr.pk2.i16",
        lhs_name="shift",
        rhs_name="value",
    )


def _v_pk_i16_binary_overlays() -> tuple[AmdgpuDescriptorOverlay, ...]:
    return (
        _v_pk_add_u16_overlay(),
        _v_pk_sub_i16_overlay(),
        _v_pk_mul_lo_u16_overlay(),
        _v_pk_min_i16_overlay(),
        _v_pk_max_i16_overlay(),
        _v_pk_min_u16_overlay(),
        _v_pk_max_u16_overlay(),
        _v_pk_lshlrev_b16_overlay(),
        _v_pk_lshrrev_b16_overlay(),
        _v_pk_ashrrev_i16_overlay(),
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


def _v_pk_add_f32_overlay() -> AmdgpuDescriptorOverlay:
    return _v_pk_binary_overlay(
        descriptor_key="amdgpu.v_pk_add_f32",
        instruction_name="V_PK_ADD_F32",
        mnemonic="v_pk_add_f32",
        semantic_tag="float.add.pk2.f32",
        units=2,
    )


def _v_pk_mul_f32_overlay() -> AmdgpuDescriptorOverlay:
    return _v_pk_binary_overlay(
        descriptor_key="amdgpu.v_pk_mul_f32",
        instruction_name="V_PK_MUL_F32",
        mnemonic="v_pk_mul_f32",
        semantic_tag="float.mul.pk2.f32",
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
            AmdgpuOperandOverlay("SRC0", _sgpr_vgpr_operand("value")),
            AmdgpuOperandOverlay("SRC1", _sgpr_vgpr_operand("denominator")),
            AmdgpuOperandOverlay("SRC2", _sgpr_vgpr_operand("numerator")),
        ),
        ignored_operands=(
            AmdgpuIgnoredOperandOverlay(
                "SDST",
                ignore_reason="fixed-architectural-vcc-scale-mask",
                fixed_encoding_value=_predefined("VCC_LO", "OPR_SDST"),
            ),
        ),
        implicit_operands=(
            _vcc_output(_vcc_result("mask"), xml_operand_required=False),
        ),
        asm_forms=_asm(
            results=("dst", "mask"),
            operands=("value", "denominator", "numerator"),
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
        constraints=_REMATERIALIZABLE_RESULT_CONSTRAINTS,
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
        constraints=_REMATERIALIZABLE_RESULT_CONSTRAINTS,
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_cvt_f32_f16_overlay(
    *, encoding_name: str = "ENC_VOP1"
) -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.v_cvt_f32_f16",
        instruction_name="V_CVT_F32_F16",
        mnemonic="v_cvt_f32_f16",
        encoding_name=encoding_name,
        semantic_tag="convert.float.f16.f32",
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            AmdgpuOperandOverlay(
                "SRC0", _vgpr_operand("input", register_part=_REG_PART_VGPR_LOW16)
            ),
        ),
        constraints=_REMATERIALIZABLE_RESULT_CONSTRAINTS,
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
                "VDST",
                _vgpr_result(
                    register_part=_REG_PART_VGPR_LOW16,
                    address_map_kind=OperandAddressMapKind.LOW_SUBSET,
                    addressable_unit_count=_D16_PARTIAL_REGISTER_ADDRESSABLE_UNIT_COUNT,
                ),
            ),
            AmdgpuOperandOverlay("SRC0", _sgpr_vgpr_operand("input")),
        ),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_pack_b32_f16_overlay() -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.v_pack_b32_f16",
        instruction_name="V_PACK_B32_F16",
        mnemonic="v_pack_b32_f16",
        encoding_name="ENC_VOP3",
        semantic_tag="pack.float.f16x2.b32",
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            AmdgpuOperandOverlay("SRC0", _f16_vgpr_operand("low")),
            AmdgpuOperandOverlay("SRC1", _f16_vgpr_operand("high")),
        ),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


_PACKED8_SOURCE_SIZE_REASON = "packed8-conversion-reads-byte-lanes-of-b32-source"


def _v_cvt_f32_packed8_overlay(
    source_type: str, source_semantics: str
) -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key=f"amdgpu.v_cvt_f32_{source_type}.{source_semantics}",
        instruction_name=f"V_CVT_F32_{source_type.upper()}",
        mnemonic=f"v_cvt_f32_{source_type}",
        encoding_name="ENC_VOP1",
        semantic_tag=f"convert.float.{source_type}.{source_semantics}.f32",
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            AmdgpuOperandOverlay(
                "SRC0",
                _sgpr_vgpr_operand("input"),
                size_exception_reason=_PACKED8_SOURCE_SIZE_REASON,
            ),
        ),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_cvt_pk_f32_packed8_overlay(
    source_type: str, source_semantics: str
) -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key=f"amdgpu.v_cvt_pk_f32_{source_type}.{source_semantics}",
        instruction_name=f"V_CVT_PK_F32_{source_type.upper()}",
        mnemonic=f"v_cvt_pk_f32_{source_type}",
        encoding_name="ENC_VOP1",
        semantic_tag=(f"convert.float.{source_type}.{source_semantics}x2.f32x2"),
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result(units=2)),
            AmdgpuOperandOverlay(
                "SRC0",
                _sgpr_vgpr_operand("input"),
                size_exception_reason=_PACKED8_SOURCE_SIZE_REASON,
            ),
        ),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_cvt_pk_f16_packed8_overlay(
    source_type: str, source_semantics: str
) -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key=f"amdgpu.v_cvt_pk_f16_{source_type}.{source_semantics}",
        instruction_name=f"V_CVT_PK_F16_{source_type.upper()}",
        mnemonic=f"v_cvt_pk_f16_{source_type}",
        encoding_name="ENC_VOP1_VGPR",
        semantic_tag=(f"convert.float.{source_type}.{source_semantics}x2.f16x2"),
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            AmdgpuOperandOverlay(
                "VSRC0",
                _vgpr_operand(
                    "input",
                    register_part=_REG_PART_VGPR_LOW16,
                    address_map_kind=OperandAddressMapKind.LOW_SUBSET,
                    addressable_unit_count=(
                        _D16_PARTIAL_REGISTER_ADDRESSABLE_UNIT_COUNT
                    ),
                ),
                size_exception_reason=_PACKED8_SOURCE_SIZE_REASON,
            ),
        ),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


# VOP3 stores byte_sel's two logical selector bits in reversed OPSEL bit order.
_PACKED8_BYTE_SELECTOR_OPSEL_VALUES = (0, 2, 1, 3)
_PACKED8_BYTE_SELECTOR_OP_SEL_LITERALS = (
    "",
    "op_sel:[0,1,0]",
    "op_sel:[1,0,0]",
    "op_sel:[1,1,0]",
)


def _v_cvt_f32_packed8_byte_overlay(
    source_type: str,
    source_semantics: str,
    byte_selector: int,
    *,
    op_sel_field: str,
) -> AmdgpuDescriptorOverlay:
    if byte_selector == 0:
        return _v_cvt_f32_packed8_overlay(source_type, source_semantics)
    return AmdgpuDescriptorOverlay(
        descriptor_key=(
            f"amdgpu.v_cvt_f32_{source_type}.{source_semantics}.byte{byte_selector}"
        ),
        instruction_name=f"V_CVT_F32_{source_type.upper()}",
        mnemonic=f"v_cvt_f32_{source_type}_byte{byte_selector}",
        encoding_name="ENC_VOP3",
        semantic_tag=(
            f"convert.float.{source_type}.{source_semantics}.byte{byte_selector}.f32"
        ),
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            AmdgpuOperandOverlay(
                "SRC0",
                _sgpr_vgpr_operand("input"),
                size_exception_reason=_PACKED8_SOURCE_SIZE_REASON,
            ),
        ),
        fixed_encoding_fields=(
            (op_sel_field, _PACKED8_BYTE_SELECTOR_OPSEL_VALUES[byte_selector]),
        ),
        asm_forms=_asm(
            native_assembly_mnemonic=f"v_cvt_f32_{source_type}",
            results=("dst",),
            operands=("input",),
            native_assembly_values=(
                _native_result("dst"),
                _native_operand("input"),
                _native_literal(f"byte_sel:{byte_selector}"),
            ),
        ),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_cvt_pk_f32_packed8_high_overlay(
    source_type: str, source_semantics: str, *, op_sel_field: str
) -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key=(f"amdgpu.v_cvt_pk_f32_{source_type}.{source_semantics}.high"),
        instruction_name=f"V_CVT_PK_F32_{source_type.upper()}",
        mnemonic=f"v_cvt_pk_f32_{source_type}_high",
        encoding_name="ENC_VOP3",
        semantic_tag=(f"convert.float.{source_type}.{source_semantics}x2.high.f32x2"),
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result(units=2)),
            AmdgpuOperandOverlay(
                "SRC0",
                _sgpr_vgpr_operand("input"),
                size_exception_reason=_PACKED8_SOURCE_SIZE_REASON,
            ),
        ),
        fixed_encoding_fields=((op_sel_field, _PACKED8_BYTE_SELECTOR_OPSEL_VALUES[2]),),
        asm_forms=_asm(
            native_assembly_mnemonic=f"v_cvt_pk_f32_{source_type}",
            results=("dst",),
            operands=("input",),
            native_assembly_values=(
                _native_result("dst"),
                _native_operand("input"),
                _native_literal("op_sel:[1,0]"),
            ),
        ),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_cvt_f16_packed8_byte_overlay(
    source_type: str, source_semantics: str, byte_selector: int
) -> AmdgpuDescriptorOverlay:
    uses_byte_selector = byte_selector != 0
    return AmdgpuDescriptorOverlay(
        descriptor_key=(
            f"amdgpu.v_cvt_f16_{source_type}.{source_semantics}.byte{byte_selector}"
        ),
        instruction_name=f"V_CVT_F16_{source_type.upper()}",
        mnemonic=f"v_cvt_f16_{source_type}_byte{byte_selector}",
        encoding_name="ENC_VOP3" if uses_byte_selector else "ENC_VOP1",
        semantic_tag=(
            f"convert.float.{source_type}.{source_semantics}.byte{byte_selector}.f16"
        ),
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay(
                "VDST",
                _vgpr_result(
                    register_part=_REG_PART_VGPR_LOW16,
                    address_map_kind=OperandAddressMapKind.LOW_SUBSET,
                    addressable_unit_count=(
                        _D16_PARTIAL_REGISTER_ADDRESSABLE_UNIT_COUNT
                    ),
                ),
            ),
            AmdgpuOperandOverlay("SRC0", _sgpr_vgpr_operand("input")),
        ),
        fixed_encoding_fields=(
            (("OPSEL", _PACKED8_BYTE_SELECTOR_OPSEL_VALUES[byte_selector]),)
            if uses_byte_selector
            else ()
        ),
        asm_forms=_asm(
            native_assembly_mnemonic=f"v_cvt_f16_{source_type}",
            results=("dst",),
            operands=("input",),
            native_assembly_values=(
                _native_result("dst"),
                _native_operand("input"),
                *(
                    (_native_literal(f"byte_sel:{byte_selector}"),)
                    if byte_selector != 0
                    else ()
                ),
            ),
        ),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


_SCALEF32_PK_PACKED8_ROWS = (
    ("fp4", "f16", 1),
    ("fp4", "bf16", 1),
    ("fp4", "f32", 2),
    ("fp8", "f16", 1),
    ("bf8", "f16", 1),
    ("fp8", "bf16", 1),
    ("bf8", "bf16", 1),
    ("fp8", "f32", 2),
    ("bf8", "f32", 2),
)


_SCALE_PK8_ROWS = (
    ("fp4", "f16", 1, 4),
    ("fp4", "bf16", 1, 4),
    ("fp4", "f32", 1, 8),
    ("fp8", "f16", 2, 4),
    ("fp8", "bf16", 2, 4),
    ("fp8", "f32", 2, 8),
    ("bf8", "f16", 2, 4),
    ("bf8", "bf16", 2, 4),
    ("bf8", "f32", 2, 8),
)


_SCALE_PK8_SCALE_SEL_IMMEDIATE = Immediate(
    "scale_sel",
    ImmediateKind.UNSIGNED,
    flags=(ImmediateFlag.DEFAULT_VALUE,),
    bit_width=4,
    unsigned_max=15,
    default_value=0,
)


def _v_cvt_scalef32_pk_packed8_overlay(
    source_type: str,
    target_type: str,
    result_units: int,
    byte_selector: int,
) -> AmdgpuDescriptorOverlay:
    descriptor_suffix = "" if byte_selector == 0 else f".byte{byte_selector}"
    mnemonic_suffix = "" if byte_selector == 0 else f"_byte{byte_selector}"
    semantic_suffix = "" if byte_selector == 0 else f".byte{byte_selector}"
    native_assembly_mnemonic = f"v_cvt_scalef32_pk_{target_type}_{source_type}"
    return AmdgpuDescriptorOverlay(
        descriptor_key=(
            f"amdgpu.v_cvt_scalef32_pk_{target_type}_{source_type}"
            f".ocp{descriptor_suffix}"
        ),
        instruction_name=(
            f"V_CVT_SCALEF32_PK_{target_type.upper()}_{source_type.upper()}"
        ),
        mnemonic=f"{native_assembly_mnemonic}{mnemonic_suffix}",
        encoding_name="ENC_VOP3",
        semantic_tag=(
            f"convert.scale.float.{source_type}.ocpx2{semantic_suffix}.{target_type}x2"
        ),
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result(units=result_units)),
            AmdgpuOperandOverlay(
                "SRC0",
                _sgpr_vgpr_operand("input"),
                size_exception_reason=_PACKED8_SOURCE_SIZE_REASON,
            ),
            AmdgpuOperandOverlay("SRC1", _sgpr_vgpr_operand("scale")),
        ),
        fixed_encoding_fields=(
            (("OP_SEL", _PACKED8_BYTE_SELECTOR_OPSEL_VALUES[byte_selector]),)
            if byte_selector != 0
            else ()
        ),
        asm_forms=_asm(
            native_assembly_mnemonic=(
                native_assembly_mnemonic if byte_selector != 0 else None
            ),
            results=("dst",),
            operands=("input", "scale"),
            native_assembly_values=(
                _native_result("dst"),
                _native_operand("input"),
                _native_operand("scale"),
                *(
                    (
                        _native_literal(
                            _PACKED8_BYTE_SELECTOR_OP_SEL_LITERALS[byte_selector]
                        ),
                    )
                    if byte_selector != 0
                    else ()
                ),
            ),
        ),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_cvt_scale_pk8_overlay(
    source_type: str,
    target_type: str,
    source_units: int,
    result_units: int,
) -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key=f"amdgpu.v_cvt_scale_pk8_{target_type}_{source_type}.ocp",
        instruction_name=f"V_CVT_SCALE_PK8_{target_type.upper()}_{source_type.upper()}",
        mnemonic=f"v_cvt_scale_pk8_{target_type}_{source_type}",
        encoding_name="ENC_VOP3",
        semantic_tag=f"convert.scale.e8m0.{source_type}.ocpx8.{target_type}x8",
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result(units=result_units)),
            AmdgpuOperandOverlay(
                "SRC0",
                _sgpr_vgpr_operand("input", units=source_units),
            ),
            AmdgpuOperandOverlay("SRC1", _sgpr_vgpr_operand("scale")),
        ),
        immediate_fields=("SCALE_SEL",),
        immediates=(_SCALE_PK8_SCALE_SEL_IMMEDIATE,),
        asm_forms=_asm(
            results=("dst",),
            operands=("input", "scale"),
            immediates=("scale_sel",),
            named_immediates=True,
            native_assembly_values=(
                _native_result("dst"),
                _native_operand("input"),
                _native_operand("scale"),
                _native_amdgpu_scale_sel_immediate("scale_sel"),
            ),
        ),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_cvt_f32_packed8_overlays(
    source_semantics: str,
) -> tuple[AmdgpuDescriptorOverlay, ...]:
    return (
        _v_cvt_f32_packed8_overlay("fp8", source_semantics),
        _v_cvt_f32_packed8_overlay("bf8", source_semantics),
        _v_cvt_pk_f32_packed8_overlay("fp8", source_semantics),
        _v_cvt_pk_f32_packed8_overlay("bf8", source_semantics),
    )


def _v_cvt_f32_packed8_selection_overlays(
    source_semantics: str, *, op_sel_field: str
) -> tuple[AmdgpuDescriptorOverlay, ...]:
    return tuple(
        _v_cvt_f32_packed8_byte_overlay(
            source_type,
            source_semantics,
            byte_selector,
            op_sel_field=op_sel_field,
        )
        for source_type in ("fp8", "bf8")
        for byte_selector in range(4)
    ) + tuple(
        overlay
        for source_type in ("fp8", "bf8")
        for overlay in (
            _v_cvt_pk_f32_packed8_overlay(source_type, source_semantics),
            _v_cvt_pk_f32_packed8_high_overlay(
                source_type, source_semantics, op_sel_field=op_sel_field
            ),
        )
    )


def _v_cvt_pk_f16_packed8_overlays(
    source_semantics: str,
) -> tuple[AmdgpuDescriptorOverlay, ...]:
    return (
        _v_cvt_pk_f16_packed8_overlay("fp8", source_semantics),
        _v_cvt_pk_f16_packed8_overlay("bf8", source_semantics),
    )


_VOP3_DESTINATION_OP_SEL = 1 << 3


def _v_cvt_pk_packed8_encode_overlay(
    target_type: str,
    source_type: str,
    target_semantics: str,
    result_part: str,
    *,
    op_sel_field: str,
) -> AmdgpuDescriptorOverlay:
    result_register_part = {
        "low": _REG_PART_VGPR_LOW16,
        "high": _REG_PART_VGPR_HIGH16,
    }[result_part]
    is_high_result = result_part == "high"
    if source_type == "f32":
        source_operands = (
            AmdgpuOperandOverlay("SRC0", _sgpr_vgpr_operand("low")),
            AmdgpuOperandOverlay("SRC1", _sgpr_vgpr_operand("high")),
        )
        native_source_values = (
            _native_operand("low"),
            _native_operand("high"),
        )
        high_native_modifier = "op_sel:[0,0,1]"
    elif source_type == "f16":
        source_operands = (AmdgpuOperandOverlay("SRC0", _sgpr_vgpr_operand("input")),)
        native_source_values = (_native_operand("input"),)
        high_native_modifier = "op_sel:[0,1]"
    else:
        raise ValueError(f"unsupported packed8 encode source type '{source_type}'")

    accumulator_operands = (
        (
            AmdgpuOperandOverlay(
                "VDST",
                Operand(
                    "acc",
                    OperandRole.OPERAND,
                    _VGPR_ALT,
                    flags=(
                        OperandFlag.IMPLICIT,
                        OperandFlag.STORAGE_CONTINUATION,
                    ),
                    register_part=_REG_PART_VGPR_LOW16,
                ),
                role_exception_reason=(
                    "the encoded destination register carries the untouched "
                    "packed byte pair"
                ),
            ),
        )
        if is_high_result
        else ()
    )
    return AmdgpuDescriptorOverlay(
        descriptor_key=(
            f"amdgpu.v_cvt_pk_{target_type}_{source_type}."
            f"{target_semantics}.{result_part}"
        ),
        instruction_name=f"V_CVT_PK_{target_type.upper()}_{source_type.upper()}",
        mnemonic=f"v_cvt_pk_{target_type}_{source_type}_{result_part}",
        encoding_name="ENC_VOP3",
        semantic_tag=(
            f"convert.float.{source_type}x2."
            f"{target_type}.{target_semantics}x2.{result_part}"
        ),
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay(
                "VDST", _vgpr_result(register_part=result_register_part)
            ),
            *accumulator_operands,
            *source_operands,
        ),
        constraints=(
            (Constraint(ConstraintKind.TIED, 0, 1),) if is_high_result else ()
        ),
        fixed_encoding_fields=(
            (op_sel_field, _VOP3_DESTINATION_OP_SEL if is_high_result else 0),
        ),
        asm_forms=_asm(
            results=("dst",),
            operands=(
                *(("acc",) if is_high_result else ()),
                *(("low", "high") if source_type == "f32" else ("input",)),
            ),
            native_assembly_mnemonic=(f"v_cvt_pk_{target_type}_{source_type}"),
            native_assembly_values=(
                _native_result("dst"),
                *native_source_values,
                *((_native_literal(high_native_modifier),) if is_high_result else ()),
            ),
        ),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_cvt_pk_packed8_from_f32_overlays(
    target_semantics: str,
    *,
    op_sel_field: str,
) -> tuple[AmdgpuDescriptorOverlay, ...]:
    return tuple(
        _v_cvt_pk_packed8_encode_overlay(
            target_type,
            "f32",
            target_semantics,
            result_part,
            op_sel_field=op_sel_field,
        )
        for target_type in ("fp8", "bf8")
        for result_part in ("low", "high")
    )


def _v_cvt_pk_packed8_from_f16_overlays(
    target_semantics: str,
    *,
    op_sel_field: str,
) -> tuple[AmdgpuDescriptorOverlay, ...]:
    return tuple(
        _v_cvt_pk_packed8_encode_overlay(
            target_type,
            "f16",
            target_semantics,
            result_part,
            op_sel_field=op_sel_field,
        )
        for target_type in ("fp8", "bf8")
        for result_part in ("low", "high")
    )


def _v_cvt_f16_packed8_byte_overlays(
    source_semantics: str,
) -> tuple[AmdgpuDescriptorOverlay, ...]:
    return tuple(
        _v_cvt_f16_packed8_byte_overlay(source_type, source_semantics, byte_selector)
        for source_type in ("fp8", "bf8")
        for byte_selector in range(4)
    )


def _v_cvt_scalef32_pk_packed8_overlays() -> tuple[AmdgpuDescriptorOverlay, ...]:
    return tuple(
        _v_cvt_scalef32_pk_packed8_overlay(*row, byte_selector)
        for row in _SCALEF32_PK_PACKED8_ROWS
        for byte_selector in range(4)
    )


def _v_cvt_scale_pk8_overlays() -> tuple[AmdgpuDescriptorOverlay, ...]:
    return tuple(_v_cvt_scale_pk8_overlay(*row) for row in _SCALE_PK8_ROWS)


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


def _v_cvt_pk_dpp16_overlay(
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
        encoding_name="VOP3_VOP_DPP16",
        encoding_condition="has_dpp16",
        semantic_tag=semantic_tag,
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            AmdgpuOperandOverlay("VSRC0", _vgpr_operand("crosslane")),
            AmdgpuOperandOverlay("SRC1", _vgpr_operand("local")),
        ),
        asm_forms=_asm(
            mnemonic=f"{mnemonic}_dpp16",
            native_assembly_mnemonic=f"{mnemonic}_e64_dpp",
            results=("dst",),
            operands=("crosslane", "local"),
            immediates=("dpp_ctrl",),
            named_immediates=True,
            native_assembly_values=(
                _native_result("dst"),
                _native_operand("crosslane"),
                _native_operand("local"),
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
    )


def _v_cvt_pk_u16_u32_dpp16_overlay() -> AmdgpuDescriptorOverlay:
    return _v_cvt_pk_dpp16_overlay(
        descriptor_key="amdgpu.v_cvt_pk_u16_u32.dpp16",
        instruction_name="V_CVT_PK_U16_U32",
        mnemonic="v_cvt_pk_u16_u32",
        semantic_tag="convert.pack.u32.u16x2.dpp",
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


def _v_cvt_pk_bf16_f32_dpp16_overlay() -> AmdgpuDescriptorOverlay:
    return _v_cvt_pk_dpp16_overlay(
        descriptor_key="amdgpu.v_cvt_pk_bf16_f32.dpp16",
        instruction_name="V_CVT_PK_BF16_F32",
        mnemonic="v_cvt_pk_bf16_f32",
        semantic_tag="convert.float.f32.bf16x2.dpp",
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
        constraints=_REMATERIALIZABLE_RESULT_CONSTRAINTS,
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_cvt_f32_ubyte_overlay(byte_ordinal: int) -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key=f"amdgpu.v_cvt_f32_ubyte{byte_ordinal}",
        instruction_name=f"V_CVT_F32_UBYTE{byte_ordinal}",
        mnemonic=f"v_cvt_f32_ubyte{byte_ordinal}",
        encoding_name="ENC_VOP1",
        semantic_tag=f"convert.unsigned.u8.byte{byte_ordinal}.f32",
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            AmdgpuOperandOverlay("SRC0", _sgpr_vgpr_operand("input")),
        ),
        constraints=_REMATERIALIZABLE_RESULT_CONSTRAINTS,
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_cvt_f32_ubyte_overlays() -> tuple[AmdgpuDescriptorOverlay, ...]:
    return tuple(_v_cvt_f32_ubyte_overlay(i) for i in range(4))


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
        constraints=_REMATERIALIZABLE_RESULT_CONSTRAINTS,
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


def _v_cmp_i32_source0_inline_vcc_overlay(
    *, predicate: str, instruction_suffix: str, semantic_suffix: str
) -> AmdgpuDescriptorOverlay:
    instruction_predicate = instruction_suffix.lower()
    return AmdgpuDescriptorOverlay(
        descriptor_key=f"amdgpu.v_cmp_{predicate}_i32.src0_inline_vcc",
        instruction_name=f"V_CMP_{instruction_suffix}_I32",
        mnemonic=f"v_cmp_{instruction_predicate}_i32",
        encoding_name="ENC_VOPC",
        semantic_tag=f"cmp.i32.{semantic_suffix}",
        schedule_class=_SCHEDULE_VALU,
        operands=(AmdgpuOperandOverlay("VSRC1", _vgpr_operand("rhs")),),
        implicit_operands=(_vcc_output(_vcc_result("mask"), operand_type="OPR_VCC"),),
        asm_forms=_asm(
            mnemonic=f"v_cmp_{instruction_predicate}_i32_src0_inline_vcc",
            native_assembly_mnemonic=f"v_cmp_{instruction_predicate}_i32",
            results=("mask",),
            operands=("rhs",),
            immediates=("lhs",),
            named_immediates=True,
            native_assembly_values=(
                _native_result("mask"),
                _native_i64_immediate("lhs"),
                _native_operand("rhs"),
            ),
        ),
        immediate_fields=("SRC0",),
        immediates=(_source_inline_u32_immediate("lhs"),),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_cmp_i32_source1_inline_vcc_overlay(
    *, predicate: str, instruction_suffix: str, semantic_suffix: str
) -> AmdgpuDescriptorOverlay:
    instruction_predicate = instruction_suffix.lower()
    return AmdgpuDescriptorOverlay(
        descriptor_key=f"amdgpu.v_cmp_{predicate}_i32.src1_inline_vcc",
        instruction_name=f"V_CMP_{instruction_suffix}_I32",
        mnemonic=f"v_cmp_{instruction_predicate}_i32",
        encoding_name="ENC_VOPC",
        semantic_tag=f"cmp.i32.{semantic_suffix}",
        schedule_class=_SCHEDULE_VALU,
        operands=(AmdgpuOperandOverlay("VSRC1", _vgpr_operand("lhs")),),
        implicit_operands=(_vcc_output(_vcc_result("mask"), operand_type="OPR_VCC"),),
        asm_forms=_asm(
            mnemonic=f"v_cmp_{instruction_predicate}_i32_src1_inline_vcc",
            native_assembly_mnemonic=f"v_cmp_{instruction_predicate}_i32",
            results=("mask",),
            operands=("lhs",),
            immediates=("rhs",),
            named_immediates=True,
            native_assembly_values=(
                _native_result("mask"),
                _native_i64_immediate("rhs"),
                _native_operand("lhs"),
            ),
        ),
        immediate_fields=("SRC0",),
        immediates=(_source_inline_u32_immediate("rhs"),),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_cmp_i32_equality_vcc_overlays() -> tuple[AmdgpuDescriptorOverlay, ...]:
    return tuple(
        overlay_builder(
            predicate=predicate,
            instruction_suffix=instruction_suffix,
            semantic_suffix=predicate,
        )
        for predicate, instruction_suffix in (("eq", "EQ"), ("ne", "NE"))
        for overlay_builder in (
            _v_cmp_i32_source0_inline_vcc_overlay,
            _v_cmp_i32_source1_inline_vcc_overlay,
        )
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


_V_CMP_OVERLAY_FAMILIES = (
    (
        _v_cmp_i32_overlay,
        _v_cmp_i32_source_overlays,
        _SOURCE_INLINE_U32_IMMEDIATE,
        (
            ("eq", "EQ", "eq"),
            ("ne", "NE", "ne"),
            ("slt", "LT", "slt"),
            ("sle", "LE", "sle"),
            ("sgt", "GT", "sgt"),
            ("sge", "GE", "sge"),
        ),
    ),
    (
        _v_cmp_u32_overlay,
        _v_cmp_u32_source_overlays,
        _SOURCE_INLINE_U32_IMMEDIATE,
        (
            ("ult", "LT", "ult"),
            ("ule", "LE", "ule"),
            ("ugt", "GT", "ugt"),
            ("uge", "GE", "uge"),
        ),
    ),
    (
        _v_cmp_f32_overlay,
        _v_cmp_f32_source_overlays,
        _SOURCE_INLINE_F32_IMMEDIATE,
        (
            ("oeq", "EQ", "oeq"),
            ("ogt", "GT", "ogt"),
            ("oge", "GE", "oge"),
            ("olt", "LT", "olt"),
            ("ole", "LE", "ole"),
            ("one", "LG", "one"),
            ("ord", "O", "ord"),
            ("ueq", "NLG", "ueq"),
            ("ugt", "NLE", "ugt"),
            ("uge", "NLT", "uge"),
            ("ult", "NGE", "ult"),
            ("ule", "NGT", "ule"),
            ("une", "NEQ", "une"),
            ("uno", "U", "uno"),
        ),
    ),
)


def _v_cmp_base_overlays() -> tuple[AmdgpuDescriptorOverlay, ...]:
    return tuple(
        base_builder(
            predicate=predicate,
            instruction_suffix=instruction_suffix,
            semantic_suffix=semantic_suffix,
        )
        for base_builder, _, _, rows in _V_CMP_OVERLAY_FAMILIES
        for predicate, instruction_suffix, semantic_suffix in rows
    )


def _v_cmp_overlays() -> tuple[AmdgpuDescriptorOverlay, ...]:
    return tuple(
        overlay
        for base_builder, source_builder, _, rows in _V_CMP_OVERLAY_FAMILIES
        for predicate, instruction_suffix, semantic_suffix in rows
        for overlay in (
            base_builder(
                predicate=predicate,
                instruction_suffix=instruction_suffix,
                semantic_suffix=semantic_suffix,
            ),
            *source_builder(
                predicate=predicate,
                instruction_suffix=instruction_suffix,
                semantic_suffix=semantic_suffix,
            ),
        )
    )


def _v_dpp_uniform_rhs_integer_compare_candidates() -> tuple[
    tuple[AmdgpuDescriptorOverlay, Immediate], ...
]:
    integer_candidates = (
        _v_add_u32_overlay("V_ADD_NC_U32"),
        _v_sub_u32_overlay("V_SUB_NC_U32", "v_sub_nc_u32"),
        _v_add3_u32_overlay(),
        _v_add_co_u32_overlay(),
        _v_add_co_ci_u32_overlay(),
        _v_sub_co_u32_overlay(),
        _v_sub_co_ci_u32_overlay(),
        _v_min_i32_overlay(),
        _v_max_i32_overlay(),
        _v_min_u32_overlay(),
        _v_max_u32_overlay(),
        _v_and_b32_overlay(),
        _v_or_b32_overlay(),
        _v_xor_b32_overlay(),
    )
    return (
        *((overlay, _SOURCE_INLINE_U32_IMMEDIATE) for overlay in integer_candidates),
        *(
            (
                base_builder(
                    predicate=predicate,
                    instruction_suffix=instruction_suffix,
                    semantic_suffix=semantic_suffix,
                ),
                inline_immediate,
            )
            for base_builder, _, inline_immediate, rows in _V_CMP_OVERLAY_FAMILIES
            for predicate, instruction_suffix, semantic_suffix in rows
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
        immediate_fields=("LITERAL", inline_field),
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


def _v_cndmask_b32_vcc_overlay(
    *,
    include_literal_form: bool = True,
) -> AmdgpuDescriptorOverlay:
    operand_forms = [
        _literal_operand_form(
            replacement_descriptor="amdgpu.v_cndmask_b32.src0_inline_vcc",
            source_operand="false_value",
            immediate_field="false_value",
        )
    ]
    if include_literal_form:
        operand_forms.append(
            _literal_operand_form(
                replacement_descriptor="amdgpu.v_cndmask_b32.src0_lit_vcc",
                source_operand="false_value",
            )
        )
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.v_cndmask_b32.vcc",
        instruction_name="V_CNDMASK_B32",
        mnemonic="v_cndmask_b32",
        encoding_name="ENC_VOP2",
        semantic_tag="select.mask.b32",
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            AmdgpuOperandOverlay("SRC0", _sgpr_vgpr_operand("false_value")),
            AmdgpuOperandOverlay("VSRC1", _vgpr_operand("true_value")),
        ),
        implicit_operands=(_vcc_input(_vcc_predicate("mask")),),
        operand_forms=tuple(operand_forms),
        asm_forms=_asm(
            mnemonic="v_cndmask_b32_vcc",
            native_assembly_mnemonic="v_cndmask_b32",
            results=("dst",),
            operands=("false_value", "true_value", "mask"),
        ),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_cndmask_b32_src0_inline_vcc_overlay() -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.v_cndmask_b32.src0_inline_vcc",
        instruction_name="V_CNDMASK_B32",
        mnemonic="v_cndmask_b32",
        encoding_name="ENC_VOP2",
        semantic_tag="select.mask.b32",
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            AmdgpuOperandOverlay("VSRC1", _vgpr_operand("true_value")),
        ),
        implicit_operands=(_vcc_input(_vcc_predicate("mask")),),
        asm_forms=_asm(
            mnemonic="v_cndmask_b32_src0_inline_vcc",
            native_assembly_mnemonic="v_cndmask_b32",
            results=("dst",),
            operands=("true_value", "mask"),
            immediates=("false_value",),
            named_immediates=True,
            native_assembly_values=(
                _native_result("dst"),
                _native_i64_immediate("false_value"),
                _native_operand("true_value"),
                _native_operand("mask"),
            ),
        ),
        immediate_fields=("SRC0",),
        immediates=(_source_inline_u32_immediate("false_value"),),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_cndmask_b32_src0_literal_vcc_overlay() -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.v_cndmask_b32.src0_lit_vcc",
        instruction_name="V_CNDMASK_B32",
        mnemonic="v_cndmask_b32",
        encoding_name="VOP2_INST_LITERAL",
        encoding_condition="has_lit",
        semantic_tag="select.mask.b32",
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            AmdgpuOperandOverlay("VSRC1", _vgpr_operand("true_value")),
        ),
        implicit_operands=(_vcc_input(_vcc_predicate("mask")),),
        asm_forms=_asm(
            mnemonic="v_cndmask_b32_src0_lit_vcc",
            native_assembly_mnemonic="v_cndmask_b32",
            results=("dst",),
            operands=("true_value", "mask"),
            immediates=("imm32",),
            native_assembly_values=(
                _native_result("dst"),
                _native_i64_immediate("imm32"),
                _native_operand("true_value"),
                _native_operand("mask"),
            ),
        ),
        immediate_fields=("LITERAL",),
        immediates=(_LITERAL_U32_IMMEDIATE,),
        fixed_encoding_fields=(("SRC0", _predefined("SRC_LITERAL", "OPR_SRC")),),
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
        _v_cndmask_b32_vcc_overlay(include_literal_form=include_literal_forms),
        _v_cndmask_b32_src0_inline_vcc_overlay(),
    )
    if not include_literal_forms:
        return overlays
    return (
        *overlays,
        _v_cndmask_b32_source_literal_overlay("src0"),
        _v_cndmask_b32_source_literal_overlay("src1"),
        _v_cndmask_b32_literal_inline_overlay("src0"),
        _v_cndmask_b32_literal_inline_overlay("src1"),
        _v_cndmask_b32_src0_literal_vcc_overlay(),
    )


def _v_cndmask_b32_dpp_overlay(
    *, descriptor_suffix: str, encoding_name: str, encoding_condition: str
) -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key=f"amdgpu.v_cndmask_b32.{descriptor_suffix}",
        instruction_name="V_CNDMASK_B32",
        mnemonic="v_cndmask_b32",
        encoding_name=encoding_name,
        encoding_condition=encoding_condition,
        semantic_tag="select.mask.b32",
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            AmdgpuOperandOverlay("VSRC0", _vgpr_operand("false_value")),
            AmdgpuOperandOverlay("VSRC1", _vgpr_operand("true_value")),
        ),
        implicit_operands=(_vcc_input(_vcc_predicate("mask")),),
        asm_forms=_asm(
            mnemonic=f"v_cndmask_b32_{descriptor_suffix}",
            native_assembly_mnemonic=(
                None if descriptor_suffix == "dpp" else "v_cndmask_b32_dpp"
            ),
            results=("dst",),
            operands=("false_value", "true_value", "mask"),
            immediates=("dpp_ctrl",),
            named_immediates=True,
            native_assembly_values=(
                _native_result("dst"),
                _native_operand("false_value"),
                _native_operand("true_value"),
                _native_operand("mask"),
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
    )


def _v_cndmask_b32_dpp_legacy_overlay() -> AmdgpuDescriptorOverlay:
    return _v_cndmask_b32_dpp_overlay(
        descriptor_suffix="dpp",
        encoding_name="VOP2_VOP_DPP",
        encoding_condition="has_dpp",
    )


def _v_cndmask_b32_dpp16_overlay() -> AmdgpuDescriptorOverlay:
    return _v_cndmask_b32_dpp_overlay(
        descriptor_suffix="dpp16",
        encoding_name="VOP2_VOP_DPP16",
        encoding_condition="has_dpp16",
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
        carrier=DescriptorCarrier.CONST,
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
    bank_mask_immediate: Immediate | None = None,
) -> AmdgpuDescriptorOverlay:
    operands = [AmdgpuOperandOverlay("VDST", _vgpr_result())]
    constraints: tuple[Constraint, ...] = ()
    asm_operands = []
    if bank_mask_immediate is not None:
        operands.append(
            AmdgpuOperandOverlay(
                "VDST",
                Operand(
                    "old_value",
                    OperandRole.OPERAND,
                    _VGPR_ALT,
                    flags=(OperandFlag.IMPLICIT,),
                ),
                role_exception_reason=(
                    "the encoded destination register is also the tied "
                    "preserved-lane input"
                ),
            )
        )
        constraints = (
            Constraint(ConstraintKind.TIED, 0, 1),
            Constraint(ConstraintKind.DESTRUCTIVE, 0, 1),
        )
        asm_operands.append("old_value")
    operands.append(AmdgpuOperandOverlay("VSRC0", _vgpr_operand("src")))
    asm_operands.append("src")

    immediate_fields = ["DPP_CTRL"]
    immediates = [_DPP_CTRL_IMMEDIATE]
    fixed_encoding_fields = [("SRC0", 250), ("ROW_MASK", 0xF)]
    if bank_mask_immediate is None:
        fixed_encoding_fields.append(("BANK_MASK", 0xF))
    else:
        immediate_fields.append("BANK_MASK")
        immediates.append(bank_mask_immediate)
    fixed_encoding_fields.append(("BOUND_CTRL", 1))

    return AmdgpuDescriptorOverlay(
        descriptor_key=descriptor_key,
        instruction_name="V_MOV_B32",
        mnemonic="v_mov_b32",
        encoding_name=encoding_name,
        encoding_condition=encoding_condition,
        semantic_tag=(
            "lane.dpp.b32"
            if bank_mask_immediate is None
            else "lane.dpp.masked_update.b32"
        ),
        schedule_class=_SCHEDULE_VALU,
        operands=tuple(operands),
        constraints=constraints,
        asm_forms=_asm(
            mnemonic=descriptor_key.removeprefix("amdgpu."),
            native_assembly_mnemonic=(
                None if descriptor_key.endswith("_dpp") else "v_mov_b32_dpp"
            ),
            results=("dst",),
            operands=tuple(asm_operands),
            immediates=tuple(immediate.field_name for immediate in immediates),
            named_immediates=True,
            native_assembly_values=(
                _native_result("dst"),
                _native_operand("src"),
                _native_amdgpu_dpp_ctrl_immediate("dpp_ctrl"),
                _native_literal("row_mask:0xf"),
                (
                    _native_literal("bank_mask:0xf")
                    if bank_mask_immediate is None
                    else _native_amdgpu_dpp_bank_mask_immediate("bank_mask")
                ),
                _native_literal("bound_ctrl:1"),
            ),
        ),
        immediate_fields=tuple(immediate_fields),
        immediates=tuple(immediates),
        fixed_encoding_fields=tuple(fixed_encoding_fields),
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


def _v_mov_b32_dpp_masked_legacy_overlay() -> AmdgpuDescriptorOverlay:
    return _v_mov_b32_dpp_overlay(
        descriptor_key="amdgpu.v_mov_b32_dpp_masked",
        encoding_name="VOP1_VOP_DPP",
        encoding_condition="has_dpp",
        bank_mask_immediate=_DPP_BANK_MASK_IMMEDIATE,
    )


def _v_mov_b32_dpp16_masked_overlay() -> AmdgpuDescriptorOverlay:
    return _v_mov_b32_dpp_overlay(
        descriptor_key="amdgpu.v_mov_b32_dpp16_masked",
        encoding_name="VOP1_VOP_DPP16",
        encoding_condition="has_dpp16",
        bank_mask_immediate=_DPP_BANK_MASK_IMMEDIATE,
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
    "_integer_bit_count_overlays",
    "_integer_bitwise_permute_overlays",
    "_integer_bitwise_shift_overlays",
    "_rdna_integer_bit_count_overlays",
    "_s_add_u32_overlay",
    "_s_add_u32_rhs_inline_overlay",
    "_s_addk_i32_overlay",
    "_s_add_u32_rhs_symbol_rel32_lo_overlay",
    "_s_addc_u32_rhs_symbol_rel32_hi_overlay",
    "_s_addc_u32_overlay",
    "_s_and_b32_overlay",
    "_s_and_b32_rhs_inline_overlay",
    "_s_and_b32_literal_overlay",
    "_s_and_b64_overlay",
    "_s_and_saveexec_b64_overlay",
    "_s_ashr_i32_overlay",
    "_s_ashr_i32_rhs_inline_overlay",
    "_s_bfe_b32_literal_overlay",
    "_s_bfe_b32_overlay",
    "_s_binary_u32_overlay",
    "_s_binary_u32_rhs_inline_overlay",
    "_s_binary_u32_literal_overlay",
    "_s_binary_u64_overlay",
    "_s_cmp_i32_overlay",
    "_s_cmp_i32_overlays",
    "_s_cmp_u64_overlay",
    "_s_cmp_u64_overlays",
    "_s_cmp_u64_src1_inline_overlay",
    "_s_cselect_b32_overlay",
    "_s_lshl_b32_overlay",
    "_s_lshl_b32_rhs_inline_overlay",
    "_s_lshl_b64_overlay",
    "_s_lshl_add_u32_overlay",
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
    "_v_add3_u32_literal_overlay",
    "_v_add3_u32_overlay",
    "_v_add_f16_overlay",
    "_v_add_f32_literal_overlay",
    "_v_add_f32_overlay",
    "_v_add_f32_src0_inline_overlay",
    "_v_add_u32_literal_overlay",
    "_v_add_u32_overlay",
    "_v_add_u32_rhs_tied_overlay",
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
    "_v_binary_f16_overlay",
    "_v_binary_f16_overlays",
    "_v_binary_f32_dpp16_overlays",
    "_v_binary_f32_dpp_legacy_overlays",
    "_v_binary_f32_dpp_overlay",
    "_v_binary_f32_dpp_variant_overlays",
    "_v_binary_f32_overlay",
    "_v_binary_f32_operand_forms",
    "_v_binary_f32_overlays",
    "_v_binary_literal_overlay",
    "_v_binary_src0_inline_f32_overlay",
    "_v_binary_src0_inline_overlay",
    "_v_binary_u32_overlay",
    "_v_binary_vop3_float_overlay",
    "_v_cmp_base_overlays",
    "_v_cmp_f32_overlay",
    "_v_cmp_f32_source_overlays",
    "_v_cmp_i32_overlay",
    "_v_cmp_i32_equality_vcc_overlays",
    "_v_cmp_i32_source0_inline_vcc_overlay",
    "_v_cmp_i32_source1_inline_vcc_overlay",
    "_v_cmp_i32_source_overlays",
    "_v_cmp_inline_operand_forms",
    "_v_cmp_overlays",
    "_v_cmp_source_inline_overlay",
    "_v_cmp_u32_overlay",
    "_v_cmp_u32_source_overlays",
    "_v_dpp_uniform_rhs_integer_compare_candidates",
    "_v_commutative_binary_f16_overlay",
    "_v_commutative_binary_vop3_float_overlay",
    "_v_cndmask_b32_literal_inline_overlay",
    "_v_cndmask_b32_overlay",
    "_v_cndmask_b32_overlays",
    "_v_cndmask_b32_dpp16_overlay",
    "_v_cndmask_b32_dpp_legacy_overlay",
    "_v_cndmask_b32_dpp_overlay",
    "_v_cndmask_b32_src0_inline_vcc_overlay",
    "_v_cndmask_b32_src0_literal_vcc_overlay",
    "_v_cndmask_b32_source_inline_overlay",
    "_v_cndmask_b32_source_literal_overlay",
    "_v_cndmask_b32_vcc_overlay",
    "_v_cvt_f16_f32_overlay",
    "_v_cvt_f32_f16_overlay",
    "_v_cvt_f32_i32_overlay",
    "_v_cvt_f32_packed8_overlays",
    "_v_cvt_f32_packed8_selection_overlays",
    "_v_cvt_f32_ubyte_overlays",
    "_v_cvt_f32_u32_overlay",
    "_v_cvt_i32_f32_overlay",
    "_v_cvt_pk_bf16_f32_dpp16_overlay",
    "_v_cvt_pk_bf16_f32_overlay",
    "_v_cvt_f16_packed8_byte_overlays",
    "_v_cvt_pk_f16_packed8_overlays",
    "_v_cvt_pk_packed8_from_f16_overlays",
    "_v_cvt_pk_packed8_from_f32_overlays",
    "_v_cvt_pk_u16_u32_dpp16_overlay",
    "_v_cvt_pk_u16_u32_overlay",
    "_v_cvt_scale_pk8_overlays",
    "_v_cvt_scalef32_pk_packed8_overlays",
    "_v_cvt_u32_f32_overlay",
    "_v_pack_b32_f16_overlay",
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
    "_v_interp_overlays",
    "_v_fma_f64_overlay",
    "_v_fma_mix_f32_overlay",
    "_v_fma_mix_f32_overlays",
    "_v_fma_mix_f32_src2_literal_overlay",
    "_v_mix_half_result_src2_literal_overlay",
    "_v_fma_mixhi_f16_overlays",
    "_v_fma_mixlo_f16_overlays",
    "_v_fmac_f16_overlay",
    "_v_fmac_f32_overlay",
    "_v_fmac_f64_overlay",
    "_v_fmamk_f16_overlay",
    "_v_fmamk_f32_overlay",
    "_v_pk_ashrrev_i16_overlay",
    "_v_pk_add_f16_overlay",
    "_v_pk_add_f32_overlay",
    "_v_pk_add_bf16_overlay",
    "_v_pk_fma_f16_overlay",
    "_v_pk_fma_f16_literal_overlays",
    "_v_pk_maximum_f16_overlay",
    "_v_pk_maxnum_f16_overlay",
    "_v_pk_minimum_f16_overlay",
    "_v_pk_minnum_f16_overlay",
    "_v_pk_mul_f16_overlay",
    "_v_pk_mul_f32_overlay",
    "_v_pk_fma_bf16_overlay",
    "_v_pk_mul_bf16_overlay",
    "_v_pk_add_u16_overlay",
    "_v_pk_lshlrev_b16_overlay",
    "_v_pk_lshrrev_b16_overlay",
    "_v_pk_max_i16_overlay",
    "_v_pk_max_u16_overlay",
    "_v_pk_min_i16_overlay",
    "_v_pk_min_u16_overlay",
    "_v_pk_mul_lo_u16_overlay",
    "_v_pk_i16_binary_overlays",
    "_v_pk_fma_f32_overlay",
    "_v_pk_fmac_f16_overlay",
    "_v_pk_mad_i16_overlay",
    "_v_pk_mad_i16_literal_overlays",
    "_v_pk_mad_u16_overlay",
    "_v_pk_mad_u16_literal_overlays",
    "_v_pk_sub_i16_overlay",
    "_v_pk_binary_overlay",
    "_v_pk_ternary_overlay",
    "_v_pk_with_op_sel_hi_field",
    "_v_perm_b32_overlay",
    "_v_perm_b32_src2_literal_overlay",
    "_v_perm_b32_src1_zero_src2_literal_overlay",
    "_v_permlanex16_b32_src12_inline_overlay",
    "_v_lshl_add_u32_shift_immediate_overlay",
    "_v_lshl_add_u32_shift_immediate_src2_literal_overlay",
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
    "_v_med3_num_f32_overlay",
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
    "_v_mov_b32_dpp16_masked_overlay",
    "_v_mov_b32_dpp_legacy_overlay",
    "_v_mov_b32_dpp_masked_legacy_overlay",
    "_v_mov_b32_dpp_overlay",
    "_v_mov_b32_literal_overlay",
    "_v_mov_b32_sdwa_overlay",
    "_v_native_f32_math_overlays",
    "_v_mul_f16_overlay",
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
    "_v_readlane_b32_src1_inline_overlay",
    "_v_readlane_b32_src1_sgpr_overlay",
    "_v_rndne_f32_overlay",
    "_v_rsq_f32_overlay",
    "_v_sin_f32_overlay",
    "_v_sqrt_f32_overlay",
    "_v_trunc_f32_overlay",
    "_v_sub_f16_overlay",
    "_v_sub_f32_literal_overlay",
    "_v_sub_f32_overlay",
    "_v_sub_f32_src0_inline_overlay",
    "_v_sub_co_ci_u32_overlay",
    "_v_sub_co_u32_overlay",
    "_v_subrev_f32_literal_overlay",
    "_v_subrev_f32_overlay",
    "_v_subrev_f32_overlays",
    "_v_subrev_f32_src0_inline_overlay",
    "_v_sub_u32_lhs_tied_overlay",
    "_v_sub_u32_overlay",
    "_v_ternary_float_overlay",
    "_v_unary_f32_overlay",
    "_v_xor_b32_literal_overlay",
    "_v_xor_b32_overlay",
    "_v_xor_b32_src0_inline_overlay",
)
