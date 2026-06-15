# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


# ruff: noqa: F403, F405

"""Workgroup memory, LDS atomic, cross-lane, and transpose descriptor overlays."""

from __future__ import annotations

from .common import *


def _ds_read_overlay(
    *,
    width_bits: int,
    units: int,
    encoding_name: str = "ENC_DS",
    fixed_encoding_fields: tuple[tuple[str, AmdgpuFixedEncodingValue], ...] = (
        ("OFFSET1", 0),
        ("GDS", 0),
    ),
) -> AmdgpuDescriptorOverlay:
    suffix = f"b{width_bits}"
    return AmdgpuDescriptorOverlay(
        descriptor_key=f"amdgpu.ds_read_{suffix}",
        instruction_name=f"DS_READ_{suffix.upper()}",
        mnemonic=f"ds_read_{suffix}",
        encoding_name=encoding_name,
        semantic_tag=f"memory.workgroup.load.u{width_bits}",
        schedule_class=_SCHEDULE_LDS_LOAD,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result(units=units)),
            AmdgpuOperandOverlay("ADDR", _vgpr_operand("addr")),
        ),
        implicit_operands=(
            _ignore_workgroup_memory(width_bits=width_bits, is_input=True),
        ),
        immediates=(_ds_offset_immediate(),),
        fixed_encoding_fields=_ds_fixed_fields_without_offset1(fixed_encoding_fields),
        effects=(_workgroup_memory_effect(EffectKind.READ, width_bits),),
        constraints=_EARLY_CLOBBER_RESULT_CONSTRAINTS,
        flags=(DescriptorFlag.SIDE_EFFECTING,),
    )


def _ds_read_u16_overlay(
    *,
    encoding_name: str = "ENC_DS",
    fixed_encoding_fields: tuple[tuple[str, AmdgpuFixedEncodingValue], ...] = (
        ("OFFSET1", 0),
        ("GDS", 0),
    ),
) -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.ds_read_u16",
        instruction_name="DS_READ_U16",
        mnemonic="ds_read_u16",
        encoding_name=encoding_name,
        semantic_tag="memory.workgroup.load.u16",
        schedule_class=_SCHEDULE_LDS_LOAD,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            AmdgpuOperandOverlay("ADDR", _vgpr_operand("addr")),
        ),
        implicit_operands=(_ignore_workgroup_memory(width_bits=16, is_input=True),),
        immediates=(_ds_offset_immediate(),),
        fixed_encoding_fields=_ds_fixed_fields_without_offset1(fixed_encoding_fields),
        effects=(_workgroup_memory_effect(EffectKind.READ, 16),),
        constraints=_EARLY_CLOBBER_RESULT_CONSTRAINTS,
        flags=(DescriptorFlag.SIDE_EFFECTING,),
    )


def _ds_load_u16_d16_overlays(
    *,
    encoding_name: str = "ENC_DS",
    fixed_encoding_fields: tuple[tuple[str, AmdgpuFixedEncodingValue], ...] = (
        ("OFFSET1", 0),
        ("GDS", 0),
    ),
) -> tuple[AmdgpuDescriptorOverlay, ...]:
    return (
        AmdgpuDescriptorOverlay(
            descriptor_key="amdgpu.ds_load_u16_d16",
            instruction_name="DS_LOAD_U16_D16",
            mnemonic="ds_load_u16_d16",
            encoding_name=encoding_name,
            semantic_tag="memory.workgroup.load.u16.d16.low",
            schedule_class=_SCHEDULE_LDS_LOAD,
            operands=(
                AmdgpuOperandOverlay(
                    "VDST",
                    _vgpr_result(register_part=_REG_PART_VGPR_LOW16),
                    size_exception_reason=_D16_PARTIAL_REGISTER_SIZE_REASON,
                ),
                AmdgpuOperandOverlay("ADDR", _vgpr_operand("addr")),
            ),
            implicit_operands=(_ignore_workgroup_memory(width_bits=16, is_input=True),),
            immediates=(_ds_offset_immediate(),),
            fixed_encoding_fields=_ds_fixed_fields_without_offset1(
                fixed_encoding_fields
            ),
            asm_forms=_asm(
                results=("dst",),
                operands=("addr",),
                immediates=("offset",),
                named_immediates=True,
            ),
            effects=(_workgroup_memory_effect(EffectKind.READ, 16),),
            constraints=_EARLY_CLOBBER_RESULT_CONSTRAINTS,
            flags=(DescriptorFlag.SIDE_EFFECTING,),
        ),
        AmdgpuDescriptorOverlay(
            descriptor_key="amdgpu.ds_load_u16_d16_hi",
            instruction_name="DS_LOAD_U16_D16_HI",
            mnemonic="ds_load_u16_d16_hi",
            encoding_name=encoding_name,
            semantic_tag="memory.workgroup.load.u16.d16.high",
            schedule_class=_SCHEDULE_LDS_LOAD,
            operands=(
                AmdgpuOperandOverlay(
                    "VDST",
                    _vgpr_result(register_part=_REG_PART_VGPR_HIGH16),
                    size_exception_reason=_D16_PARTIAL_REGISTER_SIZE_REASON,
                ),
                AmdgpuOperandOverlay(
                    "VDST",
                    Operand(
                        "src",
                        OperandRole.OPERAND,
                        _VGPR_ALT,
                        flags=(OperandFlag.IMPLICIT,),
                        register_part=_REG_PART_VGPR_LOW16,
                    ),
                    role_exception_reason=(
                        "the encoded destination register is also the tied "
                        "source value carrying the low 16 bits"
                    ),
                    size_exception_reason=_D16_PARTIAL_REGISTER_SIZE_REASON,
                ),
                AmdgpuOperandOverlay("ADDR", _vgpr_operand("addr")),
            ),
            implicit_operands=(_ignore_workgroup_memory(width_bits=16, is_input=True),),
            immediates=(_ds_offset_immediate(),),
            fixed_encoding_fields=_ds_fixed_fields_without_offset1(
                fixed_encoding_fields
            ),
            asm_forms=_asm(
                results=("dst",),
                operands=("src", "addr"),
                immediates=("offset",),
                named_immediates=True,
            ),
            effects=(_workgroup_memory_effect(EffectKind.READ, 16),),
            constraints=(
                Constraint(ConstraintKind.TIED, 0, 1),
                *_EARLY_CLOBBER_RESULT_CONSTRAINTS,
            ),
            flags=(DescriptorFlag.SIDE_EFFECTING,),
        ),
    )


def _ds_write_b16_overlay(
    *,
    encoding_name: str = "ENC_DS",
    fixed_encoding_fields: tuple[tuple[str, AmdgpuFixedEncodingValue], ...] = (
        ("OFFSET1", 0),
        ("GDS", 0),
    ),
) -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.ds_write_b16",
        instruction_name="DS_WRITE_B16",
        mnemonic="ds_write_b16",
        encoding_name=encoding_name,
        semantic_tag="memory.workgroup.store.u16.low",
        schedule_class=_SCHEDULE_LDS_STORE,
        operands=(
            AmdgpuOperandOverlay("ADDR", _vgpr_operand("addr")),
            AmdgpuOperandOverlay(
                "DATA0",
                _vgpr_operand("value", register_part=_REG_PART_VGPR_LOW16),
            ),
        ),
        implicit_operands=(_ignore_workgroup_memory(width_bits=16, is_input=False),),
        immediates=(_ds_offset_immediate(),),
        fixed_encoding_fields=_ds_fixed_fields_without_offset1(fixed_encoding_fields),
        effects=(_workgroup_memory_effect(EffectKind.WRITE, 16),),
        flags=(DescriptorFlag.SIDE_EFFECTING,),
    )


def _ds_write_overlay(
    *,
    width_bits: int,
    units: int,
    encoding_name: str = "ENC_DS",
    fixed_encoding_fields: tuple[tuple[str, AmdgpuFixedEncodingValue], ...] = (
        ("OFFSET1", 0),
        ("GDS", 0),
    ),
) -> AmdgpuDescriptorOverlay:
    suffix = f"b{width_bits}"
    return AmdgpuDescriptorOverlay(
        descriptor_key=f"amdgpu.ds_write_{suffix}",
        instruction_name=f"DS_WRITE_{suffix.upper()}",
        mnemonic=f"ds_write_{suffix}",
        encoding_name=encoding_name,
        semantic_tag=f"memory.workgroup.store.u{width_bits}",
        schedule_class=_SCHEDULE_LDS_STORE,
        operands=(
            AmdgpuOperandOverlay("ADDR", _vgpr_operand("addr")),
            AmdgpuOperandOverlay("DATA0", _vgpr_operand("value", units=units)),
        ),
        implicit_operands=(
            _ignore_workgroup_memory(width_bits=width_bits, is_input=False),
        ),
        immediates=(_ds_offset_immediate(),),
        fixed_encoding_fields=_ds_fixed_fields_without_offset1(fixed_encoding_fields),
        effects=(_workgroup_memory_effect(EffectKind.WRITE, width_bits),),
        flags=(DescriptorFlag.SIDE_EFFECTING,),
    )


def _ds_read2_overlay(
    *,
    element_width_bits: int,
    value_units: int,
    offset_encoding_id: int | None = None,
    encoding_name: str = "ENC_DS",
    fixed_encoding_fields: tuple[tuple[str, AmdgpuFixedEncodingValue], ...] = (
        ("GDS", 0),
    ),
) -> AmdgpuDescriptorOverlay:
    suffix = f"b{element_width_bits}"
    if offset_encoding_id is None:
        offset_encoding_id = (
            _ADDRESS_OFFSET_DWORD_ENCODING_ID
            if element_width_bits == 32
            else _ADDRESS_OFFSET_QWORD_ENCODING_ID
        )
    return AmdgpuDescriptorOverlay(
        descriptor_key=f"amdgpu.ds_read2_{suffix}",
        instruction_name=f"DS_READ2_{suffix.upper()}",
        mnemonic=f"ds_read2_{suffix}",
        encoding_name=encoding_name,
        semantic_tag=f"memory.workgroup.load2.u{element_width_bits}",
        schedule_class=_SCHEDULE_LDS_LOAD,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result(units=value_units * 2)),
            AmdgpuOperandOverlay("ADDR", _vgpr_operand("addr")),
        ),
        implicit_operands=(
            _ignore_workgroup_memory(width_bits=element_width_bits, is_input=True),
        ),
        immediate_fields=("OFFSET0", "OFFSET1"),
        immediates=(
            _named_offset_immediate("offset0", 8, encoding_id=offset_encoding_id),
            _named_offset_immediate("offset1", 8, encoding_id=offset_encoding_id),
        ),
        fixed_encoding_fields=fixed_encoding_fields,
        effects=(_workgroup_memory_effect(EffectKind.READ, element_width_bits * 2),),
        constraints=_EARLY_CLOBBER_RESULT_CONSTRAINTS,
        flags=(DescriptorFlag.SIDE_EFFECTING,),
    )


def _ds_write2_overlay(
    *,
    element_width_bits: int,
    value_units: int,
    offset_encoding_id: int | None = None,
    encoding_name: str = "ENC_DS",
    fixed_encoding_fields: tuple[tuple[str, AmdgpuFixedEncodingValue], ...] = (
        ("GDS", 0),
    ),
) -> AmdgpuDescriptorOverlay:
    suffix = f"b{element_width_bits}"
    if offset_encoding_id is None:
        offset_encoding_id = (
            _ADDRESS_OFFSET_DWORD_ENCODING_ID
            if element_width_bits == 32
            else _ADDRESS_OFFSET_QWORD_ENCODING_ID
        )
    return AmdgpuDescriptorOverlay(
        descriptor_key=f"amdgpu.ds_write2_{suffix}",
        instruction_name=f"DS_WRITE2_{suffix.upper()}",
        mnemonic=f"ds_write2_{suffix}",
        encoding_name=encoding_name,
        semantic_tag=f"memory.workgroup.store2.u{element_width_bits}",
        schedule_class=_SCHEDULE_LDS_STORE,
        operands=(
            AmdgpuOperandOverlay("ADDR", _vgpr_operand("addr")),
            AmdgpuOperandOverlay("DATA0", _vgpr_operand("value0", units=value_units)),
            AmdgpuOperandOverlay("DATA1", _vgpr_operand("value1", units=value_units)),
        ),
        implicit_operands=(
            _ignore_workgroup_memory(width_bits=element_width_bits, is_input=False),
        ),
        immediate_fields=("OFFSET0", "OFFSET1"),
        immediates=(
            _named_offset_immediate("offset0", 8, encoding_id=offset_encoding_id),
            _named_offset_immediate("offset1", 8, encoding_id=offset_encoding_id),
        ),
        fixed_encoding_fields=fixed_encoding_fields,
        effects=(_workgroup_memory_effect(EffectKind.WRITE, element_width_bits * 2),),
        flags=(DescriptorFlag.SIDE_EFFECTING,),
    )


def _ds_atomic_overlay(
    *,
    descriptor_key: str,
    instruction_name: str,
    mnemonic: str,
    semantic_tag: str,
    data_format_name: str,
    returns_old_value: bool,
    encoding_name: str = "ENC_DS",
    fixed_encoding_fields: tuple[tuple[str, AmdgpuFixedEncodingValue], ...] = (
        ("OFFSET1", 0),
        ("GDS", 0),
    ),
) -> AmdgpuDescriptorOverlay:
    operands: tuple[AmdgpuOperandOverlay, ...]
    if returns_old_value:
        operands = (
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            AmdgpuOperandOverlay("ADDR", _vgpr_operand("addr")),
            AmdgpuOperandOverlay("DATA0", _vgpr_operand("value")),
        )
    else:
        operands = (
            AmdgpuOperandOverlay("ADDR", _vgpr_operand("addr")),
            AmdgpuOperandOverlay("DATA0", _vgpr_operand("value")),
        )
    return AmdgpuDescriptorOverlay(
        descriptor_key=descriptor_key,
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        encoding_name=encoding_name,
        semantic_tag=semantic_tag,
        schedule_class=_SCHEDULE_LDS_ATOMIC,
        operands=operands,
        implicit_operands=(
            _ignore_workgroup_memory(
                width_bits=32, is_input=False, data_format_name=data_format_name
            ),
            _ignore_workgroup_memory(
                width_bits=32, is_input=True, data_format_name=data_format_name
            ),
        ),
        immediates=(_ds_offset_immediate(),),
        fixed_encoding_fields=_ds_fixed_fields_without_offset1(fixed_encoding_fields),
        effects=(
            _workgroup_memory_effect(EffectKind.READ, 32),
            _workgroup_memory_effect(EffectKind.WRITE, 32),
        ),
        flags=(DescriptorFlag.SIDE_EFFECTING,),
    )


def _ds_atomic_cmpstore_overlay(
    *,
    encoding_name: str = "ENC_DS",
    fixed_encoding_fields: tuple[tuple[str, AmdgpuFixedEncodingValue], ...] = (
        ("OFFSET1", 0),
        ("GDS", 0),
    ),
) -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.ds_cmpst_rtn_b32",
        instruction_name="DS_CMPST_RTN_B32",
        mnemonic="ds_cmpst_rtn_b32",
        encoding_name=encoding_name,
        semantic_tag="memory.workgroup.atomic.compare_exchange.b32.return",
        schedule_class=_SCHEDULE_LDS_ATOMIC,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            AmdgpuOperandOverlay("ADDR", _vgpr_operand("addr")),
            AmdgpuOperandOverlay("DATA0", _vgpr_operand("expected")),
            AmdgpuOperandOverlay("DATA1", _vgpr_operand("replacement")),
        ),
        implicit_operands=(
            _ignore_workgroup_memory(
                width_bits=32, is_input=False, data_format_name="FMT_NUM_B32"
            ),
            _ignore_workgroup_memory(
                width_bits=32, is_input=True, data_format_name="FMT_NUM_B32"
            ),
        ),
        immediates=(_ds_offset_immediate(),),
        fixed_encoding_fields=_ds_fixed_fields_without_offset1(fixed_encoding_fields),
        effects=(
            _workgroup_memory_effect(EffectKind.READ, 32),
            _workgroup_memory_effect(EffectKind.WRITE, 32),
        ),
        flags=(DescriptorFlag.SIDE_EFFECTING,),
    )


def _ds_atomic_overlays(
    *,
    encoding_name: str = "ENC_DS",
    fixed_encoding_fields: tuple[tuple[str, AmdgpuFixedEncodingValue], ...] = (
        ("OFFSET1", 0),
        ("GDS", 0),
    ),
    include_packed_half_add: bool = False,
) -> tuple[AmdgpuDescriptorOverlay, ...]:
    rows = [
        ("ds_add_u32", "DS_ADD_U32", "add.u32", "FMT_NUM_U32", False),
        ("ds_sub_u32", "DS_SUB_U32", "sub.u32", "FMT_NUM_U32", False),
        ("ds_min_i32", "DS_MIN_I32", "min.i32", "FMT_NUM_I32", False),
        ("ds_max_i32", "DS_MAX_I32", "max.i32", "FMT_NUM_I32", False),
        ("ds_min_u32", "DS_MIN_U32", "min.u32", "FMT_NUM_U32", False),
        ("ds_max_u32", "DS_MAX_U32", "max.u32", "FMT_NUM_U32", False),
        ("ds_and_b32", "DS_AND_B32", "and.b32", "FMT_NUM_B32", False),
        ("ds_or_b32", "DS_OR_B32", "or.b32", "FMT_NUM_B32", False),
        ("ds_xor_b32", "DS_XOR_B32", "xor.b32", "FMT_NUM_B32", False),
        ("ds_add_f32", "DS_ADD_F32", "add.f32", "FMT_NUM_F32", False),
        ("ds_min_f32", "DS_MIN_F32", "minnum.f32", "FMT_NUM_F32", False),
        ("ds_max_f32", "DS_MAX_F32", "maxnum.f32", "FMT_NUM_F32", False),
        ("ds_add_rtn_u32", "DS_ADD_RTN_U32", "add.return.u32", "FMT_NUM_U32", True),
        ("ds_sub_rtn_u32", "DS_SUB_RTN_U32", "sub.return.u32", "FMT_NUM_U32", True),
        ("ds_min_rtn_i32", "DS_MIN_RTN_I32", "min.return.i32", "FMT_NUM_I32", True),
        ("ds_max_rtn_i32", "DS_MAX_RTN_I32", "max.return.i32", "FMT_NUM_I32", True),
        ("ds_min_rtn_u32", "DS_MIN_RTN_U32", "min.return.u32", "FMT_NUM_U32", True),
        ("ds_max_rtn_u32", "DS_MAX_RTN_U32", "max.return.u32", "FMT_NUM_U32", True),
        ("ds_and_rtn_b32", "DS_AND_RTN_B32", "and.return.b32", "FMT_NUM_B32", True),
        ("ds_or_rtn_b32", "DS_OR_RTN_B32", "or.return.b32", "FMT_NUM_B32", True),
        ("ds_xor_rtn_b32", "DS_XOR_RTN_B32", "xor.return.b32", "FMT_NUM_B32", True),
        (
            "ds_wrxchg_rtn_b32",
            "DS_WRXCHG_RTN_B32",
            "exchange.return.b32",
            "FMT_NUM_B32",
            True,
        ),
        ("ds_add_rtn_f32", "DS_ADD_RTN_F32", "add.return.f32", "FMT_NUM_F32", True),
        (
            "ds_min_rtn_f32",
            "DS_MIN_RTN_F32",
            "minnum.return.f32",
            "FMT_NUM_F32",
            True,
        ),
        (
            "ds_max_rtn_f32",
            "DS_MAX_RTN_F32",
            "maxnum.return.f32",
            "FMT_NUM_F32",
            True,
        ),
    ]
    if include_packed_half_add:
        rows.extend(
            (
                (
                    "ds_pk_add_f16",
                    "DS_PK_ADD_F16",
                    "add.pk2.f16",
                    "FMT_NUM_PK2_F16",
                    False,
                ),
                (
                    "ds_pk_add_bf16",
                    "DS_PK_ADD_BF16",
                    "add.pk2.bf16",
                    "FMT_NUM_PK2_BF16",
                    False,
                ),
                (
                    "ds_pk_add_rtn_f16",
                    "DS_PK_ADD_RTN_F16",
                    "add.return.pk2.f16",
                    "FMT_NUM_PK2_F16",
                    True,
                ),
                (
                    "ds_pk_add_rtn_bf16",
                    "DS_PK_ADD_RTN_BF16",
                    "add.return.pk2.bf16",
                    "FMT_NUM_PK2_BF16",
                    True,
                ),
            )
        )
    overlays = [
        _ds_atomic_overlay(
            descriptor_key=f"amdgpu.{mnemonic}",
            instruction_name=instruction_name,
            mnemonic=mnemonic,
            semantic_tag=f"memory.workgroup.atomic.{semantic_suffix}",
            data_format_name=data_format_name,
            returns_old_value=returns_old_value,
            encoding_name=encoding_name,
            fixed_encoding_fields=fixed_encoding_fields,
        )
        for (
            mnemonic,
            instruction_name,
            semantic_suffix,
            data_format_name,
            returns_old_value,
        ) in rows
    ]
    overlays.append(
        _ds_atomic_cmpstore_overlay(
            encoding_name=encoding_name,
            fixed_encoding_fields=fixed_encoding_fields,
        )
    )
    return tuple(overlays)


def _ds_stride64_read2_overlay(
    *,
    element_width_bits: int,
    value_units: int,
    encoding_name: str = "ENC_DS",
    fixed_encoding_fields: tuple[tuple[str, AmdgpuFixedEncodingValue], ...] = (
        ("GDS", 0),
    ),
) -> AmdgpuDescriptorOverlay:
    suffix = f"b{element_width_bits}"
    offset_encoding_id = (
        _ADDRESS_OFFSET_DWORD_STRIDE64_ENCODING_ID
        if element_width_bits == 32
        else _ADDRESS_OFFSET_QWORD_STRIDE64_ENCODING_ID
    )
    return replace(
        _ds_read2_overlay(
            element_width_bits=element_width_bits,
            value_units=value_units,
            offset_encoding_id=offset_encoding_id,
            encoding_name=encoding_name,
            fixed_encoding_fields=fixed_encoding_fields,
        ),
        descriptor_key=f"amdgpu.ds_read2st64_{suffix}",
        instruction_name=f"DS_READ2ST64_{suffix.upper()}",
        mnemonic=f"ds_read2st64_{suffix}",
        semantic_tag=f"memory.workgroup.load2.stride64.u{element_width_bits}",
    )


def _ds_stride64_write2_overlay(
    *,
    element_width_bits: int,
    value_units: int,
    encoding_name: str = "ENC_DS",
    fixed_encoding_fields: tuple[tuple[str, AmdgpuFixedEncodingValue], ...] = (
        ("GDS", 0),
    ),
) -> AmdgpuDescriptorOverlay:
    suffix = f"b{element_width_bits}"
    offset_encoding_id = (
        _ADDRESS_OFFSET_DWORD_STRIDE64_ENCODING_ID
        if element_width_bits == 32
        else _ADDRESS_OFFSET_QWORD_STRIDE64_ENCODING_ID
    )
    return replace(
        _ds_write2_overlay(
            element_width_bits=element_width_bits,
            value_units=value_units,
            offset_encoding_id=offset_encoding_id,
            encoding_name=encoding_name,
            fixed_encoding_fields=fixed_encoding_fields,
        ),
        descriptor_key=f"amdgpu.ds_write2st64_{suffix}",
        instruction_name=f"DS_WRITE2ST64_{suffix.upper()}",
        mnemonic=f"ds_write2st64_{suffix}",
        semantic_tag=f"memory.workgroup.store2.stride64.u{element_width_bits}",
    )


def _ds_read_addtid_b32_overlay(
    *,
    encoding_name: str = "ENC_DS",
    fixed_encoding_fields: tuple[tuple[str, AmdgpuFixedEncodingValue], ...] = (
        ("OFFSET1", 0),
        ("GDS", 0),
    ),
) -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.ds_read_addtid_b32",
        instruction_name="DS_READ_ADDTID_B32",
        mnemonic="ds_read_addtid_b32",
        encoding_name=encoding_name,
        semantic_tag="memory.workgroup.load.addtid.u32",
        schedule_class=_SCHEDULE_LDS_LOAD,
        operands=(AmdgpuOperandOverlay("VDST", _vgpr_result()),),
        implicit_operands=(
            _ignore_workgroup_memory(width_bits=32, is_input=True),
            _implicit_m0_input(),
        ),
        immediates=(_ds_offset_immediate(),),
        asm_forms=_asm(results=("dst",), operands=("m0",), immediates=("offset",)),
        fixed_encoding_fields=_ds_fixed_fields_without_offset1(fixed_encoding_fields),
        effects=(_workgroup_memory_effect(EffectKind.READ, 32),),
        flags=(DescriptorFlag.SIDE_EFFECTING,),
    )


def _ds_write_addtid_b32_overlay(
    *,
    encoding_name: str = "ENC_DS",
    fixed_encoding_fields: tuple[tuple[str, AmdgpuFixedEncodingValue], ...] = (
        ("OFFSET1", 0),
        ("GDS", 0),
    ),
) -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.ds_write_addtid_b32",
        instruction_name="DS_WRITE_ADDTID_B32",
        mnemonic="ds_write_addtid_b32",
        encoding_name=encoding_name,
        semantic_tag="memory.workgroup.store.addtid.u32",
        schedule_class=_SCHEDULE_LDS_STORE,
        operands=(AmdgpuOperandOverlay("DATA0", _vgpr_operand("value")),),
        implicit_operands=(
            _ignore_workgroup_memory(width_bits=32, is_input=False),
            _implicit_m0_input(),
        ),
        immediates=(_ds_offset_immediate(),),
        asm_forms=_asm(operands=("value", "m0"), immediates=("offset",)),
        fixed_encoding_fields=_ds_fixed_fields_without_offset1(fixed_encoding_fields),
        effects=(_workgroup_memory_effect(EffectKind.WRITE, 32),),
        flags=(DescriptorFlag.SIDE_EFFECTING,),
    )


def _ds_crosslane_offset_immediate() -> Immediate:
    return _ds_offset_immediate()


def _ds_swizzle_b32_overlay(
    *,
    encoding_name: str = "ENC_DS",
    fixed_encoding_fields: tuple[tuple[str, AmdgpuFixedEncodingValue], ...] = (
        ("GDS", 0),
    ),
) -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.ds_swizzle_b32",
        instruction_name="DS_SWIZZLE_B32",
        mnemonic="ds_swizzle_b32",
        encoding_name=encoding_name,
        semantic_tag="memory.crosslane.swizzle.u32",
        schedule_class=_SCHEDULE_LDS_CROSSLANE,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            AmdgpuOperandOverlay("ADDR", _vgpr_operand("addr")),
        ),
        immediates=(_ds_crosslane_offset_immediate(),),
        fixed_encoding_fields=fixed_encoding_fields,
        effects=_ds_crosslane_effects(32),
    )


def _ds_permute_b32_overlay(
    *,
    descriptor_key: str = "amdgpu.ds_permute_b32",
    instruction_name: str = "DS_PERMUTE_B32",
    mnemonic: str = "ds_permute_b32",
    semantic_tag: str = "memory.crosslane.permute.u32",
    encoding_name: str = "ENC_DS",
    fixed_encoding_fields: tuple[tuple[str, AmdgpuFixedEncodingValue], ...] = (
        ("GDS", 0),
    ),
) -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key=descriptor_key,
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        encoding_name=encoding_name,
        semantic_tag=semantic_tag,
        schedule_class=_SCHEDULE_LDS_CROSSLANE,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            AmdgpuOperandOverlay("ADDR", _vgpr_operand("addr")),
            AmdgpuOperandOverlay("DATA0", _vgpr_operand("value")),
        ),
        immediates=(_ds_crosslane_offset_immediate(),),
        fixed_encoding_fields=fixed_encoding_fields,
        effects=_ds_crosslane_effects(32),
    )


def _ds_bpermute_b32_overlay(
    *,
    descriptor_key: str = "amdgpu.ds_bpermute_b32",
    instruction_name: str = "DS_BPERMUTE_B32",
    mnemonic: str = "ds_bpermute_b32",
    semantic_tag: str = "memory.crosslane.bpermute.u32",
    encoding_name: str = "ENC_DS",
    fixed_encoding_fields: tuple[tuple[str, AmdgpuFixedEncodingValue], ...] = (
        ("GDS", 0),
    ),
) -> AmdgpuDescriptorOverlay:
    return _ds_permute_b32_overlay(
        descriptor_key=descriptor_key,
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        semantic_tag=semantic_tag,
        encoding_name=encoding_name,
        fixed_encoding_fields=fixed_encoding_fields,
    )


def _ds_bpermute_fi_b32_overlay(
    *,
    encoding_name: str = "ENC_VDS",
    fixed_encoding_fields: tuple[tuple[str, AmdgpuFixedEncodingValue], ...] = (),
) -> AmdgpuDescriptorOverlay:
    return _ds_bpermute_b32_overlay(
        descriptor_key="amdgpu.ds_bpermute_fi_b32",
        instruction_name="DS_BPERMUTE_FI_B32",
        mnemonic="ds_bpermute_fi_b32",
        semantic_tag="memory.crosslane.bpermute.fi.u32",
        encoding_name=encoding_name,
        fixed_encoding_fields=fixed_encoding_fields,
    )


def _ds_crosslane_overlays(
    *,
    encoding_name: str = "ENC_DS",
    fixed_encoding_fields: tuple[tuple[str, AmdgpuFixedEncodingValue], ...] = (
        ("GDS", 0),
    ),
    include_fetch_invalid: bool = False,
) -> tuple[AmdgpuDescriptorOverlay, ...]:
    overlays = (
        _ds_swizzle_b32_overlay(
            encoding_name=encoding_name,
            fixed_encoding_fields=fixed_encoding_fields,
        ),
        _ds_permute_b32_overlay(
            encoding_name=encoding_name,
            fixed_encoding_fields=fixed_encoding_fields,
        ),
        _ds_bpermute_b32_overlay(
            encoding_name=encoding_name,
            fixed_encoding_fields=fixed_encoding_fields,
        ),
    )
    if not include_fetch_invalid:
        return overlays
    return (
        *overlays,
        _ds_bpermute_fi_b32_overlay(
            encoding_name=encoding_name,
            fixed_encoding_fields=fixed_encoding_fields,
        ),
    )


def _ds_transpose_read_overlay(
    *,
    descriptor_key: str,
    instruction_name: str,
    mnemonic: str,
    semantic_tag: str,
    width_bits: int,
    units: int,
    encoding_name: str = "ENC_DS",
    fixed_encoding_fields: tuple[tuple[str, AmdgpuFixedEncodingValue], ...] = (
        ("GDS", 0),
    ),
) -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key=descriptor_key,
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        encoding_name=encoding_name,
        semantic_tag=semantic_tag,
        schedule_class=_SCHEDULE_LDS_LOAD,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result(units=units)),
            AmdgpuOperandOverlay("ADDR", _vgpr_operand("addr")),
        ),
        implicit_operands=(
            _ignore_workgroup_memory(width_bits=width_bits, is_input=True),
        ),
        immediates=(_ds_crosslane_offset_immediate(),),
        fixed_encoding_fields=fixed_encoding_fields,
        effects=(_workgroup_memory_effect(EffectKind.READ, width_bits),),
        constraints=_EARLY_CLOBBER_RESULT_CONSTRAINTS,
        flags=(DescriptorFlag.SIDE_EFFECTING,),
    )


def _gfx950_ds_transpose_read_overlays() -> tuple[AmdgpuDescriptorOverlay, ...]:
    return (
        _ds_transpose_read_overlay(
            descriptor_key="amdgpu.ds_read_b64_tr_b4",
            instruction_name="DS_READ_B64_TR_B4",
            mnemonic="ds_read_b64_tr_b4",
            semantic_tag="memory.workgroup.transpose.load.b4.u64",
            width_bits=64,
            units=2,
        ),
        _ds_transpose_read_overlay(
            descriptor_key="amdgpu.ds_read_b96_tr_b6",
            instruction_name="DS_READ_B96_TR_B6",
            mnemonic="ds_read_b96_tr_b6",
            semantic_tag="memory.workgroup.transpose.load.b6.u96",
            width_bits=96,
            units=3,
        ),
        _ds_transpose_read_overlay(
            descriptor_key="amdgpu.ds_read_b64_tr_b8",
            instruction_name="DS_READ_B64_TR_B8",
            mnemonic="ds_read_b64_tr_b8",
            semantic_tag="memory.workgroup.transpose.load.b8.u64",
            width_bits=64,
            units=2,
        ),
        _ds_transpose_read_overlay(
            descriptor_key="amdgpu.ds_read_b64_tr_b16",
            instruction_name="DS_READ_B64_TR_B16",
            mnemonic="ds_read_b64_tr_b16",
            semantic_tag="memory.workgroup.transpose.load.b16.u64",
            width_bits=64,
            units=2,
        ),
    )


def _ds_memory_overlays(
    *,
    encoding_name: str = "ENC_DS",
    fixed_encoding_fields: tuple[tuple[str, AmdgpuFixedEncodingValue], ...] = (
        ("OFFSET1", 0),
        ("GDS", 0),
    ),
    include_packed_half_atomic_add: bool = False,
    include_u16_d16_loads: bool = False,
) -> tuple[AmdgpuDescriptorOverlay, ...]:
    widths = ((32, 1), (64, 2), (96, 3), (128, 4))
    two_addr_widths = ((32, 1), (64, 2))
    two_addr_fixed_encoding_fields = _ds_fixed_fields_without_offset1(
        fixed_encoding_fields
    )
    return (
        _ds_read_u16_overlay(
            encoding_name=encoding_name,
            fixed_encoding_fields=fixed_encoding_fields,
        ),
        *(
            _ds_read_overlay(
                width_bits=width_bits,
                units=units,
                encoding_name=encoding_name,
                fixed_encoding_fields=fixed_encoding_fields,
            )
            for width_bits, units in widths
        ),
        _ds_write_b16_overlay(
            encoding_name=encoding_name,
            fixed_encoding_fields=fixed_encoding_fields,
        ),
        *(
            _ds_write_overlay(
                width_bits=width_bits,
                units=units,
                encoding_name=encoding_name,
                fixed_encoding_fields=fixed_encoding_fields,
            )
            for width_bits, units in widths
        ),
        *_ds_atomic_overlays(
            encoding_name=encoding_name,
            fixed_encoding_fields=fixed_encoding_fields,
            include_packed_half_add=include_packed_half_atomic_add,
        ),
        *(
            _ds_load_u16_d16_overlays(
                encoding_name=encoding_name,
                fixed_encoding_fields=fixed_encoding_fields,
            )
            if include_u16_d16_loads
            else ()
        ),
        *(
            _ds_read2_overlay(
                element_width_bits=width_bits,
                value_units=units,
                encoding_name=encoding_name,
                fixed_encoding_fields=two_addr_fixed_encoding_fields,
            )
            for width_bits, units in two_addr_widths
        ),
        *(
            _ds_stride64_read2_overlay(
                element_width_bits=width_bits,
                value_units=units,
                encoding_name=encoding_name,
                fixed_encoding_fields=two_addr_fixed_encoding_fields,
            )
            for width_bits, units in two_addr_widths
        ),
        *(
            _ds_write2_overlay(
                element_width_bits=width_bits,
                value_units=units,
                encoding_name=encoding_name,
                fixed_encoding_fields=two_addr_fixed_encoding_fields,
            )
            for width_bits, units in two_addr_widths
        ),
        *(
            _ds_stride64_write2_overlay(
                element_width_bits=width_bits,
                value_units=units,
                encoding_name=encoding_name,
                fixed_encoding_fields=two_addr_fixed_encoding_fields,
            )
            for width_bits, units in two_addr_widths
        ),
        _ds_read_addtid_b32_overlay(
            encoding_name=encoding_name,
            fixed_encoding_fields=fixed_encoding_fields,
        ),
        _ds_write_addtid_b32_overlay(
            encoding_name=encoding_name,
            fixed_encoding_fields=fixed_encoding_fields,
        ),
    )


__all__ = (
    "_ds_atomic_cmpstore_overlay",
    "_ds_atomic_overlay",
    "_ds_atomic_overlays",
    "_ds_bpermute_b32_overlay",
    "_ds_bpermute_fi_b32_overlay",
    "_ds_crosslane_offset_immediate",
    "_ds_crosslane_overlays",
    "_ds_load_u16_d16_overlays",
    "_ds_memory_overlays",
    "_ds_permute_b32_overlay",
    "_ds_read2_overlay",
    "_ds_read_addtid_b32_overlay",
    "_ds_read_overlay",
    "_ds_read_u16_overlay",
    "_ds_stride64_read2_overlay",
    "_ds_stride64_write2_overlay",
    "_ds_swizzle_b32_overlay",
    "_ds_transpose_read_overlay",
    "_ds_write2_overlay",
    "_ds_write_addtid_b32_overlay",
    "_ds_write_b16_overlay",
    "_ds_write_overlay",
    "_gfx950_ds_transpose_read_overlays",
)
