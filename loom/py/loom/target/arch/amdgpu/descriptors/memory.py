# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


# ruff: noqa: F403, F405

"""Scalar, buffer, global, and global-to-LDS memory descriptor overlays."""

from __future__ import annotations

from .common import *

_MEMORY_DWORD_VECTOR_WIDTHS = ((32, 1), (64, 2), (96, 3), (128, 4))
_SMEM_DWORDX4_WIDTHS = (
    (32, 1, "dword", "DWORD"),
    (64, 2, "dwordx2", "DWORDX2"),
    (128, 4, "dwordx4", "DWORDX4"),
)

_SMEM_NARROW_LOAD_ROWS: tuple[
    tuple[str, str, int, str, AmdgpuImplicitOperandOverlay], ...
] = (
    ("i8", "I8", 8, "memory.load.i8.sign_extend", _IGNORE_GLOBAL_READ_MEMORY_I8),
    ("u8", "U8", 8, "memory.load.u8.zero_extend", _IGNORE_GLOBAL_READ_MEMORY_U8),
    ("i16", "I16", 16, "memory.load.i16.sign_extend", _IGNORE_GLOBAL_READ_MEMORY_I16),
    (
        "u16",
        "U16",
        16,
        "memory.load.u16.zero_extend",
        _IGNORE_GLOBAL_READ_MEMORY_U16,
    ),
)

_VMEM_NARROW_BYTE_LOAD_ROWS: tuple[
    tuple[str, str, int, str, AmdgpuImplicitOperandOverlay], ...
] = (
    ("u8", "UBYTE", 8, "memory.load.u8.zero_extend", _IGNORE_GLOBAL_READ_MEMORY_U8),
    ("i8", "SBYTE", 8, "memory.load.i8.sign_extend", _IGNORE_GLOBAL_READ_MEMORY_I8),
)

_VMEM_NARROW_B16_LOAD_ROWS: tuple[
    tuple[str, str, int, str, AmdgpuImplicitOperandOverlay], ...
] = (
    (
        "u16",
        "USHORT",
        16,
        "memory.load.u16.zero_extend",
        _IGNORE_GLOBAL_READ_MEMORY_U16,
    ),
    (
        "i16",
        "SSHORT",
        16,
        "memory.load.i16.sign_extend",
        _IGNORE_GLOBAL_READ_MEMORY_I16,
    ),
)

_SCRATCH_NARROW_BYTE_LOAD_ROWS: tuple[tuple[str, str, int, str, str], ...] = (
    ("u8", "UBYTE", 8, "memory.stack.load.u8.zero_extend", "FMT_NUM_U8"),
    ("i8", "SBYTE", 8, "memory.stack.load.i8.sign_extend", "FMT_NUM_I8"),
)

_SCRATCH_NARROW_B16_LOAD_ROWS: tuple[tuple[str, str, int, str, str], ...] = (
    ("u16", "USHORT", 16, "memory.stack.load.u16.zero_extend", "FMT_NUM_U16"),
    ("i16", "SSHORT", 16, "memory.stack.load.i16.sign_extend", "FMT_NUM_I16"),
)


def _buffer_operand_forms(
    *,
    off_zero_descriptor_key: str | None = None,
    vaddr_offset_descriptor_key: str | None = None,
) -> tuple[OperandForm, ...]:
    operand_forms: list[OperandForm] = []
    if off_zero_descriptor_key is not None:
        operand_forms.append(
            _buffer_off_zero_operand_form(
                replacement_descriptor=off_zero_descriptor_key
            )
        )
    if vaddr_offset_descriptor_key is not None:
        operand_forms.append(
            _buffer_soffset_offset_operand_form(
                replacement_descriptor=vaddr_offset_descriptor_key
            )
        )
    return tuple(operand_forms)


def _buffer_vaddr_offset_native_assembly_values(
    *,
    payload_kind: NativeAsmValueKind,
    payload_field_name: str,
    fixed_soffset_native_spelling: str,
) -> tuple[NativeAsmValue, ...]:
    return (
        NativeAsmValue(payload_kind, field_name=payload_field_name),
        _native_operand("vaddr"),
        _native_operand("resource"),
        _native_literal(f"{fixed_soffset_native_spelling} offen"),
    )


def _s_buffer_load_sized_overlay(
    *,
    descriptor_key: str,
    instruction_name: str,
    mnemonic: str,
    width_bits: int,
    payload_units: int,
    semantic_tag: str | None = None,
    implicit_memory: AmdgpuImplicitOperandOverlay | None = None,
    offset_field_name: str = "OFFSET",
    offset_bit_width: int = 21,
    fixed_encoding_fields: tuple[tuple[str, AmdgpuFixedEncodingValue], ...] = (),
) -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key=descriptor_key,
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        encoding_name="ENC_SMEM",
        semantic_tag=semantic_tag or f"memory.load.u{width_bits}",
        schedule_class=_SCHEDULE_SMEM_LOAD,
        operands=(
            AmdgpuOperandOverlay("SDATA", _sgpr_result(units=payload_units)),
            AmdgpuOperandOverlay("SBASE", _sgpr_resource("resource", units=4)),
            AmdgpuOperandOverlay("SOFFSET", _sgpr_operand("soffset")),
        ),
        implicit_operands=(implicit_memory or _ignore_global_read_memory(width_bits),),
        immediate_fields=(offset_field_name,),
        immediates=(_offset_immediate(offset_bit_width),),
        fixed_encoding_fields=fixed_encoding_fields,
        effects=(_global_read_effect(width_bits),),
        constraints=_EARLY_CLOBBER_RESULT_CONSTRAINTS,
        flags=(DescriptorFlag.SIDE_EFFECTING,),
    )


def _s_buffer_load_narrow_overlays(
    offset_field_name: str = "OFFSET",
    *,
    offset_bit_width: int = 21,
    fixed_encoding_fields: tuple[tuple[str, AmdgpuFixedEncodingValue], ...] = (),
) -> tuple[AmdgpuDescriptorOverlay, ...]:
    return tuple(
        _s_buffer_load_sized_overlay(
            descriptor_key=f"amdgpu.s_buffer_load_{mnemonic_suffix}",
            instruction_name=f"S_BUFFER_LOAD_{instruction_suffix}",
            mnemonic=f"s_buffer_load_{mnemonic_suffix}",
            width_bits=width_bits,
            payload_units=1,
            semantic_tag=semantic_tag,
            implicit_memory=implicit_memory,
            offset_field_name=offset_field_name,
            offset_bit_width=offset_bit_width,
            fixed_encoding_fields=fixed_encoding_fields,
        )
        for (
            mnemonic_suffix,
            instruction_suffix,
            width_bits,
            semantic_tag,
            implicit_memory,
        ) in _SMEM_NARROW_LOAD_ROWS
    )


def _s_buffer_load_dword_overlay(
    offset_field_name: str = "OFFSET",
    offset_bit_width: int = 21,
    fixed_encoding_fields: tuple[tuple[str, AmdgpuFixedEncodingValue], ...] = (),
) -> AmdgpuDescriptorOverlay:
    return _s_buffer_load_sized_overlay(
        descriptor_key="amdgpu.s_buffer_load_dword",
        instruction_name="S_BUFFER_LOAD_DWORD",
        mnemonic="s_buffer_load_dword",
        width_bits=32,
        payload_units=1,
        offset_field_name=offset_field_name,
        offset_bit_width=offset_bit_width,
        fixed_encoding_fields=fixed_encoding_fields,
    )


def _s_buffer_load_64_overlay(
    *,
    descriptor_key: str = "amdgpu.s_buffer_load_b64",
    instruction_name: str = "S_BUFFER_LOAD_B64",
    mnemonic: str = "s_buffer_load_b64",
    offset_field_name: str = "OFFSET",
    offset_bit_width: int = 21,
    fixed_encoding_fields: tuple[tuple[str, AmdgpuFixedEncodingValue], ...] = (),
) -> AmdgpuDescriptorOverlay:
    return _s_buffer_load_sized_overlay(
        descriptor_key=descriptor_key,
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        width_bits=64,
        payload_units=2,
        offset_field_name=offset_field_name,
        offset_bit_width=offset_bit_width,
        fixed_encoding_fields=fixed_encoding_fields,
    )


def _s_buffer_load_96_overlay(
    *,
    descriptor_key: str = "amdgpu.s_buffer_load_b96",
    instruction_name: str = "S_BUFFER_LOAD_B96",
    mnemonic: str = "s_buffer_load_b96",
    offset_field_name: str = "OFFSET",
    offset_bit_width: int = 21,
    fixed_encoding_fields: tuple[tuple[str, AmdgpuFixedEncodingValue], ...] = (),
) -> AmdgpuDescriptorOverlay:
    return _s_buffer_load_sized_overlay(
        descriptor_key=descriptor_key,
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        width_bits=96,
        payload_units=3,
        offset_field_name=offset_field_name,
        offset_bit_width=offset_bit_width,
        fixed_encoding_fields=fixed_encoding_fields,
    )


def _s_buffer_load_128_overlay(
    *,
    descriptor_key: str = "amdgpu.s_buffer_load_b128",
    instruction_name: str = "S_BUFFER_LOAD_B128",
    mnemonic: str = "s_buffer_load_b128",
    offset_field_name: str = "OFFSET",
    offset_bit_width: int = 21,
    fixed_encoding_fields: tuple[tuple[str, AmdgpuFixedEncodingValue], ...] = (),
) -> AmdgpuDescriptorOverlay:
    return _s_buffer_load_sized_overlay(
        descriptor_key=descriptor_key,
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        width_bits=128,
        payload_units=4,
        offset_field_name=offset_field_name,
        offset_bit_width=offset_bit_width,
        fixed_encoding_fields=fixed_encoding_fields,
    )


def _s_buffer_load_256_overlay(
    *,
    descriptor_key: str = "amdgpu.s_buffer_load_b256",
    instruction_name: str = "S_BUFFER_LOAD_B256",
    mnemonic: str = "s_buffer_load_b256",
    offset_field_name: str = "OFFSET",
    offset_bit_width: int = 21,
    fixed_encoding_fields: tuple[tuple[str, AmdgpuFixedEncodingValue], ...] = (),
) -> AmdgpuDescriptorOverlay:
    return _s_buffer_load_sized_overlay(
        descriptor_key=descriptor_key,
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        width_bits=256,
        payload_units=8,
        offset_field_name=offset_field_name,
        offset_bit_width=offset_bit_width,
        fixed_encoding_fields=fixed_encoding_fields,
    )


def _s_buffer_load_512_overlay(
    *,
    descriptor_key: str = "amdgpu.s_buffer_load_b512",
    instruction_name: str = "S_BUFFER_LOAD_B512",
    mnemonic: str = "s_buffer_load_b512",
    offset_field_name: str = "OFFSET",
    offset_bit_width: int = 21,
    fixed_encoding_fields: tuple[tuple[str, AmdgpuFixedEncodingValue], ...] = (),
) -> AmdgpuDescriptorOverlay:
    return _s_buffer_load_sized_overlay(
        descriptor_key=descriptor_key,
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        width_bits=512,
        payload_units=16,
        offset_field_name=offset_field_name,
        offset_bit_width=offset_bit_width,
        fixed_encoding_fields=fixed_encoding_fields,
    )


def _s_load_sized_overlay(
    offset_field_name: str = "OFFSET",
    offset_bit_width: int = 21,
    fixed_encoding_fields: tuple[tuple[str, AmdgpuFixedEncodingValue], ...] = (),
    *,
    descriptor_key: str,
    instruction_name: str,
    mnemonic: str,
    width_bits: int,
    payload_units: int,
    offset_only_descriptor_key: str,
    semantic_tag: str | None = None,
    implicit_memory: AmdgpuImplicitOperandOverlay | None = None,
    memory_effect: Effect | None = None,
    fixed_soffset: AmdgpuFixedEncodingValue | None = None,
) -> AmdgpuDescriptorOverlay:
    operands: tuple[AmdgpuOperandOverlay, ...] = (
        AmdgpuOperandOverlay("SDATA", _sgpr_result(units=payload_units)),
        AmdgpuOperandOverlay("SBASE", _sgpr_operand("base", units=2)),
    )
    operand_forms: tuple[OperandForm, ...]
    asm_forms: tuple[AsmForm, ...] | None
    if fixed_soffset is not None:
        fixed_encoding_fields = (*fixed_encoding_fields, ("SOFFSET", fixed_soffset))
        operand_forms = ()
        asm_forms = _asm(
            mnemonic=f"{mnemonic}_offset_only",
            native_assembly_mnemonic=mnemonic,
            results=("dst",),
            operands=("base",),
            immediates=("offset",),
            named_immediates=True,
        )
    else:
        operands = (
            *operands,
            AmdgpuOperandOverlay("SOFFSET", _sgpr_operand("soffset")),
        )
        operand_forms = (
            _soffset_zero_operand_form(
                replacement_descriptor=offset_only_descriptor_key
            ),
            _soffset_offset_operand_form(
                replacement_descriptor=offset_only_descriptor_key
            ),
        )
        asm_forms = None
    return AmdgpuDescriptorOverlay(
        descriptor_key=descriptor_key,
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        encoding_name="ENC_SMEM",
        semantic_tag=semantic_tag or f"memory.load.u{width_bits}",
        schedule_class=_SCHEDULE_SMEM_LOAD,
        operands=operands,
        implicit_operands=(implicit_memory or _ignore_global_read_memory(width_bits),),
        immediate_fields=(offset_field_name,),
        immediates=(_offset_immediate(offset_bit_width),),
        fixed_encoding_fields=fixed_encoding_fields,
        effects=(memory_effect or _global_read_effect(width_bits),),
        operand_forms=operand_forms,
        constraints=_EARLY_CLOBBER_RESULT_CONSTRAINTS,
        flags=(DescriptorFlag.SIDE_EFFECTING,),
        asm_forms=asm_forms,
    )


def _s_load_narrow_overlays(
    offset_field_name: str = "OFFSET",
    *,
    offset_bit_width: int = 21,
    fixed_encoding_fields: tuple[tuple[str, AmdgpuFixedEncodingValue], ...] = (),
    fixed_soffset: AmdgpuFixedEncodingValue | None = None,
) -> tuple[AmdgpuDescriptorOverlay, ...]:
    return tuple(
        _s_load_sized_overlay(
            descriptor_key=(
                f"amdgpu.s_load_{mnemonic_suffix}"
                if fixed_soffset is None
                else f"amdgpu.s_load_{mnemonic_suffix}_offset_only"
            ),
            instruction_name=f"S_LOAD_{instruction_suffix}",
            mnemonic=f"s_load_{mnemonic_suffix}",
            width_bits=width_bits,
            payload_units=1,
            offset_only_descriptor_key=f"amdgpu.s_load_{mnemonic_suffix}_offset_only",
            semantic_tag=semantic_tag,
            implicit_memory=implicit_memory,
            offset_field_name=offset_field_name,
            offset_bit_width=offset_bit_width,
            fixed_encoding_fields=fixed_encoding_fields,
            fixed_soffset=fixed_soffset,
        )
        for (
            mnemonic_suffix,
            instruction_suffix,
            width_bits,
            semantic_tag,
            implicit_memory,
        ) in _SMEM_NARROW_LOAD_ROWS
    )


def _s_scratch_load_dword_width_overlays(
    offset_field_name: str = "OFFSET",
    *,
    offset_bit_width: int = 21,
    fixed_encoding_fields: tuple[tuple[str, AmdgpuFixedEncodingValue], ...] = (),
    fixed_soffset: AmdgpuFixedEncodingValue | None = None,
) -> tuple[AmdgpuDescriptorOverlay, ...]:
    return tuple(
        _s_load_sized_overlay(
            descriptor_key=(
                f"amdgpu.s_scratch_load_{mnemonic_suffix}"
                if fixed_soffset is None
                else f"amdgpu.s_scratch_load_{mnemonic_suffix}_offset_only"
            ),
            instruction_name=f"S_SCRATCH_LOAD_{instruction_suffix}",
            mnemonic=f"s_scratch_load_{mnemonic_suffix}",
            width_bits=width_bits,
            payload_units=payload_units,
            offset_only_descriptor_key=(
                f"amdgpu.s_scratch_load_{mnemonic_suffix}_offset_only"
            ),
            semantic_tag=f"memory.stack.load.u{width_bits}",
            implicit_memory=_ignore_scratch_memory(
                width_bits=width_bits,
                is_input=True,
            ),
            memory_effect=_stack_memory_effect(EffectKind.READ, width_bits),
            offset_field_name=offset_field_name,
            offset_bit_width=offset_bit_width,
            fixed_encoding_fields=fixed_encoding_fields,
            fixed_soffset=fixed_soffset,
        )
        for (
            width_bits,
            payload_units,
            mnemonic_suffix,
            instruction_suffix,
        ) in _SMEM_DWORDX4_WIDTHS
    )


def _s_store_sized_overlay(
    offset_field_name: str = "OFFSET",
    offset_bit_width: int = 21,
    fixed_encoding_fields: tuple[tuple[str, AmdgpuFixedEncodingValue], ...] = (),
    *,
    descriptor_key: str,
    instruction_name: str,
    mnemonic: str,
    width_bits: int,
    payload_units: int,
    offset_only_descriptor_key: str,
    semantic_tag: str,
    memory_effect: Effect,
    implicit_memory: AmdgpuImplicitOperandOverlay | None = None,
    fixed_soffset: AmdgpuFixedEncodingValue | None = None,
) -> AmdgpuDescriptorOverlay:
    operands: tuple[AmdgpuOperandOverlay, ...] = (
        AmdgpuOperandOverlay("SDATA", _sgpr_operand("value", units=payload_units)),
        AmdgpuOperandOverlay("SBASE", _sgpr_operand("base", units=2)),
    )
    operand_forms: tuple[OperandForm, ...]
    asm_forms: tuple[AsmForm, ...] | None
    if fixed_soffset is not None:
        fixed_encoding_fields = (*fixed_encoding_fields, ("SOFFSET", fixed_soffset))
        operand_forms = ()
        asm_forms = _asm(
            mnemonic=f"{mnemonic}_offset_only",
            native_assembly_mnemonic=mnemonic,
            operands=("value", "base"),
            immediates=("offset",),
            named_immediates=True,
        )
    else:
        operands = (
            *operands,
            AmdgpuOperandOverlay("SOFFSET", _sgpr_operand("soffset")),
        )
        operand_forms = (
            _soffset_zero_operand_form(
                replacement_descriptor=offset_only_descriptor_key
            ),
            _soffset_offset_operand_form(
                replacement_descriptor=offset_only_descriptor_key
            ),
        )
        asm_forms = None
    return AmdgpuDescriptorOverlay(
        descriptor_key=descriptor_key,
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        encoding_name="ENC_SMEM",
        semantic_tag=semantic_tag,
        schedule_class=_SCHEDULE_SMEM_STORE,
        operands=operands,
        implicit_operands=(implicit_memory or _ignore_global_write_memory(width_bits),),
        immediate_fields=(offset_field_name,),
        immediates=(_offset_immediate(offset_bit_width),),
        fixed_encoding_fields=fixed_encoding_fields,
        effects=(memory_effect,),
        operand_forms=operand_forms,
        flags=(DescriptorFlag.SIDE_EFFECTING,),
        asm_forms=asm_forms,
    )


def _s_store_dword_width_overlays(
    offset_field_name: str = "OFFSET",
    *,
    offset_bit_width: int = 21,
    fixed_encoding_fields: tuple[tuple[str, AmdgpuFixedEncodingValue], ...] = (),
    fixed_soffset: AmdgpuFixedEncodingValue | None = None,
) -> tuple[AmdgpuDescriptorOverlay, ...]:
    return tuple(
        _s_store_sized_overlay(
            descriptor_key=(
                f"amdgpu.s_store_{mnemonic_suffix}"
                if fixed_soffset is None
                else f"amdgpu.s_store_{mnemonic_suffix}_offset_only"
            ),
            instruction_name=f"S_STORE_{instruction_suffix}",
            mnemonic=f"s_store_{mnemonic_suffix}",
            width_bits=width_bits,
            payload_units=payload_units,
            offset_only_descriptor_key=f"amdgpu.s_store_{mnemonic_suffix}_offset_only",
            semantic_tag=f"memory.store.u{width_bits}",
            memory_effect=_global_write_effect(width_bits),
            offset_field_name=offset_field_name,
            offset_bit_width=offset_bit_width,
            fixed_encoding_fields=fixed_encoding_fields,
            fixed_soffset=fixed_soffset,
        )
        for (
            width_bits,
            payload_units,
            mnemonic_suffix,
            instruction_suffix,
        ) in _SMEM_DWORDX4_WIDTHS
    )


def _s_scratch_store_dword_width_overlays(
    offset_field_name: str = "OFFSET",
    *,
    offset_bit_width: int = 21,
    fixed_encoding_fields: tuple[tuple[str, AmdgpuFixedEncodingValue], ...] = (),
    fixed_soffset: AmdgpuFixedEncodingValue | None = None,
) -> tuple[AmdgpuDescriptorOverlay, ...]:
    return tuple(
        _s_store_sized_overlay(
            descriptor_key=(
                f"amdgpu.s_scratch_store_{mnemonic_suffix}"
                if fixed_soffset is None
                else f"amdgpu.s_scratch_store_{mnemonic_suffix}_offset_only"
            ),
            instruction_name=f"S_SCRATCH_STORE_{instruction_suffix}",
            mnemonic=f"s_scratch_store_{mnemonic_suffix}",
            width_bits=width_bits,
            payload_units=payload_units,
            offset_only_descriptor_key=(
                f"amdgpu.s_scratch_store_{mnemonic_suffix}_offset_only"
            ),
            semantic_tag=f"memory.stack.store.u{width_bits}",
            memory_effect=_stack_memory_effect(EffectKind.WRITE, width_bits),
            implicit_memory=_ignore_scratch_memory(
                width_bits=width_bits,
                is_input=False,
            ),
            offset_field_name=offset_field_name,
            offset_bit_width=offset_bit_width,
            fixed_encoding_fields=fixed_encoding_fields,
            fixed_soffset=fixed_soffset,
        )
        for (
            width_bits,
            payload_units,
            mnemonic_suffix,
            instruction_suffix,
        ) in _SMEM_DWORDX4_WIDTHS
    )


def _s_buffer_store_sized_overlay(
    *,
    descriptor_key: str,
    instruction_name: str,
    mnemonic: str,
    width_bits: int,
    payload_units: int,
    offset_field_name: str = "OFFSET",
    offset_bit_width: int = 21,
    fixed_encoding_fields: tuple[tuple[str, AmdgpuFixedEncodingValue], ...] = (),
) -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key=descriptor_key,
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        encoding_name="ENC_SMEM",
        semantic_tag=f"memory.store.u{width_bits}",
        schedule_class=_SCHEDULE_SMEM_STORE,
        operands=(
            AmdgpuOperandOverlay("SDATA", _sgpr_operand("value", units=payload_units)),
            AmdgpuOperandOverlay("SBASE", _sgpr_resource("resource", units=4)),
            AmdgpuOperandOverlay("SOFFSET", _sgpr_operand("soffset")),
        ),
        implicit_operands=(_ignore_global_write_memory(width_bits),),
        immediate_fields=(offset_field_name,),
        immediates=(_offset_immediate(offset_bit_width),),
        fixed_encoding_fields=fixed_encoding_fields,
        effects=(_global_write_effect(width_bits),),
        flags=(DescriptorFlag.SIDE_EFFECTING,),
    )


def _s_buffer_store_dword_width_overlays(
    offset_field_name: str = "OFFSET",
    *,
    offset_bit_width: int = 21,
    fixed_encoding_fields: tuple[tuple[str, AmdgpuFixedEncodingValue], ...] = (),
) -> tuple[AmdgpuDescriptorOverlay, ...]:
    return tuple(
        _s_buffer_store_sized_overlay(
            descriptor_key=f"amdgpu.s_buffer_store_{mnemonic_suffix}",
            instruction_name=f"S_BUFFER_STORE_{instruction_suffix}",
            mnemonic=f"s_buffer_store_{mnemonic_suffix}",
            width_bits=width_bits,
            payload_units=payload_units,
            offset_field_name=offset_field_name,
            offset_bit_width=offset_bit_width,
            fixed_encoding_fields=fixed_encoding_fields,
        )
        for (
            width_bits,
            payload_units,
            mnemonic_suffix,
            instruction_suffix,
        ) in _SMEM_DWORDX4_WIDTHS
    )


def _s_load_dword_overlay(
    offset_field_name: str = "OFFSET",
    offset_bit_width: int = 21,
    fixed_encoding_fields: tuple[tuple[str, AmdgpuFixedEncodingValue], ...] = (),
    *,
    descriptor_key: str = "amdgpu.s_load_dword",
    fixed_soffset: AmdgpuFixedEncodingValue | None = None,
) -> AmdgpuDescriptorOverlay:
    return _s_load_sized_overlay(
        descriptor_key=descriptor_key,
        instruction_name="S_LOAD_DWORD",
        mnemonic="s_load_dword",
        width_bits=32,
        payload_units=1,
        offset_only_descriptor_key="amdgpu.s_load_dword_offset_only",
        offset_field_name=offset_field_name,
        offset_bit_width=offset_bit_width,
        fixed_encoding_fields=fixed_encoding_fields,
        fixed_soffset=fixed_soffset,
    )


def _s_load_dwordx2_overlay(
    offset_field_name: str = "OFFSET",
    offset_bit_width: int = 21,
    fixed_encoding_fields: tuple[tuple[str, AmdgpuFixedEncodingValue], ...] = (),
    *,
    descriptor_key: str = "amdgpu.s_load_dwordx2",
    fixed_soffset: AmdgpuFixedEncodingValue | None = None,
) -> AmdgpuDescriptorOverlay:
    return _s_load_sized_overlay(
        descriptor_key=descriptor_key,
        instruction_name="S_LOAD_DWORDX2",
        mnemonic="s_load_dwordx2",
        width_bits=64,
        payload_units=2,
        offset_only_descriptor_key="amdgpu.s_load_dwordx2_offset_only",
        offset_field_name=offset_field_name,
        offset_bit_width=offset_bit_width,
        fixed_encoding_fields=fixed_encoding_fields,
        fixed_soffset=fixed_soffset,
    )


def _s_load_96_overlay(
    offset_field_name: str = "OFFSET",
    offset_bit_width: int = 21,
    fixed_encoding_fields: tuple[tuple[str, AmdgpuFixedEncodingValue], ...] = (),
    *,
    descriptor_key: str = "amdgpu.s_load_b96",
    fixed_soffset: AmdgpuFixedEncodingValue | None = None,
) -> AmdgpuDescriptorOverlay:
    return _s_load_sized_overlay(
        descriptor_key=descriptor_key,
        instruction_name="S_LOAD_B96",
        mnemonic="s_load_b96",
        width_bits=96,
        payload_units=3,
        offset_only_descriptor_key="amdgpu.s_load_b96_offset_only",
        offset_field_name=offset_field_name,
        offset_bit_width=offset_bit_width,
        fixed_encoding_fields=fixed_encoding_fields,
        fixed_soffset=fixed_soffset,
    )


def _s_load_dwordx4_overlay(
    offset_field_name: str = "OFFSET",
    offset_bit_width: int = 21,
    fixed_encoding_fields: tuple[tuple[str, AmdgpuFixedEncodingValue], ...] = (),
    *,
    descriptor_key: str = "amdgpu.s_load_dwordx4",
    fixed_soffset: AmdgpuFixedEncodingValue | None = None,
) -> AmdgpuDescriptorOverlay:
    return _s_load_sized_overlay(
        descriptor_key=descriptor_key,
        instruction_name="S_LOAD_DWORDX4",
        mnemonic="s_load_dwordx4",
        width_bits=128,
        payload_units=4,
        offset_only_descriptor_key="amdgpu.s_load_dwordx4_offset_only",
        offset_field_name=offset_field_name,
        offset_bit_width=offset_bit_width,
        fixed_encoding_fields=fixed_encoding_fields,
        fixed_soffset=fixed_soffset,
    )


def _s_load_dwordx8_overlay(
    offset_field_name: str = "OFFSET",
    offset_bit_width: int = 21,
    fixed_encoding_fields: tuple[tuple[str, AmdgpuFixedEncodingValue], ...] = (),
    *,
    descriptor_key: str = "amdgpu.s_load_dwordx8",
    fixed_soffset: AmdgpuFixedEncodingValue | None = None,
) -> AmdgpuDescriptorOverlay:
    return _s_load_sized_overlay(
        descriptor_key=descriptor_key,
        instruction_name="S_LOAD_DWORDX8",
        mnemonic="s_load_dwordx8",
        width_bits=256,
        payload_units=8,
        offset_only_descriptor_key="amdgpu.s_load_dwordx8_offset_only",
        offset_field_name=offset_field_name,
        offset_bit_width=offset_bit_width,
        fixed_encoding_fields=fixed_encoding_fields,
        fixed_soffset=fixed_soffset,
    )


def _s_load_dwordx16_overlay(
    offset_field_name: str = "OFFSET",
    offset_bit_width: int = 21,
    fixed_encoding_fields: tuple[tuple[str, AmdgpuFixedEncodingValue], ...] = (),
    *,
    descriptor_key: str = "amdgpu.s_load_dwordx16",
    fixed_soffset: AmdgpuFixedEncodingValue | None = None,
) -> AmdgpuDescriptorOverlay:
    return _s_load_sized_overlay(
        descriptor_key=descriptor_key,
        instruction_name="S_LOAD_DWORDX16",
        mnemonic="s_load_dwordx16",
        width_bits=512,
        payload_units=16,
        offset_only_descriptor_key="amdgpu.s_load_dwordx16_offset_only",
        offset_field_name=offset_field_name,
        offset_bit_width=offset_bit_width,
        fixed_encoding_fields=fixed_encoding_fields,
        fixed_soffset=fixed_soffset,
    )


def _buffer_load_dword_overlay(
    *,
    encoding_name: str,
    resource_field_name: str,
    offset_field_name: str = "OFFSET",
    offset_bit_width: int = 12,
    cache_fields: tuple[tuple[str, int], ...] = (),
    off_zero_descriptor_key: str | None = "amdgpu.buffer_load_dword_off_zero",
    vaddr_offset_descriptor_key: str | None = ("amdgpu.buffer_load_dword_vaddr_offset"),
) -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.buffer_load_dword",
        instruction_name="BUFFER_LOAD_DWORD",
        mnemonic="buffer_load_dword",
        encoding_name=encoding_name,
        semantic_tag="memory.load.u32",
        schedule_class=_SCHEDULE_VMEM_LOAD,
        operands=(
            AmdgpuOperandOverlay("VDATA", _vgpr_result()),
            AmdgpuOperandOverlay(
                resource_field_name, _sgpr_resource("resource", units=4)
            ),
            _mubuf_vaddr_operand(),
            AmdgpuOperandOverlay("SOFFSET", _sgpr_operand("soffset")),
        ),
        implicit_operands=(_IGNORE_GLOBAL_READ_MEMORY,),
        immediate_fields=(offset_field_name, *_cache_field_names(cache_fields)),
        immediates=(
            _offset_immediate(offset_bit_width),
            *_cache_immediates(cache_fields),
        ),
        fixed_encoding_fields=(("IDXEN", 0), ("OFFEN", 1)),
        effects=(_GLOBAL_LOAD_EFFECT,),
        flags=(DescriptorFlag.SIDE_EFFECTING,),
        operand_forms=_buffer_operand_forms(
            off_zero_descriptor_key=off_zero_descriptor_key,
            vaddr_offset_descriptor_key=vaddr_offset_descriptor_key,
        ),
    )


def _buffer_load_off_zero_overlay(
    *,
    descriptor_key: str,
    instruction_name: str,
    mnemonic: str,
    semantic_tag: str,
    payload_units: int,
    memory_effect: Effect,
    implicit_memory: AmdgpuImplicitOperandOverlay,
    encoding_name: str,
    resource_field_name: str,
    offset_field_name: str = "OFFSET",
    offset_bit_width: int = 12,
    cache_fields: tuple[tuple[str, int], ...] = (),
    fixed_soffset: AmdgpuFixedEncodingValue = _MUBUF_SOFFSET_INLINE_ZERO,
) -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key=descriptor_key,
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        encoding_name=encoding_name,
        semantic_tag=semantic_tag,
        schedule_class=_SCHEDULE_VMEM_LOAD,
        operands=(
            AmdgpuOperandOverlay("VDATA", _vgpr_result(units=payload_units)),
            AmdgpuOperandOverlay(
                resource_field_name, _sgpr_resource("resource", units=4)
            ),
        ),
        implicit_operands=(implicit_memory,),
        immediate_fields=(offset_field_name, *_cache_field_names(cache_fields)),
        immediates=(
            _offset_immediate(offset_bit_width),
            *_cache_immediates(cache_fields),
        ),
        fixed_encoding_fields=(
            ("VADDR", _predefined("v0")),
            ("SOFFSET", fixed_soffset),
            ("IDXEN", 0),
            ("OFFEN", 0),
        ),
        effects=(memory_effect,),
        flags=(DescriptorFlag.SIDE_EFFECTING,),
        asm_forms=_asm(
            mnemonic=f"{mnemonic}_off_zero",
            native_assembly_mnemonic=mnemonic,
            results=("dst",),
            operands=("resource",),
            immediates=_memory_asm_immediate_names(cache_fields),
            named_immediates=True,
        ),
    )


def _buffer_load_vaddr_offset_overlay(
    *,
    descriptor_key: str,
    instruction_name: str,
    mnemonic: str,
    semantic_tag: str,
    payload_units: int,
    memory_effect: Effect,
    implicit_memory: AmdgpuImplicitOperandOverlay,
    encoding_name: str,
    resource_field_name: str,
    offset_field_name: str = "OFFSET",
    offset_bit_width: int = 12,
    cache_fields: tuple[tuple[str, int], ...] = (),
    fixed_soffset: AmdgpuFixedEncodingValue = _MUBUF_SOFFSET_INLINE_ZERO,
    fixed_soffset_native_spelling: str = "0",
) -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key=descriptor_key,
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        encoding_name=encoding_name,
        semantic_tag=semantic_tag,
        schedule_class=_SCHEDULE_VMEM_LOAD,
        operands=(
            AmdgpuOperandOverlay("VDATA", _vgpr_result(units=payload_units)),
            AmdgpuOperandOverlay(
                resource_field_name, _sgpr_resource("resource", units=4)
            ),
            _mubuf_vaddr_operand(),
        ),
        implicit_operands=(implicit_memory,),
        immediate_fields=(offset_field_name, *_cache_field_names(cache_fields)),
        immediates=(
            _offset_immediate(offset_bit_width),
            *_cache_immediates(cache_fields),
        ),
        fixed_encoding_fields=(
            ("SOFFSET", fixed_soffset),
            ("IDXEN", 0),
            ("OFFEN", 1),
        ),
        effects=(memory_effect,),
        flags=(DescriptorFlag.SIDE_EFFECTING,),
        asm_forms=_asm(
            mnemonic=f"{mnemonic}_vaddr_offset",
            native_assembly_mnemonic=mnemonic,
            results=("dst",),
            operands=("resource", "vaddr"),
            immediates=_memory_asm_immediate_names(cache_fields),
            named_immediates=True,
            native_assembly_values=_buffer_vaddr_offset_native_assembly_values(
                payload_kind=NativeAsmValueKind.RESULT,
                payload_field_name="dst",
                fixed_soffset_native_spelling=fixed_soffset_native_spelling,
            ),
        ),
    )


def _buffer_load_dword_off_zero_overlay(
    *,
    encoding_name: str,
    resource_field_name: str,
    offset_field_name: str = "OFFSET",
    offset_bit_width: int = 12,
    cache_fields: tuple[tuple[str, int], ...] = (),
) -> AmdgpuDescriptorOverlay:
    return _buffer_load_off_zero_overlay(
        descriptor_key="amdgpu.buffer_load_dword_off_zero",
        instruction_name="BUFFER_LOAD_DWORD",
        mnemonic="buffer_load_dword",
        semantic_tag="memory.load.u32",
        payload_units=1,
        memory_effect=_GLOBAL_LOAD_EFFECT,
        implicit_memory=_IGNORE_GLOBAL_READ_MEMORY,
        encoding_name=encoding_name,
        resource_field_name=resource_field_name,
        offset_field_name=offset_field_name,
        offset_bit_width=offset_bit_width,
        cache_fields=cache_fields,
    )


def _buffer_load_dword_vaddr_offset_overlay(
    *,
    encoding_name: str,
    resource_field_name: str,
    offset_field_name: str = "OFFSET",
    offset_bit_width: int = 12,
    cache_fields: tuple[tuple[str, int], ...] = (),
    fixed_soffset: AmdgpuFixedEncodingValue = _MUBUF_SOFFSET_INLINE_ZERO,
    fixed_soffset_native_spelling: str = "0",
) -> AmdgpuDescriptorOverlay:
    return _buffer_load_vaddr_offset_overlay(
        descriptor_key="amdgpu.buffer_load_dword_vaddr_offset",
        instruction_name="BUFFER_LOAD_DWORD",
        mnemonic="buffer_load_dword",
        semantic_tag="memory.load.u32",
        payload_units=1,
        memory_effect=_GLOBAL_LOAD_EFFECT,
        implicit_memory=_IGNORE_GLOBAL_READ_MEMORY,
        encoding_name=encoding_name,
        resource_field_name=resource_field_name,
        offset_field_name=offset_field_name,
        offset_bit_width=offset_bit_width,
        cache_fields=cache_fields,
        fixed_soffset=fixed_soffset,
        fixed_soffset_native_spelling=fixed_soffset_native_spelling,
    )


def _buffer_load_b16_d16_overlay(
    *,
    encoding_name: str,
    resource_field_name: str,
    offset_field_name: str = "OFFSET",
    offset_bit_width: int = 12,
    cache_fields: tuple[tuple[str, int], ...] = (),
    vaddr_offset_descriptor_key: str | None = (
        "amdgpu.buffer_load_b16_d16_vaddr_offset"
    ),
) -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.buffer_load_b16_d16",
        instruction_name="BUFFER_LOAD_SHORT_D16",
        mnemonic="buffer_load_d16_b16",
        encoding_name=encoding_name,
        semantic_tag="memory.load.u16.d16.low",
        schedule_class=_SCHEDULE_VMEM_LOAD,
        operands=(
            AmdgpuOperandOverlay(
                "VDATA",
                _vgpr_result(register_part=_REG_PART_VGPR_LOW16),
                size_exception_reason=_D16_PARTIAL_REGISTER_SIZE_REASON,
            ),
            AmdgpuOperandOverlay(
                resource_field_name, _sgpr_resource("resource", units=4)
            ),
            _mubuf_vaddr_operand(),
            AmdgpuOperandOverlay("SOFFSET", _sgpr_operand("soffset")),
        ),
        implicit_operands=(_ignore_global_read_memory(16),),
        immediate_fields=(offset_field_name, *_cache_field_names(cache_fields)),
        immediates=(
            _offset_immediate(offset_bit_width),
            *_cache_immediates(cache_fields),
        ),
        fixed_encoding_fields=(("IDXEN", 0), ("OFFEN", 1)),
        effects=(_global_read_effect(16),),
        flags=(DescriptorFlag.SIDE_EFFECTING,),
        operand_forms=_buffer_operand_forms(
            vaddr_offset_descriptor_key=vaddr_offset_descriptor_key,
        ),
    )


def _buffer_load_b16_d16_vaddr_offset_overlay(
    *,
    encoding_name: str,
    resource_field_name: str,
    offset_field_name: str = "OFFSET",
    offset_bit_width: int = 12,
    cache_fields: tuple[tuple[str, int], ...] = (),
    fixed_soffset: AmdgpuFixedEncodingValue = _MUBUF_SOFFSET_INLINE_ZERO,
    fixed_soffset_native_spelling: str = "0",
) -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.buffer_load_b16_d16_vaddr_offset",
        instruction_name="BUFFER_LOAD_SHORT_D16",
        mnemonic="buffer_load_d16_b16",
        encoding_name=encoding_name,
        semantic_tag="memory.load.u16.d16.low",
        schedule_class=_SCHEDULE_VMEM_LOAD,
        operands=(
            AmdgpuOperandOverlay(
                "VDATA",
                _vgpr_result(register_part=_REG_PART_VGPR_LOW16),
                size_exception_reason=_D16_PARTIAL_REGISTER_SIZE_REASON,
            ),
            AmdgpuOperandOverlay(
                resource_field_name, _sgpr_resource("resource", units=4)
            ),
            _mubuf_vaddr_operand(),
        ),
        implicit_operands=(_ignore_global_read_memory(16),),
        immediate_fields=(offset_field_name, *_cache_field_names(cache_fields)),
        immediates=(
            _offset_immediate(offset_bit_width),
            *_cache_immediates(cache_fields),
        ),
        fixed_encoding_fields=(
            ("SOFFSET", fixed_soffset),
            ("IDXEN", 0),
            ("OFFEN", 1),
        ),
        effects=(_global_read_effect(16),),
        flags=(DescriptorFlag.SIDE_EFFECTING,),
        asm_forms=_asm(
            mnemonic="buffer_load_d16_b16_vaddr_offset",
            native_assembly_mnemonic="buffer_load_d16_b16",
            results=("dst",),
            operands=("resource", "vaddr"),
            immediates=_memory_asm_immediate_names(cache_fields),
            named_immediates=True,
            native_assembly_values=_buffer_vaddr_offset_native_assembly_values(
                payload_kind=NativeAsmValueKind.RESULT,
                payload_field_name="dst",
                fixed_soffset_native_spelling=fixed_soffset_native_spelling,
            ),
        ),
    )


def _buffer_load_sized_overlay(
    *,
    descriptor_key: str,
    instruction_name: str,
    mnemonic: str,
    width_bits: int,
    payload_units: int,
    semantic_tag: str | None = None,
    implicit_memory: AmdgpuImplicitOperandOverlay | None = None,
    encoding_name: str,
    resource_field_name: str,
    offset_field_name: str = "OFFSET",
    offset_bit_width: int = 12,
    cache_fields: tuple[tuple[str, int], ...] = (),
    off_zero_descriptor_key: str | None = None,
    vaddr_offset_descriptor_key: str | None = None,
) -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key=descriptor_key,
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        encoding_name=encoding_name,
        semantic_tag=semantic_tag or f"memory.load.u{width_bits}",
        schedule_class=_SCHEDULE_VMEM_LOAD,
        operands=(
            AmdgpuOperandOverlay("VDATA", _vgpr_result(units=payload_units)),
            AmdgpuOperandOverlay(
                resource_field_name, _sgpr_resource("resource", units=4)
            ),
            _mubuf_vaddr_operand(),
            AmdgpuOperandOverlay("SOFFSET", _sgpr_operand("soffset")),
        ),
        immediate_fields=(offset_field_name, *_cache_field_names(cache_fields)),
        immediates=(
            _offset_immediate(offset_bit_width),
            *_cache_immediates(cache_fields),
        ),
        fixed_encoding_fields=(("IDXEN", 0), ("OFFEN", 1)),
        implicit_operands=(implicit_memory or _ignore_global_read_memory(width_bits),),
        effects=(_global_read_effect(width_bits),),
        flags=(DescriptorFlag.SIDE_EFFECTING,),
        operand_forms=_buffer_operand_forms(
            off_zero_descriptor_key=off_zero_descriptor_key,
            vaddr_offset_descriptor_key=vaddr_offset_descriptor_key,
        ),
    )


def _buffer_load_narrow_overlays(
    *,
    rows: tuple[tuple[str, str, int, str, AmdgpuImplicitOperandOverlay], ...],
    encoding_name: str,
    resource_field_name: str,
    offset_field_name: str = "OFFSET",
    offset_bit_width: int = 12,
    cache_fields: tuple[tuple[str, int], ...] = (),
    include_off_zero: bool = True,
    include_vaddr_offset: bool = True,
    fixed_soffset: AmdgpuFixedEncodingValue = _MUBUF_SOFFSET_INLINE_ZERO,
    fixed_soffset_native_spelling: str = "0",
) -> tuple[AmdgpuDescriptorOverlay, ...]:
    return tuple(
        overlay
        for (
            descriptor_suffix,
            instruction_suffix,
            width_bits,
            semantic_tag,
            implicit_memory,
        ) in rows
        for overlay in (
            _buffer_load_sized_overlay(
                descriptor_key=f"amdgpu.buffer_load_{descriptor_suffix}",
                instruction_name=f"BUFFER_LOAD_{instruction_suffix}",
                mnemonic=f"buffer_load_{descriptor_suffix}",
                width_bits=width_bits,
                payload_units=1,
                semantic_tag=semantic_tag,
                implicit_memory=implicit_memory,
                encoding_name=encoding_name,
                resource_field_name=resource_field_name,
                offset_field_name=offset_field_name,
                offset_bit_width=offset_bit_width,
                cache_fields=cache_fields,
                off_zero_descriptor_key=(
                    f"amdgpu.buffer_load_{descriptor_suffix}_off_zero"
                    if include_off_zero
                    else None
                ),
                vaddr_offset_descriptor_key=(
                    f"amdgpu.buffer_load_{descriptor_suffix}_vaddr_offset"
                    if include_vaddr_offset
                    else None
                ),
            ),
            *(
                (
                    _buffer_load_off_zero_overlay(
                        descriptor_key=(
                            f"amdgpu.buffer_load_{descriptor_suffix}_off_zero"
                        ),
                        instruction_name=f"BUFFER_LOAD_{instruction_suffix}",
                        mnemonic=f"buffer_load_{descriptor_suffix}",
                        semantic_tag=semantic_tag,
                        payload_units=1,
                        memory_effect=_global_read_effect(width_bits),
                        implicit_memory=implicit_memory,
                        encoding_name=encoding_name,
                        resource_field_name=resource_field_name,
                        offset_field_name=offset_field_name,
                        offset_bit_width=offset_bit_width,
                        cache_fields=cache_fields,
                        fixed_soffset=fixed_soffset,
                    ),
                )
                if include_off_zero
                else ()
            ),
            *(
                (
                    _buffer_load_vaddr_offset_overlay(
                        descriptor_key=(
                            f"amdgpu.buffer_load_{descriptor_suffix}_vaddr_offset"
                        ),
                        instruction_name=f"BUFFER_LOAD_{instruction_suffix}",
                        mnemonic=f"buffer_load_{descriptor_suffix}",
                        semantic_tag=semantic_tag,
                        payload_units=1,
                        memory_effect=_global_read_effect(width_bits),
                        implicit_memory=implicit_memory,
                        encoding_name=encoding_name,
                        resource_field_name=resource_field_name,
                        offset_field_name=offset_field_name,
                        offset_bit_width=offset_bit_width,
                        cache_fields=cache_fields,
                        fixed_soffset=fixed_soffset,
                        fixed_soffset_native_spelling=fixed_soffset_native_spelling,
                    ),
                )
                if include_vaddr_offset
                else ()
            ),
        )
    )


def _buffer_load_64_overlay(
    *,
    descriptor_key: str = "amdgpu.buffer_load_b64",
    instruction_name: str = "BUFFER_LOAD_B64",
    mnemonic: str = "buffer_load_b64",
    encoding_name: str,
    resource_field_name: str,
    offset_field_name: str = "OFFSET",
    offset_bit_width: int = 12,
    cache_fields: tuple[tuple[str, int], ...] = (),
    off_zero_descriptor_key: str | None = "amdgpu.buffer_load_b64_off_zero",
    vaddr_offset_descriptor_key: str | None = "amdgpu.buffer_load_b64_vaddr_offset",
) -> AmdgpuDescriptorOverlay:
    return _buffer_load_sized_overlay(
        descriptor_key=descriptor_key,
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        width_bits=64,
        payload_units=2,
        encoding_name=encoding_name,
        resource_field_name=resource_field_name,
        offset_field_name=offset_field_name,
        offset_bit_width=offset_bit_width,
        cache_fields=cache_fields,
        off_zero_descriptor_key=off_zero_descriptor_key,
        vaddr_offset_descriptor_key=vaddr_offset_descriptor_key,
    )


def _buffer_load_64_off_zero_overlay(
    *,
    descriptor_key: str = "amdgpu.buffer_load_b64_off_zero",
    instruction_name: str = "BUFFER_LOAD_B64",
    mnemonic: str = "buffer_load_b64",
    encoding_name: str,
    resource_field_name: str,
    offset_field_name: str = "OFFSET",
    offset_bit_width: int = 12,
    cache_fields: tuple[tuple[str, int], ...] = (),
) -> AmdgpuDescriptorOverlay:
    return _buffer_load_off_zero_overlay(
        descriptor_key=descriptor_key,
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        semantic_tag="memory.load.u64",
        payload_units=2,
        memory_effect=_GLOBAL_LOAD_B64_EFFECT,
        implicit_memory=_IGNORE_GLOBAL_READ_MEMORY_B64,
        encoding_name=encoding_name,
        resource_field_name=resource_field_name,
        offset_field_name=offset_field_name,
        offset_bit_width=offset_bit_width,
        cache_fields=cache_fields,
    )


def _buffer_load_64_vaddr_offset_overlay(
    *,
    descriptor_key: str = "amdgpu.buffer_load_b64_vaddr_offset",
    instruction_name: str = "BUFFER_LOAD_B64",
    mnemonic: str = "buffer_load_b64",
    encoding_name: str,
    resource_field_name: str,
    offset_field_name: str = "OFFSET",
    offset_bit_width: int = 12,
    cache_fields: tuple[tuple[str, int], ...] = (),
    fixed_soffset: AmdgpuFixedEncodingValue = _MUBUF_SOFFSET_INLINE_ZERO,
    fixed_soffset_native_spelling: str = "0",
) -> AmdgpuDescriptorOverlay:
    return _buffer_load_vaddr_offset_overlay(
        descriptor_key=descriptor_key,
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        semantic_tag="memory.load.u64",
        payload_units=2,
        memory_effect=_GLOBAL_LOAD_B64_EFFECT,
        implicit_memory=_IGNORE_GLOBAL_READ_MEMORY_B64,
        encoding_name=encoding_name,
        resource_field_name=resource_field_name,
        offset_field_name=offset_field_name,
        offset_bit_width=offset_bit_width,
        cache_fields=cache_fields,
        fixed_soffset=fixed_soffset,
        fixed_soffset_native_spelling=fixed_soffset_native_spelling,
    )


def _buffer_load_96_overlay(
    *,
    descriptor_key: str = "amdgpu.buffer_load_b96",
    instruction_name: str = "BUFFER_LOAD_B96",
    mnemonic: str = "buffer_load_b96",
    encoding_name: str,
    resource_field_name: str,
    offset_field_name: str = "OFFSET",
    offset_bit_width: int = 12,
    cache_fields: tuple[tuple[str, int], ...] = (),
    off_zero_descriptor_key: str | None = "amdgpu.buffer_load_b96_off_zero",
    vaddr_offset_descriptor_key: str | None = "amdgpu.buffer_load_b96_vaddr_offset",
) -> AmdgpuDescriptorOverlay:
    return _buffer_load_sized_overlay(
        descriptor_key=descriptor_key,
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        width_bits=96,
        payload_units=3,
        encoding_name=encoding_name,
        resource_field_name=resource_field_name,
        offset_field_name=offset_field_name,
        offset_bit_width=offset_bit_width,
        cache_fields=cache_fields,
        off_zero_descriptor_key=off_zero_descriptor_key,
        vaddr_offset_descriptor_key=vaddr_offset_descriptor_key,
    )


def _buffer_load_96_off_zero_overlay(
    *,
    descriptor_key: str = "amdgpu.buffer_load_b96_off_zero",
    instruction_name: str = "BUFFER_LOAD_B96",
    mnemonic: str = "buffer_load_b96",
    encoding_name: str,
    resource_field_name: str,
    offset_field_name: str = "OFFSET",
    offset_bit_width: int = 12,
    cache_fields: tuple[tuple[str, int], ...] = (),
) -> AmdgpuDescriptorOverlay:
    return _buffer_load_off_zero_overlay(
        descriptor_key=descriptor_key,
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        semantic_tag="memory.load.u96",
        payload_units=3,
        memory_effect=_GLOBAL_LOAD_B96_EFFECT,
        implicit_memory=_IGNORE_GLOBAL_READ_MEMORY_B96,
        encoding_name=encoding_name,
        resource_field_name=resource_field_name,
        offset_field_name=offset_field_name,
        offset_bit_width=offset_bit_width,
        cache_fields=cache_fields,
    )


def _buffer_load_96_vaddr_offset_overlay(
    *,
    descriptor_key: str = "amdgpu.buffer_load_b96_vaddr_offset",
    instruction_name: str = "BUFFER_LOAD_B96",
    mnemonic: str = "buffer_load_b96",
    encoding_name: str,
    resource_field_name: str,
    offset_field_name: str = "OFFSET",
    offset_bit_width: int = 12,
    cache_fields: tuple[tuple[str, int], ...] = (),
    fixed_soffset: AmdgpuFixedEncodingValue = _MUBUF_SOFFSET_INLINE_ZERO,
    fixed_soffset_native_spelling: str = "0",
) -> AmdgpuDescriptorOverlay:
    return _buffer_load_vaddr_offset_overlay(
        descriptor_key=descriptor_key,
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        semantic_tag="memory.load.u96",
        payload_units=3,
        memory_effect=_GLOBAL_LOAD_B96_EFFECT,
        implicit_memory=_IGNORE_GLOBAL_READ_MEMORY_B96,
        encoding_name=encoding_name,
        resource_field_name=resource_field_name,
        offset_field_name=offset_field_name,
        offset_bit_width=offset_bit_width,
        cache_fields=cache_fields,
        fixed_soffset=fixed_soffset,
        fixed_soffset_native_spelling=fixed_soffset_native_spelling,
    )


def _buffer_load_128_overlay(
    *,
    descriptor_key: str = "amdgpu.buffer_load_b128",
    instruction_name: str = "BUFFER_LOAD_B128",
    mnemonic: str = "buffer_load_b128",
    encoding_name: str,
    resource_field_name: str,
    offset_field_name: str = "OFFSET",
    offset_bit_width: int = 12,
    cache_fields: tuple[tuple[str, int], ...] = (),
    off_zero_descriptor_key: str | None = "amdgpu.buffer_load_b128_off_zero",
    vaddr_offset_descriptor_key: str | None = "amdgpu.buffer_load_b128_vaddr_offset",
) -> AmdgpuDescriptorOverlay:
    return _buffer_load_sized_overlay(
        descriptor_key=descriptor_key,
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        width_bits=128,
        payload_units=4,
        encoding_name=encoding_name,
        resource_field_name=resource_field_name,
        offset_field_name=offset_field_name,
        offset_bit_width=offset_bit_width,
        cache_fields=cache_fields,
        off_zero_descriptor_key=off_zero_descriptor_key,
        vaddr_offset_descriptor_key=vaddr_offset_descriptor_key,
    )


def _buffer_load_128_off_zero_overlay(
    *,
    descriptor_key: str = "amdgpu.buffer_load_b128_off_zero",
    instruction_name: str = "BUFFER_LOAD_B128",
    mnemonic: str = "buffer_load_b128",
    encoding_name: str,
    resource_field_name: str,
    offset_field_name: str = "OFFSET",
    offset_bit_width: int = 12,
    cache_fields: tuple[tuple[str, int], ...] = (),
) -> AmdgpuDescriptorOverlay:
    return _buffer_load_off_zero_overlay(
        descriptor_key=descriptor_key,
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        semantic_tag="memory.load.u128",
        payload_units=4,
        memory_effect=_GLOBAL_LOAD_B128_EFFECT,
        implicit_memory=_IGNORE_GLOBAL_READ_MEMORY_B128,
        encoding_name=encoding_name,
        resource_field_name=resource_field_name,
        offset_field_name=offset_field_name,
        offset_bit_width=offset_bit_width,
        cache_fields=cache_fields,
    )


def _buffer_load_128_vaddr_offset_overlay(
    *,
    descriptor_key: str = "amdgpu.buffer_load_b128_vaddr_offset",
    instruction_name: str = "BUFFER_LOAD_B128",
    mnemonic: str = "buffer_load_b128",
    encoding_name: str,
    resource_field_name: str,
    offset_field_name: str = "OFFSET",
    offset_bit_width: int = 12,
    cache_fields: tuple[tuple[str, int], ...] = (),
    fixed_soffset: AmdgpuFixedEncodingValue = _MUBUF_SOFFSET_INLINE_ZERO,
    fixed_soffset_native_spelling: str = "0",
) -> AmdgpuDescriptorOverlay:
    return _buffer_load_vaddr_offset_overlay(
        descriptor_key=descriptor_key,
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        semantic_tag="memory.load.u128",
        payload_units=4,
        memory_effect=_GLOBAL_LOAD_B128_EFFECT,
        implicit_memory=_IGNORE_GLOBAL_READ_MEMORY_B128,
        encoding_name=encoding_name,
        resource_field_name=resource_field_name,
        offset_field_name=offset_field_name,
        offset_bit_width=offset_bit_width,
        cache_fields=cache_fields,
        fixed_soffset=fixed_soffset,
        fixed_soffset_native_spelling=fixed_soffset_native_spelling,
    )


def _buffer_load_lds_overlay(
    *,
    descriptor_key: str,
    instruction_name: str,
    mnemonic: str,
    asm_mnemonic: str,
    semantic_tag: str,
    global_width_bits: int,
    workgroup_width_bits: int,
    implicit_memory: AmdgpuImplicitOperandOverlay,
    encoding_name: str,
    resource_field_name: str,
    offset_field_name: str = "OFFSET",
    offset_bit_width: int = 12,
    cache_fields: tuple[tuple[str, int], ...] = (),
    off_zero_descriptor_key: str | None = None,
    vaddr_offset_descriptor_key: str | None = None,
) -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key=descriptor_key,
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        encoding_name=encoding_name,
        semantic_tag=semantic_tag,
        schedule_class=_SCHEDULE_VMEM_LOAD_LDS,
        operands=(
            AmdgpuOperandOverlay(
                resource_field_name, _sgpr_resource("resource", units=4)
            ),
            _mubuf_vaddr_operand(),
            AmdgpuOperandOverlay("SOFFSET", _sgpr_operand("soffset")),
        ),
        ignored_operands=(
            AmdgpuIgnoredOperandOverlay(
                "VDATA",
                ignore_reason="lds-bit-has-no-vgpr-result",
                fixed_encoding_value=_predefined("v0", "OPR_VGPR"),
            ),
        ),
        implicit_operands=(
            implicit_memory,
            _implicit_m0_input(xml_operand_required=False),
        ),
        immediate_fields=(offset_field_name, *_cache_field_names(cache_fields)),
        immediates=(
            _offset_immediate(offset_bit_width),
            *_cache_immediates(cache_fields),
        ),
        fixed_encoding_fields=(("IDXEN", 0), ("OFFEN", 1), ("LDS", 1)),
        effects=_global_to_lds_effects(
            global_width_bits,
            workgroup_width_bits=workgroup_width_bits,
        ),
        flags=(DescriptorFlag.SIDE_EFFECTING,),
        operand_forms=_buffer_operand_forms(
            off_zero_descriptor_key=off_zero_descriptor_key,
            vaddr_offset_descriptor_key=vaddr_offset_descriptor_key,
        ),
        asm_forms=_asm(
            mnemonic=asm_mnemonic,
            native_assembly_mnemonic=mnemonic,
            operands=("resource", "vaddr", "soffset", "m0"),
            immediates=_memory_asm_immediate_names(cache_fields),
            named_immediates=True,
        ),
    )


def _buffer_load_lds_vaddr_offset_overlay(
    *,
    descriptor_key: str,
    instruction_name: str,
    mnemonic: str,
    asm_mnemonic: str,
    semantic_tag: str,
    global_width_bits: int,
    workgroup_width_bits: int,
    implicit_memory: AmdgpuImplicitOperandOverlay,
    encoding_name: str,
    resource_field_name: str,
    offset_field_name: str = "OFFSET",
    offset_bit_width: int = 12,
    cache_fields: tuple[tuple[str, int], ...] = (),
) -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key=descriptor_key,
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        encoding_name=encoding_name,
        semantic_tag=semantic_tag,
        schedule_class=_SCHEDULE_VMEM_LOAD_LDS,
        operands=(
            AmdgpuOperandOverlay(
                resource_field_name, _sgpr_resource("resource", units=4)
            ),
            _mubuf_vaddr_operand(),
        ),
        ignored_operands=(
            AmdgpuIgnoredOperandOverlay(
                "VDATA",
                ignore_reason="lds-bit-has-no-vgpr-result",
                fixed_encoding_value=_predefined("v0", "OPR_VGPR"),
            ),
        ),
        implicit_operands=(
            implicit_memory,
            _implicit_m0_input(xml_operand_required=False),
        ),
        immediate_fields=(offset_field_name, *_cache_field_names(cache_fields)),
        immediates=(
            _offset_immediate(offset_bit_width),
            *_cache_immediates(cache_fields),
        ),
        fixed_encoding_fields=(
            ("SOFFSET", _MUBUF_SOFFSET_INLINE_ZERO),
            ("IDXEN", 0),
            ("OFFEN", 1),
            ("LDS", 1),
        ),
        effects=_global_to_lds_effects(
            global_width_bits,
            workgroup_width_bits=workgroup_width_bits,
        ),
        flags=(DescriptorFlag.SIDE_EFFECTING,),
        asm_forms=_asm(
            mnemonic=f"{asm_mnemonic}_vaddr_offset",
            native_assembly_mnemonic=mnemonic,
            operands=("resource", "vaddr", "m0"),
            immediates=_memory_asm_immediate_names(cache_fields),
            named_immediates=True,
        ),
    )


def _buffer_load_lds_off_zero_overlay(
    *,
    descriptor_key: str,
    instruction_name: str,
    mnemonic: str,
    asm_mnemonic: str,
    semantic_tag: str,
    global_width_bits: int,
    workgroup_width_bits: int,
    implicit_memory: AmdgpuImplicitOperandOverlay,
    encoding_name: str,
    resource_field_name: str,
    offset_field_name: str = "OFFSET",
    offset_bit_width: int = 12,
    cache_fields: tuple[tuple[str, int], ...] = (),
) -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key=descriptor_key,
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        encoding_name=encoding_name,
        semantic_tag=semantic_tag,
        schedule_class=_SCHEDULE_VMEM_LOAD_LDS,
        operands=(
            AmdgpuOperandOverlay(
                resource_field_name, _sgpr_resource("resource", units=4)
            ),
        ),
        ignored_operands=(
            AmdgpuIgnoredOperandOverlay(
                "VDATA",
                ignore_reason="lds-bit-has-no-vgpr-result",
                fixed_encoding_value=_predefined("v0", "OPR_VGPR"),
            ),
        ),
        implicit_operands=(
            implicit_memory,
            _implicit_m0_input(xml_operand_required=False),
        ),
        immediate_fields=(offset_field_name, *_cache_field_names(cache_fields)),
        immediates=(
            _offset_immediate(offset_bit_width),
            *_cache_immediates(cache_fields),
        ),
        fixed_encoding_fields=(
            ("VADDR", _predefined("v0")),
            ("SOFFSET", _MUBUF_SOFFSET_INLINE_ZERO),
            ("IDXEN", 0),
            ("OFFEN", 0),
            ("LDS", 1),
        ),
        effects=_global_to_lds_effects(
            global_width_bits,
            workgroup_width_bits=workgroup_width_bits,
        ),
        flags=(DescriptorFlag.SIDE_EFFECTING,),
        asm_forms=_asm(
            mnemonic=f"{asm_mnemonic}_off_zero",
            native_assembly_mnemonic=mnemonic,
            operands=("resource", "m0"),
            immediates=_memory_asm_immediate_names(cache_fields),
            named_immediates=True,
        ),
    )


_BUFFER_LOAD_LDS_CDNA3_VARIANTS = (
    (
        "UBYTE",
        "ubyte",
        "memory.global_to_workgroup.u8.zero_extend",
        8,
        8,
        _IGNORE_GLOBAL_READ_MEMORY_U8,
    ),
    (
        "SBYTE",
        "sbyte",
        "memory.global_to_workgroup.i8.sign_extend",
        8,
        8,
        _IGNORE_GLOBAL_READ_MEMORY_I8,
    ),
    (
        "USHORT",
        "ushort",
        "memory.global_to_workgroup.u16.zero_extend",
        16,
        16,
        _IGNORE_GLOBAL_READ_MEMORY_U16,
    ),
    (
        "SSHORT",
        "sshort",
        "memory.global_to_workgroup.i16.sign_extend",
        16,
        16,
        _IGNORE_GLOBAL_READ_MEMORY_I16,
    ),
    (
        "DWORD",
        "dword",
        "memory.global_to_workgroup.u32",
        32,
        32,
        _IGNORE_GLOBAL_READ_MEMORY,
    ),
)

_BUFFER_LOAD_LDS_GFX950_VARIANTS = (
    *_BUFFER_LOAD_LDS_CDNA3_VARIANTS,
    (
        "DWORDX3",
        "dwordx3",
        "memory.global_to_workgroup.u96",
        96,
        96,
        _IGNORE_GLOBAL_READ_MEMORY_B96,
    ),
    (
        "DWORDX4",
        "dwordx4",
        "memory.global_to_workgroup.u128",
        128,
        128,
        _IGNORE_GLOBAL_READ_MEMORY_B128,
    ),
)


def _buffer_load_lds_overlays(
    *,
    encoding_name: str,
    resource_field_name: str,
    cache_fields: tuple[tuple[str, int], ...] = (),
    variants: tuple[
        tuple[str, str, str, int, int, AmdgpuImplicitOperandOverlay], ...
    ] = (_BUFFER_LOAD_LDS_GFX950_VARIANTS),
) -> tuple[AmdgpuDescriptorOverlay, ...]:
    return tuple(
        overlay
        for (
            instruction_suffix,
            mnemonic_suffix,
            semantic_tag,
            global_width_bits,
            workgroup_width_bits,
            implicit_memory,
        ) in variants
        for overlay in (
            _buffer_load_lds_overlay(
                descriptor_key=f"amdgpu.buffer_load_lds_{mnemonic_suffix}",
                instruction_name=f"BUFFER_LOAD_{instruction_suffix}",
                mnemonic=f"buffer_load_{mnemonic_suffix}",
                asm_mnemonic=f"buffer_load_lds_{mnemonic_suffix}",
                semantic_tag=semantic_tag,
                global_width_bits=global_width_bits,
                workgroup_width_bits=workgroup_width_bits,
                implicit_memory=implicit_memory,
                encoding_name=encoding_name,
                resource_field_name=resource_field_name,
                cache_fields=cache_fields,
                off_zero_descriptor_key=(
                    f"amdgpu.buffer_load_lds_{mnemonic_suffix}_off_zero"
                ),
                vaddr_offset_descriptor_key=(
                    f"amdgpu.buffer_load_lds_{mnemonic_suffix}_vaddr_offset"
                ),
            ),
            _buffer_load_lds_off_zero_overlay(
                descriptor_key=f"amdgpu.buffer_load_lds_{mnemonic_suffix}_off_zero",
                instruction_name=f"BUFFER_LOAD_{instruction_suffix}",
                mnemonic=f"buffer_load_{mnemonic_suffix}",
                asm_mnemonic=f"buffer_load_lds_{mnemonic_suffix}",
                semantic_tag=semantic_tag,
                global_width_bits=global_width_bits,
                workgroup_width_bits=workgroup_width_bits,
                implicit_memory=implicit_memory,
                encoding_name=encoding_name,
                resource_field_name=resource_field_name,
                cache_fields=cache_fields,
            ),
            _buffer_load_lds_vaddr_offset_overlay(
                descriptor_key=f"amdgpu.buffer_load_lds_{mnemonic_suffix}_vaddr_offset",
                instruction_name=f"BUFFER_LOAD_{instruction_suffix}",
                mnemonic=f"buffer_load_{mnemonic_suffix}",
                asm_mnemonic=f"buffer_load_lds_{mnemonic_suffix}",
                semantic_tag=semantic_tag,
                global_width_bits=global_width_bits,
                workgroup_width_bits=workgroup_width_bits,
                implicit_memory=implicit_memory,
                encoding_name=encoding_name,
                resource_field_name=resource_field_name,
                cache_fields=cache_fields,
            ),
        )
    )


def _buffer_store_dword_overlay(
    *,
    encoding_name: str,
    resource_field_name: str,
    offset_field_name: str = "OFFSET",
    offset_bit_width: int = 12,
    cache_fields: tuple[tuple[str, int], ...] = (),
    off_zero_descriptor_key: str | None = "amdgpu.buffer_store_dword_off_zero",
    vaddr_offset_descriptor_key: str | None = (
        "amdgpu.buffer_store_dword_vaddr_offset"
    ),
) -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.buffer_store_dword",
        instruction_name="BUFFER_STORE_DWORD",
        mnemonic="buffer_store_dword",
        encoding_name=encoding_name,
        semantic_tag="memory.store.u32",
        schedule_class=_SCHEDULE_VMEM_STORE,
        operands=(
            AmdgpuOperandOverlay("VDATA", _vgpr_operand("value")),
            AmdgpuOperandOverlay(
                resource_field_name, _sgpr_resource("resource", units=4)
            ),
            _mubuf_vaddr_operand(),
            AmdgpuOperandOverlay("SOFFSET", _sgpr_operand("soffset")),
        ),
        implicit_operands=(_IGNORE_GLOBAL_WRITE_MEMORY,),
        immediate_fields=(offset_field_name, *_cache_field_names(cache_fields)),
        immediates=(
            _offset_immediate(offset_bit_width),
            *_cache_immediates(cache_fields),
        ),
        fixed_encoding_fields=(("IDXEN", 0), ("OFFEN", 1)),
        effects=(_GLOBAL_STORE_EFFECT,),
        flags=(DescriptorFlag.SIDE_EFFECTING,),
        operand_forms=_buffer_operand_forms(
            off_zero_descriptor_key=off_zero_descriptor_key,
            vaddr_offset_descriptor_key=vaddr_offset_descriptor_key,
        ),
    )


def _buffer_store_off_zero_overlay(
    *,
    descriptor_key: str,
    instruction_name: str,
    mnemonic: str,
    semantic_tag: str,
    payload_units: int,
    memory_effect: Effect,
    implicit_memory: AmdgpuImplicitOperandOverlay,
    encoding_name: str,
    resource_field_name: str,
    offset_field_name: str = "OFFSET",
    offset_bit_width: int = 12,
    cache_fields: tuple[tuple[str, int], ...] = (),
) -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key=descriptor_key,
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        encoding_name=encoding_name,
        semantic_tag=semantic_tag,
        schedule_class=_SCHEDULE_VMEM_STORE,
        operands=(
            AmdgpuOperandOverlay("VDATA", _vgpr_operand("value", units=payload_units)),
            AmdgpuOperandOverlay(
                resource_field_name, _sgpr_resource("resource", units=4)
            ),
        ),
        implicit_operands=(implicit_memory,),
        immediate_fields=(offset_field_name, *_cache_field_names(cache_fields)),
        immediates=(
            _offset_immediate(offset_bit_width),
            *_cache_immediates(cache_fields),
        ),
        fixed_encoding_fields=(
            ("VADDR", _predefined("v0")),
            ("SOFFSET", _MUBUF_SOFFSET_INLINE_ZERO),
            ("IDXEN", 0),
            ("OFFEN", 0),
        ),
        effects=(memory_effect,),
        flags=(DescriptorFlag.SIDE_EFFECTING,),
        asm_forms=_asm(
            mnemonic=f"{mnemonic}_off_zero",
            native_assembly_mnemonic=mnemonic,
            operands=("value", "resource"),
            immediates=_memory_asm_immediate_names(cache_fields),
            named_immediates=True,
        ),
    )


def _buffer_store_vaddr_offset_overlay(
    *,
    descriptor_key: str,
    instruction_name: str,
    mnemonic: str,
    semantic_tag: str,
    payload_units: int,
    memory_effect: Effect,
    implicit_memory: AmdgpuImplicitOperandOverlay,
    encoding_name: str,
    resource_field_name: str,
    offset_field_name: str = "OFFSET",
    offset_bit_width: int = 12,
    cache_fields: tuple[tuple[str, int], ...] = (),
    fixed_soffset: AmdgpuFixedEncodingValue = _MUBUF_SOFFSET_INLINE_ZERO,
    fixed_soffset_native_spelling: str = "0",
) -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key=descriptor_key,
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        encoding_name=encoding_name,
        semantic_tag=semantic_tag,
        schedule_class=_SCHEDULE_VMEM_STORE,
        operands=(
            AmdgpuOperandOverlay("VDATA", _vgpr_operand("value", units=payload_units)),
            AmdgpuOperandOverlay(
                resource_field_name, _sgpr_resource("resource", units=4)
            ),
            _mubuf_vaddr_operand(),
        ),
        implicit_operands=(implicit_memory,),
        immediate_fields=(offset_field_name, *_cache_field_names(cache_fields)),
        immediates=(
            _offset_immediate(offset_bit_width),
            *_cache_immediates(cache_fields),
        ),
        fixed_encoding_fields=(
            ("SOFFSET", fixed_soffset),
            ("IDXEN", 0),
            ("OFFEN", 1),
        ),
        effects=(memory_effect,),
        flags=(DescriptorFlag.SIDE_EFFECTING,),
        asm_forms=_asm(
            mnemonic=f"{mnemonic}_vaddr_offset",
            native_assembly_mnemonic=mnemonic,
            operands=("value", "resource", "vaddr"),
            immediates=_memory_asm_immediate_names(cache_fields),
            named_immediates=True,
            native_assembly_values=_buffer_vaddr_offset_native_assembly_values(
                payload_kind=NativeAsmValueKind.OPERAND,
                payload_field_name="value",
                fixed_soffset_native_spelling=fixed_soffset_native_spelling,
            ),
        ),
    )


def _buffer_store_dword_off_zero_overlay(
    *,
    encoding_name: str,
    resource_field_name: str,
    offset_field_name: str = "OFFSET",
    offset_bit_width: int = 12,
    cache_fields: tuple[tuple[str, int], ...] = (),
) -> AmdgpuDescriptorOverlay:
    return _buffer_store_off_zero_overlay(
        descriptor_key="amdgpu.buffer_store_dword_off_zero",
        instruction_name="BUFFER_STORE_DWORD",
        mnemonic="buffer_store_dword",
        semantic_tag="memory.store.u32",
        payload_units=1,
        memory_effect=_GLOBAL_STORE_EFFECT,
        implicit_memory=_IGNORE_GLOBAL_WRITE_MEMORY,
        encoding_name=encoding_name,
        resource_field_name=resource_field_name,
        offset_field_name=offset_field_name,
        offset_bit_width=offset_bit_width,
        cache_fields=cache_fields,
    )


def _buffer_store_dword_vaddr_offset_overlay(
    *,
    encoding_name: str,
    resource_field_name: str,
    offset_field_name: str = "OFFSET",
    offset_bit_width: int = 12,
    cache_fields: tuple[tuple[str, int], ...] = (),
    fixed_soffset: AmdgpuFixedEncodingValue = _MUBUF_SOFFSET_INLINE_ZERO,
    fixed_soffset_native_spelling: str = "0",
) -> AmdgpuDescriptorOverlay:
    return _buffer_store_vaddr_offset_overlay(
        descriptor_key="amdgpu.buffer_store_dword_vaddr_offset",
        instruction_name="BUFFER_STORE_DWORD",
        mnemonic="buffer_store_dword",
        semantic_tag="memory.store.u32",
        payload_units=1,
        memory_effect=_GLOBAL_STORE_EFFECT,
        implicit_memory=_IGNORE_GLOBAL_WRITE_MEMORY,
        encoding_name=encoding_name,
        resource_field_name=resource_field_name,
        offset_field_name=offset_field_name,
        offset_bit_width=offset_bit_width,
        cache_fields=cache_fields,
        fixed_soffset=fixed_soffset,
        fixed_soffset_native_spelling=fixed_soffset_native_spelling,
    )


def _buffer_store_b16_overlay(
    *,
    encoding_name: str,
    resource_field_name: str,
    offset_field_name: str = "OFFSET",
    offset_bit_width: int = 12,
    cache_fields: tuple[tuple[str, int], ...] = (),
    vaddr_offset_descriptor_key: str | None = ("amdgpu.buffer_store_b16_vaddr_offset"),
) -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.buffer_store_b16",
        instruction_name="BUFFER_STORE_SHORT",
        mnemonic="buffer_store_short",
        encoding_name=encoding_name,
        semantic_tag="memory.store.u16.low",
        schedule_class=_SCHEDULE_VMEM_STORE,
        operands=(
            AmdgpuOperandOverlay(
                "VDATA",
                _vgpr_operand("value", register_part=_REG_PART_VGPR_LOW16),
                size_exception_reason=_D16_PARTIAL_REGISTER_SIZE_REASON,
            ),
            AmdgpuOperandOverlay(
                resource_field_name, _sgpr_resource("resource", units=4)
            ),
            _mubuf_vaddr_operand(),
            AmdgpuOperandOverlay("SOFFSET", _sgpr_operand("soffset")),
        ),
        implicit_operands=(_ignore_global_write_memory(16),),
        immediate_fields=(offset_field_name, *_cache_field_names(cache_fields)),
        immediates=(
            _offset_immediate(offset_bit_width),
            *_cache_immediates(cache_fields),
        ),
        fixed_encoding_fields=(("IDXEN", 0), ("OFFEN", 1)),
        effects=(_global_write_effect(16),),
        flags=(DescriptorFlag.SIDE_EFFECTING,),
        operand_forms=_buffer_operand_forms(
            vaddr_offset_descriptor_key=vaddr_offset_descriptor_key,
        ),
    )


def _buffer_store_b16_vaddr_offset_overlay(
    *,
    encoding_name: str,
    resource_field_name: str,
    offset_field_name: str = "OFFSET",
    offset_bit_width: int = 12,
    cache_fields: tuple[tuple[str, int], ...] = (),
    fixed_soffset: AmdgpuFixedEncodingValue = _MUBUF_SOFFSET_INLINE_ZERO,
    fixed_soffset_native_spelling: str = "0",
) -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.buffer_store_b16_vaddr_offset",
        instruction_name="BUFFER_STORE_SHORT",
        mnemonic="buffer_store_short",
        encoding_name=encoding_name,
        semantic_tag="memory.store.u16.low",
        schedule_class=_SCHEDULE_VMEM_STORE,
        operands=(
            AmdgpuOperandOverlay(
                "VDATA",
                _vgpr_operand("value", register_part=_REG_PART_VGPR_LOW16),
                size_exception_reason=_D16_PARTIAL_REGISTER_SIZE_REASON,
            ),
            AmdgpuOperandOverlay(
                resource_field_name, _sgpr_resource("resource", units=4)
            ),
            _mubuf_vaddr_operand(),
        ),
        implicit_operands=(_ignore_global_write_memory(16),),
        immediate_fields=(offset_field_name, *_cache_field_names(cache_fields)),
        immediates=(
            _offset_immediate(offset_bit_width),
            *_cache_immediates(cache_fields),
        ),
        fixed_encoding_fields=(
            ("SOFFSET", fixed_soffset),
            ("IDXEN", 0),
            ("OFFEN", 1),
        ),
        effects=(_global_write_effect(16),),
        flags=(DescriptorFlag.SIDE_EFFECTING,),
        asm_forms=_asm(
            mnemonic="buffer_store_short_vaddr_offset",
            native_assembly_mnemonic="buffer_store_short",
            operands=("value", "resource", "vaddr"),
            immediates=_memory_asm_immediate_names(cache_fields),
            named_immediates=True,
            native_assembly_values=_buffer_vaddr_offset_native_assembly_values(
                payload_kind=NativeAsmValueKind.OPERAND,
                payload_field_name="value",
                fixed_soffset_native_spelling=fixed_soffset_native_spelling,
            ),
        ),
    )


def _buffer_store_b8_overlay(
    *,
    encoding_name: str,
    resource_field_name: str,
    offset_field_name: str = "OFFSET",
    offset_bit_width: int = 12,
    cache_fields: tuple[tuple[str, int], ...] = (),
    vaddr_offset_descriptor_key: str | None = "amdgpu.buffer_store_b8_vaddr_offset",
) -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.buffer_store_b8",
        instruction_name="BUFFER_STORE_BYTE",
        mnemonic="buffer_store_b8",
        encoding_name=encoding_name,
        semantic_tag="memory.store.u8",
        schedule_class=_SCHEDULE_VMEM_STORE,
        operands=(
            AmdgpuOperandOverlay("VDATA", _vgpr_operand("value")),
            AmdgpuOperandOverlay(
                resource_field_name, _sgpr_resource("resource", units=4)
            ),
            _mubuf_vaddr_operand(),
            AmdgpuOperandOverlay("SOFFSET", _sgpr_operand("soffset")),
        ),
        implicit_operands=(_ignore_global_write_memory(8),),
        immediate_fields=(offset_field_name, *_cache_field_names(cache_fields)),
        immediates=(
            _offset_immediate(offset_bit_width),
            *_cache_immediates(cache_fields),
        ),
        fixed_encoding_fields=(("IDXEN", 0), ("OFFEN", 1)),
        effects=(_global_write_effect(8),),
        flags=(DescriptorFlag.SIDE_EFFECTING,),
        operand_forms=_buffer_operand_forms(
            vaddr_offset_descriptor_key=vaddr_offset_descriptor_key,
        ),
    )


def _buffer_store_b8_vaddr_offset_overlay(
    *,
    encoding_name: str,
    resource_field_name: str,
    offset_field_name: str = "OFFSET",
    offset_bit_width: int = 12,
    cache_fields: tuple[tuple[str, int], ...] = (),
    fixed_soffset: AmdgpuFixedEncodingValue = _MUBUF_SOFFSET_INLINE_ZERO,
    fixed_soffset_native_spelling: str = "0",
) -> AmdgpuDescriptorOverlay:
    return _buffer_store_vaddr_offset_overlay(
        descriptor_key="amdgpu.buffer_store_b8_vaddr_offset",
        instruction_name="BUFFER_STORE_BYTE",
        mnemonic="buffer_store_b8",
        semantic_tag="memory.store.u8",
        payload_units=1,
        memory_effect=_global_write_effect(8),
        implicit_memory=_ignore_global_write_memory(8),
        encoding_name=encoding_name,
        resource_field_name=resource_field_name,
        offset_field_name=offset_field_name,
        offset_bit_width=offset_bit_width,
        cache_fields=cache_fields,
        fixed_soffset=fixed_soffset,
        fixed_soffset_native_spelling=fixed_soffset_native_spelling,
    )


def _buffer_store_sized_overlay(
    *,
    descriptor_key: str,
    instruction_name: str,
    mnemonic: str,
    width_bits: int,
    payload_units: int,
    encoding_name: str,
    resource_field_name: str,
    offset_field_name: str = "OFFSET",
    offset_bit_width: int = 12,
    cache_fields: tuple[tuple[str, int], ...] = (),
    off_zero_descriptor_key: str | None = None,
    vaddr_offset_descriptor_key: str | None = None,
) -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key=descriptor_key,
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        encoding_name=encoding_name,
        semantic_tag=f"memory.store.u{width_bits}",
        schedule_class=_SCHEDULE_VMEM_STORE,
        operands=(
            AmdgpuOperandOverlay("VDATA", _vgpr_operand("value", units=payload_units)),
            AmdgpuOperandOverlay(
                resource_field_name, _sgpr_resource("resource", units=4)
            ),
            _mubuf_vaddr_operand(),
            AmdgpuOperandOverlay("SOFFSET", _sgpr_operand("soffset")),
        ),
        implicit_operands=(_ignore_global_write_memory(width_bits),),
        immediate_fields=(offset_field_name, *_cache_field_names(cache_fields)),
        immediates=(
            _offset_immediate(offset_bit_width),
            *_cache_immediates(cache_fields),
        ),
        fixed_encoding_fields=(("IDXEN", 0), ("OFFEN", 1)),
        effects=(_global_write_effect(width_bits),),
        flags=(DescriptorFlag.SIDE_EFFECTING,),
        operand_forms=_buffer_operand_forms(
            off_zero_descriptor_key=off_zero_descriptor_key,
            vaddr_offset_descriptor_key=vaddr_offset_descriptor_key,
        ),
    )


def _buffer_store_64_overlay(
    *,
    descriptor_key: str = "amdgpu.buffer_store_b64",
    instruction_name: str = "BUFFER_STORE_B64",
    mnemonic: str = "buffer_store_b64",
    encoding_name: str,
    resource_field_name: str,
    offset_field_name: str = "OFFSET",
    offset_bit_width: int = 12,
    cache_fields: tuple[tuple[str, int], ...] = (),
    off_zero_descriptor_key: str | None = "amdgpu.buffer_store_b64_off_zero",
    vaddr_offset_descriptor_key: str | None = "amdgpu.buffer_store_b64_vaddr_offset",
) -> AmdgpuDescriptorOverlay:
    return _buffer_store_sized_overlay(
        descriptor_key=descriptor_key,
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        width_bits=64,
        payload_units=2,
        encoding_name=encoding_name,
        resource_field_name=resource_field_name,
        offset_field_name=offset_field_name,
        offset_bit_width=offset_bit_width,
        cache_fields=cache_fields,
        off_zero_descriptor_key=off_zero_descriptor_key,
        vaddr_offset_descriptor_key=vaddr_offset_descriptor_key,
    )


def _buffer_store_64_off_zero_overlay(
    *,
    descriptor_key: str = "amdgpu.buffer_store_b64_off_zero",
    instruction_name: str = "BUFFER_STORE_B64",
    mnemonic: str = "buffer_store_b64",
    encoding_name: str,
    resource_field_name: str,
    offset_field_name: str = "OFFSET",
    offset_bit_width: int = 12,
    cache_fields: tuple[tuple[str, int], ...] = (),
) -> AmdgpuDescriptorOverlay:
    return _buffer_store_off_zero_overlay(
        descriptor_key=descriptor_key,
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        semantic_tag="memory.store.u64",
        payload_units=2,
        memory_effect=_GLOBAL_STORE_B64_EFFECT,
        implicit_memory=_IGNORE_GLOBAL_WRITE_MEMORY_B64,
        encoding_name=encoding_name,
        resource_field_name=resource_field_name,
        offset_field_name=offset_field_name,
        offset_bit_width=offset_bit_width,
        cache_fields=cache_fields,
    )


def _buffer_store_64_vaddr_offset_overlay(
    *,
    descriptor_key: str = "amdgpu.buffer_store_b64_vaddr_offset",
    instruction_name: str = "BUFFER_STORE_B64",
    mnemonic: str = "buffer_store_b64",
    encoding_name: str,
    resource_field_name: str,
    offset_field_name: str = "OFFSET",
    offset_bit_width: int = 12,
    cache_fields: tuple[tuple[str, int], ...] = (),
    fixed_soffset: AmdgpuFixedEncodingValue = _MUBUF_SOFFSET_INLINE_ZERO,
    fixed_soffset_native_spelling: str = "0",
) -> AmdgpuDescriptorOverlay:
    return _buffer_store_vaddr_offset_overlay(
        descriptor_key=descriptor_key,
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        semantic_tag="memory.store.u64",
        payload_units=2,
        memory_effect=_GLOBAL_STORE_B64_EFFECT,
        implicit_memory=_IGNORE_GLOBAL_WRITE_MEMORY_B64,
        encoding_name=encoding_name,
        resource_field_name=resource_field_name,
        offset_field_name=offset_field_name,
        offset_bit_width=offset_bit_width,
        cache_fields=cache_fields,
        fixed_soffset=fixed_soffset,
        fixed_soffset_native_spelling=fixed_soffset_native_spelling,
    )


def _buffer_store_96_overlay(
    *,
    descriptor_key: str = "amdgpu.buffer_store_b96",
    instruction_name: str = "BUFFER_STORE_B96",
    mnemonic: str = "buffer_store_b96",
    encoding_name: str,
    resource_field_name: str,
    offset_field_name: str = "OFFSET",
    offset_bit_width: int = 12,
    cache_fields: tuple[tuple[str, int], ...] = (),
    off_zero_descriptor_key: str | None = "amdgpu.buffer_store_b96_off_zero",
    vaddr_offset_descriptor_key: str | None = "amdgpu.buffer_store_b96_vaddr_offset",
) -> AmdgpuDescriptorOverlay:
    return _buffer_store_sized_overlay(
        descriptor_key=descriptor_key,
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        width_bits=96,
        payload_units=3,
        encoding_name=encoding_name,
        resource_field_name=resource_field_name,
        offset_field_name=offset_field_name,
        offset_bit_width=offset_bit_width,
        cache_fields=cache_fields,
        off_zero_descriptor_key=off_zero_descriptor_key,
        vaddr_offset_descriptor_key=vaddr_offset_descriptor_key,
    )


def _buffer_store_96_off_zero_overlay(
    *,
    descriptor_key: str = "amdgpu.buffer_store_b96_off_zero",
    instruction_name: str = "BUFFER_STORE_B96",
    mnemonic: str = "buffer_store_b96",
    encoding_name: str,
    resource_field_name: str,
    offset_field_name: str = "OFFSET",
    offset_bit_width: int = 12,
    cache_fields: tuple[tuple[str, int], ...] = (),
) -> AmdgpuDescriptorOverlay:
    return _buffer_store_off_zero_overlay(
        descriptor_key=descriptor_key,
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        semantic_tag="memory.store.u96",
        payload_units=3,
        memory_effect=_GLOBAL_STORE_B96_EFFECT,
        implicit_memory=_IGNORE_GLOBAL_WRITE_MEMORY_B96,
        encoding_name=encoding_name,
        resource_field_name=resource_field_name,
        offset_field_name=offset_field_name,
        offset_bit_width=offset_bit_width,
        cache_fields=cache_fields,
    )


def _buffer_store_96_vaddr_offset_overlay(
    *,
    descriptor_key: str = "amdgpu.buffer_store_b96_vaddr_offset",
    instruction_name: str = "BUFFER_STORE_B96",
    mnemonic: str = "buffer_store_b96",
    encoding_name: str,
    resource_field_name: str,
    offset_field_name: str = "OFFSET",
    offset_bit_width: int = 12,
    cache_fields: tuple[tuple[str, int], ...] = (),
    fixed_soffset: AmdgpuFixedEncodingValue = _MUBUF_SOFFSET_INLINE_ZERO,
    fixed_soffset_native_spelling: str = "0",
) -> AmdgpuDescriptorOverlay:
    return _buffer_store_vaddr_offset_overlay(
        descriptor_key=descriptor_key,
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        semantic_tag="memory.store.u96",
        payload_units=3,
        memory_effect=_GLOBAL_STORE_B96_EFFECT,
        implicit_memory=_IGNORE_GLOBAL_WRITE_MEMORY_B96,
        encoding_name=encoding_name,
        resource_field_name=resource_field_name,
        offset_field_name=offset_field_name,
        offset_bit_width=offset_bit_width,
        cache_fields=cache_fields,
        fixed_soffset=fixed_soffset,
        fixed_soffset_native_spelling=fixed_soffset_native_spelling,
    )


def _buffer_store_128_overlay(
    *,
    descriptor_key: str = "amdgpu.buffer_store_b128",
    instruction_name: str = "BUFFER_STORE_B128",
    mnemonic: str = "buffer_store_b128",
    encoding_name: str,
    resource_field_name: str,
    offset_field_name: str = "OFFSET",
    offset_bit_width: int = 12,
    cache_fields: tuple[tuple[str, int], ...] = (),
    off_zero_descriptor_key: str | None = "amdgpu.buffer_store_b128_off_zero",
    vaddr_offset_descriptor_key: str | None = "amdgpu.buffer_store_b128_vaddr_offset",
) -> AmdgpuDescriptorOverlay:
    return _buffer_store_sized_overlay(
        descriptor_key=descriptor_key,
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        width_bits=128,
        payload_units=4,
        encoding_name=encoding_name,
        resource_field_name=resource_field_name,
        offset_field_name=offset_field_name,
        offset_bit_width=offset_bit_width,
        cache_fields=cache_fields,
        off_zero_descriptor_key=off_zero_descriptor_key,
        vaddr_offset_descriptor_key=vaddr_offset_descriptor_key,
    )


def _buffer_store_128_off_zero_overlay(
    *,
    descriptor_key: str = "amdgpu.buffer_store_b128_off_zero",
    instruction_name: str = "BUFFER_STORE_B128",
    mnemonic: str = "buffer_store_b128",
    encoding_name: str,
    resource_field_name: str,
    offset_field_name: str = "OFFSET",
    offset_bit_width: int = 12,
    cache_fields: tuple[tuple[str, int], ...] = (),
) -> AmdgpuDescriptorOverlay:
    return _buffer_store_off_zero_overlay(
        descriptor_key=descriptor_key,
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        semantic_tag="memory.store.u128",
        payload_units=4,
        memory_effect=_GLOBAL_STORE_B128_EFFECT,
        implicit_memory=_IGNORE_GLOBAL_WRITE_MEMORY_B128,
        encoding_name=encoding_name,
        resource_field_name=resource_field_name,
        offset_field_name=offset_field_name,
        offset_bit_width=offset_bit_width,
        cache_fields=cache_fields,
    )


def _buffer_store_128_vaddr_offset_overlay(
    *,
    descriptor_key: str = "amdgpu.buffer_store_b128_vaddr_offset",
    instruction_name: str = "BUFFER_STORE_B128",
    mnemonic: str = "buffer_store_b128",
    encoding_name: str,
    resource_field_name: str,
    offset_field_name: str = "OFFSET",
    offset_bit_width: int = 12,
    cache_fields: tuple[tuple[str, int], ...] = (),
    fixed_soffset: AmdgpuFixedEncodingValue = _MUBUF_SOFFSET_INLINE_ZERO,
    fixed_soffset_native_spelling: str = "0",
) -> AmdgpuDescriptorOverlay:
    return _buffer_store_vaddr_offset_overlay(
        descriptor_key=descriptor_key,
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        semantic_tag="memory.store.u128",
        payload_units=4,
        memory_effect=_GLOBAL_STORE_B128_EFFECT,
        implicit_memory=_IGNORE_GLOBAL_WRITE_MEMORY_B128,
        encoding_name=encoding_name,
        resource_field_name=resource_field_name,
        offset_field_name=offset_field_name,
        offset_bit_width=offset_bit_width,
        cache_fields=cache_fields,
        fixed_soffset=fixed_soffset,
        fixed_soffset_native_spelling=fixed_soffset_native_spelling,
    )


def _buffer_b16_memory_overlays(
    *,
    encoding_name: str,
    resource_field_name: str,
    offset_field_name: str = "OFFSET",
    offset_bit_width: int = 12,
    cache_fields: tuple[tuple[str, int], ...] = (),
    include_off_zero: bool = True,
    include_vaddr_offset: bool = True,
    fixed_soffset: AmdgpuFixedEncodingValue = _MUBUF_SOFFSET_INLINE_ZERO,
    fixed_soffset_native_spelling: str = "0",
) -> tuple[AmdgpuDescriptorOverlay, ...]:
    return (
        *_buffer_load_narrow_overlays(
            rows=_VMEM_NARROW_B16_LOAD_ROWS,
            encoding_name=encoding_name,
            resource_field_name=resource_field_name,
            offset_field_name=offset_field_name,
            offset_bit_width=offset_bit_width,
            cache_fields=cache_fields,
            include_off_zero=include_off_zero,
            include_vaddr_offset=include_vaddr_offset,
            fixed_soffset=fixed_soffset,
            fixed_soffset_native_spelling=fixed_soffset_native_spelling,
        ),
        _buffer_load_b16_d16_overlay(
            encoding_name=encoding_name,
            resource_field_name=resource_field_name,
            offset_field_name=offset_field_name,
            offset_bit_width=offset_bit_width,
            cache_fields=cache_fields,
            vaddr_offset_descriptor_key=(
                "amdgpu.buffer_load_b16_d16_vaddr_offset"
                if include_vaddr_offset
                else None
            ),
        ),
        *(
            (
                _buffer_load_b16_d16_vaddr_offset_overlay(
                    encoding_name=encoding_name,
                    resource_field_name=resource_field_name,
                    offset_field_name=offset_field_name,
                    offset_bit_width=offset_bit_width,
                    cache_fields=cache_fields,
                    fixed_soffset=fixed_soffset,
                    fixed_soffset_native_spelling=fixed_soffset_native_spelling,
                ),
            )
            if include_vaddr_offset
            else ()
        ),
        _buffer_store_b16_overlay(
            encoding_name=encoding_name,
            resource_field_name=resource_field_name,
            offset_field_name=offset_field_name,
            offset_bit_width=offset_bit_width,
            cache_fields=cache_fields,
            vaddr_offset_descriptor_key=(
                "amdgpu.buffer_store_b16_vaddr_offset" if include_vaddr_offset else None
            ),
        ),
        *(
            (
                _buffer_store_b16_vaddr_offset_overlay(
                    encoding_name=encoding_name,
                    resource_field_name=resource_field_name,
                    offset_field_name=offset_field_name,
                    offset_bit_width=offset_bit_width,
                    cache_fields=cache_fields,
                    fixed_soffset=fixed_soffset,
                    fixed_soffset_native_spelling=fixed_soffset_native_spelling,
                ),
            )
            if include_vaddr_offset
            else ()
        ),
    )


def _buffer_byte_memory_overlays(
    *,
    encoding_name: str,
    resource_field_name: str,
    offset_field_name: str = "OFFSET",
    offset_bit_width: int = 12,
    cache_fields: tuple[tuple[str, int], ...] = (),
    include_off_zero: bool = True,
    include_vaddr_offset: bool = True,
    fixed_soffset: AmdgpuFixedEncodingValue = _MUBUF_SOFFSET_INLINE_ZERO,
    fixed_soffset_native_spelling: str = "0",
) -> tuple[AmdgpuDescriptorOverlay, ...]:
    return (
        *_buffer_load_narrow_overlays(
            rows=_VMEM_NARROW_BYTE_LOAD_ROWS,
            encoding_name=encoding_name,
            resource_field_name=resource_field_name,
            offset_field_name=offset_field_name,
            offset_bit_width=offset_bit_width,
            cache_fields=cache_fields,
            include_off_zero=include_off_zero,
            include_vaddr_offset=include_vaddr_offset,
            fixed_soffset=fixed_soffset,
            fixed_soffset_native_spelling=fixed_soffset_native_spelling,
        ),
        _buffer_store_b8_overlay(
            encoding_name=encoding_name,
            resource_field_name=resource_field_name,
            offset_field_name=offset_field_name,
            offset_bit_width=offset_bit_width,
            cache_fields=cache_fields,
            vaddr_offset_descriptor_key=(
                "amdgpu.buffer_store_b8_vaddr_offset" if include_vaddr_offset else None
            ),
        ),
        *(
            (
                _buffer_store_b8_vaddr_offset_overlay(
                    encoding_name=encoding_name,
                    resource_field_name=resource_field_name,
                    offset_field_name=offset_field_name,
                    offset_bit_width=offset_bit_width,
                    cache_fields=cache_fields,
                    fixed_soffset=fixed_soffset,
                    fixed_soffset_native_spelling=fixed_soffset_native_spelling,
                ),
            )
            if include_vaddr_offset
            else ()
        ),
    )


def _global_load_overlay(
    *,
    descriptor_key: str,
    instruction_name: str,
    mnemonic: str,
    encoding_name: str,
    address_field_name: str,
    data_field_name: str,
    offset_field_name: str,
    offset_bit_width: int,
    saddr_off: AmdgpuFixedEncodingValue | None,
    width_bits: int,
    units: int,
    address_units: int,
    implicit_m0: bool = False,
    global_read_memory: AmdgpuImplicitOperandOverlay | None = None,
    semantic_tag: str | None = None,
    cache_fields: tuple[tuple[str, int], ...] = (),
) -> AmdgpuDescriptorOverlay:
    implicit_operands: tuple[AmdgpuImplicitOperandOverlay, ...] = (
        global_read_memory or _ignore_global_read_memory(width_bits),
    )
    if implicit_m0:
        implicit_operands += (_implicit_m0_input(),)
    operands: tuple[AmdgpuOperandOverlay, ...] = (
        AmdgpuOperandOverlay(data_field_name, _vgpr_result(units=units)),
        _global_addr_operand(
            address_field_name, units=address_units, has_saddr=saddr_off is None
        ),
    )
    fixed_encoding_fields: tuple[tuple[str, AmdgpuFixedEncodingValue], ...] = ()
    if saddr_off is None:
        operands += (AmdgpuOperandOverlay("SADDR", _sgpr_operand("saddr", units=2)),)
    else:
        fixed_encoding_fields = (("SADDR", saddr_off),)
    return AmdgpuDescriptorOverlay(
        descriptor_key=descriptor_key,
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        encoding_name=encoding_name,
        semantic_tag=semantic_tag or f"memory.load.u{width_bits}",
        schedule_class=_SCHEDULE_VMEM_LOAD,
        operands=operands,
        implicit_operands=implicit_operands,
        immediate_fields=(offset_field_name, *_cache_field_names(cache_fields)),
        immediates=(
            _signed_offset_immediate(offset_bit_width),
            *_cache_immediates(cache_fields),
        ),
        fixed_encoding_fields=fixed_encoding_fields,
        effects=(_global_read_effect(width_bits),),
        flags=(DescriptorFlag.SIDE_EFFECTING,),
        asm_forms=_global_saddr_asm(
            mnemonic=mnemonic,
            results=("dst",),
            operands=("addr", "saddr"),
            implicit_m0=implicit_m0,
            immediates=_memory_asm_immediate_names(cache_fields),
        )
        if saddr_off is None
        else _global_vaddr_asm(
            mnemonic=mnemonic,
            results=("dst",),
            operands=("addr",),
            immediates=_memory_asm_immediate_names(cache_fields),
        ),
    )


def _global_load_b16_d16_overlay(
    *,
    descriptor_key: str,
    encoding_name: str,
    address_field_name: str,
    data_field_name: str,
    offset_field_name: str,
    offset_bit_width: int,
    saddr_off: AmdgpuFixedEncodingValue | None,
    address_units: int,
    implicit_m0: bool = False,
    cache_fields: tuple[tuple[str, int], ...] = (),
) -> AmdgpuDescriptorOverlay:
    implicit_operands: tuple[AmdgpuImplicitOperandOverlay, ...] = (
        _ignore_global_read_memory(16),
    )
    if implicit_m0:
        implicit_operands += (_implicit_m0_input(),)
    operands: tuple[AmdgpuOperandOverlay, ...] = (
        AmdgpuOperandOverlay(
            data_field_name,
            _vgpr_result(register_part=_REG_PART_VGPR_LOW16),
            size_exception_reason=_D16_PARTIAL_REGISTER_SIZE_REASON,
        ),
        _global_addr_operand(
            address_field_name, units=address_units, has_saddr=saddr_off is None
        ),
    )
    fixed_encoding_fields: tuple[tuple[str, AmdgpuFixedEncodingValue], ...] = ()
    if saddr_off is None:
        operands += (AmdgpuOperandOverlay("SADDR", _sgpr_operand("saddr", units=2)),)
    else:
        fixed_encoding_fields = (("SADDR", saddr_off),)
    return AmdgpuDescriptorOverlay(
        descriptor_key=descriptor_key,
        instruction_name="GLOBAL_LOAD_SHORT_D16",
        mnemonic="global_load_d16_b16",
        encoding_name=encoding_name,
        semantic_tag="memory.load.u16.d16.low",
        schedule_class=_SCHEDULE_VMEM_LOAD,
        operands=operands,
        implicit_operands=implicit_operands,
        immediate_fields=(offset_field_name, *_cache_field_names(cache_fields)),
        immediates=(
            _signed_offset_immediate(offset_bit_width),
            *_cache_immediates(cache_fields),
        ),
        fixed_encoding_fields=fixed_encoding_fields,
        effects=(_global_read_effect(16),),
        flags=(DescriptorFlag.SIDE_EFFECTING,),
        asm_forms=_global_saddr_asm(
            mnemonic="global_load_d16_b16",
            results=("dst",),
            operands=("addr", "saddr"),
            implicit_m0=implicit_m0,
            immediates=_memory_asm_immediate_names(cache_fields),
        )
        if saddr_off is None
        else _global_vaddr_asm(
            mnemonic="global_load_d16_b16",
            results=("dst",),
            operands=("addr",),
            immediates=_memory_asm_immediate_names(cache_fields),
        ),
    )


def _global_load_narrow_overlays(
    *,
    rows: tuple[tuple[str, str, int, str, AmdgpuImplicitOperandOverlay], ...],
    mnemonic_suffixes: tuple[str, ...],
    encoding_name: str,
    address_field_name: str,
    data_field_name: str,
    offset_field_name: str,
    offset_bit_width: int,
    saddr_off: AmdgpuFixedEncodingValue | None,
    address_units: int,
    descriptor_key_suffix: str = "",
    implicit_m0: bool = False,
    cache_fields: tuple[tuple[str, int], ...] = (),
) -> tuple[AmdgpuDescriptorOverlay, ...]:
    return tuple(
        _global_load_overlay(
            descriptor_key=(
                f"amdgpu.global_load_{descriptor_suffix}{descriptor_key_suffix}"
            ),
            instruction_name=f"GLOBAL_LOAD_{instruction_suffix}",
            mnemonic=f"global_load_{mnemonic_suffix}",
            encoding_name=encoding_name,
            address_field_name=address_field_name,
            data_field_name=data_field_name,
            offset_field_name=offset_field_name,
            offset_bit_width=offset_bit_width,
            saddr_off=saddr_off,
            width_bits=width_bits,
            units=1,
            address_units=address_units,
            implicit_m0=implicit_m0,
            global_read_memory=implicit_memory,
            semantic_tag=semantic_tag,
            cache_fields=cache_fields,
        )
        for (
            descriptor_suffix,
            instruction_suffix,
            width_bits,
            semantic_tag,
            implicit_memory,
        ), mnemonic_suffix in zip(rows, mnemonic_suffixes, strict=True)
    )


def _scratch_load_overlay(
    *,
    descriptor_key: str,
    instruction_name: str,
    mnemonic: str,
    encoding_name: str,
    address_field_name: str,
    data_field_name: str,
    offset_field_name: str,
    offset_bit_width: int,
    offset_signed: bool,
    width_bits: int,
    units: int,
    semantic_tag: str | None = None,
    implicit_memory: AmdgpuImplicitOperandOverlay | None = None,
    fixed_vaddr: AmdgpuFixedEncodingValue | None,
    fixed_saddr: AmdgpuFixedEncodingValue | None,
    implicit_flat_scratch: bool,
    implicit_m0: bool,
    cache_fields: tuple[tuple[str, int], ...] = (),
) -> AmdgpuDescriptorOverlay:
    operands: tuple[AmdgpuOperandOverlay, ...] = (
        AmdgpuOperandOverlay(data_field_name, _vgpr_result(units=units)),
    )
    implicit_operands: tuple[AmdgpuImplicitOperandOverlay, ...] = (
        implicit_memory or _ignore_scratch_memory(width_bits=width_bits, is_input=True),
    )
    if implicit_flat_scratch:
        implicit_operands = (*implicit_operands, _IGNORE_FLAT_SCRATCH_INPUT)
    if implicit_m0:
        implicit_operands = (*implicit_operands, _implicit_m0_clobber())
    fixed_encoding_fields: tuple[tuple[str, AmdgpuFixedEncodingValue], ...] = (
        ("SVE", 1 if fixed_vaddr is None else 0),
    )
    if fixed_vaddr is None:
        operands += (AmdgpuOperandOverlay(address_field_name, _vgpr_operand("addr")),)
    else:
        fixed_encoding_fields += ((address_field_name, fixed_vaddr),)
    if fixed_saddr is None:
        operands += (AmdgpuOperandOverlay("SADDR", _sgpr_operand("saddr")),)
    else:
        fixed_encoding_fields += (("SADDR", fixed_saddr),)
    if fixed_vaddr is None and fixed_saddr is None:
        asm_mnemonic = f"{mnemonic}_svs"
        asm_operands: tuple[str, ...] = ("addr", "saddr")
    elif fixed_vaddr is None:
        asm_mnemonic = f"{mnemonic}_vaddr"
        asm_operands = ("addr",)
    elif fixed_saddr is None:
        asm_mnemonic = f"{mnemonic}_saddr"
        asm_operands = ("saddr",)
    else:
        asm_mnemonic = f"{mnemonic}_offset_only"
        asm_operands = ()
    return AmdgpuDescriptorOverlay(
        descriptor_key=descriptor_key,
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        encoding_name=encoding_name,
        semantic_tag=semantic_tag or f"memory.stack.load.u{width_bits}",
        schedule_class=_SCHEDULE_VMEM_LOAD,
        operands=operands,
        implicit_operands=implicit_operands,
        immediate_fields=(offset_field_name, *_cache_field_names(cache_fields)),
        immediates=(
            _signed_offset_immediate(offset_bit_width)
            if offset_signed
            else _offset_immediate(offset_bit_width),
            *_cache_immediates(cache_fields),
        ),
        fixed_encoding_fields=fixed_encoding_fields,
        effects=(_stack_memory_effect(EffectKind.READ, width_bits),),
        constraints=_EARLY_CLOBBER_RESULT_CONSTRAINTS,
        flags=(DescriptorFlag.SIDE_EFFECTING,),
        asm_forms=_asm(
            mnemonic=asm_mnemonic,
            native_assembly_mnemonic=mnemonic
            if asm_mnemonic.endswith("_offset_only")
            else None,
            results=("dst",),
            operands=asm_operands,
            immediates=_memory_asm_immediate_names(cache_fields),
            named_immediates=True,
        ),
    )


def _scratch_store_overlay(
    *,
    descriptor_key: str,
    instruction_name: str,
    mnemonic: str,
    encoding_name: str,
    address_field_name: str,
    data_field_name: str,
    offset_field_name: str,
    offset_bit_width: int,
    offset_signed: bool,
    width_bits: int,
    units: int,
    fixed_vaddr: AmdgpuFixedEncodingValue | None,
    fixed_saddr: AmdgpuFixedEncodingValue | None,
    implicit_flat_scratch: bool,
    implicit_m0: bool,
    cache_fields: tuple[tuple[str, int], ...] = (),
) -> AmdgpuDescriptorOverlay:
    operands: tuple[AmdgpuOperandOverlay, ...] = ()
    implicit_operands: tuple[AmdgpuImplicitOperandOverlay, ...] = (
        _ignore_scratch_memory(width_bits=width_bits, is_input=False),
    )
    if implicit_flat_scratch:
        implicit_operands = (*implicit_operands, _IGNORE_FLAT_SCRATCH_INPUT)
    if implicit_m0:
        implicit_operands = (*implicit_operands, _implicit_m0_clobber())
    fixed_encoding_fields: tuple[tuple[str, AmdgpuFixedEncodingValue], ...] = (
        ("SVE", 1 if fixed_vaddr is None else 0),
    )
    if fixed_vaddr is None:
        operands += (AmdgpuOperandOverlay(address_field_name, _vgpr_operand("addr")),)
    else:
        fixed_encoding_fields += ((address_field_name, fixed_vaddr),)
    operands += (
        AmdgpuOperandOverlay(data_field_name, _vgpr_operand("value", units=units)),
    )
    if fixed_saddr is None:
        operands += (AmdgpuOperandOverlay("SADDR", _sgpr_operand("saddr")),)
    else:
        fixed_encoding_fields += (("SADDR", fixed_saddr),)
    if fixed_vaddr is None and fixed_saddr is None:
        asm_mnemonic = f"{mnemonic}_svs"
        asm_operands: tuple[str, ...] = ("addr", "value", "saddr")
    elif fixed_vaddr is None:
        asm_mnemonic = f"{mnemonic}_vaddr"
        asm_operands = ("addr", "value")
    elif fixed_saddr is None:
        asm_mnemonic = f"{mnemonic}_saddr"
        asm_operands = ("value", "saddr")
    else:
        asm_mnemonic = f"{mnemonic}_offset_only"
        asm_operands = ("value",)
    return AmdgpuDescriptorOverlay(
        descriptor_key=descriptor_key,
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        encoding_name=encoding_name,
        semantic_tag=f"memory.stack.store.u{width_bits}",
        schedule_class=_SCHEDULE_VMEM_STORE,
        operands=operands,
        implicit_operands=implicit_operands,
        immediate_fields=(offset_field_name, *_cache_field_names(cache_fields)),
        immediates=(
            _signed_offset_immediate(offset_bit_width)
            if offset_signed
            else _offset_immediate(offset_bit_width),
            *_cache_immediates(cache_fields),
        ),
        fixed_encoding_fields=fixed_encoding_fields,
        effects=(_stack_memory_effect(EffectKind.WRITE, width_bits),),
        flags=(DescriptorFlag.SIDE_EFFECTING,),
        asm_forms=_asm(
            mnemonic=asm_mnemonic,
            native_assembly_mnemonic=mnemonic
            if asm_mnemonic.endswith("_offset_only")
            else None,
            operands=asm_operands,
            immediates=_memory_asm_immediate_names(cache_fields),
            named_immediates=True,
        ),
    )


def _scratch_load_narrow_overlays(
    *,
    rows: tuple[tuple[str, str, int, str, str], ...],
    mnemonic_suffixes: tuple[str, ...],
    encoding_name: str,
    address_field_name: str,
    data_field_name: str,
    offset_field_name: str,
    offset_bit_width: int,
    offset_signed: bool,
    fixed_vaddr: AmdgpuFixedEncodingValue | None,
    fixed_saddr: AmdgpuFixedEncodingValue | None,
    implicit_flat_scratch: bool,
    implicit_m0: bool,
    descriptor_key_suffix: str = "",
    cache_fields: tuple[tuple[str, int], ...] = (),
) -> tuple[AmdgpuDescriptorOverlay, ...]:
    return tuple(
        _scratch_load_overlay(
            descriptor_key=(
                f"amdgpu.scratch_load_{descriptor_suffix}{descriptor_key_suffix}"
            ),
            instruction_name=f"SCRATCH_LOAD_{instruction_suffix}",
            mnemonic=f"scratch_load_{mnemonic_suffix}",
            encoding_name=encoding_name,
            address_field_name=address_field_name,
            data_field_name=data_field_name,
            offset_field_name=offset_field_name,
            offset_bit_width=offset_bit_width,
            offset_signed=offset_signed,
            width_bits=width_bits,
            units=1,
            semantic_tag=semantic_tag,
            implicit_memory=_ignore_scratch_memory(
                width_bits=width_bits,
                is_input=True,
                data_format_name=data_format_name,
            ),
            fixed_vaddr=fixed_vaddr,
            fixed_saddr=fixed_saddr,
            implicit_flat_scratch=implicit_flat_scratch,
            implicit_m0=implicit_m0,
            cache_fields=cache_fields,
        )
        for (
            descriptor_suffix,
            instruction_suffix,
            width_bits,
            semantic_tag,
            data_format_name,
        ), mnemonic_suffix in zip(rows, mnemonic_suffixes, strict=True)
    )


def _scratch_memory_overlays(
    *,
    instruction_suffixes: tuple[str, ...],
    mnemonic_suffixes: tuple[str, ...],
    encoding_name: str,
    address_field_name: str,
    load_data_field_name: str,
    store_data_field_name: str,
    offset_field_name: str,
    offset_bit_width: int,
    offset_signed: bool,
    fixed_vaddr: AmdgpuFixedEncodingValue | None,
    fixed_saddr: AmdgpuFixedEncodingValue | None,
    implicit_flat_scratch: bool = False,
    implicit_m0: bool = False,
    descriptor_key_suffix: str = "",
    narrow_byte_load_mnemonic_suffixes: tuple[str, str] = ("u8", "i8"),
    narrow_b16_load_mnemonic_suffixes: tuple[str, str] = ("u16", "i16"),
    cache_fields: tuple[tuple[str, int], ...] = (),
) -> tuple[AmdgpuDescriptorOverlay, ...]:
    return (
        *_scratch_load_narrow_overlays(
            rows=_SCRATCH_NARROW_BYTE_LOAD_ROWS,
            mnemonic_suffixes=narrow_byte_load_mnemonic_suffixes,
            encoding_name=encoding_name,
            address_field_name=address_field_name,
            data_field_name=load_data_field_name,
            offset_field_name=offset_field_name,
            offset_bit_width=offset_bit_width,
            offset_signed=offset_signed,
            fixed_vaddr=fixed_vaddr,
            fixed_saddr=fixed_saddr,
            implicit_flat_scratch=implicit_flat_scratch,
            implicit_m0=implicit_m0,
            descriptor_key_suffix=descriptor_key_suffix,
            cache_fields=cache_fields,
        ),
        *_scratch_load_narrow_overlays(
            rows=_SCRATCH_NARROW_B16_LOAD_ROWS,
            mnemonic_suffixes=narrow_b16_load_mnemonic_suffixes,
            encoding_name=encoding_name,
            address_field_name=address_field_name,
            data_field_name=load_data_field_name,
            offset_field_name=offset_field_name,
            offset_bit_width=offset_bit_width,
            offset_signed=offset_signed,
            fixed_vaddr=fixed_vaddr,
            fixed_saddr=fixed_saddr,
            implicit_flat_scratch=implicit_flat_scratch,
            implicit_m0=implicit_m0,
            descriptor_key_suffix=descriptor_key_suffix,
            cache_fields=cache_fields,
        ),
        *(
            _scratch_load_overlay(
                descriptor_key=(
                    f"amdgpu.scratch_load_b{width_bits}{descriptor_key_suffix}"
                ),
                instruction_name=f"SCRATCH_LOAD_{instruction_suffix}",
                mnemonic=f"scratch_load_{mnemonic_suffix}",
                encoding_name=encoding_name,
                address_field_name=address_field_name,
                data_field_name=load_data_field_name,
                offset_field_name=offset_field_name,
                offset_bit_width=offset_bit_width,
                offset_signed=offset_signed,
                width_bits=width_bits,
                units=units,
                fixed_vaddr=fixed_vaddr,
                fixed_saddr=fixed_saddr,
                implicit_flat_scratch=implicit_flat_scratch,
                implicit_m0=implicit_m0,
                cache_fields=cache_fields,
            )
            for (width_bits, units), instruction_suffix, mnemonic_suffix in zip(
                _MEMORY_DWORD_VECTOR_WIDTHS,
                instruction_suffixes,
                mnemonic_suffixes,
                strict=True,
            )
        ),
        *(
            _scratch_store_overlay(
                descriptor_key=(
                    f"amdgpu.scratch_store_b{width_bits}{descriptor_key_suffix}"
                ),
                instruction_name=f"SCRATCH_STORE_{instruction_suffix}",
                mnemonic=f"scratch_store_{mnemonic_suffix}",
                encoding_name=encoding_name,
                address_field_name=address_field_name,
                data_field_name=store_data_field_name,
                offset_field_name=offset_field_name,
                offset_bit_width=offset_bit_width,
                offset_signed=offset_signed,
                width_bits=width_bits,
                units=units,
                fixed_vaddr=fixed_vaddr,
                fixed_saddr=fixed_saddr,
                implicit_flat_scratch=implicit_flat_scratch,
                implicit_m0=implicit_m0,
                cache_fields=cache_fields,
            )
            for (width_bits, units), instruction_suffix, mnemonic_suffix in zip(
                _MEMORY_DWORD_VECTOR_WIDTHS,
                instruction_suffixes,
                mnemonic_suffixes,
                strict=True,
            )
        ),
    )


def _global_store_overlay(
    *,
    descriptor_key: str,
    instruction_name: str,
    mnemonic: str,
    encoding_name: str,
    address_field_name: str,
    data_field_name: str,
    offset_field_name: str,
    offset_bit_width: int,
    saddr_off: AmdgpuFixedEncodingValue | None,
    width_bits: int,
    units: int,
    address_units: int,
    implicit_m0: bool = False,
    cache_fields: tuple[tuple[str, int], ...] = (),
) -> AmdgpuDescriptorOverlay:
    implicit_operands: tuple[AmdgpuImplicitOperandOverlay, ...] = (
        _ignore_global_write_memory(width_bits),
    )
    if implicit_m0:
        implicit_operands += (_implicit_m0_input(),)
    operands: tuple[AmdgpuOperandOverlay, ...] = (
        _global_addr_operand(
            address_field_name, units=address_units, has_saddr=saddr_off is None
        ),
        AmdgpuOperandOverlay(data_field_name, _vgpr_operand("value", units=units)),
    )
    fixed_encoding_fields: tuple[tuple[str, AmdgpuFixedEncodingValue], ...] = ()
    if saddr_off is None:
        operands += (AmdgpuOperandOverlay("SADDR", _sgpr_operand("saddr", units=2)),)
    else:
        fixed_encoding_fields = (("SADDR", saddr_off),)
    return AmdgpuDescriptorOverlay(
        descriptor_key=descriptor_key,
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        encoding_name=encoding_name,
        semantic_tag=f"memory.store.u{width_bits}",
        schedule_class=_SCHEDULE_VMEM_STORE,
        operands=operands,
        implicit_operands=implicit_operands,
        immediate_fields=(offset_field_name, *_cache_field_names(cache_fields)),
        immediates=(
            _signed_offset_immediate(offset_bit_width),
            *_cache_immediates(cache_fields),
        ),
        fixed_encoding_fields=fixed_encoding_fields,
        effects=(_global_write_effect(width_bits),),
        flags=(DescriptorFlag.SIDE_EFFECTING,),
        asm_forms=_global_saddr_asm(
            mnemonic=mnemonic,
            operands=("addr", "value", "saddr"),
            implicit_m0=implicit_m0,
            immediates=_memory_asm_immediate_names(cache_fields),
        )
        if saddr_off is None
        else _global_vaddr_asm(
            mnemonic=mnemonic,
            operands=("addr", "value"),
            immediates=_memory_asm_immediate_names(cache_fields),
        ),
    )


def _global_store_b16_overlay(
    *,
    descriptor_key: str,
    encoding_name: str,
    address_field_name: str,
    data_field_name: str,
    offset_field_name: str,
    offset_bit_width: int,
    saddr_off: AmdgpuFixedEncodingValue | None,
    address_units: int,
    implicit_m0: bool = False,
    cache_fields: tuple[tuple[str, int], ...] = (),
) -> AmdgpuDescriptorOverlay:
    implicit_operands: tuple[AmdgpuImplicitOperandOverlay, ...] = (
        _ignore_global_write_memory(16),
    )
    if implicit_m0:
        implicit_operands += (_implicit_m0_input(),)
    operands: tuple[AmdgpuOperandOverlay, ...] = (
        _global_addr_operand(
            address_field_name, units=address_units, has_saddr=saddr_off is None
        ),
        AmdgpuOperandOverlay(
            data_field_name,
            _vgpr_operand("value", register_part=_REG_PART_VGPR_LOW16),
            size_exception_reason=_D16_PARTIAL_REGISTER_SIZE_REASON,
        ),
    )
    fixed_encoding_fields: tuple[tuple[str, AmdgpuFixedEncodingValue], ...] = ()
    if saddr_off is None:
        operands += (AmdgpuOperandOverlay("SADDR", _sgpr_operand("saddr", units=2)),)
    else:
        fixed_encoding_fields = (("SADDR", saddr_off),)
    return AmdgpuDescriptorOverlay(
        descriptor_key=descriptor_key,
        instruction_name="GLOBAL_STORE_SHORT",
        mnemonic="global_store_short",
        encoding_name=encoding_name,
        semantic_tag="memory.store.u16.low",
        schedule_class=_SCHEDULE_VMEM_STORE,
        operands=operands,
        implicit_operands=implicit_operands,
        immediate_fields=(offset_field_name, *_cache_field_names(cache_fields)),
        immediates=(
            _signed_offset_immediate(offset_bit_width),
            *_cache_immediates(cache_fields),
        ),
        fixed_encoding_fields=fixed_encoding_fields,
        effects=(_global_write_effect(16),),
        flags=(DescriptorFlag.SIDE_EFFECTING,),
        asm_forms=_global_saddr_asm(
            mnemonic="global_store_short",
            operands=("addr", "value", "saddr"),
            implicit_m0=implicit_m0,
            immediates=_memory_asm_immediate_names(cache_fields),
        )
        if saddr_off is None
        else _global_vaddr_asm(
            mnemonic="global_store_short",
            operands=("addr", "value"),
            immediates=_memory_asm_immediate_names(cache_fields),
        ),
    )


def _global_load_lds_overlay(
    *,
    descriptor_key: str,
    instruction_name: str,
    mnemonic: str,
    semantic_tag: str,
    global_width_bits: int,
    workgroup_width_bits: int,
    address_units: int,
    saddr_off: AmdgpuFixedEncodingValue | None,
    offset_field_name: str = "OFFSET",
    offset_bit_width: int = 13,
    cache_fields: tuple[tuple[str, int], ...] = (),
) -> AmdgpuDescriptorOverlay:
    operands: tuple[AmdgpuOperandOverlay, ...] = (
        _global_addr_operand("ADDR", units=address_units, has_saddr=saddr_off is None),
    )
    fixed_encoding_fields: tuple[tuple[str, AmdgpuFixedEncodingValue], ...] = ()
    if saddr_off is None:
        operands += (AmdgpuOperandOverlay("SADDR", _sgpr_operand("saddr", units=2)),)
    else:
        fixed_encoding_fields = (("SADDR", saddr_off),)
    return AmdgpuDescriptorOverlay(
        descriptor_key=descriptor_key,
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        encoding_name="ENC_FLAT_GLBL",
        semantic_tag=semantic_tag,
        schedule_class=_SCHEDULE_VMEM_LOAD_LDS,
        operands=operands,
        ignored_operands=(
            AmdgpuIgnoredOperandOverlay(
                "VDST",
                ignore_reason="legacy-lds-dma-has-no-vgpr-result",
                fixed_encoding_value=_predefined("v0", "OPR_VGPR"),
            ),
        ),
        implicit_operands=(_implicit_m0_input(),),
        immediate_fields=(offset_field_name, *_cache_field_names(cache_fields)),
        immediates=(
            _signed_offset_immediate(offset_bit_width),
            *_cache_immediates(cache_fields),
        ),
        fixed_encoding_fields=fixed_encoding_fields,
        effects=_global_to_lds_effects(
            global_width_bits,
            workgroup_width_bits=workgroup_width_bits,
        ),
        flags=(DescriptorFlag.SIDE_EFFECTING,),
        asm_forms=_global_saddr_asm(
            mnemonic=mnemonic,
            operands=("addr", "saddr"),
            implicit_m0=True,
            immediates=_memory_asm_immediate_names(cache_fields),
        )
        if saddr_off is None
        else _asm(
            mnemonic=f"{mnemonic}_vaddr",
            operands=("addr", "m0"),
            immediates=_memory_asm_immediate_names(cache_fields),
            named_immediates=True,
        ),
    )


_GLOBAL_LOAD_LDS_BASE_VARIANTS = (
    (
        "UBYTE",
        "ubyte",
        "memory.global_to_workgroup.u8.zero_extend",
        8,
        32,
    ),
    (
        "SBYTE",
        "sbyte",
        "memory.global_to_workgroup.i8.sign_extend",
        8,
        32,
    ),
    (
        "USHORT",
        "ushort",
        "memory.global_to_workgroup.u16.zero_extend",
        16,
        32,
    ),
    (
        "SSHORT",
        "sshort",
        "memory.global_to_workgroup.i16.sign_extend",
        16,
        32,
    ),
    ("DWORD", "dword", "memory.global_to_workgroup.u32", 32, 32),
)

_GLOBAL_LOAD_LDS_CDNA3_VARIANTS = _GLOBAL_LOAD_LDS_BASE_VARIANTS

_GLOBAL_LOAD_LDS_GFX950_VARIANTS = (
    *_GLOBAL_LOAD_LDS_BASE_VARIANTS,
    ("DWORDX3", "dwordx3", "memory.global_to_workgroup.u96", 96, 96),
    ("DWORDX4", "dwordx4", "memory.global_to_workgroup.u128", 128, 128),
)


def _global_load_lds_overlays(
    *,
    descriptor_key_suffix: str = "",
    address_units: int,
    saddr_off: AmdgpuFixedEncodingValue | None,
    cache_fields: tuple[tuple[str, int], ...] = (),
    variants: tuple[tuple[str, str, str, int, int], ...] = (
        _GLOBAL_LOAD_LDS_GFX950_VARIANTS
    ),
) -> tuple[AmdgpuDescriptorOverlay, ...]:
    return tuple(
        _global_load_lds_overlay(
            descriptor_key=(
                f"amdgpu.global_load_lds_{mnemonic_suffix}{descriptor_key_suffix}"
            ),
            instruction_name=f"GLOBAL_LOAD_LDS_{instruction_suffix}",
            mnemonic=f"global_load_lds_{mnemonic_suffix}",
            semantic_tag=semantic_tag,
            global_width_bits=global_width_bits,
            workgroup_width_bits=workgroup_width_bits,
            address_units=address_units,
            saddr_off=saddr_off,
            cache_fields=cache_fields,
        )
        for (
            instruction_suffix,
            mnemonic_suffix,
            semantic_tag,
            global_width_bits,
            workgroup_width_bits,
        ) in variants
    )


def _global_b16_memory_overlays(
    *,
    encoding_name: str,
    address_field_name: str,
    load_data_field_name: str,
    store_data_field_name: str,
    offset_field_name: str,
    offset_bit_width: int,
    saddr_off: AmdgpuFixedEncodingValue | None,
    address_units: int,
    descriptor_key_suffix: str = "",
    load_mnemonic_suffixes: tuple[str, str] = ("u16", "i16"),
    implicit_m0: bool = False,
    cache_fields: tuple[tuple[str, int], ...] = (),
) -> tuple[AmdgpuDescriptorOverlay, ...]:
    return (
        *_global_load_narrow_overlays(
            rows=_VMEM_NARROW_B16_LOAD_ROWS,
            mnemonic_suffixes=load_mnemonic_suffixes,
            encoding_name=encoding_name,
            address_field_name=address_field_name,
            data_field_name=load_data_field_name,
            offset_field_name=offset_field_name,
            offset_bit_width=offset_bit_width,
            saddr_off=saddr_off,
            address_units=address_units,
            descriptor_key_suffix=descriptor_key_suffix,
            implicit_m0=implicit_m0,
            cache_fields=cache_fields,
        ),
        _global_load_b16_d16_overlay(
            descriptor_key=f"amdgpu.global_load_b16_d16{descriptor_key_suffix}",
            encoding_name=encoding_name,
            address_field_name=address_field_name,
            data_field_name=load_data_field_name,
            offset_field_name=offset_field_name,
            offset_bit_width=offset_bit_width,
            saddr_off=saddr_off,
            address_units=address_units,
            implicit_m0=implicit_m0,
            cache_fields=cache_fields,
        ),
        _global_store_b16_overlay(
            descriptor_key=f"amdgpu.global_store_b16{descriptor_key_suffix}",
            encoding_name=encoding_name,
            address_field_name=address_field_name,
            data_field_name=store_data_field_name,
            offset_field_name=offset_field_name,
            offset_bit_width=offset_bit_width,
            saddr_off=saddr_off,
            address_units=address_units,
            implicit_m0=implicit_m0,
            cache_fields=cache_fields,
        ),
    )


def _global_byte_memory_overlays(
    *,
    encoding_name: str,
    address_field_name: str,
    load_data_field_name: str,
    store_data_field_name: str,
    offset_field_name: str,
    offset_bit_width: int,
    saddr_off: AmdgpuFixedEncodingValue | None,
    address_units: int,
    descriptor_key_suffix: str = "",
    load_mnemonic_suffixes: tuple[str, str] = ("u8", "i8"),
    implicit_m0: bool = False,
    cache_fields: tuple[tuple[str, int], ...] = (),
) -> tuple[AmdgpuDescriptorOverlay, ...]:
    return (
        *_global_load_narrow_overlays(
            rows=_VMEM_NARROW_BYTE_LOAD_ROWS,
            mnemonic_suffixes=load_mnemonic_suffixes,
            encoding_name=encoding_name,
            address_field_name=address_field_name,
            data_field_name=load_data_field_name,
            offset_field_name=offset_field_name,
            offset_bit_width=offset_bit_width,
            saddr_off=saddr_off,
            address_units=address_units,
            descriptor_key_suffix=descriptor_key_suffix,
            implicit_m0=implicit_m0,
            cache_fields=cache_fields,
        ),
        _global_store_overlay(
            descriptor_key=f"amdgpu.global_store_b8{descriptor_key_suffix}",
            instruction_name="GLOBAL_STORE_BYTE",
            mnemonic="global_store_b8",
            encoding_name=encoding_name,
            address_field_name=address_field_name,
            data_field_name=store_data_field_name,
            offset_field_name=offset_field_name,
            offset_bit_width=offset_bit_width,
            saddr_off=saddr_off,
            width_bits=8,
            units=1,
            address_units=address_units,
            implicit_m0=implicit_m0,
            cache_fields=cache_fields,
        ),
    )


def _global_memory_overlays(
    *,
    instruction_suffixes: tuple[str, ...],
    mnemonic_suffixes: tuple[str, ...],
    encoding_name: str,
    address_field_name: str,
    load_data_field_name: str,
    store_data_field_name: str,
    offset_field_name: str,
    offset_bit_width: int,
    saddr_off: AmdgpuFixedEncodingValue | None,
    address_units: int,
    descriptor_key_suffix: str = "",
    implicit_m0: bool = False,
    cache_fields: tuple[tuple[str, int], ...] = (),
) -> tuple[AmdgpuDescriptorOverlay, ...]:
    return (
        *(
            _global_load_overlay(
                descriptor_key=(
                    f"amdgpu.global_load_b{width_bits}{descriptor_key_suffix}"
                ),
                instruction_name=f"GLOBAL_LOAD_{instruction_suffix}",
                mnemonic=f"global_load_{mnemonic_suffix}",
                encoding_name=encoding_name,
                address_field_name=address_field_name,
                data_field_name=load_data_field_name,
                offset_field_name=offset_field_name,
                offset_bit_width=offset_bit_width,
                saddr_off=saddr_off,
                width_bits=width_bits,
                units=units,
                address_units=address_units,
                implicit_m0=implicit_m0,
                cache_fields=cache_fields,
            )
            for (width_bits, units), instruction_suffix, mnemonic_suffix in zip(
                _MEMORY_DWORD_VECTOR_WIDTHS,
                instruction_suffixes,
                mnemonic_suffixes,
                strict=True,
            )
        ),
        *(
            _global_store_overlay(
                descriptor_key=(
                    f"amdgpu.global_store_b{width_bits}{descriptor_key_suffix}"
                ),
                instruction_name=f"GLOBAL_STORE_{instruction_suffix}",
                mnemonic=f"global_store_{mnemonic_suffix}",
                encoding_name=encoding_name,
                address_field_name=address_field_name,
                data_field_name=store_data_field_name,
                offset_field_name=offset_field_name,
                offset_bit_width=offset_bit_width,
                saddr_off=saddr_off,
                width_bits=width_bits,
                units=units,
                address_units=address_units,
                implicit_m0=implicit_m0,
                cache_fields=cache_fields,
            )
            for (width_bits, units), instruction_suffix, mnemonic_suffix in zip(
                _MEMORY_DWORD_VECTOR_WIDTHS,
                instruction_suffixes,
                mnemonic_suffixes,
                strict=True,
            )
        ),
    )


__all__ = (
    "_BUFFER_LOAD_LDS_CDNA3_VARIANTS",
    "_BUFFER_LOAD_LDS_GFX950_VARIANTS",
    "_GLOBAL_LOAD_LDS_CDNA3_VARIANTS",
    "_GLOBAL_LOAD_LDS_GFX950_VARIANTS",
    "_MEMORY_DWORD_VECTOR_WIDTHS",
    "_SMEM_DWORDX4_WIDTHS",
    "_buffer_b16_memory_overlays",
    "_buffer_byte_memory_overlays",
    "_buffer_load_128_off_zero_overlay",
    "_buffer_load_128_overlay",
    "_buffer_load_128_vaddr_offset_overlay",
    "_buffer_load_96_off_zero_overlay",
    "_buffer_load_96_overlay",
    "_buffer_load_96_vaddr_offset_overlay",
    "_buffer_load_64_off_zero_overlay",
    "_buffer_load_64_overlay",
    "_buffer_load_64_vaddr_offset_overlay",
    "_buffer_load_b16_d16_overlay",
    "_buffer_load_b16_d16_vaddr_offset_overlay",
    "_buffer_load_dword_off_zero_overlay",
    "_buffer_load_dword_overlay",
    "_buffer_load_dword_vaddr_offset_overlay",
    "_buffer_load_lds_off_zero_overlay",
    "_buffer_load_lds_overlay",
    "_buffer_load_lds_overlays",
    "_buffer_load_lds_vaddr_offset_overlay",
    "_buffer_load_off_zero_overlay",
    "_buffer_load_vaddr_offset_overlay",
    "_buffer_store_128_off_zero_overlay",
    "_buffer_store_128_overlay",
    "_buffer_store_128_vaddr_offset_overlay",
    "_buffer_store_96_off_zero_overlay",
    "_buffer_store_96_overlay",
    "_buffer_store_96_vaddr_offset_overlay",
    "_buffer_store_64_off_zero_overlay",
    "_buffer_store_64_overlay",
    "_buffer_store_64_vaddr_offset_overlay",
    "_buffer_store_b16_overlay",
    "_buffer_store_b16_vaddr_offset_overlay",
    "_buffer_store_b8_overlay",
    "_buffer_store_b8_vaddr_offset_overlay",
    "_buffer_store_dword_off_zero_overlay",
    "_buffer_store_dword_overlay",
    "_buffer_store_dword_vaddr_offset_overlay",
    "_buffer_store_off_zero_overlay",
    "_buffer_store_vaddr_offset_overlay",
    "_global_b16_memory_overlays",
    "_global_byte_memory_overlays",
    "_global_load_b16_d16_overlay",
    "_global_load_lds_overlay",
    "_global_load_lds_overlays",
    "_global_load_overlay",
    "_global_memory_overlays",
    "_global_store_b16_overlay",
    "_global_store_overlay",
    "_s_buffer_load_128_overlay",
    "_s_buffer_load_256_overlay",
    "_s_buffer_load_512_overlay",
    "_s_buffer_load_64_overlay",
    "_s_buffer_load_96_overlay",
    "_s_buffer_load_dword_overlay",
    "_s_buffer_load_narrow_overlays",
    "_s_buffer_store_dword_width_overlays",
    "_s_load_96_overlay",
    "_s_load_dword_overlay",
    "_s_load_dwordx16_overlay",
    "_s_load_dwordx2_overlay",
    "_s_load_dwordx4_overlay",
    "_s_load_dwordx8_overlay",
    "_s_load_narrow_overlays",
    "_s_scratch_load_dword_width_overlays",
    "_s_scratch_store_dword_width_overlays",
    "_s_store_dword_width_overlays",
    "_scratch_memory_overlays",
)
