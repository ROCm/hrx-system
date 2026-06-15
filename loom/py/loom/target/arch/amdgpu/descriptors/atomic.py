# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


# ruff: noqa: F403, F405

"""Global, flat, and buffer atomic descriptor overlays."""

from __future__ import annotations

from .common import *


def _flat_atomic_asm(
    *,
    mnemonic: str,
    returns_old_value: bool,
    implicit_m0: bool,
    cache_fields: tuple[tuple[str, int], ...],
) -> tuple[AsmForm, ...]:
    operands: tuple[str, ...] = ("addr", "value")
    if implicit_m0:
        operands = (*operands, "m0")
    return _asm(
        mnemonic=f"{mnemonic}_rtn" if returns_old_value else mnemonic,
        native_assembly_mnemonic=mnemonic if returns_old_value else None,
        results=("dst",) if returns_old_value else (),
        operands=operands,
        immediates=_memory_asm_immediate_names(cache_fields),
        named_immediates=True,
    )


def _buffer_atomic_asm(
    *,
    mnemonic: str,
    returns_old_value: bool,
    cache_fields: tuple[tuple[str, int], ...],
) -> tuple[AsmForm, ...]:
    return _asm(
        mnemonic=f"{mnemonic}_rtn" if returns_old_value else mnemonic,
        native_assembly_mnemonic=mnemonic if returns_old_value else None,
        results=("dst",) if returns_old_value else (),
        operands=("value", "resource", "vaddr", "soffset"),
        immediates=_memory_asm_immediate_names(cache_fields),
        named_immediates=True,
    )


def _global_atomic_overlay(
    *,
    descriptor_key: str,
    instruction_name: str,
    mnemonic: str,
    semantic_tag: str,
    data_format_name: str,
    returns_old_value: bool,
    encoding_name: str,
    address_field_name: str,
    data_field_name: str,
    offset_field_name: str,
    offset_bit_width: int,
    return_field_name: str,
    return_field_value: int,
    cache_fields: tuple[tuple[str, int], ...],
    cache_immediate_field_names: tuple[str, ...],
    saddr_off: AmdgpuFixedEncodingValue | None,
    address_units: int,
    implicit_m0: bool,
) -> AmdgpuDescriptorOverlay:
    schedule_class = (
        _SCHEDULE_VMEM_ATOMIC_RETURN
        if returns_old_value
        else _SCHEDULE_VMEM_ATOMIC_NO_RETURN
    )
    counter_id = _COUNTER_VMEM_LOAD if returns_old_value else _COUNTER_VMEM_STORE
    ignored_operands: tuple[AmdgpuIgnoredOperandOverlay, ...] = ()
    if returns_old_value:
        operands: tuple[AmdgpuOperandOverlay, ...] = (
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            _global_addr_operand(
                address_field_name, units=address_units, has_saddr=saddr_off is None
            ),
            AmdgpuOperandOverlay(data_field_name, _vgpr_operand("value")),
        )
    else:
        operands = (
            _global_addr_operand(
                address_field_name, units=address_units, has_saddr=saddr_off is None
            ),
            AmdgpuOperandOverlay(data_field_name, _vgpr_operand("value")),
        )
        ignored_operands = (
            AmdgpuIgnoredOperandOverlay(
                "VDST",
                ignore_reason="no-return-global-atomic-has-no-vgpr-result",
                fixed_encoding_value=_predefined("v0", "OPR_VGPR"),
            ),
        )

    cache_immediate_fields = tuple(
        (field_name, bit_width)
        for field_name, bit_width in cache_fields
        if field_name in cache_immediate_field_names
    )
    cache_immediate_defaults = {
        return_field_name: return_field_value if returns_old_value else 0
    }
    fixed_encoding_fields: tuple[tuple[str, AmdgpuFixedEncodingValue], ...] = tuple(
        (
            field_name,
            return_field_value
            if returns_old_value and field_name == return_field_name
            else 0,
        )
        for field_name, _bit_width in cache_fields
        if field_name not in cache_immediate_field_names
    )
    if saddr_off is None:
        operands += (AmdgpuOperandOverlay("SADDR", _sgpr_operand("saddr", units=2)),)
    else:
        fixed_encoding_fields += (("SADDR", saddr_off),)
    implicit_operands: tuple[AmdgpuImplicitOperandOverlay, ...] = (
        _ignore_global_atomic_memory(data_format_name=data_format_name, is_input=False),
        _ignore_global_atomic_memory(data_format_name=data_format_name, is_input=True),
    )
    if implicit_m0:
        implicit_operands += (_implicit_m0_input(),)
    return AmdgpuDescriptorOverlay(
        descriptor_key=descriptor_key,
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        encoding_name=encoding_name,
        semantic_tag=semantic_tag,
        schedule_class=schedule_class,
        operands=operands,
        ignored_operands=ignored_operands,
        implicit_operands=implicit_operands,
        immediate_fields=(
            offset_field_name,
            *_cache_field_names(cache_immediate_fields),
        ),
        immediates=(
            _signed_offset_immediate(offset_bit_width),
            *_cache_immediates_with_defaults(
                cache_immediate_fields, cache_immediate_defaults
            ),
        ),
        fixed_encoding_fields=fixed_encoding_fields,
        effects=_global_atomic_effects(32, counter_id=counter_id),
        flags=(DescriptorFlag.SIDE_EFFECTING,),
        asm_forms=_global_saddr_asm(
            mnemonic=f"{mnemonic}_rtn" if returns_old_value else mnemonic,
            results=("dst",) if returns_old_value else (),
            operands=("addr", "value", "saddr"),
            implicit_m0=implicit_m0,
            immediates=_memory_asm_immediate_names(cache_immediate_fields),
        )
        if saddr_off is None
        else _global_vaddr_asm(
            mnemonic=f"{mnemonic}_rtn" if returns_old_value else mnemonic,
            results=("dst",) if returns_old_value else (),
            operands=("addr", "value"),
            immediates=_memory_asm_immediate_names(cache_immediate_fields),
        ),
    )


def _global_atomic_cmpswap_overlay(
    *,
    encoding_name: str,
    address_field_name: str,
    data_field_name: str,
    offset_field_name: str,
    offset_bit_width: int,
    return_field_name: str,
    return_field_value: int,
    cache_fields: tuple[tuple[str, int], ...],
    cache_immediate_field_names: tuple[str, ...],
    saddr_off: AmdgpuFixedEncodingValue | None,
    address_units: int,
    descriptor_key_suffix: str,
    implicit_m0: bool,
) -> AmdgpuDescriptorOverlay:
    cache_immediate_fields = tuple(
        (field_name, bit_width)
        for field_name, bit_width in cache_fields
        if field_name in cache_immediate_field_names
    )
    cache_immediate_defaults = {return_field_name: return_field_value}
    fixed_encoding_fields: tuple[tuple[str, AmdgpuFixedEncodingValue], ...] = tuple(
        (
            field_name,
            return_field_value if field_name == return_field_name else 0,
        )
        for field_name, _bit_width in cache_fields
        if field_name not in cache_immediate_field_names
    )
    operands: tuple[AmdgpuOperandOverlay, ...] = (
        AmdgpuOperandOverlay("VDST", _vgpr_result()),
        _global_addr_operand(
            address_field_name, units=address_units, has_saddr=saddr_off is None
        ),
        AmdgpuOperandOverlay(data_field_name, _vgpr_operand("value", units=2)),
    )
    if saddr_off is None:
        operands += (AmdgpuOperandOverlay("SADDR", _sgpr_operand("saddr", units=2)),)
    else:
        fixed_encoding_fields += (("SADDR", saddr_off),)
    implicit_operands: tuple[AmdgpuImplicitOperandOverlay, ...] = (
        _ignore_global_atomic_memory(data_format_name="FMT_NUM_U32", is_input=False),
        _ignore_global_atomic_memory(data_format_name="FMT_NUM_U32", is_input=True),
    )
    if implicit_m0:
        implicit_operands += (_implicit_m0_input(),)
    return AmdgpuDescriptorOverlay(
        descriptor_key=(f"amdgpu.global_atomic_cmpswap_b32_rtn{descriptor_key_suffix}"),
        instruction_name="GLOBAL_ATOMIC_CMPSWAP",
        mnemonic="global_atomic_cmpswap_b32",
        encoding_name=encoding_name,
        semantic_tag="memory.global.atomic.compare_exchange.b32.return",
        schedule_class=_SCHEDULE_VMEM_ATOMIC_RETURN,
        operands=operands,
        implicit_operands=implicit_operands,
        immediate_fields=(
            offset_field_name,
            *_cache_field_names(cache_immediate_fields),
        ),
        immediates=(
            _signed_offset_immediate(offset_bit_width),
            *_cache_immediates_with_defaults(
                cache_immediate_fields, cache_immediate_defaults
            ),
        ),
        fixed_encoding_fields=fixed_encoding_fields,
        effects=_global_atomic_effects(32, counter_id=_COUNTER_VMEM_LOAD),
        flags=(DescriptorFlag.SIDE_EFFECTING,),
        asm_forms=_global_saddr_asm(
            mnemonic="global_atomic_cmpswap_b32_rtn",
            results=("dst",),
            operands=("addr", "value", "saddr"),
            implicit_m0=implicit_m0,
            immediates=_memory_asm_immediate_names(cache_immediate_fields),
        )
        if saddr_off is None
        else _global_vaddr_asm(
            mnemonic="global_atomic_cmpswap_b32_rtn",
            results=("dst",),
            operands=("addr", "value"),
            immediates=_memory_asm_immediate_names(cache_immediate_fields),
        ),
    )


_GLOBAL_ATOMIC_DEFAULT_ROWS = (
    ("add_u32", "GLOBAL_ATOMIC_ADD_U32", "add.u32", "FMT_NUM_U32", True),
    ("sub_u32", "GLOBAL_ATOMIC_SUB_U32", "sub.u32", "FMT_NUM_U32", True),
    ("min_i32", "GLOBAL_ATOMIC_MIN_I32", "min.i32", "FMT_NUM_I32", True),
    ("max_i32", "GLOBAL_ATOMIC_MAX_I32", "max.i32", "FMT_NUM_I32", True),
    ("min_u32", "GLOBAL_ATOMIC_MIN_U32", "min.u32", "FMT_NUM_U32", True),
    ("max_u32", "GLOBAL_ATOMIC_MAX_U32", "max.u32", "FMT_NUM_U32", True),
    ("and_b32", "GLOBAL_ATOMIC_AND_B32", "and.b32", "FMT_NUM_B32", True),
    ("or_b32", "GLOBAL_ATOMIC_OR_B32", "or.b32", "FMT_NUM_B32", True),
    ("xor_b32", "GLOBAL_ATOMIC_XOR_B32", "xor.b32", "FMT_NUM_B32", True),
    (
        "swap_b32",
        "GLOBAL_ATOMIC_SWAP_B32",
        "exchange.b32",
        "FMT_NUM_B32",
        False,
    ),
    ("add_f32", "GLOBAL_ATOMIC_ADD_F32", "add.f32", "FMT_NUM_F32", True),
    ("min_f32", "GLOBAL_ATOMIC_MIN_F32", "minnum.f32", "FMT_NUM_F32", True),
    ("max_f32", "GLOBAL_ATOMIC_MAX_F32", "maxnum.f32", "FMT_NUM_F32", True),
)

_GLOBAL_ATOMIC_GFX940_ROWS = (
    ("add_f32", "GLOBAL_ATOMIC_ADD_F32", "add.f32", "FMT_NUM_F32", True),
)


def _global_atomic_overlays(
    *,
    encoding_name: str,
    address_field_name: str,
    data_field_name: str,
    offset_field_name: str,
    offset_bit_width: int,
    return_field_name: str,
    return_field_value: int,
    cache_fields: tuple[tuple[str, int], ...],
    cache_immediate_field_names: tuple[str, ...] = (),
    saddr_off: AmdgpuFixedEncodingValue | None,
    address_units: int,
    descriptor_key_suffix: str = "",
    include_packed_half_add: bool = False,
    implicit_m0: bool = False,
    rows: tuple[tuple[str, str, str, str, bool], ...] = _GLOBAL_ATOMIC_DEFAULT_ROWS,
) -> tuple[AmdgpuDescriptorOverlay, ...]:
    effective_rows = list(rows)
    if include_packed_half_add:
        effective_rows.extend(
            (
                (
                    "pk_add_f16",
                    "GLOBAL_ATOMIC_PK_ADD_F16",
                    "add.pk2.f16",
                    "FMT_NUM_PK2_F16",
                    True,
                ),
                (
                    "pk_add_bf16",
                    "GLOBAL_ATOMIC_PK_ADD_BF16",
                    "add.pk2.bf16",
                    "FMT_NUM_PK2_BF16",
                    True,
                ),
            )
        )
    overlays: list[AmdgpuDescriptorOverlay] = []
    for (
        mnemonic_suffix,
        instruction_name,
        semantic_suffix,
        data_format_name,
        has_no_return_form,
    ) in effective_rows:
        if has_no_return_form:
            overlays.append(
                _global_atomic_overlay(
                    descriptor_key=(
                        f"amdgpu.global_atomic_{mnemonic_suffix}{descriptor_key_suffix}"
                    ),
                    instruction_name=instruction_name,
                    mnemonic=f"global_atomic_{mnemonic_suffix}",
                    semantic_tag=f"memory.global.atomic.{semantic_suffix}",
                    data_format_name=data_format_name,
                    returns_old_value=False,
                    encoding_name=encoding_name,
                    address_field_name=address_field_name,
                    data_field_name=data_field_name,
                    offset_field_name=offset_field_name,
                    offset_bit_width=offset_bit_width,
                    return_field_name=return_field_name,
                    return_field_value=return_field_value,
                    cache_fields=cache_fields,
                    cache_immediate_field_names=cache_immediate_field_names,
                    saddr_off=saddr_off,
                    address_units=address_units,
                    implicit_m0=implicit_m0,
                )
            )
        overlays.append(
            _global_atomic_overlay(
                descriptor_key=(
                    f"amdgpu.global_atomic_{mnemonic_suffix}_rtn{descriptor_key_suffix}"
                ),
                instruction_name=instruction_name,
                mnemonic=f"global_atomic_{mnemonic_suffix}",
                semantic_tag=f"memory.global.atomic.{semantic_suffix}.return",
                data_format_name=data_format_name,
                returns_old_value=True,
                encoding_name=encoding_name,
                address_field_name=address_field_name,
                data_field_name=data_field_name,
                offset_field_name=offset_field_name,
                offset_bit_width=offset_bit_width,
                return_field_name=return_field_name,
                return_field_value=return_field_value,
                cache_fields=cache_fields,
                cache_immediate_field_names=cache_immediate_field_names,
                saddr_off=saddr_off,
                address_units=address_units,
                implicit_m0=implicit_m0,
            )
        )
    overlays.append(
        _global_atomic_cmpswap_overlay(
            encoding_name=encoding_name,
            address_field_name=address_field_name,
            data_field_name=data_field_name,
            offset_field_name=offset_field_name,
            offset_bit_width=offset_bit_width,
            return_field_name=return_field_name,
            return_field_value=return_field_value,
            cache_fields=cache_fields,
            cache_immediate_field_names=cache_immediate_field_names,
            saddr_off=saddr_off,
            address_units=address_units,
            descriptor_key_suffix=descriptor_key_suffix,
            implicit_m0=implicit_m0,
        )
    )
    return tuple(overlays)


_FLAT_ATOMIC_GFX11_ROWS = (
    ("add_u32", "add_u32", "FLAT_ATOMIC_ADD_U32", "add.u32", "FMT_NUM_U32", True),
    ("sub_u32", "sub_u32", "FLAT_ATOMIC_SUB_U32", "sub.u32", "FMT_NUM_U32", True),
    ("min_i32", "min_i32", "FLAT_ATOMIC_MIN_I32", "min.i32", "FMT_NUM_I32", True),
    ("max_i32", "max_i32", "FLAT_ATOMIC_MAX_I32", "max.i32", "FMT_NUM_I32", True),
    ("min_u32", "min_u32", "FLAT_ATOMIC_MIN_U32", "min.u32", "FMT_NUM_U32", True),
    ("max_u32", "max_u32", "FLAT_ATOMIC_MAX_U32", "max.u32", "FMT_NUM_U32", True),
    ("and_b32", "and_b32", "FLAT_ATOMIC_AND_B32", "and.b32", "FMT_NUM_B32", True),
    ("or_b32", "or_b32", "FLAT_ATOMIC_OR_B32", "or.b32", "FMT_NUM_B32", True),
    ("xor_b32", "xor_b32", "FLAT_ATOMIC_XOR_B32", "xor.b32", "FMT_NUM_B32", True),
    (
        "swap_b32",
        "swap_b32",
        "FLAT_ATOMIC_SWAP_B32",
        "exchange.b32",
        "FMT_NUM_B32",
        False,
    ),
    ("add_f32", "add_f32", "FLAT_ATOMIC_ADD_F32", "add.f32", "FMT_NUM_F32", True),
    (
        "min_f32",
        "min_f32",
        "FLAT_ATOMIC_MIN_F32",
        "minnum.f32",
        "FMT_NUM_F32",
        True,
    ),
    (
        "max_f32",
        "max_f32",
        "FLAT_ATOMIC_MAX_F32",
        "maxnum.f32",
        "FMT_NUM_F32",
        True,
    ),
)

_FLAT_ATOMIC_GFX12_ROWS = (
    *_FLAT_ATOMIC_GFX11_ROWS[:-2],
    (
        "pk_add_f16",
        "pk_add_f16",
        "FLAT_ATOMIC_PK_ADD_F16",
        "add.pk2.f16",
        "FMT_NUM_PK2_F16",
        True,
    ),
    (
        "pk_add_bf16",
        "pk_add_bf16",
        "FLAT_ATOMIC_PK_ADD_BF16",
        "add.pk2.bf16",
        "FMT_NUM_PK2_BF16",
        True,
    ),
    (
        "min_f32",
        "min_f32",
        "FLAT_ATOMIC_MIN_NUM_F32",
        "minnum.f32",
        "FMT_NUM_F32",
        True,
    ),
    (
        "max_f32",
        "max_f32",
        "FLAT_ATOMIC_MAX_NUM_F32",
        "maxnum.f32",
        "FMT_NUM_F32",
        True,
    ),
)

_FLAT_ATOMIC_GFX950_ROWS = (
    ("add_u32", "add", "FLAT_ATOMIC_ADD", "add.u32", "FMT_NUM_U32", True),
    ("sub_u32", "sub", "FLAT_ATOMIC_SUB", "sub.u32", "FMT_NUM_U32", True),
    ("min_i32", "smin", "FLAT_ATOMIC_SMIN", "min.i32", "FMT_NUM_I32", True),
    ("max_i32", "smax", "FLAT_ATOMIC_SMAX", "max.i32", "FMT_NUM_I32", True),
    ("min_u32", "umin", "FLAT_ATOMIC_UMIN", "min.u32", "FMT_NUM_U32", True),
    ("max_u32", "umax", "FLAT_ATOMIC_UMAX", "max.u32", "FMT_NUM_U32", True),
    ("and_b32", "and", "FLAT_ATOMIC_AND", "and.b32", "FMT_NUM_B32", True),
    ("or_b32", "or", "FLAT_ATOMIC_OR", "or.b32", "FMT_NUM_B32", True),
    ("xor_b32", "xor", "FLAT_ATOMIC_XOR", "xor.b32", "FMT_NUM_B32", True),
    ("swap_b32", "swap", "FLAT_ATOMIC_SWAP", "exchange.b32", "FMT_NUM_B32", False),
    ("add_f32", "add_f32", "FLAT_ATOMIC_ADD_F32", "add.f32", "FMT_NUM_F32", True),
    (
        "pk_add_f16",
        "pk_add_f16",
        "FLAT_ATOMIC_PK_ADD_F16",
        "add.pk2.f16",
        "FMT_NUM_PK2_F16",
        True,
    ),
    (
        "pk_add_bf16",
        "pk_add_bf16",
        "FLAT_ATOMIC_PK_ADD_BF16",
        "add.pk2.bf16",
        "FMT_NUM_PK2_BF16",
        True,
    ),
)


def _flat_atomic_overlay(
    *,
    descriptor_key: str,
    instruction_name: str,
    mnemonic: str,
    semantic_tag: str,
    data_format_name: str,
    returns_old_value: bool,
    encoding_name: str,
    address_field_name: str,
    data_field_name: str,
    offset_field_name: str,
    offset_bit_width: int,
    offset_signed: bool,
    return_field_name: str,
    return_field_value: int,
    cache_fields: tuple[tuple[str, int], ...],
    cache_immediate_field_names: tuple[str, ...],
    implicit_flat_scratch: bool,
    implicit_m0: bool,
    allow_accumulator_operands: bool,
) -> AmdgpuDescriptorOverlay:
    schedule_class = (
        _SCHEDULE_VMEM_ATOMIC_RETURN
        if returns_old_value
        else _SCHEDULE_VMEM_ATOMIC_NO_RETURN
    )
    counter_id = _COUNTER_VMEM_LOAD if returns_old_value else _COUNTER_VMEM_STORE
    result_operand = (
        _vgpr_agpr_result() if allow_accumulator_operands else _vgpr_result()
    )
    data_operand = (
        _vgpr_agpr_operand("value")
        if allow_accumulator_operands
        else _vgpr_operand("value")
    )
    ignored_operands: tuple[AmdgpuIgnoredOperandOverlay, ...] = ()
    if returns_old_value:
        operands: tuple[AmdgpuOperandOverlay, ...] = (
            AmdgpuOperandOverlay("VDST", result_operand),
            AmdgpuOperandOverlay(address_field_name, _vgpr_operand("addr", units=2)),
            AmdgpuOperandOverlay(data_field_name, data_operand),
        )
    else:
        operands = (
            AmdgpuOperandOverlay(address_field_name, _vgpr_operand("addr", units=2)),
            AmdgpuOperandOverlay(data_field_name, data_operand),
        )
        ignored_operands = (
            AmdgpuIgnoredOperandOverlay(
                "VDST",
                ignore_reason="no-return-flat-atomic-has-no-vgpr-result",
                fixed_encoding_value=_predefined("v0", "OPR_VGPR"),
            ),
        )

    cache_immediate_fields = tuple(
        (field_name, bit_width)
        for field_name, bit_width in cache_fields
        if field_name in cache_immediate_field_names
    )
    cache_immediate_defaults = {
        return_field_name: return_field_value if returns_old_value else 0
    }
    fixed_encoding_fields: tuple[tuple[str, AmdgpuFixedEncodingValue], ...] = tuple(
        (
            field_name,
            return_field_value
            if returns_old_value and field_name == return_field_name
            else 0,
        )
        for field_name, _bit_width in cache_fields
        if field_name not in cache_immediate_field_names
    )
    implicit_operands: tuple[AmdgpuImplicitOperandOverlay, ...] = (
        _ignore_generic_atomic_memory(
            data_format_name=data_format_name, is_input=False
        ),
    )
    if implicit_flat_scratch:
        implicit_operands += (_IGNORE_FLAT_SCRATCH_INPUT,)
    implicit_operands += (
        _ignore_generic_atomic_memory(data_format_name=data_format_name, is_input=True),
    )
    if implicit_m0:
        implicit_operands += (_implicit_m0_input(),)
    offset_immediate = (
        _signed_offset_immediate(offset_bit_width)
        if offset_signed
        else _offset_immediate(offset_bit_width)
    )
    return AmdgpuDescriptorOverlay(
        descriptor_key=descriptor_key,
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        encoding_name=encoding_name,
        semantic_tag=semantic_tag,
        schedule_class=schedule_class,
        operands=operands,
        ignored_operands=ignored_operands,
        implicit_operands=implicit_operands,
        immediate_fields=(
            offset_field_name,
            *_cache_field_names(cache_immediate_fields),
        ),
        immediates=(
            offset_immediate,
            *_cache_immediates_with_defaults(
                cache_immediate_fields, cache_immediate_defaults
            ),
        ),
        fixed_encoding_fields=fixed_encoding_fields,
        effects=_generic_atomic_effects(32, counter_id=counter_id),
        flags=(DescriptorFlag.SIDE_EFFECTING,),
        asm_forms=_flat_atomic_asm(
            mnemonic=mnemonic,
            returns_old_value=returns_old_value,
            implicit_m0=implicit_m0,
            cache_fields=cache_immediate_fields,
        ),
    )


def _flat_atomic_cmpswap_overlay(
    *,
    instruction_name: str,
    encoding_name: str,
    address_field_name: str,
    data_field_name: str,
    offset_field_name: str,
    offset_bit_width: int,
    offset_signed: bool,
    return_field_name: str,
    return_field_value: int,
    cache_fields: tuple[tuple[str, int], ...],
    cache_immediate_field_names: tuple[str, ...],
    implicit_flat_scratch: bool,
    implicit_m0: bool,
    allow_accumulator_operands: bool,
) -> AmdgpuDescriptorOverlay:
    cache_immediate_fields = tuple(
        (field_name, bit_width)
        for field_name, bit_width in cache_fields
        if field_name in cache_immediate_field_names
    )
    cache_immediate_defaults = {return_field_name: return_field_value}
    fixed_encoding_fields: tuple[tuple[str, AmdgpuFixedEncodingValue], ...] = tuple(
        (
            field_name,
            return_field_value if field_name == return_field_name else 0,
        )
        for field_name, _bit_width in cache_fields
        if field_name not in cache_immediate_field_names
    )
    result_operand = (
        _vgpr_agpr_result() if allow_accumulator_operands else _vgpr_result()
    )
    data_operand = (
        _vgpr_agpr_operand("value", units=2)
        if allow_accumulator_operands
        else _vgpr_operand("value", units=2)
    )
    implicit_operands: tuple[AmdgpuImplicitOperandOverlay, ...] = (
        _ignore_generic_atomic_memory(data_format_name="FMT_NUM_U32", is_input=False),
    )
    if implicit_flat_scratch:
        implicit_operands += (_IGNORE_FLAT_SCRATCH_INPUT,)
    implicit_operands += (
        _ignore_generic_atomic_memory(data_format_name="FMT_NUM_U32", is_input=True),
    )
    if implicit_m0:
        implicit_operands += (_implicit_m0_input(),)
    offset_immediate = (
        _signed_offset_immediate(offset_bit_width)
        if offset_signed
        else _offset_immediate(offset_bit_width)
    )
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.flat_atomic_cmpswap_b32_rtn",
        instruction_name=instruction_name,
        mnemonic="flat_atomic_cmpswap_b32",
        encoding_name=encoding_name,
        semantic_tag="memory.generic.atomic.compare_exchange.b32.return",
        schedule_class=_SCHEDULE_VMEM_ATOMIC_RETURN,
        operands=(
            AmdgpuOperandOverlay("VDST", result_operand),
            AmdgpuOperandOverlay(address_field_name, _vgpr_operand("addr", units=2)),
            AmdgpuOperandOverlay(data_field_name, data_operand),
        ),
        implicit_operands=implicit_operands,
        immediate_fields=(
            offset_field_name,
            *_cache_field_names(cache_immediate_fields),
        ),
        immediates=(
            offset_immediate,
            *_cache_immediates_with_defaults(
                cache_immediate_fields, cache_immediate_defaults
            ),
        ),
        fixed_encoding_fields=fixed_encoding_fields,
        effects=_generic_atomic_effects(32, counter_id=_COUNTER_VMEM_LOAD),
        flags=(DescriptorFlag.SIDE_EFFECTING,),
        asm_forms=_flat_atomic_asm(
            mnemonic="flat_atomic_cmpswap_b32",
            returns_old_value=True,
            implicit_m0=implicit_m0,
            cache_fields=cache_immediate_fields,
        ),
    )


def _flat_atomic_overlays(
    *,
    rows: tuple[tuple[str, str, str, str, str, bool], ...],
    cmpswap_instruction_name: str,
    encoding_name: str,
    address_field_name: str,
    data_field_name: str,
    offset_field_name: str,
    offset_bit_width: int,
    offset_signed: bool,
    return_field_name: str,
    return_field_value: int,
    cache_fields: tuple[tuple[str, int], ...],
    cache_immediate_field_names: tuple[str, ...] = (),
    implicit_flat_scratch: bool,
    implicit_m0: bool = False,
    allow_accumulator_operands: bool = False,
) -> tuple[AmdgpuDescriptorOverlay, ...]:
    overlays: list[AmdgpuDescriptorOverlay] = []
    for (
        descriptor_suffix,
        mnemonic_suffix,
        instruction_name,
        semantic_suffix,
        data_format_name,
        has_no_return_form,
    ) in rows:
        if has_no_return_form:
            overlays.append(
                _flat_atomic_overlay(
                    descriptor_key=f"amdgpu.flat_atomic_{descriptor_suffix}",
                    instruction_name=instruction_name,
                    mnemonic=f"flat_atomic_{mnemonic_suffix}",
                    semantic_tag=f"memory.generic.atomic.{semantic_suffix}",
                    data_format_name=data_format_name,
                    returns_old_value=False,
                    encoding_name=encoding_name,
                    address_field_name=address_field_name,
                    data_field_name=data_field_name,
                    offset_field_name=offset_field_name,
                    offset_bit_width=offset_bit_width,
                    offset_signed=offset_signed,
                    return_field_name=return_field_name,
                    return_field_value=return_field_value,
                    cache_fields=cache_fields,
                    cache_immediate_field_names=cache_immediate_field_names,
                    implicit_flat_scratch=implicit_flat_scratch,
                    implicit_m0=implicit_m0,
                    allow_accumulator_operands=allow_accumulator_operands,
                )
            )
        overlays.append(
            _flat_atomic_overlay(
                descriptor_key=f"amdgpu.flat_atomic_{descriptor_suffix}_rtn",
                instruction_name=instruction_name,
                mnemonic=f"flat_atomic_{mnemonic_suffix}",
                semantic_tag=f"memory.generic.atomic.{semantic_suffix}.return",
                data_format_name=data_format_name,
                returns_old_value=True,
                encoding_name=encoding_name,
                address_field_name=address_field_name,
                data_field_name=data_field_name,
                offset_field_name=offset_field_name,
                offset_bit_width=offset_bit_width,
                offset_signed=offset_signed,
                return_field_name=return_field_name,
                return_field_value=return_field_value,
                cache_fields=cache_fields,
                cache_immediate_field_names=cache_immediate_field_names,
                implicit_flat_scratch=implicit_flat_scratch,
                implicit_m0=implicit_m0,
                allow_accumulator_operands=allow_accumulator_operands,
            )
        )
    overlays.append(
        _flat_atomic_cmpswap_overlay(
            instruction_name=cmpswap_instruction_name,
            encoding_name=encoding_name,
            address_field_name=address_field_name,
            data_field_name=data_field_name,
            offset_field_name=offset_field_name,
            offset_bit_width=offset_bit_width,
            offset_signed=offset_signed,
            return_field_name=return_field_name,
            return_field_value=return_field_value,
            cache_fields=cache_fields,
            cache_immediate_field_names=cache_immediate_field_names,
            implicit_flat_scratch=implicit_flat_scratch,
            implicit_m0=implicit_m0,
            allow_accumulator_operands=allow_accumulator_operands,
        )
    )
    return tuple(overlays)


def _buffer_atomic_overlay(
    *,
    descriptor_key: str,
    instruction_name: str,
    mnemonic: str,
    semantic_tag: str,
    data_format_name: str,
    returns_old_value: bool,
    encoding_name: str,
    resource_field_name: str,
    offset_field_name: str,
    offset_bit_width: int,
    return_field_name: str,
    return_field_value: int,
    cache_fields: tuple[tuple[str, int], ...],
    cache_immediate_field_names: tuple[str, ...],
) -> AmdgpuDescriptorOverlay:
    schedule_class = (
        _SCHEDULE_VMEM_ATOMIC_RETURN
        if returns_old_value
        else _SCHEDULE_VMEM_ATOMIC_NO_RETURN
    )
    counter_id = _COUNTER_VMEM_LOAD if returns_old_value else _COUNTER_VMEM_STORE
    constraints: tuple[Constraint, ...]
    if returns_old_value:
        operands: tuple[AmdgpuOperandOverlay, ...] = (
            AmdgpuOperandOverlay("VDATA", _vgpr_result()),
            AmdgpuOperandOverlay(
                "VDATA",
                _vgpr_operand("value"),
                role_exception_reason=_BUFFER_ATOMIC_VDATA_INPUT_REASON,
            ),
        )
        constraints = _DESTRUCTIVE_BUFFER_ATOMIC_CONSTRAINTS
    else:
        operands = (
            AmdgpuOperandOverlay(
                "VDATA",
                _vgpr_operand("value"),
                role_exception_reason=_BUFFER_ATOMIC_VDATA_INPUT_REASON,
            ),
        )
        constraints = ()

    operands += (
        AmdgpuOperandOverlay(resource_field_name, _sgpr_resource("resource", units=4)),
        _mubuf_vaddr_operand(),
        AmdgpuOperandOverlay("SOFFSET", _sgpr_operand("soffset")),
    )
    cache_immediate_fields = tuple(
        (field_name, bit_width)
        for field_name, bit_width in cache_fields
        if field_name in cache_immediate_field_names
    )
    cache_immediate_defaults = {
        return_field_name: return_field_value if returns_old_value else 0
    }
    fixed_encoding_fields = tuple(
        (
            field_name,
            return_field_value
            if returns_old_value and field_name == return_field_name
            else 0,
        )
        for field_name, _bit_width in cache_fields
        if field_name not in cache_immediate_field_names
    )
    return AmdgpuDescriptorOverlay(
        descriptor_key=descriptor_key,
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        encoding_name=encoding_name,
        semantic_tag=semantic_tag,
        schedule_class=schedule_class,
        operands=operands,
        implicit_operands=(
            _ignore_global_atomic_memory(
                data_format_name=data_format_name, is_input=False
            ),
            _ignore_global_atomic_memory(
                data_format_name=data_format_name, is_input=True
            ),
        ),
        immediate_fields=(
            offset_field_name,
            *_cache_field_names(cache_immediate_fields),
        ),
        immediates=(
            _offset_immediate(offset_bit_width),
            *_cache_immediates_with_defaults(
                cache_immediate_fields, cache_immediate_defaults
            ),
        ),
        fixed_encoding_fields=(("IDXEN", 0), ("OFFEN", 1), *fixed_encoding_fields),
        effects=_global_atomic_effects(32, counter_id=counter_id),
        constraints=constraints,
        flags=(DescriptorFlag.SIDE_EFFECTING,),
        asm_forms=_buffer_atomic_asm(
            mnemonic=mnemonic,
            returns_old_value=returns_old_value,
            cache_fields=cache_immediate_fields,
        ),
    )


def _buffer_atomic_cmpswap_overlay(
    *,
    encoding_name: str,
    resource_field_name: str,
    offset_field_name: str,
    offset_bit_width: int,
    return_field_name: str,
    return_field_value: int,
    cache_fields: tuple[tuple[str, int], ...],
    cache_immediate_field_names: tuple[str, ...],
) -> AmdgpuDescriptorOverlay:
    cache_immediate_fields = tuple(
        (field_name, bit_width)
        for field_name, bit_width in cache_fields
        if field_name in cache_immediate_field_names
    )
    cache_immediate_defaults = {return_field_name: return_field_value}
    fixed_encoding_fields = tuple(
        (
            field_name,
            return_field_value if field_name == return_field_name else 0,
        )
        for field_name, _bit_width in cache_fields
        if field_name not in cache_immediate_field_names
    )
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.buffer_atomic_cmpswap_b32_rtn",
        instruction_name="BUFFER_ATOMIC_CMPSWAP",
        mnemonic="buffer_atomic_cmpswap_b32",
        encoding_name=encoding_name,
        semantic_tag="memory.global.atomic.compare_exchange.b32.return",
        schedule_class=_SCHEDULE_VMEM_ATOMIC_RETURN,
        operands=(
            AmdgpuOperandOverlay("VDATA", _vgpr_result(units=2)),
            AmdgpuOperandOverlay(
                "VDATA",
                _vgpr_operand("value", units=2),
                role_exception_reason=_BUFFER_ATOMIC_VDATA_INPUT_REASON,
            ),
            AmdgpuOperandOverlay(
                resource_field_name, _sgpr_resource("resource", units=4)
            ),
            _mubuf_vaddr_operand(),
            AmdgpuOperandOverlay("SOFFSET", _sgpr_operand("soffset")),
        ),
        implicit_operands=(
            _ignore_global_atomic_memory(
                data_format_name="FMT_NUM_U32", is_input=False
            ),
            _ignore_global_atomic_memory(data_format_name="FMT_NUM_U32", is_input=True),
        ),
        immediate_fields=(
            offset_field_name,
            *_cache_field_names(cache_immediate_fields),
        ),
        immediates=(
            _offset_immediate(offset_bit_width),
            *_cache_immediates_with_defaults(
                cache_immediate_fields, cache_immediate_defaults
            ),
        ),
        fixed_encoding_fields=(("IDXEN", 0), ("OFFEN", 1), *fixed_encoding_fields),
        effects=_global_atomic_effects(32, counter_id=_COUNTER_VMEM_LOAD),
        constraints=_DESTRUCTIVE_BUFFER_ATOMIC_CONSTRAINTS,
        flags=(DescriptorFlag.SIDE_EFFECTING,),
        asm_forms=_buffer_atomic_asm(
            mnemonic="buffer_atomic_cmpswap_b32",
            returns_old_value=True,
            cache_fields=cache_immediate_fields,
        ),
    )


def _buffer_atomic_overlays(
    *,
    encoding_name: str,
    resource_field_name: str,
    offset_field_name: str,
    offset_bit_width: int,
    return_field_name: str,
    return_field_value: int,
    cache_fields: tuple[tuple[str, int], ...],
    cache_immediate_field_names: tuple[str, ...] = (),
    include_packed_half_add: bool = False,
) -> tuple[AmdgpuDescriptorOverlay, ...]:
    rows = [
        ("add_u32", "BUFFER_ATOMIC_ADD_U32", "add.u32", "FMT_NUM_U32", True),
        ("sub_u32", "BUFFER_ATOMIC_SUB_U32", "sub.u32", "FMT_NUM_U32", True),
        ("min_i32", "BUFFER_ATOMIC_MIN_I32", "min.i32", "FMT_NUM_I32", True),
        ("max_i32", "BUFFER_ATOMIC_MAX_I32", "max.i32", "FMT_NUM_I32", True),
        ("min_u32", "BUFFER_ATOMIC_MIN_U32", "min.u32", "FMT_NUM_U32", True),
        ("max_u32", "BUFFER_ATOMIC_MAX_U32", "max.u32", "FMT_NUM_U32", True),
        ("and_b32", "BUFFER_ATOMIC_AND_B32", "and.b32", "FMT_NUM_B32", True),
        ("or_b32", "BUFFER_ATOMIC_OR_B32", "or.b32", "FMT_NUM_B32", True),
        ("xor_b32", "BUFFER_ATOMIC_XOR_B32", "xor.b32", "FMT_NUM_B32", True),
        (
            "swap_b32",
            "BUFFER_ATOMIC_SWAP_B32",
            "exchange.b32",
            "FMT_NUM_B32",
            False,
        ),
        ("add_f32", "BUFFER_ATOMIC_ADD_F32", "add.f32", "FMT_NUM_F32", True),
        ("min_f32", "BUFFER_ATOMIC_MIN_F32", "minnum.f32", "FMT_NUM_F32", True),
        ("max_f32", "BUFFER_ATOMIC_MAX_F32", "maxnum.f32", "FMT_NUM_F32", True),
    ]
    if include_packed_half_add:
        rows.extend(
            (
                (
                    "pk_add_f16",
                    "BUFFER_ATOMIC_PK_ADD_F16",
                    "add.pk2.f16",
                    "FMT_NUM_PK2_F16",
                    True,
                ),
                (
                    "pk_add_bf16",
                    "BUFFER_ATOMIC_PK_ADD_BF16",
                    "add.pk2.bf16",
                    "FMT_NUM_PK2_BF16",
                    True,
                ),
            )
        )
    overlays: list[AmdgpuDescriptorOverlay] = []
    for (
        mnemonic_suffix,
        instruction_name,
        semantic_suffix,
        data_format_name,
        has_no_return_form,
    ) in rows:
        if has_no_return_form:
            overlays.append(
                _buffer_atomic_overlay(
                    descriptor_key=f"amdgpu.buffer_atomic_{mnemonic_suffix}",
                    instruction_name=instruction_name,
                    mnemonic=f"buffer_atomic_{mnemonic_suffix}",
                    semantic_tag=f"memory.global.atomic.{semantic_suffix}",
                    data_format_name=data_format_name,
                    returns_old_value=False,
                    encoding_name=encoding_name,
                    resource_field_name=resource_field_name,
                    offset_field_name=offset_field_name,
                    offset_bit_width=offset_bit_width,
                    return_field_name=return_field_name,
                    return_field_value=return_field_value,
                    cache_fields=cache_fields,
                    cache_immediate_field_names=cache_immediate_field_names,
                )
            )
        overlays.append(
            _buffer_atomic_overlay(
                descriptor_key=f"amdgpu.buffer_atomic_{mnemonic_suffix}_rtn",
                instruction_name=instruction_name,
                mnemonic=f"buffer_atomic_{mnemonic_suffix}",
                semantic_tag=f"memory.global.atomic.{semantic_suffix}.return",
                data_format_name=data_format_name,
                returns_old_value=True,
                encoding_name=encoding_name,
                resource_field_name=resource_field_name,
                offset_field_name=offset_field_name,
                offset_bit_width=offset_bit_width,
                return_field_name=return_field_name,
                return_field_value=return_field_value,
                cache_fields=cache_fields,
                cache_immediate_field_names=cache_immediate_field_names,
            )
        )
    overlays.append(
        _buffer_atomic_cmpswap_overlay(
            encoding_name=encoding_name,
            resource_field_name=resource_field_name,
            offset_field_name=offset_field_name,
            offset_bit_width=offset_bit_width,
            return_field_name=return_field_name,
            return_field_value=return_field_value,
            cache_fields=cache_fields,
            cache_immediate_field_names=cache_immediate_field_names,
        )
    )
    return tuple(overlays)


__all__ = (
    "_FLAT_ATOMIC_GFX11_ROWS",
    "_FLAT_ATOMIC_GFX12_ROWS",
    "_FLAT_ATOMIC_GFX950_ROWS",
    "_GLOBAL_ATOMIC_DEFAULT_ROWS",
    "_GLOBAL_ATOMIC_GFX940_ROWS",
    "_buffer_atomic_cmpswap_overlay",
    "_buffer_atomic_overlay",
    "_buffer_atomic_overlays",
    "_flat_atomic_cmpswap_overlay",
    "_flat_atomic_overlay",
    "_flat_atomic_overlays",
    "_global_atomic_cmpswap_overlay",
    "_global_atomic_overlay",
    "_global_atomic_overlays",
)
