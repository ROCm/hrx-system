# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


# ruff: noqa: F403, F405

"""Gfx125x tensor-memory packet descriptors."""

from __future__ import annotations

from loom.target.arch.amdgpu.encoding import AMDGPU_ENCODING_FORMAT_VIMAGE

from .common import *


def _encoded_tensor_dgroup(field_name: str, units: int) -> Operand:
    return replace(
        _sgpr_resource(field_name.lower(), units=units),
        encoding_field_id=amdgpu_encoding_field_id(field_name),
    )


def _encoded_tensor_cache_immediates() -> tuple[Immediate, ...]:
    return tuple(
        replace(
            immediate,
            encoding_field_id=amdgpu_encoding_field_id(field_name),
        )
        for immediate, (field_name, _bit_width) in zip(
            _cache_immediates(_GFX12_VECTOR_CACHE_FIELDS),
            _GFX12_VECTOR_CACHE_FIELDS,
            strict=True,
        )
    )


def _tensor_load_to_lds_effects() -> tuple[Effect, Effect]:
    return (
        Effect(
            EffectKind.READ,
            memory_space=MemorySpace.GLOBAL,
            flags=(EffectFlag.DEPENDENCY,),
            counter_id=_COUNTER_TENSOR,
        ),
        Effect(
            EffectKind.WRITE,
            memory_space=MemorySpace.WORKGROUP,
            flags=(EffectFlag.DEPENDENCY,),
            counter_id=_COUNTER_TENSOR,
        ),
    )


def _tensor_load_to_lds_descriptor(dgroup_count: int) -> Descriptor:
    if dgroup_count not in (2, 4):
        raise ValueError("gfx125x tensor loads require two or four dgroups")
    operands = (
        _encoded_tensor_dgroup("VADDR0", 4),
        _encoded_tensor_dgroup("VADDR1", 8),
        *(
            (
                _encoded_tensor_dgroup("VADDR2", 4),
                _encoded_tensor_dgroup("VADDR3", 4),
            )
            if dgroup_count == 4
            else ()
        ),
    )
    fixed_fields = (
        EncodingFieldValue(amdgpu_encoding_field_id("DIM"), 1),
        EncodingFieldValue(amdgpu_encoding_field_id("R128"), 0),
        EncodingFieldValue(amdgpu_encoding_field_id("D16"), 0),
        EncodingFieldValue(amdgpu_encoding_field_id("A16"), 0),
        EncodingFieldValue(amdgpu_encoding_field_id("DMASK"), 1),
        EncodingFieldValue(amdgpu_encoding_field_id("VDATA"), 0),
        EncodingFieldValue(amdgpu_encoding_field_id("RSRC"), 0),
        EncodingFieldValue(amdgpu_encoding_field_id("TFE"), 0),
        EncodingFieldValue(amdgpu_encoding_field_id("VADDR4"), 0x7C),
        *(
            (
                EncodingFieldValue(amdgpu_encoding_field_id("VADDR2"), 0x7C),
                EncodingFieldValue(amdgpu_encoding_field_id("VADDR3"), 0x7C),
            )
            if dgroup_count == 2
            else ()
        ),
    )
    operand_names = tuple(operand.field_name for operand in operands)
    cache_immediate_names = tuple(
        field_name.lower() for field_name, _bit_width in _GFX12_VECTOR_CACHE_FIELDS
    )
    return Descriptor(
        key=f"amdgpu.tensor_load_to_lds.d{dgroup_count}",
        mnemonic=f"tensor_load_to_lds_d{dgroup_count}",
        semantic_tag="memory.tensor.load.to_lds",
        operands=operands,
        immediates=_encoded_tensor_cache_immediates(),
        encoding_field_values=fixed_fields,
        asm_forms=_asm(
            mnemonic=f"tensor_load_to_lds_d{dgroup_count}",
            native_assembly_mnemonic="tensor_load_to_lds",
            operands=operand_names,
            immediates=cache_immediate_names,
            named_immediates=True,
            native_assembly_values=(
                *(_native_operand(name) for name in operand_names),
                _native_amdgpu_named_flag_immediate("nv", name="nv"),
                _native_amdgpu_named_i64_immediate("scope", name="scope"),
                _native_amdgpu_named_i64_immediate("th", name="th"),
            ),
        ),
        effects=_tensor_load_to_lds_effects(),
        schedule_class=_SCHEDULE_TENSOR_LOAD_LDS,
        encoding_format_id=AMDGPU_ENCODING_FORMAT_VIMAGE,
        encoding_id=0xC4,
        flags=(DescriptorFlag.SIDE_EFFECTING,),
    )


def _s_wait_tensorcnt_descriptor() -> Descriptor:
    return Descriptor(
        key="amdgpu.s_wait_tensorcnt",
        mnemonic="s_wait_tensorcnt",
        semantic_tag="control.waitcnt.tensor",
        operands=(),
        immediates=(
            Immediate(
                "tensorcnt",
                ImmediateKind.UNSIGNED,
                bit_width=16,
                encoding_id=_WAIT_COUNTER_TENSOR_ENCODING_ID,
                encoding_field_id=amdgpu_encoding_field_id("SIMM16"),
                unsigned_max=(2**16) - 1,
            ),
        ),
        asm_forms=_asm(
            immediates=("tensorcnt",),
            named_immediates=True,
            native_assembly_values=(_native_i64_immediate("tensorcnt"),),
        ),
        effects=(_TENSOR_WAIT_EFFECT,),
        schedule_class=_SCHEDULE_WAIT_TENSOR,
        encoding_format_id=AMDGPU_ENCODING_FORMAT_SOPP,
        # The SOPP OP field excludes the format's fixed high encoding bit.
        # OP=0x4B combines with ENC_SOPP to produce 0xBFCB0000.
        encoding_id=0x4B,
        flags=(DescriptorFlag.SIDE_EFFECTING,),
    )


def _gfx125x_tensor_descriptors() -> tuple[Descriptor, ...]:
    return (
        _tensor_load_to_lds_descriptor(2),
        _tensor_load_to_lds_descriptor(4),
        _s_wait_tensorcnt_descriptor(),
    )


__all__ = (
    "_gfx125x_tensor_descriptors",
    "_s_wait_tensorcnt_descriptor",
    "_tensor_load_to_lds_descriptor",
)
