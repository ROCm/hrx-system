# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


# ruff: noqa: F403, F405

"""Gfx125x workgroup-cluster async-to-LDS descriptors."""

from __future__ import annotations

from loom.target.arch.amdgpu.encoding import (
    AMDGPU_ENCODING_FORMAT_SOPP,
    AMDGPU_ENCODING_FORMAT_VGLOBAL,
)

from .common import *


def _encoded_operand(operand: Operand, field_name: str) -> Operand:
    return replace(operand, encoding_field_id=amdgpu_encoding_field_id(field_name))


def _encoded_cluster_immediates() -> tuple[Immediate, ...]:
    offset = replace(
        _signed_offset_immediate(24),
        encoding_field_id=amdgpu_encoding_field_id("IOFFSET"),
    )
    cache = tuple(
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
    return (offset, *cache)


def _cluster_load_async_to_lds_effects(width_bits: int) -> tuple[Effect, Effect]:
    return (
        Effect(
            EffectKind.READ,
            memory_space=MemorySpace.GLOBAL,
            flags=(EffectFlag.DEPENDENCY,),
            counter_id=_COUNTER_ASYNC,
            width_bits=width_bits,
        ),
        Effect(
            EffectKind.WRITE,
            memory_space=MemorySpace.WORKGROUP,
            flags=(EffectFlag.DEPENDENCY,),
            counter_id=_COUNTER_ASYNC,
            width_bits=width_bits,
        ),
    )


def _cluster_load_async_to_lds_descriptor(
    width_bits: int, encoding_id: int
) -> Descriptor:
    mnemonic = f"cluster_load_async_to_lds_b{width_bits}"
    return Descriptor(
        key=f"amdgpu.{mnemonic}",
        mnemonic=mnemonic,
        semantic_tag=f"memory.cluster.load.to_lds.u{width_bits}",
        operands=(
            _encoded_operand(_vgpr_operand("lds_addr"), "VDST"),
            _encoded_operand(_vgpr_operand("addr"), "VADDR"),
            _encoded_operand(_sgpr_operand("saddr", units=2), "SADDR"),
            _m0_implicit_resource(),
        ),
        immediates=_encoded_cluster_immediates(),
        asm_forms=_asm(
            operands=("lds_addr", "addr", "saddr", "m0"),
            immediates=_memory_asm_immediate_names(_GFX12_VECTOR_CACHE_FIELDS),
            named_immediates=True,
            native_assembly_values=(
                _native_operand("lds_addr"),
                _native_operand("addr"),
                _native_operand("saddr"),
                _native_amdgpu_named_i64_immediate("offset"),
                _native_amdgpu_named_flag_immediate("nv"),
                _native_amdgpu_gfx12_scope_immediate("scope"),
                _native_amdgpu_gfx12_load_temporal_immediate("th"),
            ),
        ),
        effects=_cluster_load_async_to_lds_effects(width_bits),
        schedule_class=_SCHEDULE_CLUSTER_LOAD_LDS,
        encoding_format_id=AMDGPU_ENCODING_FORMAT_VGLOBAL,
        encoding_id=encoding_id,
        flags=(DescriptorFlag.SIDE_EFFECTING,),
    )


def _s_wait_asynccnt_descriptor() -> Descriptor:
    return Descriptor(
        key="amdgpu.s_wait_asynccnt",
        mnemonic="s_wait_asynccnt",
        semantic_tag="control.waitcnt.async",
        operands=(),
        immediates=(
            Immediate(
                "asynccnt",
                ImmediateKind.UNSIGNED,
                bit_width=16,
                encoding_id=_WAIT_COUNTER_ASYNC_ENCODING_ID,
                encoding_field_id=amdgpu_encoding_field_id("SIMM16"),
                unsigned_max=(2**16) - 1,
            ),
        ),
        asm_forms=_asm(
            immediates=("asynccnt",),
            named_immediates=True,
            native_assembly_values=(_native_i64_immediate("asynccnt"),),
        ),
        effects=(_ASYNC_WAIT_EFFECT,),
        schedule_class=_SCHEDULE_WAIT_ASYNC,
        encoding_format_id=AMDGPU_ENCODING_FORMAT_SOPP,
        # The SOPP OP field excludes the format's fixed high encoding bit.
        # OP=0x4A combines with ENC_SOPP to produce 0xBFCA0000.
        encoding_id=0x4A,
        flags=(DescriptorFlag.SIDE_EFFECTING,),
    )


def _gfx125x_cluster_descriptors() -> tuple[Descriptor, ...]:
    return (
        _cluster_load_async_to_lds_descriptor(8, 0x6A),
        _cluster_load_async_to_lds_descriptor(32, 0x6B),
        _cluster_load_async_to_lds_descriptor(64, 0x6C),
        _cluster_load_async_to_lds_descriptor(128, 0x6D),
        _s_wait_asynccnt_descriptor(),
    )


__all__ = (
    "_cluster_load_async_to_lds_descriptor",
    "_gfx125x_cluster_descriptors",
    "_s_wait_asynccnt_descriptor",
)
