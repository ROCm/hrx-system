# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


# ruff: noqa: F403, F405

"""AMDGPU core descriptor-set assembly."""

from __future__ import annotations

from .alu import *
from .atomic import *
from .cdna import *
from .common import *
from .control import *
from .matrix import *
from .memory import *
from .rdna3 import *
from .rdna4 import *
from .workgroup import *

_CDNA_DWORD_MEMORY_INSTRUCTION_SUFFIXES = (
    "DWORD",
    "DWORDX2",
    "DWORDX3",
    "DWORDX4",
)
_CDNA_DWORD_MEMORY_MNEMONIC_SUFFIXES = (
    "dword",
    "dwordx2",
    "dwordx3",
    "dwordx4",
)
_BYTE_MEMORY_INSTRUCTION_SUFFIXES = ("B32", "B64", "B96", "B128")
_BYTE_MEMORY_MNEMONIC_SUFFIXES = ("b32", "b64", "b96", "b128")

_RDNA4_VBUFFER_DWORD_WIDTH_OVERLAY_ROWS = (
    (_buffer_load_dword_overlay, _buffer_load_dword_vaddr_offset_overlay),
    (_buffer_load_64_overlay, _buffer_load_64_vaddr_offset_overlay),
    (_buffer_load_96_overlay, _buffer_load_96_vaddr_offset_overlay),
    (_buffer_load_128_overlay, _buffer_load_128_vaddr_offset_overlay),
    (_buffer_store_dword_overlay, _buffer_store_dword_vaddr_offset_overlay),
    (_buffer_store_64_overlay, _buffer_store_64_vaddr_offset_overlay),
    (_buffer_store_96_overlay, _buffer_store_96_vaddr_offset_overlay),
    (_buffer_store_128_overlay, _buffer_store_128_vaddr_offset_overlay),
)

_S_LOAD_DWORD_WIDTH_OVERLAY_ROWS = (
    (_s_load_dword_overlay, "amdgpu.s_load_dword_offset_only", False),
    (_s_load_dwordx2_overlay, "amdgpu.s_load_dwordx2_offset_only", False),
    (_s_load_96_overlay, "amdgpu.s_load_b96_offset_only", True),
    (_s_load_dwordx4_overlay, "amdgpu.s_load_dwordx4_offset_only", False),
    (_s_load_dwordx8_overlay, "amdgpu.s_load_dwordx8_offset_only", False),
    (_s_load_dwordx16_overlay, "amdgpu.s_load_dwordx16_offset_only", False),
)

_S_BUFFER_LOAD_WIDTH_OVERLAY_ROWS = (
    (_s_buffer_load_dword_overlay, False),
    (_s_buffer_load_64_overlay, False),
    (_s_buffer_load_96_overlay, True),
    (_s_buffer_load_128_overlay, False),
    (_s_buffer_load_256_overlay, False),
    (_s_buffer_load_512_overlay, False),
)

_CDNA_S_BUFFER_LOAD_WIDTH_OVERLAY_ROWS = (
    (_s_buffer_load_dword_overlay, None, None, None),
    (
        _s_buffer_load_64_overlay,
        "amdgpu.s_buffer_load_dwordx2",
        "S_BUFFER_LOAD_DWORDX2",
        "s_buffer_load_dwordx2",
    ),
    (
        _s_buffer_load_128_overlay,
        "amdgpu.s_buffer_load_dwordx4",
        "S_BUFFER_LOAD_DWORDX4",
        "s_buffer_load_dwordx4",
    ),
    (
        _s_buffer_load_256_overlay,
        "amdgpu.s_buffer_load_dwordx8",
        "S_BUFFER_LOAD_DWORDX8",
        "s_buffer_load_dwordx8",
    ),
    (
        _s_buffer_load_512_overlay,
        "amdgpu.s_buffer_load_dwordx16",
        "S_BUFFER_LOAD_DWORDX16",
        "s_buffer_load_dwordx16",
    ),
)


def _rdna4_vbuffer_dword_width_overlays() -> tuple[AmdgpuDescriptorOverlay, ...]:
    overlays: list[AmdgpuDescriptorOverlay] = []
    for base_overlay, vaddr_offset_overlay in _RDNA4_VBUFFER_DWORD_WIDTH_OVERLAY_ROWS:
        overlays.append(
            base_overlay(
                encoding_name="ENC_VBUFFER",
                resource_field_name="RSRC",
                offset_field_name="IOFFSET",
                offset_bit_width=24,
                cache_fields=_GFX12_VECTOR_CACHE_FIELDS,
                off_zero_descriptor_key=None,
            )
        )
        overlays.append(
            vaddr_offset_overlay(
                encoding_name="ENC_VBUFFER",
                resource_field_name="RSRC",
                offset_field_name="IOFFSET",
                offset_bit_width=24,
                cache_fields=_GFX12_VECTOR_CACHE_FIELDS,
                fixed_soffset=_VBUFFER_SOFFSET_NULL,
                fixed_soffset_native_spelling="null",
            )
        )
    return tuple(overlays)


def _s_load_dword_width_overlays(
    offset_field_name: str = "OFFSET",
    *,
    offset_bit_width: int = 21,
    fixed_encoding_fields: tuple[tuple[str, AmdgpuFixedEncodingValue], ...] = (),
    include_b96: bool = False,
) -> tuple[AmdgpuDescriptorOverlay, ...]:
    return tuple(
        overlay(
            offset_field_name,
            offset_bit_width=offset_bit_width,
            fixed_encoding_fields=fixed_encoding_fields,
        )
        for (
            overlay,
            _offset_only_descriptor_key,
            requires_b96,
        ) in _S_LOAD_DWORD_WIDTH_OVERLAY_ROWS
        if include_b96 or not requires_b96
    )


def _s_load_dword_offset_only_overlays(
    offset_field_name: str = "OFFSET",
    *,
    offset_bit_width: int = 21,
    fixed_encoding_fields: tuple[tuple[str, AmdgpuFixedEncodingValue], ...] = (),
    fixed_soffset: AmdgpuFixedEncodingValue,
    include_b96: bool = False,
) -> tuple[AmdgpuDescriptorOverlay, ...]:
    return tuple(
        overlay(
            offset_field_name,
            offset_bit_width=offset_bit_width,
            fixed_encoding_fields=fixed_encoding_fields,
            descriptor_key=offset_only_descriptor_key,
            fixed_soffset=fixed_soffset,
        )
        for (
            overlay,
            offset_only_descriptor_key,
            requires_b96,
        ) in _S_LOAD_DWORD_WIDTH_OVERLAY_ROWS
        if include_b96 or not requires_b96
    )


def _s_buffer_load_width_overlays(
    offset_field_name: str = "OFFSET",
    *,
    offset_bit_width: int = 21,
    fixed_encoding_fields: tuple[tuple[str, AmdgpuFixedEncodingValue], ...] = (),
    include_b96: bool = False,
) -> tuple[AmdgpuDescriptorOverlay, ...]:
    return tuple(
        overlay(
            offset_field_name=offset_field_name,
            offset_bit_width=offset_bit_width,
            fixed_encoding_fields=fixed_encoding_fields,
        )
        for (
            overlay,
            requires_b96,
        ) in _S_BUFFER_LOAD_WIDTH_OVERLAY_ROWS
        if include_b96 or not requires_b96
    )


def _cdna_s_buffer_load_width_overlays(
    *,
    fixed_encoding_fields: tuple[tuple[str, AmdgpuFixedEncodingValue], ...],
) -> tuple[AmdgpuDescriptorOverlay, ...]:
    overlays: list[AmdgpuDescriptorOverlay] = []
    for (
        overlay,
        descriptor_key,
        instruction_name,
        mnemonic,
    ) in _CDNA_S_BUFFER_LOAD_WIDTH_OVERLAY_ROWS:
        overlay_kwargs = {"fixed_encoding_fields": fixed_encoding_fields}
        if descriptor_key is not None:
            overlay_kwargs.update(
                {
                    "descriptor_key": descriptor_key,
                    "instruction_name": instruction_name,
                    "mnemonic": mnemonic,
                }
            )
        overlays.append(overlay(**overlay_kwargs))
    return tuple(overlays)


def _cdna_scalar_fma_overlays() -> tuple[AmdgpuDescriptorOverlay, ...]:
    return (
        _v_mad_f16_overlay(),
        _v_mac_f16_overlay(),
        _v_madak_f16_overlay(),
        _v_madmk_f16_overlay(),
        _v_fma_f16_overlay(),
        _v_fma_f64_overlay(),
        _v_fmac_f64_overlay(),
    )


def _rdna_scalar_fma_overlays() -> tuple[AmdgpuDescriptorOverlay, ...]:
    return (
        _v_fma_f16_overlay(),
        _v_fmac_f16_overlay(),
        _v_fmaak_f16_overlay(),
        _v_fmamk_f16_overlay(),
        _v_fma_f64_overlay(),
    )


def _rdna_scalar_domain_fma_overlays() -> tuple[AmdgpuDescriptorOverlay, ...]:
    return (
        _s_fmaak_f32_overlay(),
        _s_fmamk_f32_overlay(),
        _s_fmac_f32_overlay(),
        _s_fmac_f16_overlay(),
    )


def _cdna_core_overlays(
    *,
    global_load_lds_variants: tuple[tuple[str, str, str, int, int], ...],
    buffer_load_lds_variants: tuple[
        tuple[str, str, str, int, int, AmdgpuImplicitOperandOverlay], ...
    ],
    include_v_dot2_f32_bf16: bool,
    include_v_cvt_pk_bf16_f32: bool,
    include_ds_transpose_reads: bool,
    matrix_overlays: tuple[AmdgpuDescriptorOverlay, ...],
) -> tuple[AmdgpuDescriptorOverlay, ...]:
    return (
        _s_add_u32_overlay(),
        _s_add_u32_rhs_inline_overlay(),
        _s_addk_i32_overlay(),
        _s_addc_u32_overlay(),
        _s_sub_u32_overlay(),
        _s_sub_u32_rhs_inline_overlay(),
        _s_mul_i32_overlay(),
        _s_mul_i32_rhs_inline_overlay(),
        _s_mulk_i32_overlay(),
        _s_mul_hi_u32_overlay(),
        _s_min_i32_overlay(),
        _s_max_i32_overlay(),
        _s_min_u32_overlay(),
        _s_max_u32_overlay(),
        _s_cselect_b32_overlay(),
        *_s_cmp_i32_overlays(),
        *_s_cmp_u64_overlays(),
        _s_and_saveexec_b64_overlay("default"),
        _v_add_u32_overlay("V_ADD_U32"),
        _v_add_u32_src0_inline_overlay("V_ADD_U32"),
        _v_add_u32_literal_overlay("V_ADD_U32"),
        _v_add_co_u32_overlay(),
        _v_add_co_ci_u32_overlay(
            instruction_name="V_ADDC_CO_U32", mnemonic="v_addc_co_u32"
        ),
        _v_sub_u32_overlay("V_SUB_U32", "v_sub_u32"),
        _v_mov_b32_literal_overlay(),
        _v_mov_b32_copy_overlay(),
        _v_mov_b32_dpp_legacy_overlay(),
        _v_mov_b32_sdwa_overlay(),
        _v_mul_lo_u32_overlay(),
        _v_mul_hi_u32_overlay(),
        _v_mul_u32_u24_overlay(),
        _v_mul_u32_u24_src0_inline_overlay(),
        _v_mul_u32_u24_literal_overlay(),
        _v_mad_u32_u24_overlay(include_literal_forms=False),
        _v_min_i32_overlay(),
        _v_max_i32_overlay(),
        _v_min_u32_overlay(),
        _v_max_u32_overlay(),
        _v_readfirstlane_b32_overlay(),
        *_integer_bitwise_shift_overlays(),
        *_integer_bitwise_permute_overlays(),
        *_v_binary_f32_overlays(),
        _v_fma_f32_overlay(),
        _v_fmaak_f32_overlay(),
        _v_fmac_f32_overlay(),
        _v_fmamk_f32_overlay(),
        *_cdna_scalar_fma_overlays(),
        _v_pk_fmac_f16_overlay(),
        _v_pk_fma_f16_overlay(),
        _v_pk_mad_i16_overlay(),
        _v_pk_mad_u16_overlay(),
        _v_pk_fma_f32_overlay(),
        *_v_mad_mix_f32_overlays(op_sel_field="OP_SEL", op_sel_hi_field="OP_SEL_HI"),
        *_v_mad_mixlo_f16_overlays(op_sel_field="OP_SEL", op_sel_hi_field="OP_SEL_HI"),
        *_v_mad_mixhi_f16_overlays(op_sel_field="OP_SEL", op_sel_hi_field="OP_SEL_HI"),
        *_v_native_f32_math_overlays(),
        _v_sqrt_f32_overlay(),
        _v_rsq_f32_overlay(),
        _v_rcp_f32_overlay(),
        _v_cvt_f32_f16_overlay(),
        _v_cvt_f16_f32_overlay(),
        _v_cvt_pk_u16_u32_overlay(),
        *((_v_cvt_pk_bf16_f32_overlay(),) if include_v_cvt_pk_bf16_f32 else ()),
        _v_cvt_f32_i32_overlay(),
        _v_cvt_f32_u32_overlay(),
        _v_cvt_i32_f32_overlay(),
        _v_cvt_u32_f32_overlay(),
        *_v_cmp_overlays(),
        *_v_cndmask_b32_overlays(include_literal_forms=False),
        *_s_load_dword_width_overlays(
            fixed_encoding_fields=_CDNA_SMEM_SGPR_IMM_FIXED_FIELDS
        ),
        *_s_scratch_load_dword_width_overlays(
            fixed_encoding_fields=_CDNA_SMEM_SGPR_IMM_FIXED_FIELDS
        ),
        *_s_load_dword_offset_only_overlays(
            fixed_encoding_fields=_CDNA_SMEM_OFFSET_ONLY_FIXED_FIELDS,
            fixed_soffset=0,
        ),
        *_s_scratch_load_dword_width_overlays(
            fixed_encoding_fields=_CDNA_SMEM_OFFSET_ONLY_FIXED_FIELDS,
            fixed_soffset=0,
        ),
        *_cdna_s_buffer_load_width_overlays(
            fixed_encoding_fields=_CDNA_SMEM_SGPR_IMM_FIXED_FIELDS
        ),
        *_s_store_dword_width_overlays(
            fixed_encoding_fields=_CDNA_SMEM_SGPR_IMM_FIXED_FIELDS
        ),
        *_s_scratch_store_dword_width_overlays(
            fixed_encoding_fields=_CDNA_SMEM_SGPR_IMM_FIXED_FIELDS
        ),
        *_s_store_dword_width_overlays(
            fixed_encoding_fields=_CDNA_SMEM_OFFSET_ONLY_FIXED_FIELDS,
            fixed_soffset=0,
        ),
        *_s_scratch_store_dword_width_overlays(
            fixed_encoding_fields=_CDNA_SMEM_OFFSET_ONLY_FIXED_FIELDS,
            fixed_soffset=0,
        ),
        *_s_buffer_store_dword_width_overlays(
            fixed_encoding_fields=_CDNA_SMEM_SGPR_IMM_FIXED_FIELDS
        ),
        _buffer_load_dword_overlay(
            encoding_name="ENC_MUBUF",
            resource_field_name="SRSRC",
            cache_fields=_GFX950_VECTOR_CACHE_FIELDS,
        ),
        _buffer_load_dword_off_zero_overlay(
            encoding_name="ENC_MUBUF",
            resource_field_name="SRSRC",
            cache_fields=_GFX950_VECTOR_CACHE_FIELDS,
        ),
        _buffer_load_dword_vaddr_offset_overlay(
            encoding_name="ENC_MUBUF",
            resource_field_name="SRSRC",
            cache_fields=_GFX950_VECTOR_CACHE_FIELDS,
        ),
        _buffer_load_64_overlay(
            descriptor_key="amdgpu.buffer_load_dwordx2",
            instruction_name="BUFFER_LOAD_DWORDX2",
            mnemonic="buffer_load_dwordx2",
            encoding_name="ENC_MUBUF",
            resource_field_name="SRSRC",
            cache_fields=_GFX950_VECTOR_CACHE_FIELDS,
            off_zero_descriptor_key="amdgpu.buffer_load_dwordx2_off_zero",
            vaddr_offset_descriptor_key="amdgpu.buffer_load_dwordx2_vaddr_offset",
        ),
        _buffer_load_64_off_zero_overlay(
            descriptor_key="amdgpu.buffer_load_dwordx2_off_zero",
            instruction_name="BUFFER_LOAD_DWORDX2",
            mnemonic="buffer_load_dwordx2",
            encoding_name="ENC_MUBUF",
            resource_field_name="SRSRC",
            cache_fields=_GFX950_VECTOR_CACHE_FIELDS,
        ),
        _buffer_load_64_vaddr_offset_overlay(
            descriptor_key="amdgpu.buffer_load_dwordx2_vaddr_offset",
            instruction_name="BUFFER_LOAD_DWORDX2",
            mnemonic="buffer_load_dwordx2",
            encoding_name="ENC_MUBUF",
            resource_field_name="SRSRC",
            cache_fields=_GFX950_VECTOR_CACHE_FIELDS,
        ),
        _buffer_load_96_overlay(
            descriptor_key="amdgpu.buffer_load_dwordx3",
            instruction_name="BUFFER_LOAD_DWORDX3",
            mnemonic="buffer_load_dwordx3",
            encoding_name="ENC_MUBUF",
            resource_field_name="SRSRC",
            cache_fields=_GFX950_VECTOR_CACHE_FIELDS,
            off_zero_descriptor_key="amdgpu.buffer_load_dwordx3_off_zero",
            vaddr_offset_descriptor_key="amdgpu.buffer_load_dwordx3_vaddr_offset",
        ),
        _buffer_load_96_off_zero_overlay(
            descriptor_key="amdgpu.buffer_load_dwordx3_off_zero",
            instruction_name="BUFFER_LOAD_DWORDX3",
            mnemonic="buffer_load_dwordx3",
            encoding_name="ENC_MUBUF",
            resource_field_name="SRSRC",
            cache_fields=_GFX950_VECTOR_CACHE_FIELDS,
        ),
        _buffer_load_96_vaddr_offset_overlay(
            descriptor_key="amdgpu.buffer_load_dwordx3_vaddr_offset",
            instruction_name="BUFFER_LOAD_DWORDX3",
            mnemonic="buffer_load_dwordx3",
            encoding_name="ENC_MUBUF",
            resource_field_name="SRSRC",
            cache_fields=_GFX950_VECTOR_CACHE_FIELDS,
        ),
        _buffer_load_128_overlay(
            descriptor_key="amdgpu.buffer_load_dwordx4",
            instruction_name="BUFFER_LOAD_DWORDX4",
            mnemonic="buffer_load_dwordx4",
            encoding_name="ENC_MUBUF",
            resource_field_name="SRSRC",
            cache_fields=_GFX950_VECTOR_CACHE_FIELDS,
            off_zero_descriptor_key="amdgpu.buffer_load_dwordx4_off_zero",
            vaddr_offset_descriptor_key="amdgpu.buffer_load_dwordx4_vaddr_offset",
        ),
        _buffer_load_128_off_zero_overlay(
            descriptor_key="amdgpu.buffer_load_dwordx4_off_zero",
            instruction_name="BUFFER_LOAD_DWORDX4",
            mnemonic="buffer_load_dwordx4",
            encoding_name="ENC_MUBUF",
            resource_field_name="SRSRC",
            cache_fields=_GFX950_VECTOR_CACHE_FIELDS,
        ),
        _buffer_load_128_vaddr_offset_overlay(
            descriptor_key="amdgpu.buffer_load_dwordx4_vaddr_offset",
            instruction_name="BUFFER_LOAD_DWORDX4",
            mnemonic="buffer_load_dwordx4",
            encoding_name="ENC_MUBUF",
            resource_field_name="SRSRC",
            cache_fields=_GFX950_VECTOR_CACHE_FIELDS,
        ),
        _buffer_store_dword_overlay(
            encoding_name="ENC_MUBUF",
            resource_field_name="SRSRC",
            cache_fields=_GFX950_VECTOR_CACHE_FIELDS,
        ),
        _buffer_store_dword_off_zero_overlay(
            encoding_name="ENC_MUBUF",
            resource_field_name="SRSRC",
            cache_fields=_GFX950_VECTOR_CACHE_FIELDS,
        ),
        _buffer_store_dword_vaddr_offset_overlay(
            encoding_name="ENC_MUBUF",
            resource_field_name="SRSRC",
            cache_fields=_GFX950_VECTOR_CACHE_FIELDS,
        ),
        _buffer_store_64_overlay(
            descriptor_key="amdgpu.buffer_store_dwordx2",
            instruction_name="BUFFER_STORE_DWORDX2",
            mnemonic="buffer_store_dwordx2",
            encoding_name="ENC_MUBUF",
            resource_field_name="SRSRC",
            cache_fields=_GFX950_VECTOR_CACHE_FIELDS,
            off_zero_descriptor_key="amdgpu.buffer_store_dwordx2_off_zero",
            vaddr_offset_descriptor_key="amdgpu.buffer_store_dwordx2_vaddr_offset",
        ),
        _buffer_store_64_off_zero_overlay(
            descriptor_key="amdgpu.buffer_store_dwordx2_off_zero",
            instruction_name="BUFFER_STORE_DWORDX2",
            mnemonic="buffer_store_dwordx2",
            encoding_name="ENC_MUBUF",
            resource_field_name="SRSRC",
            cache_fields=_GFX950_VECTOR_CACHE_FIELDS,
        ),
        _buffer_store_64_vaddr_offset_overlay(
            descriptor_key="amdgpu.buffer_store_dwordx2_vaddr_offset",
            instruction_name="BUFFER_STORE_DWORDX2",
            mnemonic="buffer_store_dwordx2",
            encoding_name="ENC_MUBUF",
            resource_field_name="SRSRC",
            cache_fields=_GFX950_VECTOR_CACHE_FIELDS,
        ),
        _buffer_store_96_overlay(
            descriptor_key="amdgpu.buffer_store_dwordx3",
            instruction_name="BUFFER_STORE_DWORDX3",
            mnemonic="buffer_store_dwordx3",
            encoding_name="ENC_MUBUF",
            resource_field_name="SRSRC",
            cache_fields=_GFX950_VECTOR_CACHE_FIELDS,
            off_zero_descriptor_key="amdgpu.buffer_store_dwordx3_off_zero",
            vaddr_offset_descriptor_key="amdgpu.buffer_store_dwordx3_vaddr_offset",
        ),
        _buffer_store_96_off_zero_overlay(
            descriptor_key="amdgpu.buffer_store_dwordx3_off_zero",
            instruction_name="BUFFER_STORE_DWORDX3",
            mnemonic="buffer_store_dwordx3",
            encoding_name="ENC_MUBUF",
            resource_field_name="SRSRC",
            cache_fields=_GFX950_VECTOR_CACHE_FIELDS,
        ),
        _buffer_store_96_vaddr_offset_overlay(
            descriptor_key="amdgpu.buffer_store_dwordx3_vaddr_offset",
            instruction_name="BUFFER_STORE_DWORDX3",
            mnemonic="buffer_store_dwordx3",
            encoding_name="ENC_MUBUF",
            resource_field_name="SRSRC",
            cache_fields=_GFX950_VECTOR_CACHE_FIELDS,
        ),
        _buffer_store_128_overlay(
            descriptor_key="amdgpu.buffer_store_dwordx4",
            instruction_name="BUFFER_STORE_DWORDX4",
            mnemonic="buffer_store_dwordx4",
            encoding_name="ENC_MUBUF",
            resource_field_name="SRSRC",
            cache_fields=_GFX950_VECTOR_CACHE_FIELDS,
            off_zero_descriptor_key="amdgpu.buffer_store_dwordx4_off_zero",
            vaddr_offset_descriptor_key="amdgpu.buffer_store_dwordx4_vaddr_offset",
        ),
        _buffer_store_128_off_zero_overlay(
            descriptor_key="amdgpu.buffer_store_dwordx4_off_zero",
            instruction_name="BUFFER_STORE_DWORDX4",
            mnemonic="buffer_store_dwordx4",
            encoding_name="ENC_MUBUF",
            resource_field_name="SRSRC",
            cache_fields=_GFX950_VECTOR_CACHE_FIELDS,
        ),
        _buffer_store_128_vaddr_offset_overlay(
            descriptor_key="amdgpu.buffer_store_dwordx4_vaddr_offset",
            instruction_name="BUFFER_STORE_DWORDX4",
            mnemonic="buffer_store_dwordx4",
            encoding_name="ENC_MUBUF",
            resource_field_name="SRSRC",
            cache_fields=_GFX950_VECTOR_CACHE_FIELDS,
        ),
        *_buffer_byte_memory_overlays(
            encoding_name="ENC_MUBUF",
            resource_field_name="SRSRC",
            cache_fields=_GFX950_VECTOR_CACHE_FIELDS,
        ),
        *_buffer_b16_memory_overlays(
            encoding_name="ENC_MUBUF",
            resource_field_name="SRSRC",
            cache_fields=_GFX950_VECTOR_CACHE_FIELDS,
        ),
        *_buffer_load_lds_overlays(
            encoding_name="ENC_MUBUF",
            resource_field_name="SRSRC",
            cache_fields=_GFX950_VECTOR_CACHE_FIELDS,
            variants=buffer_load_lds_variants,
        ),
        *_global_memory_overlays(
            instruction_suffixes=_CDNA_DWORD_MEMORY_INSTRUCTION_SUFFIXES,
            mnemonic_suffixes=_CDNA_DWORD_MEMORY_MNEMONIC_SUFFIXES,
            encoding_name="ENC_FLAT_GLBL",
            address_field_name="ADDR",
            load_data_field_name="VDST",
            store_data_field_name="DATA",
            offset_field_name="OFFSET",
            offset_bit_width=13,
            saddr_off=_GLOBAL_GFX950_SADDR_OFF,
            address_units=2,
            implicit_m0=True,
            cache_fields=_GFX950_VECTOR_CACHE_FIELDS,
        ),
        *_global_byte_memory_overlays(
            encoding_name="ENC_FLAT_GLBL",
            address_field_name="ADDR",
            load_data_field_name="VDST",
            store_data_field_name="DATA",
            offset_field_name="OFFSET",
            offset_bit_width=13,
            saddr_off=_GLOBAL_GFX950_SADDR_OFF,
            address_units=2,
            implicit_m0=True,
            load_mnemonic_suffixes=("ubyte", "sbyte"),
            cache_fields=_GFX950_VECTOR_CACHE_FIELDS,
        ),
        *_global_b16_memory_overlays(
            encoding_name="ENC_FLAT_GLBL",
            address_field_name="ADDR",
            load_data_field_name="VDST",
            store_data_field_name="DATA",
            offset_field_name="OFFSET",
            offset_bit_width=13,
            saddr_off=_GLOBAL_GFX950_SADDR_OFF,
            address_units=2,
            implicit_m0=True,
            load_mnemonic_suffixes=("ushort", "sshort"),
            cache_fields=_GFX950_VECTOR_CACHE_FIELDS,
        ),
        *_global_memory_overlays(
            instruction_suffixes=_CDNA_DWORD_MEMORY_INSTRUCTION_SUFFIXES,
            mnemonic_suffixes=_CDNA_DWORD_MEMORY_MNEMONIC_SUFFIXES,
            encoding_name="ENC_FLAT_GLBL",
            address_field_name="ADDR",
            load_data_field_name="VDST",
            store_data_field_name="DATA",
            offset_field_name="OFFSET",
            offset_bit_width=13,
            saddr_off=None,
            address_units=1,
            descriptor_key_suffix="_saddr",
            implicit_m0=True,
            cache_fields=_GFX950_VECTOR_CACHE_FIELDS,
        ),
        *_global_byte_memory_overlays(
            encoding_name="ENC_FLAT_GLBL",
            address_field_name="ADDR",
            load_data_field_name="VDST",
            store_data_field_name="DATA",
            offset_field_name="OFFSET",
            offset_bit_width=13,
            saddr_off=None,
            address_units=1,
            descriptor_key_suffix="_saddr",
            implicit_m0=True,
            load_mnemonic_suffixes=("ubyte", "sbyte"),
            cache_fields=_GFX950_VECTOR_CACHE_FIELDS,
        ),
        *_global_b16_memory_overlays(
            encoding_name="ENC_FLAT_GLBL",
            address_field_name="ADDR",
            load_data_field_name="VDST",
            store_data_field_name="DATA",
            offset_field_name="OFFSET",
            offset_bit_width=13,
            saddr_off=None,
            address_units=1,
            descriptor_key_suffix="_saddr",
            implicit_m0=True,
            load_mnemonic_suffixes=("ushort", "sshort"),
            cache_fields=_GFX950_VECTOR_CACHE_FIELDS,
        ),
        *_scratch_memory_overlays(
            instruction_suffixes=_CDNA_DWORD_MEMORY_INSTRUCTION_SUFFIXES,
            mnemonic_suffixes=_BYTE_MEMORY_MNEMONIC_SUFFIXES,
            encoding_name="ENC_FLAT_SCRATCH",
            address_field_name="ADDR",
            load_data_field_name="VDST",
            store_data_field_name="DATA",
            offset_field_name="OFFSET",
            offset_bit_width=13,
            offset_signed=True,
            fixed_vaddr=None,
            fixed_saddr=_SCRATCH_CDNA_SADDR_OFF,
            implicit_flat_scratch=True,
            implicit_m0=True,
            descriptor_key_suffix="_vaddr",
            narrow_byte_load_mnemonic_suffixes=("ubyte", "sbyte"),
            narrow_b16_load_mnemonic_suffixes=("ushort", "sshort"),
            cache_fields=_GFX950_VECTOR_CACHE_FIELDS,
        ),
        *_scratch_memory_overlays(
            instruction_suffixes=_CDNA_DWORD_MEMORY_INSTRUCTION_SUFFIXES,
            mnemonic_suffixes=_BYTE_MEMORY_MNEMONIC_SUFFIXES,
            encoding_name="ENC_FLAT_SCRATCH",
            address_field_name="ADDR",
            load_data_field_name="VDST",
            store_data_field_name="DATA",
            offset_field_name="OFFSET",
            offset_bit_width=13,
            offset_signed=True,
            fixed_vaddr=0,
            fixed_saddr=_SCRATCH_CDNA_SADDR_OFF,
            implicit_flat_scratch=True,
            implicit_m0=True,
            descriptor_key_suffix="_offset_only",
            narrow_byte_load_mnemonic_suffixes=("ubyte", "sbyte"),
            narrow_b16_load_mnemonic_suffixes=("ushort", "sshort"),
            cache_fields=_GFX950_VECTOR_CACHE_FIELDS,
        ),
        *_global_load_lds_overlays(
            address_units=2,
            saddr_off=_GLOBAL_GFX950_SADDR_OFF,
            cache_fields=_GFX950_VECTOR_CACHE_FIELDS,
            variants=global_load_lds_variants,
        ),
        *_global_load_lds_overlays(
            address_units=1,
            saddr_off=None,
            descriptor_key_suffix="_saddr",
            cache_fields=_GFX950_VECTOR_CACHE_FIELDS,
            variants=global_load_lds_variants,
        ),
        *_global_atomic_overlays(
            rows=_GLOBAL_ATOMIC_GFX940_ROWS,
            encoding_name="ENC_FLAT_GLBL",
            address_field_name="ADDR",
            data_field_name="DATA",
            offset_field_name="OFFSET",
            offset_bit_width=13,
            return_field_name="SC0",
            return_field_value=1,
            cache_fields=_GFX950_VECTOR_CACHE_FIELDS,
            saddr_off=None,
            address_units=1,
            descriptor_key_suffix="_saddr",
            implicit_m0=True,
        ),
        *_flat_atomic_overlays(
            rows=_FLAT_ATOMIC_GFX950_ROWS,
            cmpswap_instruction_name="FLAT_ATOMIC_CMPSWAP",
            encoding_name="ENC_FLAT",
            address_field_name="ADDR",
            data_field_name="DATA",
            offset_field_name="OFFSET",
            offset_bit_width=12,
            offset_signed=False,
            return_field_name="SC0",
            return_field_value=1,
            cache_fields=_GFX950_VECTOR_CACHE_FIELDS,
            implicit_flat_scratch=True,
            implicit_m0=True,
            allow_accumulator_operands=True,
        ),
        *_ds_memory_overlays(include_packed_half_atomic_add=True),
        *_ds_crosslane_overlays(),
        _v_dot2_f32_f16_overlay(),
        *((_v_dot2_f32_bf16_overlay(),) if include_v_dot2_f32_bf16 else ()),
        _v_dot4_i32_i8_overlay(signedness_modifiers=False),
        _v_dot4_u32_u8_overlay(),
        _v_dot8_i32_i4_overlay(signedness_modifiers=False),
        _v_dot8_u32_u4_overlay(),
        *(_gfx950_ds_transpose_read_overlays() if include_ds_transpose_reads else ()),
        *matrix_overlays,
        _s_barrier_overlay(),
        *_gfx950_cache_control_overlays(),
        _s_waitcnt_overlay(
            effects=(
                _VMEM_LOAD_WAIT_EFFECT,
                _VMEM_STORE_WAIT_EFFECT,
                _LDS_WAIT_EFFECT,
                _SMEM_WAIT_EFFECT,
            ),
            lgkmcnt_immediate=_LGKMCNT_4BIT_IMMEDIATE,
        ),
    )


def _gfx940_core_overlays() -> tuple[AmdgpuDescriptorOverlay, ...]:
    return _cdna_core_overlays(
        global_load_lds_variants=_GLOBAL_LOAD_LDS_CDNA3_VARIANTS,
        buffer_load_lds_variants=_BUFFER_LOAD_LDS_CDNA3_VARIANTS,
        include_v_dot2_f32_bf16=False,
        include_v_cvt_pk_bf16_f32=False,
        include_ds_transpose_reads=False,
        matrix_overlays=(*_cdna3_mfma_overlays(), *_cdna3_smfmac_overlays()),
    )


def _gfx950_core_overlays() -> tuple[AmdgpuDescriptorOverlay, ...]:
    return _cdna_core_overlays(
        global_load_lds_variants=_GLOBAL_LOAD_LDS_GFX950_VARIANTS,
        buffer_load_lds_variants=_BUFFER_LOAD_LDS_GFX950_VARIANTS,
        include_v_dot2_f32_bf16=True,
        include_v_cvt_pk_bf16_f32=True,
        include_ds_transpose_reads=True,
        matrix_overlays=(*_cdna4_mfma_overlays(), *_cdna4_smfmac_overlays()),
    )


def _gfx940_core_overlay_descriptors(
    spec: AmdgpuIsaFactSource,
) -> tuple[Descriptor, ...]:
    return _with_execution_mask_state_reads(
        materialize_amdgpu_descriptor_overlays(spec, _gfx940_core_overlays())
    )


def _gfx950_core_overlay_descriptors(
    spec: AmdgpuIsaFactSource,
) -> tuple[Descriptor, ...]:
    return _with_execution_mask_state_reads(
        materialize_amdgpu_descriptor_overlays(spec, _gfx950_core_overlays())
    )


def _gfx11_core_overlays() -> tuple[AmdgpuDescriptorOverlay, ...]:
    return (
        _s_add_u32_overlay(),
        _s_add_u32_rhs_inline_overlay(),
        _s_addk_i32_overlay(),
        _s_addc_u32_overlay(),
        _s_sub_u32_overlay(),
        _s_sub_u32_rhs_inline_overlay(),
        _s_mul_i32_overlay(),
        _s_mul_i32_rhs_inline_overlay(),
        _s_mulk_i32_overlay(),
        _s_mul_hi_u32_overlay(),
        _s_min_i32_overlay(),
        _s_max_i32_overlay(),
        _s_min_u32_overlay(),
        _s_max_u32_overlay(),
        _s_cselect_b32_overlay(),
        *_s_cmp_i32_overlays(),
        *_s_cmp_u64_overlays(),
        _s_and_saveexec_b64_overlay("Nothas_lit_0_Nothas_lit_1"),
        _v_add_u32_overlay("V_ADD_NC_U32"),
        _v_add_u32_src0_inline_overlay("V_ADD_NC_U32"),
        _v_add_u32_literal_overlay("V_ADD_NC_U32"),
        _v_add_co_u32_overlay(),
        _v_add_co_ci_u32_overlay(),
        _v_sub_u32_overlay("V_SUB_NC_U32", "v_sub_nc_u32"),
        _v_mov_b32_literal_overlay(),
        _v_mov_b32_copy_overlay(),
        _v_mov_b32_dpp16_overlay(),
        _v_mul_lo_u32_overlay(),
        _v_mul_hi_u32_overlay(),
        _v_mul_u32_u24_overlay(),
        _v_mul_u32_u24_src0_inline_overlay(),
        _v_mul_u32_u24_literal_overlay(),
        _v_mad_u32_u24_overlay(),
        _v_mad_u32_u24_literal_overlay("src0"),
        _v_mad_u32_u24_literal_overlay("src1"),
        _v_mad_u32_u24_literal_overlay("src2"),
        _v_min_i32_overlay(),
        _v_max_i32_overlay(),
        _v_min_u32_overlay(),
        _v_max_u32_overlay(),
        _v_readfirstlane_b32_overlay(),
        *_integer_bitwise_shift_overlays(),
        *_integer_bitwise_permute_overlays(),
        *_v_binary_f32_overlays(),
        *_v_subrev_f32_overlays(),
        _v_fma_f32_overlay(),
        _v_fmaak_f32_overlay(),
        _v_fmac_f32_overlay(),
        _v_fmamk_f32_overlay(),
        *_rdna_scalar_fma_overlays(),
        _v_pk_fmac_f16_overlay(),
        _v_pk_fma_f16_overlay(include_literal_forms=True),
        *_v_pk_fma_f16_literal_overlays(),
        _v_pk_mad_i16_overlay(include_literal_forms=True),
        *_v_pk_mad_i16_literal_overlays(),
        _v_pk_mad_u16_overlay(include_literal_forms=True),
        *_v_pk_mad_u16_literal_overlays(),
        *_v_fma_mix_f32_overlays(op_sel_field="OP_SEL", op_sel_hi_field="OP_SEL_HI"),
        *_v_fma_mixlo_f16_overlays(op_sel_field="OP_SEL", op_sel_hi_field="OP_SEL_HI"),
        *_v_fma_mixhi_f16_overlays(op_sel_field="OP_SEL", op_sel_hi_field="OP_SEL_HI"),
        *_v_native_f32_math_overlays(),
        _v_sqrt_f32_overlay(),
        _v_rsq_f32_overlay(),
        _v_rcp_f32_overlay(),
        _v_cvt_f32_f16_overlay(),
        _v_cvt_f16_f32_overlay(),
        _v_cvt_pk_u16_u32_overlay(),
        _v_cvt_f32_i32_overlay(),
        _v_cvt_f32_u32_overlay(),
        _v_cvt_i32_f32_overlay(),
        _v_cvt_u32_f32_overlay(),
        *_v_cmp_overlays(),
        *_v_cndmask_b32_overlays(),
        *_s_load_dword_width_overlays(),
        *_s_load_dword_offset_only_overlays(fixed_soffset=_predefined("NULL")),
        *_s_buffer_load_width_overlays(),
        _buffer_load_dword_overlay(
            encoding_name="ENC_MUBUF",
            resource_field_name="SRSRC",
            cache_fields=_GFX9_11_VECTOR_CACHE_FIELDS,
        ),
        _buffer_load_dword_off_zero_overlay(
            encoding_name="ENC_MUBUF",
            resource_field_name="SRSRC",
            cache_fields=_GFX9_11_VECTOR_CACHE_FIELDS,
        ),
        _buffer_load_dword_vaddr_offset_overlay(
            encoding_name="ENC_MUBUF",
            resource_field_name="SRSRC",
            cache_fields=_GFX9_11_VECTOR_CACHE_FIELDS,
        ),
        _buffer_load_64_overlay(
            encoding_name="ENC_MUBUF",
            resource_field_name="SRSRC",
            cache_fields=_GFX9_11_VECTOR_CACHE_FIELDS,
        ),
        _buffer_load_64_off_zero_overlay(
            encoding_name="ENC_MUBUF",
            resource_field_name="SRSRC",
            cache_fields=_GFX9_11_VECTOR_CACHE_FIELDS,
        ),
        _buffer_load_64_vaddr_offset_overlay(
            encoding_name="ENC_MUBUF",
            resource_field_name="SRSRC",
            cache_fields=_GFX9_11_VECTOR_CACHE_FIELDS,
        ),
        _buffer_load_96_overlay(
            encoding_name="ENC_MUBUF",
            resource_field_name="SRSRC",
            cache_fields=_GFX9_11_VECTOR_CACHE_FIELDS,
        ),
        _buffer_load_96_off_zero_overlay(
            encoding_name="ENC_MUBUF",
            resource_field_name="SRSRC",
            cache_fields=_GFX9_11_VECTOR_CACHE_FIELDS,
        ),
        _buffer_load_96_vaddr_offset_overlay(
            encoding_name="ENC_MUBUF",
            resource_field_name="SRSRC",
            cache_fields=_GFX9_11_VECTOR_CACHE_FIELDS,
        ),
        _buffer_load_128_overlay(
            encoding_name="ENC_MUBUF",
            resource_field_name="SRSRC",
            cache_fields=_GFX9_11_VECTOR_CACHE_FIELDS,
        ),
        _buffer_load_128_off_zero_overlay(
            encoding_name="ENC_MUBUF",
            resource_field_name="SRSRC",
            cache_fields=_GFX9_11_VECTOR_CACHE_FIELDS,
        ),
        _buffer_load_128_vaddr_offset_overlay(
            encoding_name="ENC_MUBUF",
            resource_field_name="SRSRC",
            cache_fields=_GFX9_11_VECTOR_CACHE_FIELDS,
        ),
        _buffer_store_dword_overlay(
            encoding_name="ENC_MUBUF",
            resource_field_name="SRSRC",
            cache_fields=_GFX9_11_VECTOR_CACHE_FIELDS,
        ),
        _buffer_store_dword_off_zero_overlay(
            encoding_name="ENC_MUBUF",
            resource_field_name="SRSRC",
            cache_fields=_GFX9_11_VECTOR_CACHE_FIELDS,
        ),
        _buffer_store_dword_vaddr_offset_overlay(
            encoding_name="ENC_MUBUF",
            resource_field_name="SRSRC",
            cache_fields=_GFX9_11_VECTOR_CACHE_FIELDS,
        ),
        _buffer_store_64_overlay(
            encoding_name="ENC_MUBUF",
            resource_field_name="SRSRC",
            cache_fields=_GFX9_11_VECTOR_CACHE_FIELDS,
        ),
        _buffer_store_64_off_zero_overlay(
            encoding_name="ENC_MUBUF",
            resource_field_name="SRSRC",
            cache_fields=_GFX9_11_VECTOR_CACHE_FIELDS,
        ),
        _buffer_store_64_vaddr_offset_overlay(
            encoding_name="ENC_MUBUF",
            resource_field_name="SRSRC",
            cache_fields=_GFX9_11_VECTOR_CACHE_FIELDS,
        ),
        _buffer_store_96_overlay(
            encoding_name="ENC_MUBUF",
            resource_field_name="SRSRC",
            cache_fields=_GFX9_11_VECTOR_CACHE_FIELDS,
        ),
        _buffer_store_96_off_zero_overlay(
            encoding_name="ENC_MUBUF",
            resource_field_name="SRSRC",
            cache_fields=_GFX9_11_VECTOR_CACHE_FIELDS,
        ),
        _buffer_store_96_vaddr_offset_overlay(
            encoding_name="ENC_MUBUF",
            resource_field_name="SRSRC",
            cache_fields=_GFX9_11_VECTOR_CACHE_FIELDS,
        ),
        _buffer_store_128_overlay(
            encoding_name="ENC_MUBUF",
            resource_field_name="SRSRC",
            cache_fields=_GFX9_11_VECTOR_CACHE_FIELDS,
        ),
        _buffer_store_128_off_zero_overlay(
            encoding_name="ENC_MUBUF",
            resource_field_name="SRSRC",
            cache_fields=_GFX9_11_VECTOR_CACHE_FIELDS,
        ),
        _buffer_store_128_vaddr_offset_overlay(
            encoding_name="ENC_MUBUF",
            resource_field_name="SRSRC",
            cache_fields=_GFX9_11_VECTOR_CACHE_FIELDS,
        ),
        *_buffer_byte_memory_overlays(
            encoding_name="ENC_MUBUF",
            resource_field_name="SRSRC",
            cache_fields=_GFX9_11_VECTOR_CACHE_FIELDS,
        ),
        *_buffer_b16_memory_overlays(
            encoding_name="ENC_MUBUF",
            resource_field_name="SRSRC",
            cache_fields=_GFX9_11_VECTOR_CACHE_FIELDS,
        ),
        *_buffer_atomic_overlays(
            encoding_name="ENC_MUBUF",
            resource_field_name="SRSRC",
            offset_field_name="OFFSET",
            offset_bit_width=12,
            return_field_name="GLC",
            return_field_value=1,
            cache_fields=_GFX9_11_VECTOR_CACHE_FIELDS,
        ),
        *_global_memory_overlays(
            instruction_suffixes=_BYTE_MEMORY_INSTRUCTION_SUFFIXES,
            mnemonic_suffixes=_BYTE_MEMORY_MNEMONIC_SUFFIXES,
            encoding_name="ENC_FLAT_GLOBAL",
            address_field_name="ADDR",
            load_data_field_name="VDST",
            store_data_field_name="DATA",
            offset_field_name="OFFSET",
            offset_bit_width=13,
            saddr_off=_GLOBAL_SADDR_OFF,
            address_units=2,
            cache_fields=_GFX9_11_VECTOR_CACHE_FIELDS,
        ),
        *_global_byte_memory_overlays(
            encoding_name="ENC_FLAT_GLOBAL",
            address_field_name="ADDR",
            load_data_field_name="VDST",
            store_data_field_name="DATA",
            offset_field_name="OFFSET",
            offset_bit_width=13,
            saddr_off=_GLOBAL_SADDR_OFF,
            address_units=2,
            cache_fields=_GFX9_11_VECTOR_CACHE_FIELDS,
        ),
        *_global_b16_memory_overlays(
            encoding_name="ENC_FLAT_GLOBAL",
            address_field_name="ADDR",
            load_data_field_name="VDST",
            store_data_field_name="DATA",
            offset_field_name="OFFSET",
            offset_bit_width=13,
            saddr_off=_GLOBAL_SADDR_OFF,
            address_units=2,
            cache_fields=_GFX9_11_VECTOR_CACHE_FIELDS,
        ),
        *_global_memory_overlays(
            instruction_suffixes=_BYTE_MEMORY_INSTRUCTION_SUFFIXES,
            mnemonic_suffixes=_BYTE_MEMORY_MNEMONIC_SUFFIXES,
            encoding_name="ENC_FLAT_GLOBAL",
            address_field_name="ADDR",
            load_data_field_name="VDST",
            store_data_field_name="DATA",
            offset_field_name="OFFSET",
            offset_bit_width=13,
            saddr_off=None,
            address_units=1,
            descriptor_key_suffix="_saddr",
            cache_fields=_GFX9_11_VECTOR_CACHE_FIELDS,
        ),
        *_global_byte_memory_overlays(
            encoding_name="ENC_FLAT_GLOBAL",
            address_field_name="ADDR",
            load_data_field_name="VDST",
            store_data_field_name="DATA",
            offset_field_name="OFFSET",
            offset_bit_width=13,
            saddr_off=None,
            address_units=1,
            descriptor_key_suffix="_saddr",
            cache_fields=_GFX9_11_VECTOR_CACHE_FIELDS,
        ),
        *_global_b16_memory_overlays(
            encoding_name="ENC_FLAT_GLOBAL",
            address_field_name="ADDR",
            load_data_field_name="VDST",
            store_data_field_name="DATA",
            offset_field_name="OFFSET",
            offset_bit_width=13,
            saddr_off=None,
            address_units=1,
            descriptor_key_suffix="_saddr",
            cache_fields=_GFX9_11_VECTOR_CACHE_FIELDS,
        ),
        *_scratch_memory_overlays(
            instruction_suffixes=_BYTE_MEMORY_INSTRUCTION_SUFFIXES,
            mnemonic_suffixes=_BYTE_MEMORY_MNEMONIC_SUFFIXES,
            encoding_name="ENC_FLAT_SCRATCH",
            address_field_name="ADDR",
            load_data_field_name="VDST",
            store_data_field_name="DATA",
            offset_field_name="OFFSET",
            offset_bit_width=13,
            offset_signed=True,
            fixed_vaddr=None,
            fixed_saddr=_predefined("NULL"),
            implicit_flat_scratch=True,
            descriptor_key_suffix="_vaddr",
            cache_fields=_GFX9_11_VECTOR_CACHE_FIELDS,
        ),
        *_scratch_memory_overlays(
            instruction_suffixes=_BYTE_MEMORY_INSTRUCTION_SUFFIXES,
            mnemonic_suffixes=_BYTE_MEMORY_MNEMONIC_SUFFIXES,
            encoding_name="ENC_FLAT_SCRATCH",
            address_field_name="ADDR",
            load_data_field_name="VDST",
            store_data_field_name="DATA",
            offset_field_name="OFFSET",
            offset_bit_width=13,
            offset_signed=True,
            fixed_vaddr=0,
            fixed_saddr=_predefined("NULL"),
            implicit_flat_scratch=True,
            descriptor_key_suffix="_offset_only",
            cache_fields=_GFX9_11_VECTOR_CACHE_FIELDS,
        ),
        *_global_atomic_overlays(
            encoding_name="ENC_FLAT_GLOBAL",
            address_field_name="ADDR",
            data_field_name="DATA",
            offset_field_name="OFFSET",
            offset_bit_width=13,
            return_field_name="GLC",
            return_field_value=1,
            cache_fields=_GFX9_11_VECTOR_CACHE_FIELDS,
            saddr_off=None,
            address_units=1,
            descriptor_key_suffix="_saddr",
        ),
        *_flat_atomic_overlays(
            rows=_FLAT_ATOMIC_GFX11_ROWS,
            cmpswap_instruction_name="FLAT_ATOMIC_CMPSWAP_B32",
            encoding_name="ENC_FLAT",
            address_field_name="ADDR",
            data_field_name="DATA",
            offset_field_name="OFFSET",
            offset_bit_width=13,
            offset_signed=True,
            return_field_name="GLC",
            return_field_value=1,
            cache_fields=_GFX9_11_VECTOR_CACHE_FIELDS,
            implicit_flat_scratch=True,
        ),
        *_ds_memory_overlays(include_u16_d16_loads=True),
        *_ds_crosslane_overlays(),
        _v_dot2_f32_f16_overlay(),
        _v_dot2_f32_bf16_overlay(),
        _v_dot4_i32_i8_overlay(signedness_modifiers=True),
        _v_dot4_i32_iu8_overlay(lhs_signed=True, rhs_signed=False),
        _v_dot4_i32_iu8_overlay(lhs_signed=False, rhs_signed=True),
        _v_dot4_u32_u8_overlay(),
        _v_dot8_i32_i4_overlay(signedness_modifiers=True),
        _v_dot8_i32_iu4_overlay(lhs_signed=True, rhs_signed=False),
        _v_dot8_i32_iu4_overlay(lhs_signed=False, rhs_signed=True),
        _v_dot8_u32_u4_overlay(),
        *_with_zero_accumulator_form(_v_wmma_f32_16x16x16_f16_overlay(input_units=8)),
        *_with_zero_accumulator_form(_v_wmma_f32_16x16x16_bf16_overlay(input_units=8)),
        *_with_zero_accumulator_form(
            _v_wmma_f16_16x16x16_f16_overlay(input_units=8, accumulator_units=8)
        ),
        *_with_zero_accumulator_form(
            _v_wmma_bf16_16x16x16_bf16_overlay(input_units=8, accumulator_units=8)
        ),
        *_with_zero_accumulator_form(_v_wmma_i32_16x16x16_iu8_overlay(operand_units=4)),
        *_with_zero_accumulator_form(_v_wmma_i32_16x16x16_iu4_overlay(operand_units=2)),
        _s_barrier_overlay(),
        *_gfx11_cache_control_overlays(),
        _s_waitcnt_overlay(
            effects=(
                _VMEM_LOAD_WAIT_EFFECT,
                _LDS_WAIT_EFFECT,
                _SMEM_WAIT_EFFECT,
            )
        ),
        _s_waitcnt_vscnt_overlay(),
        _s_waitcnt_depctr_overlay(),
        _s_wait_idle_overlay(),
    )


def _gfx11_core_overlay_descriptors(
    spec: AmdgpuIsaFactSource,
) -> tuple[Descriptor, ...]:
    return _with_execution_mask_state_reads(
        materialize_amdgpu_descriptor_overlays(spec, _gfx11_core_overlays())
    )


def _gfx117x_core_overlays() -> tuple[AmdgpuDescriptorOverlay, ...]:
    return (*_gfx11_core_overlays(), *_rdna_scalar_domain_fma_overlays())


def _gfx117x_core_overlay_descriptors(
    spec: AmdgpuIsaFactSource,
) -> tuple[Descriptor, ...]:
    return _with_execution_mask_state_reads(
        materialize_amdgpu_descriptor_overlays(spec, _gfx117x_core_overlays())
    )


def _rdna4_core_overlays() -> tuple[AmdgpuDescriptorOverlay, ...]:
    return (
        _s_add_u32_overlay(),
        _s_add_u32_rhs_inline_overlay(),
        _s_addk_i32_overlay(instruction_name="S_ADDK_CO_I32", mnemonic="s_addk_co_i32"),
        _s_addc_u32_overlay(),
        _s_sub_u32_overlay(),
        _s_sub_u32_rhs_inline_overlay(),
        _s_mul_i32_overlay(),
        _s_mul_i32_rhs_inline_overlay(),
        _s_mulk_i32_overlay(),
        _s_mul_hi_u32_overlay(),
        _s_min_i32_overlay(),
        _s_max_i32_overlay(),
        _s_min_u32_overlay(),
        _s_max_u32_overlay(),
        _s_cselect_b32_overlay(),
        *_s_cmp_i32_overlays(),
        *_s_cmp_u64_overlays(),
        _s_and_saveexec_b64_overlay("Nothas_lit_0_Nothas_lit_1"),
        _v_add_u32_overlay("V_ADD_NC_U32"),
        _v_add_u32_src0_inline_overlay("V_ADD_NC_U32"),
        _v_add_u32_literal_overlay("V_ADD_NC_U32"),
        _v_add_co_u32_overlay(),
        _v_add_co_ci_u32_overlay(),
        _v_sub_u32_overlay("V_SUB_NC_U32", "v_sub_nc_u32"),
        _v_mov_b32_literal_overlay(),
        _v_mov_b32_copy_overlay(),
        _v_mov_b32_dpp16_overlay(),
        _v_mul_lo_u32_overlay(),
        _v_mul_hi_u32_overlay(),
        _v_mul_u32_u24_overlay(),
        _v_mul_u32_u24_src0_inline_overlay(),
        _v_mul_u32_u24_literal_overlay(),
        _v_mad_u32_u24_overlay(),
        _v_mad_u32_u24_literal_overlay("src0"),
        _v_mad_u32_u24_literal_overlay("src1"),
        _v_mad_u32_u24_literal_overlay("src2"),
        _v_min_i32_overlay(),
        _v_max_i32_overlay(),
        _v_min_u32_overlay(),
        _v_max_u32_overlay(),
        _v_readfirstlane_b32_overlay(),
        *_integer_bitwise_shift_overlays(),
        *_integer_bitwise_permute_overlays(),
        *_v_binary_f32_overlays(),
        *_v_subrev_f32_overlays(),
        _v_fma_f32_overlay(),
        _v_fmaak_f32_overlay(),
        _v_fmac_f32_overlay(),
        _v_fmamk_f32_overlay(),
        *_rdna_scalar_domain_fma_overlays(),
        *_rdna_scalar_fma_overlays(),
        _v_pk_fmac_f16_overlay(),
        _v_pk_fma_f16_overlay(include_literal_forms=True),
        *_v_pk_fma_f16_literal_overlays(),
        _v_pk_mad_i16_overlay(include_literal_forms=True),
        *_v_pk_mad_i16_literal_overlays(),
        _v_pk_mad_u16_overlay(include_literal_forms=True),
        *_v_pk_mad_u16_literal_overlays(),
        *_v_fma_mix_f32_overlays(),
        *_v_fma_mixlo_f16_overlays(),
        *_v_fma_mixhi_f16_overlays(),
        *_v_native_f32_math_overlays(),
        _v_sqrt_f32_overlay(),
        _v_rsq_f32_overlay(),
        _v_rcp_f32_overlay(),
        _v_cvt_f32_f16_overlay(),
        _v_cvt_f16_f32_overlay(),
        _v_cvt_pk_u16_u32_overlay(),
        _v_cvt_f32_i32_overlay(),
        _v_cvt_f32_u32_overlay(),
        _v_cvt_i32_f32_overlay(),
        _v_cvt_u32_f32_overlay(),
        *_v_cmp_overlays(),
        *_v_cndmask_b32_overlays(),
        *_s_load_dword_width_overlays("IOFFSET", offset_bit_width=24, include_b96=True),
        *_s_load_narrow_overlays("IOFFSET", offset_bit_width=24),
        *_s_load_dword_offset_only_overlays(
            "IOFFSET",
            offset_bit_width=24,
            fixed_soffset=_predefined("NULL"),
            include_b96=True,
        ),
        *_s_load_narrow_overlays(
            "IOFFSET", offset_bit_width=24, fixed_soffset=_predefined("NULL")
        ),
        *_s_buffer_load_width_overlays(
            "IOFFSET", offset_bit_width=24, include_b96=True
        ),
        *_s_buffer_load_narrow_overlays("IOFFSET", offset_bit_width=24),
        *_rdna4_vbuffer_dword_width_overlays(),
        *_buffer_byte_memory_overlays(
            encoding_name="ENC_VBUFFER",
            resource_field_name="RSRC",
            offset_field_name="IOFFSET",
            offset_bit_width=24,
            cache_fields=_GFX12_VECTOR_CACHE_FIELDS,
            include_off_zero=False,
            fixed_soffset=_VBUFFER_SOFFSET_NULL,
            fixed_soffset_native_spelling="null",
        ),
        *_buffer_b16_memory_overlays(
            encoding_name="ENC_VBUFFER",
            resource_field_name="RSRC",
            offset_field_name="IOFFSET",
            offset_bit_width=24,
            cache_fields=_GFX12_VECTOR_CACHE_FIELDS,
            include_off_zero=False,
            fixed_soffset=_VBUFFER_SOFFSET_NULL,
            fixed_soffset_native_spelling="null",
        ),
        *_buffer_atomic_overlays(
            encoding_name="ENC_VBUFFER",
            resource_field_name="RSRC",
            offset_field_name="IOFFSET",
            offset_bit_width=24,
            return_field_name="TH",
            return_field_value=_GFX12_TH_ATOMIC_RETURN_VALUE,
            cache_fields=_GFX12_VECTOR_CACHE_FIELDS,
            cache_immediate_field_names=_GFX12_ATOMIC_CACHE_IMMEDIATE_FIELDS,
            include_packed_half_add=True,
        ),
        *_global_memory_overlays(
            instruction_suffixes=_BYTE_MEMORY_INSTRUCTION_SUFFIXES,
            mnemonic_suffixes=_BYTE_MEMORY_MNEMONIC_SUFFIXES,
            encoding_name="ENC_VGLOBAL",
            address_field_name="VADDR",
            load_data_field_name="VDST",
            store_data_field_name="VSRC",
            offset_field_name="IOFFSET",
            offset_bit_width=24,
            saddr_off=_GLOBAL_SADDR_OFF,
            address_units=2,
            cache_fields=_GFX12_VECTOR_CACHE_FIELDS,
        ),
        *_global_byte_memory_overlays(
            encoding_name="ENC_VGLOBAL",
            address_field_name="VADDR",
            load_data_field_name="VDST",
            store_data_field_name="VSRC",
            offset_field_name="IOFFSET",
            offset_bit_width=24,
            saddr_off=_GLOBAL_SADDR_OFF,
            address_units=2,
            cache_fields=_GFX12_VECTOR_CACHE_FIELDS,
        ),
        *_global_b16_memory_overlays(
            encoding_name="ENC_VGLOBAL",
            address_field_name="VADDR",
            load_data_field_name="VDST",
            store_data_field_name="VSRC",
            offset_field_name="IOFFSET",
            offset_bit_width=24,
            saddr_off=_GLOBAL_SADDR_OFF,
            address_units=2,
            cache_fields=_GFX12_VECTOR_CACHE_FIELDS,
        ),
        *_global_memory_overlays(
            instruction_suffixes=_BYTE_MEMORY_INSTRUCTION_SUFFIXES,
            mnemonic_suffixes=_BYTE_MEMORY_MNEMONIC_SUFFIXES,
            encoding_name="ENC_VGLOBAL",
            address_field_name="VADDR",
            load_data_field_name="VDST",
            store_data_field_name="VSRC",
            offset_field_name="IOFFSET",
            offset_bit_width=24,
            saddr_off=None,
            address_units=1,
            descriptor_key_suffix="_saddr",
            cache_fields=_GFX12_VECTOR_CACHE_FIELDS,
        ),
        *_global_byte_memory_overlays(
            encoding_name="ENC_VGLOBAL",
            address_field_name="VADDR",
            load_data_field_name="VDST",
            store_data_field_name="VSRC",
            offset_field_name="IOFFSET",
            offset_bit_width=24,
            saddr_off=None,
            address_units=1,
            descriptor_key_suffix="_saddr",
            cache_fields=_GFX12_VECTOR_CACHE_FIELDS,
        ),
        *_global_b16_memory_overlays(
            encoding_name="ENC_VGLOBAL",
            address_field_name="VADDR",
            load_data_field_name="VDST",
            store_data_field_name="VSRC",
            offset_field_name="IOFFSET",
            offset_bit_width=24,
            saddr_off=None,
            address_units=1,
            descriptor_key_suffix="_saddr",
            cache_fields=_GFX12_VECTOR_CACHE_FIELDS,
        ),
        *_scratch_memory_overlays(
            instruction_suffixes=_BYTE_MEMORY_INSTRUCTION_SUFFIXES,
            mnemonic_suffixes=_BYTE_MEMORY_MNEMONIC_SUFFIXES,
            encoding_name="ENC_VSCRATCH",
            address_field_name="VADDR",
            load_data_field_name="VDST",
            store_data_field_name="VSRC",
            offset_field_name="IOFFSET",
            offset_bit_width=24,
            offset_signed=True,
            fixed_vaddr=None,
            fixed_saddr=_predefined("NULL"),
            descriptor_key_suffix="_vaddr",
            cache_fields=_GFX12_VECTOR_CACHE_FIELDS,
        ),
        *_scratch_memory_overlays(
            instruction_suffixes=_BYTE_MEMORY_INSTRUCTION_SUFFIXES,
            mnemonic_suffixes=_BYTE_MEMORY_MNEMONIC_SUFFIXES,
            encoding_name="ENC_VSCRATCH",
            address_field_name="VADDR",
            load_data_field_name="VDST",
            store_data_field_name="VSRC",
            offset_field_name="IOFFSET",
            offset_bit_width=24,
            offset_signed=True,
            fixed_vaddr=0,
            fixed_saddr=_predefined("NULL"),
            descriptor_key_suffix="_offset_only",
            cache_fields=_GFX12_VECTOR_CACHE_FIELDS,
        ),
        *_global_atomic_overlays(
            encoding_name="ENC_VGLOBAL",
            address_field_name="VADDR",
            data_field_name="VSRC",
            offset_field_name="IOFFSET",
            offset_bit_width=24,
            return_field_name="TH",
            return_field_value=_GFX12_TH_ATOMIC_RETURN_VALUE,
            cache_fields=_GFX12_VECTOR_CACHE_FIELDS,
            cache_immediate_field_names=_GFX12_ATOMIC_CACHE_IMMEDIATE_FIELDS,
            saddr_off=None,
            address_units=1,
            descriptor_key_suffix="_saddr",
            include_packed_half_add=True,
        ),
        *_flat_atomic_overlays(
            rows=_FLAT_ATOMIC_GFX12_ROWS,
            cmpswap_instruction_name="FLAT_ATOMIC_CMPSWAP_B32",
            encoding_name="ENC_VFLAT",
            address_field_name="VADDR",
            data_field_name="VSRC",
            offset_field_name="IOFFSET",
            offset_bit_width=24,
            offset_signed=True,
            return_field_name="TH",
            return_field_value=_GFX12_TH_ATOMIC_RETURN_VALUE,
            cache_fields=_GFX12_VECTOR_CACHE_FIELDS,
            cache_immediate_field_names=_GFX12_ATOMIC_CACHE_IMMEDIATE_FIELDS,
            implicit_flat_scratch=False,
        ),
        *_ds_memory_overlays(
            encoding_name="ENC_VDS",
            fixed_encoding_fields=(("OFFSET1", 0),),
            include_packed_half_atomic_add=True,
            include_u16_d16_loads=True,
        ),
        *_ds_crosslane_overlays(
            encoding_name="ENC_VDS",
            fixed_encoding_fields=(),
            include_fetch_invalid=True,
        ),
        _v_dot2_f32_f16_overlay(op_sel_hi_field="OPSEL_HI"),
        _v_dot2_f32_bf16_overlay(op_sel_hi_field="OPSEL_HI"),
        _v_dot4_i32_i8_overlay(op_sel_hi_field="OPSEL_HI", signedness_modifiers=True),
        _v_dot4_i32_iu8_overlay(
            op_sel_hi_field="OPSEL_HI", lhs_signed=True, rhs_signed=False
        ),
        _v_dot4_i32_iu8_overlay(
            op_sel_hi_field="OPSEL_HI", lhs_signed=False, rhs_signed=True
        ),
        _v_dot4_u32_u8_overlay(op_sel_hi_field="OPSEL_HI"),
        _v_dot8_i32_i4_overlay(op_sel_hi_field="OPSEL_HI", signedness_modifiers=True),
        _v_dot8_i32_iu4_overlay(
            op_sel_hi_field="OPSEL_HI", lhs_signed=True, rhs_signed=False
        ),
        _v_dot8_i32_iu4_overlay(
            op_sel_hi_field="OPSEL_HI", lhs_signed=False, rhs_signed=True
        ),
        _v_dot8_u32_u4_overlay(op_sel_hi_field="OPSEL_HI"),
        _v_dot4_f32_packed8_overlay(lhs_type="fp8", rhs_type="bf8"),
        _v_dot4_f32_packed8_overlay(lhs_type="bf8", rhs_type="fp8"),
        _v_dot4_f32_packed8_overlay(lhs_type="fp8", rhs_type="fp8"),
        _v_dot4_f32_packed8_overlay(lhs_type="bf8", rhs_type="bf8"),
        *_with_zero_accumulator_form(_v_wmma_f32_16x16x16_f16_overlay()),
        *_with_zero_accumulator_form(_v_wmma_f32_16x16x16_bf16_overlay()),
        *_with_zero_accumulator_form(_v_wmma_f16_16x16x16_f16_overlay()),
        *_with_zero_accumulator_form(_v_wmma_bf16_16x16x16_bf16_overlay()),
        *_with_zero_accumulator_form(_v_wmma_i32_16x16x16_iu8_overlay()),
        *_with_zero_accumulator_form(_v_wmma_i32_16x16x16_iu4_overlay()),
        *_with_zero_accumulator_form(
            _v_wmma_f32_16x16x16_packed8_overlay(lhs_type="fp8", rhs_type="fp8")
        ),
        *_with_zero_accumulator_form(
            _v_wmma_f32_16x16x16_packed8_overlay(lhs_type="fp8", rhs_type="bf8")
        ),
        *_with_zero_accumulator_form(
            _v_wmma_f32_16x16x16_packed8_overlay(lhs_type="bf8", rhs_type="fp8")
        ),
        *_with_zero_accumulator_form(
            _v_wmma_f32_16x16x16_packed8_overlay(lhs_type="bf8", rhs_type="bf8")
        ),
        *_with_zero_accumulator_form(_v_wmma_i32_16x16x32_iu4_overlay()),
        *_rdna4_swmmac_overlays(),
        *_gfx12_cache_control_overlays(),
        *_gfx12_prefetch_overlays(),
        _s_wait_loadcnt_overlay(),
        _s_wait_storecnt_overlay(),
        _s_wait_dscnt_overlay(),
        _s_wait_kmcnt_overlay(),
        _s_wait_alu_overlay(),
        _s_wait_idle_overlay(),
    )


def _gfx12_core_overlays() -> tuple[AmdgpuDescriptorOverlay, ...]:
    return _rdna4_core_overlays()


def _gfx12_core_overlay_descriptors(
    spec: AmdgpuIsaFactSource,
) -> tuple[Descriptor, ...]:
    return _with_execution_mask_state_reads(
        materialize_amdgpu_descriptor_overlays(spec, _gfx12_core_overlays())
    )


def _gfx1250_core_overlays() -> tuple[AmdgpuDescriptorOverlay, ...]:
    return _rdna4_core_overlays()


def _gfx1250_core_overlay_descriptors(
    spec: AmdgpuIsaFactSource,
) -> tuple[Descriptor, ...]:
    return _with_execution_mask_state_reads(
        materialize_amdgpu_descriptor_overlays(spec, _gfx1250_core_overlays())
    )


def _amdgpu_core_descriptor_set_bases() -> tuple[DescriptorSet, ...]:
    return (
        _AMDGPU_CDNA3_CORE_DESCRIPTOR_SET_BASE,
        _AMDGPU_CDNA4_CORE_DESCRIPTOR_SET_BASE,
        _AMDGPU_RDNA3_CORE_DESCRIPTOR_SET_BASE,
        _AMDGPU_RDNA3_5_CORE_DESCRIPTOR_SET_BASE,
        _AMDGPU_RDNA4_CORE_DESCRIPTOR_SET_BASE,
        _AMDGPU_RDNA4_GFX125X_CORE_DESCRIPTOR_SET_BASE,
    )


def _amdgpu_descriptor_ref_key_set() -> set[str]:
    keys: set[str] = set()
    keys.update(_MANUAL_SCALAR_DESCRIPTOR_KEYS)
    keys.update(descriptor.key for descriptor in _hal_buffer_descriptor_pseudos())
    for descriptor_set in _amdgpu_core_descriptor_set_bases():
        keys.update(descriptor.key for descriptor in descriptor_set.descriptors)
    for overlays in (
        _gfx940_core_overlays(),
        _gfx950_core_overlays(),
        _gfx11_core_overlays(),
        _gfx117x_core_overlays(),
        _gfx12_core_overlays(),
        _gfx1250_core_overlays(),
    ):
        keys.update(overlay.descriptor_key for overlay in overlays)
    return keys


__all__ = (
    "_AMDGPU_CDNA3_CORE_DESCRIPTOR_SET_BASE",
    "_AMDGPU_CDNA4_CORE_DESCRIPTOR_SET_BASE",
    "_AMDGPU_RDNA3_CORE_DESCRIPTOR_SET_BASE",
    "_AMDGPU_RDNA3_5_CORE_DESCRIPTOR_SET_BASE",
    "_AMDGPU_RDNA4_CORE_DESCRIPTOR_SET_BASE",
    "_AMDGPU_RDNA4_GFX125X_CORE_DESCRIPTOR_SET_BASE",
    "_amdgpu_core_descriptor_set_bases",
    "_amdgpu_descriptor_ref_key_set",
    "_cdna_core_overlays",
    "_gfx125x_reg_classes",
    "_gfx11_core_overlay_descriptors",
    "_gfx11_core_overlays",
    "_gfx117x_core_overlay_descriptors",
    "_gfx117x_core_overlays",
    "_gfx1250_core_overlay_descriptors",
    "_gfx1250_core_overlays",
    "_gfx12_core_overlay_descriptors",
    "_gfx12_core_overlays",
    "_gfx940_core_overlay_descriptors",
    "_gfx940_core_overlays",
    "_gfx950_core_overlay_descriptors",
    "_gfx950_core_overlays",
)
