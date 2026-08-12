# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


# ruff: noqa: F403, F405

"""AMDGPU core descriptor-set assembly."""

from __future__ import annotations

from dataclasses import replace
from functools import cache

from loom.target.arch.amdgpu.encoding import (
    AMDGPU_GFX125X_VOP3_SCALE_SEL_BIT_COUNT,
    AMDGPU_GFX125X_VOP3_SCALE_SEL_BIT_OFFSET,
)
from loom.target.arch.amdgpu.isa_xml import (
    AmdgpuIsaBitRange,
    AmdgpuIsaEncoding,
    AmdgpuIsaEncodingField,
    AmdgpuIsaFactSource,
    AmdgpuIsaFunctionalGroup,
    AmdgpuIsaInstruction,
    AmdgpuIsaInstructionEncoding,
    AmdgpuIsaInstructionFlags,
    AmdgpuIsaOperand,
)

from .alu import *
from .atomic import *
from .cdna import *
from .common import *
from .control import *
from .matrix import *
from .memory import *
from .rdna3 import *
from .rdna4 import *
from .rdna4m import *
from .rdna35 import *
from .scalar_float import *
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

_CDNA_FLAT_LOAD_MNEMONICS = (
    "flat_load_ubyte",
    "flat_load_sbyte",
    "flat_load_ushort",
    "flat_load_sshort",
    "flat_load_dword",
    "flat_load_dwordx2",
    "flat_load_dwordx3",
    "flat_load_dwordx4",
)
_CDNA_FLAT_STORE_MNEMONICS = (
    "flat_store_byte",
    "flat_store_short",
    "flat_store_dword",
    "flat_store_dwordx2",
    "flat_store_dwordx3",
    "flat_store_dwordx4",
)
_RDNA_FLAT_LOAD_MNEMONICS = (
    "flat_load_u8",
    "flat_load_i8",
    "flat_load_u16",
    "flat_load_i16",
    "flat_load_dword",
    "flat_load_dwordx2",
    "flat_load_dwordx3",
    "flat_load_dwordx4",
)
_RDNA_FLAT_STORE_MNEMONICS = (
    "flat_store_b8",
    "flat_store_b16",
    "flat_store_b32",
    "flat_store_b64",
    "flat_store_b96",
    "flat_store_b128",
)

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


def _amdgpu_descriptor_overlay_intersection(
    *members: tuple[AmdgpuDescriptorOverlay, ...],
) -> tuple[AmdgpuDescriptorOverlay, ...]:
    if not members:
        raise ValueError("AMDGPU generic descriptor overlay set has no members")
    member_maps = []
    for rows in members:
        rows_by_key = {row.descriptor_key: row for row in rows}
        if len(rows_by_key) != len(rows):
            raise ValueError("AMDGPU descriptor overlay set has duplicate keys")
        member_maps.append(rows_by_key)
    return tuple(
        row
        for row in members[0]
        if all(
            member_map.get(row.descriptor_key) == row for member_map in member_maps[1:]
        )
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
    packed8_source_semantics: str,
    global_load_lds_variants: tuple[tuple[str, str, str, int, int], ...],
    buffer_load_lds_variants: tuple[
        tuple[str, str, str, int, int, AmdgpuImplicitOperandOverlay], ...
    ],
    include_v_dot2_f32_bf16: bool,
    include_v_cvt_pk_bf16_f32: bool,
    include_v_cvt_scalef32_pk_packed8: bool,
    include_ds_transpose_reads: bool,
    matrix_overlays: tuple[AmdgpuDescriptorOverlay, ...],
) -> tuple[AmdgpuDescriptorOverlay, ...]:
    return (
        _s_add_u32_overlay(),
        _s_add_u32_rhs_inline_overlay(),
        _s_addk_i32_overlay(),
        _s_add_u32_rhs_symbol_rel32_lo_overlay(),
        _s_addc_u32_overlay(),
        _s_addc_u32_rhs_symbol_rel32_hi_overlay(),
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
        *_integer_bit_count_overlays(),
        _s_and_saveexec_b64_overlay("default"),
        _v_add_u32_overlay("V_ADD_U32"),
        _v_add_u32_src0_inline_overlay("V_ADD_U32"),
        _v_add_u32_literal_overlay("V_ADD_U32"),
        _v_add3_u32_overlay(include_literal_forms=False),
        _v_add_co_u32_overlay(),
        _v_add_co_ci_u32_overlay(
            instruction_name="V_ADDC_CO_U32", mnemonic="v_addc_co_u32"
        ),
        _v_sub_co_u32_overlay(),
        _v_sub_co_ci_u32_overlay(
            instruction_name="V_SUBB_CO_U32", mnemonic="v_subb_co_u32"
        ),
        _v_sub_u32_overlay("V_SUB_U32", "v_sub_u32"),
        _v_mov_b32_literal_overlay(),
        _v_mov_b32_copy_overlay(),
        _v_mov_b32_dpp_legacy_overlay(),
        _v_mov_b32_dpp_masked_legacy_overlay(),
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
        _v_readlane_b32_src1_inline_overlay(),
        *_integer_bitwise_shift_overlays(include_vop3_literal_forms=False),
        *_integer_bitwise_permute_overlays(include_vop3_literal_forms=False),
        *_v_binary_f32_overlays(),
        *_v_binary_f16_overlays(),
        _v_med3_num_f32_overlay(),
        *_v_binary_f32_dpp_legacy_overlays(),
        _v_fma_f32_overlay(),
        _v_fmaak_f32_overlay(),
        _v_fmac_f32_overlay(),
        _v_fmamk_f32_overlay(),
        *_cdna_scalar_fma_overlays(),
        _v_pk_fmac_f16_overlay(),
        _v_pk_fma_f16_overlay(),
        _v_pk_add_f16_overlay(),
        _v_pk_mul_f16_overlay(),
        _v_pk_minnum_f16_overlay(),
        _v_pk_maxnum_f16_overlay(),
        *_v_pk_i16_binary_overlays(),
        _v_pk_mad_i16_overlay(),
        _v_pk_mad_u16_overlay(),
        _v_pk_add_f32_overlay(),
        _v_pk_mul_f32_overlay(),
        _v_pk_fma_f32_overlay(),
        *_v_mad_mix_f32_overlays(op_sel_field="OP_SEL", op_sel_hi_field="OP_SEL_HI"),
        *_v_mad_mixlo_f16_overlays(op_sel_field="OP_SEL", op_sel_hi_field="OP_SEL_HI"),
        *_v_mad_mixhi_f16_overlays(op_sel_field="OP_SEL", op_sel_hi_field="OP_SEL_HI"),
        *_v_native_f32_math_overlays(),
        _v_sqrt_f32_overlay(),
        _v_rsq_f32_overlay(),
        _v_rcp_f32_overlay(),
        _v_div_scale_f32_overlay(),
        _v_div_fmas_f32_overlay(),
        _v_div_fixup_f32_overlay(),
        _v_cvt_f32_f16_overlay(),
        _v_cvt_f16_f32_overlay(),
        *_v_cvt_f32_packed8_overlays(packed8_source_semantics),
        *_v_cvt_pk_packed8_from_f32_overlays(
            packed8_source_semantics,
            op_sel_field="OP_SEL",
        ),
        *(
            _v_cvt_scalef32_pk_packed8_overlays()
            if include_v_cvt_scalef32_pk_packed8
            else ()
        ),
        _v_cvt_pk_u16_u32_overlay(),
        *((_v_cvt_pk_bf16_f32_overlay(),) if include_v_cvt_pk_bf16_f32 else ()),
        _v_cvt_f32_i32_overlay(),
        _v_cvt_f32_u32_overlay(),
        *_v_cvt_f32_ubyte_overlays(),
        _v_cvt_i32_f32_overlay(),
        _v_cvt_u32_f32_overlay(),
        *_v_cmp_overlays(),
        *_v_cmp_i32_equality_vcc_overlays(),
        *_v_cndmask_b32_overlays(include_literal_forms=False),
        _v_cndmask_b32_dpp_legacy_overlay(),
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
            narrow_store_mnemonic_suffixes=("byte", "short"),
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
            narrow_store_mnemonic_suffixes=("byte", "short"),
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
            cmpswap_mnemonic_suffix="x2",
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
        *_flat_memory_overlays(
            load_mnemonics=_CDNA_FLAT_LOAD_MNEMONICS,
            store_mnemonics=_CDNA_FLAT_STORE_MNEMONICS,
            encoding_name="ENC_FLAT",
            address_field_name="ADDR",
            load_data_field_name="VDST",
            store_data_field_name="DATA",
            offset_field_name="OFFSET",
            offset_bit_width=12,
            offset_signed=False,
            implicit_flat_scratch=True,
            implicit_m0=True,
            allow_accumulator_results=True,
            allow_accumulator_operands=True,
            cache_fields=_GFX950_VECTOR_CACHE_FIELDS,
        ),
        *_flat_atomic_overlays(
            rows=_FLAT_ATOMIC_GFX950_ROWS,
            cmpswap_instruction_name="FLAT_ATOMIC_CMPSWAP",
            cmpswap_mnemonic_suffix="x2",
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
        _s_sendmsg_overlay(),
        _s_sethalt_overlay(),
        _s_trap_overlay(),
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


@cache
def _gfx940_core_overlays() -> tuple[AmdgpuDescriptorOverlay, ...]:
    return _cdna_core_overlays(
        packed8_source_semantics="fnuz",
        global_load_lds_variants=_GLOBAL_LOAD_LDS_CDNA3_VARIANTS,
        buffer_load_lds_variants=_BUFFER_LOAD_LDS_CDNA3_VARIANTS,
        include_v_dot2_f32_bf16=False,
        include_v_cvt_pk_bf16_f32=False,
        include_v_cvt_scalef32_pk_packed8=False,
        include_ds_transpose_reads=False,
        matrix_overlays=(*_cdna3_mfma_overlays(), *_cdna3_smfmac_overlays()),
    )


@cache
def _gfx950_core_overlays() -> tuple[AmdgpuDescriptorOverlay, ...]:
    return _cdna_core_overlays(
        packed8_source_semantics="ocp",
        global_load_lds_variants=_GLOBAL_LOAD_LDS_GFX950_VARIANTS,
        buffer_load_lds_variants=_BUFFER_LOAD_LDS_GFX950_VARIANTS,
        include_v_dot2_f32_bf16=True,
        include_v_cvt_pk_bf16_f32=True,
        include_v_cvt_scalef32_pk_packed8=True,
        include_ds_transpose_reads=True,
        matrix_overlays=(*_cdna4_mfma_overlays(), *_cdna4_smfmac_overlays()),
    )


@cache
def _gfx9_4_generic_core_overlays() -> tuple[AmdgpuDescriptorOverlay, ...]:
    common_overlays = _amdgpu_descriptor_overlay_intersection(
        _gfx940_core_overlays(), _gfx950_core_overlays()
    )
    # ROCm's generic processor omits packed FP8/BF8 matrix operations because
    # gfx942 uses FNUZ operands while gfx950 uses OCP operands. Their packets
    # compare equal here, but the shared encodings do not have shared numeric
    # semantics.
    return tuple(
        overlay
        for overlay in common_overlays
        if not (
            overlay.semantic_tag is not None
            and overlay.semantic_tag.startswith("matrix.")
            and (".fp8" in overlay.semantic_tag or ".bf8" in overlay.semantic_tag)
        )
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


def _gfx9_4_generic_core_overlay_descriptors(
    spec: AmdgpuIsaFactSource,
) -> tuple[Descriptor, ...]:
    return _with_execution_mask_state_reads(
        materialize_amdgpu_descriptor_overlays(spec, _gfx9_4_generic_core_overlays())
    )


@cache
def _gfx11_core_overlays() -> tuple[AmdgpuDescriptorOverlay, ...]:
    return (
        _s_add_u32_overlay(),
        _s_add_u32_rhs_inline_overlay(),
        _s_addk_i32_overlay(),
        _s_add_u32_rhs_symbol_rel32_lo_overlay(),
        _s_addc_u32_overlay(),
        _s_addc_u32_rhs_symbol_rel32_hi_overlay(),
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
        *_rdna_integer_bit_count_overlays("Nothas_lit_0_Nothas_lit_1"),
        _s_and_saveexec_b64_overlay("Nothas_lit_0_Nothas_lit_1"),
        _v_add_u32_overlay("V_ADD_NC_U32"),
        _v_add_u32_src0_inline_overlay("V_ADD_NC_U32"),
        _v_add_u32_literal_overlay("V_ADD_NC_U32"),
        _v_add3_u32_overlay(),
        _v_add3_u32_literal_overlay("src0"),
        _v_add3_u32_literal_overlay("src1"),
        _v_add3_u32_literal_overlay("src2"),
        _v_add_co_u32_overlay(),
        _v_add_co_ci_u32_overlay(),
        _v_sub_co_u32_overlay(),
        _v_sub_co_ci_u32_overlay(),
        _v_sub_u32_overlay("V_SUB_NC_U32", "v_sub_nc_u32"),
        _v_mov_b32_literal_overlay(),
        _v_mov_b32_copy_overlay(),
        _v_mov_b32_dpp16_overlay(),
        _v_mov_b32_dpp16_masked_overlay(),
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
        _v_readlane_b32_src1_inline_overlay(),
        _v_readlane_b32_src1_sgpr_overlay(),
        *_integer_bitwise_shift_overlays(),
        *_integer_bitwise_permute_overlays(),
        _v_permlanex16_b32_src12_inline_overlay(),
        *_v_binary_f32_overlays(),
        *_v_binary_f16_overlays(),
        _v_med3_num_f32_overlay(),
        *_v_binary_f32_dpp16_overlays(),
        *_v_subrev_f32_overlays(),
        _v_fma_f32_overlay(),
        _v_fmaak_f32_overlay(),
        *_v_interp_overlays(),
        _v_fmac_f32_overlay(),
        _v_fmamk_f32_overlay(),
        *_rdna_scalar_fma_overlays(),
        _v_pk_fmac_f16_overlay(),
        _v_pk_fma_f16_overlay(include_literal_forms=True),
        *_v_pk_fma_f16_literal_overlays(),
        _v_pk_add_f16_overlay(),
        _v_pk_mul_f16_overlay(),
        _v_pk_minnum_f16_overlay(),
        _v_pk_maxnum_f16_overlay(),
        *_v_pk_i16_binary_overlays(),
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
        _v_div_scale_f32_overlay(),
        _v_div_fmas_f32_overlay(),
        _v_div_fixup_f32_overlay(),
        _v_cvt_f32_f16_overlay(encoding_name="ENC_VOP3"),
        _v_cvt_f16_f32_overlay(),
        _v_cvt_pk_u16_u32_overlay(),
        _v_cvt_pk_u16_u32_dpp16_overlay(),
        _v_cvt_f32_i32_overlay(),
        _v_cvt_f32_u32_overlay(),
        *_v_cvt_f32_ubyte_overlays(),
        _v_cvt_i32_f32_overlay(),
        _v_cvt_u32_f32_overlay(),
        *_v_cmp_overlays(),
        *_v_cmp_i32_equality_vcc_overlays(),
        *_v_cndmask_b32_overlays(),
        _v_cndmask_b32_dpp16_overlay(),
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
            include_d16_hi_loads=True,
        ),
        *_buffer_atomic_overlays(
            rows=_BUFFER_ATOMIC_GFX11_ROWS,
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
            include_d16_hi_loads=True,
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
            include_d16_hi_loads=True,
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
            rows=_GLOBAL_ATOMIC_GFX11_ROWS,
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
        *_flat_memory_overlays(
            load_mnemonics=_RDNA_FLAT_LOAD_MNEMONICS,
            store_mnemonics=_RDNA_FLAT_STORE_MNEMONICS,
            encoding_name="ENC_FLAT",
            address_field_name="ADDR",
            load_data_field_name="VDST",
            store_data_field_name="DATA",
            offset_field_name="OFFSET",
            offset_bit_width=13,
            offset_signed=True,
            implicit_flat_scratch=True,
            fixed_saddr=_predefined("NULL", "OPR_SREG"),
            cache_fields=_GFX9_11_VECTOR_CACHE_FIELDS,
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
        *_with_zero_accumulator_form(
            _v_wmma_f32_16x16x16_f16_overlay(
                descriptor_key_suffix=".w64",
                low_mnemonic_suffix="_w64",
                input_units=8,
                accumulator_units=4,
                accumulator_size_exception_reason=_WMMA_GFX11_WAVE64_ACCUMULATOR_SIZE_REASON,
            )
        ),
        *_with_zero_accumulator_form(_v_wmma_f32_16x16x16_bf16_overlay(input_units=8)),
        *_with_zero_accumulator_form(
            _v_wmma_f32_16x16x16_bf16_overlay(
                descriptor_key_suffix=".w64",
                low_mnemonic_suffix="_w64",
                input_units=8,
                accumulator_units=4,
                accumulator_size_exception_reason=_WMMA_GFX11_WAVE64_ACCUMULATOR_SIZE_REASON,
            )
        ),
        *_with_zero_accumulator_form(
            _v_wmma_f16_16x16x16_f16_overlay(input_units=8, accumulator_units=8)
        ),
        *_with_zero_accumulator_form(
            _v_wmma_f16_16x16x16_f16_overlay(
                descriptor_key_suffix=".w64",
                low_mnemonic_suffix="_w64",
                input_units=8,
                accumulator_units=4,
                accumulator_size_exception_reason=_WMMA_GFX11_WAVE64_ACCUMULATOR_SIZE_REASON,
            )
        ),
        *_with_zero_accumulator_form(
            _v_wmma_bf16_16x16x16_bf16_overlay(input_units=8, accumulator_units=8)
        ),
        *_with_zero_accumulator_form(
            _v_wmma_bf16_16x16x16_bf16_overlay(
                descriptor_key_suffix=".w64",
                low_mnemonic_suffix="_w64",
                input_units=8,
                accumulator_units=4,
                accumulator_size_exception_reason=_WMMA_GFX11_WAVE64_ACCUMULATOR_SIZE_REASON,
            )
        ),
        *_with_zero_accumulator_form(
            _v_wmma_i32_16x16x16_iu8_overlay(
                operand_units=4,
                mirrors_sign_select_to_high_halves=True,
            )
        ),
        *_with_zero_accumulator_form(
            _v_wmma_i32_16x16x16_iu8_overlay(
                descriptor_key_suffix=".w64",
                low_mnemonic_suffix="_w64",
                operand_units=4,
                accumulator_units=4,
                accumulator_size_exception_reason=_WMMA_GFX11_WAVE64_ACCUMULATOR_SIZE_REASON,
                mirrors_sign_select_to_high_halves=True,
            )
        ),
        *_with_zero_accumulator_form(
            _v_wmma_i32_16x16x16_iu4_overlay(
                operand_units=2,
                mirrors_sign_select_to_high_halves=True,
            )
        ),
        *_with_zero_accumulator_form(
            _v_wmma_i32_16x16x16_iu4_overlay(
                descriptor_key_suffix=".w64",
                low_mnemonic_suffix="_w64",
                operand_units=2,
                accumulator_units=4,
                accumulator_size_exception_reason=_WMMA_GFX11_WAVE64_ACCUMULATOR_SIZE_REASON,
                mirrors_sign_select_to_high_halves=True,
            )
        ),
        _s_barrier_overlay(),
        _s_sendmsg_overlay(),
        _s_sendmsg_rtn_b32_overlay(),
        _s_sethalt_overlay(),
        _s_trap_overlay(),
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


@cache
def _gfx115x_core_overlays() -> tuple[AmdgpuDescriptorOverlay, ...]:
    return (
        *_gfx11_core_overlays(),
        *_s_float_arithmetic_overlays(),
        *_s_float_compare_overlays(),
        *_s_float_conversion_overlays(),
        *_rdna_scalar_domain_fma_overlays(),
        *_rdna35_dpp16_f32_uniform_rhs_overlays(),
        *_rdna35_dpp8_f32_uniform_rhs_overlays(),
        *_rdna35_dpp_integer_compare_uniform_rhs_overlays(),
    )


def _gfx115x_core_overlay_descriptors(
    spec: AmdgpuIsaFactSource,
) -> tuple[Descriptor, ...]:
    return _with_execution_mask_state_reads(
        materialize_amdgpu_descriptor_overlays(spec, _gfx115x_core_overlays())
    )


def _gfx12_wmma_16x16_overlays(
    *, wave_size: int, op_sel_hi_field: str
) -> tuple[AmdgpuDescriptorOverlay, ...]:
    if wave_size not in (32, 64):
        raise ValueError(f"unsupported WMMA wave size {wave_size}")
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

    overlays = (
        _v_wmma_f32_16x16x16_f16_overlay(
            descriptor_key_suffix=descriptor_key_suffix,
            low_mnemonic_suffix=low_mnemonic_suffix,
            input_units=units(4),
            accumulator_units=units(8),
            input_size_exception_reason=size_exception_reason(4),
            accumulator_size_exception_reason=size_exception_reason(8),
            op_sel_hi_field=op_sel_hi_field,
        ),
        _v_wmma_f32_16x16x16_bf16_overlay(
            descriptor_key_suffix=descriptor_key_suffix,
            low_mnemonic_suffix=low_mnemonic_suffix,
            input_units=units(4),
            accumulator_units=units(8),
            input_size_exception_reason=size_exception_reason(4),
            accumulator_size_exception_reason=size_exception_reason(8),
            op_sel_hi_field=op_sel_hi_field,
        ),
        _v_wmma_f16_16x16x16_f16_overlay(
            descriptor_key_suffix=descriptor_key_suffix,
            low_mnemonic_suffix=low_mnemonic_suffix,
            input_units=units(4),
            accumulator_units=units(4),
            input_size_exception_reason=size_exception_reason(4),
            accumulator_size_exception_reason=size_exception_reason(4),
            op_sel_hi_field=op_sel_hi_field,
        ),
        _v_wmma_bf16_16x16x16_bf16_overlay(
            descriptor_key_suffix=descriptor_key_suffix,
            low_mnemonic_suffix=low_mnemonic_suffix,
            input_units=units(4),
            accumulator_units=units(4),
            input_size_exception_reason=size_exception_reason(4),
            accumulator_size_exception_reason=size_exception_reason(4),
            op_sel_hi_field=op_sel_hi_field,
        ),
        _v_wmma_i32_16x16x16_iu8_overlay(
            descriptor_key_suffix=descriptor_key_suffix,
            low_mnemonic_suffix=low_mnemonic_suffix,
            operand_units=units(2),
            accumulator_units=units(8),
            operand_size_exception_reason=size_exception_reason(2),
            accumulator_size_exception_reason=size_exception_reason(8),
            op_sel_hi_field=op_sel_hi_field,
        ),
        _v_wmma_i32_16x16x16_iu4_overlay(
            descriptor_key_suffix=descriptor_key_suffix,
            low_mnemonic_suffix=low_mnemonic_suffix,
            operand_units=units(1),
            accumulator_units=units(8),
            operand_size_exception_reason=size_exception_reason(1),
            accumulator_size_exception_reason=size_exception_reason(8),
            op_sel_hi_field=op_sel_hi_field,
        ),
        *(
            _v_wmma_f32_16x16x16_packed8_overlay(
                lhs_type=lhs_type,
                rhs_type=rhs_type,
                descriptor_key_suffix=descriptor_key_suffix,
                low_mnemonic_suffix=low_mnemonic_suffix,
                input_units=units(2),
                accumulator_units=units(8),
                input_size_exception_reason=size_exception_reason(2),
                accumulator_size_exception_reason=size_exception_reason(8),
                op_sel_hi_field=op_sel_hi_field,
            )
            for lhs_type, rhs_type in (
                ("fp8", "fp8"),
                ("fp8", "bf8"),
                ("bf8", "fp8"),
                ("bf8", "bf8"),
            )
        ),
        _v_wmma_i32_16x16x32_iu4_overlay(
            descriptor_key_suffix=descriptor_key_suffix,
            low_mnemonic_suffix=low_mnemonic_suffix,
            operand_units=units(2),
            accumulator_units=units(8),
            operand_size_exception_reason=size_exception_reason(2),
            accumulator_size_exception_reason=size_exception_reason(8),
            op_sel_hi_field=op_sel_hi_field,
        ),
    )
    return tuple(
        variant
        for overlay in overlays
        for variant in _with_zero_accumulator_form(overlay)
    )


def _rdna4m_minmax_overlays() -> tuple[AmdgpuDescriptorOverlay, ...]:
    numeric_f32_overlays = (
        _v_min_f32_overlay(instruction_name="V_MIN_NUM_F32", mnemonic="v_min_num_f32"),
        _v_min_f32_literal_overlay(
            instruction_name="V_MIN_NUM_F32", mnemonic="v_min_num_f32"
        ),
        _v_min_f32_src0_inline_overlay(
            instruction_name="V_MIN_NUM_F32", mnemonic="v_min_num_f32"
        ),
        _v_max_f32_overlay(instruction_name="V_MAX_NUM_F32", mnemonic="v_max_num_f32"),
        _v_max_f32_literal_overlay(
            instruction_name="V_MAX_NUM_F32", mnemonic="v_max_num_f32"
        ),
        _v_max_f32_src0_inline_overlay(
            instruction_name="V_MAX_NUM_F32", mnemonic="v_max_num_f32"
        ),
        *(
            _v_binary_f32_dpp_overlay(
                descriptor_key=f"amdgpu.v_{operation}_f32.dpp16",
                instruction_name=f"V_{operation.upper()}_NUM_F32",
                mnemonic=f"v_{operation}_num_f32",
                semantic_tag=f"float.{semantic}.f32",
                encoding_name="VOP2_VOP_DPP16",
                encoding_condition="has_dpp16",
            )
            for operation, semantic in (("min", "minnum"), ("max", "maxnum"))
        ),
        *(
            overlay
            for operation, instruction_name, semantic in (
                ("min", "V_MIN_NUM_F32", "minnum"),
                ("max", "V_MAX_NUM_F32", "maxnum"),
            )
            for overlay in _v_binary_f32_dpp16_uniform_rhs_overlays(
                operation=operation,
                instruction_name=instruction_name,
                low_mnemonic=f"v_{operation}_f32",
                semantic=semantic,
            )
        ),
        *(
            overlay
            for operation, instruction_name, semantic in (
                ("min", "V_MIN_NUM_F32", "minnum"),
                ("max", "V_MAX_NUM_F32", "maxnum"),
            )
            for overlay in _v_binary_f32_dpp8_uniform_rhs_overlays(
                operation=operation,
                instruction_name=instruction_name,
                low_mnemonic=f"v_{operation}_f32",
                semantic=semantic,
            )
        ),
    )
    numeric_scalar_overlays = (
        *(
            _v_commutative_binary_f16_overlay(
                descriptor_key=f"amdgpu.v_{operation}_f16",
                instruction_name=f"V_{operation.upper()}_NUM_F16",
                mnemonic=f"v_{operation}_num_f16",
                semantic_tag=f"float.{semantic}.f16",
            )
            for operation, semantic in (("min", "minnum"), ("max", "maxnum"))
        ),
        *(
            _v_commutative_binary_vop3_float_overlay(
                descriptor_key=f"amdgpu.v_{operation}_f64",
                instruction_name=f"V_{operation.upper()}_NUM_F64",
                mnemonic=f"v_{operation}_num_f64",
                semantic_tag=f"float.{semantic}.f64",
                element_bit_width=64,
            )
            for operation, semantic in (("min", "minnum"), ("max", "maxnum"))
        ),
    )
    ieee_scalar_overlays = tuple(
        _v_commutative_binary_vop3_float_overlay(
            descriptor_key=f"amdgpu.v_{operation}_{type_suffix}",
            instruction_name=f"V_{operation.upper()}_{type_suffix.upper()}",
            mnemonic=f"v_{operation}_{type_suffix}",
            semantic_tag=f"float.{operation}.{type_suffix}",
            element_bit_width=element_bit_width,
        )
        for type_suffix, element_bit_width in (("f16", 16), ("f32", 32), ("f64", 64))
        for operation in ("minimum", "maximum")
    )
    ternary_overlays = tuple(
        _v_ternary_float_overlay(
            descriptor_key=f"amdgpu.v_{operation}_{type_suffix}",
            instruction_name=f"V_{operation.upper()}_{type_suffix.upper()}",
            mnemonic=f"v_{operation}_{type_suffix}",
            semantic_tag=f"float.{operation}.{type_suffix}",
            element_bit_width=element_bit_width,
        )
        for type_suffix, element_bit_width in (("f16", 16), ("f32", 32))
        for operation in (
            "min3_num",
            "max3_num",
            "med3_num",
            "minmax_num",
            "maxmin_num",
            "minimum3",
            "maximum3",
            "minimummaximum",
            "maximumminimum",
        )
    )
    return (
        *numeric_f32_overlays,
        *numeric_scalar_overlays,
        _v_pk_minnum_f16_overlay(
            instruction_name="V_PK_MIN_NUM_F16", mnemonic="v_pk_min_num_f16"
        ),
        _v_pk_maxnum_f16_overlay(
            instruction_name="V_PK_MAX_NUM_F16", mnemonic="v_pk_max_num_f16"
        ),
        *ieee_scalar_overlays,
        _v_pk_minimum_f16_overlay(),
        _v_pk_maximum_f16_overlay(),
        *ternary_overlays,
    )


@cache
def _rdna4m_core_overlays() -> tuple[AmdgpuDescriptorOverlay, ...]:
    minmax_overlays = _rdna4m_minmax_overlays()
    minmax_descriptor_keys = {overlay.descriptor_key for overlay in minmax_overlays}
    return (
        *(
            overlay
            for overlay in _gfx115x_core_overlays()
            if not (overlay.semantic_tag or "").startswith("matrix.")
            and overlay.descriptor_key not in minmax_descriptor_keys
        ),
        *minmax_overlays,
        *_v_cvt_f32_packed8_selection_overlays("ocp", op_sel_field="OP_SEL"),
        *_v_cvt_pk_packed8_from_f32_overlays("ocp", op_sel_field="OP_SEL"),
        *(
            _v_dot4_f32_packed8_overlay(lhs_type=lhs_type, rhs_type=rhs_type)
            for lhs_type, rhs_type in (
                ("fp8", "bf8"),
                ("bf8", "fp8"),
                ("fp8", "fp8"),
                ("bf8", "bf8"),
            )
        ),
        *_gfx12_wmma_16x16_overlays(wave_size=32, op_sel_hi_field="OP_SEL_HI"),
        *_gfx12_wmma_16x16_overlays(wave_size=64, op_sel_hi_field="OP_SEL_HI"),
        *_v_swmmac_16x16_overlays(wave_size=32, op_sel_hi_field="OP_SEL_HI"),
        *_v_swmmac_16x16_overlays(wave_size=64, op_sel_hi_field="OP_SEL_HI"),
    )


def _rdna4m_core_overlay_descriptors(
    spec: AmdgpuIsaFactSource,
) -> tuple[Descriptor, ...]:
    spec = _rdna4m_spec_with_supplemental_instruction_facts(spec)
    return _with_execution_mask_state_reads(
        materialize_amdgpu_descriptor_overlays(spec, _rdna4m_core_overlays())
    )


@cache
def _rdna4_core_overlays() -> tuple[AmdgpuDescriptorOverlay, ...]:
    return (
        _s_add_u32_overlay(),
        _s_add_u32_rhs_inline_overlay(),
        _s_addk_i32_overlay(instruction_name="S_ADDK_CO_I32", mnemonic="s_addk_co_i32"),
        _s_add_u32_rhs_symbol_rel32_lo_overlay(),
        _s_addc_u32_overlay(),
        _s_addc_u32_rhs_symbol_rel32_hi_overlay(),
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
        *_rdna_integer_bit_count_overlays("Nothas_lit_0_Nothas_lit_1"),
        _s_and_saveexec_b64_overlay("Nothas_lit_0_Nothas_lit_1"),
        _v_add_u32_overlay("V_ADD_NC_U32"),
        _v_add_u32_src0_inline_overlay("V_ADD_NC_U32"),
        _v_add_u32_literal_overlay("V_ADD_NC_U32"),
        _v_add3_u32_overlay(),
        _v_add3_u32_literal_overlay("src0"),
        _v_add3_u32_literal_overlay("src1"),
        _v_add3_u32_literal_overlay("src2"),
        _v_add_co_u32_overlay(),
        _v_add_co_ci_u32_overlay(),
        _v_sub_co_u32_overlay(),
        _v_sub_co_ci_u32_overlay(),
        _v_sub_u32_overlay("V_SUB_NC_U32", "v_sub_nc_u32"),
        _v_mov_b32_literal_overlay(),
        _v_mov_b32_copy_overlay(),
        _v_mov_b32_dpp16_overlay(),
        _v_mov_b32_dpp16_masked_overlay(),
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
        _v_readlane_b32_src1_inline_overlay(),
        *_integer_bitwise_shift_overlays(),
        *_integer_bitwise_permute_overlays(),
        _v_permlanex16_b32_src12_inline_overlay(),
        *_v_binary_f32_overlays(),
        *_v_binary_f16_overlays(),
        _v_med3_num_f32_overlay(
            instruction_name="V_MED3_NUM_F32",
            mnemonic="v_med3_num_f32",
        ),
        *_v_binary_f32_dpp16_overlays(),
        *_v_subrev_f32_overlays(),
        *_s_float_arithmetic_overlays(),
        *_s_float_compare_overlays(),
        *_s_float_conversion_overlays(),
        _v_fma_f32_overlay(),
        _v_fmaak_f32_overlay(),
        *_v_interp_overlays(op_sel_field="OPSEL"),
        _v_fmac_f32_overlay(),
        _v_fmamk_f32_overlay(),
        *_rdna_scalar_domain_fma_overlays(),
        *_rdna_scalar_fma_overlays(),
        _v_pk_fmac_f16_overlay(),
        *_v_pk_with_op_sel_hi_field(
            (
                _v_pk_fma_f16_overlay(include_literal_forms=True),
                *_v_pk_fma_f16_literal_overlays(),
                _v_pk_add_f16_overlay(),
                _v_pk_mul_f16_overlay(),
                _v_pk_minnum_f16_overlay(
                    instruction_name="V_PK_MIN_NUM_F16",
                    mnemonic="v_pk_min_num_f16",
                ),
                _v_pk_maxnum_f16_overlay(
                    instruction_name="V_PK_MAX_NUM_F16",
                    mnemonic="v_pk_max_num_f16",
                ),
                _v_pk_minimum_f16_overlay(),
                _v_pk_maximum_f16_overlay(),
                *_v_pk_i16_binary_overlays(),
                _v_pk_mad_i16_overlay(include_literal_forms=True),
                *_v_pk_mad_i16_literal_overlays(),
                _v_pk_mad_u16_overlay(include_literal_forms=True),
                *_v_pk_mad_u16_literal_overlays(),
            ),
            "OPSEL_HI",
        ),
        *_v_fma_mix_f32_overlays(),
        *_v_fma_mixlo_f16_overlays(),
        *_v_fma_mixhi_f16_overlays(),
        *_v_native_f32_math_overlays(),
        _v_sqrt_f32_overlay(),
        _v_rsq_f32_overlay(),
        _v_rcp_f32_overlay(),
        _v_div_scale_f32_overlay(),
        _v_div_fmas_f32_overlay(),
        _v_div_fixup_f32_overlay(),
        _v_cvt_f32_f16_overlay(encoding_name="ENC_VOP3"),
        _v_cvt_f16_f32_overlay(),
        *_v_cvt_f32_packed8_selection_overlays("ocp", op_sel_field="OPSEL"),
        *_v_cvt_pk_packed8_from_f32_overlays(
            "ocp",
            op_sel_field="OPSEL",
        ),
        _v_cvt_pk_u16_u32_overlay(),
        _v_cvt_pk_u16_u32_dpp16_overlay(),
        _v_cvt_f32_i32_overlay(),
        _v_cvt_f32_u32_overlay(),
        *_v_cvt_f32_ubyte_overlays(),
        _v_cvt_i32_f32_overlay(),
        _v_cvt_u32_f32_overlay(),
        *_v_cmp_overlays(),
        *_v_cmp_i32_equality_vcc_overlays(),
        *_v_cndmask_b32_overlays(),
        _v_cndmask_b32_dpp16_overlay(),
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
            rows=_BUFFER_ATOMIC_GFX12_ROWS,
            encoding_name="ENC_VBUFFER",
            resource_field_name="RSRC",
            offset_field_name="IOFFSET",
            offset_bit_width=24,
            return_field_name="TH",
            return_field_value=_GFX12_TH_ATOMIC_RETURN_VALUE,
            cache_fields=_GFX12_VECTOR_CACHE_FIELDS,
            cache_immediate_field_names=_GFX12_ATOMIC_CACHE_IMMEDIATE_FIELDS,
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
            rows=_GLOBAL_ATOMIC_GFX12_ROWS,
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
        ),
        *_flat_memory_overlays(
            load_mnemonics=_RDNA_FLAT_LOAD_MNEMONICS,
            store_mnemonics=_RDNA_FLAT_STORE_MNEMONICS,
            encoding_name="ENC_VFLAT",
            address_field_name="VADDR",
            load_data_field_name="VDST",
            store_data_field_name="VSRC",
            offset_field_name="IOFFSET",
            offset_bit_width=24,
            offset_signed=True,
            implicit_flat_scratch=False,
            cache_fields=_GFX12_VECTOR_CACHE_FIELDS,
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
        _v_dot4_f32_packed8_overlay(
            lhs_type="fp8", rhs_type="bf8", op_sel_hi_field="OPSEL_HI"
        ),
        _v_dot4_f32_packed8_overlay(
            lhs_type="bf8", rhs_type="fp8", op_sel_hi_field="OPSEL_HI"
        ),
        _v_dot4_f32_packed8_overlay(
            lhs_type="fp8", rhs_type="fp8", op_sel_hi_field="OPSEL_HI"
        ),
        _v_dot4_f32_packed8_overlay(
            lhs_type="bf8", rhs_type="bf8", op_sel_hi_field="OPSEL_HI"
        ),
        *_gfx12_wmma_16x16_overlays(wave_size=32, op_sel_hi_field="OPSEL_HI"),
        *_gfx12_wmma_16x16_overlays(wave_size=64, op_sel_hi_field="OPSEL_HI"),
        *_v_swmmac_16x16_overlays(wave_size=32, op_sel_hi_field="OPSEL_HI"),
        *_v_swmmac_16x16_overlays(wave_size=64, op_sel_hi_field="OPSEL_HI"),
        _s_barrier_signal_all_overlay(),
        _s_barrier_wait_all_overlay(),
        _s_sendmsg_overlay(),
        _s_sendmsg_rtn_b32_overlay(),
        _s_sethalt_overlay(),
        _s_trap_overlay(),
        *_gfx12_cache_control_overlays(),
        *_gfx12_prefetch_overlays(),
        _s_wait_loadcnt_overlay(),
        _s_wait_storecnt_overlay(),
        _s_wait_dscnt_overlay(),
        _s_wait_kmcnt_overlay(),
        _s_wait_alu_overlay(),
        _s_wait_idle_overlay(),
    )


@cache
def _gfx12_core_overlays() -> tuple[AmdgpuDescriptorOverlay, ...]:
    return _rdna4_core_overlays()


def _gfx12_core_overlay_descriptors(
    spec: AmdgpuIsaFactSource,
) -> tuple[Descriptor, ...]:
    return _with_execution_mask_state_reads(
        materialize_amdgpu_descriptor_overlays(spec, _gfx12_core_overlays())
    )


@cache
def _gfx125x_core_overlays() -> tuple[AmdgpuDescriptorOverlay, ...]:
    return (
        *(
            overlay
            for overlay in _rdna4_core_overlays()
            if not (overlay.semantic_tag or "").startswith(
                ("float.interpolation.", "matrix.wmma.")
            )
        ),
        _s_getreg_b32_cluster_workgroup_flat_id_overlay(),
        *_v_cvt_f16_packed8_byte_overlays("ocp"),
        *_v_cvt_pk_f16_packed8_overlays("ocp"),
        *_v_cvt_pk_packed8_from_f16_overlays(
            "ocp",
            op_sel_field="OPSEL",
        ),
        *_v_cvt_scale_pk8_overlays(),
        _v_cvt_pk_bf16_f32_overlay(),
        _v_cvt_pk_bf16_f32_dpp16_overlay(),
        *_v_pk_with_op_sel_hi_field(
            (
                _v_pk_add_bf16_overlay(),
                _v_pk_mul_bf16_overlay(),
                _v_pk_fma_bf16_overlay(),
            ),
            "OPSEL_HI",
        ),
    )


_GFX125X_SUPPLEMENTAL_INSTRUCTION_FLAGS = AmdgpuIsaInstructionFlags(
    is_branch=False,
    is_conditional_branch=False,
    is_indirect_branch=False,
    is_program_terminator=False,
    is_immediately_executed=False,
)


def _gfx125x_supplemental_bit_range(
    bit_offset: int, bit_count: int
) -> AmdgpuIsaBitRange:
    return AmdgpuIsaBitRange(order=0, bit_offset=bit_offset, bit_count=bit_count)


def _gfx125x_supplemental_encoding_field(
    name: str, *ranges: AmdgpuIsaBitRange
) -> AmdgpuIsaEncodingField:
    return AmdgpuIsaEncodingField(name=name, is_conditional=False, ranges=ranges)


def _gfx125x_vop3_scale_sel_field() -> AmdgpuIsaEncodingField:
    return _gfx125x_supplemental_encoding_field(
        "SCALE_SEL",
        _gfx125x_supplemental_bit_range(
            AMDGPU_GFX125X_VOP3_SCALE_SEL_BIT_OFFSET,
            AMDGPU_GFX125X_VOP3_SCALE_SEL_BIT_COUNT,
        ),
    )


def _gfx125x_supplement_encoding(
    encoding: AmdgpuIsaEncoding,
) -> AmdgpuIsaEncoding:
    if encoding.name != "ENC_VOP3":
        return encoding
    if any(field.name == "SCALE_SEL" for field in encoding.fields):
        return encoding
    return replace(
        encoding,
        fields=(*encoding.fields, _gfx125x_vop3_scale_sel_field()),
    )


def _gfx125x_spec_with_supplemental_encoding_facts(
    spec: AmdgpuIsaFactSource,
) -> AmdgpuIsaFactSource:
    encodings = tuple(
        _gfx125x_supplement_encoding(encoding) for encoding in spec.encodings
    )
    if encodings == spec.encodings:
        return spec
    return replace(spec, encodings=encodings)


def _gfx125x_supplemental_vop1_source(
    field_name: str,
    data_format_name: str,
    *,
    operand_type: str = "OPR_VGPR",
) -> AmdgpuIsaOperand:
    return AmdgpuIsaOperand(
        order=2,
        field_name=field_name,
        data_format_name=data_format_name,
        operand_type=operand_type,
        size_bits=32,
        is_input=True,
        is_output=False,
        is_implicit=False,
        is_binary_microcode_required=True,
    )


def _gfx125x_supplemental_vop1_result(
    data_format_name: str,
    *,
    size_bits: int = 32,
) -> AmdgpuIsaOperand:
    return AmdgpuIsaOperand(
        order=1,
        field_name="VDST",
        data_format_name=data_format_name,
        operand_type="OPR_VGPR",
        size_bits=size_bits,
        is_input=False,
        is_output=True,
        is_implicit=False,
        is_binary_microcode_required=True,
    )


def _gfx125x_supplemental_vop3_source(
    order: int,
    field_name: str,
    data_format_name: str,
    *,
    size_bits: int = 32,
    operand_type: str = "OPR_SRC",
) -> AmdgpuIsaOperand:
    return AmdgpuIsaOperand(
        order=order,
        field_name=field_name,
        data_format_name=data_format_name,
        operand_type=operand_type,
        size_bits=size_bits,
        is_input=True,
        is_output=False,
        is_implicit=False,
        is_binary_microcode_required=True,
    )


def _gfx125x_supplemental_vop3_result(
    data_format_name: str,
    *,
    size_bits: int = 32,
) -> AmdgpuIsaOperand:
    return AmdgpuIsaOperand(
        order=1,
        field_name="VDST",
        data_format_name=data_format_name,
        operand_type="OPR_VGPR",
        size_bits=size_bits,
        is_input=False,
        is_output=True,
        is_implicit=False,
        is_binary_microcode_required=True,
    )


def _gfx125x_supplemental_instruction(
    *,
    name: str,
    encoding_name: str,
    opcode: int,
    operands: tuple[AmdgpuIsaOperand, ...],
    additional_encodings: tuple[AmdgpuIsaInstructionEncoding, ...] = (),
) -> AmdgpuIsaInstruction:
    return AmdgpuIsaInstruction(
        name=name,
        aliases=(),
        flags=_GFX125X_SUPPLEMENTAL_INSTRUCTION_FLAGS,
        encodings=(
            AmdgpuIsaInstructionEncoding(
                encoding_name=encoding_name,
                condition_name="default",
                opcode=opcode,
                operands=operands,
            ),
            *additional_encodings,
        ),
        functional_groups=(AmdgpuIsaFunctionalGroup("VALU", ("NOT_ASSIGNED",)),),
    )


_GFX125X_CVT_SCALE_PK8_ROWS = (
    ("V_CVT_SCALE_PK8_F16_FP4", 0x29F, 32, "FMT_NUM_PK2_F16", 128),
    ("V_CVT_SCALE_PK8_BF16_FP4", 0x2A0, 32, "FMT_NUM_PK2_BF16", 128),
    ("V_CVT_SCALE_PK8_F32_FP4", 0x2A1, 32, "FMT_NUM_F32", 256),
    ("V_CVT_SCALE_PK8_F16_FP8", 0x2A8, 64, "FMT_NUM_PK2_F16", 128),
    ("V_CVT_SCALE_PK8_BF16_FP8", 0x2A9, 64, "FMT_NUM_PK2_BF16", 128),
    ("V_CVT_SCALE_PK8_F32_FP8", 0x2AA, 64, "FMT_NUM_F32", 256),
    ("V_CVT_SCALE_PK8_F16_BF8", 0x2AB, 64, "FMT_NUM_PK2_F16", 128),
    ("V_CVT_SCALE_PK8_BF16_BF8", 0x2AC, 64, "FMT_NUM_PK2_BF16", 128),
    ("V_CVT_SCALE_PK8_F32_BF8", 0x2AD, 64, "FMT_NUM_F32", 256),
)


def _gfx125x_supplemental_cvt_scale_pk8_instruction(
    name: str,
    opcode: int,
    source_bits: int,
    result_format_name: str,
    result_bits: int,
) -> AmdgpuIsaInstruction:
    return _gfx125x_supplemental_instruction(
        name=name,
        encoding_name="ENC_VOP3",
        opcode=opcode,
        operands=(
            _gfx125x_supplemental_vop3_result(
                result_format_name,
                size_bits=result_bits,
            ),
            _gfx125x_supplemental_vop3_source(
                2,
                "SRC0",
                "FMT_NUM_UINT",
                size_bits=source_bits,
            ),
            _gfx125x_supplemental_vop3_source(3, "SRC1", "FMT_NUM_UINT"),
        ),
    )


_GFX125X_SUPPLEMENTAL_INSTRUCTIONS = (
    *(
        _gfx125x_supplemental_cvt_scale_pk8_instruction(*row)
        for row in _GFX125X_CVT_SCALE_PK8_ROWS
    ),
    _gfx125x_supplemental_instruction(
        name="V_CVT_F16_FP8",
        encoding_name="ENC_VOP1",
        opcode=0x77,
        operands=(
            _gfx125x_supplemental_vop1_result("FMT_NUM_F16", size_bits=16),
            _gfx125x_supplemental_vop1_source(
                "SRC0",
                "FMT_NUM_UINT",
                operand_type="OPR_SRC",
            ),
        ),
        additional_encodings=(
            AmdgpuIsaInstructionEncoding(
                encoding_name="ENC_VOP3",
                condition_name="default",
                opcode=0x1F7,
                operands=(
                    _gfx125x_supplemental_vop3_result(
                        "FMT_NUM_F16",
                        size_bits=16,
                    ),
                    _gfx125x_supplemental_vop3_source(
                        2,
                        "SRC0",
                        "FMT_NUM_UINT",
                        size_bits=32,
                    ),
                ),
            ),
        ),
    ),
    _gfx125x_supplemental_instruction(
        name="V_CVT_F16_BF8",
        encoding_name="ENC_VOP1",
        opcode=0x78,
        operands=(
            _gfx125x_supplemental_vop1_result("FMT_NUM_F16", size_bits=16),
            _gfx125x_supplemental_vop1_source(
                "SRC0",
                "FMT_NUM_UINT",
                operand_type="OPR_SRC",
            ),
        ),
        additional_encodings=(
            AmdgpuIsaInstructionEncoding(
                encoding_name="ENC_VOP3",
                condition_name="default",
                opcode=0x1F8,
                operands=(
                    _gfx125x_supplemental_vop3_result(
                        "FMT_NUM_F16",
                        size_bits=16,
                    ),
                    _gfx125x_supplemental_vop3_source(
                        2,
                        "SRC0",
                        "FMT_NUM_UINT",
                        size_bits=32,
                    ),
                ),
            ),
        ),
    ),
    _gfx125x_supplemental_instruction(
        name="V_CVT_PK_F16_FP8",
        encoding_name="ENC_VOP1_VGPR",
        opcode=0xEB,
        operands=(
            _gfx125x_supplemental_vop1_result("FMT_NUM_PK2_F16"),
            _gfx125x_supplemental_vop1_source("VSRC0", "FMT_NUM_UINT"),
        ),
    ),
    _gfx125x_supplemental_instruction(
        name="V_CVT_PK_F16_BF8",
        encoding_name="ENC_VOP1_VGPR",
        opcode=0xED,
        operands=(
            _gfx125x_supplemental_vop1_result("FMT_NUM_PK2_F16"),
            _gfx125x_supplemental_vop1_source("VSRC0", "FMT_NUM_UINT"),
        ),
    ),
    _gfx125x_supplemental_instruction(
        name="V_CVT_PK_FP8_F16",
        encoding_name="ENC_VOP3",
        opcode=0x372,
        operands=(
            _gfx125x_supplemental_vop3_result(
                "FMT_NUM_PK2_FP8",
                size_bits=16,
            ),
            _gfx125x_supplemental_vop3_source(
                2,
                "SRC0",
                "FMT_NUM_PK2_F16",
            ),
        ),
    ),
    _gfx125x_supplemental_instruction(
        name="V_CVT_PK_BF8_F16",
        encoding_name="ENC_VOP3",
        opcode=0x373,
        operands=(
            _gfx125x_supplemental_vop3_result(
                "FMT_NUM_PK2_BF8",
                size_bits=16,
            ),
            _gfx125x_supplemental_vop3_source(
                2,
                "SRC0",
                "FMT_NUM_PK2_F16",
            ),
        ),
    ),
    _gfx125x_supplemental_instruction(
        name="V_CVT_PK_BF16_F32",
        encoding_name="ENC_VOP3",
        opcode=877,
        operands=(
            _gfx125x_supplemental_vop3_result("FMT_NUM_PK2_BF16"),
            _gfx125x_supplemental_vop3_source(2, "SRC0", "FMT_NUM_F32"),
            _gfx125x_supplemental_vop3_source(3, "SRC1", "FMT_NUM_F32"),
        ),
        additional_encodings=(
            AmdgpuIsaInstructionEncoding(
                encoding_name="VOP3_VOP_DPP16",
                condition_name="has_dpp16",
                opcode=877,
                operands=(
                    _gfx125x_supplemental_vop3_result("FMT_NUM_PK2_BF16"),
                    _gfx125x_supplemental_vop3_source(
                        2,
                        "VSRC0",
                        "FMT_NUM_F32",
                        operand_type="OPR_VGPR",
                    ),
                    _gfx125x_supplemental_vop3_source(
                        3,
                        "SRC1",
                        "FMT_NUM_F32",
                        operand_type="OPR_SRC_VGPR",
                    ),
                ),
            ),
        ),
    ),
    _gfx125x_supplemental_instruction(
        name="V_PK_ADD_BF16",
        encoding_name="ENC_VOP3P",
        opcode=0x23,
        operands=(
            _gfx125x_supplemental_vop3_result("FMT_NUM_PK2_BF16"),
            _gfx125x_supplemental_vop3_source(2, "SRC0", "FMT_NUM_PK2_BF16"),
            _gfx125x_supplemental_vop3_source(3, "SRC1", "FMT_NUM_PK2_BF16"),
        ),
    ),
    _gfx125x_supplemental_instruction(
        name="V_PK_MUL_BF16",
        encoding_name="ENC_VOP3P",
        opcode=0x2A,
        operands=(
            _gfx125x_supplemental_vop3_result("FMT_NUM_PK2_BF16"),
            _gfx125x_supplemental_vop3_source(2, "SRC0", "FMT_NUM_PK2_BF16"),
            _gfx125x_supplemental_vop3_source(3, "SRC1", "FMT_NUM_PK2_BF16"),
        ),
    ),
    _gfx125x_supplemental_instruction(
        name="V_PK_FMA_BF16",
        encoding_name="ENC_VOP3P",
        opcode=0x11,
        operands=(
            _gfx125x_supplemental_vop3_result("FMT_NUM_PK2_BF16"),
            _gfx125x_supplemental_vop3_source(2, "SRC0", "FMT_NUM_PK2_BF16"),
            _gfx125x_supplemental_vop3_source(3, "SRC1", "FMT_NUM_PK2_BF16"),
            _gfx125x_supplemental_vop3_source(4, "SRC2", "FMT_NUM_PK2_BF16"),
        ),
    ),
)


def _gfx125x_spec_with_supplemental_instruction_facts(
    spec: AmdgpuIsaFactSource,
) -> AmdgpuIsaFactSource:
    existing_instruction_names = spec.instruction_map(include_aliases=True)
    supplemental_instructions = tuple(
        instruction
        for instruction in _GFX125X_SUPPLEMENTAL_INSTRUCTIONS
        if instruction.name not in existing_instruction_names
    )
    if not supplemental_instructions:
        return spec

    # The pinned RDNA4 ISA XML lacks these gfx125x instructions even though the
    # ROCm LLVM assembler accepts them. Keep the supplemental facts narrow to
    # gfx125x and let normal overlay materialization validate each descriptor.
    return replace(spec, instructions=(*spec.instructions, *supplemental_instructions))


def _gfx125x_core_overlay_descriptors(
    spec: AmdgpuIsaFactSource,
) -> tuple[Descriptor, ...]:
    spec = _gfx125x_spec_with_supplemental_encoding_facts(spec)
    spec = _gfx125x_spec_with_supplemental_instruction_facts(spec)
    descriptors = materialize_amdgpu_descriptor_overlays(spec, _gfx125x_core_overlays())
    return _with_execution_mask_state_reads(
        _with_gfx125x_inherited_matrix_schedules(descriptors)
    )


def _gfx1251_core_overlay_projection(
    descriptors: tuple[Descriptor, ...],
) -> tuple[Descriptor, ...]:
    return _with_xdl_latency_tiers(descriptors)


def _gfx1250_a0_core_overlay_projection(
    descriptors: tuple[Descriptor, ...],
) -> tuple[Descriptor, ...]:
    return _with_gfx1250_a0_matrix_schedules(descriptors)


@cache
def _gfx12_5_generic_core_overlays() -> tuple[AmdgpuDescriptorOverlay, ...]:
    return tuple(
        overlay
        for overlay in _amdgpu_descriptor_overlay_intersection(_gfx125x_core_overlays())
        if not (overlay.semantic_tag or "").startswith("matrix.swmmac.")
    )


def _gfx12_5_generic_core_overlay_descriptors(
    spec: AmdgpuIsaFactSource,
) -> tuple[Descriptor, ...]:
    spec = _gfx125x_spec_with_supplemental_encoding_facts(spec)
    spec = _gfx125x_spec_with_supplemental_instruction_facts(spec)
    descriptors = materialize_amdgpu_descriptor_overlays(
        spec, _gfx12_5_generic_core_overlays()
    )
    return _with_execution_mask_state_reads(
        _with_xdl_latency_tiers(
            _with_gfx125x_inherited_matrix_schedules(descriptors),
            regular_schedule_class=_SCHEDULE_MATRIX_XDL_ESTIMATED_16,
            slow_schedule_class=_SCHEDULE_MATRIX_XDL_ESTIMATED_32,
        )
    )


def _amdgpu_core_descriptor_set_bases() -> tuple[DescriptorSet, ...]:
    return (
        _AMDGPU_CDNA3_CORE_DESCRIPTOR_SET_BASE,
        _AMDGPU_CDNA4_CORE_DESCRIPTOR_SET_BASE,
        _AMDGPU_GFX9_4_GENERIC_CORE_DESCRIPTOR_SET_BASE,
        _AMDGPU_GFX11_GENERIC_CORE_DESCRIPTOR_SET_BASE,
        _AMDGPU_GFX12_GENERIC_CORE_DESCRIPTOR_SET_BASE,
        _AMDGPU_GFX12_5_GENERIC_CORE_DESCRIPTOR_SET_BASE,
        _AMDGPU_RDNA3_CORE_DESCRIPTOR_SET_BASE,
        _AMDGPU_RDNA3_5_CORE_DESCRIPTOR_SET_BASE,
        _AMDGPU_RDNA4M_CORE_DESCRIPTOR_SET_BASE,
        _AMDGPU_RDNA4_CORE_DESCRIPTOR_SET_BASE,
        _AMDGPU_RDNA4_GFX1250_A0_CORE_DESCRIPTOR_SET_BASE,
        _AMDGPU_RDNA4_GFX1251_CORE_DESCRIPTOR_SET_BASE,
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
        _gfx115x_core_overlays(),
        _rdna4m_core_overlays(),
        _gfx12_core_overlays(),
        _gfx125x_core_overlays(),
    ):
        keys.update(overlay.descriptor_key for overlay in overlays)
    return keys


__all__ = (
    "_AMDGPU_CDNA3_CORE_DESCRIPTOR_SET_BASE",
    "_AMDGPU_CDNA4_CORE_DESCRIPTOR_SET_BASE",
    "_AMDGPU_GFX9_4_GENERIC_CORE_DESCRIPTOR_SET_BASE",
    "_AMDGPU_GFX11_GENERIC_CORE_DESCRIPTOR_SET_BASE",
    "_AMDGPU_GFX12_GENERIC_CORE_DESCRIPTOR_SET_BASE",
    "_AMDGPU_GFX12_5_GENERIC_CORE_DESCRIPTOR_SET_BASE",
    "_AMDGPU_RDNA3_CORE_DESCRIPTOR_SET_BASE",
    "_AMDGPU_RDNA3_5_CORE_DESCRIPTOR_SET_BASE",
    "_AMDGPU_RDNA4M_CORE_DESCRIPTOR_SET_BASE",
    "_AMDGPU_RDNA4_CORE_DESCRIPTOR_SET_BASE",
    "_AMDGPU_RDNA4_GFX1250_A0_CORE_DESCRIPTOR_SET_BASE",
    "_AMDGPU_RDNA4_GFX1251_CORE_DESCRIPTOR_SET_BASE",
    "_AMDGPU_RDNA4_GFX125X_CORE_DESCRIPTOR_SET_BASE",
    "_amdgpu_core_descriptor_set_bases",
    "_amdgpu_descriptor_ref_key_set",
    "_cdna_core_overlays",
    "_gfx125x_reg_classes",
    "_gfx11_core_overlay_descriptors",
    "_gfx11_core_overlays",
    "_gfx115x_core_overlay_descriptors",
    "_gfx115x_core_overlays",
    "_gfx12_5_generic_core_overlay_descriptors",
    "_gfx12_5_generic_core_overlays",
    "_gfx125x_core_overlay_descriptors",
    "_gfx125x_core_overlays",
    "_gfx1250_a0_core_overlay_projection",
    "_gfx1251_core_overlay_projection",
    "_gfx12_core_overlay_descriptors",
    "_gfx12_core_overlays",
    "_rdna4m_core_overlay_descriptors",
    "_rdna4m_core_overlays",
    "_rdna4m_minmax_overlays",
    "_gfx940_core_overlay_descriptors",
    "_gfx940_core_overlays",
    "_gfx950_core_overlay_descriptors",
    "_gfx950_core_overlays",
    "_gfx9_4_generic_core_overlay_descriptors",
    "_gfx9_4_generic_core_overlays",
)
