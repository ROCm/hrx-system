# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


# ruff: noqa: F403, F405

"""XML-free AMDGPU descriptor projections for source-to-low contracts."""

from __future__ import annotations

from .alu import *
from .categories import *
from .common import *
from .matrix import *
from .sets import *


def _contract_overlay_builders_from_overlays(
    overlays: Sequence[AmdgpuDescriptorOverlay],
) -> dict[str, Callable[[], AmdgpuDescriptorOverlay]]:
    def _overlay_builder(
        overlay: AmdgpuDescriptorOverlay,
    ) -> Callable[[], AmdgpuDescriptorOverlay]:
        def build() -> AmdgpuDescriptorOverlay:
            return overlay

        return build

    return {overlay.descriptor_key: _overlay_builder(overlay) for overlay in overlays}


_AMDGPU_CONTRACT_DESCRIPTOR_OVERLAY_BUILDERS: dict[
    str,
    Callable[[], AmdgpuDescriptorOverlay],
] = {
    "amdgpu.s_mov_b32": _s_mov_b32_contract_overlay,
    "amdgpu.s_add_u32": _s_add_u32_overlay,
    "amdgpu.s_sub_u32": _s_sub_u32_overlay,
    "amdgpu.s_mul_i32": _s_mul_i32_overlay,
    "amdgpu.s_mul_hi_u32": _s_mul_hi_u32_overlay,
    "amdgpu.s_min_i32": _s_min_i32_overlay,
    "amdgpu.s_max_i32": _s_max_i32_overlay,
    "amdgpu.s_min_u32": _s_min_u32_overlay,
    "amdgpu.s_max_u32": _s_max_u32_overlay,
    "amdgpu.s_cselect_b32": _s_cselect_b32_overlay,
    "amdgpu.v_mov_b32": _v_mov_b32_literal_overlay,
    "amdgpu.v_add_u32": lambda: _v_add_u32_overlay("V_ADD_NC_U32"),
    "amdgpu.v_add_u32.src0_inline": lambda: _v_add_u32_src0_inline_overlay(
        "V_ADD_NC_U32"
    ),
    "amdgpu.v_add_u32.lit": lambda: _v_add_u32_literal_overlay("V_ADD_NC_U32"),
    "amdgpu.v_sub_u32": lambda: _v_sub_u32_overlay("V_SUB_NC_U32", "v_sub_nc_u32"),
    "amdgpu.v_mul_lo_u32": _v_mul_lo_u32_overlay,
    "amdgpu.v_mul_hi_u32": _v_mul_hi_u32_overlay,
    "amdgpu.v_mul_u32_u24": _v_mul_u32_u24_overlay,
    "amdgpu.v_mul_u32_u24.src0_inline": _v_mul_u32_u24_src0_inline_overlay,
    "amdgpu.v_mul_u32_u24.lit": _v_mul_u32_u24_literal_overlay,
    "amdgpu.v_mad_u32_u24": _v_mad_u32_u24_overlay,
    "amdgpu.v_mad_u32_u24.src0_lit": lambda: _v_mad_u32_u24_literal_overlay("src0"),
    "amdgpu.v_mad_u32_u24.src1_lit": lambda: _v_mad_u32_u24_literal_overlay("src1"),
    "amdgpu.v_mad_u32_u24.src2_lit": lambda: _v_mad_u32_u24_literal_overlay("src2"),
    "amdgpu.v_min_i32": _v_min_i32_overlay,
    "amdgpu.v_max_i32": _v_max_i32_overlay,
    "amdgpu.v_min_u32": _v_min_u32_overlay,
    "amdgpu.v_max_u32": _v_max_u32_overlay,
    "amdgpu.v_add_f32": _v_add_f32_overlay,
    "amdgpu.v_add_f32.lit": _v_add_f32_literal_overlay,
    "amdgpu.v_add_f32.src0_inline": _v_add_f32_src0_inline_overlay,
    "amdgpu.v_sub_f32": _v_sub_f32_overlay,
    "amdgpu.v_sub_f32.lit": _v_sub_f32_literal_overlay,
    "amdgpu.v_sub_f32.src0_inline": _v_sub_f32_src0_inline_overlay,
    "amdgpu.v_mul_f32": _v_mul_f32_overlay,
    "amdgpu.v_mul_f32.lit": _v_mul_f32_literal_overlay,
    "amdgpu.v_mul_f32.src0_inline": _v_mul_f32_src0_inline_overlay,
    "amdgpu.v_min_f32": _v_min_f32_overlay,
    "amdgpu.v_min_f32.lit": _v_min_f32_literal_overlay,
    "amdgpu.v_min_f32.src0_inline": _v_min_f32_src0_inline_overlay,
    "amdgpu.v_max_f32": _v_max_f32_overlay,
    "amdgpu.v_max_f32.lit": _v_max_f32_literal_overlay,
    "amdgpu.v_max_f32.src0_inline": _v_max_f32_src0_inline_overlay,
    "amdgpu.v_fma_f32": _v_fma_f32_overlay,
    "amdgpu.v_fmaak_f32": _v_fmaak_f32_overlay,
    "amdgpu.v_fmac_f32": _v_fmac_f32_overlay,
    "amdgpu.v_fmamk_f32": _v_fmamk_f32_overlay,
    "amdgpu.v_pk_fmac_f16": _v_pk_fmac_f16_overlay,
    "amdgpu.v_pk_fma_f16": _v_pk_fma_f16_overlay,
    **_contract_overlay_builders_from_overlays(_v_pk_fma_f16_literal_overlays()),
    "amdgpu.v_pk_mad_i16": _v_pk_mad_i16_overlay,
    **_contract_overlay_builders_from_overlays(_v_pk_mad_i16_literal_overlays()),
    "amdgpu.v_pk_mad_u16": _v_pk_mad_u16_overlay,
    **_contract_overlay_builders_from_overlays(_v_pk_mad_u16_literal_overlays()),
    "amdgpu.v_pk_fma_f32": _v_pk_fma_f32_overlay,
    **_contract_overlay_builders_from_overlays(_v_fma_mix_f32_overlays()),
    **_contract_overlay_builders_from_overlays(_v_fma_mixlo_f16_overlays()),
    **_contract_overlay_builders_from_overlays(_v_fma_mixhi_f16_overlays()),
    **_contract_overlay_builders_from_overlays(_v_mad_mix_f32_overlays()),
    **_contract_overlay_builders_from_overlays(_v_mad_mixlo_f16_overlays()),
    **_contract_overlay_builders_from_overlays(_v_mad_mixhi_f16_overlays()),
    "amdgpu.v_exp_f32": _v_exp_f32_overlay,
    "amdgpu.v_log_f32": _v_log_f32_overlay,
    "amdgpu.v_sin_f32": _v_sin_f32_overlay,
    "amdgpu.v_cos_f32": _v_cos_f32_overlay,
    "amdgpu.v_sqrt_f32": _v_sqrt_f32_overlay,
    "amdgpu.v_rsq_f32": _v_rsq_f32_overlay,
    "amdgpu.v_rcp_f32": _v_rcp_f32_overlay,
    "amdgpu.v_cvt_f32_f16": _v_cvt_f32_f16_overlay,
    "amdgpu.v_cvt_f16_f32": _v_cvt_f16_f32_overlay,
    "amdgpu.v_cvt_f32_i32": _v_cvt_f32_i32_overlay,
    "amdgpu.v_cvt_i32_f32": _v_cvt_i32_f32_overlay,
    "amdgpu.v_cvt_f32_u32": _v_cvt_f32_u32_overlay,
    "amdgpu.v_cvt_u32_f32": _v_cvt_u32_f32_overlay,
    "amdgpu.v_dot2_f32_f16": _v_dot2_f32_f16_overlay,
    "amdgpu.v_dot2_f32_bf16": _v_dot2_f32_bf16_overlay,
    "amdgpu.v_dot4_i32_i8": lambda: _v_dot4_i32_i8_overlay(
        signedness_modifiers=False,
    ),
    "amdgpu.v_dot4_u32_u8": _v_dot4_u32_u8_overlay,
    "amdgpu.v_dot4_i32_iu8.u8s8": lambda: _v_dot4_i32_iu8_overlay(
        lhs_signed=False,
        rhs_signed=True,
    ),
    "amdgpu.v_dot4_i32_iu8.s8u8": lambda: _v_dot4_i32_iu8_overlay(
        lhs_signed=True,
        rhs_signed=False,
    ),
    "amdgpu.v_dot8_i32_i4": lambda: _v_dot8_i32_i4_overlay(
        signedness_modifiers=False,
    ),
    "amdgpu.v_dot8_u32_u4": _v_dot8_u32_u4_overlay,
    "amdgpu.v_dot8_i32_iu4.s4u4": lambda: _v_dot8_i32_iu4_overlay(
        lhs_signed=True,
        rhs_signed=False,
    ),
    "amdgpu.v_dot8_i32_iu4.u4s4": lambda: _v_dot8_i32_iu4_overlay(
        lhs_signed=False,
        rhs_signed=True,
    ),
    "amdgpu.v_dot4_f32_fp8_bf8": lambda: _v_dot4_f32_packed8_overlay(
        lhs_type="fp8",
        rhs_type="bf8",
    ),
    "amdgpu.v_dot4_f32_bf8_fp8": lambda: _v_dot4_f32_packed8_overlay(
        lhs_type="bf8",
        rhs_type="fp8",
    ),
    "amdgpu.v_dot4_f32_fp8_fp8": lambda: _v_dot4_f32_packed8_overlay(
        lhs_type="fp8",
        rhs_type="fp8",
    ),
    "amdgpu.v_dot4_f32_bf8_bf8": lambda: _v_dot4_f32_packed8_overlay(
        lhs_type="bf8",
        rhs_type="bf8",
    ),
    **_contract_overlay_builders_from_overlays(_integer_bitwise_shift_overlays()),
    **_contract_overlay_builders_from_overlays(_s_cmp_i32_overlays()),
    **_contract_overlay_builders_from_overlays(_s_cmp_u64_overlays()),
    **_contract_overlay_builders_from_overlays(_v_cmp_overlays()),
}


def build_amdgpu_contract_descriptor_set(
    *,
    key: str,
    descriptor_keys: Sequence[str],
) -> DescriptorSet:
    """Builds an XML-free descriptor projection for source-to-low contracts."""

    descriptors = []
    for descriptor_key in descriptor_keys:
        try:
            overlay = _AMDGPU_CONTRACT_DESCRIPTOR_OVERLAY_BUILDERS[descriptor_key]()
        except KeyError as exc:
            raise ValueError(
                f"AMDGPU contract descriptor '{descriptor_key}' is not registered"
            ) from exc
        descriptors.append(_amdgpu_contract_descriptor_from_overlay(overlay))
    return replace(
        _AMDGPU_RDNA3_CORE_DESCRIPTOR_SET_BASE,
        key=key,
        feature_key=None,
        c_header_path=Path("loom/src/loom/target/arch/amdgpu/refs/target_refs.h"),
        c_source_path=Path("loom/src/loom/target/arch/amdgpu/refs/target_refs.c"),
        header_guard="LOOM_TARGET_ARCH_AMDGPU_REFS_TARGET_REFS_H_",
        public_header="loom/target/arch/amdgpu/refs/target_refs.h",
        function_name="loom_amdgpu_contract_descriptor_set",
        c_table_prefix=_amdgpu_contract_fragment_prefix(key),
        c_enum_prefix="LOOM_AMDGPU",
        descriptors=_categorize_amdgpu_descriptors(tuple(descriptors)),
    )


def _amdgpu_contract_fragment_prefix(key: str) -> str:
    parts = tuple(
        part for part in key.replace("_", ".").replace("-", ".").split(".") if part
    )
    if not parts:
        raise ValueError("AMDGPU contract descriptor set key must be non-empty")
    return "".join(part[:1].upper() + part[1:] for part in parts)


def _amdgpu_contract_descriptor_from_overlay(
    overlay: AmdgpuDescriptorOverlay,
) -> Descriptor:
    operands = tuple(
        operand_overlay.descriptor_operand for operand_overlay in overlay.operands
    ) + tuple(
        implicit_overlay.descriptor_operand
        for implicit_overlay in overlay.implicit_operands
        if implicit_overlay.descriptor_operand is not None
    )
    descriptor = Descriptor(
        key=overlay.descriptor_key,
        mnemonic=overlay.mnemonic or overlay.instruction_name.lower(),
        semantic_tag=overlay.semantic_tag,
        operands=tuple(
            operand for operand in operands if operand.role is OperandRole.RESULT
        )
        + tuple(
            operand for operand in operands if operand.role is not OperandRole.RESULT
        ),
        immediates=overlay.immediates,
        effects=overlay.effects,
        constraints=overlay.constraints,
        operand_forms=overlay.operand_forms,
        feature_mask_words=overlay.feature_mask_words,
        schedule_class=overlay.schedule_class,
        flags=overlay.flags,
    )
    return _with_execution_mask_state_read(descriptor)


__all__ = (
    "_AMDGPU_CONTRACT_DESCRIPTOR_OVERLAY_BUILDERS",
    "_amdgpu_contract_descriptor_from_overlay",
    "_amdgpu_contract_fragment_prefix",
    "_contract_overlay_builders_from_overlays",
    "build_amdgpu_contract_descriptor_set",
)
