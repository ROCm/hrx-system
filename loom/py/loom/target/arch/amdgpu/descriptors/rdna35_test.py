# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from loom.target.arch.amdgpu.descriptors.alu import (
    _v_dpp_uniform_rhs_integer_compare_candidates,
)
from loom.target.arch.amdgpu.descriptors.common import (
    _REG_SGPR,
    _REG_VGPR,
    _SOURCE_INLINE_F32_ENCODING_ID,
)
from loom.target.arch.amdgpu.descriptors.rdna35 import (
    _RDNA35_DPP_F32_ROWS,
    _rdna35_dpp8_f32_uniform_rhs_overlays,
    _rdna35_dpp16_f32_uniform_rhs_overlays,
    _rdna35_dpp_integer_compare_uniform_rhs_overlays,
)
from loom.target.arch.amdgpu.descriptors.sets import (
    _gfx11_core_overlays,
    _gfx115x_core_overlays,
    _rdna4m_core_overlays,
)
from loom.target.arch.amdgpu.encoding import amdgpu_encoding_field_id


def test_rdna35_dpp_f32_overlays_model_uniform_rhs_forms() -> None:
    families = (
        (
            "dpp16",
            "VOP3_VOP_DPP16",
            "has_dpp16",
            (
                ("SRC0", 250),
                ("ROW_MASK", 0xF),
                ("BANK_MASK", 0xF),
                ("BOUND_CTRL", 1),
            ),
            _rdna35_dpp16_f32_uniform_rhs_overlays(),
        ),
        (
            "dpp8",
            "VOP3_VOP_DPP8",
            "has_dpp8",
            (("SRC0", 233),),
            _rdna35_dpp8_f32_uniform_rhs_overlays(),
        ),
    )
    for suffix, encoding_name, encoding_condition, fixed_fields, family in families:
        overlays = {overlay.descriptor_key: overlay for overlay in family}
        assert len(overlays) == 2 * len(_RDNA35_DPP_F32_ROWS)
        for operation, instruction_name, mnemonic, semantic in _RDNA35_DPP_F32_ROWS:
            descriptor_key = f"amdgpu.v_{operation}_f32"
            sgpr_overlay = overlays[f"{descriptor_key}.{suffix}_sgpr"]
            inline_overlay = overlays[f"{descriptor_key}.{suffix}_src1_inline"]

            for overlay in (sgpr_overlay, inline_overlay):
                assert overlay.instruction_name == instruction_name
                assert overlay.mnemonic == mnemonic
                assert overlay.encoding_name == encoding_name
                assert overlay.encoding_condition == encoding_condition
                assert overlay.semantic_tag == f"float.{semantic}.f32"
                assert overlay.fixed_encoding_fields == fixed_fields

            assert tuple(
                operand.xml_field_name for operand in sgpr_overlay.operands
            ) == ("VDST", "VSRC0", "SRC1")
            assert (
                sgpr_overlay.operands[-1].descriptor_operand.reg_alts[0].reg_class
                == _REG_SGPR
            )
            assert tuple(
                operand.xml_field_name for operand in inline_overlay.operands
            ) == ("VDST", "VSRC0")
            assert inline_overlay.immediates[0].encoding_id == (
                _SOURCE_INLINE_F32_ENCODING_ID
            )
            assert inline_overlay.immediates[0].encoding_field_id == (
                amdgpu_encoding_field_id("SRC1")
            )


def test_rdna35_dpp_integer_compare_overlays_derive_semantic_families() -> None:
    candidates = _v_dpp_uniform_rhs_integer_compare_candidates()
    overlays = {
        overlay.descriptor_key: overlay
        for overlay in _rdna35_dpp_integer_compare_uniform_rhs_overlays()
    }
    assert len(overlays) == 4 * len(candidates)

    for base_overlay, inline_immediate in candidates:
        has_scalar_destination = any(
            operand.xml_field_name == "SDST" for operand in base_overlay.operands
        )
        for suffix in ("dpp16", "dpp8"):
            expected_encoding = (
                f"VOP3_SDST_ENC_VOP_{suffix.upper()}"
                if has_scalar_destination
                else f"VOP3_VOP_{suffix.upper()}"
            )
            sgpr_overlay = overlays[f"{base_overlay.descriptor_key}.{suffix}_sgpr"]
            inline_overlay = overlays[
                f"{base_overlay.descriptor_key}.{suffix}_src1_inline"
            ]

            for overlay in (sgpr_overlay, inline_overlay):
                assert overlay.instruction_name == base_overlay.instruction_name
                assert overlay.semantic_tag == base_overlay.semantic_tag
                assert overlay.encoding_name == expected_encoding
                assert overlay.flags == base_overlay.flags

            crosslane_operand = next(
                operand
                for operand in sgpr_overlay.operands
                if operand.xml_field_name == "VSRC0"
            )
            rhs_operand = next(
                operand
                for operand in sgpr_overlay.operands
                if operand.xml_field_name == "SRC1"
            )
            assert crosslane_operand.descriptor_operand.reg_alts[0].reg_class == (
                _REG_VGPR
            )
            assert rhs_operand.descriptor_operand.reg_alts[0].reg_class == _REG_SGPR
            assert all(
                operand.xml_field_name != "SRC1" for operand in inline_overlay.operands
            )
            assert inline_overlay.immediates[0].encoding_id == (
                inline_immediate.encoding_id
            )
            assert inline_overlay.immediates[0].encoding_field_id == (
                amdgpu_encoding_field_id("SRC1")
            )


def test_rdna35_dpp_selectors_map_to_their_native_encoding_fields() -> None:
    dpp16_selector = _rdna35_dpp16_f32_uniform_rhs_overlays()[0].immediates[-1]
    assert dpp16_selector.encoding_field_id == amdgpu_encoding_field_id("DPP_CTRL")
    assert not dpp16_selector.encoding_slices

    dpp8_selector = _rdna35_dpp8_f32_uniform_rhs_overlays()[0].immediates[-1]
    assert dpp8_selector.encoding_field_id == 0
    assert tuple(
        (
            encoding_slice.encoding_field_id,
            encoding_slice.source_bit_offset,
            encoding_slice.bit_count,
        )
        for encoding_slice in dpp8_selector.encoding_slices
    ) == tuple(
        (amdgpu_encoding_field_id(f"LANE_SEL_{lane}"), lane * 3, 3) for lane in range(8)
    )


def test_rdna35_dpp_uniform_rhs_overlays_remain_exact_target_only() -> None:
    exact_keys = {
        overlay.descriptor_key
        for family in (
            _rdna35_dpp16_f32_uniform_rhs_overlays(),
            _rdna35_dpp8_f32_uniform_rhs_overlays(),
            _rdna35_dpp_integer_compare_uniform_rhs_overlays(),
        )
        for overlay in family
    }
    gfx115x_keys = {overlay.descriptor_key for overlay in _gfx115x_core_overlays()}
    generic_keys = {overlay.descriptor_key for overlay in _gfx11_core_overlays()}
    rdna4m_overlays = {
        overlay.descriptor_key: overlay for overlay in _rdna4m_core_overlays()
    }

    assert exact_keys <= gfx115x_keys
    assert exact_keys.isdisjoint(generic_keys)
    assert exact_keys <= rdna4m_overlays.keys()
    for operation in ("min", "max"):
        for suffix in (
            "dpp16_sgpr",
            "dpp16_src1_inline",
            "dpp8_sgpr",
            "dpp8_src1_inline",
        ):
            overlay = rdna4m_overlays[f"amdgpu.v_{operation}_f32.{suffix}"]
            assert overlay.instruction_name == f"V_{operation.upper()}_NUM_F32"
            assert overlay.mnemonic == f"v_{operation}_f32"
            assert overlay.asm_forms[0].native_assembly_mnemonic == (
                f"v_{operation}_num_f32_e64_dpp"
            )
