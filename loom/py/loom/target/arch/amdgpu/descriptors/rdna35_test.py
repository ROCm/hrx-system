# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from loom.target.arch.amdgpu.descriptors.common import (
    _REG_SGPR,
    _SOURCE_INLINE_F32_ENCODING_ID,
)
from loom.target.arch.amdgpu.descriptors.rdna35 import (
    _RDNA35_DPP16_F32_ROWS,
    _rdna35_dpp16_f32_uniform_rhs_overlays,
)
from loom.target.arch.amdgpu.descriptors.sets import (
    _gfx11_generic_core_overlays,
    _gfx115x_core_overlays,
    _rdna4m_core_overlays,
)


def test_rdna35_dpp16_f32_overlays_model_uniform_rhs_forms() -> None:
    overlays = {
        overlay.descriptor_key: overlay
        for overlay in _rdna35_dpp16_f32_uniform_rhs_overlays()
    }

    assert len(overlays) == 2 * len(_RDNA35_DPP16_F32_ROWS)
    for operation, instruction_name, mnemonic, semantic in _RDNA35_DPP16_F32_ROWS:
        descriptor_key = f"amdgpu.v_{operation}_f32"
        sgpr_overlay = overlays[f"{descriptor_key}.dpp16_sgpr"]
        inline_overlay = overlays[f"{descriptor_key}.dpp16_src1_inline"]

        for overlay in (sgpr_overlay, inline_overlay):
            assert overlay.instruction_name == instruction_name
            assert overlay.mnemonic == mnemonic
            assert overlay.encoding_name == "VOP3_VOP_DPP16"
            assert overlay.encoding_condition == "has_dpp16"
            assert overlay.semantic_tag == f"float.{semantic}.f32"
            assert overlay.fixed_encoding_fields == (
                ("SRC0", 250),
                ("ROW_MASK", 0xF),
                ("BANK_MASK", 0xF),
                ("BOUND_CTRL", 1),
            )

        assert tuple(operand.xml_field_name for operand in sgpr_overlay.operands) == (
            "VDST",
            "VSRC0",
            "SRC1",
        )
        assert sgpr_overlay.operands[-1].descriptor_operand.reg_alts[0].reg_class == (
            _REG_SGPR
        )
        assert sgpr_overlay.immediate_fields == ("DPP_CTRL",)

        assert tuple(operand.xml_field_name for operand in inline_overlay.operands) == (
            "VDST",
            "VSRC0",
        )
        assert inline_overlay.immediate_fields == ("SRC1", "DPP_CTRL")
        assert inline_overlay.immediates[0].encoding_id == (
            _SOURCE_INLINE_F32_ENCODING_ID
        )


def test_rdna35_dpp16_f32_overlays_remain_exact_target_only() -> None:
    exact_keys = {
        overlay.descriptor_key for overlay in _rdna35_dpp16_f32_uniform_rhs_overlays()
    }
    gfx115x_keys = {overlay.descriptor_key for overlay in _gfx115x_core_overlays()}
    generic_keys = {
        overlay.descriptor_key for overlay in _gfx11_generic_core_overlays()
    }
    rdna4m_overlays = {
        overlay.descriptor_key: overlay for overlay in _rdna4m_core_overlays()
    }

    assert exact_keys <= gfx115x_keys
    assert exact_keys.isdisjoint(generic_keys)
    assert exact_keys <= rdna4m_overlays.keys()
    for operation in ("min", "max"):
        for suffix in ("dpp16_sgpr", "dpp16_src1_inline"):
            overlay = rdna4m_overlays[f"amdgpu.v_{operation}_f32.{suffix}"]
            assert overlay.instruction_name == f"V_{operation.upper()}_NUM_F32"
            assert overlay.mnemonic == f"v_{operation}_num_f32"
