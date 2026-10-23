# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Arithmetic descriptor coverage across native AMDGPU families."""

import pytest

from loom.target.arch.amdgpu.descriptors import _REG_SGPR, _REG_VGPR
from loom.target.arch.amdgpu.descriptors.api import (
    _AMDGPU_CORE_DESCRIPTOR_SET_BUILDERS,
)
from loom.target.low_descriptors import OperandAddressMapKind, OperandRole, RegClassAlt


@pytest.mark.parametrize("target", _AMDGPU_CORE_DESCRIPTOR_SET_BUILDERS)
def test_physical_lane_id_packets_span_every_family(target: str) -> None:
    overlays = {
        overlay.descriptor_key: overlay
        for overlay in _AMDGPU_CORE_DESCRIPTOR_SET_BUILDERS[target].overlay_rows()
    }
    low = overlays["amdgpu.v_mbcnt_lo_u32_b32.lane_id"]
    high = overlays["amdgpu.v_mbcnt_hi_u32_b32.lane_id"]

    assert low.instruction_name == "V_MBCNT_LO_U32_B32"
    assert tuple(operand.descriptor_operand.field_name for operand in low.operands) == (
        "dst",
    )
    assert tuple(
        (field_name, value.value_name, value.operand_type)
        for field_name, value in low.fixed_encoding_fields
    ) == (("SRC0", "-1", "OPR_SRC"), ("SRC1", "0", "OPR_SRC"))

    assert high.instruction_name == "V_MBCNT_HI_U32_B32"
    assert tuple(
        operand.descriptor_operand.field_name for operand in high.operands
    ) == ("dst", "low_lane")
    assert tuple(
        (field_name, value.value_name, value.operand_type)
        for field_name, value in high.fixed_encoding_fields
    ) == (("SRC0", "-1", "OPR_SRC"),)


@pytest.mark.parametrize("target", _AMDGPU_CORE_DESCRIPTOR_SET_BUILDERS)
@pytest.mark.parametrize("bit_width", [32, 64])
def test_fma_residuals_use_operand_modifiers_on_every_family(target, bit_width) -> None:
    overlays = {
        overlay.descriptor_key: overlay
        for overlay in _AMDGPU_CORE_DESCRIPTOR_SET_BUILDERS[target].overlay_rows()
    }
    for suffix, fields in (
        ("", ("dst", "a", "b", "c")),
        (".neg_a", ("dst", "a", "b", "c")),
        (".neg_a_one", ("dst", "a", "b")),
    ):
        descriptor = overlays[f"amdgpu.v_fma_f{bit_width}{suffix}"]
        assert descriptor.instruction_name == f"V_FMA_F{bit_width}"
        assert descriptor.immediates == ()
        operands = tuple(operand.descriptor_operand for operand in descriptor.operands)
        assert tuple(operand.field_name for operand in operands) == fields
        assert all(operand.unit_count == bit_width // 32 for operand in operands)
        fixed = dict(descriptor.fixed_encoding_fields)
        assert fixed.get("NEG", 0) == (1 if suffix else 0)
        assert ("SRC2" in fixed) == (suffix == ".neg_a_one")


@pytest.mark.parametrize("target", _AMDGPU_CORE_DESCRIPTOR_SET_BUILDERS)
@pytest.mark.parametrize("bit_width", [16, 32, 64])
def test_exponent_descriptors_preserve_mixed_operand_and_result_widths(
    target, bit_width: int
) -> None:
    overlays = {
        overlay.descriptor_key: overlay
        for overlay in _AMDGPU_CORE_DESCRIPTOR_SET_BUILDERS[target].overlay_rows()
    }
    exponent_width = min(bit_width, 32)
    for operation, result_width in (
        ("ldexp", bit_width),
        ("frexp_mant", bit_width),
        (f"frexp_exp_i{exponent_width}", exponent_width),
    ):
        descriptor = overlays[f"amdgpu.v_{operation}_f{bit_width}"]
        assert descriptor.instruction_name == f"V_{operation.upper()}_F{bit_width}"
        result, input_value, *exponents = (
            operand.descriptor_operand for operand in descriptor.operands
        )
        assert result.role is OperandRole.RESULT
        assert result.unit_count == max(1, result_width // 32)
        assert input_value.role is OperandRole.OPERAND
        assert input_value.unit_count == max(1, bit_width // 32)
        if bit_width == 16:
            assert input_value.reg_alts == (
                RegClassAlt(_REG_VGPR, register_part="amdgpu.vgpr.low16"),
            )
            assert result.reg_alts == input_value.reg_alts
        else:
            assert input_value.reg_alts == (
                RegClassAlt(_REG_SGPR),
                RegClassAlt(_REG_VGPR),
            )
        assert len(exponents) == (operation == "ldexp")
        if exponents:
            assert exponents[0].unit_count == 1
            assert exponents[0].reg_alts == input_value.reg_alts


@pytest.mark.parametrize("target", _AMDGPU_CORE_DESCRIPTOR_SET_BUILDERS)
def test_unsigned_subtract_accepts_an_inline_lhs_on_every_family(target) -> None:
    overlays = {
        overlay.descriptor_key: overlay
        for overlay in _AMDGPU_CORE_DESCRIPTOR_SET_BUILDERS[target].overlay_rows()
    }
    descriptor = overlays["amdgpu.v_sub_u32.src0_inline"]

    assert descriptor.instruction_name == (
        "V_SUB_U32" if target.startswith(("cdna", "gfx9_4")) else "V_SUB_NC_U32"
    )
    assert descriptor.semantic_tag == "integer.sub.u32"
    assert tuple(
        operand.descriptor_operand.field_name for operand in descriptor.operands
    ) == ("dst", "rhs")
    assert descriptor.immediates[0].field_name == "imm32"
    assert descriptor.immediates[0].unsigned_max == 64


@pytest.mark.parametrize("target", _AMDGPU_CORE_DESCRIPTOR_SET_BUILDERS)
def test_wide_shifts_preserve_placement_and_require_a_representable_neighbor(
    target,
) -> None:
    overlays = {
        overlay.descriptor_key: overlay
        for overlay in _AMDGPU_CORE_DESCRIPTOR_SET_BUILDERS[target].overlay_rows()
    }
    needs_neighbor = target.startswith(("cdna", "gfx9_4"))
    for mnemonic in ("v_lshlrev_b64", "v_lshrrev_b64", "v_ashrrev_i64"):
        descriptor = overlays[f"amdgpu.{mnemonic}"]
        result, shift, value = (
            operand.descriptor_operand for operand in descriptor.operands
        )
        assert result.unit_count == value.unit_count == 2
        assert shift.unit_count == 1
        assert shift.reg_alts == (RegClassAlt(_REG_SGPR), RegClassAlt(_REG_VGPR))
        assert shift.addressable_unit_count == (255 if needs_neighbor else 0)
        assert shift.address_map_kind is (
            OperandAddressMapKind.LOW_SUBSET
            if needs_neighbor
            else OperandAddressMapKind.DIRECT
        )
        inline = overlays[f"amdgpu.{mnemonic}.src0_inline"]
        assert tuple(operand.descriptor_operand for operand in inline.operands) == (
            result,
            value,
        )
