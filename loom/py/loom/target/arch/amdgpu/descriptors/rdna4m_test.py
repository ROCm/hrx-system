# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from loom.target.arch.amdgpu.descriptors.common import (
    _RESOURCE_SWMMAC,
    _RESOURCE_VALU,
    _RESOURCE_WMMA,
    _SCHEDULE_SWMMAC,
    _SCHEDULE_WMMA,
)
from loom.target.arch.amdgpu.descriptors.rdna4m import (
    _AMDGPU_RDNA4M_CORE_DESCRIPTOR_SET_BASE,
    _RDNA4M_IEEE_MINMAX_INSTRUCTION_ROWS,
    _RDNA4M_NUMERIC_MINMAX_INSTRUCTION_ROWS,
    _RDNA4M_SUPPLEMENTAL_FP8_INSTRUCTIONS,
    _RDNA4M_SUPPLEMENTAL_MATRIX_INSTRUCTIONS,
)
from loom.target.arch.amdgpu.descriptors.sets import (
    _gfx115x_core_overlays,
    _rdna4m_core_overlays,
    _rdna4m_minmax_overlays,
)
from loom.target.low_descriptors import (
    IssueUse,
    LatencyKind,
    ModelQuality,
)


def test_rdna4m_minmax_manifest_matches_llvm_mc_opcodes() -> None:
    assert _RDNA4M_NUMERIC_MINMAX_INSTRUCTION_ROWS == (
        ("V_MIN_F32", "V_MIN_NUM_F32", 0x00F),
        ("V_MAX_F32", "V_MAX_NUM_F32", 0x010),
        ("V_MIN_F16", "V_MIN_NUM_F16", 0x03A),
        ("V_MAX_F16", "V_MAX_NUM_F16", 0x039),
        ("V_MIN_F64", "V_MIN_NUM_F64", 0x329),
        ("V_MAX_F64", "V_MAX_NUM_F64", 0x32A),
        ("V_PK_MIN_F16", "V_PK_MIN_NUM_F16", 0x011),
        ("V_PK_MAX_F16", "V_PK_MAX_NUM_F16", 0x012),
        ("V_MIN3_F32", "V_MIN3_NUM_F32", 0x219),
        ("V_MAX3_F32", "V_MAX3_NUM_F32", 0x21C),
        ("V_MIN3_F16", "V_MIN3_NUM_F16", 0x249),
        ("V_MAX3_F16", "V_MAX3_NUM_F16", 0x24C),
        ("V_MED3_F32", "V_MED3_NUM_F32", 0x231),
        ("V_MED3_F16", "V_MED3_NUM_F16", 0x232),
        ("V_MINMAX_F32", "V_MINMAX_NUM_F32", 0x25F),
        ("V_MAXMIN_F32", "V_MAXMIN_NUM_F32", 0x25E),
        ("V_MINMAX_F16", "V_MINMAX_NUM_F16", 0x261),
        ("V_MAXMIN_F16", "V_MAXMIN_NUM_F16", 0x260),
    )
    assert _RDNA4M_IEEE_MINMAX_INSTRUCTION_ROWS == (
        ("V_MIN_F32", "V_MINIMUM_F32", 0x365),
        ("V_MAX_F32", "V_MAXIMUM_F32", 0x366),
        ("V_MIN_F16", "V_MINIMUM_F16", 0x367),
        ("V_MAX_F16", "V_MAXIMUM_F16", 0x368),
        ("V_MIN_F64", "V_MINIMUM_F64", 0x341),
        ("V_MAX_F64", "V_MAXIMUM_F64", 0x342),
        ("V_PK_MIN_F16", "V_PK_MINIMUM_F16", 0x01D),
        ("V_PK_MAX_F16", "V_PK_MAXIMUM_F16", 0x01E),
        ("V_MIN3_F32", "V_MINIMUM3_F32", 0x22D),
        ("V_MAX3_F32", "V_MAXIMUM3_F32", 0x22E),
        ("V_MIN3_F16", "V_MINIMUM3_F16", 0x22F),
        ("V_MAX3_F16", "V_MAXIMUM3_F16", 0x230),
        ("V_MINMAX_F32", "V_MINIMUMMAXIMUM_F32", 0x26C),
        ("V_MAXMIN_F32", "V_MAXIMUMMINIMUM_F32", 0x26D),
        ("V_MINMAX_F16", "V_MINIMUMMAXIMUM_F16", 0x26E),
        ("V_MAXMIN_F16", "V_MAXIMUMMINIMUM_F16", 0x26F),
    )


def test_rdna4m_minmax_overlays_replace_legacy_instruction_names() -> None:
    minmax_overlays = _rdna4m_minmax_overlays()
    expected_instruction_names = {
        instruction_name
        for _, instruction_name, _ in (
            *_RDNA4M_NUMERIC_MINMAX_INSTRUCTION_ROWS,
            *_RDNA4M_IEEE_MINMAX_INSTRUCTION_ROWS,
        )
    }
    expected_overlays = {overlay.descriptor_key: overlay for overlay in minmax_overlays}
    core_overlays = _rdna4m_core_overlays()
    core_family_overlays = tuple(
        overlay
        for overlay in core_overlays
        if overlay.descriptor_key in expected_overlays
    )
    legacy_instruction_names = {
        source_name
        for source_name, instruction_name, _ in (
            *_RDNA4M_NUMERIC_MINMAX_INSTRUCTION_ROWS,
            *_RDNA4M_IEEE_MINMAX_INSTRUCTION_ROWS,
        )
        if source_name != instruction_name
    }

    assert {overlay.instruction_name for overlay in minmax_overlays} == (
        expected_instruction_names
    )
    assert legacy_instruction_names.isdisjoint(
        overlay.instruction_name for overlay in core_overlays
    )
    assert len(core_family_overlays) == len(expected_overlays)
    assert {
        overlay.descriptor_key: overlay.instruction_name
        for overlay in core_family_overlays
    } == {
        descriptor_key: overlay.instruction_name
        for descriptor_key, overlay in expected_overlays.items()
    }


def test_rdna4m_fp8_manifest_matches_llvm_mc_opcodes() -> None:
    encodings = {
        (instruction.name, encoding.encoding_name): encoding.opcode
        for instruction in _RDNA4M_SUPPLEMENTAL_FP8_INSTRUCTIONS
        for encoding in instruction.encodings
    }

    assert encodings == {
        ("V_CVT_F32_FP8", "ENC_VOP1"): 0x6C,
        ("V_CVT_F32_FP8", "ENC_VOP3"): 0x1EC,
        ("V_CVT_F32_BF8", "ENC_VOP1"): 0x6D,
        ("V_CVT_F32_BF8", "ENC_VOP3"): 0x1ED,
        ("V_CVT_PK_F32_FP8", "ENC_VOP1"): 0x6E,
        ("V_CVT_PK_F32_FP8", "ENC_VOP3"): 0x1EE,
        ("V_CVT_PK_F32_BF8", "ENC_VOP1"): 0x6F,
        ("V_CVT_PK_F32_BF8", "ENC_VOP3"): 0x1EF,
        ("V_CVT_PK_FP8_F32", "ENC_VOP3"): 0x369,
        ("V_CVT_PK_BF8_F32", "ENC_VOP3"): 0x36A,
        ("V_DOT4_F32_FP8_BF8", "ENC_VOP3P"): 0x24,
        ("V_DOT4_F32_BF8_FP8", "ENC_VOP3P"): 0x25,
        ("V_DOT4_F32_FP8_FP8", "ENC_VOP3P"): 0x26,
        ("V_DOT4_F32_BF8_BF8", "ENC_VOP3P"): 0x27,
    }


def test_rdna4m_fp8_manifest_models_exact_operand_shapes() -> None:
    default_encodings = {
        instruction.name: instruction.encodings[0]
        for instruction in _RDNA4M_SUPPLEMENTAL_FP8_INSTRUCTIONS
    }

    assert tuple(
        (operand.field_name, operand.data_format_name, operand.size_bits)
        for operand in default_encodings["V_CVT_PK_F32_FP8"].operands
    ) == (
        ("VDST", "FMT_NUM_PK2_F32", 64),
        ("SRC0", "FMT_NUM_PK2_FP8", 16),
    )
    assert tuple(
        (operand.field_name, operand.data_format_name, operand.size_bits)
        for operand in default_encodings["V_CVT_PK_FP8_F32"].operands
    ) == (
        ("VDST", "FMT_NUM_PK2_FP8", 16),
        ("SRC0", "FMT_NUM_F32", 32),
        ("SRC1", "FMT_NUM_F32", 32),
    )
    assert tuple(
        (operand.field_name, operand.data_format_name, operand.size_bits)
        for operand in default_encodings["V_DOT4_F32_FP8_BF8"].operands
    ) == (
        ("VDST", "FMT_NUM_F32", 32),
        ("SRC0", "FMT_NUM_PK4_FP8", 32),
        ("SRC1", "FMT_NUM_PK4_BF8", 32),
        ("SRC2", "FMT_NUM_F32", 32),
    )


def test_rdna4m_native_fp8_descriptors_do_not_leak_to_rdna35() -> None:
    rdna4m_keys = {overlay.descriptor_key for overlay in _rdna4m_core_overlays()}
    rdna35_keys = {overlay.descriptor_key for overlay in _gfx115x_core_overlays()}
    native_fp8_keys = {
        "amdgpu.v_cvt_f32_fp8.ocp",
        "amdgpu.v_cvt_f32_fp8.ocp.byte1",
        "amdgpu.v_cvt_f32_fp8.ocp.byte2",
        "amdgpu.v_cvt_f32_fp8.ocp.byte3",
        "amdgpu.v_cvt_f32_bf8.ocp",
        "amdgpu.v_cvt_f32_bf8.ocp.byte1",
        "amdgpu.v_cvt_f32_bf8.ocp.byte2",
        "amdgpu.v_cvt_f32_bf8.ocp.byte3",
        "amdgpu.v_cvt_pk_f32_fp8.ocp",
        "amdgpu.v_cvt_pk_f32_fp8.ocp.high",
        "amdgpu.v_cvt_pk_f32_bf8.ocp",
        "amdgpu.v_cvt_pk_f32_bf8.ocp.high",
        "amdgpu.v_cvt_pk_fp8_f32.ocp.low",
        "amdgpu.v_cvt_pk_fp8_f32.ocp.high",
        "amdgpu.v_cvt_pk_bf8_f32.ocp.low",
        "amdgpu.v_cvt_pk_bf8_f32.ocp.high",
        "amdgpu.v_dot4_f32_fp8_bf8",
        "amdgpu.v_dot4_f32_bf8_fp8",
        "amdgpu.v_dot4_f32_fp8_fp8",
        "amdgpu.v_dot4_f32_bf8_bf8",
    }

    assert native_fp8_keys <= rdna4m_keys
    assert native_fp8_keys.isdisjoint(rdna35_keys)


def test_rdna4m_packed_fp8_dots_pin_no_op_modifiers() -> None:
    descriptors = {
        overlay.descriptor_key: overlay for overlay in _rdna4m_core_overlays()
    }

    for lhs_type in ("fp8", "bf8"):
        for rhs_type in ("fp8", "bf8"):
            descriptor_key = f"amdgpu.v_dot4_f32_{lhs_type}_{rhs_type}"
            assert descriptors[descriptor_key].fixed_encoding_fields == (
                ("OP_SEL_HI", 0x7),
            )


def test_rdna4m_matrix_manifest_matches_llvm_mc_opcodes() -> None:
    opcodes = {
        instruction.name: instruction.encodings[0].opcode
        for instruction in _RDNA4M_SUPPLEMENTAL_MATRIX_INSTRUCTIONS
    }

    assert opcodes == {
        "V_WMMA_F32_16X16X16_F16": 0x40,
        "V_WMMA_F32_16X16X16_BF16": 0x41,
        "V_WMMA_F16_16X16X16_F16": 0x42,
        "V_WMMA_BF16_16X16X16_BF16": 0x43,
        "V_WMMA_I32_16X16X16_IU8": 0x44,
        "V_WMMA_I32_16X16X16_IU4": 0x45,
        "V_WMMA_F32_16X16X16_FP8_FP8": 0x46,
        "V_WMMA_F32_16X16X16_FP8_BF8": 0x47,
        "V_WMMA_F32_16X16X16_BF8_FP8": 0x48,
        "V_WMMA_F32_16X16X16_BF8_BF8": 0x49,
        "V_WMMA_I32_16X16X32_IU4": 0x4A,
        "V_SWMMAC_F32_16X16X32_F16": 0x50,
        "V_SWMMAC_F32_16X16X32_BF16": 0x51,
        "V_SWMMAC_F16_16X16X32_F16": 0x52,
        "V_SWMMAC_BF16_16X16X32_BF16": 0x53,
        "V_SWMMAC_I32_16X16X32_IU8": 0x54,
        "V_SWMMAC_I32_16X16X32_IU4": 0x55,
        "V_SWMMAC_I32_16X16X64_IU4": 0x56,
        "V_SWMMAC_F32_16X16X32_FP8_FP8": 0x57,
        "V_SWMMAC_F32_16X16X32_FP8_BF8": 0x58,
        "V_SWMMAC_F32_16X16X32_BF8_FP8": 0x59,
        "V_SWMMAC_F32_16X16X32_BF8_BF8": 0x5A,
    }


def test_rdna4m_matrix_manifest_models_register_shapes() -> None:
    encodings = {
        instruction.name: instruction.encodings[0]
        for instruction in _RDNA4M_SUPPLEMENTAL_MATRIX_INSTRUCTIONS
    }

    def operand_shapes(
        instruction_name: str,
    ) -> tuple[tuple[str | None, str | None, str, int, bool, bool], ...]:
        return tuple(
            (
                operand.field_name,
                operand.data_format_name,
                operand.operand_type,
                operand.size_bits,
                operand.is_input,
                operand.is_output,
            )
            for operand in encodings[instruction_name].operands
        )

    assert operand_shapes("V_WMMA_F32_16X16X16_FP8_BF8") == (
        ("VDST", "FMT_WMMA_DC_16X16_F32", "OPR_VGPR", 256, False, True),
        ("SRC0", "FMT_WMMA_AB_16X16_FP8", "OPR_SRC_VGPR", 64, True, False),
        ("SRC1", "FMT_WMMA_AB_16X16_BF8", "OPR_SRC_VGPR", 64, True, False),
        (
            "SRC2",
            "FMT_WMMA_DC_16X16_F32",
            "OPR_SRC_VGPR_OR_INLINE",
            256,
            True,
            False,
        ),
    )
    assert operand_shapes("V_WMMA_I32_16X16X32_IU4") == (
        ("VDST", "FMT_WMMA_DC_16X16_I32", "OPR_VGPR", 256, False, True),
        ("SRC0", "FMT_WMMA_AB_16X32_IU4", "OPR_SRC_VGPR", 64, True, False),
        ("SRC1", "FMT_WMMA_AB_16X32_IU4", "OPR_SRC_VGPR", 64, True, False),
        (
            "SRC2",
            "FMT_WMMA_DC_16X16_I32",
            "OPR_SRC_VGPR_OR_INLINE",
            256,
            True,
            False,
        ),
    )
    assert operand_shapes("V_SWMMAC_F16_16X16X32_F16") == (
        ("VDST", "FMT_WMMA_DC_16X16_F16", "OPR_VGPR", 128, False, True),
        ("SRC0", "FMT_WMMA_AB_16X16_F16", "OPR_SRC_VGPR", 128, True, False),
        ("SRC1", "FMT_WMMA_AB_16X32_F16", "OPR_SRC_VGPR", 256, True, False),
        ("SRC2", "FMT_WMMA_INDEX_SET", "OPR_SRC_VGPR", 32, True, False),
    )
    assert operand_shapes("V_SWMMAC_I32_16X16X64_IU4") == (
        ("VDST", "FMT_WMMA_DC_16X16_I32", "OPR_VGPR", 256, False, True),
        ("SRC0", "FMT_WMMA_AB_16X32_IU4", "OPR_SRC_VGPR", 64, True, False),
        ("SRC1", "FMT_WMMA_AB_16X64_IU4", "OPR_SRC_VGPR", 128, True, False),
        ("SRC2", "FMT_WMMA_INDEX_SET", "OPR_SRC_VGPR", 32, True, False),
    )


def test_rdna4m_matrix_descriptors_cover_manifest_and_architecture_delta() -> None:
    matrix_instruction_names = {
        instruction.name for instruction in _RDNA4M_SUPPLEMENTAL_MATRIX_INSTRUCTIONS
    }
    rdna4m_instruction_names = {
        overlay.instruction_name for overlay in _rdna4m_core_overlays()
    }
    rdna35_instruction_names = {
        overlay.instruction_name for overlay in _gfx115x_core_overlays()
    }

    assert matrix_instruction_names <= rdna4m_instruction_names
    assert matrix_instruction_names - rdna35_instruction_names == {
        "V_WMMA_F32_16X16X16_FP8_FP8",
        "V_WMMA_F32_16X16X16_FP8_BF8",
        "V_WMMA_F32_16X16X16_BF8_FP8",
        "V_WMMA_F32_16X16X16_BF8_BF8",
        "V_WMMA_I32_16X16X32_IU4",
        "V_SWMMAC_F32_16X16X32_F16",
        "V_SWMMAC_F32_16X16X32_BF16",
        "V_SWMMAC_F16_16X16X32_F16",
        "V_SWMMAC_BF16_16X16X32_BF16",
        "V_SWMMAC_I32_16X16X32_IU8",
        "V_SWMMAC_I32_16X16X32_IU4",
        "V_SWMMAC_I32_16X16X64_IU4",
        "V_SWMMAC_F32_16X16X32_FP8_FP8",
        "V_SWMMAC_F32_16X16X32_FP8_BF8",
        "V_SWMMAC_F32_16X16X32_BF8_FP8",
        "V_SWMMAC_F32_16X16X32_BF8_BF8",
    }


def test_rdna4m_wave64_matrix_descriptors_model_half_width_abi() -> None:
    matrix_instruction_names = {
        instruction.name for instruction in _RDNA4M_SUPPLEMENTAL_MATRIX_INSTRUCTIONS
    }
    descriptors = {
        overlay.descriptor_key: overlay
        for overlay in _rdna4m_core_overlays()
        if overlay.instruction_name in matrix_instruction_names
    }
    wave32_descriptors = {
        key: descriptor
        for key, descriptor in descriptors.items()
        if not key.endswith(".acc_zero") and ".w64" not in key
    }

    assert len(wave32_descriptors) == len(matrix_instruction_names)
    for descriptor_key, wave32_descriptor in wave32_descriptors.items():
        wave64_descriptor = descriptors[f"{descriptor_key}.w64"]
        assert wave64_descriptor.mnemonic == f"{wave32_descriptor.mnemonic}_w64"
        assert wave64_descriptor.asm_forms is not None
        assert wave64_descriptor.asm_forms[0].native_assembly_mnemonic == (
            wave32_descriptor.mnemonic
        )
        for wave32_operand, wave64_operand in zip(
            wave32_descriptor.operands,
            wave64_descriptor.operands,
            strict=True,
        ):
            wave32_units = wave32_operand.descriptor_operand.unit_count
            wave64_units = wave64_operand.descriptor_operand.unit_count
            assert wave64_units == max(1, wave32_units // 2)
            assert (wave64_operand.size_exception_reason is not None) == (
                wave64_units != wave32_units
            )


def test_rdna4m_matrix_schedule_matches_llvm_gfx11_model() -> None:
    schedule_classes = {
        schedule_class.name: schedule_class
        for schedule_class in _AMDGPU_RDNA4M_CORE_DESCRIPTOR_SET_BASE.schedule_classes
    }

    for schedule_class_name, matrix_resource_name in (
        (_SCHEDULE_WMMA, _RESOURCE_WMMA),
        (_SCHEDULE_SWMMAC, _RESOURCE_SWMMAC),
    ):
        schedule_class = schedule_classes[schedule_class_name]
        assert schedule_class.latency_kind is LatencyKind.ESTIMATE
        assert schedule_class.latency_cycles == 5
        assert schedule_class.schedule_distance_cycles == 0
        assert schedule_class.issue_uses == (
            IssueUse(_RESOURCE_VALU, cycles=1, units=1),
            IssueUse(matrix_resource_name, cycles=1, units=1),
        )
        assert schedule_class.model_quality is ModelQuality.ESTIMATED
