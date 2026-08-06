# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from loom.target.arch.amdgpu.descriptors.rdna4m import (
    _RDNA4M_SUPPLEMENTAL_FP8_INSTRUCTIONS,
)
from loom.target.arch.amdgpu.descriptors.sets import (
    _gfx115x_core_overlays,
    _rdna4m_core_overlays,
)


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
