# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import pytest

from loom.gen.target.arch.amdgpu.lower import amdgpu_narrow_float_tables
from loom.gen.target.arch.amdgpu.lower.amdgpu_narrow_float_tables import (
    _Fp8DecodePlanDescriptorRow,
    _Fp8FormatRow,
    _Fp8NativeDescriptorRefRow,
    _Fp8ScaledDescriptorRefRow,
)
from loom.ir import ScalarTypeKind


def test_fp8_decode_plan_descriptor_rows_emit_data_only() -> None:
    source = amdgpu_narrow_float_tables._emit_fp8_decode_plan_descriptor_rows()

    assert "LOOM_AMDGPU_FP8_DECODE_PLAN_DESCRIPTOR_ROW(" in source
    assert "LOOM_AMDGPU_DESCRIPTOR_REF_V_BFE_U32_OFFSET_WIDTH_INLINE" in source
    assert "LOOM_AMDGPU_DESCRIPTOR_REF_V_PK_LSHLREV_B16" in source
    assert "LOOM_AMDGPU_DESCRIPTOR_REF_V_PK_MAD_U16" in source
    assert "LOOM_AMDGPU_DESCRIPTOR_REF_V_PK_MAD_U16_SRC2_LIT" in source
    assert "bfe_u32_descriptor" in source
    assert "pk_lshlrev_b16_descriptor" in source
    assert "pk_mad_u16_descriptor" in source
    assert "pk_mad_u16_src2_literal_descriptor" in source
    assert "LOOM_AMDGPU_FP8_DECODE_PLAN_FLAG_HAS_BFE_U32" in source
    assert "LOOM_AMDGPU_FP8_DECODE_PLAN_FLAG_HAS_PK_LSHLREV_B16" in source
    assert "LOOM_AMDGPU_FP8_DECODE_PLAN_FLAG_HAS_PK_MAD_U16" in source
    assert "LOOM_AMDGPU_FP8_DECODE_PLAN_FLAG_HAS_PK_MAD_U16_SRC2_LITERAL" in source
    assert "typedef " not in source
    assert "struct " not in source
    assert "#include" not in source
    assert "\nif " not in source
    assert "\nreturn " not in source


def test_fp8_native_descriptor_refs_emit_data_only() -> None:
    source = amdgpu_narrow_float_tables._emit_fp8_native_descriptor_ref_rows()

    assert "LOOM_AMDGPU_FP8_NATIVE_DESCRIPTOR_REF_ROW(" in source
    assert "    0,\n    LOOM_SCALAR_TYPE_F8E4M3,\n    LOOM_SCALAR_TYPE_F32," in source
    assert "LOOM_SCALAR_TYPE_F8E4M3" in source
    assert "LOOM_SCALAR_TYPE_F16" in source
    assert "LOOM_AMDGPU_DESCRIPTOR_REF_V_CVT_F32_FP8" in source
    assert "LOOM_AMDGPU_DESCRIPTOR_REF_V_CVT_PK_F16_FP8" in source
    assert "LOOM_AMDGPU_DESCRIPTOR_REF_V_CVT_PK_F32_BF8" in source
    assert "switch " not in source
    assert "\ncase " not in source
    assert "\nreturn " not in source


def test_fp8_scaled_descriptor_refs_emit_data_only() -> None:
    source = amdgpu_narrow_float_tables._emit_fp8_scaled_descriptor_ref_rows()

    assert "LOOM_AMDGPU_FP8_SCALED_DESCRIPTOR_REF_ROW(" in source
    assert "    0,\n    LOOM_SCALAR_TYPE_F8E4M3,\n    LOOM_SCALAR_TYPE_F16," in source
    assert "LOOM_SCALAR_TYPE_F8E4M3" in source
    assert "LOOM_SCALAR_TYPE_BF16" in source
    assert "LOOM_AMDGPU_DESCRIPTOR_REF_V_CVT_SCALEF32_PK_BF16_FP8" in source
    assert "LOOM_AMDGPU_DESCRIPTOR_REF_V_CVT_SCALEF32_PK_F32_BF8" in source
    assert "LOOM_AMDGPU_DESCRIPTOR_REF_V_CVT_SCALE_PK8_BF16_FP8" in source
    assert "LOOM_AMDGPU_DESCRIPTOR_REF_V_CVT_SCALE_PK8_F32_BF8" in source
    assert "switch " not in source
    assert "\ncase " not in source
    assert "\nreturn " not in source


def test_fp8_subnormal_table_rows_emit_data_only() -> None:
    source = amdgpu_narrow_float_tables._emit_fp8_subnormal_table_rows()

    assert "LOOM_AMDGPU_FP8_SUBNORMAL_TABLE_ROW(" in source
    assert "LOOM_SCALAR_TYPE_F8E4M3" in source
    assert "LOOM_SCALAR_TYPE_F8E5M2" in source
    assert "LOOM_SCALAR_TYPE_FP8_SPECIAL_POLICY_FINITE_NAN" in source
    assert "LOOM_SCALAR_TYPE_FP8_SPECIAL_POLICY_IEEE" in source
    assert "UINT32_C(0xC0800000)" in source
    assert "UINT32_C(0x3C3C3C3C)" in source
    assert "UINT32_C(0x37800000)" in source
    assert "UINT32_C(0x23222120)" in source
    assert "UINT32_C(0x07060504)" in source
    assert "switch " not in source
    assert "\ncase " not in source
    assert "\nreturn " not in source


def test_fp8_packed_repair_reason_rows_emit_data_only() -> None:
    source = amdgpu_narrow_float_tables._emit_fp8_packed_repair_reason_rows()

    assert source.count("LOOM_AMDGPU_FP8_PACKED_BF16_REPAIR_REASON_ROW(") == 16
    assert source.count("LOOM_AMDGPU_FP8_PACKED_F16_REPAIR_REASON_ROW(") == 4
    assert "fp8_packed_bf16_decode" in source
    assert "fp8_packed_bf16_decode_repair_zero_subnormal" in source
    assert "fp8_packed_bf16_decode_repair_zero_subnormal_inf_nan" in source
    assert "fp8_packed_f16_decode" in source
    assert "fp8_packed_f16_decode_repair_zero_subnormal" in source
    assert "typedef " not in source
    assert "struct " not in source
    assert "#include" not in source
    assert "switch " not in source
    assert "\ncase " not in source
    assert "\nreturn " not in source


def test_fp8_decode_plan_descriptor_rows_reject_missing_descriptor_ref() -> None:
    with pytest.raises(
        ValueError,
        match=(
            r"AMDGPU FP8 decode plan descriptor table requires missing "
            r"descriptor refs: amdgpu\.missing"
        ),
    ):
        amdgpu_narrow_float_tables._emit_fp8_decode_plan_descriptor_rows(
            rows=(
                _Fp8DecodePlanDescriptorRow(
                    "amdgpu.missing",
                    "bfe_u32_descriptor",
                    "LOOM_AMDGPU_FP8_DECODE_PLAN_FLAG_HAS_BFE_U32",
                ),
            ),
            descriptor_ref_key_set=set(),
        )


def test_fp8_decode_plan_descriptor_rows_reject_duplicate_plan_field() -> None:
    with pytest.raises(
        ValueError,
        match=(
            r"AMDGPU FP8 decode plan descriptor table contains duplicate "
            r"plan fields: bfe_u32_descriptor"
        ),
    ):
        amdgpu_narrow_float_tables._emit_fp8_decode_plan_descriptor_rows(
            rows=(
                _Fp8DecodePlanDescriptorRow(
                    "amdgpu.v_bfe_u32.offset_width_inline",
                    "bfe_u32_descriptor",
                    "LOOM_AMDGPU_FP8_DECODE_PLAN_FLAG_HAS_BFE_U32",
                ),
                _Fp8DecodePlanDescriptorRow(
                    "amdgpu.v_perm_b32",
                    "bfe_u32_descriptor",
                    "LOOM_AMDGPU_FP8_DECODE_PLAN_FLAG_HAS_PERM_B32",
                ),
            ),
        )


def test_fp8_decode_plan_descriptor_rows_reject_duplicate_capability_flag() -> None:
    with pytest.raises(
        ValueError,
        match=(
            r"AMDGPU FP8 decode plan descriptor table contains duplicate "
            r"capability flags: LOOM_AMDGPU_FP8_DECODE_PLAN_FLAG_HAS_BFE_U32"
        ),
    ):
        amdgpu_narrow_float_tables._emit_fp8_decode_plan_descriptor_rows(
            rows=(
                _Fp8DecodePlanDescriptorRow(
                    "amdgpu.v_bfe_u32.offset_width_inline",
                    "bfe_u32_descriptor",
                    "LOOM_AMDGPU_FP8_DECODE_PLAN_FLAG_HAS_BFE_U32",
                ),
                _Fp8DecodePlanDescriptorRow(
                    "amdgpu.v_perm_b32",
                    "perm_b32_descriptor",
                    "LOOM_AMDGPU_FP8_DECODE_PLAN_FLAG_HAS_BFE_U32",
                ),
            ),
        )


def test_fp8_subnormal_table_rows_reject_missing_dense_format_row() -> None:
    with pytest.raises(
        ValueError,
        match=(
            r"AMDGPU FP8 subnormal table must cover dense FP8/BF8 rows in "
            r"order: expected F8E4M3, F8E5M2; got F8E4M3"
        ),
    ):
        amdgpu_narrow_float_tables._emit_fp8_subnormal_table_rows(
            rows=(
                _Fp8FormatRow(
                    ScalarTypeKind.F8E4M3,
                    exponent_bits=4,
                    mantissa_bits=3,
                    exponent_bias=7,
                    special_policy="LOOM_SCALAR_TYPE_FP8_SPECIAL_POLICY_FINITE_NAN",
                ),
            ),
        )


def test_fp8_native_descriptor_refs_reject_missing_descriptor_ref() -> None:
    with pytest.raises(
        ValueError,
        match=(
            r"AMDGPU FP8 native conversion descriptor table requires missing "
            r"descriptor refs: amdgpu\.missing"
        ),
    ):
        amdgpu_narrow_float_tables._emit_fp8_native_descriptor_ref_rows(
            rows=(
                _Fp8NativeDescriptorRefRow(
                    ScalarTypeKind.F8E4M3,
                    ScalarTypeKind.F32,
                    "amdgpu.missing",
                    "amdgpu.v_cvt_pk_f32_fp8",
                ),
            ),
            descriptor_ref_key_set={"amdgpu.v_cvt_pk_f32_fp8"},
        )


def test_fp8_native_descriptor_refs_reject_duplicate_type_pair() -> None:
    with pytest.raises(
        ValueError,
        match=(
            r"AMDGPU FP8 native conversion descriptor table contains duplicate "
            r"source/result rows: F8E4M3->F32"
        ),
    ):
        amdgpu_narrow_float_tables._emit_fp8_native_descriptor_ref_rows(
            rows=(
                _Fp8NativeDescriptorRefRow(
                    ScalarTypeKind.F8E4M3,
                    ScalarTypeKind.F32,
                    "amdgpu.v_cvt_f32_fp8",
                    "amdgpu.v_cvt_pk_f32_fp8",
                ),
                _Fp8NativeDescriptorRefRow(
                    ScalarTypeKind.F8E4M3,
                    ScalarTypeKind.F32,
                    "amdgpu.v_cvt_f32_fp8",
                    "amdgpu.v_cvt_pk_f32_fp8",
                ),
            ),
        )


def test_fp8_scaled_descriptor_refs_reject_missing_descriptor_ref() -> None:
    with pytest.raises(
        ValueError,
        match=(
            r"AMDGPU FP8 scaled conversion descriptor table requires missing "
            r"descriptor refs: amdgpu\.missing"
        ),
    ):
        amdgpu_narrow_float_tables._emit_fp8_scaled_descriptor_ref_rows(
            rows=(
                _Fp8ScaledDescriptorRefRow(
                    ScalarTypeKind.F8E4M3,
                    ScalarTypeKind.BF16,
                    "amdgpu.missing",
                    "amdgpu.v_cvt_scale_pk8_bf16_fp8",
                ),
            ),
            descriptor_ref_key_set={"amdgpu.v_cvt_scale_pk8_bf16_fp8"},
        )


def test_fp8_scaled_descriptor_refs_reject_duplicate_type_pair() -> None:
    with pytest.raises(
        ValueError,
        match=(
            r"AMDGPU FP8 scaled conversion descriptor table contains duplicate "
            r"source/result rows: F8E4M3->BF16"
        ),
    ):
        amdgpu_narrow_float_tables._emit_fp8_scaled_descriptor_ref_rows(
            rows=(
                _Fp8ScaledDescriptorRefRow(
                    ScalarTypeKind.F8E4M3,
                    ScalarTypeKind.BF16,
                    "amdgpu.v_cvt_scalef32_pk_bf16_fp8",
                    "amdgpu.v_cvt_scale_pk8_bf16_fp8",
                ),
                _Fp8ScaledDescriptorRefRow(
                    ScalarTypeKind.F8E4M3,
                    ScalarTypeKind.BF16,
                    "amdgpu.v_cvt_scalef32_pk_bf16_fp8",
                    "amdgpu.v_cvt_scale_pk8_bf16_fp8",
                ),
            ),
        )
