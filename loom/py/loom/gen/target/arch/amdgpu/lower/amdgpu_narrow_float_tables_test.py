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
    _Fp8EncodedOperandSchemaRequirementRow,
    _Fp8FormatRow,
    _Fp8NativeDescriptorRefRow,
    _Fp8ScaledDescriptorRefRow,
)
from loom.ir import ScalarTypeKind


def test_f8e4m3_exact_bf16_via_f16_covers_every_finite_payload() -> None:
    row = next(row for row in amdgpu_narrow_float_tables._FP8_FORMAT_ROWS if row.source_type == ScalarTypeKind.F8E4M3)
    exponent_mask = (1 << row.exponent_bits) - 1
    mantissa_mask = (1 << row.mantissa_bits) - 1

    for payload in range(256):
        magnitude = payload & 0x7F
        exponent = (magnitude >> row.mantissa_bits) & exponent_mask
        mantissa = magnitude & mantissa_mask
        if exponent == exponent_mask and mantissa == mantissa_mask:
            continue

        sign = 0x8000 if payload & 0x80 else 0
        if exponent == 0:
            expected_magnitude = amdgpu_narrow_float_tables._fp8_subnormal_bf16_payload(row, mantissa)
            f16_magnitude = amdgpu_narrow_float_tables._fp8_subnormal_f16_payload(row, mantissa)
        else:
            expected_magnitude = ((exponent - row.exponent_bias + 127) << 7) | (mantissa << (7 - row.mantissa_bits))
            f16_magnitude = ((exponent - row.exponent_bias + 15) << 10) | (mantissa << (10 - row.mantissa_bits))

        rebased_magnitude = f16_magnitude >> 3
        if magnitude != 0:
            rebased_magnitude += 0x3800
        assert sign | rebased_magnitude == sign | expected_magnitude


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
            r"AMDGPU FP8 subnormal table must cover dense FP8 format bits in "
            r"order"
        ),
    ):
        amdgpu_narrow_float_tables._emit_fp8_subnormal_table_rows(
            rows=(
                _Fp8FormatRow(
                    "LOOM_VALUE_FACT_NUMERIC_FORMAT_F8_E4M3",
                    "f8e4m3",
                    ScalarTypeKind.F8E4M3,
                    exponent_bits=4,
                    mantissa_bits=3,
                    exponent_bias=7,
                    special_policy="LOOM_SCALAR_TYPE_FP8_SPECIAL_POLICY_FINITE_NAN",
                ),
            ),
        )


def test_fp8_encoded_operand_schema_requirement_rows_reject_kind_gap() -> None:
    with pytest.raises(
        ValueError,
        match=(
            r"AMDGPU FP8 encoded operand schema requirement table must cover "
            r"dense schema kinds in order"
        ),
    ):
        amdgpu_narrow_float_tables._emit_fp8_encoded_operand_schema_requirement_rows(
            rows=(
                _Fp8EncodedOperandSchemaRequirementRow(
                    "LOOM_AMDGPU_FP8_ENCODED_OPERAND_SCHEMA_KIND_UNSCALED",
                    "LOOM_VALUE_FACT_NUMERIC_FORMAT_NONE",
                    "LOOM_VALUE_FACT_SCALE_TOPOLOGY_NONE",
                    "LOOM_VALUE_FACT_AFFINE_POLICY_NONE",
                    0,
                    "LOOM_AMDGPU_FP8_SCALE_GROUP_MODE_NONE",
                ),
                _Fp8EncodedOperandSchemaRequirementRow(
                    "LOOM_AMDGPU_FP8_ENCODED_OPERAND_SCHEMA_KIND_SCALE_E8M0",
                    "LOOM_VALUE_FACT_NUMERIC_FORMAT_F8_E8M0",
                    "LOOM_VALUE_FACT_SCALE_TOPOLOGY_BLOCK_1D",
                    "LOOM_VALUE_FACT_AFFINE_POLICY_SCALE_ONLY",
                    1,
                    "LOOM_AMDGPU_FP8_SCALE_GROUP_MODE_OCTETS_MAX4",
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
                    "LOOM_VALUE_FACT_NUMERIC_FORMAT_F8_E4M3",
                    ScalarTypeKind.F32,
                    "amdgpu.missing",
                    "amdgpu.v_cvt_pk_f32_fp8.ocp",
                ),
            ),
            descriptor_ref_key_set={"amdgpu.v_cvt_pk_f32_fp8.ocp"},
        )


def test_fp8_native_descriptor_refs_reject_partial_byte_select_family() -> None:
    with pytest.raises(
        ValueError,
        match=(
            r"AMDGPU FP8 native conversion descriptor table byte-select family "
            r"has 1 entries instead of 4"
        ),
    ):
        amdgpu_narrow_float_tables._emit_fp8_native_descriptor_ref_rows(
            rows=(
                _Fp8NativeDescriptorRefRow(
                    "LOOM_VALUE_FACT_NUMERIC_FORMAT_F8_E4M3",
                    ScalarTypeKind.F16,
                    None,
                    "amdgpu.v_cvt_pk_f16_fp8.ocp",
                    ("amdgpu.v_cvt_f16_fp8.ocp.byte0",),
                ),
            ),
            descriptor_ref_key_set={
                "amdgpu.v_cvt_pk_f16_fp8.ocp",
                "amdgpu.v_cvt_f16_fp8.ocp.byte0",
            },
        )


def test_fp8_native_descriptor_refs_reject_duplicate_type_pair() -> None:
    with pytest.raises(
        ValueError,
        match=(
            r"AMDGPU FP8 native conversion descriptor table contains duplicate "
            r"source/result rows: LOOM_VALUE_FACT_NUMERIC_FORMAT_F8_E4M3->F32"
        ),
    ):
        amdgpu_narrow_float_tables._emit_fp8_native_descriptor_ref_rows(
            rows=(
                _Fp8NativeDescriptorRefRow(
                    "LOOM_VALUE_FACT_NUMERIC_FORMAT_F8_E4M3",
                    ScalarTypeKind.F32,
                    "amdgpu.v_cvt_f32_fp8.ocp",
                    "amdgpu.v_cvt_pk_f32_fp8.ocp",
                ),
                _Fp8NativeDescriptorRefRow(
                    "LOOM_VALUE_FACT_NUMERIC_FORMAT_F8_E4M3",
                    ScalarTypeKind.F32,
                    "amdgpu.v_cvt_f32_fp8.ocp",
                    "amdgpu.v_cvt_pk_f32_fp8.ocp",
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
                    "LOOM_VALUE_FACT_NUMERIC_FORMAT_F8_E4M3",
                    ScalarTypeKind.BF16,
                    "amdgpu.missing",
                    "amdgpu.v_cvt_scale_pk8_bf16_fp8.ocp",
                ),
            ),
            descriptor_ref_key_set={"amdgpu.v_cvt_scale_pk8_bf16_fp8.ocp"},
        )


def test_fp8_scaled_descriptor_refs_reject_duplicate_type_pair() -> None:
    with pytest.raises(
        ValueError,
        match=(
            r"AMDGPU FP8 scaled conversion descriptor table contains duplicate "
            r"source/result rows: LOOM_VALUE_FACT_NUMERIC_FORMAT_F8_E4M3->BF16"
        ),
    ):
        amdgpu_narrow_float_tables._emit_fp8_scaled_descriptor_ref_rows(
            rows=(
                _Fp8ScaledDescriptorRefRow(
                    "LOOM_VALUE_FACT_NUMERIC_FORMAT_F8_E4M3",
                    ScalarTypeKind.BF16,
                    "amdgpu.v_cvt_scalef32_pk_bf16_fp8.ocp",
                    "amdgpu.v_cvt_scale_pk8_bf16_fp8.ocp",
                ),
                _Fp8ScaledDescriptorRefRow(
                    "LOOM_VALUE_FACT_NUMERIC_FORMAT_F8_E4M3",
                    ScalarTypeKind.BF16,
                    "amdgpu.v_cvt_scalef32_pk_bf16_fp8.ocp",
                    "amdgpu.v_cvt_scale_pk8_bf16_fp8.ocp",
                ),
            ),
        )
