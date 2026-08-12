# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Generator: AMDGPU narrow-float descriptor tables."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path


def _ensure_runtime_py_on_path() -> None:
    runtime_py = Path(__file__).resolve().parents[6]
    runtime_py_string = str(runtime_py)
    if runtime_py_string not in sys.path:
        sys.path.insert(0, runtime_py_string)


_ensure_runtime_py_on_path()

from loom.dialect.encoding.numeric_formats import (  # noqa: E402
    FP8_FORMATS,
    Fp8SpecialPolicy,
)
from loom.gen.support.files import write_text_file  # noqa: E402
from loom.gen.support.generated_file import line_comment_header  # noqa: E402
from loom.gen.target.arch.amdgpu.lower.candidates.validation import (  # noqa: E402
    required_descriptor_ref_constant_name,
)
from loom.ir import ScalarTypeKind  # noqa: E402
from loom.target.arch.amdgpu.descriptors import amdgpu_descriptor_ref_keys  # noqa: E402

_GENERATOR = "loom.gen.target.arch.amdgpu.lower.amdgpu_narrow_float_tables"


@dataclass(frozen=True)
class _Fp8DecodePlanDescriptorRow:
    descriptor_key: str
    plan_field: str
    present_flag: str


@dataclass(frozen=True)
class _Fp8NativeDescriptorRefRow:
    source_format: str
    result_type: ScalarTypeKind
    lane_descriptor_key: str | None
    pair_descriptor_key: str
    byte_select_descriptor_keys: tuple[str, ...] = ()


@dataclass(frozen=True)
class _Fp8ScaledDescriptorRefRow:
    source_format: str
    result_type: ScalarTypeKind
    scalef32_pair_descriptor_key: str
    e8m0_pk8_descriptor_key: str


@dataclass(frozen=True)
class _Fp8FormatRow:
    source_format: str
    keyword: str
    source_type: ScalarTypeKind
    exponent_bits: int
    mantissa_bits: int
    exponent_bias: int
    special_policy: str


@dataclass(frozen=True)
class _Fp8EncodedOperandSchemaRequirementRow:
    kind: str
    scale_format: str
    scale_topology: str
    affine_policy: str
    scale_operand_count: int
    scale_group_mode: str


@dataclass(frozen=True)
class _Fp8PackedRepairBit:
    flag_name: str
    suffix: str
    bit_value: int


_FP8_PACKED_REPAIR_BITS = (
    _Fp8PackedRepairBit(
        "LOOM_AMDGPU_FP8_PACKED_U16_REPAIR_ZERO",
        "zero",
        1 << 0,
    ),
    _Fp8PackedRepairBit(
        "LOOM_AMDGPU_FP8_PACKED_U16_REPAIR_SUBNORMAL",
        "subnormal",
        1 << 1,
    ),
    _Fp8PackedRepairBit(
        "LOOM_AMDGPU_FP8_PACKED_U16_REPAIR_INF",
        "inf",
        1 << 3,
    ),
    _Fp8PackedRepairBit(
        "LOOM_AMDGPU_FP8_PACKED_U16_REPAIR_NAN",
        "nan",
        1 << 2,
    ),
)


_FP8_SOURCE_FORMAT_CONSTANTS = {
    "f8e4m3": "LOOM_VALUE_FACT_NUMERIC_FORMAT_F8_E4M3",
    "f8e5m2": "LOOM_VALUE_FACT_NUMERIC_FORMAT_F8_E5M2",
    "f8e4m3fn": "LOOM_VALUE_FACT_NUMERIC_FORMAT_F8_E4M3FN",
    "f8e4m3fnuz": "LOOM_VALUE_FACT_NUMERIC_FORMAT_F8_E4M3FNUZ",
    "f8e5m2fnuz": "LOOM_VALUE_FACT_NUMERIC_FORMAT_F8_E5M2FNUZ",
}

_FP8_SPECIAL_POLICY_CONSTANTS = {
    Fp8SpecialPolicy.IEEE: "LOOM_SCALAR_TYPE_FP8_SPECIAL_POLICY_IEEE",
    Fp8SpecialPolicy.FINITE_NAN: "LOOM_SCALAR_TYPE_FP8_SPECIAL_POLICY_FINITE_NAN",
    Fp8SpecialPolicy.FINITE_NAN_UNSIGNED_ZERO: ("LOOM_SCALAR_TYPE_FP8_SPECIAL_POLICY_FINITE_NAN_UNSIGNED_ZERO"),
}

_FP8_FORMAT_ROWS = tuple(
    _Fp8FormatRow(
        source_format=_FP8_SOURCE_FORMAT_CONSTANTS[numeric_format.keyword],
        keyword=numeric_format.keyword,
        source_type=numeric_format.carrier_type,
        exponent_bits=numeric_format.exponent_bits,
        mantissa_bits=numeric_format.mantissa_bits,
        exponent_bias=numeric_format.exponent_bias,
        special_policy=_FP8_SPECIAL_POLICY_CONSTANTS[numeric_format.special_policy],
    )
    for numeric_format in FP8_FORMATS
)

_FP8_FORMAT_INDEX_BY_NAME = {row.source_format: index for index, row in enumerate(_FP8_FORMAT_ROWS)}


_FP8_ENCODED_OPERAND_SCHEMA_REQUIREMENT_ROWS = (
    _Fp8EncodedOperandSchemaRequirementRow(
        "LOOM_AMDGPU_FP8_ENCODED_OPERAND_SCHEMA_KIND_UNSCALED",
        "LOOM_VALUE_FACT_NUMERIC_FORMAT_NONE",
        "LOOM_VALUE_FACT_SCALE_TOPOLOGY_NONE",
        "LOOM_VALUE_FACT_AFFINE_POLICY_NONE",
        0,
        "LOOM_AMDGPU_FP8_SCALE_GROUP_MODE_NONE",
    ),
    _Fp8EncodedOperandSchemaRequirementRow(
        "LOOM_AMDGPU_FP8_ENCODED_OPERAND_SCHEMA_KIND_SCALE_F32",
        "LOOM_VALUE_FACT_NUMERIC_FORMAT_F32",
        "LOOM_VALUE_FACT_SCALE_TOPOLOGY_BLOCK_1D",
        "LOOM_VALUE_FACT_AFFINE_POLICY_SCALE_ONLY",
        1,
        "LOOM_AMDGPU_FP8_SCALE_GROUP_MODE_ALL_LANES",
    ),
    _Fp8EncodedOperandSchemaRequirementRow(
        "LOOM_AMDGPU_FP8_ENCODED_OPERAND_SCHEMA_KIND_SCALE_E8M0",
        "LOOM_VALUE_FACT_NUMERIC_FORMAT_F8_E8M0",
        "LOOM_VALUE_FACT_SCALE_TOPOLOGY_BLOCK_1D",
        "LOOM_VALUE_FACT_AFFINE_POLICY_SCALE_ONLY",
        1,
        "LOOM_AMDGPU_FP8_SCALE_GROUP_MODE_OCTETS_MAX4",
    ),
)


_FP8_DECODE_PLAN_DESCRIPTOR_ROWS = (
    _Fp8DecodePlanDescriptorRow(
        "amdgpu.v_bfe_u32.offset_width_inline",
        "bfe_u32_descriptor",
        "LOOM_AMDGPU_FP8_DECODE_PLAN_FLAG_HAS_BFE_U32",
    ),
    _Fp8DecodePlanDescriptorRow(
        "amdgpu.v_cmp_eq_i32.src1_inline",
        "compare_eq_i32_src1_inline_descriptor",
        "LOOM_AMDGPU_FP8_DECODE_PLAN_FLAG_NONE",
    ),
    _Fp8DecodePlanDescriptorRow(
        "amdgpu.v_cmp_ne_i32.src1_inline",
        "compare_ne_i32_src1_inline_descriptor",
        "LOOM_AMDGPU_FP8_DECODE_PLAN_FLAG_HAS_CMP_NE_I32_SRC1_INLINE",
    ),
    _Fp8DecodePlanDescriptorRow(
        "amdgpu.s_cmp_lg_u64",
        "compare_lg_u64_descriptor",
        "LOOM_AMDGPU_FP8_DECODE_PLAN_FLAG_HAS_CMP_LG_U64",
    ),
    _Fp8DecodePlanDescriptorRow(
        "amdgpu.s_cmp_lg_u64.src1_inline",
        "compare_lg_u64_src1_inline_descriptor",
        "LOOM_AMDGPU_FP8_DECODE_PLAN_FLAG_HAS_CMP_LG_U64_SRC1_INLINE",
    ),
    _Fp8DecodePlanDescriptorRow(
        "amdgpu.v_cmp_uge_u32.src1_inline",
        "compare_uge_u32_src1_inline_descriptor",
        "LOOM_AMDGPU_FP8_DECODE_PLAN_FLAG_NONE",
    ),
    _Fp8DecodePlanDescriptorRow(
        "amdgpu.v_cmp_ult_u32.src1_inline",
        "compare_ult_u32_src1_inline_descriptor",
        "LOOM_AMDGPU_FP8_DECODE_PLAN_FLAG_NONE",
    ),
    _Fp8DecodePlanDescriptorRow(
        "amdgpu.v_cvt_pk_u16_u32",
        "pack_u16_descriptor",
        "LOOM_AMDGPU_FP8_DECODE_PLAN_FLAG_HAS_PACK_U16",
    ),
    _Fp8DecodePlanDescriptorRow(
        "amdgpu.v_perm_b32",
        "perm_b32_descriptor",
        "LOOM_AMDGPU_FP8_DECODE_PLAN_FLAG_HAS_PERM_B32",
    ),
    _Fp8DecodePlanDescriptorRow(
        "amdgpu.v_cvt_pk_bf16_f32",
        "native_bf16_pack_descriptor",
        "LOOM_AMDGPU_FP8_DECODE_PLAN_FLAG_HAS_NATIVE_BF16_PACK",
    ),
    _Fp8DecodePlanDescriptorRow(
        "amdgpu.v_add3_u32.src2_lit",
        "add3_src2_literal_descriptor",
        "LOOM_AMDGPU_FP8_DECODE_PLAN_FLAG_HAS_ADD3_SRC2_LITERAL",
    ),
    _Fp8DecodePlanDescriptorRow(
        "amdgpu.v_lshl_add_u32.shift_imm",
        "lshl_add_u32_shift_imm_descriptor",
        "LOOM_AMDGPU_FP8_DECODE_PLAN_FLAG_HAS_LSHL_ADD_U32_SHIFT_IMM",
    ),
    _Fp8DecodePlanDescriptorRow(
        "amdgpu.v_lshl_add_u32.shift_imm.src2_lit",
        "lshl_add_u32_shift_imm_src2_literal_descriptor",
        "LOOM_AMDGPU_FP8_DECODE_PLAN_FLAG_HAS_LSHL_ADD_U32_SHIFT_IMM_SRC2_LITERAL",
    ),
    _Fp8DecodePlanDescriptorRow(
        "amdgpu.v_perm_b32.src2_lit",
        "perm_b32_src2_literal_descriptor",
        "LOOM_AMDGPU_FP8_DECODE_PLAN_FLAG_HAS_PERM_B32_SRC2_LITERAL",
    ),
    _Fp8DecodePlanDescriptorRow(
        "amdgpu.v_perm_b32.src1_zero_src2_lit",
        "perm_b32_src1_zero_src2_literal_descriptor",
        "LOOM_AMDGPU_FP8_DECODE_PLAN_FLAG_HAS_PERM_B32_SRC1_ZERO_SRC2_LIT",
    ),
    _Fp8DecodePlanDescriptorRow(
        "amdgpu.v_pk_min_u16",
        "pk_min_u16_descriptor",
        "LOOM_AMDGPU_FP8_DECODE_PLAN_FLAG_HAS_PK_MIN_U16",
    ),
    _Fp8DecodePlanDescriptorRow(
        "amdgpu.v_pk_mul_lo_u16",
        "pk_mul_lo_u16_descriptor",
        "LOOM_AMDGPU_FP8_DECODE_PLAN_FLAG_HAS_PK_MUL_LO_U16",
    ),
    _Fp8DecodePlanDescriptorRow(
        "amdgpu.v_pk_add_u16",
        "pk_add_u16_descriptor",
        "LOOM_AMDGPU_FP8_DECODE_PLAN_FLAG_HAS_PK_ADD_U16",
    ),
    _Fp8DecodePlanDescriptorRow(
        "amdgpu.v_pk_lshlrev_b16",
        "pk_lshlrev_b16_descriptor",
        "LOOM_AMDGPU_FP8_DECODE_PLAN_FLAG_HAS_PK_LSHLREV_B16",
    ),
    _Fp8DecodePlanDescriptorRow(
        "amdgpu.v_pk_lshrrev_b16",
        "pk_lshrrev_b16_descriptor",
        "LOOM_AMDGPU_FP8_DECODE_PLAN_FLAG_HAS_PK_LSHRREV_B16",
    ),
    _Fp8DecodePlanDescriptorRow(
        "amdgpu.v_pk_mul_f16",
        "pk_mul_f16_descriptor",
        "LOOM_AMDGPU_FP8_DECODE_PLAN_FLAG_HAS_PK_MUL_F16",
    ),
    _Fp8DecodePlanDescriptorRow(
        "amdgpu.v_pk_mad_u16",
        "pk_mad_u16_descriptor",
        "LOOM_AMDGPU_FP8_DECODE_PLAN_FLAG_HAS_PK_MAD_U16",
    ),
    _Fp8DecodePlanDescriptorRow(
        "amdgpu.v_pk_mad_u16.src2_lit",
        "pk_mad_u16_src2_literal_descriptor",
        "LOOM_AMDGPU_FP8_DECODE_PLAN_FLAG_HAS_PK_MAD_U16_SRC2_LITERAL",
    ),
    _Fp8DecodePlanDescriptorRow(
        "amdgpu.v_pk_max_u16",
        "pk_max_u16_descriptor",
        "LOOM_AMDGPU_FP8_DECODE_PLAN_FLAG_HAS_PK_MAX_U16",
    ),
    _Fp8DecodePlanDescriptorRow(
        "amdgpu.v_pk_ashrrev_i16",
        "pk_ashrrev_i16_descriptor",
        "LOOM_AMDGPU_FP8_DECODE_PLAN_FLAG_HAS_PK_ASHRREV_I16",
    ),
    _Fp8DecodePlanDescriptorRow(
        "amdgpu.v_bfi_b32",
        "bfi_b32_descriptor",
        "LOOM_AMDGPU_FP8_DECODE_PLAN_FLAG_HAS_BFI_B32",
    ),
    _Fp8DecodePlanDescriptorRow(
        "amdgpu.v_bfi_b32.src0_lit",
        "bfi_b32_src0_literal_descriptor",
        "LOOM_AMDGPU_FP8_DECODE_PLAN_FLAG_HAS_BFI_B32_SRC0_LITERAL",
    ),
)

_FP8_NATIVE_DESCRIPTOR_REF_ROWS = (
    _Fp8NativeDescriptorRefRow(
        "LOOM_VALUE_FACT_NUMERIC_FORMAT_F8_E5M2",
        ScalarTypeKind.F32,
        "amdgpu.v_cvt_f32_bf8.ocp",
        "amdgpu.v_cvt_pk_f32_bf8.ocp",
    ),
    _Fp8NativeDescriptorRefRow(
        "LOOM_VALUE_FACT_NUMERIC_FORMAT_F8_E4M3FN",
        ScalarTypeKind.F32,
        "amdgpu.v_cvt_f32_fp8.ocp",
        "amdgpu.v_cvt_pk_f32_fp8.ocp",
    ),
    _Fp8NativeDescriptorRefRow(
        "LOOM_VALUE_FACT_NUMERIC_FORMAT_F8_E4M3FNUZ",
        ScalarTypeKind.F32,
        "amdgpu.v_cvt_f32_fp8.fnuz",
        "amdgpu.v_cvt_pk_f32_fp8.fnuz",
    ),
    _Fp8NativeDescriptorRefRow(
        "LOOM_VALUE_FACT_NUMERIC_FORMAT_F8_E5M2FNUZ",
        ScalarTypeKind.F32,
        "amdgpu.v_cvt_f32_bf8.fnuz",
        "amdgpu.v_cvt_pk_f32_bf8.fnuz",
    ),
    _Fp8NativeDescriptorRefRow(
        "LOOM_VALUE_FACT_NUMERIC_FORMAT_F8_E5M2",
        ScalarTypeKind.F16,
        None,
        "amdgpu.v_cvt_pk_f16_bf8.ocp",
        tuple(f"amdgpu.v_cvt_f16_bf8.ocp.byte{byte_selector}" for byte_selector in range(4)),
    ),
    _Fp8NativeDescriptorRefRow(
        "LOOM_VALUE_FACT_NUMERIC_FORMAT_F8_E4M3FN",
        ScalarTypeKind.F16,
        None,
        "amdgpu.v_cvt_pk_f16_fp8.ocp",
        tuple(f"amdgpu.v_cvt_f16_fp8.ocp.byte{byte_selector}" for byte_selector in range(4)),
    ),
)

_FP8_SCALED_DESCRIPTOR_REF_ROWS = (
    _Fp8ScaledDescriptorRefRow(
        "LOOM_VALUE_FACT_NUMERIC_FORMAT_F8_E5M2",
        ScalarTypeKind.F16,
        "amdgpu.v_cvt_scalef32_pk_f16_bf8.ocp",
        "amdgpu.v_cvt_scale_pk8_f16_bf8.ocp",
    ),
    _Fp8ScaledDescriptorRefRow(
        "LOOM_VALUE_FACT_NUMERIC_FORMAT_F8_E4M3FN",
        ScalarTypeKind.F16,
        "amdgpu.v_cvt_scalef32_pk_f16_fp8.ocp",
        "amdgpu.v_cvt_scale_pk8_f16_fp8.ocp",
    ),
    _Fp8ScaledDescriptorRefRow(
        "LOOM_VALUE_FACT_NUMERIC_FORMAT_F8_E5M2",
        ScalarTypeKind.BF16,
        "amdgpu.v_cvt_scalef32_pk_bf16_bf8.ocp",
        "amdgpu.v_cvt_scale_pk8_bf16_bf8.ocp",
    ),
    _Fp8ScaledDescriptorRefRow(
        "LOOM_VALUE_FACT_NUMERIC_FORMAT_F8_E4M3FN",
        ScalarTypeKind.BF16,
        "amdgpu.v_cvt_scalef32_pk_bf16_fp8.ocp",
        "amdgpu.v_cvt_scale_pk8_bf16_fp8.ocp",
    ),
    _Fp8ScaledDescriptorRefRow(
        "LOOM_VALUE_FACT_NUMERIC_FORMAT_F8_E5M2",
        ScalarTypeKind.F32,
        "amdgpu.v_cvt_scalef32_pk_f32_bf8.ocp",
        "amdgpu.v_cvt_scale_pk8_f32_bf8.ocp",
    ),
    _Fp8ScaledDescriptorRefRow(
        "LOOM_VALUE_FACT_NUMERIC_FORMAT_F8_E4M3FN",
        ScalarTypeKind.F32,
        "amdgpu.v_cvt_scalef32_pk_f32_fp8.ocp",
        "amdgpu.v_cvt_scale_pk8_f32_fp8.ocp",
    ),
)


def _generated_header() -> list[str]:
    return [
        "// Copyright 2026 The IREE Authors",
        "//",
        "// Licensed under the Apache License v2.0 with LLVM Exceptions.",
        "// See https://llvm.org/LICENSE.txt for license information.",
        "// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception",
        "",
        *line_comment_header("//", generator=_GENERATOR),
        "",
    ]


def _scalar_type_constant_name(scalar_type: ScalarTypeKind) -> str:
    return f"LOOM_SCALAR_TYPE_{scalar_type.name}"


def _validate_unique_source_result_rows(
    table_name: str,
    rows: Sequence[_Fp8NativeDescriptorRefRow | _Fp8ScaledDescriptorRefRow],
) -> None:
    seen: set[tuple[str, ScalarTypeKind]] = set()
    duplicates: list[str] = []
    for row in rows:
        if row.source_format not in _FP8_FORMAT_INDEX_BY_NAME:
            raise ValueError(f"{table_name} contains unsupported source format: {row.source_format}")
        key = (row.source_format, row.result_type)
        if key in seen:
            duplicates.append(f"{row.source_format}->{row.result_type.name}")
        seen.add(key)
    if duplicates:
        raise ValueError(f"{table_name} contains duplicate source/result rows: " + ", ".join(sorted(duplicates)))


def _validate_unique_strings(
    table_name: str,
    value_name: str,
    values: Sequence[str],
) -> None:
    seen: set[str] = set()
    duplicates: list[str] = []
    for value in values:
        if value in seen:
            duplicates.append(value)
        seen.add(value)
    if duplicates:
        raise ValueError(f"{table_name} contains duplicate {value_name}: " + ", ".join(sorted(duplicates)))


def _validate_fp8_decode_plan_descriptor_rows(
    rows: Sequence[_Fp8DecodePlanDescriptorRow],
) -> None:
    table_name = "AMDGPU FP8 decode plan descriptor table"
    _validate_unique_strings(
        table_name,
        "plan fields",
        [row.plan_field for row in rows],
    )
    _validate_unique_strings(
        table_name,
        "descriptor keys",
        [row.descriptor_key for row in rows],
    )
    _validate_unique_strings(
        table_name,
        "capability flags",
        [row.present_flag for row in rows if row.present_flag != "LOOM_AMDGPU_FP8_DECODE_PLAN_FLAG_NONE"],
    )


def _validate_fp8_format_rows(rows: Sequence[_Fp8FormatRow]) -> None:
    table_name = "AMDGPU FP8 subnormal table"
    _validate_unique_strings(
        table_name,
        "source formats",
        [row.source_format for row in rows],
    )
    expected_formats = (
        "LOOM_VALUE_FACT_NUMERIC_FORMAT_F8_E4M3",
        "LOOM_VALUE_FACT_NUMERIC_FORMAT_F8_E5M2",
        "LOOM_VALUE_FACT_NUMERIC_FORMAT_F8_E4M3FN",
        "LOOM_VALUE_FACT_NUMERIC_FORMAT_F8_E4M3FNUZ",
        "LOOM_VALUE_FACT_NUMERIC_FORMAT_F8_E5M2FNUZ",
    )
    if tuple(row.source_format for row in rows) != expected_formats:
        names = ", ".join(row.source_format for row in rows)
        raise ValueError(f"{table_name} must cover dense FP8 format bits in order: expected " + ", ".join(expected_formats) + f"; got {names}")
    for row in rows:
        if row.exponent_bits + row.mantissa_bits != 7:
            raise ValueError(f"{table_name} row {row.source_format} must describe a signless 7-bit FP8 payload")
        if row.mantissa_bits not in (2, 3):
            raise ValueError(f"{table_name} row {row.source_format} has unsupported mantissa bits: {row.mantissa_bits}")
        if row.source_type not in (
            ScalarTypeKind.F8E4M3,
            ScalarTypeKind.F8E5M2,
        ):
            raise ValueError(f"{table_name} row {row.source_format} has unsupported storage type: {row.source_type.name}")


def _validate_fp8_encoded_operand_schema_requirement_rows(
    rows: Sequence[_Fp8EncodedOperandSchemaRequirementRow],
) -> None:
    table_name = "AMDGPU FP8 encoded operand schema requirement table"
    expected_kinds = (
        "LOOM_AMDGPU_FP8_ENCODED_OPERAND_SCHEMA_KIND_UNSCALED",
        "LOOM_AMDGPU_FP8_ENCODED_OPERAND_SCHEMA_KIND_SCALE_F32",
        "LOOM_AMDGPU_FP8_ENCODED_OPERAND_SCHEMA_KIND_SCALE_E8M0",
    )
    actual_kinds = tuple(row.kind for row in rows)
    if actual_kinds != expected_kinds:
        raise ValueError(f"{table_name} must cover dense schema kinds in order: expected " + ", ".join(expected_kinds) + "; got " + ", ".join(actual_kinds))
    for row in rows:
        if row.scale_operand_count < 0:
            raise ValueError(f"{table_name} row {row.kind} has negative scale operand count")


def _fp8_decode_plan_descriptor_initializer(
    row: _Fp8DecodePlanDescriptorRow,
    descriptor_ref_key_set: set[str] | None = None,
) -> str:
    descriptor_ref = required_descriptor_ref_constant_name(
        "AMDGPU FP8 decode plan descriptor table",
        row.descriptor_key,
        descriptor_ref_key_set,
    )
    return "\n".join(
        [
            "    {",
            f"        .descriptor_ref = {descriptor_ref},",
            "        .descriptor_offset =",
            f"            offsetof(loom_amdgpu_fp8_decode_plan_t, {row.plan_field}),",
            f"        .present_flag = {row.present_flag},",
            "    },",
        ]
    )


def _fp8_numeric_format_flags_expr(formats: Sequence[str]) -> str:
    if not formats:
        return "LOOM_VALUE_FACT_NUMERIC_FORMAT_NONE"
    return " | ".join(formats)


def _fp8_encoded_operand_schema_requirement_initializer(
    row: _Fp8EncodedOperandSchemaRequirementRow,
) -> str:
    return "\n".join(
        [
            f"    [{row.kind}] = {{",
            f"        .scale_format = {row.scale_format},",
            f"        .scale_topology = {row.scale_topology},",
            f"        .affine_policy = {row.affine_policy},",
            f"        .scale_operand_count = {row.scale_operand_count},",
            f"        .scale_group_mode = {row.scale_group_mode},",
            "    },",
        ]
    )


def _fp8_native_descriptor_ref_initializer(
    row_index: int,
    row: _Fp8NativeDescriptorRefRow,
    descriptor_ref_key_set: set[str] | None = None,
) -> str:
    if len(row.byte_select_descriptor_keys) not in (0, 4):
        raise ValueError(f"AMDGPU FP8 native conversion descriptor table byte-select family has {len(row.byte_select_descriptor_keys)} entries instead of 4")
    lane_descriptor_ref = (
        "LOOM_AMDGPU_DESCRIPTOR_REF_NONE"
        if row.lane_descriptor_key is None
        else required_descriptor_ref_constant_name(
            "AMDGPU FP8 native conversion descriptor table",
            row.lane_descriptor_key,
            descriptor_ref_key_set,
        )
    )
    pair_descriptor_ref = required_descriptor_ref_constant_name(
        "AMDGPU FP8 native conversion descriptor table",
        row.pair_descriptor_key,
        descriptor_ref_key_set,
    )
    byte_select_descriptor_refs = tuple(
        required_descriptor_ref_constant_name(
            "AMDGPU FP8 native conversion descriptor table",
            descriptor_key,
            descriptor_ref_key_set,
        )
        for descriptor_key in row.byte_select_descriptor_keys
    )
    if not byte_select_descriptor_refs:
        byte_select_descriptor_refs = ("LOOM_AMDGPU_DESCRIPTOR_REF_NONE",) * 4
    return "\n".join(
        [
            f"    [{row_index}] = {{",
            f"        .source_format = {row.source_format},",
            f"        .result_element_type = {_scalar_type_constant_name(row.result_type)},",
            "        .refs = {",
            f"            .lane = {lane_descriptor_ref},",
            f"            .pair = {pair_descriptor_ref},",
            "            .byte_select = {",
            f"                {', '.join(byte_select_descriptor_refs)},",
            "            },",
            "        },",
            "    },",
        ]
    )


def _fp8_native_descriptor_ref_index_initializer(
    row_index: int,
    row: _Fp8NativeDescriptorRefRow,
) -> str:
    return f"    [{_FP8_FORMAT_INDEX_BY_NAME[row.source_format]}][{_scalar_type_constant_name(row.result_type)}] = {row_index + 1},"


def _fp8_scaled_descriptor_ref_initializer(
    row_index: int,
    row: _Fp8ScaledDescriptorRefRow,
    descriptor_ref_key_set: set[str] | None = None,
) -> str:
    scalef32_pair_descriptor_ref = required_descriptor_ref_constant_name(
        "AMDGPU FP8 scaled conversion descriptor table",
        row.scalef32_pair_descriptor_key,
        descriptor_ref_key_set,
    )
    e8m0_pk8_descriptor_ref = required_descriptor_ref_constant_name(
        "AMDGPU FP8 scaled conversion descriptor table",
        row.e8m0_pk8_descriptor_key,
        descriptor_ref_key_set,
    )
    return "\n".join(
        [
            f"    [{row_index}] = {{",
            f"        .source_format = {row.source_format},",
            f"        .result_element_type = {_scalar_type_constant_name(row.result_type)},",
            "        .scalef32_pair_descriptor_ref =",
            f"            {scalef32_pair_descriptor_ref},",
            f"        .e8m0_pk8_descriptor_ref = {e8m0_pk8_descriptor_ref},",
            "    },",
        ]
    )


def _fp8_scaled_descriptor_ref_index_initializer(
    row_index: int,
    row: _Fp8ScaledDescriptorRefRow,
) -> str:
    return f"    [{_FP8_FORMAT_INDEX_BY_NAME[row.source_format]}][{_scalar_type_constant_name(row.result_type)}] = {row_index + 1},"


def _fp8_subnormal_bf16_payload(row: _Fp8FormatRow, mantissa: int) -> int:
    if mantissa == 0:
        return 0
    leading_index = 0
    for i in range(1, row.mantissa_bits):
        if mantissa & (1 << i):
            leading_index = i
    exponent = 128 - row.exponent_bias - row.mantissa_bits + leading_index
    fraction = (mantissa << (7 - leading_index)) & 0x7F
    return (exponent << 7) | fraction


def _fp8_subnormal_f16_payload(row: _Fp8FormatRow, mantissa: int) -> int:
    if mantissa == 0:
        return 0
    source_power = 1 - row.exponent_bias - row.mantissa_bits
    leading_index = 0
    for i in range(1, row.mantissa_bits):
        if mantissa & (1 << i):
            leading_index = i
    exponent = source_power + leading_index
    f16_exponent = exponent + 15
    if f16_exponent > 0:
        fraction = (mantissa << (10 - leading_index)) & 0x3FF
        return (f16_exponent << 10) | fraction

    subnormal_shift = source_power + 24
    assert subnormal_shift >= 0
    return mantissa << subnormal_shift


def _fp8_subnormal_table_word(row: _Fp8FormatRow, mantissa_base: int) -> int:
    if row.mantissa_bits != 2:
        return 0
    return _fp8_subnormal_bf16_payload(row, mantissa_base) | (_fp8_subnormal_bf16_payload(row, mantissa_base + 1) << 16)


def _fp8_subnormal_bf16_byte_table_word(row: _Fp8FormatRow, byte_index: int, mantissa_base: int) -> int:
    table_word = 0
    for i in range(4):
        payload = _fp8_subnormal_bf16_payload(row, mantissa_base + i)
        table_word |= ((payload >> (byte_index * 8)) & 0xFF) << (i * 8)
    return table_word


def _fp8_subnormal_f16_byte_table_word(row: _Fp8FormatRow, byte_index: int, mantissa_base: int) -> int:
    table_word = 0
    for i in range(4):
        payload = _fp8_subnormal_f16_payload(row, mantissa_base + i)
        table_word |= ((payload >> (byte_index * 8)) & 0xFF) << (i * 8)
    return table_word


def _hex_u32(value: int) -> str:
    return f"UINT32_C(0x{value:08X})"


def _fp8_format_initializer(row: _Fp8FormatRow) -> str:
    return "\n".join(
        [
            "    {",
            f"        .source_format = {row.source_format},",
            f"        .element_type = {_scalar_type_constant_name(row.source_type)},",
            "        .format = {",
            f"            .exponent_bits = {row.exponent_bits},",
            f"            .mantissa_bits = {row.mantissa_bits},",
            f"            .exponent_bias = {row.exponent_bias},",
            f"            .special_policy = {row.special_policy},",
            "        },",
            "    },",
        ]
    )


def _fp8_subnormal_table_initializer(row: _Fp8FormatRow) -> str:
    return "\n".join(
        [
            "    {",
            f"        .source_format = {row.source_format},",
            f"        .element_type = {_scalar_type_constant_name(row.source_type)},",
            "        .format = {",
            f"            .exponent_bits = {row.exponent_bits},",
            f"            .mantissa_bits = {row.mantissa_bits},",
            f"            .exponent_bias = {row.exponent_bias},",
            f"            .special_policy = {row.special_policy},",
            "        },",
            "        .subnormal_bf16_table_words = {",
            f"            {_hex_u32(_fp8_subnormal_table_word(row, 0))},",
            f"            {_hex_u32(_fp8_subnormal_table_word(row, 2))},",
            "        },",
            "        .subnormal_bf16_byte_table_words = {",
            "            {",
            f"                {_hex_u32(_fp8_subnormal_bf16_byte_table_word(row, 0, 0))},",
            f"                {_hex_u32(_fp8_subnormal_bf16_byte_table_word(row, 0, 4))},",
            "            },",
            "            {",
            f"                {_hex_u32(_fp8_subnormal_bf16_byte_table_word(row, 1, 0))},",
            f"                {_hex_u32(_fp8_subnormal_bf16_byte_table_word(row, 1, 4))},",
            "            },",
            "        },",
            "        .subnormal_f16_byte_table_words = {",
            "            {",
            f"                {_hex_u32(_fp8_subnormal_f16_byte_table_word(row, 0, 0))},",
            f"                {_hex_u32(_fp8_subnormal_f16_byte_table_word(row, 0, 4))},",
            "            },",
            "            {",
            f"                {_hex_u32(_fp8_subnormal_f16_byte_table_word(row, 1, 0))},",
            f"                {_hex_u32(_fp8_subnormal_f16_byte_table_word(row, 1, 4))},",
            "            },",
            "        },",
            "    },",
        ]
    )


def _fp8_encoded_operand_format_initializer(source_type: ScalarTypeKind, formats: Sequence[str]) -> str:
    return "\n".join(
        [
            f"    [{_scalar_type_constant_name(source_type)}] = {{",
            f"        .element_type = {_scalar_type_constant_name(source_type)},",
            "        .encoded_operand_formats =",
            f"            {_fp8_numeric_format_flags_expr(formats)},",
            "    },",
        ]
    )


def _fp8_packed_repair_expression(repair_mask: int) -> str:
    flag_names = [repair_bit.flag_name for repair_bit in _FP8_PACKED_REPAIR_BITS if repair_mask & repair_bit.bit_value]
    if not flag_names:
        return "LOOM_AMDGPU_FP8_PACKED_U16_REPAIR_NONE"
    return " | ".join(flag_names)


def _fp8_packed_repair_reason(base_name: str, repair_mask: int) -> str:
    suffixes = [repair_bit.suffix for repair_bit in _FP8_PACKED_REPAIR_BITS if repair_mask & repair_bit.bit_value]
    if not suffixes:
        return base_name
    return f"{base_name}_repair_{'_'.join(suffixes)}"


def _fp8_packed_repair_reason_initializer(base_name: str, repair_mask: int) -> str:
    return "\n".join(
        [
            f"    [{_fp8_packed_repair_expression(repair_mask)}] =",
            f'        IREE_SVL("{_fp8_packed_repair_reason(base_name, repair_mask)}"),',
        ]
    )


def _emit_fp8_subnormal_table_rows(
    rows: Sequence[_Fp8FormatRow] = _FP8_FORMAT_ROWS,
) -> str:
    _validate_fp8_format_rows(rows)
    return (
        "\n".join(
            [
                *_generated_header(),
                *(_fp8_subnormal_table_initializer(row) for row in rows),
            ]
        )
        + "\n"
    )


def _emit_fp8_decode_plan_descriptor_rows(
    rows: Sequence[_Fp8DecodePlanDescriptorRow] = _FP8_DECODE_PLAN_DESCRIPTOR_ROWS,
    descriptor_ref_key_set: set[str] | None = None,
) -> str:
    _validate_fp8_decode_plan_descriptor_rows(rows)
    known_refs = descriptor_ref_key_set if descriptor_ref_key_set is not None else set(amdgpu_descriptor_ref_keys())
    return (
        "\n".join(
            [
                *_generated_header(),
                *(_fp8_decode_plan_descriptor_initializer(row, known_refs) for row in rows),
            ]
        )
        + "\n"
    )


def _emit_fp8_encoded_operand_schema_requirement_rows(
    rows: Sequence[_Fp8EncodedOperandSchemaRequirementRow] = _FP8_ENCODED_OPERAND_SCHEMA_REQUIREMENT_ROWS,
) -> str:
    _validate_fp8_encoded_operand_schema_requirement_rows(rows)
    return (
        "\n".join(
            [
                *_generated_header(),
                *(_fp8_encoded_operand_schema_requirement_initializer(row) for row in rows),
            ]
        )
        + "\n"
    )


def _emit_fp8_native_descriptor_ref_rows(
    rows: Sequence[_Fp8NativeDescriptorRefRow] = _FP8_NATIVE_DESCRIPTOR_REF_ROWS,
    descriptor_ref_key_set: set[str] | None = None,
) -> str:
    known_refs = descriptor_ref_key_set if descriptor_ref_key_set is not None else set(amdgpu_descriptor_ref_keys())
    _validate_unique_source_result_rows("AMDGPU FP8 native conversion descriptor table", rows)
    return (
        "\n".join(
            [
                *_generated_header(),
                *(_fp8_native_descriptor_ref_initializer(i, row, known_refs) for i, row in enumerate(rows)),
            ]
        )
        + "\n"
    )


def _emit_fp8_scaled_descriptor_ref_rows(
    rows: Sequence[_Fp8ScaledDescriptorRefRow] = _FP8_SCALED_DESCRIPTOR_REF_ROWS,
    descriptor_ref_key_set: set[str] | None = None,
) -> str:
    known_refs = descriptor_ref_key_set if descriptor_ref_key_set is not None else set(amdgpu_descriptor_ref_keys())
    _validate_unique_source_result_rows("AMDGPU FP8 scaled conversion descriptor table", rows)
    return (
        "\n".join(
            [
                *_generated_header(),
                *(_fp8_scaled_descriptor_ref_initializer(i, row, known_refs) for i, row in enumerate(rows)),
            ]
        )
        + "\n"
    )


def _emit_counts_header() -> str:
    lines = [
        *_generated_header(),
        "#ifndef LOOM_TARGET_ARCH_AMDGPU_LOWER_ENCODING_FP8_TABLE_COUNTS_H_",
        "#define LOOM_TARGET_ARCH_AMDGPU_LOWER_ENCODING_FP8_TABLE_COUNTS_H_",
        "",
        f"#define LOOM_AMDGPU_FP8_FORMAT_ROW_COUNT {len(_FP8_FORMAT_ROWS)}u",
        f"#define LOOM_AMDGPU_FP8_NATIVE_DESCRIPTOR_REF_ROW_COUNT {len(_FP8_NATIVE_DESCRIPTOR_REF_ROWS)}u",
        f"#define LOOM_AMDGPU_FP8_SCALED_DESCRIPTOR_REF_ROW_COUNT {len(_FP8_SCALED_DESCRIPTOR_REF_ROWS)}u",
        f"#define LOOM_AMDGPU_FP8_DECODE_PLAN_DESCRIPTOR_ROW_COUNT {len(_FP8_DECODE_PLAN_DESCRIPTOR_ROWS)}u",
        "",
        "#endif  // LOOM_TARGET_ARCH_AMDGPU_LOWER_ENCODING_FP8_TABLE_COUNTS_H_",
    ]
    return "\n".join(lines) + "\n"


def _emit_source(*, public_header: str) -> str:
    _validate_fp8_format_rows(_FP8_FORMAT_ROWS)
    _validate_fp8_encoded_operand_schema_requirement_rows(_FP8_ENCODED_OPERAND_SCHEMA_REQUIREMENT_ROWS)
    _validate_fp8_decode_plan_descriptor_rows(_FP8_DECODE_PLAN_DESCRIPTOR_ROWS)
    _validate_unique_source_result_rows(
        "AMDGPU FP8 native conversion descriptor table",
        _FP8_NATIVE_DESCRIPTOR_REF_ROWS,
    )
    _validate_unique_source_result_rows(
        "AMDGPU FP8 scaled conversion descriptor table",
        _FP8_SCALED_DESCRIPTOR_REF_ROWS,
    )
    descriptor_ref_key_set = set(amdgpu_descriptor_ref_keys())
    source_types = (ScalarTypeKind.F8E4M3, ScalarTypeKind.F8E5M2)
    lines = [
        *_generated_header(),
        f'#include "{public_header}"',
        "",
        "const iree_string_view_t kLoomAmdgpuFp8PackedBf16RepairReasons",
        "    [LOOM_AMDGPU_FP8_PACKED_U16_REPAIR_REASON_COUNT] = {",
        *(_fp8_packed_repair_reason_initializer("fp8_packed_bf16_decode", repair_mask) for repair_mask in range(16)),
        "};",
        "",
        "const iree_string_view_t kLoomAmdgpuFp8PackedF16RepairReasons",
        "    [LOOM_AMDGPU_FP8_PACKED_F16_REPAIR_REASON_COUNT] = {",
        *(_fp8_packed_repair_reason_initializer("fp8_packed_f16_decode", repair_mask) for repair_mask in range(4)),
        "};",
        "",
        "const loom_amdgpu_fp8_encoded_operand_schema_requirement_t",
        "    kLoomAmdgpuFp8EncodedOperandSchemaRequirements",
        "        [LOOM_AMDGPU_FP8_ENCODED_OPERAND_SCHEMA_KIND_SCALE_E8M0 + 1] = {",
        *(_fp8_encoded_operand_schema_requirement_initializer(row) for row in _FP8_ENCODED_OPERAND_SCHEMA_REQUIREMENT_ROWS),
        "};",
        "",
        "const loom_amdgpu_fp8_encoded_operand_format_row_t",
        "    kLoomAmdgpuFp8EncodedOperandFormatRows[LOOM_SCALAR_TYPE_COUNT_] = {",
        *(
            _fp8_encoded_operand_format_initializer(
                source_type,
                tuple(row.source_format for row in _FP8_FORMAT_ROWS if row.source_type == source_type),
            )
            for source_type in source_types
        ),
        "};",
        "",
        "const loom_amdgpu_fp8_format_row_t kLoomAmdgpuFp8FormatRows",
        "    [LOOM_AMDGPU_FP8_FORMAT_ROW_COUNT] = {",
        *(_fp8_format_initializer(row) for row in _FP8_FORMAT_ROWS),
        "};",
        "",
        "const loom_amdgpu_fp8_subnormal_table_row_t",
        "    kLoomAmdgpuFp8SubnormalTableRows",
        "        [LOOM_AMDGPU_FP8_FORMAT_ROW_COUNT] = {",
        *(_fp8_subnormal_table_initializer(row) for row in _FP8_FORMAT_ROWS),
        "};",
        "",
        "const loom_amdgpu_fp8_native_descriptor_ref_row_t",
        "    kLoomAmdgpuFp8NativeDescriptorRefRows",
        "        [LOOM_AMDGPU_FP8_NATIVE_DESCRIPTOR_REF_ROW_COUNT] = {",
        *(_fp8_native_descriptor_ref_initializer(row_index, row, descriptor_ref_key_set) for row_index, row in enumerate(_FP8_NATIVE_DESCRIPTOR_REF_ROWS)),
        "};",
        "",
        "const uint8_t kLoomAmdgpuFp8NativeDescriptorRefRowIndex",
        "    [LOOM_AMDGPU_FP8_FORMAT_ROW_COUNT][LOOM_SCALAR_TYPE_COUNT_] = {",
        *(_fp8_native_descriptor_ref_index_initializer(row_index, row) for row_index, row in enumerate(_FP8_NATIVE_DESCRIPTOR_REF_ROWS)),
        "};",
        "",
        "const loom_amdgpu_fp8_scaled_descriptor_ref_row_t",
        "    kLoomAmdgpuFp8ScaledDescriptorRefRows",
        "        [LOOM_AMDGPU_FP8_SCALED_DESCRIPTOR_REF_ROW_COUNT] = {",
        *(_fp8_scaled_descriptor_ref_initializer(row_index, row, descriptor_ref_key_set) for row_index, row in enumerate(_FP8_SCALED_DESCRIPTOR_REF_ROWS)),
        "};",
        "",
        "const uint8_t kLoomAmdgpuFp8ScaledDescriptorRefRowIndex",
        "    [LOOM_AMDGPU_FP8_FORMAT_ROW_COUNT][LOOM_SCALAR_TYPE_COUNT_] = {",
        *(_fp8_scaled_descriptor_ref_index_initializer(row_index, row) for row_index, row in enumerate(_FP8_SCALED_DESCRIPTOR_REF_ROWS)),
        "};",
        "",
        "const loom_amdgpu_fp8_decode_plan_descriptor_row_t",
        "    kLoomAmdgpuFp8DecodePlanDescriptorRows",
        "        [LOOM_AMDGPU_FP8_DECODE_PLAN_DESCRIPTOR_ROW_COUNT] = {",
        *(_fp8_decode_plan_descriptor_initializer(row, descriptor_ref_key_set) for row in _FP8_DECODE_PLAN_DESCRIPTOR_ROWS),
        "};",
    ]
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate AMDGPU narrow-float lowering tables.")
    parser.add_argument(
        "--counts-header",
        required=True,
        type=Path,
        help="Generated FP8 table count header path.",
    )
    parser.add_argument(
        "--source",
        required=True,
        type=Path,
        help="Generated FP8 table source path.",
    )
    parser.add_argument(
        "--public-header",
        default="loom/target/arch/amdgpu/lower/encoding/fp8_tables.h",
        help="Public include path for the generated source.",
    )
    args = parser.parse_args(argv)

    write_text_file(args.counts_header, _emit_counts_header())
    write_text_file(args.source, _emit_source(public_header=args.public_header))
    return 0


if __name__ == "__main__":
    sys.exit(main())
