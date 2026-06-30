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
    source_type: ScalarTypeKind
    result_type: ScalarTypeKind
    lane_descriptor_key: str | None
    pair_descriptor_key: str


@dataclass(frozen=True)
class _Fp8ScaleF32DescriptorRefRow:
    source_type: ScalarTypeKind
    result_type: ScalarTypeKind
    descriptor_key: str


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
        ScalarTypeKind.F8E4M3,
        ScalarTypeKind.F32,
        "amdgpu.v_cvt_f32_fp8",
        "amdgpu.v_cvt_pk_f32_fp8",
    ),
    _Fp8NativeDescriptorRefRow(
        ScalarTypeKind.F8E5M2,
        ScalarTypeKind.F32,
        "amdgpu.v_cvt_f32_bf8",
        "amdgpu.v_cvt_pk_f32_bf8",
    ),
    _Fp8NativeDescriptorRefRow(
        ScalarTypeKind.F8E4M3,
        ScalarTypeKind.F16,
        None,
        "amdgpu.v_cvt_pk_f16_fp8",
    ),
    _Fp8NativeDescriptorRefRow(
        ScalarTypeKind.F8E5M2,
        ScalarTypeKind.F16,
        None,
        "amdgpu.v_cvt_pk_f16_bf8",
    ),
)

_FP8_SCALEF32_DESCRIPTOR_REF_ROWS = (
    _Fp8ScaleF32DescriptorRefRow(
        ScalarTypeKind.F8E4M3,
        ScalarTypeKind.F16,
        "amdgpu.v_cvt_scalef32_pk_f16_fp8",
    ),
    _Fp8ScaleF32DescriptorRefRow(
        ScalarTypeKind.F8E5M2,
        ScalarTypeKind.F16,
        "amdgpu.v_cvt_scalef32_pk_f16_bf8",
    ),
    _Fp8ScaleF32DescriptorRefRow(
        ScalarTypeKind.F8E4M3,
        ScalarTypeKind.BF16,
        "amdgpu.v_cvt_scalef32_pk_bf16_fp8",
    ),
    _Fp8ScaleF32DescriptorRefRow(
        ScalarTypeKind.F8E5M2,
        ScalarTypeKind.BF16,
        "amdgpu.v_cvt_scalef32_pk_bf16_bf8",
    ),
    _Fp8ScaleF32DescriptorRefRow(
        ScalarTypeKind.F8E4M3,
        ScalarTypeKind.F32,
        "amdgpu.v_cvt_scalef32_pk_f32_fp8",
    ),
    _Fp8ScaleF32DescriptorRefRow(
        ScalarTypeKind.F8E5M2,
        ScalarTypeKind.F32,
        "amdgpu.v_cvt_scalef32_pk_f32_bf8",
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
            "LOOM_AMDGPU_FP8_DECODE_PLAN_DESCRIPTOR_ROW(",
            f"    {descriptor_ref}, {row.plan_field},",
            f"    {row.present_flag}),",
        ]
    )


def _fp8_native_descriptor_ref_initializer(
    row: _Fp8NativeDescriptorRefRow,
    descriptor_ref_key_set: set[str] | None = None,
) -> str:
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
    return "\n".join(
        [
            "LOOM_AMDGPU_FP8_NATIVE_DESCRIPTOR_REF_ROW(",
            f"    {_scalar_type_constant_name(row.source_type)},",
            f"    {_scalar_type_constant_name(row.result_type)},",
            f"    {lane_descriptor_ref}, {pair_descriptor_ref}),",
        ]
    )


def _fp8_scalef32_descriptor_ref_initializer(
    row: _Fp8ScaleF32DescriptorRefRow,
    descriptor_ref_key_set: set[str] | None = None,
) -> str:
    descriptor_ref = required_descriptor_ref_constant_name(
        "AMDGPU FP8 scaleF32 descriptor table",
        row.descriptor_key,
        descriptor_ref_key_set,
    )
    return "\n".join(
        [
            "LOOM_AMDGPU_FP8_SCALEF32_DESCRIPTOR_REF_ROW(",
            f"    {_scalar_type_constant_name(row.source_type)},",
            f"    {_scalar_type_constant_name(row.result_type)},",
            f"    {descriptor_ref}),",
        ]
    )


def _emit_fp8_decode_plan_descriptor_rows(
    rows: Sequence[_Fp8DecodePlanDescriptorRow] = _FP8_DECODE_PLAN_DESCRIPTOR_ROWS,
    descriptor_ref_key_set: set[str] | None = None,
) -> str:
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


def _emit_fp8_native_descriptor_ref_rows(
    rows: Sequence[_Fp8NativeDescriptorRefRow] = _FP8_NATIVE_DESCRIPTOR_REF_ROWS,
    descriptor_ref_key_set: set[str] | None = None,
) -> str:
    known_refs = descriptor_ref_key_set if descriptor_ref_key_set is not None else set(amdgpu_descriptor_ref_keys())
    return (
        "\n".join(
            [
                *_generated_header(),
                *(_fp8_native_descriptor_ref_initializer(row, known_refs) for row in rows),
            ]
        )
        + "\n"
    )


def _emit_fp8_scalef32_descriptor_ref_rows(
    rows: Sequence[_Fp8ScaleF32DescriptorRefRow] = _FP8_SCALEF32_DESCRIPTOR_REF_ROWS,
    descriptor_ref_key_set: set[str] | None = None,
) -> str:
    known_refs = descriptor_ref_key_set if descriptor_ref_key_set is not None else set(amdgpu_descriptor_ref_keys())
    return (
        "\n".join(
            [
                *_generated_header(),
                *(_fp8_scalef32_descriptor_ref_initializer(row, known_refs) for row in rows),
            ]
        )
        + "\n"
    )


def _write_output(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents, encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate AMDGPU narrow-float lowering tables.")
    parser.add_argument(
        "--fp8-decode-plan-descriptor-rows",
        type=Path,
        help="Generated FP8 decode-plan descriptor row fragment path.",
    )
    parser.add_argument(
        "--fp8-native-descriptor-ref-rows",
        type=Path,
        help="Generated native unscaled FP8/BF8 descriptor row fragment path.",
    )
    parser.add_argument(
        "--fp8-scalef32-descriptor-ref-rows",
        type=Path,
        help="Generated FP8/BF8 scaleF32 descriptor row fragment path.",
    )
    args = parser.parse_args(argv)

    if args.fp8_decode_plan_descriptor_rows is None and args.fp8_native_descriptor_ref_rows is None and args.fp8_scalef32_descriptor_ref_rows is None:
        parser.error("at least one output path is required")
    if args.fp8_decode_plan_descriptor_rows is not None:
        _write_output(
            args.fp8_decode_plan_descriptor_rows,
            _emit_fp8_decode_plan_descriptor_rows(),
        )
    if args.fp8_native_descriptor_ref_rows is not None:
        _write_output(
            args.fp8_native_descriptor_ref_rows,
            _emit_fp8_native_descriptor_ref_rows(),
        )
    if args.fp8_scalef32_descriptor_ref_rows is not None:
        _write_output(
            args.fp8_scalef32_descriptor_ref_rows,
            _emit_fp8_scalef32_descriptor_ref_rows(),
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
