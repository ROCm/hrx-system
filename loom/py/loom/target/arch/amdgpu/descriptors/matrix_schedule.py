# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""AMDGPU matrix scheduling-model facts."""

from __future__ import annotations

from dataclasses import dataclass

from ..matrix_formats import AMDGPU_F8F6F4_MATRIX_PHYSICAL_FORMATS


@dataclass(frozen=True, slots=True)
class _AmdgpuMatrixTiming:
    """LLVM scheduling-model timing for one AMDGPU matrix instruction."""

    latency_cycles: int
    reciprocal_throughput_cycles: int


def _f8f6f4_physical_timing_rows(
    descriptor_key_prefix: str, timing: _AmdgpuMatrixTiming
) -> dict[str, _AmdgpuMatrixTiming]:
    return {
        f"{descriptor_key_prefix}_{lhs_format.token}_{rhs_format.token}": timing
        for lhs_format in AMDGPU_F8F6F4_MATRIX_PHYSICAL_FORMATS
        for rhs_format in AMDGPU_F8F6F4_MATRIX_PHYSICAL_FORMATS
    }


# These are the matrix operations that LLVM exposes on gfx9-4-generic. The
# gfx942 and gfx950 rows come from the ROCm LLVM scheduling models; the generic
# rows conservatively take the larger member value for each field.
_AMDGPU_GFX942_PORTABLE_MATRIX_TIMINGS = {
    "amdgpu.v_mfma_f32_16x16x1_4b_f32": _AmdgpuMatrixTiming(12, 8),
    "amdgpu.v_mfma_f32_16x16x4_4b_f16": _AmdgpuMatrixTiming(12, 8),
    "amdgpu.v_mfma_f32_16x16x4_f32": _AmdgpuMatrixTiming(12, 8),
    "amdgpu.v_mfma_f32_16x16x4_4b_bf16": _AmdgpuMatrixTiming(12, 8),
    "amdgpu.v_mfma_f32_16x16x16_f16": _AmdgpuMatrixTiming(8, 4),
    "amdgpu.v_mfma_f32_16x16x16_bf16": _AmdgpuMatrixTiming(8, 4),
    "amdgpu.v_mfma_f32_32x32x1_2b_f32": _AmdgpuMatrixTiming(20, 16),
    "amdgpu.v_mfma_f32_32x32x2_f32": _AmdgpuMatrixTiming(20, 16),
    "amdgpu.v_mfma_f32_32x32x4_2b_f16": _AmdgpuMatrixTiming(20, 16),
    "amdgpu.v_mfma_f32_32x32x4_2b_bf16": _AmdgpuMatrixTiming(20, 16),
    "amdgpu.v_mfma_f32_32x32x8_f16": _AmdgpuMatrixTiming(12, 8),
    "amdgpu.v_mfma_f32_32x32x8_bf16": _AmdgpuMatrixTiming(12, 8),
    "amdgpu.v_mfma_f32_4x4x1_16b_f32": _AmdgpuMatrixTiming(6, 2),
    "amdgpu.v_mfma_f32_4x4x4_16b_f16": _AmdgpuMatrixTiming(6, 2),
    "amdgpu.v_mfma_f32_4x4x4_16b_bf16": _AmdgpuMatrixTiming(6, 2),
    "amdgpu.v_mfma_f64_16x16x4_f64": _AmdgpuMatrixTiming(12, 8),
    "amdgpu.v_mfma_f64_4x4x4_4b_f64": _AmdgpuMatrixTiming(8, 4),
    "amdgpu.v_mfma_i32_16x16x4_4b_i8": _AmdgpuMatrixTiming(12, 8),
    "amdgpu.v_mfma_i32_16x16x32_i8": _AmdgpuMatrixTiming(8, 4),
    "amdgpu.v_mfma_i32_32x32x4_2b_i8": _AmdgpuMatrixTiming(20, 16),
    "amdgpu.v_mfma_i32_32x32x16_i8": _AmdgpuMatrixTiming(12, 8),
    "amdgpu.v_mfma_i32_4x4x4_16b_i8": _AmdgpuMatrixTiming(6, 2),
    "amdgpu.v_smfmac_f32_16x16x32_f16": _AmdgpuMatrixTiming(8, 4),
    "amdgpu.v_smfmac_f32_16x16x32_bf16": _AmdgpuMatrixTiming(8, 4),
    "amdgpu.v_smfmac_i32_16x16x64_i8": _AmdgpuMatrixTiming(8, 4),
    "amdgpu.v_smfmac_f32_32x32x16_f16": _AmdgpuMatrixTiming(12, 8),
    "amdgpu.v_smfmac_f32_32x32x16_bf16": _AmdgpuMatrixTiming(12, 8),
    "amdgpu.v_smfmac_i32_32x32x32_i8": _AmdgpuMatrixTiming(12, 8),
}

_AMDGPU_GFX9_4_PACKED8_MATRIX_TIMINGS = {
    "amdgpu.v_mfma_f32_16x16x32_bf8_bf8": _AmdgpuMatrixTiming(8, 4),
    "amdgpu.v_mfma_f32_16x16x32_bf8_fp8": _AmdgpuMatrixTiming(8, 4),
    "amdgpu.v_mfma_f32_16x16x32_fp8_bf8": _AmdgpuMatrixTiming(8, 4),
    "amdgpu.v_mfma_f32_16x16x32_fp8_fp8": _AmdgpuMatrixTiming(8, 4),
    "amdgpu.v_mfma_f32_32x32x16_bf8_bf8": _AmdgpuMatrixTiming(12, 8),
    "amdgpu.v_mfma_f32_32x32x16_bf8_fp8": _AmdgpuMatrixTiming(12, 8),
    "amdgpu.v_mfma_f32_32x32x16_fp8_bf8": _AmdgpuMatrixTiming(12, 8),
    "amdgpu.v_mfma_f32_32x32x16_fp8_fp8": _AmdgpuMatrixTiming(12, 8),
    "amdgpu.v_smfmac_f32_16x16x64_bf8_bf8": _AmdgpuMatrixTiming(8, 4),
    "amdgpu.v_smfmac_f32_16x16x64_bf8_fp8": _AmdgpuMatrixTiming(8, 4),
    "amdgpu.v_smfmac_f32_16x16x64_fp8_bf8": _AmdgpuMatrixTiming(8, 4),
    "amdgpu.v_smfmac_f32_16x16x64_fp8_fp8": _AmdgpuMatrixTiming(8, 4),
    "amdgpu.v_smfmac_f32_32x32x32_bf8_bf8": _AmdgpuMatrixTiming(12, 8),
    "amdgpu.v_smfmac_f32_32x32x32_bf8_fp8": _AmdgpuMatrixTiming(12, 8),
    "amdgpu.v_smfmac_f32_32x32x32_fp8_bf8": _AmdgpuMatrixTiming(12, 8),
    "amdgpu.v_smfmac_f32_32x32x32_fp8_fp8": _AmdgpuMatrixTiming(12, 8),
}

_AMDGPU_GFX942_MATRIX_TIMINGS = {
    **_AMDGPU_GFX942_PORTABLE_MATRIX_TIMINGS,
    **_AMDGPU_GFX9_4_PACKED8_MATRIX_TIMINGS,
    "amdgpu.v_mfma_f32_16x16x8_xf32": _AmdgpuMatrixTiming(8, 4),
    "amdgpu.v_mfma_f32_32x32x4_xf32": _AmdgpuMatrixTiming(12, 8),
}

_AMDGPU_GFX950_MATRIX_TIMINGS = {
    **_AMDGPU_GFX942_PORTABLE_MATRIX_TIMINGS,
    "amdgpu.v_mfma_f64_16x16x4_f64": _AmdgpuMatrixTiming(20, 16),
    **_AMDGPU_GFX9_4_PACKED8_MATRIX_TIMINGS,
    "amdgpu.v_mfma_f32_16x16x32_f16": _AmdgpuMatrixTiming(8, 4),
    "amdgpu.v_mfma_f32_16x16x32_bf16": _AmdgpuMatrixTiming(8, 4),
    "amdgpu.v_mfma_f32_32x32x16_f16": _AmdgpuMatrixTiming(12, 8),
    "amdgpu.v_mfma_f32_32x32x16_bf16": _AmdgpuMatrixTiming(12, 8),
    "amdgpu.v_mfma_i32_16x16x64_i8": _AmdgpuMatrixTiming(8, 4),
    "amdgpu.v_mfma_i32_32x32x32_i8": _AmdgpuMatrixTiming(12, 8),
    **_f8f6f4_physical_timing_rows(
        "amdgpu.v_mfma_f32_16x16x128_f8f6f4",
        _AmdgpuMatrixTiming(8, 4),
    ),
    **_f8f6f4_physical_timing_rows(
        "amdgpu.v_mfma_f32_32x32x64_f8f6f4",
        _AmdgpuMatrixTiming(12, 8),
    ),
    **_f8f6f4_physical_timing_rows(
        "amdgpu.v_mfma_scale_f32_16x16x128_f8f6f4",
        _AmdgpuMatrixTiming(8, 4),
    ),
    **_f8f6f4_physical_timing_rows(
        "amdgpu.v_mfma_scale_f32_32x32x64_f8f6f4",
        _AmdgpuMatrixTiming(12, 8),
    ),
    "amdgpu.v_smfmac_f32_16x16x64_f16": _AmdgpuMatrixTiming(8, 4),
    "amdgpu.v_smfmac_f32_16x16x64_bf16": _AmdgpuMatrixTiming(8, 4),
    "amdgpu.v_smfmac_f32_16x16x128_bf8_bf8": _AmdgpuMatrixTiming(8, 4),
    "amdgpu.v_smfmac_f32_16x16x128_bf8_fp8": _AmdgpuMatrixTiming(8, 4),
    "amdgpu.v_smfmac_f32_16x16x128_fp8_bf8": _AmdgpuMatrixTiming(8, 4),
    "amdgpu.v_smfmac_f32_16x16x128_fp8_fp8": _AmdgpuMatrixTiming(8, 4),
    "amdgpu.v_smfmac_i32_16x16x128_i8": _AmdgpuMatrixTiming(8, 4),
    "amdgpu.v_smfmac_f32_32x32x32_f16": _AmdgpuMatrixTiming(12, 8),
    "amdgpu.v_smfmac_f32_32x32x32_bf16": _AmdgpuMatrixTiming(12, 8),
    "amdgpu.v_smfmac_f32_32x32x64_bf8_bf8": _AmdgpuMatrixTiming(12, 8),
    "amdgpu.v_smfmac_f32_32x32x64_bf8_fp8": _AmdgpuMatrixTiming(12, 8),
    "amdgpu.v_smfmac_f32_32x32x64_fp8_bf8": _AmdgpuMatrixTiming(12, 8),
    "amdgpu.v_smfmac_f32_32x32x64_fp8_fp8": _AmdgpuMatrixTiming(12, 8),
    "amdgpu.v_smfmac_i32_32x32x64_i8": _AmdgpuMatrixTiming(12, 8),
}

_AMDGPU_GFX9_4_GENERIC_MATRIX_TIMINGS = {
    descriptor_key: _AmdgpuMatrixTiming(
        latency_cycles=max(
            gfx942_timing.latency_cycles,
            _AMDGPU_GFX950_MATRIX_TIMINGS[descriptor_key].latency_cycles,
        ),
        reciprocal_throughput_cycles=max(
            gfx942_timing.reciprocal_throughput_cycles,
            _AMDGPU_GFX950_MATRIX_TIMINGS[descriptor_key].reciprocal_throughput_cycles,
        ),
    )
    for descriptor_key, gfx942_timing in (
        _AMDGPU_GFX942_PORTABLE_MATRIX_TIMINGS.items()
    )
}


def _amdgpu_mfma_has_qualified_timing(descriptor_key: str) -> bool:
    return (
        descriptor_key in _AMDGPU_GFX942_MATRIX_TIMINGS
        or descriptor_key in _AMDGPU_GFX950_MATRIX_TIMINGS
    )


__all__ = (
    "_AmdgpuMatrixTiming",
    "_AMDGPU_GFX942_MATRIX_TIMINGS",
    "_AMDGPU_GFX950_MATRIX_TIMINGS",
    "_AMDGPU_GFX9_4_GENERIC_MATRIX_TIMINGS",
    "_amdgpu_mfma_has_qualified_timing",
)
