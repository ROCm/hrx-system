// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU matrix contract type definitions.
//
// This file describes the target-native matrix primitives that Loom can select
// from a higher-level tile.contract after shapes, encodings, layouts, and value
// facts are refined enough to make the choice structural. The descriptors are
// intentionally data-only: lowering code can query exact shape/type/feature
// requirements without hard-coding AMDGPU intrinsic names throughout generic
// tile/vector passes.

#ifndef LOOM_TARGET_ARCH_AMDGPU_MATRIX_TYPES_H_
#define LOOM_TARGET_ARCH_AMDGPU_MATRIX_TYPES_H_

#include "iree/base/api.h"
#include "loom/analysis/contract_roles.h"
#include "loom/analysis/matrix_fragment_layout.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"
#include "loom/target/arch/amdgpu/target_info_defs.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum loom_amdgpu_matrix_feature_flag_bits_e {
  // Processor supports the common gfx908-era MFMA subset.
  LOOM_AMDGPU_MATRIX_FEATURE_MFMA_GFX908 = UINT64_C(1) << 0,
  // Processor supports MFMA variants gated to gfx908 and gfx90a.
  LOOM_AMDGPU_MATRIX_FEATURE_MFMA_GFX908_GFX90A = UINT64_C(1) << 1,
  // Processor supports gfx90a BF16 1k MFMA variants.
  LOOM_AMDGPU_MATRIX_FEATURE_MFMA_GFX90A_BF16_1K = UINT64_C(1) << 2,
  // Processor supports gfx90a F64 MFMA variants.
  LOOM_AMDGPU_MATRIX_FEATURE_MFMA_GFX90A_F64 = UINT64_C(1) << 3,
  // Processor supports gfx940 FP8/BF8 MFMA variants.
  LOOM_AMDGPU_MATRIX_FEATURE_MFMA_GFX940_FP8 = UINT64_C(1) << 4,
  // Processor supports gfx950 F16/BF16/I8 MFMA shape variants.
  LOOM_AMDGPU_MATRIX_FEATURE_MFMA_GFX950 = UINT64_C(1) << 5,
  // Processor supports gfx950 scaled F8/F6/F4 MFMA variants.
  LOOM_AMDGPU_MATRIX_FEATURE_MFMA_GFX950_SCALE_F8F6F4 = UINT64_C(1) << 6,
  // Processor supports gfx940 F16/BF16/I8 sparse MFMA variants.
  LOOM_AMDGPU_MATRIX_FEATURE_SMFMAC_GFX940 = UINT64_C(1) << 7,
  // Processor supports gfx950 sparse MFMA accumulate variants.
  LOOM_AMDGPU_MATRIX_FEATURE_SMFMAC_GFX950 = UINT64_C(1) << 8,
  // Processor supports gfx11 WMMA variants.
  LOOM_AMDGPU_MATRIX_FEATURE_WMMA_GFX11 = UINT64_C(1) << 9,
  // Processor supports gfx12 WMMA FP8/BF8/IU4 variants.
  LOOM_AMDGPU_MATRIX_FEATURE_WMMA_GFX12 = UINT64_C(1) << 10,
  // Processor supports gfx12 SWMMAC sparse variants.
  LOOM_AMDGPU_MATRIX_FEATURE_SWMMAC_GFX12 = UINT64_C(1) << 11,
  // Processor supports gfx1250 WMMA modifier/reuse variants.
  LOOM_AMDGPU_MATRIX_FEATURE_WMMA_GFX1250 = UINT64_C(1) << 12,
  // Processor supports gfx1250 WMMA scaled F8/F6/F4 and F4 variants.
  LOOM_AMDGPU_MATRIX_FEATURE_WMMA_GFX1250_SCALE_F8F6F4 = UINT64_C(1) << 13,
  // Processor supports gfx1250 SWMMAC modifier/reuse variants.
  LOOM_AMDGPU_MATRIX_FEATURE_SWMMAC_GFX1250 = UINT64_C(1) << 14,
  // Processor supports gfx940 XF32 MFMA variants.
  LOOM_AMDGPU_MATRIX_FEATURE_MFMA_GFX940_XF32 = UINT64_C(1) << 15,
  // Processor supports gfx940 FP8/BF8 sparse MFMA variants.
  LOOM_AMDGPU_MATRIX_FEATURE_SMFMAC_GFX940_FP8 = UINT64_C(1) << 16,
  // Processor supports gfx940 I8 MFMA shape variants.
  LOOM_AMDGPU_MATRIX_FEATURE_MFMA_GFX940_I8 = UINT64_C(1) << 17,
} loom_amdgpu_matrix_feature_flag_bits_t;

// Bitset of loom_amdgpu_matrix_feature_flag_bits_t values.
typedef uint64_t loom_amdgpu_matrix_feature_bits_t;

typedef struct loom_amdgpu_matrix_feature_info_t {
  // Single matrix feature bit represented by |name|.
  loom_amdgpu_matrix_feature_bits_t feature_bit;
  // Stable capability/report name for |feature_bit|.
  iree_string_view_t name;
} loom_amdgpu_matrix_feature_info_t;

enum loom_amdgpu_matrix_wave_size_bits_e {
  // Contract may be selected for wave32 code generation.
  LOOM_AMDGPU_MATRIX_WAVE_SIZE_32 = 1u << 0,
  // Contract may be selected for wave64 code generation.
  LOOM_AMDGPU_MATRIX_WAVE_SIZE_64 = 1u << 1,
  // Contract may be selected for either wave32 or wave64 code generation.
  LOOM_AMDGPU_MATRIX_WAVE_SIZE_ANY =
      LOOM_AMDGPU_MATRIX_WAVE_SIZE_32 | LOOM_AMDGPU_MATRIX_WAVE_SIZE_64,
};

// Bitset of loom_amdgpu_matrix_wave_size_bits_e values.
typedef uint32_t loom_amdgpu_matrix_wave_size_bits_t;

typedef enum loom_amdgpu_matrix_family_e {
  // Unknown or uninitialized matrix contract family.
  LOOM_AMDGPU_MATRIX_FAMILY_UNKNOWN = 0,
  // Matrix FMA instruction family.
  LOOM_AMDGPU_MATRIX_FAMILY_MFMA = 1,
  // Sparse matrix FMA instruction family.
  LOOM_AMDGPU_MATRIX_FAMILY_SMFMAC = 2,
  // Wave matrix multiply-accumulate instruction family.
  LOOM_AMDGPU_MATRIX_FAMILY_WMMA = 3,
  // Sparse wave matrix multiply-accumulate instruction family.
  LOOM_AMDGPU_MATRIX_FAMILY_SWMMAC = 4,
} loom_amdgpu_matrix_family_t;

typedef enum loom_amdgpu_matrix_numeric_type_e {
  // Unknown or uninitialized numeric payload.
  LOOM_AMDGPU_MATRIX_NUMERIC_UNKNOWN = 0,
  // IEEE f64 payload.
  LOOM_AMDGPU_MATRIX_NUMERIC_F64 = 1,
  // IEEE f32 payload.
  LOOM_AMDGPU_MATRIX_NUMERIC_F32 = 2,
  // IEEE f16 payload.
  LOOM_AMDGPU_MATRIX_NUMERIC_F16 = 3,
  // BF16 payload.
  LOOM_AMDGPU_MATRIX_NUMERIC_BF16 = 4,
  // AMD XF32 payload.
  LOOM_AMDGPU_MATRIX_NUMERIC_XF32 = 5,
  // Signed i32 payload.
  LOOM_AMDGPU_MATRIX_NUMERIC_I32 = 6,
  // Signed i8 payload.
  LOOM_AMDGPU_MATRIX_NUMERIC_I8 = 7,
  // Per-operand sign-selected 8-bit integer payload.
  LOOM_AMDGPU_MATRIX_NUMERIC_IU8 = 8,
  // Signed i4 payload packed into byte-like storage.
  LOOM_AMDGPU_MATRIX_NUMERIC_I4 = 9,
  // Per-operand sign-selected 4-bit integer payload.
  LOOM_AMDGPU_MATRIX_NUMERIC_IU4 = 10,
  // AMD FP8 payload.
  LOOM_AMDGPU_MATRIX_NUMERIC_FP8 = 11,
  // AMD BF8 payload.
  LOOM_AMDGPU_MATRIX_NUMERIC_BF8 = 12,
  // AMD FP6 payload.
  LOOM_AMDGPU_MATRIX_NUMERIC_FP6 = 13,
  // AMD BF6 payload.
  LOOM_AMDGPU_MATRIX_NUMERIC_BF6 = 14,
  // AMD FP4 payload.
  LOOM_AMDGPU_MATRIX_NUMERIC_FP4 = 15,
  // Selector-driven 8-bit AMD FP8/BF8 payload family.
  LOOM_AMDGPU_MATRIX_NUMERIC_F8 = 16,
  // Selector-driven 6-bit AMD FP6/BF6 payload family.
  LOOM_AMDGPU_MATRIX_NUMERIC_F6 = 17,
  // Selector-driven AMD F8/F6/F4 payload family.
  LOOM_AMDGPU_MATRIX_NUMERIC_F8F6F4 = 18,
} loom_amdgpu_matrix_numeric_type_t;

typedef enum loom_amdgpu_matrix_scale_kind_e {
  // Contract has no explicit scale operands.
  LOOM_AMDGPU_MATRIX_SCALE_NONE = 0,
  // Contract uses 32-bit scale exponent operands.
  LOOM_AMDGPU_MATRIX_SCALE_32 = 1,
  // Contract uses 16-bit scale exponent operands packed into 64-bit operands.
  LOOM_AMDGPU_MATRIX_SCALE_16 = 2,
} loom_amdgpu_matrix_scale_kind_t;

typedef enum loom_amdgpu_matrix_format_selector_e {
  // LLVM selector value for FP8 E4M3-family matrix payloads.
  LOOM_AMDGPU_MATRIX_FORMAT_SELECTOR_FP8 = 0,
  // LLVM selector value for BF8/E5M2-family matrix payloads.
  LOOM_AMDGPU_MATRIX_FORMAT_SELECTOR_BF8 = 1,
  // LLVM selector value for FP6 E2M3-family matrix payloads.
  LOOM_AMDGPU_MATRIX_FORMAT_SELECTOR_FP6 = 2,
  // LLVM selector value for BF6/E3M2-family matrix payloads.
  LOOM_AMDGPU_MATRIX_FORMAT_SELECTOR_BF6 = 3,
  // LLVM selector value for FP4 matrix payloads.
  LOOM_AMDGPU_MATRIX_FORMAT_SELECTOR_FP4 = 4,
} loom_amdgpu_matrix_format_selector_t;

typedef enum loom_amdgpu_matrix_scale_format_selector_e {
  // LLVM selector value for E8M0 scale payloads.
  LOOM_AMDGPU_MATRIX_SCALE_FORMAT_SELECTOR_E8M0 = 0,
  // LLVM selector value for FP8 E4M3-family scale payloads.
  LOOM_AMDGPU_MATRIX_SCALE_FORMAT_SELECTOR_FP8_E4M3 = 2,
} loom_amdgpu_matrix_scale_format_selector_t;

// Bitset of loom_amdgpu_matrix_scale_format_selector_t values.
typedef uint8_t loom_amdgpu_matrix_scale_format_selector_bits_t;

typedef enum loom_amdgpu_matrix_fragment_coordinate_flag_bits_e {
  // Coordinate carries an independent block or batch value.
  LOOM_AMDGPU_MATRIX_FRAGMENT_COORDINATE_BLOCK =
      LOOM_MATRIX_FRAGMENT_COORDINATE_BLOCK,
  // Coordinate carries an M/result-row value.
  LOOM_AMDGPU_MATRIX_FRAGMENT_COORDINATE_ROW =
      LOOM_MATRIX_FRAGMENT_COORDINATE_ROW,
  // Coordinate carries an N/result-column value.
  LOOM_AMDGPU_MATRIX_FRAGMENT_COORDINATE_COLUMN =
      LOOM_MATRIX_FRAGMENT_COORDINATE_COLUMN,
  // Coordinate carries a K/reduction value.
  LOOM_AMDGPU_MATRIX_FRAGMENT_COORDINATE_REDUCTION =
      LOOM_MATRIX_FRAGMENT_COORDINATE_REDUCTION,
} loom_amdgpu_matrix_fragment_coordinate_flag_bits_t;

// Bitset of loom_amdgpu_matrix_fragment_coordinate_flag_bits_t values.
typedef loom_matrix_fragment_coordinate_flags_t
    loom_amdgpu_matrix_fragment_coordinate_flags_t;

typedef enum loom_amdgpu_matrix_fragment_layout_kind_e {
  // No target-owned fragment layout is attached.
  LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_UNKNOWN = 0,
  // RDNA3 WMMAR3 16x16x16 f16 input, f32 accumulator/result layout.
  LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_RDNA3_WMMAR3_F32_16X16X16_F16 = 1,
  // RDNA3 WMMAR3 16x16x16 bf16 input, f32 accumulator/result layout.
  LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_RDNA3_WMMAR3_F32_16X16X16_BF16 = 2,
  // CDNA MFMA 16x16x16 f16 input, f32 accumulator/result layout.
  LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_CDNA_MFMA_F32_16X16X16_F16 = 3,
  // CDNA MFMA 16x16x16 bf16 input, f32 accumulator/result layout.
  LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_CDNA_MFMA_F32_16X16X16_BF16 = 4,
  // CDNA MFMA 16x16x4 f32 input, f32 accumulator/result layout.
  LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_CDNA_MFMA_F32_16X16X4_F32 = 5,
  // RDNA3 WMMAR3 16x16x16 f16 input, f16 accumulator/result layout.
  LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_RDNA3_WMMAR3_F16_16X16X16_F16 = 6,
  // RDNA3 WMMAR3 16x16x16 bf16 input, bf16 accumulator/result layout.
  LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_RDNA3_WMMAR3_BF16_16X16X16_BF16 = 7,
  // RDNA3 WMMAR3 wave64 16x16x16 f16 input, f32 accumulator/result layout.
  LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_RDNA3_WMMAR3_F32_16X16X16_F16_W64 = 8,
  // RDNA3 WMMAR3 wave64 16x16x16 bf16 input, f32 accumulator/result layout.
  LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_RDNA3_WMMAR3_F32_16X16X16_BF16_W64 = 9,
  // RDNA3 WMMAR3 wave64 16x16x16 f16 input, f16 accumulator/result layout.
  LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_RDNA3_WMMAR3_F16_16X16X16_F16_W64 = 10,
  // RDNA3 WMMAR3 wave64 16x16x16 bf16 input, bf16 accumulator/result layout.
  LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_RDNA3_WMMAR3_BF16_16X16X16_BF16_W64 = 11,
  // RDNA4 WMMA 16x16x16 f16 input, f16 accumulator/result layout.
  LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_RDNA4_WMMA_F16_16X16X16_F16 = 12,
  // RDNA4 WMMA 16x16x16 bf16 input, bf16 accumulator/result layout.
  LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_RDNA4_WMMA_BF16_16X16X16_BF16 = 13,
  // RDNA4 gfx1250 WMMA 16x16x32 f16 input, f16 accumulator/result layout.
  LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_RDNA4_WMMA_F16_16X16X32_F16 = 14,
  // RDNA4 gfx1250 WMMA 16x16x32 bf16 input, bf16 accumulator/result layout.
  LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_RDNA4_WMMA_BF16_16X16X32_BF16 = 15,
  // RDNA4 WMMA 16x16x16 f16 input, f32 accumulator/result layout.
  LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_RDNA4_WMMA_F32_16X16X16_F16 = 16,
  // RDNA4 WMMA 16x16x16 bf16 input, f32 accumulator/result layout.
  LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_RDNA4_WMMA_F32_16X16X16_BF16 = 17,
  // RDNA4 gfx1250 WMMA 16x16x32 f16 input, f32 accumulator/result layout.
  LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_RDNA4_WMMA_F32_16X16X32_F16 = 18,
  // RDNA4 gfx1250 WMMA 16x16x32 bf16 input, f32 accumulator/result layout.
  LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_RDNA4_WMMA_F32_16X16X32_BF16 = 19,
  // RDNA4 gfx1250 WMMA 16x16x4 f32 input, f32 accumulator/result layout.
  LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_RDNA4_WMMA_F32_16X16X4_F32 = 20,
  // CDNA MFMA 16x16x32 packed fp8/bf8 input, f32 accumulator/result layout.
  LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_CDNA_MFMA_F32_16X16X32_PACKED8 = 21,
  // RDNA4 WMMA 16x16x16 packed fp8/bf8 input, f32 accumulator/result layout.
  LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_RDNA4_WMMA_F32_16X16X16_PACKED8 = 22,
  // RDNA4 gfx1250 WMMA 16x16x64 packed fp8/bf8 input, f32
  // accumulator/result layout.
  LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_RDNA4_WMMA_F32_16X16X64_PACKED8 = 23,
  // RDNA4 gfx1250 WMMA 16x16x128 packed fp8/bf8 input, f32
  // accumulator/result layout.
  LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_RDNA4_WMMA_F32_16X16X128_PACKED8 = 24,
  // CDNA4 MFMA 16x16x32 f16 input, f32 accumulator/result layout.
  LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_CDNA_MFMA_F32_16X16X32_F16 = 25,
  // CDNA4 MFMA 16x16x32 bf16 input, f32 accumulator/result layout.
  LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_CDNA_MFMA_F32_16X16X32_BF16 = 26,
  // CDNA4 MFMA 32x32x16 f16 input, f32 accumulator/result layout.
  LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_CDNA_MFMA_F32_32X32X16_F16 = 27,
  // CDNA4 MFMA 32x32x16 bf16 input, f32 accumulator/result layout.
  LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_CDNA_MFMA_F32_32X32X16_BF16 = 28,
  // CDNA MFMA 32x32x16 packed fp8/bf8 input, f32 accumulator/result layout.
  LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_CDNA_MFMA_F32_32X32X16_PACKED8 = 29,
  // CDNA MFMA 16x16x8 packed 16-bit input, f32 accumulator/result layout.
  LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_CDNA_MFMA_F32_16X16X8_PACKED16 = 30,
  // CDNA MFMA 16x16x8 xf32 input, f32 accumulator/result layout.
  LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_CDNA_MFMA_F32_16X16X8_XF32 = 31,
  // CDNA MFMA 32x32x4 packed 16-bit input, f32 accumulator/result layout.
  LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_CDNA_MFMA_F32_32X32X4_PACKED16 = 32,
  // CDNA MFMA 32x32x8 packed 16-bit input, f32 accumulator/result layout.
  LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_CDNA_MFMA_F32_32X32X8_PACKED16 = 33,
  // CDNA MFMA 32x32x4 xf32 input, f32 accumulator/result layout.
  LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_CDNA_MFMA_F32_32X32X4_XF32 = 34,
  // CDNA MFMA 32x32x2 f32 input, f32 accumulator/result layout.
  LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_CDNA_MFMA_F32_32X32X2_F32 = 35,
  // CDNA SMFMAC 16x16x32 packed 16-bit input, 32-bit result layout.
  LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_CDNA_SMFMAC_32BIT_16X16X32_PACKED16 = 36,
  // CDNA SMFMAC 16x16x64 packed 8-bit input, 32-bit result layout.
  LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_CDNA_SMFMAC_32BIT_16X16X64_PACKED8 = 37,
  // CDNA SMFMAC 16x16x64 packed 16-bit input, 32-bit result layout.
  LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_CDNA_SMFMAC_32BIT_16X16X64_PACKED16 = 38,
  // CDNA SMFMAC 16x16x128 packed 8-bit input, 32-bit result layout.
  LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_CDNA_SMFMAC_32BIT_16X16X128_PACKED8 = 39,
  // CDNA SMFMAC 32x32x16 packed 16-bit input, 32-bit result layout.
  LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_CDNA_SMFMAC_32BIT_32X32X16_PACKED16 = 40,
  // CDNA SMFMAC 32x32x32 packed 8-bit input, 32-bit result layout.
  LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_CDNA_SMFMAC_32BIT_32X32X32_PACKED8 = 41,
  // CDNA SMFMAC 32x32x32 packed 16-bit input, 32-bit result layout.
  LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_CDNA_SMFMAC_32BIT_32X32X32_PACKED16 = 42,
  // CDNA SMFMAC 32x32x64 packed 8-bit input, 32-bit result layout.
  LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_CDNA_SMFMAC_32BIT_32X32X64_PACKED8 = 43,
  // RDNA4 SWMMAC 16x16x32 packed 16-bit input, 32-bit result layout.
  LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_RDNA4_SWMMAC_32BIT_16X16X32_PACKED16 = 44,
  // RDNA4 SWMMAC 16x16x32 packed 16-bit input/result layout.
  LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_RDNA4_SWMMAC_16BIT_16X16X32_PACKED16 = 45,
  // RDNA4 SWMMAC 16x16x32 packed 8-bit input, 32-bit result layout.
  LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_RDNA4_SWMMAC_32BIT_16X16X32_PACKED8 = 46,
  // RDNA4 SWMMAC 16x16x32 packed 4-bit input, 32-bit result layout.
  LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_RDNA4_SWMMAC_32BIT_16X16X32_PACKED4 = 47,
  // RDNA4 SWMMAC 16x16x64 packed 4-bit input, 32-bit result layout.
  LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_RDNA4_SWMMAC_32BIT_16X16X64_PACKED4 = 48,
  // GFX1250 SWMMAC 16x16x64 packed 16-bit input, 32-bit result layout.
  LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_GFX1250_SWMMAC_32BIT_16X16X64_PACKED16 =
      49,
  // GFX1250 SWMMAC 16x16x64 packed 16-bit input/result layout.
  LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_GFX1250_SWMMAC_16BIT_16X16X64_PACKED16 =
      50,
  // GFX1250 SWMMAC 16x16x128 packed 8-bit input, 32-bit result layout.
  LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_GFX1250_SWMMAC_32BIT_16X16X128_PACKED8 =
      51,
  // GFX1250 SWMMAC 16x16x128 packed 8-bit input, 16-bit result layout.
  LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_GFX1250_SWMMAC_16BIT_16X16X128_PACKED8 =
      52,
  // CDNA MFMA 16 independent 4x4x1 f32-input, f32-result blocks.
  LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_CDNA_MFMA_F32_4X4X1_F32_16B = 53,
  // CDNA MFMA 16 independent 4x4x2 bf16-input, f32-result blocks.
  LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_CDNA_MFMA_F32_4X4X2_BF16_16B = 54,
  // CDNA MFMA 16 independent 4x4x4 packed-16-input, f32-result blocks.
  LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_CDNA_MFMA_F32_4X4X4_PACKED16_16B = 55,
  // CDNA MFMA 16 independent 4x4x4 i8-input, i32-result blocks.
  LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_CDNA_MFMA_I32_4X4X4_I8_16B = 56,
  // CDNA MFMA 4 independent 16x16x1 f32-input, f32-result blocks.
  LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_CDNA_MFMA_F32_16X16X1_F32_4B = 57,
  // CDNA MFMA 4 independent 16x16x2 bf16-input, f32-result blocks.
  LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_CDNA_MFMA_F32_16X16X2_BF16_4B = 58,
  // CDNA MFMA 4 independent 16x16x4 packed-16-input, f32-result blocks.
  LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_CDNA_MFMA_F32_16X16X4_PACKED16_4B = 59,
  // CDNA MFMA 4 independent 16x16x4 i8-input, i32-result blocks.
  LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_CDNA_MFMA_I32_16X16X4_I8_4B = 60,
  // CDNA MFMA 2 independent 32x32x1 f32-input, f32-result blocks.
  LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_CDNA_MFMA_F32_32X32X1_F32_2B = 61,
  // CDNA MFMA 2 independent 32x32x2 bf16-input, f32-result blocks.
  LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_CDNA_MFMA_F32_32X32X2_BF16_2B = 62,
  // CDNA MFMA 2 independent 32x32x4 packed-16-input, f32-result blocks.
  LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_CDNA_MFMA_F32_32X32X4_PACKED16_2B = 63,
  // CDNA MFMA 2 independent 32x32x4 i8-input, i32-result blocks.
  LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_CDNA_MFMA_I32_32X32X4_I8_2B = 64,
  // CDNA MFMA 4 independent 4x4x4 f64-input/result blocks.
  LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_CDNA_MFMA_F64_4X4X4_F64_4B = 65,
  // RDNA3 WMMAR3 wave32 16x16x16 packed iu8 input, i32 result layout.
  LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_RDNA3_WMMAR3_I32_16X16X16_IU8 = 66,
  // RDNA3 WMMAR3 wave64 16x16x16 packed iu8 input, i32 result layout.
  LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_RDNA3_WMMAR3_I32_16X16X16_IU8_W64 = 67,
  // RDNA3 WMMAR3 wave32 16x16x16 packed iu4 input, i32 result layout.
  LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_RDNA3_WMMAR3_I32_16X16X16_IU4 = 68,
  // RDNA3 WMMAR3 wave64 16x16x16 packed iu4 input, i32 result layout.
  LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_RDNA3_WMMAR3_I32_16X16X16_IU4_W64 = 69,
  // RDNA4 WMMA 16x16x16 packed iu8 input, i32 result layout.
  LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_RDNA4_WMMA_I32_16X16X16_IU8 = 70,
  // RDNA4 WMMA 16x16x16 packed iu4 input, i32 result layout.
  LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_RDNA4_WMMA_I32_16X16X16_IU4 = 71,
  // RDNA4 WMMA 16x16x32 packed iu4 input, i32 result layout.
  LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_RDNA4_WMMA_I32_16X16X32_IU4 = 72,
  // RDNA4 WMMA 16x16x64 packed iu8 input, i32 result layout.
  LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_RDNA4_WMMA_I32_16X16X64_IU8 = 73,
  // CDNA MFMA 16x16x4 f64 input, f64 accumulator/result layout.
  LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_CDNA_MFMA_F64_16X16X4_F64 = 74,
  // GFX12.5 WMMA 16x16x128 f8/f8 input, f32 result layout.
  LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_GFX125X_WMMA_F32_16X16X128_F8_F8 = 75,
  // GFX12.5 WMMA 16x16x128 f8/f6 input, f32 result layout.
  LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_GFX125X_WMMA_F32_16X16X128_F8_F6 = 76,
  // GFX12.5 WMMA 16x16x128 f8/f4 input, f32 result layout.
  LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_GFX125X_WMMA_F32_16X16X128_F8_F4 = 77,
  // GFX12.5 WMMA 16x16x128 f6/f8 input, f32 result layout.
  LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_GFX125X_WMMA_F32_16X16X128_F6_F8 = 78,
  // GFX12.5 WMMA 16x16x128 f6/f6 input, f32 result layout.
  LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_GFX125X_WMMA_F32_16X16X128_F6_F6 = 79,
  // GFX12.5 WMMA 16x16x128 f6/f4 input, f32 result layout.
  LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_GFX125X_WMMA_F32_16X16X128_F6_F4 = 80,
  // GFX12.5 WMMA 16x16x128 f4/f8 input, f32 result layout.
  LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_GFX125X_WMMA_F32_16X16X128_F4_F8 = 81,
  // GFX12.5 WMMA 16x16x128 f4/f6 input, f32 result layout.
  LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_GFX125X_WMMA_F32_16X16X128_F4_F6 = 82,
  // GFX12.5 WMMA 16x16x128 f4/f4 input, f32 result layout.
  LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_GFX125X_WMMA_F32_16X16X128_F4_F4 = 83,
  // RDNA4 wave64 WMMA 16x16x16 f16 input, f32 result layout.
  LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_RDNA4_WMMA_F32_16X16X16_F16_W64 = 84,
  // RDNA4 wave64 WMMA 16x16x16 bf16 input, f32 result layout.
  LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_RDNA4_WMMA_F32_16X16X16_BF16_W64 = 85,
  // RDNA4 wave64 WMMA 16x16x16 f16 input/result layout.
  LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_RDNA4_WMMA_F16_16X16X16_F16_W64 = 86,
  // RDNA4 wave64 WMMA 16x16x16 bf16 input/result layout.
  LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_RDNA4_WMMA_BF16_16X16X16_BF16_W64 = 87,
  // RDNA4 wave64 WMMA 16x16x16 packed fp8/bf8 input, f32 result layout.
  LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_RDNA4_WMMA_F32_16X16X16_PACKED8_W64 = 88,
  // RDNA4 wave64 WMMA 16x16x16 packed iu8 input, i32 result layout.
  LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_RDNA4_WMMA_I32_16X16X16_IU8_W64 = 89,
  // RDNA4 wave64 WMMA 16x16x16 packed iu4 input, i32 result layout.
  LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_RDNA4_WMMA_I32_16X16X16_IU4_W64 = 90,
  // RDNA4 wave64 WMMA 16x16x32 packed iu4 input, i32 result layout.
  LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_RDNA4_WMMA_I32_16X16X32_IU4_W64 = 91,
  // RDNA4 wave64 SWMMAC 16x16x32 packed 16-bit input, 32-bit result layout.
  LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_RDNA4_SWMMAC_32BIT_16X16X32_PACKED16_W64 =
      92,
  // RDNA4 wave64 SWMMAC 16x16x32 packed 16-bit input/result layout.
  LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_RDNA4_SWMMAC_16BIT_16X16X32_PACKED16_W64 =
      93,
  // RDNA4 wave64 SWMMAC 16x16x32 packed 8-bit input, 32-bit result layout.
  LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_RDNA4_SWMMAC_32BIT_16X16X32_PACKED8_W64 =
      94,
  // RDNA4 wave64 SWMMAC 16x16x32 packed 4-bit input, 32-bit result layout.
  LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_RDNA4_SWMMAC_32BIT_16X16X32_PACKED4_W64 =
      95,
  // RDNA4 wave64 SWMMAC 16x16x64 packed 4-bit input, 32-bit result layout.
  LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_RDNA4_SWMMAC_32BIT_16X16X64_PACKED4_W64 =
      96,
  // Total number of fragment layout table slots, including UNKNOWN.
  LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_COUNT = 97,
} loom_amdgpu_matrix_fragment_layout_kind_t;

typedef enum loom_amdgpu_matrix_contract_flag_bits_e {
  // Contract consumes an explicit sparse index operand.
  LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SPARSE = 1u << 0,
  // Contract consumes explicit scale operands.
  LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SCALED = 1u << 1,
  // Contract consumes matrix-format selector operands.
  LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_MATRIX_FORMATS = 1u << 2,
  // Contract consumes matrix reuse immediate operands.
  LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_REUSE = 1u << 3,
  // Contract consumes a clamp immediate operand.
  LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_CLAMP = 1u << 4,
  // Contract consumes A/B sign-selection immediate operands.
  LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SIGN_SELECT = 1u << 5,
  // Contract consumes A/B operand modifier immediate operands.
  LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_AB_MODIFIERS = 1u << 6,
  // Contract consumes a C accumulator modifier immediate operand.
  LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_C_MODIFIER = 1u << 7,
  // Contract consumes a GFX11/GFX12 op_sel operand.
  LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_OPSEL = 1u << 8,
  // A zero scale can refine to an unscaled contract with the same shape.
  LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_ZERO_SCALE_FALLBACK = 1u << 9,
  // Contract consumes scale-format selector operands.
  LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SCALE_FORMATS = 1u << 10,
} loom_amdgpu_matrix_contract_flag_bits_t;

// Bitset of loom_amdgpu_matrix_contract_flag_bits_t values.
typedef uint32_t loom_amdgpu_matrix_contract_flags_t;

typedef enum loom_amdgpu_matrix_contract_source_requirement_flag_bits_e {
  // Source-level lowering requires a target-owned fragment layout.
  LOOM_AMDGPU_MATRIX_CONTRACT_SOURCE_REQUIREMENT_FRAGMENT_LAYOUT = 1u << 0,
} loom_amdgpu_matrix_contract_source_requirement_flag_bits_t;

// Bitset of loom_amdgpu_matrix_contract_source_requirement_flag_bits_t values.
typedef uint32_t loom_amdgpu_matrix_contract_source_requirement_flags_t;

// Matrix contract does not have a target-low descriptor mapping yet.
#define LOOM_AMDGPU_MATRIX_LOW_DESCRIPTOR_REF_NONE \
  LOOM_AMDGPU_DESCRIPTOR_REF_NONE

// AMDGPU descriptors use the generic block/M/N/K matrix tile shape record.
typedef loom_matrix_fragment_tile_shape_t loom_amdgpu_matrix_tile_shape_t;

typedef struct loom_amdgpu_matrix_payload_shape_t {
  // Logical numeric type represented by this operand or result.
  loom_amdgpu_matrix_numeric_type_t numeric_type;
  // Number of 32-bit VGPR payload registers in a fixed LLVM signature. Zero
  // means the descriptor needs a later wave- or format-specific fragment
  // signature.
  uint16_t register_count;
  // Number of logical scalar elements represented by the payload. Zero means
  // the descriptor needs a later wave- or format-specific fragment signature.
  uint16_t element_count;
} loom_amdgpu_matrix_payload_shape_t;

// AMDGPU descriptors use the generic role-local lane/register layout record.
typedef loom_matrix_fragment_role_layout_t
    loom_amdgpu_matrix_fragment_role_layout_t;

// AMDGPU descriptors use the generic matrix fragment layout record.
typedef loom_matrix_fragment_layout_t loom_amdgpu_matrix_fragment_layout_t;

// AMDGPU descriptors use the generic logical coordinate record.
typedef loom_matrix_fragment_coordinate_t
    loom_amdgpu_matrix_fragment_coordinate_t;

typedef struct loom_amdgpu_matrix_contract_descriptor_t {
  // Stable Loom descriptor name used by tests, diagnostics, and target logs.
  iree_string_view_t name;
  // Stable target-low descriptor ref selected by this descriptor, or NONE.
  loom_amdgpu_descriptor_ref_t low_descriptor_ref;
  // LLVM AMDGPU intrinsic name selected by this descriptor for LLVM lowering.
  iree_string_view_t llvm_intrinsic_name;
  // AMDGPU instruction family used by this descriptor.
  loom_amdgpu_matrix_family_t family;
  // Processor feature bits required before this descriptor is legal.
  loom_amdgpu_matrix_feature_bits_t required_feature_bits;
  // Wave sizes for which this descriptor is legal.
  loom_amdgpu_matrix_wave_size_bits_t wave_size_bits;
  // Optional immediate operands or semantic decorations required by the call.
  loom_amdgpu_matrix_contract_flags_t flags;
  // Source-level support requirements that must be met before selection.
  loom_amdgpu_matrix_contract_source_requirement_flags_t
      source_requirement_flags;
  // Logical tile shape consumed and produced by one instruction.
  loom_amdgpu_matrix_tile_shape_t tile_shape;
  // Matrix A payload shape.
  loom_amdgpu_matrix_payload_shape_t lhs_payload;
  // Matrix B payload shape.
  loom_amdgpu_matrix_payload_shape_t rhs_payload;
  // Accumulator payload shape.
  loom_amdgpu_matrix_payload_shape_t accumulator_payload;
  // Result payload shape.
  loom_amdgpu_matrix_payload_shape_t result_payload;
  // Explicit scale operand kind.
  loom_amdgpu_matrix_scale_kind_t scale_kind;
  // Fixed scale-format selector bits accepted when the descriptor ABI has no
  // scale-format selector immediates. Zero means unconstrained or selector
  // driven.
  loom_amdgpu_matrix_scale_format_selector_bits_t
      implicit_scale_format_selector_bits;
  // Target-owned fragment lane/register layout kind.
  loom_amdgpu_matrix_fragment_layout_kind_t fragment_layout_kind;
} loom_amdgpu_matrix_contract_descriptor_t;

enum loom_amdgpu_matrix_contract_rejection_bits_e {
  // No rejection reason was recorded.
  LOOM_AMDGPU_MATRIX_CONTRACT_REJECTION_NONE = 0u,
  // The requested family rejected every descriptor.
  LOOM_AMDGPU_MATRIX_CONTRACT_REJECTION_FAMILY = 1u << 0,
  // The requested tile shape rejected every family-compatible descriptor.
  LOOM_AMDGPU_MATRIX_CONTRACT_REJECTION_TILE_SHAPE = 1u << 1,
  // The requested matrix A payload rejected every shape-compatible descriptor.
  LOOM_AMDGPU_MATRIX_CONTRACT_REJECTION_LHS_PAYLOAD = 1u << 2,
  // The requested matrix B payload rejected every shape-compatible descriptor.
  LOOM_AMDGPU_MATRIX_CONTRACT_REJECTION_RHS_PAYLOAD = 1u << 3,
  // The requested accumulator payload rejected every shape-compatible
  // descriptor.
  LOOM_AMDGPU_MATRIX_CONTRACT_REJECTION_ACCUMULATOR_PAYLOAD = 1u << 4,
  // The requested result payload rejected every shape-compatible descriptor.
  LOOM_AMDGPU_MATRIX_CONTRACT_REJECTION_RESULT_PAYLOAD = 1u << 5,
  // The requested scale kind rejected every payload-compatible descriptor.
  LOOM_AMDGPU_MATRIX_CONTRACT_REJECTION_SCALE_KIND = 1u << 6,
  // The selected processor feature bits rejected every semantic candidate.
  LOOM_AMDGPU_MATRIX_CONTRACT_REJECTION_FEATURES = 1u << 7,
  // The selected wave size rejected every feature-compatible candidate.
  LOOM_AMDGPU_MATRIX_CONTRACT_REJECTION_WAVE_SIZE = 1u << 8,
  // A candidate required a sparse index fact or operand that was unavailable.
  LOOM_AMDGPU_MATRIX_CONTRACT_REJECTION_MISSING_SPARSE = 1u << 9,
  // A candidate required scale operands that were unavailable.
  LOOM_AMDGPU_MATRIX_CONTRACT_REJECTION_MISSING_SCALE = 1u << 10,
  // A candidate required matrix-format selectors that were unavailable.
  LOOM_AMDGPU_MATRIX_CONTRACT_REJECTION_MISSING_MATRIX_FORMATS = 1u << 11,
  // A candidate required reuse operands that were unavailable.
  LOOM_AMDGPU_MATRIX_CONTRACT_REJECTION_MISSING_REUSE = 1u << 12,
  // A candidate required a clamp operand that was unavailable.
  LOOM_AMDGPU_MATRIX_CONTRACT_REJECTION_MISSING_CLAMP = 1u << 13,
  // A candidate required sign-selection operands that were unavailable.
  LOOM_AMDGPU_MATRIX_CONTRACT_REJECTION_MISSING_SIGN_SELECT = 1u << 14,
  // A candidate required A/B operand modifiers that were unavailable.
  LOOM_AMDGPU_MATRIX_CONTRACT_REJECTION_MISSING_AB_MODIFIERS = 1u << 15,
  // A candidate required a C accumulator modifier that was unavailable.
  LOOM_AMDGPU_MATRIX_CONTRACT_REJECTION_MISSING_C_MODIFIER = 1u << 16,
  // A candidate required op_sel operands that were unavailable.
  LOOM_AMDGPU_MATRIX_CONTRACT_REJECTION_MISSING_OPSEL = 1u << 17,
  // A candidate required scale-format selectors that were unavailable.
  LOOM_AMDGPU_MATRIX_CONTRACT_REJECTION_MISSING_SCALE_FORMATS = 1u << 18,
  // A candidate has fixed scale-format semantics that do not match the source.
  LOOM_AMDGPU_MATRIX_CONTRACT_REJECTION_SCALE_FORMAT = 1u << 19,
  // The request required target flags that the remaining candidates do not
  // carry.
  LOOM_AMDGPU_MATRIX_CONTRACT_REJECTION_REQUIRED_FLAGS = 1u << 20,
  // The request itself was invalid or absent.
  LOOM_AMDGPU_MATRIX_CONTRACT_REJECTION_INVALID_REQUEST = 1u << 21,
};

// Bitset of loom_amdgpu_matrix_contract_rejection_bits_e values.
typedef uint32_t loom_amdgpu_matrix_contract_rejection_bits_t;

typedef struct loom_amdgpu_matrix_contract_match_request_t {
  // Optional instruction family. UNKNOWN allows any family.
  loom_amdgpu_matrix_family_t family;
  // Required logical tile shape.
  loom_amdgpu_matrix_tile_shape_t tile_shape;
  // Required matrix A payload facts. Zero register/element counts are ignored.
  loom_amdgpu_matrix_payload_shape_t lhs_payload;
  // Required matrix B payload facts. Zero register/element counts are ignored.
  loom_amdgpu_matrix_payload_shape_t rhs_payload;
  // Required accumulator payload facts. Zero register/element counts are
  // ignored.
  loom_amdgpu_matrix_payload_shape_t accumulator_payload;
  // Required result payload facts. Zero register/element counts are ignored.
  loom_amdgpu_matrix_payload_shape_t result_payload;
  // Required scale operand kind.
  loom_amdgpu_matrix_scale_kind_t scale_kind;
  // LHS scale-format selector bit proven by the source schema, or zero.
  loom_amdgpu_matrix_scale_format_selector_bits_t
      lhs_scale_format_selector_bits;
  // RHS scale-format selector bit proven by the source schema, or zero.
  loom_amdgpu_matrix_scale_format_selector_bits_t
      rhs_scale_format_selector_bits;
  // Processor feature bits available to the target.
  loom_amdgpu_matrix_feature_bits_t feature_bits;
  // Concrete wave size selected for the target. Use 0 when not yet selected.
  uint32_t wave_size;
  // Contract flag classes for which the request has facts or operands
  // available. Descriptor ABI flags must be present here before selection.
  loom_amdgpu_matrix_contract_flags_t available_flags;
  // Contract flag classes that the selected descriptor must carry.
  loom_amdgpu_matrix_contract_flags_t required_flags;
} loom_amdgpu_matrix_contract_match_request_t;

typedef struct loom_amdgpu_matrix_contract_match_diagnostic_t {
  // Structural rejection reason selected for user-facing diagnostics.
  loom_amdgpu_matrix_contract_rejection_bits_t rejection_bits;
  // Number of descriptors scanned.
  iree_host_size_t descriptor_count;
  // Number of descriptors that matched the requested family.
  iree_host_size_t family_candidate_count;
  // Number of family-compatible descriptors that matched the tile shape.
  iree_host_size_t shape_candidate_count;
  // Number of shape-compatible descriptors that matched all payload facts.
  iree_host_size_t payload_candidate_count;
  // Number of payload-compatible descriptors that matched the scale kind.
  iree_host_size_t scale_candidate_count;
  // Number of scale-compatible descriptors that matched flag requirements.
  iree_host_size_t flag_candidate_count;
  // Number of flag-compatible descriptors that matched scale-format facts.
  iree_host_size_t scale_format_candidate_count;
  // Number of flag-compatible descriptors available for the target features.
  iree_host_size_t feature_candidate_count;
  // Number of feature-compatible descriptors legal for the selected wave size.
  iree_host_size_t wave_candidate_count;
} loom_amdgpu_matrix_contract_match_diagnostic_t;

#ifdef __cplusplus
}
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_MATRIX_TYPES_H_
