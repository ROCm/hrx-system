// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Shared AMDGPU FP8 payload decode helpers.

#ifndef LOOM_TARGET_ARCH_AMDGPU_LOWER_FP8_DECODE_H_
#define LOOM_TARGET_ARCH_AMDGPU_LOWER_FP8_DECODE_H_

#include <stdint.h>

#include "loom/codegen/low/lower/lower.h"
#include "loom/ir/scalar_type.h"
#include "loom/target/arch/amdgpu/lower/plan.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum loom_amdgpu_fp8_decode_plan_flag_bits_e {
  LOOM_AMDGPU_FP8_DECODE_PLAN_FLAG_NONE = 0u,
  LOOM_AMDGPU_FP8_DECODE_PLAN_FLAG_HAS_BFE_U32 = 1u << 0,
  LOOM_AMDGPU_FP8_DECODE_PLAN_FLAG_HAS_PACK_U16 = 1u << 1,
  LOOM_AMDGPU_FP8_DECODE_PLAN_FLAG_HAS_PERM_B32 = 1u << 2,
  LOOM_AMDGPU_FP8_DECODE_PLAN_FLAG_HAS_NATIVE_F32_PAIR = 1u << 3,
  LOOM_AMDGPU_FP8_DECODE_PLAN_FLAG_HAS_NATIVE_BF16_PACK = 1u << 4,
  LOOM_AMDGPU_FP8_DECODE_PLAN_FLAG_HAS_ADD3_SRC2_LITERAL = 1u << 5,
  LOOM_AMDGPU_FP8_DECODE_PLAN_FLAG_HAS_LSHL_ADD_U32_SHIFT_IMM = 1u << 6,
} loom_amdgpu_fp8_decode_plan_flag_bits_t;
typedef uint32_t loom_amdgpu_fp8_decode_plan_flags_t;

// Per-value facts that can simplify the generated special-value decode path.
// These describe the actual value being decoded, not the full source FP8 type.
typedef enum loom_amdgpu_fp8_decode_value_flag_bits_e {
  LOOM_AMDGPU_FP8_DECODE_VALUE_FLAG_NONE = 0u,
  LOOM_AMDGPU_FP8_DECODE_VALUE_FLAG_NOT_NAN = 1u << 0,
  LOOM_AMDGPU_FP8_DECODE_VALUE_FLAG_NOT_INF = 1u << 1,
} loom_amdgpu_fp8_decode_value_flag_bits_t;
typedef uint32_t loom_amdgpu_fp8_decode_value_flags_t;

enum {
  LOOM_AMDGPU_FP8_BF16_BYTE_COUNT = 2u,
  LOOM_AMDGPU_FP8_BF16_BYTE_TABLE_WORD_COUNT = 2u,
};

typedef struct loom_amdgpu_fp8_to_f32_descriptor_refs_t {
  // Scalar FP8-to-F32 conversion descriptor.
  loom_amdgpu_descriptor_ref_t lane;
  // Packed pair FP8-to-F32 conversion descriptor.
  loom_amdgpu_descriptor_ref_t pair;
} loom_amdgpu_fp8_to_f32_descriptor_refs_t;

typedef struct loom_amdgpu_fp8_decode_plan_t {
  // Available native packet helpers selected from the active descriptor set.
  loom_amdgpu_fp8_decode_plan_flags_t flags;
  // Parsed FP8 source format.
  loom_scalar_type_fp8_format_t format;
  // Packed unsigned BF16 subnormal payload tables for two-bit mantissas.
  uint32_t subnormal_bf16_table_words[2];
  // Packed BF16 subnormal payload byte tables for three-bit mantissas.
  uint32_t subnormal_bf16_byte_table_words
      [LOOM_AMDGPU_FP8_BF16_BYTE_COUNT]
      [LOOM_AMDGPU_FP8_BF16_BYTE_TABLE_WORD_COUNT];
  // Unsigned VGPR bitfield extract descriptor used to isolate packed bytes.
  loom_low_lower_resolved_descriptor_t bfe_u32_descriptor;
  // Optional signed equality compare descriptor with an inline RHS operand.
  loom_low_lower_resolved_descriptor_t compare_eq_i32_src1_inline_descriptor;
  // Optional unsigned-greater-equal compare descriptor with inline RHS operand.
  loom_low_lower_resolved_descriptor_t compare_uge_u32_src1_inline_descriptor;
  // Optional unsigned-less-than compare descriptor with an inline RHS operand.
  loom_low_lower_resolved_descriptor_t compare_ult_u32_src1_inline_descriptor;
  // Packed low-16-bit pair descriptor used to combine BF16 lanes.
  loom_low_lower_resolved_descriptor_t pack_u16_descriptor;
  // Byte permute descriptor used to select tiny FP8 subnormal BF16 tables.
  loom_low_lower_resolved_descriptor_t perm_b32_descriptor;
  // Native packed-pair FP8-to-F32 conversion descriptor.
  loom_low_lower_resolved_descriptor_t native_f32_pair_descriptor;
  // Native F32-pair-to-BF16-pair conversion descriptor.
  loom_low_lower_resolved_descriptor_t native_bf16_pack_descriptor;
  // Integer three-input add descriptor with a source-2 literal.
  loom_low_lower_resolved_descriptor_t add3_src2_literal_descriptor;
  // Integer left-shift-add descriptor with an immediate shift.
  loom_low_lower_resolved_descriptor_t lshl_add_u32_shift_imm_descriptor;
} loom_amdgpu_fp8_decode_plan_t;

// Returns the native FP8-to-F32 conversion descriptor refs for |element_type|.
bool loom_amdgpu_fp8_to_f32_descriptor_refs(
    loom_scalar_type_t source_element_type,
    loom_amdgpu_fp8_to_f32_descriptor_refs_t* out_refs);

// Selects descriptor helpers and initializes format tables for |element_type|.
iree_status_t loom_amdgpu_select_fp8_decode_plan(
    loom_low_lower_context_t* context, loom_scalar_type_t element_type,
    loom_amdgpu_fp8_decode_plan_t* out_plan);

// Emits one 16-bit BF16 bit payload from an unsigned FP8 byte payload.
iree_status_t loom_amdgpu_emit_fp8_to_bf16_lane(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fp8_decode_plan_t* plan, loom_value_id_t low_byte,
    loom_amdgpu_fp8_decode_value_flags_t value_flags, loom_type_t vgpr_type,
    loom_type_t mask_type, loom_value_id_t* out_lane);

// Emits one packed VGPR containing two BF16 bit payloads.
iree_status_t loom_amdgpu_emit_packed_bf16_pair(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fp8_decode_plan_t* plan, loom_value_id_t low_element,
    loom_value_id_t high_element, loom_type_t vgpr_type,
    loom_value_id_t* out_low_packet);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_LOWER_FP8_DECODE_H_
