// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ir/attribute.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/target/arch/amdgpu/lower/constants.h"
#include "loom/target/arch/amdgpu/lower/emit.h"
#include "loom/target/arch/amdgpu/lower/encoding/fp8.h"
#include "loom/target/arch/amdgpu/lower/topology.h"
#include "loom/target/arch/amdgpu/lower/types.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"

enum {
  LOOM_AMDGPU_FP8_DECODE_U16_PAIR_SELECTOR_COUNT = 3u,
  LOOM_AMDGPU_FP8_DECODE_U16_PAIR_NO_SIGN_MASK = 0x007F007Fu,
  LOOM_AMDGPU_FP8_DECODE_U16_PAIR_HIGH_BYTE_NO_SIGN_MASK = 0x7F007F00u,
  LOOM_AMDGPU_FP8_DECODE_U16_PAIR_SIGN_MASK = 0x00800080u,
  LOOM_AMDGPU_FP8_DECODE_U16_PAIR_SIGN_INSERT_MASK = 0x80008000u,
  LOOM_AMDGPU_FP8_DECODE_U16_PAIR_LOW_BYTE_MASK = 0x000000FFu,
  LOOM_AMDGPU_FP8_DECODE_U16_PAIR_ONE_MASK = 0x00010001u,
  LOOM_AMDGPU_FP8_DECODE_U16_PAIR_ASHR_15_MASK = 0x000F000Fu,
  LOOM_AMDGPU_FP8_DECODE_U16_PAIR_HIGH_BIT_MASK = 0x80008000u,
  LOOM_AMDGPU_FP8_DECODE_U16_PAIR_EQ_ZERO_BIAS = 0xFFFFFFFFu,
  LOOM_AMDGPU_FP8_DECODE_BF16_PAIR_INFINITY_BITS = 0x7F807F80u,
  LOOM_AMDGPU_FP8_DECODE_BF16_PAIR_QUIET_NAN_BITS = 0x7FC07FC0u,
};

typedef struct loom_amdgpu_fp8_packed_u16_repair_state_t {
  // SGPR shift constant used to expand packed condition lanes into BFI masks.
  loom_value_id_t low_mask_shift;
} loom_amdgpu_fp8_packed_u16_repair_state_t;

typedef struct loom_amdgpu_fp8_packed_u16_repair_split_t {
  // Block entered when the combined condition has at least one active lane.
  loom_block_t* repair_block;
  // Block entered when the combined condition is empty.
  loom_block_t* normal_block;
  // Block joining repair and normal payloads.
  loom_block_t* continuation_block;
  // Continuation arguments carrying one value per packed pair.
  loom_value_id_t continuation_values[LOOM_AMDGPU_MAX_PACKED_32BIT_REGISTERS];
} loom_amdgpu_fp8_packed_u16_repair_split_t;

typedef struct loom_amdgpu_fp8_packed_u16_pair_base_t {
  // FP8 byte-pair expanded into the low byte of each packed 16-bit lane.
  loom_value_id_t expanded_pair;
  // Expanded pair with sign bits masked out.
  loom_value_id_t source_no_sign;
  // Shifted sign bits ready to merge into packed 16-bit payload lanes.
  loom_value_id_t sign_bits;
} loom_amdgpu_fp8_packed_u16_pair_base_t;

typedef struct loom_amdgpu_fp8_packed_u16_pair_state_t {
  // Pair base shared by packed 16-bit FP8 decode paths.
  loom_amdgpu_fp8_packed_u16_pair_base_t base;
  // Packed 16-bit payload produced by the fast normal-value expansion.
  loom_value_id_t normal_payload;
} loom_amdgpu_fp8_packed_u16_pair_state_t;

typedef struct loom_amdgpu_fp8_packed_u16_permute_state_t {
  // VGPR zero used as the second byte source by non-unary permute forms.
  loom_value_id_t low_zero;
  // Lazily materialized SGPR selectors indexed by source byte offset.
  loom_value_id_t low_selectors[LOOM_AMDGPU_FP8_DECODE_U16_PAIR_SELECTOR_COUNT];
  // Left shift applied to selectors for the requested packed lane placement.
  uint32_t selector_shift;
} loom_amdgpu_fp8_packed_u16_permute_state_t;

static uint32_t loom_amdgpu_fp8_decode_packed_u16(uint32_t value) {
  return value | (value << 16);
}

static uint32_t loom_amdgpu_fp8_decode_packed_lt_bias(uint32_t threshold) {
  const uint32_t lane_bias = UINT32_C(0x10000) - threshold;
  return loom_amdgpu_fp8_decode_packed_u16(lane_bias);
}

static uint32_t loom_amdgpu_fp8_decode_packed_ge_bias(uint32_t threshold) {
  const uint32_t lane_bias = UINT32_C(0x8000) - threshold;
  return loom_amdgpu_fp8_decode_packed_u16(lane_bias);
}

static uint32_t loom_amdgpu_fp8_decode_subnormal_threshold(
    const loom_scalar_type_fp8_format_t* format) {
  return UINT32_C(1) << format->mantissa_bits;
}

static uint32_t loom_amdgpu_fp8_decode_top_exponent_no_sign(
    const loom_scalar_type_fp8_format_t* format) {
  return ((UINT32_C(1) << format->exponent_bits) - 1u) << format->mantissa_bits;
}

static uint32_t loom_amdgpu_fp8_decode_finite_nan_no_sign(
    const loom_scalar_type_fp8_format_t* format) {
  return loom_amdgpu_fp8_decode_top_exponent_no_sign(format) |
         ((UINT32_C(1) << format->mantissa_bits) - 1u);
}

static bool loom_amdgpu_fp8_decode_value_is_finite(
    loom_amdgpu_fp8_decode_value_flags_t value_flags) {
  return iree_all_bits_set(value_flags,
                           LOOM_AMDGPU_FP8_DECODE_VALUE_FLAG_NOT_NAN |
                               LOOM_AMDGPU_FP8_DECODE_VALUE_FLAG_NOT_INF);
}

static bool loom_amdgpu_fp8_decode_plan_has_packed_zero_repair(
    const loom_amdgpu_fp8_decode_plan_t* plan) {
  return iree_any_bit_set(
      plan->capabilities,
      LOOM_AMDGPU_FP8_DECODE_PLAN_CAPABILITY_PACKED_ZERO_REPAIR);
}

static bool loom_amdgpu_fp8_decode_plan_has_mask_repair_split(
    const loom_amdgpu_fp8_decode_plan_t* plan) {
  return iree_any_bit_set(
      plan->capabilities,
      LOOM_AMDGPU_FP8_DECODE_PLAN_CAPABILITY_MASK_REPAIR_SPLIT);
}

static bool loom_amdgpu_fp8_decode_plan_has_inline_sgpr64_zero_compare(
    const loom_amdgpu_fp8_decode_plan_t* plan) {
  return iree_any_bit_set(
      plan->capabilities,
      LOOM_AMDGPU_FP8_DECODE_PLAN_CAPABILITY_INLINE_SGPR64_ZERO_COMPARE);
}

static bool loom_amdgpu_fp8_decode_plan_has_combined_finite_nan_condition(
    const loom_amdgpu_fp8_decode_plan_t* plan) {
  return iree_any_bit_set(
      plan->capabilities,
      LOOM_AMDGPU_FP8_DECODE_PLAN_CAPABILITY_COMBINED_FINITE_NAN_CONDITION);
}

static bool loom_amdgpu_fp8_decode_plan_has_combined_non_normal_condition(
    const loom_amdgpu_fp8_decode_plan_t* plan) {
  return iree_any_bit_set(
      plan->capabilities,
      LOOM_AMDGPU_FP8_DECODE_PLAN_CAPABILITY_COMBINED_NON_NORMAL_CONDITION);
}

static bool loom_amdgpu_can_emit_fp8_pair_to_packed_u16_finite_path(
    const loom_amdgpu_fp8_decode_plan_t* plan,
    loom_amdgpu_fp8_decode_value_flags_t value_flags,
    loom_amdgpu_fp8_decode_plan_capabilities_t normal_payload_capability) {
  const loom_amdgpu_fp8_decode_plan_flags_t required_plan_flags =
      LOOM_AMDGPU_FP8_DECODE_PLAN_FLAG_HAS_PERM_B32;
  if (!iree_all_bits_set(plan->flags, required_plan_flags) ||
      !iree_any_bit_set(plan->capabilities, normal_payload_capability) ||
      !loom_amdgpu_fp8_decode_value_is_finite(value_flags) ||
      !iree_any_bit_set(value_flags,
                        LOOM_AMDGPU_FP8_DECODE_VALUE_FLAG_NOT_SUBNORMAL)) {
    return false;
  }
  if (iree_any_bit_set(value_flags,
                       LOOM_AMDGPU_FP8_DECODE_VALUE_FLAG_NON_ZERO)) {
    return true;
  }
  return loom_amdgpu_fp8_decode_plan_has_packed_zero_repair(plan);
}

static iree_status_t loom_amdgpu_emit_fp8_decode_merge_low_high_bytes(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fp8_decode_plan_t* plan, loom_value_id_t low_byte_payload,
    loom_value_id_t high_byte_payload, loom_type_t vgpr_type,
    loom_value_id_t* out_value) {
  *out_value = LOOM_VALUE_ID_INVALID;
  if (iree_any_bit_set(
          plan->flags,
          LOOM_AMDGPU_FP8_DECODE_PLAN_FLAG_HAS_LSHL_ADD_U32_SHIFT_IMM)) {
    return loom_amdgpu_emit_resolved_vgpr_lshl_add_u32(
        context, source_op, &plan->lshl_add_u32_shift_imm_descriptor,
        high_byte_payload, low_byte_payload, 8, vgpr_type, out_value);
  }

  loom_value_id_t shifted_high_payload = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_shift(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32_LIT, 8,
      high_byte_payload, vgpr_type, &shifted_high_payload));
  return loom_amdgpu_emit_vgpr_binary(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_OR_B32, low_byte_payload,
      shifted_high_payload, vgpr_type, out_value);
}

enum {
  LOOM_AMDGPU_FP8_DECODE_U16_BYTE_PAIR_LOW_SELECTOR = 0x00050004u,
  LOOM_AMDGPU_FP8_DECODE_U16_BYTE_PAIR_MIDDLE_SELECTOR = 0x00060005u,
  LOOM_AMDGPU_FP8_DECODE_U16_BYTE_PAIR_HIGH_SELECTOR = 0x00070006u,
};

static const uint32_t kLoomAmdgpuFp8DecodeU16BytePairSelectors[] = {
    LOOM_AMDGPU_FP8_DECODE_U16_BYTE_PAIR_LOW_SELECTOR,
    LOOM_AMDGPU_FP8_DECODE_U16_BYTE_PAIR_MIDDLE_SELECTOR,
    LOOM_AMDGPU_FP8_DECODE_U16_BYTE_PAIR_HIGH_SELECTOR,
};
static_assert(IREE_ARRAYSIZE(kLoomAmdgpuFp8DecodeU16BytePairSelectors) ==
                  LOOM_AMDGPU_FP8_DECODE_U16_PAIR_SELECTOR_COUNT,
              "packed FP8 pair selector state must cover every byte offset");

typedef struct loom_amdgpu_fp8_normal_packed_u16_payload_state_t {
  // Packed per-16-bit-lane exponent-bias addend.
  uint32_t packed_exponent_bias;
  // SGPR constant with the per-16-bit-lane payload shift amount.
  loom_value_id_t low_payload_shift;
  // SGPR constant with the per-16-bit-lane payload multiplier.
  loom_value_id_t low_payload_multiplier;
  // SGPR constant with the per-16-bit-lane exponent-bias addend.
  loom_value_id_t low_exponent_bias;
} loom_amdgpu_fp8_normal_packed_u16_payload_state_t;

static iree_status_t loom_amdgpu_emit_fp8_normal_packed_u16_payload_state(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fp8_decode_plan_t* plan, uint32_t payload_shift,
    uint32_t packed_exponent_bias, loom_type_t sgpr_type,
    loom_amdgpu_fp8_normal_packed_u16_payload_state_t* out_state) {
  *out_state = (loom_amdgpu_fp8_normal_packed_u16_payload_state_t){
      .packed_exponent_bias = packed_exponent_bias,
      .low_payload_shift = LOOM_VALUE_ID_INVALID,
      .low_payload_multiplier = LOOM_VALUE_ID_INVALID,
      .low_exponent_bias = LOOM_VALUE_ID_INVALID,
  };
  const bool use_multiply_add =
      packed_exponent_bias != 0 &&
      iree_any_bit_set(plan->flags,
                       LOOM_AMDGPU_FP8_DECODE_PLAN_FLAG_HAS_PK_MAD_U16);
  const bool use_literal_multiply_add =
      use_multiply_add &&
      iree_any_bit_set(
          plan->flags,
          LOOM_AMDGPU_FP8_DECODE_PLAN_FLAG_HAS_PK_MAD_U16_SRC2_LITERAL);
  if (use_multiply_add) {
    const uint32_t payload_multiplier = UINT32_C(1) << payload_shift;
    const uint32_t packed_payload_multiplier =
        payload_multiplier | (payload_multiplier << 16);
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32,
        packed_payload_multiplier, sgpr_type,
        &out_state->low_payload_multiplier));
  } else {
    const uint32_t packed_payload_shift = payload_shift | (payload_shift << 16);
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32,
        packed_payload_shift, sgpr_type, &out_state->low_payload_shift));
  }
  if (packed_exponent_bias == 0) {
    return iree_ok_status();
  }
  if (use_literal_multiply_add) {
    return iree_ok_status();
  }
  return loom_amdgpu_emit_const_u32(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32,
      packed_exponent_bias, sgpr_type, &out_state->low_exponent_bias);
}

static iree_status_t loom_amdgpu_emit_fp8_pk_mad_u16(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fp8_decode_plan_t* plan, loom_value_id_t low_lhs,
    loom_value_id_t low_rhs, uint32_t addend,
    loom_value_id_t low_materialized_addend, loom_type_t vgpr_type,
    loom_value_id_t* out_payload) {
  if (iree_any_bit_set(
          plan->flags,
          LOOM_AMDGPU_FP8_DECODE_PLAN_FLAG_HAS_PK_MAD_U16_SRC2_LITERAL)) {
    loom_named_attr_t attrs[1] = {0};
    iree_host_size_t attr_count = 0;
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_append_i64_attr(context, IREE_SV("imm32"), addend, attrs,
                                    IREE_ARRAYSIZE(attrs), &attr_count));
    const loom_value_id_t operands[] = {low_lhs, low_rhs};
    loom_op_t* low_op = NULL;
    IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
        context, &plan->pk_mad_u16_src2_literal_descriptor, operands,
        IREE_ARRAYSIZE(operands), loom_make_named_attr_slice(attrs, attr_count),
        &vgpr_type, 1, /*tied_results=*/NULL, /*tied_result_count=*/0,
        source_op->location, &low_op));
    *out_payload = loom_value_slice_get(loom_low_op_results(low_op), 0);
    return iree_ok_status();
  }

  IREE_ASSERT_NE(low_materialized_addend, LOOM_VALUE_ID_INVALID);
  return loom_amdgpu_emit_resolved_vgpr_ternary(
      context, source_op, &plan->pk_mad_u16_descriptor, low_lhs, low_rhs,
      low_materialized_addend, vgpr_type, out_payload);
}

static iree_status_t loom_amdgpu_emit_fp8_normal_packed_u16_payload(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fp8_decode_plan_t* plan,
    const loom_amdgpu_fp8_normal_packed_u16_payload_state_t* state,
    loom_value_id_t low_source_no_sign_pair, loom_type_t vgpr_type,
    loom_value_id_t* out_payload) {
  *out_payload = LOOM_VALUE_ID_INVALID;
  if (state->low_payload_multiplier != LOOM_VALUE_ID_INVALID) {
    return loom_amdgpu_emit_fp8_pk_mad_u16(
        context, source_op, plan, low_source_no_sign_pair,
        state->low_payload_multiplier, state->packed_exponent_bias,
        state->low_exponent_bias, vgpr_type, out_payload);
  }

  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_resolved_vgpr_binary(
      context, source_op, &plan->pk_lshlrev_b16_descriptor,
      state->low_payload_shift, low_source_no_sign_pair, vgpr_type,
      out_payload));
  if (state->low_exponent_bias == LOOM_VALUE_ID_INVALID) {
    return iree_ok_status();
  }

  return loom_amdgpu_emit_resolved_vgpr_binary(
      context, source_op, &plan->pk_add_u16_descriptor, *out_payload,
      state->low_exponent_bias, vgpr_type, out_payload);
}

typedef struct loom_amdgpu_fp8_subnormal_u16_byte_table_values_t {
  // SGPR table words indexed by output byte and packed word.
  loom_value_id_t words[LOOM_AMDGPU_FP8_U16_BYTE_COUNT]
                       [LOOM_AMDGPU_FP8_U16_BYTE_TABLE_WORD_COUNT];
} loom_amdgpu_fp8_subnormal_u16_byte_table_values_t;

static iree_status_t loom_amdgpu_emit_fp8_subnormal_u16_byte_table_values(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const uint32_t table_words[LOOM_AMDGPU_FP8_U16_BYTE_COUNT]
                              [LOOM_AMDGPU_FP8_U16_BYTE_TABLE_WORD_COUNT],
    loom_type_t sgpr_type,
    loom_amdgpu_fp8_subnormal_u16_byte_table_values_t* out_tables) {
  *out_tables = (loom_amdgpu_fp8_subnormal_u16_byte_table_values_t){0};
  for (uint32_t byte_index = 0; byte_index < LOOM_AMDGPU_FP8_U16_BYTE_COUNT;
       ++byte_index) {
    for (uint32_t word_index = 0;
         word_index < LOOM_AMDGPU_FP8_U16_BYTE_TABLE_WORD_COUNT; ++word_index) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
          context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32,
          table_words[byte_index][word_index], sgpr_type,
          &out_tables->words[byte_index][word_index]));
    }
  }
  return iree_ok_status();
}

static iree_status_t
loom_amdgpu_emit_fp8_subnormal_packed_u16_bits_from_byte_tables(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fp8_decode_plan_t* plan,
    const loom_amdgpu_fp8_subnormal_u16_byte_table_values_t* tables,
    loom_value_id_t low_source_no_sign_pair, loom_type_t vgpr_type,
    loom_value_id_t* out_bits) {
  *out_bits = LOOM_VALUE_ID_INVALID;
  loom_value_id_t low_payload_bytes[LOOM_AMDGPU_FP8_U16_BYTE_COUNT] = {0};
  for (uint32_t byte_index = 0; byte_index < LOOM_AMDGPU_FP8_U16_BYTE_COUNT;
       ++byte_index) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_resolved_vgpr_ternary(
        context, source_op, &plan->perm_b32_descriptor,
        tables->words[byte_index][1], tables->words[byte_index][0],
        low_source_no_sign_pair, vgpr_type, &low_payload_bytes[byte_index]));
  }

  return loom_amdgpu_emit_fp8_decode_merge_low_high_bytes(
      context, source_op, plan, low_payload_bytes[0], low_payload_bytes[1],
      vgpr_type, out_bits);
}

static iree_status_t loom_amdgpu_emit_fp8_packed_u16_pair_lt_condition(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fp8_decode_plan_t* plan, loom_value_id_t low_value_pair,
    uint32_t lane_bias, loom_type_t vgpr_type, loom_type_t sgpr_type,
    loom_value_id_t* out_condition) {
  *out_condition = LOOM_VALUE_ID_INVALID;
  loom_value_id_t low_bias = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32, lane_bias,
      sgpr_type, &low_bias));
  return loom_amdgpu_emit_resolved_vgpr_binary(
      context, source_op, &plan->pk_add_u16_descriptor, low_value_pair,
      low_bias, vgpr_type, out_condition);
}

static iree_status_t loom_amdgpu_emit_fp8_packed_u16_pair_ge_condition(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fp8_decode_plan_t* plan, loom_value_id_t low_value_pair,
    uint32_t lane_threshold, loom_type_t vgpr_type, loom_type_t sgpr_type,
    loom_value_id_t* out_condition) {
  *out_condition = LOOM_VALUE_ID_INVALID;
  loom_value_id_t low_bias = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32,
      loom_amdgpu_fp8_decode_packed_ge_bias(lane_threshold), sgpr_type,
      &low_bias));
  return loom_amdgpu_emit_resolved_vgpr_binary(
      context, source_op, &plan->pk_add_u16_descriptor, low_value_pair,
      low_bias, vgpr_type, out_condition);
}

static iree_status_t loom_amdgpu_emit_fp8_packed_u16_pair_condition_with_bias(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fp8_decode_plan_t* plan, loom_value_id_t low_value_pair,
    loom_value_id_t low_bias, loom_type_t vgpr_type,
    loom_value_id_t* out_condition) {
  *out_condition = LOOM_VALUE_ID_INVALID;
  return loom_amdgpu_emit_resolved_vgpr_binary(
      context, source_op, &plan->pk_add_u16_descriptor, low_value_pair,
      low_bias, vgpr_type, out_condition);
}

static iree_status_t
loom_amdgpu_emit_fp8_packed_u16_mask_from_condition_with_shift(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fp8_decode_plan_t* plan, loom_value_id_t low_condition,
    loom_value_id_t low_shift, loom_type_t vgpr_type,
    loom_value_id_t* out_mask) {
  *out_mask = LOOM_VALUE_ID_INVALID;
  return loom_amdgpu_emit_resolved_vgpr_binary(
      context, source_op, &plan->pk_ashrrev_i16_descriptor, low_shift,
      low_condition, vgpr_type, out_mask);
}

static iree_status_t loom_amdgpu_get_fp8_packed_u16_mask_shift(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_type_t sgpr_type,
    loom_amdgpu_fp8_packed_u16_repair_state_t* repair_state) {
  if (repair_state->low_mask_shift != LOOM_VALUE_ID_INVALID) {
    return iree_ok_status();
  }
  return loom_amdgpu_emit_const_u32(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32,
      LOOM_AMDGPU_FP8_DECODE_U16_PAIR_ASHR_15_MASK, sgpr_type,
      &repair_state->low_mask_shift);
}

static iree_status_t loom_amdgpu_emit_fp8_packed_u16_pair_eq_condition(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fp8_decode_plan_t* plan, loom_value_id_t low_value_pair,
    uint32_t lane_value, loom_type_t vgpr_type, loom_type_t sgpr_type,
    loom_value_id_t* out_condition) {
  *out_condition = LOOM_VALUE_ID_INVALID;
  loom_value_id_t low_delta = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary_immediate(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_XOR_B32_LIT,
      low_value_pair, lane_value, vgpr_type, &low_delta));
  return loom_amdgpu_emit_fp8_packed_u16_pair_lt_condition(
      context, source_op, plan, low_delta,
      LOOM_AMDGPU_FP8_DECODE_U16_PAIR_EQ_ZERO_BIAS, vgpr_type, sgpr_type,
      out_condition);
}

static iree_status_t
loom_amdgpu_emit_fp8_packed_u16_pair_eq_condition_with_bias(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fp8_decode_plan_t* plan, loom_value_id_t low_value_pair,
    uint32_t lane_value, loom_value_id_t low_eq_zero_bias,
    loom_type_t vgpr_type, loom_value_id_t* out_condition) {
  *out_condition = LOOM_VALUE_ID_INVALID;
  loom_value_id_t low_delta = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary_immediate(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_XOR_B32_LIT,
      low_value_pair, lane_value, vgpr_type, &low_delta));
  return loom_amdgpu_emit_fp8_packed_u16_pair_condition_with_bias(
      context, source_op, plan, low_delta, low_eq_zero_bias, vgpr_type,
      out_condition);
}

static iree_status_t loom_amdgpu_emit_fp8_packed_u16_pair_select(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fp8_decode_plan_t* plan, loom_value_id_t low_false_value,
    loom_value_id_t low_true_value, loom_value_id_t low_mask,
    loom_type_t vgpr_type, loom_value_id_t* out_value) {
  *out_value = LOOM_VALUE_ID_INVALID;
  if (iree_any_bit_set(plan->flags,
                       LOOM_AMDGPU_FP8_DECODE_PLAN_FLAG_HAS_BFI_B32)) {
    const loom_value_id_t operands[] = {
        low_mask,
        low_true_value,
        low_false_value,
    };
    loom_op_t* low_op = NULL;
    IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
        context, &plan->bfi_b32_descriptor, operands, IREE_ARRAYSIZE(operands),
        loom_named_attr_slice_empty(), &vgpr_type, 1,
        /*tied_results=*/NULL, /*tied_result_count=*/0, source_op->location,
        &low_op));
    *out_value = loom_value_slice_get(loom_low_op_results(low_op), 0);
    return iree_ok_status();
  }
  loom_value_id_t low_delta = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_XOR_B32, low_false_value,
      low_true_value, vgpr_type, &low_delta));
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32, low_delta,
      low_mask, vgpr_type, &low_delta));
  return loom_amdgpu_emit_vgpr_binary(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_XOR_B32, low_false_value,
      low_delta, vgpr_type, out_value);
}

static iree_status_t loom_amdgpu_emit_fp8_decode_vgpr_or_reduce(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_value_id_t* values, iree_host_size_t value_count,
    loom_type_t vgpr_type, loom_value_id_t* out_value) {
  IREE_ASSERT_GT(value_count, 0u);
  IREE_ASSERT_LE(value_count, LOOM_AMDGPU_MAX_PACKED_32BIT_REGISTERS);
  loom_value_id_t current[LOOM_AMDGPU_MAX_PACKED_32BIT_REGISTERS] = {0};
  for (iree_host_size_t i = 0; i < value_count; ++i) {
    current[i] = values[i];
  }

  iree_host_size_t current_count = value_count;
  while (current_count > 1) {
    iree_host_size_t next_count = 0;
    iree_host_size_t i = 0;
    for (; i + 1 < current_count; i += 2) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary(
          context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_OR_B32, current[i],
          current[i + 1], vgpr_type, &current[next_count++]));
    }
    if (i < current_count) {
      current[next_count++] = current[i];
    }
    current_count = next_count;
  }

  *out_value = current[0];
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_fp8_decode_resolved_vgpr_binary_reduce(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_low_lower_resolved_descriptor_t* descriptor,
    const loom_value_id_t* values, iree_host_size_t value_count,
    loom_type_t vgpr_type, loom_value_id_t* out_value) {
  IREE_ASSERT_GT(value_count, 0u);
  IREE_ASSERT_LE(value_count, LOOM_AMDGPU_MAX_PACKED_32BIT_REGISTERS);
  loom_value_id_t current[LOOM_AMDGPU_MAX_PACKED_32BIT_REGISTERS] = {0};
  for (iree_host_size_t i = 0; i < value_count; ++i) {
    current[i] = values[i];
  }

  iree_host_size_t current_count = value_count;
  while (current_count > 1) {
    iree_host_size_t next_count = 0;
    iree_host_size_t i = 0;
    for (; i + 1 < current_count; i += 2) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_resolved_vgpr_binary(
          context, source_op, descriptor, current[i], current[i + 1], vgpr_type,
          &current[next_count++]));
    }
    if (i < current_count) {
      current[next_count++] = current[i];
    }
    current_count = next_count;
  }

  *out_value = current[0];
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_fp8_decode_sgpr64_nonzero_scc(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fp8_decode_plan_t* plan, loom_value_id_t low_value,
    loom_value_id_t low_zero_sgpr64, loom_value_id_t* out_scc) {
  *out_scc = LOOM_VALUE_ID_INVALID;
  loom_type_t scc_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_scc_type(context, &scc_type));

  if (loom_amdgpu_fp8_decode_plan_has_inline_sgpr64_zero_compare(plan)) {
    loom_named_attr_t attrs[1] = {0};
    iree_host_size_t attr_count = 0;
    IREE_RETURN_IF_ERROR(loom_amdgpu_append_i64_attr(
        context, IREE_SV("rhs"), 0, attrs, IREE_ARRAYSIZE(attrs), &attr_count));
    const loom_value_id_t operands[] = {low_value};
    loom_op_t* compare_op = NULL;
    IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
        context, &plan->compare_lg_u64_src1_inline_descriptor, operands,
        IREE_ARRAYSIZE(operands), loom_make_named_attr_slice(attrs, attr_count),
        &scc_type, 1, /*tied_results=*/NULL, /*tied_result_count=*/0,
        source_op->location, &compare_op));
    *out_scc = loom_value_slice_get(loom_low_op_results(compare_op), 0);
    return iree_ok_status();
  }

  IREE_ASSERT_NE(low_zero_sgpr64, LOOM_VALUE_ID_INVALID);
  const loom_value_id_t operands[] = {low_value, low_zero_sgpr64};
  loom_op_t* compare_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
      context, &plan->compare_lg_u64_descriptor, operands,
      IREE_ARRAYSIZE(operands), loom_named_attr_slice_empty(), &scc_type, 1,
      /*tied_results=*/NULL, /*tied_result_count=*/0, source_op->location,
      &compare_op));
  *out_scc = loom_value_slice_get(loom_low_op_results(compare_op), 0);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_fp8_packed_u16_repair_split(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fp8_decode_plan_t* plan,
    loom_value_id_t low_combined_condition, iree_host_size_t pair_count,
    loom_type_t vgpr_type, loom_type_t mask_type,
    loom_value_id_t low_zero_sgpr64,
    loom_amdgpu_fp8_packed_u16_repair_split_t* out_split) {
  *out_split = (loom_amdgpu_fp8_packed_u16_repair_split_t){0};
  loom_value_id_t low_has_condition_lanes = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_resolved_vgpr_compare_immediate(
      context, source_op, &plan->compare_ne_i32_src1_inline_descriptor,
      LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_NE_I32, low_combined_condition, 0,
      vgpr_type, mask_type, &low_has_condition_lanes));

  loom_builder_t* builder = loom_low_lower_context_builder(context);
  IREE_ASSERT(builder->ip.before_op == NULL);

  const uint32_t wavefront_size =
      loom_amdgpu_target_wavefront_size(loom_low_lower_context_bundle(context));

  loom_value_id_t low_has_condition_scc = LOOM_VALUE_ID_INVALID;
  if (wavefront_size == 32) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_lane_mask_nonzero_scc(
        context, source_op, low_has_condition_lanes, wavefront_size,
        &low_has_condition_scc));
  } else {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp8_decode_sgpr64_nonzero_scc(
        context, source_op, plan, low_has_condition_lanes, low_zero_sgpr64,
        &low_has_condition_scc));
  }

  loom_block_t* hot_block = builder->ip.block;
  loom_region_t* low_region = hot_block->parent_region;
  IREE_RETURN_IF_ERROR(loom_region_insert_block(
      builder->module, low_region, (uint16_t)(hot_block->region_index + 1),
      &out_split->repair_block));
  IREE_RETURN_IF_ERROR(loom_region_insert_block(
      builder->module, low_region,
      (uint16_t)(out_split->repair_block->region_index + 1),
      &out_split->normal_block));
  IREE_RETURN_IF_ERROR(loom_region_insert_block(
      builder->module, low_region,
      (uint16_t)(out_split->normal_block->region_index + 1),
      &out_split->continuation_block));

  for (iree_host_size_t i = 0; i < pair_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_builder_define_block_arg(
        builder, out_split->continuation_block, vgpr_type,
        &out_split->continuation_values[i]));
  }

  loom_op_t* cond_branch_op = NULL;
  return loom_low_cond_br_build(
      builder, low_has_condition_scc, out_split->repair_block,
      out_split->normal_block, source_op->location, &cond_branch_op);
}

static iree_status_t loom_amdgpu_emit_fp8_packed_u16_split_condition(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_value_id_t* low_conditions, iree_host_size_t condition_count,
    loom_type_t vgpr_type, loom_value_id_t* out_condition) {
  *out_condition = LOOM_VALUE_ID_INVALID;
  loom_value_id_t low_combined_condition = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp8_decode_vgpr_or_reduce(
      context, source_op, low_conditions, condition_count, vgpr_type,
      &low_combined_condition));
  return loom_amdgpu_emit_vgpr_binary_immediate(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT,
      low_combined_condition, LOOM_AMDGPU_FP8_DECODE_U16_PAIR_HIGH_BIT_MASK,
      vgpr_type, out_condition);
}

static iree_status_t loom_amdgpu_emit_fp8_packed_u16_zero_repaired_payload(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fp8_decode_plan_t* plan,
    loom_value_id_t low_source_no_sign, loom_value_id_t low_normal_payload,
    loom_amdgpu_fp8_decode_value_flags_t value_flags,
    loom_value_id_t low_one_mask, loom_type_t vgpr_type,
    loom_value_id_t* out_payload) {
  *out_payload = low_normal_payload;
  if (iree_any_bit_set(value_flags,
                       LOOM_AMDGPU_FP8_DECODE_VALUE_FLAG_NON_ZERO)) {
    return iree_ok_status();
  }
  IREE_ASSERT_NE(low_one_mask, LOOM_VALUE_ID_INVALID);
  loom_value_id_t low_nonzero_mask = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_resolved_vgpr_binary(
      context, source_op, &plan->pk_min_u16_descriptor, low_source_no_sign,
      low_one_mask, vgpr_type, &low_nonzero_mask));
  return loom_amdgpu_emit_resolved_vgpr_binary(
      context, source_op, &plan->pk_mul_lo_u16_descriptor, low_normal_payload,
      low_nonzero_mask, vgpr_type, out_payload);
}

static iree_status_t loom_amdgpu_emit_fp8_packed_u16_permute_state(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fp8_decode_plan_t* plan, uint32_t selector_shift,
    loom_type_t vgpr_type,
    loom_amdgpu_fp8_packed_u16_permute_state_t* out_state) {
  *out_state = (loom_amdgpu_fp8_packed_u16_permute_state_t){
      .low_zero = LOOM_VALUE_ID_INVALID,
      .low_selectors = {LOOM_VALUE_ID_INVALID, LOOM_VALUE_ID_INVALID,
                        LOOM_VALUE_ID_INVALID},
      .selector_shift = selector_shift,
  };
  if (iree_any_bit_set(
          plan->flags,
          LOOM_AMDGPU_FP8_DECODE_PLAN_FLAG_HAS_PERM_B32_SRC1_ZERO_SRC2_LIT)) {
    return iree_ok_status();
  }
  return loom_amdgpu_emit_const_u32(context, source_op,
                                    LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32, 0,
                                    vgpr_type, &out_state->low_zero);
}

static iree_status_t loom_amdgpu_emit_fp8_packed_u16_pair_permute(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fp8_decode_plan_t* plan,
    const loom_amdgpu_fp8_packed_u16_pair_source_t* pair_source,
    loom_amdgpu_fp8_packed_u16_permute_state_t* state, loom_type_t vgpr_type,
    loom_type_t sgpr_type, loom_value_id_t* out_expanded_pair) {
  const uint32_t selector =
      kLoomAmdgpuFp8DecodeU16BytePairSelectors[pair_source->byte_offset]
      << state->selector_shift;
  if (iree_any_bit_set(
          plan->flags,
          LOOM_AMDGPU_FP8_DECODE_PLAN_FLAG_HAS_PERM_B32_SRC1_ZERO_SRC2_LIT)) {
    return loom_amdgpu_emit_resolved_vgpr_unary_immediate(
        context, source_op, &plan->perm_b32_src1_zero_src2_literal_descriptor,
        pair_source->source_register, selector, vgpr_type, out_expanded_pair);
  }
  IREE_ASSERT_NE(state->low_zero, LOOM_VALUE_ID_INVALID);
  if (iree_any_bit_set(
          plan->flags,
          LOOM_AMDGPU_FP8_DECODE_PLAN_FLAG_HAS_PERM_B32_SRC2_LITERAL)) {
    return loom_amdgpu_emit_resolved_vgpr_binary_immediate(
        context, source_op, &plan->perm_b32_src2_literal_descriptor,
        pair_source->source_register, state->low_zero, selector, vgpr_type,
        out_expanded_pair);
  }

  loom_value_id_t* low_selector =
      &state->low_selectors[pair_source->byte_offset];
  if (*low_selector == LOOM_VALUE_ID_INVALID) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32, selector,
        sgpr_type, low_selector));
  }
  return loom_amdgpu_emit_resolved_vgpr_ternary(
      context, source_op, &plan->perm_b32_descriptor,
      pair_source->source_register, state->low_zero, *low_selector, vgpr_type,
      out_expanded_pair);
}

static iree_status_t loom_amdgpu_emit_fp8_packed_u16_pair_base(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fp8_decode_plan_t* plan,
    const loom_amdgpu_fp8_packed_u16_pair_source_t* pair_source,
    loom_amdgpu_fp8_packed_u16_permute_state_t* permute_state,
    loom_type_t vgpr_type, loom_type_t sgpr_type,
    loom_amdgpu_fp8_packed_u16_pair_base_t* out_base) {
  IREE_ASSERT_GE(pair_source->live_lane_count, 1u);
  IREE_ASSERT_LE(pair_source->live_lane_count, 2u);
  *out_base = (loom_amdgpu_fp8_packed_u16_pair_base_t){
      .expanded_pair = LOOM_VALUE_ID_INVALID,
      .source_no_sign = LOOM_VALUE_ID_INVALID,
      .sign_bits = LOOM_VALUE_ID_INVALID,
  };

  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp8_packed_u16_pair_permute(
      context, source_op, plan, pair_source, permute_state, vgpr_type,
      sgpr_type, &out_base->expanded_pair));

  if (pair_source->live_lane_count == 1u) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary_immediate(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT,
        out_base->expanded_pair, LOOM_AMDGPU_FP8_DECODE_U16_PAIR_LOW_BYTE_MASK,
        vgpr_type, &out_base->expanded_pair));
    const uint32_t high_lane_normal =
        loom_amdgpu_fp8_decode_subnormal_threshold(&plan->format) << 16;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary_immediate(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_OR_B32_LIT,
        out_base->expanded_pair, high_lane_normal, vgpr_type,
        &out_base->expanded_pair));
  }

  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary_immediate(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT,
      out_base->expanded_pair, LOOM_AMDGPU_FP8_DECODE_U16_PAIR_NO_SIGN_MASK,
      vgpr_type, &out_base->source_no_sign));

  if (iree_any_bit_set(
          plan->flags,
          LOOM_AMDGPU_FP8_DECODE_PLAN_FLAG_HAS_BFI_B32_SRC0_LITERAL)) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_shift(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32_LIT, 8,
        out_base->expanded_pair, vgpr_type, &out_base->sign_bits));
  } else {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary_immediate(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT,
        out_base->expanded_pair, LOOM_AMDGPU_FP8_DECODE_U16_PAIR_SIGN_MASK,
        vgpr_type, &out_base->sign_bits));
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_shift(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32_LIT, 8,
        out_base->sign_bits, vgpr_type, &out_base->sign_bits));
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_fp8_packed_u16_pair_state(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fp8_decode_plan_t* plan,
    const loom_amdgpu_fp8_packed_u16_pair_source_t* pair_source,
    const loom_amdgpu_fp8_normal_packed_u16_payload_state_t*
        normal_payload_state,
    loom_amdgpu_fp8_packed_u16_permute_state_t* permute_state,
    loom_type_t vgpr_type, loom_type_t sgpr_type,
    loom_amdgpu_fp8_packed_u16_pair_state_t* out_state) {
  *out_state = (loom_amdgpu_fp8_packed_u16_pair_state_t){
      .base = {0},
      .normal_payload = LOOM_VALUE_ID_INVALID,
  };

  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp8_packed_u16_pair_base(
      context, source_op, plan, pair_source, permute_state, vgpr_type,
      sgpr_type, &out_state->base));

  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp8_normal_packed_u16_payload(
      context, source_op, plan, normal_payload_state,
      out_state->base.source_no_sign, vgpr_type, &out_state->normal_payload));
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_fp8_packed_u16_pair_states(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fp8_decode_plan_t* plan,
    const loom_amdgpu_fp8_packed_u16_pair_source_t* pair_sources,
    iree_host_size_t pair_count, uint32_t payload_shift,
    uint32_t packed_exponent_bias, loom_type_t vgpr_type, loom_type_t sgpr_type,
    loom_amdgpu_fp8_packed_u16_pair_state_t* out_pair_states) {
  loom_amdgpu_fp8_packed_u16_permute_state_t permute_state;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp8_packed_u16_permute_state(
      context, source_op, plan, /*selector_shift=*/0, vgpr_type,
      &permute_state));

  loom_amdgpu_fp8_normal_packed_u16_payload_state_t normal_payload_state;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp8_normal_packed_u16_payload_state(
      context, source_op, plan, payload_shift, packed_exponent_bias, sgpr_type,
      &normal_payload_state));

  for (iree_host_size_t i = 0; i < pair_count; ++i) {
    IREE_ASSERT_GE(pair_sources[i].live_lane_count, 1u);
    IREE_ASSERT_LE(pair_sources[i].live_lane_count, 2u);
    IREE_ASSERT_LT(pair_sources[i].byte_offset,
                   IREE_ARRAYSIZE(kLoomAmdgpuFp8DecodeU16BytePairSelectors));
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp8_packed_u16_pair_state(
        context, source_op, plan, &pair_sources[i], &normal_payload_state,
        &permute_state, vgpr_type, sgpr_type, &out_pair_states[i]));
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_fp8_packed_u16_pair_sign(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fp8_decode_plan_t* plan,
    const loom_amdgpu_fp8_packed_u16_pair_base_t* base,
    loom_value_id_t low_finite_payload, loom_type_t vgpr_type,
    loom_value_id_t* out_packet) {
  *out_packet = LOOM_VALUE_ID_INVALID;
  if (iree_any_bit_set(
          plan->flags,
          LOOM_AMDGPU_FP8_DECODE_PLAN_FLAG_HAS_BFI_B32_SRC0_LITERAL)) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_resolved_vgpr_binary_immediate(
        context, source_op, &plan->bfi_b32_src0_literal_descriptor,
        base->sign_bits, low_finite_payload,
        LOOM_AMDGPU_FP8_DECODE_U16_PAIR_SIGN_INSERT_MASK, vgpr_type,
        out_packet));
  } else {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_OR_B32,
        base->sign_bits, low_finite_payload, vgpr_type, out_packet));
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_fp8_packed_u16_pair_signs(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fp8_decode_plan_t* plan,
    const loom_amdgpu_fp8_packed_u16_pair_state_t* pair_states,
    const loom_value_id_t* finite_payloads, iree_host_size_t pair_count,
    loom_type_t vgpr_type, loom_value_id_t* out_packets) {
  for (iree_host_size_t i = 0; i < pair_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp8_packed_u16_pair_sign(
        context, source_op, plan, &pair_states[i].base, finite_payloads[i],
        vgpr_type, &out_packets[i]));
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_fp8_packed_u16_subnormal_condition(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fp8_decode_plan_t* plan,
    loom_value_id_t low_source_no_sign, loom_type_t vgpr_type,
    loom_type_t sgpr_type, loom_value_id_t* out_condition) {
  return loom_amdgpu_emit_fp8_packed_u16_pair_lt_condition(
      context, source_op, plan, low_source_no_sign,
      loom_amdgpu_fp8_decode_packed_lt_bias(
          loom_amdgpu_fp8_decode_subnormal_threshold(&plan->format)),
      vgpr_type, sgpr_type, out_condition);
}

static iree_status_t
loom_amdgpu_emit_fp8_packed_u16_combined_subnormal_condition(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fp8_decode_plan_t* plan,
    const loom_amdgpu_fp8_packed_u16_pair_state_t* pair_states,
    iree_host_size_t pair_count, loom_type_t vgpr_type, loom_type_t sgpr_type,
    loom_value_id_t* out_condition) {
  *out_condition = LOOM_VALUE_ID_INVALID;
  if (pair_count > 1 &&
      iree_any_bit_set(plan->flags,
                       LOOM_AMDGPU_FP8_DECODE_PLAN_FLAG_HAS_PK_MIN_U16)) {
    loom_value_id_t low_sources[LOOM_AMDGPU_MAX_PACKED_32BIT_REGISTERS] = {0};
    for (iree_host_size_t i = 0; i < pair_count; ++i) {
      low_sources[i] = pair_states[i].base.source_no_sign;
    }
    loom_value_id_t low_min_source_no_sign = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_emit_fp8_decode_resolved_vgpr_binary_reduce(
            context, source_op, &plan->pk_min_u16_descriptor, low_sources,
            pair_count, vgpr_type, &low_min_source_no_sign));
    loom_value_id_t low_condition = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp8_packed_u16_subnormal_condition(
        context, source_op, plan, low_min_source_no_sign, vgpr_type, sgpr_type,
        &low_condition));
    return loom_amdgpu_emit_fp8_packed_u16_split_condition(
        context, source_op, &low_condition, 1, vgpr_type, out_condition);
  }

  loom_value_id_t repair_conditions[LOOM_AMDGPU_MAX_PACKED_32BIT_REGISTERS] = {
      0};
  for (iree_host_size_t i = 0; i < pair_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp8_packed_u16_subnormal_condition(
        context, source_op, plan, pair_states[i].base.source_no_sign, vgpr_type,
        sgpr_type, &repair_conditions[i]));
  }
  return loom_amdgpu_emit_fp8_packed_u16_split_condition(
      context, source_op, repair_conditions, pair_count, vgpr_type,
      out_condition);
}

static iree_status_t loom_amdgpu_emit_fp8_packed_u16_subnormal_repair_payloads(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fp8_decode_plan_t* plan,
    const loom_amdgpu_fp8_packed_u16_pair_state_t* pair_states,
    iree_host_size_t pair_count,
    const uint32_t table_words[LOOM_AMDGPU_FP8_U16_BYTE_COUNT]
                              [LOOM_AMDGPU_FP8_U16_BYTE_TABLE_WORD_COUNT],
    loom_type_t vgpr_type, loom_type_t sgpr_type,
    loom_amdgpu_fp8_packed_u16_repair_state_t* repair_state,
    const loom_value_id_t* low_subnormal_markers,
    loom_value_id_t* out_repair_payloads) {
  loom_amdgpu_fp8_subnormal_u16_byte_table_values_t subnormal_tables;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp8_subnormal_u16_byte_table_values(
      context, source_op, table_words, sgpr_type, &subnormal_tables));
  loom_value_id_t low_subnormal_bias = LOOM_VALUE_ID_INVALID;
  if (low_subnormal_markers == NULL) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32,
        loom_amdgpu_fp8_decode_packed_lt_bias(
            loom_amdgpu_fp8_decode_subnormal_threshold(&plan->format)),
        sgpr_type, &low_subnormal_bias));
  }
  IREE_RETURN_IF_ERROR(loom_amdgpu_get_fp8_packed_u16_mask_shift(
      context, source_op, sgpr_type, repair_state));
  for (iree_host_size_t i = 0; i < pair_count; ++i) {
    loom_value_id_t low_subnormal_payload = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_emit_fp8_subnormal_packed_u16_bits_from_byte_tables(
            context, source_op, plan, &subnormal_tables,
            pair_states[i].base.source_no_sign, vgpr_type,
            &low_subnormal_payload));
    loom_value_id_t low_repair_condition = low_subnormal_markers != NULL
                                               ? low_subnormal_markers[i]
                                               : LOOM_VALUE_ID_INVALID;
    if (low_repair_condition == LOOM_VALUE_ID_INVALID) {
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_emit_fp8_packed_u16_pair_condition_with_bias(
              context, source_op, plan, pair_states[i].base.source_no_sign,
              low_subnormal_bias, vgpr_type, &low_repair_condition));
    }
    loom_value_id_t low_repair_mask = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_emit_fp8_packed_u16_mask_from_condition_with_shift(
            context, source_op, plan, low_repair_condition,
            repair_state->low_mask_shift, vgpr_type, &low_repair_mask));
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp8_packed_u16_pair_select(
        context, source_op, plan, pair_states[i].normal_payload,
        low_subnormal_payload, low_repair_mask, vgpr_type,
        &out_repair_payloads[i]));
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_fp8_packed_u16_subnormal_repair_group_if(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fp8_decode_plan_t* plan,
    const loom_amdgpu_fp8_packed_u16_pair_state_t* pair_states,
    iree_host_size_t pair_count, loom_value_id_t low_combined_repair_condition,
    const uint32_t table_words[LOOM_AMDGPU_FP8_U16_BYTE_COUNT]
                              [LOOM_AMDGPU_FP8_U16_BYTE_TABLE_WORD_COUNT],
    loom_type_t vgpr_type, loom_type_t sgpr_type, loom_type_t mask_type,
    loom_value_id_t low_zero_sgpr64, loom_value_id_t* out_finite_payloads) {
  loom_amdgpu_fp8_packed_u16_repair_split_t split = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp8_packed_u16_repair_split(
      context, source_op, plan, low_combined_repair_condition, pair_count,
      vgpr_type, mask_type, low_zero_sgpr64, &split));
  loom_builder_t* builder = loom_low_lower_context_builder(context);

  loom_value_id_t repair_payloads[LOOM_AMDGPU_MAX_PACKED_32BIT_REGISTERS] = {0};
  loom_builder_set_block(builder, split.repair_block);
  loom_amdgpu_fp8_packed_u16_repair_state_t repair_state = {
      .low_mask_shift = LOOM_VALUE_ID_INVALID,
  };
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_emit_fp8_packed_u16_subnormal_repair_payloads(
          context, source_op, plan, pair_states, pair_count, table_words,
          vgpr_type, sgpr_type, &repair_state,
          /*low_subnormal_markers=*/NULL, repair_payloads));
  loom_op_t* repair_branch_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_low_br_build(builder, split.continuation_block, repair_payloads,
                        pair_count, source_op->location, &repair_branch_op));

  loom_value_id_t normal_payloads[LOOM_AMDGPU_MAX_PACKED_32BIT_REGISTERS] = {0};
  loom_builder_set_block(builder, split.normal_block);
  for (iree_host_size_t i = 0; i < pair_count; ++i) {
    normal_payloads[i] = pair_states[i].normal_payload;
  }
  loom_op_t* normal_branch_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_low_br_build(builder, split.continuation_block, normal_payloads,
                        pair_count, source_op->location, &normal_branch_op));

  loom_builder_set_block(builder, split.continuation_block);
  for (iree_host_size_t i = 0; i < pair_count; ++i) {
    out_finite_payloads[i] = split.continuation_values[i];
  }
  return iree_ok_status();
}

iree_status_t loom_amdgpu_emit_fp8_pairs_to_packed_f16_finite(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fp8_decode_plan_t* plan,
    const loom_amdgpu_fp8_packed_u16_pair_source_t* pair_sources,
    iree_host_size_t pair_count,
    loom_amdgpu_fp8_decode_value_flags_t value_flags, loom_type_t vgpr_type,
    loom_type_t sgpr_type, loom_type_t mask_type,
    loom_value_id_t* out_low_packets) {
  IREE_ASSERT_GT(pair_count, 0u);
  IREE_ASSERT_LE(pair_count, LOOM_AMDGPU_MAX_PACKED_32BIT_REGISTERS);
  IREE_ASSERT(
      loom_amdgpu_can_emit_fp8_pair_to_packed_f16_finite(plan, value_flags));
  for (iree_host_size_t i = 0; i < pair_count; ++i) {
    out_low_packets[i] = LOOM_VALUE_ID_INVALID;
  }
  for (iree_host_size_t i = 0; i < pair_count; ++i) {
    IREE_ASSERT_GE(pair_sources[i].live_lane_count, 1u);
    IREE_ASSERT_LE(pair_sources[i].live_lane_count, 2u);
    IREE_ASSERT_LT(pair_sources[i].byte_offset,
                   IREE_ARRAYSIZE(kLoomAmdgpuFp8DecodeU16BytePairSelectors));
  }

  const uint32_t exponent_bias = (15u - plan->format.exponent_bias) << 10;
  const uint32_t packed_exponent_bias = exponent_bias | (exponent_bias << 16);
  const bool can_use_normal_path =
      loom_amdgpu_can_emit_fp8_pair_to_packed_u16_finite_path(
          plan, value_flags,
          LOOM_AMDGPU_FP8_DECODE_PLAN_CAPABILITY_PACKED_NORMAL_F16_PAYLOAD);
  const bool use_exact_repair = !can_use_normal_path;

  const bool value_non_zero =
      iree_any_bit_set(value_flags, LOOM_AMDGPU_FP8_DECODE_VALUE_FLAG_NON_ZERO);
  const bool value_not_subnormal = iree_any_bit_set(
      value_flags, LOOM_AMDGPU_FP8_DECODE_VALUE_FLAG_NOT_SUBNORMAL);
  const bool needs_repair_condition =
      use_exact_repair && !(value_not_subnormal && value_non_zero);
  const bool use_repair_split =
      needs_repair_condition &&
      loom_amdgpu_fp8_decode_plan_has_mask_repair_split(plan);
  loom_value_id_t low_one_mask = LOOM_VALUE_ID_INVALID;
  if (!use_exact_repair && !value_non_zero) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32,
        LOOM_AMDGPU_FP8_DECODE_U16_PAIR_ONE_MASK, sgpr_type, &low_one_mask));
  }
  const bool needs_sgpr64_zero_compare =
      use_repair_split &&
      !loom_amdgpu_fp8_decode_plan_has_inline_sgpr64_zero_compare(plan);
  loom_value_id_t low_zero_sgpr64 = LOOM_VALUE_ID_INVALID;
  if (needs_sgpr64_zero_compare) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_sgpr64_constant_u64(
        context, source_op, 0, &low_zero_sgpr64));
  }

  loom_amdgpu_fp8_packed_u16_pair_state_t
      pair_states[LOOM_AMDGPU_MAX_PACKED_32BIT_REGISTERS] = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp8_packed_u16_pair_states(
      context, source_op, plan, pair_sources, pair_count,
      10u - plan->format.mantissa_bits, packed_exponent_bias, vgpr_type,
      sgpr_type, pair_states));

  loom_value_id_t finite_payloads[LOOM_AMDGPU_MAX_PACKED_32BIT_REGISTERS] = {0};
  if (use_exact_repair) {
    if (use_repair_split) {
      loom_value_id_t low_combined_repair_condition = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_emit_fp8_packed_u16_combined_subnormal_condition(
              context, source_op, plan, pair_states, pair_count, vgpr_type,
              sgpr_type, &low_combined_repair_condition));
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_emit_fp8_packed_u16_subnormal_repair_group_if(
              context, source_op, plan, pair_states, pair_count,
              low_combined_repair_condition,
              plan->subnormal_f16_byte_table_words, vgpr_type, sgpr_type,
              mask_type, low_zero_sgpr64, finite_payloads));
    } else {
      loom_amdgpu_fp8_packed_u16_repair_state_t repair_state = {
          .low_mask_shift = LOOM_VALUE_ID_INVALID,
      };
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_emit_fp8_packed_u16_subnormal_repair_payloads(
              context, source_op, plan, pair_states, pair_count,
              plan->subnormal_f16_byte_table_words, vgpr_type, sgpr_type,
              &repair_state, /*low_subnormal_markers=*/NULL, finite_payloads));
    }
  } else if (!value_non_zero) {
    for (iree_host_size_t i = 0; i < pair_count; ++i) {
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_emit_fp8_packed_u16_zero_repaired_payload(
              context, source_op, plan, pair_states[i].base.source_no_sign,
              pair_states[i].normal_payload, value_flags, low_one_mask,
              vgpr_type, &finite_payloads[i]));
    }
  } else {
    for (iree_host_size_t i = 0; i < pair_count; ++i) {
      finite_payloads[i] = pair_states[i].normal_payload;
    }
  }

  return loom_amdgpu_emit_fp8_packed_u16_pair_signs(
      context, source_op, plan, pair_states, finite_payloads, pair_count,
      vgpr_type, out_low_packets);
}

typedef struct loom_amdgpu_fp8_packed_bf16_non_normal_markers_t {
  // Packed per-pair values biased so subnormal lanes have the sign bit set.
  loom_value_id_t shifted_pairs[LOOM_AMDGPU_MAX_PACKED_32BIT_REGISTERS];
  // Threshold in shifted-pair space where upper-end repairs begin.
  uint32_t upper_shifted_threshold;
} loom_amdgpu_fp8_packed_bf16_non_normal_markers_t;

typedef struct loom_amdgpu_fp8_exact_bf16_via_f16_state_t {
  // Byte expansion state shared by every packed pair.
  loom_amdgpu_fp8_packed_u16_permute_state_t permute;
  // Packed per-lane right shift placing FP8 fields into F16 fields.
  loom_value_id_t low_input_shift;
  // Packed F16 power-of-two scale correcting the FP8 exponent bias.
  loom_value_id_t low_scale;
  // Packed per-lane shift dropping unused F16 mantissa bits.
  loom_value_id_t low_output_shift;
  // Packed per-lane one used to distinguish zero from nonzero payloads.
  loom_value_id_t low_one;
  // Packed BF16 exponent-bias addend.
  loom_value_id_t low_bf16_bias;
} loom_amdgpu_fp8_exact_bf16_via_f16_state_t;

static iree_status_t loom_amdgpu_emit_fp8_exact_bf16_via_f16_state(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fp8_decode_plan_t* plan, loom_type_t vgpr_type,
    loom_type_t sgpr_type,
    loom_amdgpu_fp8_exact_bf16_via_f16_state_t* out_state) {
  *out_state = (loom_amdgpu_fp8_exact_bf16_via_f16_state_t){
      .permute = {0},
      .low_input_shift = LOOM_VALUE_ID_INVALID,
      .low_scale = LOOM_VALUE_ID_INVALID,
      .low_output_shift = LOOM_VALUE_ID_INVALID,
      .low_one = LOOM_VALUE_ID_INVALID,
      .low_bf16_bias = LOOM_VALUE_ID_INVALID,
  };

  const uint32_t input_shift = plan->format.mantissa_bits - 2u;
  const uint32_t packed_input_shift = input_shift | (input_shift << 16);
  const uint32_t scale_bits = (30u - plan->format.exponent_bias) << 10;
  const uint32_t packed_scale_bits = scale_bits | (scale_bits << 16);
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp8_packed_u16_permute_state(
      context, source_op, plan, /*selector_shift=*/8, vgpr_type,
      &out_state->permute));
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32,
      packed_scale_bits, sgpr_type, &out_state->low_scale));
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32,
      UINT32_C(0x00030003), sgpr_type, &out_state->low_output_shift));
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32,
      LOOM_AMDGPU_FP8_DECODE_U16_PAIR_ONE_MASK, sgpr_type,
      &out_state->low_one));
  if (packed_input_shift == LOOM_AMDGPU_FP8_DECODE_U16_PAIR_ONE_MASK) {
    out_state->low_input_shift = out_state->low_one;
  } else {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32,
        packed_input_shift, sgpr_type, &out_state->low_input_shift));
  }
  return loom_amdgpu_emit_const_u32(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32,
      UINT32_C(0x38003800), sgpr_type, &out_state->low_bf16_bias);
}

static iree_status_t loom_amdgpu_emit_fp8_pairs_to_packed_bf16_via_f16(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fp8_decode_plan_t* plan,
    const loom_amdgpu_fp8_packed_u16_pair_source_t* pair_sources,
    iree_host_size_t pair_count, loom_type_t vgpr_type, loom_type_t sgpr_type,
    loom_value_id_t* out_low_packets) {
  loom_amdgpu_fp8_exact_bf16_via_f16_state_t state;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp8_exact_bf16_via_f16_state(
      context, source_op, plan, vgpr_type, sgpr_type, &state));

  for (iree_host_size_t i = 0; i < pair_count; ++i) {
    IREE_ASSERT_GE(pair_sources[i].live_lane_count, 1u);
    IREE_ASSERT_LE(pair_sources[i].live_lane_count, 2u);
    IREE_ASSERT_LT(pair_sources[i].byte_offset,
                   LOOM_AMDGPU_FP8_DECODE_U16_PAIR_SELECTOR_COUNT);
    loom_amdgpu_fp8_packed_u16_pair_base_t base = {
        .expanded_pair = LOOM_VALUE_ID_INVALID,
        .source_no_sign = LOOM_VALUE_ID_INVALID,
        .sign_bits = LOOM_VALUE_ID_INVALID,
    };
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp8_packed_u16_pair_permute(
        context, source_op, plan, &pair_sources[i], &state.permute, vgpr_type,
        sgpr_type, &base.expanded_pair));
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary_immediate(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT,
        base.expanded_pair,
        LOOM_AMDGPU_FP8_DECODE_U16_PAIR_HIGH_BYTE_NO_SIGN_MASK, vgpr_type,
        &base.source_no_sign));
    if (iree_any_bit_set(
            plan->flags,
            LOOM_AMDGPU_FP8_DECODE_PLAN_FLAG_HAS_BFI_B32_SRC0_LITERAL)) {
      base.sign_bits = base.expanded_pair;
    } else {
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary_immediate(
          context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT,
          base.expanded_pair, LOOM_AMDGPU_FP8_DECODE_U16_PAIR_SIGN_INSERT_MASK,
          vgpr_type, &base.sign_bits));
    }

    loom_value_id_t low_f16_input = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_resolved_vgpr_binary(
        context, source_op, &plan->pk_lshrrev_b16_descriptor,
        state.low_input_shift, base.source_no_sign, vgpr_type, &low_f16_input));
    loom_value_id_t low_nonzero = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_resolved_vgpr_binary(
        context, source_op, &plan->pk_min_u16_descriptor, low_f16_input,
        state.low_one, vgpr_type, &low_nonzero));
    loom_value_id_t low_f16_value = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_resolved_vgpr_binary(
        context, source_op, &plan->pk_mul_f16_descriptor, low_f16_input,
        state.low_scale, vgpr_type, &low_f16_value));
    loom_value_id_t low_shifted_bf16 = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_resolved_vgpr_binary(
        context, source_op, &plan->pk_lshrrev_b16_descriptor,
        state.low_output_shift, low_f16_value, vgpr_type, &low_shifted_bf16));
    loom_value_id_t low_unsigned_bf16 = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_resolved_vgpr_ternary(
        context, source_op, &plan->pk_mad_u16_descriptor, low_nonzero,
        state.low_bf16_bias, low_shifted_bf16, vgpr_type, &low_unsigned_bf16));
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp8_packed_u16_pair_sign(
        context, source_op, plan, &base, low_unsigned_bf16, vgpr_type,
        &out_low_packets[i]));
  }
  return iree_ok_status();
}
static iree_status_t loom_amdgpu_emit_fp8_packed_bf16_pair_nan_condition(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fp8_decode_plan_t* plan,
    const loom_amdgpu_fp8_packed_u16_pair_state_t* state, loom_type_t vgpr_type,
    loom_type_t sgpr_type, loom_value_id_t* out_condition) {
  *out_condition = LOOM_VALUE_ID_INVALID;
  const loom_scalar_type_fp8_format_t* format = &plan->format;
  if (format->special_policy ==
      LOOM_SCALAR_TYPE_FP8_SPECIAL_POLICY_FINITE_NAN) {
    return loom_amdgpu_emit_fp8_packed_u16_pair_eq_condition(
        context, source_op, plan, state->base.source_no_sign,
        loom_amdgpu_fp8_decode_packed_u16(
            loom_amdgpu_fp8_decode_finite_nan_no_sign(format)),
        vgpr_type, sgpr_type, out_condition);
  }
  return loom_amdgpu_emit_fp8_packed_u16_pair_ge_condition(
      context, source_op, plan, state->base.source_no_sign,
      loom_amdgpu_fp8_decode_top_exponent_no_sign(format) + 1u, vgpr_type,
      sgpr_type, out_condition);
}

static iree_status_t
loom_amdgpu_emit_fp8_packed_bf16_pair_nan_condition_with_bias(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fp8_decode_plan_t* plan,
    const loom_amdgpu_fp8_packed_u16_pair_state_t* state,
    loom_value_id_t low_eq_zero_bias, loom_type_t vgpr_type,
    loom_value_id_t* out_condition) {
  *out_condition = LOOM_VALUE_ID_INVALID;
  return loom_amdgpu_emit_fp8_packed_u16_pair_eq_condition_with_bias(
      context, source_op, plan, state->base.source_no_sign,
      loom_amdgpu_fp8_decode_packed_u16(
          loom_amdgpu_fp8_decode_finite_nan_no_sign(&plan->format)),
      low_eq_zero_bias, vgpr_type, out_condition);
}

static iree_status_t loom_amdgpu_emit_fp8_packed_bf16_combined_eq_condition(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fp8_decode_plan_t* plan,
    const loom_amdgpu_fp8_packed_u16_pair_state_t* pair_states,
    iree_host_size_t pair_count, uint32_t packed_value, loom_type_t vgpr_type,
    loom_type_t sgpr_type, loom_value_id_t* out_condition) {
  *out_condition = LOOM_VALUE_ID_INVALID;
  loom_value_id_t low_deltas[LOOM_AMDGPU_MAX_PACKED_32BIT_REGISTERS] = {0};
  for (iree_host_size_t i = 0; i < pair_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary_immediate(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_XOR_B32_LIT,
        pair_states[i].base.source_no_sign, packed_value, vgpr_type,
        &low_deltas[i]));
  }
  loom_value_id_t low_min_delta = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp8_decode_resolved_vgpr_binary_reduce(
      context, source_op, &plan->pk_min_u16_descriptor, low_deltas, pair_count,
      vgpr_type, &low_min_delta));
  loom_value_id_t low_condition = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp8_packed_u16_pair_lt_condition(
      context, source_op, plan, low_min_delta,
      LOOM_AMDGPU_FP8_DECODE_U16_PAIR_EQ_ZERO_BIAS, vgpr_type, sgpr_type,
      &low_condition));
  return loom_amdgpu_emit_fp8_packed_u16_split_condition(
      context, source_op, &low_condition, 1, vgpr_type, out_condition);
}

static iree_status_t loom_amdgpu_emit_fp8_packed_bf16_combined_ge_condition(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fp8_decode_plan_t* plan,
    const loom_amdgpu_fp8_packed_u16_pair_state_t* pair_states,
    iree_host_size_t pair_count, uint32_t lane_threshold, loom_type_t vgpr_type,
    loom_type_t sgpr_type, loom_value_id_t* out_condition) {
  *out_condition = LOOM_VALUE_ID_INVALID;
  loom_value_id_t low_sources[LOOM_AMDGPU_MAX_PACKED_32BIT_REGISTERS] = {0};
  for (iree_host_size_t i = 0; i < pair_count; ++i) {
    low_sources[i] = pair_states[i].base.source_no_sign;
  }
  loom_value_id_t low_max_source_no_sign = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp8_decode_resolved_vgpr_binary_reduce(
      context, source_op, &plan->pk_max_u16_descriptor, low_sources, pair_count,
      vgpr_type, &low_max_source_no_sign));
  loom_value_id_t low_condition = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp8_packed_u16_pair_ge_condition(
      context, source_op, plan, low_max_source_no_sign, lane_threshold,
      vgpr_type, sgpr_type, &low_condition));
  return loom_amdgpu_emit_fp8_packed_u16_split_condition(
      context, source_op, &low_condition, 1, vgpr_type, out_condition);
}

static iree_status_t
loom_amdgpu_emit_fp8_packed_bf16_combined_finite_nan_condition(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fp8_decode_plan_t* plan,
    const loom_amdgpu_fp8_packed_u16_pair_state_t* pair_states,
    iree_host_size_t pair_count, loom_type_t vgpr_type, loom_type_t sgpr_type,
    loom_value_id_t* out_condition) {
  const uint32_t finite_nan_no_sign =
      loom_amdgpu_fp8_decode_finite_nan_no_sign(&plan->format);
  if (iree_any_bit_set(plan->flags,
                       LOOM_AMDGPU_FP8_DECODE_PLAN_FLAG_HAS_PK_MAX_U16)) {
    return loom_amdgpu_emit_fp8_packed_bf16_combined_ge_condition(
        context, source_op, plan, pair_states, pair_count, finite_nan_no_sign,
        vgpr_type, sgpr_type, out_condition);
  }
  return loom_amdgpu_emit_fp8_packed_bf16_combined_eq_condition(
      context, source_op, plan, pair_states, pair_count,
      loom_amdgpu_fp8_decode_packed_u16(finite_nan_no_sign), vgpr_type,
      sgpr_type, out_condition);
}

static iree_status_t
loom_amdgpu_emit_fp8_packed_bf16_combined_non_normal_condition(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fp8_decode_plan_t* plan,
    const loom_amdgpu_fp8_packed_u16_pair_state_t* pair_states,
    iree_host_size_t pair_count, loom_type_t vgpr_type, loom_type_t sgpr_type,
    uint32_t upper_repair_threshold,
    loom_amdgpu_fp8_packed_bf16_non_normal_markers_t* out_markers,
    loom_value_id_t* out_condition) {
  *out_condition = LOOM_VALUE_ID_INVALID;
  if (out_markers != NULL) {
    *out_markers = (loom_amdgpu_fp8_packed_bf16_non_normal_markers_t){0};
  }
  IREE_ASSERT(
      loom_amdgpu_fp8_decode_plan_has_combined_non_normal_condition(plan));
  IREE_ASSERT_GT(pair_count, 0);
  const uint32_t subnormal_threshold =
      loom_amdgpu_fp8_decode_subnormal_threshold(&plan->format);
  IREE_ASSERT_GE(upper_repair_threshold, subnormal_threshold);
  const uint32_t upper_shifted_threshold =
      upper_repair_threshold - subnormal_threshold;
  if (out_markers != NULL) {
    out_markers->upper_shifted_threshold = upper_shifted_threshold;
  }

  // Biasing by -subnormal_threshold wraps subnormal lanes above the normal
  // range, letting one packed maximum detect either low or upper-end repairs.
  loom_value_id_t low_threshold_bias = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32,
      loom_amdgpu_fp8_decode_packed_lt_bias(subnormal_threshold), sgpr_type,
      &low_threshold_bias));
  loom_value_id_t low_shifted_pairs[LOOM_AMDGPU_MAX_PACKED_32BIT_REGISTERS] = {
      0};
  for (iree_host_size_t i = 0; i < pair_count; ++i) {
    loom_value_id_t low_shifted = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_resolved_vgpr_binary(
        context, source_op, &plan->pk_add_u16_descriptor,
        pair_states[i].base.source_no_sign, low_threshold_bias, vgpr_type,
        &low_shifted));
    if (out_markers != NULL) {
      out_markers->shifted_pairs[i] = low_shifted;
    }
    low_shifted_pairs[i] = low_shifted;
  }
  loom_value_id_t low_max_shifted = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp8_decode_resolved_vgpr_binary_reduce(
      context, source_op, &plan->pk_max_u16_descriptor, low_shifted_pairs,
      pair_count, vgpr_type, &low_max_shifted));
  loom_value_id_t low_condition = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp8_packed_u16_pair_ge_condition(
      context, source_op, plan, low_max_shifted, upper_shifted_threshold,
      vgpr_type, sgpr_type, &low_condition));
  const loom_value_id_t repair_conditions[] = {
      // The threshold bias already marks subnormal lanes with the high bit.
      low_max_shifted,
      // Upper-end repairs are detected in the normal biased range.
      low_condition,
  };
  return loom_amdgpu_emit_fp8_packed_u16_split_condition(
      context, source_op, repair_conditions, IREE_ARRAYSIZE(repair_conditions),
      vgpr_type, out_condition);
}

static iree_status_t loom_amdgpu_emit_fp8_packed_bf16_nan_repair_packets(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fp8_decode_plan_t* plan,
    const loom_amdgpu_fp8_packed_u16_pair_state_t* pair_states,
    const loom_value_id_t* no_nan_packets, iree_host_size_t pair_count,
    bool has_hot_nan_conditions, loom_value_id_t* nan_conditions,
    loom_type_t vgpr_type, loom_type_t sgpr_type,
    loom_amdgpu_fp8_packed_u16_repair_state_t* repair_state,
    loom_value_id_t* out_packets) {
  loom_value_id_t low_eq_zero_bias = LOOM_VALUE_ID_INVALID;
  loom_value_id_t low_nan_ge_bias = LOOM_VALUE_ID_INVALID;
  if (!has_hot_nan_conditions) {
    if (plan->format.special_policy ==
        LOOM_SCALAR_TYPE_FP8_SPECIAL_POLICY_FINITE_NAN) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
          context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32,
          LOOM_AMDGPU_FP8_DECODE_U16_PAIR_EQ_ZERO_BIAS, sgpr_type,
          &low_eq_zero_bias));
    } else {
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
          context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32,
          loom_amdgpu_fp8_decode_packed_ge_bias(
              loom_amdgpu_fp8_decode_top_exponent_no_sign(&plan->format) + 1u),
          sgpr_type, &low_nan_ge_bias));
    }
  }
  IREE_RETURN_IF_ERROR(loom_amdgpu_get_fp8_packed_u16_mask_shift(
      context, source_op, sgpr_type, repair_state));
  loom_value_id_t low_quiet_nan = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32,
      LOOM_AMDGPU_FP8_DECODE_BF16_PAIR_QUIET_NAN_BITS, vgpr_type,
      &low_quiet_nan));
  for (iree_host_size_t i = 0; i < pair_count; ++i) {
    if (!has_hot_nan_conditions) {
      if (plan->format.special_policy ==
          LOOM_SCALAR_TYPE_FP8_SPECIAL_POLICY_FINITE_NAN) {
        IREE_RETURN_IF_ERROR(
            loom_amdgpu_emit_fp8_packed_bf16_pair_nan_condition_with_bias(
                context, source_op, plan, &pair_states[i], low_eq_zero_bias,
                vgpr_type, &nan_conditions[i]));
      } else {
        IREE_RETURN_IF_ERROR(
            loom_amdgpu_emit_fp8_packed_u16_pair_condition_with_bias(
                context, source_op, plan, pair_states[i].base.source_no_sign,
                low_nan_ge_bias, vgpr_type, &nan_conditions[i]));
      }
    }
    loom_value_id_t nan_mask = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_emit_fp8_packed_u16_mask_from_condition_with_shift(
            context, source_op, plan, nan_conditions[i],
            repair_state->low_mask_shift, vgpr_type, &nan_mask));
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp8_packed_u16_pair_select(
        context, source_op, plan, no_nan_packets[i], low_quiet_nan, nan_mask,
        vgpr_type, &out_packets[i]));
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_fp8_packed_bf16_inf_repair_packets(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fp8_decode_plan_t* plan,
    const loom_amdgpu_fp8_packed_u16_pair_state_t* pair_states,
    const loom_value_id_t* no_inf_packets, iree_host_size_t pair_count,
    loom_type_t vgpr_type, loom_type_t sgpr_type,
    loom_amdgpu_fp8_packed_u16_repair_state_t* repair_state,
    loom_value_id_t* out_packets) {
  loom_value_id_t low_eq_zero_bias = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32,
      LOOM_AMDGPU_FP8_DECODE_U16_PAIR_EQ_ZERO_BIAS, sgpr_type,
      &low_eq_zero_bias));
  IREE_RETURN_IF_ERROR(loom_amdgpu_get_fp8_packed_u16_mask_shift(
      context, source_op, sgpr_type, repair_state));
  loom_value_id_t low_infinity_payload = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32,
      LOOM_AMDGPU_FP8_DECODE_BF16_PAIR_INFINITY_BITS, vgpr_type,
      &low_infinity_payload));
  const uint32_t infinity_no_sign = loom_amdgpu_fp8_decode_packed_u16(
      loom_amdgpu_fp8_decode_top_exponent_no_sign(&plan->format));
  for (iree_host_size_t i = 0; i < pair_count; ++i) {
    loom_value_id_t low_infinity_packet = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp8_packed_u16_pair_sign(
        context, source_op, plan, &pair_states[i].base, low_infinity_payload,
        vgpr_type, &low_infinity_packet));
    loom_value_id_t low_inf_condition = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_emit_fp8_packed_u16_pair_eq_condition_with_bias(
            context, source_op, plan, pair_states[i].base.source_no_sign,
            infinity_no_sign, low_eq_zero_bias, vgpr_type, &low_inf_condition));
    loom_value_id_t low_inf_mask = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_emit_fp8_packed_u16_mask_from_condition_with_shift(
            context, source_op, plan, low_inf_condition,
            repair_state->low_mask_shift, vgpr_type, &low_inf_mask));
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp8_packed_u16_pair_select(
        context, source_op, plan, no_inf_packets[i], low_infinity_packet,
        low_inf_mask, vgpr_type, &out_packets[i]));
  }
  return iree_ok_status();
}

static iree_status_t
loom_amdgpu_emit_fp8_packed_bf16_ieee_top_exponent_repair_packets(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fp8_decode_plan_t* plan,
    const loom_amdgpu_fp8_packed_u16_pair_state_t* pair_states,
    const loom_value_id_t* no_top_exponent_packets, iree_host_size_t pair_count,
    bool has_hot_top_exponent_conditions,
    loom_value_id_t* top_exponent_conditions, loom_type_t vgpr_type,
    loom_type_t sgpr_type,
    loom_amdgpu_fp8_packed_u16_repair_state_t* repair_state,
    loom_value_id_t* out_packets) {
  IREE_ASSERT_EQ(plan->format.special_policy,
                 LOOM_SCALAR_TYPE_FP8_SPECIAL_POLICY_IEEE);

  loom_value_id_t low_top_exponent_ge_bias = LOOM_VALUE_ID_INVALID;
  if (!has_hot_top_exponent_conditions) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32,
        loom_amdgpu_fp8_decode_packed_ge_bias(
            loom_amdgpu_fp8_decode_top_exponent_no_sign(&plan->format)),
        sgpr_type, &low_top_exponent_ge_bias));
  }
  loom_value_id_t low_nan_ge_bias = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32,
      loom_amdgpu_fp8_decode_packed_ge_bias(
          loom_amdgpu_fp8_decode_top_exponent_no_sign(&plan->format) + 1u),
      sgpr_type, &low_nan_ge_bias));
  IREE_RETURN_IF_ERROR(loom_amdgpu_get_fp8_packed_u16_mask_shift(
      context, source_op, sgpr_type, repair_state));
  loom_value_id_t low_infinity_payload = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32,
      LOOM_AMDGPU_FP8_DECODE_BF16_PAIR_INFINITY_BITS, vgpr_type,
      &low_infinity_payload));
  loom_value_id_t low_quiet_nan = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32,
      LOOM_AMDGPU_FP8_DECODE_BF16_PAIR_QUIET_NAN_BITS, vgpr_type,
      &low_quiet_nan));

  for (iree_host_size_t i = 0; i < pair_count; ++i) {
    loom_value_id_t low_top_exponent_condition = top_exponent_conditions[i];
    if (!has_hot_top_exponent_conditions) {
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_emit_fp8_packed_u16_pair_condition_with_bias(
              context, source_op, plan, pair_states[i].base.source_no_sign,
              low_top_exponent_ge_bias, vgpr_type,
              &low_top_exponent_condition));
    }
    loom_value_id_t low_top_exponent_mask = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_emit_fp8_packed_u16_mask_from_condition_with_shift(
            context, source_op, plan, low_top_exponent_condition,
            repair_state->low_mask_shift, vgpr_type, &low_top_exponent_mask));

    loom_value_id_t low_nan_condition = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_emit_fp8_packed_u16_pair_condition_with_bias(
            context, source_op, plan, pair_states[i].base.source_no_sign,
            low_nan_ge_bias, vgpr_type, &low_nan_condition));
    loom_value_id_t low_nan_mask = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_emit_fp8_packed_u16_mask_from_condition_with_shift(
            context, source_op, plan, low_nan_condition,
            repair_state->low_mask_shift, vgpr_type, &low_nan_mask));

    loom_value_id_t low_infinity_packet = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp8_packed_u16_pair_sign(
        context, source_op, plan, &pair_states[i].base, low_infinity_payload,
        vgpr_type, &low_infinity_packet));
    loom_value_id_t low_top_exponent_packet = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp8_packed_u16_pair_select(
        context, source_op, plan, low_infinity_packet, low_quiet_nan,
        low_nan_mask, vgpr_type, &low_top_exponent_packet));
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp8_packed_u16_pair_select(
        context, source_op, plan, no_top_exponent_packets[i],
        low_top_exponent_packet, low_top_exponent_mask, vgpr_type,
        &out_packets[i]));
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_fp8_packed_bf16_nan_group_if(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fp8_decode_plan_t* plan,
    const loom_amdgpu_fp8_packed_u16_pair_state_t* pair_states,
    const loom_value_id_t* no_nan_packets, iree_host_size_t pair_count,
    loom_type_t vgpr_type, loom_type_t sgpr_type, loom_type_t mask_type,
    loom_value_id_t low_zero_sgpr64, loom_value_id_t* out_packets) {
  loom_value_id_t nan_conditions[LOOM_AMDGPU_MAX_PACKED_32BIT_REGISTERS] = {0};
  bool has_hot_nan_conditions = false;
  loom_value_id_t low_combined_nan_condition = LOOM_VALUE_ID_INVALID;
  if (loom_amdgpu_fp8_decode_plan_has_combined_finite_nan_condition(plan)) {
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_emit_fp8_packed_bf16_combined_finite_nan_condition(
            context, source_op, plan, pair_states, pair_count, vgpr_type,
            sgpr_type, &low_combined_nan_condition));
  } else {
    has_hot_nan_conditions = true;
    for (iree_host_size_t i = 0; i < pair_count; ++i) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp8_packed_bf16_pair_nan_condition(
          context, source_op, plan, &pair_states[i], vgpr_type, sgpr_type,
          &nan_conditions[i]));
    }
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp8_packed_u16_split_condition(
        context, source_op, nan_conditions, pair_count, vgpr_type,
        &low_combined_nan_condition));
  }

  loom_amdgpu_fp8_packed_u16_repair_split_t split = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp8_packed_u16_repair_split(
      context, source_op, plan, low_combined_nan_condition, pair_count,
      vgpr_type, mask_type, low_zero_sgpr64, &split));
  loom_builder_t* builder = loom_low_lower_context_builder(context);

  loom_value_id_t repair_packets[LOOM_AMDGPU_MAX_PACKED_32BIT_REGISTERS] = {0};
  loom_builder_set_block(builder, split.repair_block);
  loom_amdgpu_fp8_packed_u16_repair_state_t repair_state = {
      .low_mask_shift = LOOM_VALUE_ID_INVALID,
  };
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp8_packed_bf16_nan_repair_packets(
      context, source_op, plan, pair_states, no_nan_packets, pair_count,
      has_hot_nan_conditions, nan_conditions, vgpr_type, sgpr_type,
      &repair_state, repair_packets));
  loom_op_t* repair_branch_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_low_br_build(builder, split.continuation_block, repair_packets,
                        pair_count, source_op->location, &repair_branch_op));

  loom_builder_set_block(builder, split.normal_block);
  loom_op_t* normal_branch_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_low_br_build(builder, split.continuation_block, no_nan_packets,
                        pair_count, source_op->location, &normal_branch_op));

  loom_builder_set_block(builder, split.continuation_block);
  for (iree_host_size_t i = 0; i < pair_count; ++i) {
    out_packets[i] = split.continuation_values[i];
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_fp8_packed_bf16_top_exponent_condition(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fp8_decode_plan_t* plan,
    const loom_amdgpu_fp8_packed_u16_pair_state_t* state,
    loom_amdgpu_fp8_packed_u16_repairs_t top_exponent_repairs,
    loom_type_t vgpr_type, loom_type_t sgpr_type,
    loom_value_id_t* out_condition) {
  *out_condition = LOOM_VALUE_ID_INVALID;
  const uint32_t top_exponent_no_sign =
      loom_amdgpu_fp8_decode_top_exponent_no_sign(&plan->format);
  const bool repair_inf = iree_any_bit_set(
      top_exponent_repairs, LOOM_AMDGPU_FP8_PACKED_U16_REPAIR_INF);
  const bool repair_nan = iree_any_bit_set(
      top_exponent_repairs, LOOM_AMDGPU_FP8_PACKED_U16_REPAIR_NAN);
  IREE_ASSERT(repair_inf || repair_nan);
  if (repair_inf && repair_nan) {
    return loom_amdgpu_emit_fp8_packed_u16_pair_ge_condition(
        context, source_op, plan, state->base.source_no_sign,
        top_exponent_no_sign, vgpr_type, sgpr_type, out_condition);
  }
  if (repair_inf) {
    return loom_amdgpu_emit_fp8_packed_u16_pair_eq_condition(
        context, source_op, plan, state->base.source_no_sign,
        top_exponent_no_sign, vgpr_type, sgpr_type, out_condition);
  }
  return loom_amdgpu_emit_fp8_packed_bf16_pair_nan_condition(
      context, source_op, plan, state, vgpr_type, sgpr_type, out_condition);
}

static iree_status_t
loom_amdgpu_emit_fp8_packed_bf16_combined_top_exponent_condition(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fp8_decode_plan_t* plan,
    const loom_amdgpu_fp8_packed_u16_pair_state_t* pair_states,
    iree_host_size_t pair_count,
    loom_amdgpu_fp8_packed_u16_repairs_t top_exponent_repairs,
    loom_type_t vgpr_type, loom_type_t sgpr_type,
    loom_value_id_t* out_pair_conditions, bool* out_pair_conditions_available,
    loom_value_id_t* out_condition) {
  *out_condition = LOOM_VALUE_ID_INVALID;
  *out_pair_conditions_available = false;
  const bool repair_inf = iree_any_bit_set(
      top_exponent_repairs, LOOM_AMDGPU_FP8_PACKED_U16_REPAIR_INF);
  const bool repair_nan = iree_any_bit_set(
      top_exponent_repairs, LOOM_AMDGPU_FP8_PACKED_U16_REPAIR_NAN);
  IREE_ASSERT(repair_inf || repair_nan);

  const uint32_t top_exponent_no_sign =
      loom_amdgpu_fp8_decode_top_exponent_no_sign(&plan->format);
  if (!repair_nan && repair_inf &&
      iree_any_bit_set(plan->flags,
                       LOOM_AMDGPU_FP8_DECODE_PLAN_FLAG_HAS_PK_MIN_U16)) {
    return loom_amdgpu_emit_fp8_packed_bf16_combined_eq_condition(
        context, source_op, plan, pair_states, pair_count,
        loom_amdgpu_fp8_decode_packed_u16(top_exponent_no_sign), vgpr_type,
        sgpr_type, out_condition);
  }

  if (iree_any_bit_set(plan->flags,
                       LOOM_AMDGPU_FP8_DECODE_PLAN_FLAG_HAS_PK_MAX_U16)) {
    const uint32_t threshold =
        repair_inf ? top_exponent_no_sign : top_exponent_no_sign + 1u;
    return loom_amdgpu_emit_fp8_packed_bf16_combined_ge_condition(
        context, source_op, plan, pair_states, pair_count, threshold, vgpr_type,
        sgpr_type, out_condition);
  }

  for (iree_host_size_t i = 0; i < pair_count; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_emit_fp8_packed_bf16_top_exponent_condition(
            context, source_op, plan, &pair_states[i], top_exponent_repairs,
            vgpr_type, sgpr_type, &out_pair_conditions[i]));
  }
  *out_pair_conditions_available = true;
  return loom_amdgpu_emit_fp8_packed_u16_split_condition(
      context, source_op, out_pair_conditions, pair_count, vgpr_type,
      out_condition);
}

static iree_status_t
loom_amdgpu_emit_fp8_packed_bf16_top_exponent_repair_packets(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fp8_decode_plan_t* plan,
    const loom_amdgpu_fp8_packed_u16_pair_state_t* pair_states,
    const loom_value_id_t* no_top_exponent_packets, iree_host_size_t pair_count,
    loom_amdgpu_fp8_packed_u16_repairs_t top_exponent_repairs,
    bool has_hot_top_exponent_conditions,
    loom_value_id_t* top_exponent_conditions, loom_type_t vgpr_type,
    loom_type_t sgpr_type,
    loom_amdgpu_fp8_packed_u16_repair_state_t* repair_state,
    loom_value_id_t* out_packets) {
  const bool repair_inf = iree_any_bit_set(
      top_exponent_repairs, LOOM_AMDGPU_FP8_PACKED_U16_REPAIR_INF);
  const bool repair_nan = iree_any_bit_set(
      top_exponent_repairs, LOOM_AMDGPU_FP8_PACKED_U16_REPAIR_NAN);
  IREE_ASSERT(repair_inf || repair_nan);
  if (repair_inf && repair_nan) {
    return loom_amdgpu_emit_fp8_packed_bf16_ieee_top_exponent_repair_packets(
        context, source_op, plan, pair_states, no_top_exponent_packets,
        pair_count, has_hot_top_exponent_conditions, top_exponent_conditions,
        vgpr_type, sgpr_type, repair_state, out_packets);
  }

  loom_value_id_t inf_repaired_packets[LOOM_AMDGPU_MAX_PACKED_32BIT_REGISTERS] =
      {0};
  const loom_value_id_t* no_nan_packets = no_top_exponent_packets;
  if (repair_inf) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp8_packed_bf16_inf_repair_packets(
        context, source_op, plan, pair_states, no_top_exponent_packets,
        pair_count, vgpr_type, sgpr_type, repair_state, inf_repaired_packets));
    no_nan_packets = inf_repaired_packets;
  }
  if (repair_nan) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp8_packed_bf16_nan_repair_packets(
        context, source_op, plan, pair_states, no_nan_packets, pair_count,
        has_hot_top_exponent_conditions && !repair_inf, top_exponent_conditions,
        vgpr_type, sgpr_type, repair_state, out_packets));
    return iree_ok_status();
  }
  for (iree_host_size_t i = 0; i < pair_count; ++i) {
    out_packets[i] = no_nan_packets[i];
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_fp8_packed_bf16_top_exponent_group_if(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fp8_decode_plan_t* plan,
    const loom_amdgpu_fp8_packed_u16_pair_state_t* pair_states,
    const loom_value_id_t* no_top_exponent_packets, iree_host_size_t pair_count,
    loom_amdgpu_fp8_packed_u16_repairs_t top_exponent_repairs,
    loom_type_t vgpr_type, loom_type_t sgpr_type, loom_type_t mask_type,
    loom_value_id_t low_zero_sgpr64, loom_value_id_t* out_packets) {
  loom_value_id_t
      top_exponent_conditions[LOOM_AMDGPU_MAX_PACKED_32BIT_REGISTERS] = {0};
  bool has_hot_top_exponent_conditions = false;
  loom_value_id_t low_combined_top_exponent_condition = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_emit_fp8_packed_bf16_combined_top_exponent_condition(
          context, source_op, plan, pair_states, pair_count,
          top_exponent_repairs, vgpr_type, sgpr_type, top_exponent_conditions,
          &has_hot_top_exponent_conditions,
          &low_combined_top_exponent_condition));

  loom_amdgpu_fp8_packed_u16_repair_split_t split = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp8_packed_u16_repair_split(
      context, source_op, plan, low_combined_top_exponent_condition, pair_count,
      vgpr_type, mask_type, low_zero_sgpr64, &split));
  loom_builder_t* builder = loom_low_lower_context_builder(context);

  loom_value_id_t repair_packets[LOOM_AMDGPU_MAX_PACKED_32BIT_REGISTERS] = {0};
  loom_builder_set_block(builder, split.repair_block);
  loom_amdgpu_fp8_packed_u16_repair_state_t repair_state = {
      .low_mask_shift = LOOM_VALUE_ID_INVALID,
  };
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_emit_fp8_packed_bf16_top_exponent_repair_packets(
          context, source_op, plan, pair_states, no_top_exponent_packets,
          pair_count, top_exponent_repairs, has_hot_top_exponent_conditions,
          top_exponent_conditions, vgpr_type, sgpr_type, &repair_state,
          repair_packets));
  loom_op_t* repair_branch_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_low_br_build(builder, split.continuation_block, repair_packets,
                        pair_count, source_op->location, &repair_branch_op));

  loom_builder_set_block(builder, split.normal_block);
  loom_op_t* normal_branch_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_br_build(
      builder, split.continuation_block, no_top_exponent_packets, pair_count,
      source_op->location, &normal_branch_op));

  loom_builder_set_block(builder, split.continuation_block);
  for (iree_host_size_t i = 0; i < pair_count; ++i) {
    out_packets[i] = split.continuation_values[i];
  }
  return iree_ok_status();
}

static bool loom_amdgpu_fp8_packed_bf16_combined_non_normal_threshold(
    const loom_amdgpu_fp8_decode_plan_t* plan,
    loom_amdgpu_fp8_packed_u16_repairs_t top_exponent_repairs,
    uint32_t* out_threshold) {
  *out_threshold = 0;
  if (!loom_amdgpu_fp8_decode_plan_has_combined_non_normal_condition(plan)) {
    return false;
  }
  if (plan->format.special_policy == LOOM_SCALAR_TYPE_FP8_SPECIAL_POLICY_IEEE) {
    if (top_exponent_repairs != (LOOM_AMDGPU_FP8_PACKED_U16_REPAIR_INF |
                                 LOOM_AMDGPU_FP8_PACKED_U16_REPAIR_NAN)) {
      return false;
    }
    *out_threshold = loom_amdgpu_fp8_decode_top_exponent_no_sign(&plan->format);
    return true;
  }

  IREE_ASSERT_EQ(plan->format.special_policy,
                 LOOM_SCALAR_TYPE_FP8_SPECIAL_POLICY_FINITE_NAN);
  *out_threshold = loom_amdgpu_fp8_decode_finite_nan_no_sign(&plan->format);
  return true;
}

static iree_status_t loom_amdgpu_emit_fp8_packed_bf16_combined_repair_condition(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fp8_decode_plan_t* plan,
    const loom_amdgpu_fp8_packed_u16_pair_state_t* pair_states,
    iree_host_size_t pair_count,
    loom_amdgpu_fp8_packed_u16_repairs_t top_exponent_repairs,
    loom_type_t vgpr_type, loom_type_t sgpr_type,
    loom_value_id_t* top_exponent_conditions,
    bool* out_top_exponent_conditions_available,
    loom_value_id_t* out_condition) {
  *out_condition = LOOM_VALUE_ID_INVALID;
  *out_top_exponent_conditions_available = false;

  loom_value_id_t low_combined_repair_condition = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_emit_fp8_packed_u16_combined_subnormal_condition(
          context, source_op, plan, pair_states, pair_count, vgpr_type,
          sgpr_type, &low_combined_repair_condition));
  loom_value_id_t low_combined_special_condition = LOOM_VALUE_ID_INVALID;
  if (plan->format.special_policy == LOOM_SCALAR_TYPE_FP8_SPECIAL_POLICY_IEEE) {
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_emit_fp8_packed_bf16_combined_top_exponent_condition(
            context, source_op, plan, pair_states, pair_count,
            top_exponent_repairs, vgpr_type, sgpr_type, top_exponent_conditions,
            out_top_exponent_conditions_available,
            &low_combined_special_condition));
  } else {
    IREE_ASSERT_EQ(plan->format.special_policy,
                   LOOM_SCALAR_TYPE_FP8_SPECIAL_POLICY_FINITE_NAN);
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_emit_fp8_packed_bf16_combined_finite_nan_condition(
            context, source_op, plan, pair_states, pair_count, vgpr_type,
            sgpr_type, &low_combined_special_condition));
  }
  return loom_amdgpu_emit_vgpr_binary(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_OR_B32,
      low_combined_repair_condition, low_combined_special_condition, vgpr_type,
      out_condition);
}

static iree_status_t loom_amdgpu_emit_fp8_packed_bf16_special_group_if(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fp8_decode_plan_t* plan,
    const loom_amdgpu_fp8_packed_u16_pair_state_t* pair_states,
    iree_host_size_t pair_count,
    loom_amdgpu_fp8_packed_u16_repairs_t top_exponent_repairs,
    loom_type_t vgpr_type, loom_type_t sgpr_type, loom_type_t mask_type,
    loom_value_id_t low_zero_sgpr64, loom_value_id_t* out_packets) {
  loom_value_id_t low_combined_condition = LOOM_VALUE_ID_INVALID;
  loom_value_id_t
      top_exponent_conditions[LOOM_AMDGPU_MAX_PACKED_32BIT_REGISTERS] = {0};
  bool has_hot_top_exponent_conditions = false;
  uint32_t non_normal_upper_threshold = 0;
  loom_amdgpu_fp8_packed_bf16_non_normal_markers_t non_normal_markers = {0};
  bool has_non_normal_markers = false;
  if (loom_amdgpu_fp8_packed_bf16_combined_non_normal_threshold(
          plan, top_exponent_repairs, &non_normal_upper_threshold)) {
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_emit_fp8_packed_bf16_combined_non_normal_condition(
            context, source_op, plan, pair_states, pair_count, vgpr_type,
            sgpr_type, non_normal_upper_threshold, &non_normal_markers,
            &low_combined_condition));
    has_non_normal_markers = true;
  } else {
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_emit_fp8_packed_bf16_combined_repair_condition(
            context, source_op, plan, pair_states, pair_count,
            top_exponent_repairs, vgpr_type, sgpr_type, top_exponent_conditions,
            &has_hot_top_exponent_conditions, &low_combined_condition));
  }

  const bool use_ieee_top_exponent_repair =
      plan->format.special_policy == LOOM_SCALAR_TYPE_FP8_SPECIAL_POLICY_IEEE;
  loom_amdgpu_fp8_packed_u16_repair_split_t split = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp8_packed_u16_repair_split(
      context, source_op, plan, low_combined_condition, pair_count, vgpr_type,
      mask_type, low_zero_sgpr64, &split));
  loom_builder_t* builder = loom_low_lower_context_builder(context);

  loom_value_id_t repair_payloads[LOOM_AMDGPU_MAX_PACKED_32BIT_REGISTERS] = {0};
  loom_value_id_t
      subnormal_repaired_packets[LOOM_AMDGPU_MAX_PACKED_32BIT_REGISTERS] = {0};
  loom_value_id_t repair_packets[LOOM_AMDGPU_MAX_PACKED_32BIT_REGISTERS] = {0};
  loom_builder_set_block(builder, split.repair_block);
  loom_amdgpu_fp8_packed_u16_repair_state_t repair_state = {
      .low_mask_shift = LOOM_VALUE_ID_INVALID,
  };
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_emit_fp8_packed_u16_subnormal_repair_payloads(
          context, source_op, plan, pair_states, pair_count,
          plan->subnormal_bf16_byte_table_words, vgpr_type, sgpr_type,
          &repair_state,
          has_non_normal_markers ? non_normal_markers.shifted_pairs : NULL,
          repair_payloads));
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp8_packed_u16_pair_signs(
      context, source_op, plan, pair_states, repair_payloads, pair_count,
      vgpr_type, subnormal_repaired_packets));
  if (use_ieee_top_exponent_repair) {
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_emit_fp8_packed_bf16_top_exponent_repair_packets(
            context, source_op, plan, pair_states, subnormal_repaired_packets,
            pair_count, top_exponent_repairs, has_hot_top_exponent_conditions,
            top_exponent_conditions, vgpr_type, sgpr_type, &repair_state,
            repair_packets));
  } else {
    loom_value_id_t nan_conditions[LOOM_AMDGPU_MAX_PACKED_32BIT_REGISTERS] = {
        0};
    bool has_hot_nan_conditions = false;
    if (has_non_normal_markers) {
      loom_value_id_t low_nan_ge_bias = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
          context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32,
          loom_amdgpu_fp8_decode_packed_ge_bias(
              non_normal_markers.upper_shifted_threshold),
          sgpr_type, &low_nan_ge_bias));
      for (iree_host_size_t i = 0; i < pair_count; ++i) {
        IREE_RETURN_IF_ERROR(
            loom_amdgpu_emit_fp8_packed_u16_pair_condition_with_bias(
                context, source_op, plan, non_normal_markers.shifted_pairs[i],
                low_nan_ge_bias, vgpr_type, &nan_conditions[i]));
      }
      has_hot_nan_conditions = true;
    }
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp8_packed_bf16_nan_repair_packets(
        context, source_op, plan, pair_states, subnormal_repaired_packets,
        pair_count, has_hot_nan_conditions, nan_conditions, vgpr_type,
        sgpr_type, &repair_state, repair_packets));
  }
  loom_op_t* repair_branch_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_low_br_build(builder, split.continuation_block, repair_packets,
                        pair_count, source_op->location, &repair_branch_op));

  loom_value_id_t normal_payloads[LOOM_AMDGPU_MAX_PACKED_32BIT_REGISTERS] = {0};
  loom_value_id_t normal_packets[LOOM_AMDGPU_MAX_PACKED_32BIT_REGISTERS] = {0};
  loom_builder_set_block(builder, split.normal_block);
  for (iree_host_size_t i = 0; i < pair_count; ++i) {
    normal_payloads[i] = pair_states[i].normal_payload;
  }
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp8_packed_u16_pair_signs(
      context, source_op, plan, pair_states, normal_payloads, pair_count,
      vgpr_type, normal_packets));
  loom_op_t* normal_branch_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_low_br_build(builder, split.continuation_block, normal_packets,
                        pair_count, source_op->location, &normal_branch_op));

  loom_builder_set_block(builder, split.continuation_block);
  for (iree_host_size_t i = 0; i < pair_count; ++i) {
    out_packets[i] = split.continuation_values[i];
  }
  return iree_ok_status();
}

iree_status_t loom_amdgpu_emit_fp8_pairs_to_packed_bf16(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fp8_decode_plan_t* plan,
    const loom_amdgpu_fp8_packed_u16_pair_source_t* pair_sources,
    iree_host_size_t pair_count,
    loom_amdgpu_fp8_decode_value_flags_t value_flags, loom_type_t vgpr_type,
    loom_type_t sgpr_type, loom_type_t mask_type,
    loom_value_id_t* out_low_packets) {
  IREE_ASSERT_GT(pair_count, 0u);
  IREE_ASSERT_LE(pair_count, LOOM_AMDGPU_MAX_PACKED_32BIT_REGISTERS);
  IREE_ASSERT(loom_amdgpu_can_emit_fp8_pair_to_packed_bf16(plan, value_flags));
  for (iree_host_size_t i = 0; i < pair_count; ++i) {
    out_low_packets[i] = LOOM_VALUE_ID_INVALID;
  }

  if (loom_amdgpu_fp8_selects_exact_bf16_via_f16(plan, value_flags)) {
    return loom_amdgpu_emit_fp8_pairs_to_packed_bf16_via_f16(
        context, source_op, plan, pair_sources, pair_count, vgpr_type,
        sgpr_type, out_low_packets);
  }

  const uint32_t exponent_bias = (127u - plan->format.exponent_bias) << 7;
  const uint32_t packed_exponent_bias = exponent_bias | (exponent_bias << 16);
  const bool can_use_normal_path =
      loom_amdgpu_can_emit_fp8_pair_to_packed_u16_finite_path(
          plan, value_flags,
          LOOM_AMDGPU_FP8_DECODE_PLAN_CAPABILITY_PACKED_NORMAL_BF16_PAYLOAD);

  const bool value_non_zero =
      iree_any_bit_set(value_flags, LOOM_AMDGPU_FP8_DECODE_VALUE_FLAG_NON_ZERO);
  const bool value_not_subnormal = iree_any_bit_set(
      value_flags, LOOM_AMDGPU_FP8_DECODE_VALUE_FLAG_NOT_SUBNORMAL);
  const bool value_not_nan =
      iree_any_bit_set(value_flags, LOOM_AMDGPU_FP8_DECODE_VALUE_FLAG_NOT_NAN);
  const bool can_use_zero_repaired_normal_payload =
      value_non_zero ||
      loom_amdgpu_fp8_decode_plan_has_packed_zero_repair(plan);
  const bool use_exact_repair = !can_use_normal_path;
  const bool needs_repair_condition =
      use_exact_repair &&
      !(value_not_subnormal && can_use_zero_repaired_normal_payload);
  const bool use_repair_split =
      needs_repair_condition &&
      loom_amdgpu_fp8_decode_plan_has_mask_repair_split(plan);
  const bool needs_nan_repair = use_exact_repair && !value_not_nan;
  const bool value_not_inf =
      iree_any_bit_set(value_flags, LOOM_AMDGPU_FP8_DECODE_VALUE_FLAG_NOT_INF);
  const bool needs_inf_repair =
      use_exact_repair &&
      plan->format.special_policy == LOOM_SCALAR_TYPE_FP8_SPECIAL_POLICY_IEEE &&
      !value_not_inf;
  const loom_amdgpu_fp8_packed_u16_repairs_t top_exponent_repairs =
      (needs_nan_repair ? LOOM_AMDGPU_FP8_PACKED_U16_REPAIR_NAN
                        : LOOM_AMDGPU_FP8_PACKED_U16_REPAIR_NONE) |
      (needs_inf_repair ? LOOM_AMDGPU_FP8_PACKED_U16_REPAIR_INF
                        : LOOM_AMDGPU_FP8_PACKED_U16_REPAIR_NONE);
  const bool use_nan_repair_split =
      needs_nan_repair && pair_count > 1 &&
      loom_amdgpu_fp8_decode_plan_has_mask_repair_split(plan);
  const bool use_top_exponent_repair_split =
      plan->format.special_policy == LOOM_SCALAR_TYPE_FP8_SPECIAL_POLICY_IEEE &&
      top_exponent_repairs != LOOM_AMDGPU_FP8_PACKED_U16_REPAIR_NONE &&
      pair_count > 1 && loom_amdgpu_fp8_decode_plan_has_mask_repair_split(plan);
  const bool use_combined_ieee_special_repair_split =
      use_repair_split && use_top_exponent_repair_split &&
      plan->format.special_policy == LOOM_SCALAR_TYPE_FP8_SPECIAL_POLICY_IEEE;
  const bool use_combined_finite_nan_special_repair_split =
      use_repair_split && use_nan_repair_split &&
      plan->format.special_policy ==
          LOOM_SCALAR_TYPE_FP8_SPECIAL_POLICY_FINITE_NAN;
  const bool use_combined_special_repair_split =
      use_combined_ieee_special_repair_split ||
      use_combined_finite_nan_special_repair_split;
  const bool needs_branchless_zero_repair =
      !value_non_zero && (!use_exact_repair || !needs_repair_condition);
  const bool needs_sgpr64_zero_compare =
      (use_repair_split || use_nan_repair_split ||
       use_top_exponent_repair_split) &&
      !loom_amdgpu_fp8_decode_plan_has_inline_sgpr64_zero_compare(plan);

  loom_value_id_t low_one_mask = LOOM_VALUE_ID_INVALID;
  if (needs_branchless_zero_repair) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32,
        LOOM_AMDGPU_FP8_DECODE_U16_PAIR_ONE_MASK, sgpr_type, &low_one_mask));
  }
  loom_value_id_t low_zero_sgpr64 = LOOM_VALUE_ID_INVALID;
  if (needs_sgpr64_zero_compare) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_sgpr64_constant_u64(
        context, source_op, 0, &low_zero_sgpr64));
  }

  loom_amdgpu_fp8_packed_u16_pair_state_t
      pair_states[LOOM_AMDGPU_MAX_PACKED_32BIT_REGISTERS] = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp8_packed_u16_pair_states(
      context, source_op, plan, pair_sources, pair_count,
      7u - plan->format.mantissa_bits, packed_exponent_bias, vgpr_type,
      sgpr_type, pair_states));

  loom_value_id_t finite_payloads[LOOM_AMDGPU_MAX_PACKED_32BIT_REGISTERS] = {0};
  if (use_exact_repair) {
    if (use_combined_special_repair_split) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp8_packed_bf16_special_group_if(
          context, source_op, plan, pair_states, pair_count,
          use_combined_ieee_special_repair_split
              ? top_exponent_repairs
              : LOOM_AMDGPU_FP8_PACKED_U16_REPAIR_NONE,
          vgpr_type, sgpr_type, mask_type, low_zero_sgpr64, out_low_packets));
      return iree_ok_status();
    } else if (!needs_repair_condition) {
      for (iree_host_size_t i = 0; i < pair_count; ++i) {
        IREE_RETURN_IF_ERROR(
            loom_amdgpu_emit_fp8_packed_u16_zero_repaired_payload(
                context, source_op, plan, pair_states[i].base.source_no_sign,
                pair_states[i].normal_payload, value_flags, low_one_mask,
                vgpr_type, &finite_payloads[i]));
      }
    } else if (use_repair_split) {
      loom_value_id_t low_combined_repair_condition = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_emit_fp8_packed_u16_combined_subnormal_condition(
              context, source_op, plan, pair_states, pair_count, vgpr_type,
              sgpr_type, &low_combined_repair_condition));
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_emit_fp8_packed_u16_subnormal_repair_group_if(
              context, source_op, plan, pair_states, pair_count,
              low_combined_repair_condition,
              plan->subnormal_bf16_byte_table_words, vgpr_type, sgpr_type,
              mask_type, low_zero_sgpr64, finite_payloads));
    } else {
      loom_amdgpu_fp8_packed_u16_repair_state_t repair_state = {
          .low_mask_shift = LOOM_VALUE_ID_INVALID,
      };
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_emit_fp8_packed_u16_subnormal_repair_payloads(
              context, source_op, plan, pair_states, pair_count,
              plan->subnormal_bf16_byte_table_words, vgpr_type, sgpr_type,
              &repair_state, /*low_subnormal_markers=*/NULL, finite_payloads));
    }
  } else if (!value_non_zero) {
    for (iree_host_size_t i = 0; i < pair_count; ++i) {
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_emit_fp8_packed_u16_zero_repaired_payload(
              context, source_op, plan, pair_states[i].base.source_no_sign,
              pair_states[i].normal_payload, value_flags, low_one_mask,
              vgpr_type, &finite_payloads[i]));
    }
  } else {
    for (iree_host_size_t i = 0; i < pair_count; ++i) {
      finite_payloads[i] = pair_states[i].normal_payload;
    }
  }

  loom_value_id_t no_nan_packets[LOOM_AMDGPU_MAX_PACKED_32BIT_REGISTERS] = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp8_packed_u16_pair_signs(
      context, source_op, plan, pair_states, finite_payloads, pair_count,
      vgpr_type, no_nan_packets));
  if (use_top_exponent_repair_split) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp8_packed_bf16_top_exponent_group_if(
        context, source_op, plan, pair_states, no_nan_packets, pair_count,
        top_exponent_repairs, vgpr_type, sgpr_type, mask_type, low_zero_sgpr64,
        out_low_packets));
    return iree_ok_status();
  }
  loom_value_id_t no_inf_packets[LOOM_AMDGPU_MAX_PACKED_32BIT_REGISTERS] = {0};
  if (needs_inf_repair) {
    loom_amdgpu_fp8_packed_u16_repair_state_t repair_state = {
        .low_mask_shift = LOOM_VALUE_ID_INVALID,
    };
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp8_packed_bf16_inf_repair_packets(
        context, source_op, plan, pair_states, no_nan_packets, pair_count,
        vgpr_type, sgpr_type, &repair_state, no_inf_packets));
  } else {
    for (iree_host_size_t i = 0; i < pair_count; ++i) {
      no_inf_packets[i] = no_nan_packets[i];
    }
  }
  if (use_nan_repair_split) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp8_packed_bf16_nan_group_if(
        context, source_op, plan, pair_states, no_inf_packets, pair_count,
        vgpr_type, sgpr_type, mask_type, low_zero_sgpr64, out_low_packets));
  } else if (needs_nan_repair) {
    loom_value_id_t nan_conditions[LOOM_AMDGPU_MAX_PACKED_32BIT_REGISTERS] = {
        0};
    loom_amdgpu_fp8_packed_u16_repair_state_t repair_state = {
        .low_mask_shift = LOOM_VALUE_ID_INVALID,
    };
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp8_packed_bf16_nan_repair_packets(
        context, source_op, plan, pair_states, no_inf_packets, pair_count,
        /*has_hot_nan_conditions=*/false, nan_conditions, vgpr_type, sgpr_type,
        &repair_state, out_low_packets));
  } else {
    for (iree_host_size_t i = 0; i < pair_count; ++i) {
      out_low_packets[i] = no_inf_packets[i];
    }
  }
  return iree_ok_status();
}
