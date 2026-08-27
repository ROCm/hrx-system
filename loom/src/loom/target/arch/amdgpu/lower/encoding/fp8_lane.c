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
  LOOM_AMDGPU_FP8_DECODE_BF16_SIGN_INSERT_MASK = 0x00008000u,
  LOOM_AMDGPU_FP8_DECODE_F32_SIGN_INSERT_MASK = 0x80000000u,
};

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

static iree_status_t loom_amdgpu_emit_fp8_apply_special_values(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fp8_decode_plan_t* plan, loom_value_id_t low_finite_bits,
    loom_value_id_t low_sign_bits, loom_value_id_t low_source_no_sign,
    loom_value_id_t low_source_shifted,
    loom_amdgpu_fp8_decode_value_flags_t value_flags, uint32_t quiet_nan_bits,
    uint32_t infinity_magnitude_bits, loom_type_t vgpr_type,
    loom_type_t mask_type, loom_value_id_t* out_lane) {
  const loom_scalar_type_fp8_format_t* format = &plan->format;
  const bool value_not_nan =
      iree_any_bit_set(value_flags, LOOM_AMDGPU_FP8_DECODE_VALUE_FLAG_NOT_NAN);
  const bool value_not_inf =
      iree_any_bit_set(value_flags, LOOM_AMDGPU_FP8_DECODE_VALUE_FLAG_NOT_INF);

  if (value_not_nan && value_not_inf) {
    *out_lane = low_finite_bits;
    return iree_ok_status();
  }

  if (format->special_policy ==
      LOOM_SCALAR_TYPE_FP8_SPECIAL_POLICY_FINITE_NAN_UNSIGNED_ZERO) {
    if (value_not_nan) {
      *out_lane = low_finite_bits;
      return iree_ok_status();
    }

    loom_value_id_t low_quiet_nan = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32,
        quiet_nan_bits, vgpr_type, &low_quiet_nan));
    loom_value_id_t low_source_byte = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary_immediate(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT,
        low_source_shifted, UINT32_C(0xFF), vgpr_type, &low_source_byte));
    loom_value_id_t low_is_nan = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_resolved_vgpr_compare_immediate(
        context, source_op, &plan->compare_eq_i32_src1_inline_descriptor,
        LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_EQ_I32, low_source_byte,
        UINT32_C(0x80), vgpr_type, mask_type, &low_is_nan));
    return loom_amdgpu_emit_vgpr_select(context, source_op, low_finite_bits,
                                        low_quiet_nan, low_is_nan, vgpr_type,
                                        out_lane);
  }

  if (format->special_policy ==
      LOOM_SCALAR_TYPE_FP8_SPECIAL_POLICY_FINITE_NAN) {
    if (value_not_nan) {
      *out_lane = low_finite_bits;
      return iree_ok_status();
    }

    loom_value_id_t low_quiet_nan = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32,
        quiet_nan_bits, vgpr_type, &low_quiet_nan));
    loom_value_id_t low_is_nan = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_resolved_vgpr_compare_immediate(
        context, source_op, &plan->compare_eq_i32_src1_inline_descriptor,
        LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_EQ_I32, low_source_no_sign,
        loom_amdgpu_fp8_decode_finite_nan_no_sign(format), vgpr_type, mask_type,
        &low_is_nan));
    return loom_amdgpu_emit_vgpr_select(context, source_op, low_finite_bits,
                                        low_quiet_nan, low_is_nan, vgpr_type,
                                        out_lane);
  }

  const uint32_t top_exponent_no_sign =
      loom_amdgpu_fp8_decode_top_exponent_no_sign(format);
  if (value_not_inf) {
    loom_value_id_t low_quiet_nan = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32,
        quiet_nan_bits, vgpr_type, &low_quiet_nan));
    loom_value_id_t low_is_top_exponent = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_resolved_vgpr_compare_immediate(
        context, source_op, &plan->compare_uge_u32_src1_inline_descriptor,
        LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_UGE_U32, low_source_no_sign,
        top_exponent_no_sign, vgpr_type, mask_type, &low_is_top_exponent));
    return loom_amdgpu_emit_vgpr_select(context, source_op, low_finite_bits,
                                        low_quiet_nan, low_is_top_exponent,
                                        vgpr_type, out_lane);
  }

  if (value_not_nan) {
    loom_value_id_t low_infinity_bits = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary_immediate(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_OR_B32_LIT,
        low_sign_bits, infinity_magnitude_bits, vgpr_type, &low_infinity_bits));
    loom_value_id_t low_is_infinity = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_resolved_vgpr_compare_immediate(
        context, source_op, &plan->compare_eq_i32_src1_inline_descriptor,
        LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_EQ_I32, low_source_no_sign,
        top_exponent_no_sign, vgpr_type, mask_type, &low_is_infinity));
    return loom_amdgpu_emit_vgpr_select(context, source_op, low_finite_bits,
                                        low_infinity_bits, low_is_infinity,
                                        vgpr_type, out_lane);
  }

  loom_value_id_t low_quiet_nan = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32, quiet_nan_bits,
      vgpr_type, &low_quiet_nan));
  loom_value_id_t low_infinity_bits = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary_immediate(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_OR_B32_LIT,
      low_sign_bits, infinity_magnitude_bits, vgpr_type, &low_infinity_bits));
  loom_value_id_t low_is_infinity = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_resolved_vgpr_compare_immediate(
      context, source_op, &plan->compare_eq_i32_src1_inline_descriptor,
      LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_EQ_I32, low_source_no_sign,
      top_exponent_no_sign, vgpr_type, mask_type, &low_is_infinity));
  loom_value_id_t low_top_bits = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_select(
      context, source_op, low_quiet_nan, low_infinity_bits, low_is_infinity,
      vgpr_type, &low_top_bits));
  loom_value_id_t low_is_top_exponent = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_resolved_vgpr_compare_immediate(
      context, source_op, &plan->compare_uge_u32_src1_inline_descriptor,
      LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_UGE_U32, low_source_no_sign,
      top_exponent_no_sign, vgpr_type, mask_type, &low_is_top_exponent));
  return loom_amdgpu_emit_vgpr_select(context, source_op, low_finite_bits,
                                      low_top_bits, low_is_top_exponent,
                                      vgpr_type, out_lane);
}

static iree_status_t loom_amdgpu_emit_fp8_subnormal_bf16_bits(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fp8_decode_plan_t* plan,
    loom_value_id_t low_source_no_sign, loom_type_t vgpr_type,
    loom_type_t mask_type, loom_value_id_t* out_bf16_bits) {
  *out_bf16_bits = LOOM_VALUE_ID_INVALID;
  const loom_scalar_type_fp8_format_t* format = &plan->format;
  loom_value_id_t low_leading_index = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32, 0, vgpr_type,
      &low_leading_index));
  for (uint8_t i = 1; i < format->mantissa_bits; ++i) {
    loom_value_id_t low_has_leading_index = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_resolved_vgpr_compare_immediate(
        context, source_op, &plan->compare_uge_u32_src1_inline_descriptor,
        LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_UGE_U32, low_source_no_sign,
        UINT32_C(1) << i, vgpr_type, mask_type, &low_has_leading_index));
    loom_value_id_t low_candidate = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32, i, vgpr_type,
        &low_candidate));
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_select(
        context, source_op, low_leading_index, low_candidate,
        low_has_leading_index, vgpr_type, &low_leading_index));
  }

  loom_value_id_t low_exponent_bits = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary_immediate(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_ADD_U32_LIT,
      low_leading_index, 128u - format->exponent_bias - format->mantissa_bits,
      vgpr_type, &low_exponent_bits));
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_shift(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32_LIT, 7,
      low_exponent_bits, vgpr_type, &low_exponent_bits));

  loom_value_id_t low_shift_base = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32, 7, vgpr_type,
      &low_shift_base));
  loom_value_id_t low_fraction_shift = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_SUB_U32, low_shift_base,
      low_leading_index, vgpr_type, &low_fraction_shift));
  loom_value_id_t low_fraction_bits = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32,
      low_fraction_shift, low_source_no_sign, vgpr_type, &low_fraction_bits));
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary_immediate(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT,
      low_fraction_bits, UINT32_C(0x7F), vgpr_type, &low_fraction_bits));

  loom_value_id_t low_nonzero_bits = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_OR_B32,
      low_exponent_bits, low_fraction_bits, vgpr_type, &low_nonzero_bits));
  loom_value_id_t low_zero_bits = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32, 0, vgpr_type,
      &low_zero_bits));
  loom_value_id_t low_source_is_zero = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_resolved_vgpr_compare_immediate(
      context, source_op, &plan->compare_eq_i32_src1_inline_descriptor,
      LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_EQ_I32, low_source_no_sign, 0, vgpr_type,
      mask_type, &low_source_is_zero));
  return loom_amdgpu_emit_vgpr_select(context, source_op, low_nonzero_bits,
                                      low_zero_bits, low_source_is_zero,
                                      vgpr_type, out_bf16_bits);
}

static iree_status_t loom_amdgpu_emit_fp8_decode_lshl_add_u32_src2_literal(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fp8_decode_plan_t* plan, loom_value_id_t value,
    uint32_t shift, uint32_t addend, loom_type_t vgpr_type,
    loom_value_id_t* out_value) {
  *out_value = LOOM_VALUE_ID_INVALID;
  loom_named_attr_t attrs[2] = {0};
  iree_host_size_t attr_count = 0;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_append_i64_attr(context, IREE_SV("shift"), shift, attrs,
                                  IREE_ARRAYSIZE(attrs), &attr_count));
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_append_i64_attr(context, IREE_SV("imm32"), addend, attrs,
                                  IREE_ARRAYSIZE(attrs), &attr_count));
  const loom_value_id_t operands[] = {value};
  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
      context, &plan->lshl_add_u32_shift_imm_src2_literal_descriptor, operands,
      IREE_ARRAYSIZE(operands), loom_make_named_attr_slice(attrs, attr_count),
      &vgpr_type, 1, /*tied_results=*/NULL, /*tied_result_count=*/0,
      source_op->location, &low_op));
  *out_value = loom_value_slice_get(loom_low_op_results(low_op), 0);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_fp8_decode_lshl_add_u32_materialized(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fp8_decode_plan_t* plan, loom_value_id_t value,
    uint32_t shift, uint32_t addend,
    loom_amdgpu_descriptor_ref_t addend_descriptor_ref, loom_type_t addend_type,
    loom_type_t vgpr_type, loom_value_id_t* out_value) {
  *out_value = LOOM_VALUE_ID_INVALID;
  loom_value_id_t low_addend = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(context, source_op,
                                                  addend_descriptor_ref, addend,
                                                  addend_type, &low_addend));
  return loom_amdgpu_emit_resolved_vgpr_lshl_add_u32(
      context, source_op, &plan->lshl_add_u32_shift_imm_descriptor, value,
      low_addend, shift, vgpr_type, out_value);
}

static iree_status_t loom_amdgpu_emit_fp8_normal_payload(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fp8_decode_plan_t* plan,
    loom_value_id_t low_source_no_sign, uint32_t payload_shift,
    uint32_t exponent_bias, loom_type_t vgpr_type,
    loom_value_id_t* out_payload) {
  *out_payload = LOOM_VALUE_ID_INVALID;
  if (iree_any_bit_set(
          plan->flags,
          LOOM_AMDGPU_FP8_DECODE_PLAN_FLAG_HAS_LSHL_ADD_U32_SHIFT_IMM_SRC2_LITERAL)) {
    return loom_amdgpu_emit_fp8_decode_lshl_add_u32_src2_literal(
        context, source_op, plan, low_source_no_sign, payload_shift,
        exponent_bias, vgpr_type, out_payload);
  }
  if (iree_any_bit_set(
          plan->flags,
          LOOM_AMDGPU_FP8_DECODE_PLAN_FLAG_HAS_LSHL_ADD_U32_SHIFT_IMM)) {
    return loom_amdgpu_emit_fp8_decode_lshl_add_u32_materialized(
        context, source_op, plan, low_source_no_sign, payload_shift,
        exponent_bias, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32, vgpr_type,
        vgpr_type, out_payload);
  }

  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_shift(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32_LIT,
      payload_shift, low_source_no_sign, vgpr_type, out_payload));
  return loom_amdgpu_emit_vgpr_binary_immediate(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_ADD_U32_LIT,
      *out_payload, exponent_bias, vgpr_type, out_payload);
}

// Merges byte-disjoint low/high FP8 table payloads after shifting the high
// payload into byte 1. The add form is equivalent to OR for these payloads.
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

static iree_status_t loom_amdgpu_emit_fp8_subnormal_bf16_bits_permute(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fp8_decode_plan_t* plan,
    loom_value_id_t low_source_no_sign, loom_type_t vgpr_type,
    loom_type_t sgpr_type, loom_value_id_t* out_bf16_bits) {
  *out_bf16_bits = LOOM_VALUE_ID_INVALID;
  if (!iree_any_bit_set(plan->flags,
                        LOOM_AMDGPU_FP8_DECODE_PLAN_FLAG_HAS_PERM_B32) ||
      plan->format.mantissa_bits != 2) {
    return iree_ok_status();
  }

  loom_value_id_t low_byte_offset = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_shift(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32_LIT, 1,
      low_source_no_sign, vgpr_type, &low_byte_offset));
  loom_value_id_t low_high_byte_offset = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary_immediate(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_ADD_U32_LIT,
      low_byte_offset, 1, vgpr_type, &low_high_byte_offset));
  loom_value_id_t low_selector = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp8_decode_merge_low_high_bytes(
      context, source_op, plan, low_byte_offset, low_high_byte_offset,
      vgpr_type, &low_selector));

  loom_value_id_t low_table_lo = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32,
      plan->subnormal_bf16_table_words[0], sgpr_type, &low_table_lo));
  loom_value_id_t low_table_hi = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32,
      plan->subnormal_bf16_table_words[1], sgpr_type, &low_table_hi));

  loom_value_id_t low_unsigned_bits = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_resolved_vgpr_ternary(
      context, source_op, &plan->perm_b32_descriptor, low_table_hi,
      low_table_lo, low_selector, vgpr_type, &low_unsigned_bits));

  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary_immediate(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT,
      low_unsigned_bits, UINT32_C(0xFFFF), vgpr_type, &low_unsigned_bits));
  *out_bf16_bits = low_unsigned_bits;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_fp8_subnormal_bf16_bits_byte_tables(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fp8_decode_plan_t* plan,
    loom_value_id_t low_source_no_sign, loom_type_t vgpr_type,
    loom_type_t sgpr_type, loom_value_id_t* out_bf16_bits) {
  *out_bf16_bits = LOOM_VALUE_ID_INVALID;
  if (!iree_any_bit_set(plan->flags,
                        LOOM_AMDGPU_FP8_DECODE_PLAN_FLAG_HAS_PERM_B32) ||
      plan->format.mantissa_bits != 3) {
    return iree_ok_status();
  }

  loom_value_id_t low_payload_bytes[LOOM_AMDGPU_FP8_BF16_BYTE_COUNT] = {0};
  for (uint32_t byte_index = 0; byte_index < LOOM_AMDGPU_FP8_BF16_BYTE_COUNT;
       ++byte_index) {
    loom_value_id_t low_table_lo = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32,
        plan->subnormal_bf16_byte_table_words[byte_index][0], sgpr_type,
        &low_table_lo));
    loom_value_id_t low_table_hi = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32,
        plan->subnormal_bf16_byte_table_words[byte_index][1], sgpr_type,
        &low_table_hi));

    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_resolved_vgpr_ternary(
        context, source_op, &plan->perm_b32_descriptor, low_table_hi,
        low_table_lo, low_source_no_sign, vgpr_type,
        &low_payload_bytes[byte_index]));
  }

  loom_value_id_t low_unsigned_bits = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp8_decode_merge_low_high_bytes(
      context, source_op, plan, low_payload_bytes[0], low_payload_bytes[1],
      vgpr_type, &low_unsigned_bits));
  *out_bf16_bits = low_unsigned_bits;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_fp8_normal_or_zero_lane_payload(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fp8_decode_plan_t* plan,
    loom_value_id_t low_source_no_sign, loom_value_id_t low_normal_payload,
    loom_amdgpu_fp8_decode_value_flags_t value_flags, loom_type_t vgpr_type,
    loom_type_t mask_type, loom_value_id_t* out_payload) {
  *out_payload = LOOM_VALUE_ID_INVALID;
  if (iree_any_bit_set(value_flags,
                       LOOM_AMDGPU_FP8_DECODE_VALUE_FLAG_NON_ZERO)) {
    *out_payload = low_normal_payload;
    return iree_ok_status();
  }

  loom_value_id_t low_zero_payload = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32, 0, vgpr_type,
      &low_zero_payload));
  loom_value_id_t low_source_is_zero = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_resolved_vgpr_compare_immediate(
      context, source_op, &plan->compare_eq_i32_src1_inline_descriptor,
      LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_EQ_I32, low_source_no_sign, 0, vgpr_type,
      mask_type, &low_source_is_zero));
  return loom_amdgpu_emit_vgpr_select(context, source_op, low_normal_payload,
                                      low_zero_payload, low_source_is_zero,
                                      vgpr_type, out_payload);
}

typedef struct loom_amdgpu_fp8_lane_result_format_t {
  // Bit shift that moves the FP8 mantissa into the destination payload.
  uint32_t payload_shift;
  // Bit shift that moves the exponent-bias delta into the destination payload.
  uint32_t exponent_bias_shift;
  // Bit shift that moves the FP8 sign bit into the destination sign position.
  uint32_t sign_shift;
  // BFI mask that selects the destination sign bit.
  uint32_t sign_insert_mask;
  // Destination quiet-NaN payload bits.
  uint32_t quiet_nan_bits;
  // Destination infinity magnitude bits.
  uint32_t infinity_magnitude_bits;
} loom_amdgpu_fp8_lane_result_format_t;

static const loom_amdgpu_fp8_lane_result_format_t
    kLoomAmdgpuFp8LaneBf16ResultFormat = {
        .payload_shift = 7,
        .exponent_bias_shift = 7,
        .sign_shift = 8,
        .sign_insert_mask = LOOM_AMDGPU_FP8_DECODE_BF16_SIGN_INSERT_MASK,
        .quiet_nan_bits = 0x7FC0,
        .infinity_magnitude_bits = 0x7F80,
};

static const loom_amdgpu_fp8_lane_result_format_t
    kLoomAmdgpuFp8LaneF32ResultFormat = {
        .payload_shift = 23,
        .exponent_bias_shift = 23,
        .sign_shift = 24,
        .sign_insert_mask = LOOM_AMDGPU_FP8_DECODE_F32_SIGN_INSERT_MASK,
        .quiet_nan_bits = 0x7FC00000,
        .infinity_magnitude_bits = 0x7F800000,
};

typedef struct loom_amdgpu_fp8_lane_bits_t {
  // Source payload with the FP8 sign bit cleared.
  loom_value_id_t low_source_no_sign;
  // FP8 sign bit shifted to the destination sign position.
  loom_value_id_t low_sign_bits;
  // Whether finite lanes can insert the sign with BFI instead of OR.
  bool use_finite_sign_insert;
} loom_amdgpu_fp8_lane_bits_t;

static iree_status_t loom_amdgpu_emit_fp8_lane_bits(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fp8_decode_plan_t* plan, loom_value_id_t low_byte,
    loom_amdgpu_fp8_decode_value_flags_t value_flags,
    const loom_amdgpu_fp8_lane_result_format_t* result_format,
    loom_type_t vgpr_type, loom_amdgpu_fp8_lane_bits_t* out_bits) {
  *out_bits = (loom_amdgpu_fp8_lane_bits_t){
      .low_source_no_sign = LOOM_VALUE_ID_INVALID,
      .low_sign_bits = LOOM_VALUE_ID_INVALID,
      .use_finite_sign_insert =
          loom_amdgpu_fp8_decode_value_is_finite(value_flags) &&
          iree_any_bit_set(
              plan->flags,
              LOOM_AMDGPU_FP8_DECODE_PLAN_FLAG_HAS_BFI_B32_SRC0_LITERAL),
  };

  if (!out_bits->use_finite_sign_insert) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary_immediate(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT, low_byte,
        UINT32_C(0x80), vgpr_type, &out_bits->low_sign_bits));
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_shift(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32_LIT,
        result_format->sign_shift, out_bits->low_sign_bits, vgpr_type,
        &out_bits->low_sign_bits));
  }

  return loom_amdgpu_emit_vgpr_binary_immediate(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT, low_byte,
      UINT32_C(0x7F), vgpr_type, &out_bits->low_source_no_sign);
}

static iree_status_t loom_amdgpu_emit_fp8_lane_normal_payload(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fp8_decode_plan_t* plan,
    const loom_amdgpu_fp8_lane_result_format_t* result_format,
    loom_value_id_t low_source_no_sign, loom_type_t vgpr_type,
    loom_value_id_t* out_payload) {
  return loom_amdgpu_emit_fp8_normal_payload(
      context, source_op, plan, low_source_no_sign,
      result_format->payload_shift - plan->format.mantissa_bits,
      (127u - plan->format.exponent_bias) << result_format->exponent_bias_shift,
      vgpr_type, out_payload);
}

static iree_status_t loom_amdgpu_emit_fp8_signed_lane_payload(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fp8_decode_plan_t* plan, loom_value_id_t low_byte,
    const loom_amdgpu_fp8_lane_bits_t* lane_bits,
    loom_value_id_t low_finite_payload,
    loom_amdgpu_fp8_decode_value_flags_t value_flags,
    const loom_amdgpu_fp8_lane_result_format_t* result_format,
    loom_type_t vgpr_type, loom_type_t mask_type, loom_value_id_t* out_lane) {
  if (lane_bits->use_finite_sign_insert) {
    loom_value_id_t low_shifted_sign_source = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_shift(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32_LIT,
        result_format->sign_shift, low_byte, vgpr_type,
        &low_shifted_sign_source));
    return loom_amdgpu_emit_resolved_vgpr_binary_immediate(
        context, source_op, &plan->bfi_b32_src0_literal_descriptor,
        low_shifted_sign_source, low_finite_payload,
        result_format->sign_insert_mask, vgpr_type, out_lane);
  }

  loom_value_id_t low_finite_bits = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_OR_B32,
      lane_bits->low_sign_bits, low_finite_payload, vgpr_type,
      &low_finite_bits));
  return loom_amdgpu_emit_fp8_apply_special_values(
      context, source_op, plan, low_finite_bits, lane_bits->low_sign_bits,
      lane_bits->low_source_no_sign, low_byte, value_flags,
      result_format->quiet_nan_bits, result_format->infinity_magnitude_bits,
      vgpr_type, mask_type, out_lane);
}

iree_status_t loom_amdgpu_emit_fp8_to_bf16_lane(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fp8_decode_plan_t* plan, loom_value_id_t low_byte,
    loom_amdgpu_fp8_decode_value_flags_t value_flags, loom_type_t vgpr_type,
    loom_type_t sgpr_type, loom_type_t mask_type, loom_value_id_t* out_lane) {
  *out_lane = LOOM_VALUE_ID_INVALID;
  const loom_scalar_type_fp8_format_t* format = &plan->format;
  const bool value_not_subnormal = iree_any_bit_set(
      value_flags, LOOM_AMDGPU_FP8_DECODE_VALUE_FLAG_NOT_SUBNORMAL);
  const bool value_non_zero =
      iree_any_bit_set(value_flags, LOOM_AMDGPU_FP8_DECODE_VALUE_FLAG_NON_ZERO);
  loom_amdgpu_fp8_lane_bits_t lane_bits;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp8_lane_bits(
      context, source_op, plan, low_byte, value_flags,
      &kLoomAmdgpuFp8LaneBf16ResultFormat, vgpr_type, &lane_bits));

  loom_value_id_t low_normal_payload = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp8_lane_normal_payload(
      context, source_op, plan, &kLoomAmdgpuFp8LaneBf16ResultFormat,
      lane_bits.low_source_no_sign, vgpr_type, &low_normal_payload));

  loom_value_id_t low_finite_payload = LOOM_VALUE_ID_INVALID;
  if (value_not_subnormal && value_non_zero) {
    low_finite_payload = low_normal_payload;
  } else if (value_not_subnormal) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp8_normal_or_zero_lane_payload(
        context, source_op, plan, lane_bits.low_source_no_sign,
        low_normal_payload, value_flags, vgpr_type, mask_type,
        &low_finite_payload));
  } else {
    loom_value_id_t low_subnormal_payload = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp8_subnormal_bf16_bits_permute(
        context, source_op, plan, lane_bits.low_source_no_sign, vgpr_type,
        sgpr_type, &low_subnormal_payload));
    if (low_subnormal_payload == LOOM_VALUE_ID_INVALID) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp8_subnormal_bf16_bits_byte_tables(
          context, source_op, plan, lane_bits.low_source_no_sign, vgpr_type,
          sgpr_type, &low_subnormal_payload));
    }
    if (low_subnormal_payload == LOOM_VALUE_ID_INVALID) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp8_subnormal_bf16_bits(
          context, source_op, plan, lane_bits.low_source_no_sign, vgpr_type,
          mask_type, &low_subnormal_payload));
    }
    loom_value_id_t low_exponent_is_zero = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_resolved_vgpr_compare_immediate(
        context, source_op, &plan->compare_ult_u32_src1_inline_descriptor,
        LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_ULT_U32, lane_bits.low_source_no_sign,
        UINT32_C(1) << format->mantissa_bits, vgpr_type, mask_type,
        &low_exponent_is_zero));
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_select(
        context, source_op, low_normal_payload, low_subnormal_payload,
        low_exponent_is_zero, vgpr_type, &low_finite_payload));
  }

  return loom_amdgpu_emit_fp8_signed_lane_payload(
      context, source_op, plan, low_byte, &lane_bits, low_finite_payload,
      value_flags, &kLoomAmdgpuFp8LaneBf16ResultFormat, vgpr_type, mask_type,
      out_lane);
}

iree_status_t loom_amdgpu_try_emit_fp8_not_subnormal_to_f32_lane(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fp8_decode_plan_t* plan, loom_value_id_t low_byte,
    loom_amdgpu_fp8_decode_value_flags_t value_flags, loom_type_t vgpr_type,
    loom_type_t mask_type, loom_value_id_t* out_lane) {
  *out_lane = LOOM_VALUE_ID_INVALID;
  if (!iree_any_bit_set(value_flags,
                        LOOM_AMDGPU_FP8_DECODE_VALUE_FLAG_NOT_SUBNORMAL)) {
    return iree_ok_status();
  }
  loom_amdgpu_fp8_lane_bits_t lane_bits;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp8_lane_bits(
      context, source_op, plan, low_byte, value_flags,
      &kLoomAmdgpuFp8LaneF32ResultFormat, vgpr_type, &lane_bits));

  loom_value_id_t low_normal_payload = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp8_lane_normal_payload(
      context, source_op, plan, &kLoomAmdgpuFp8LaneF32ResultFormat,
      lane_bits.low_source_no_sign, vgpr_type, &low_normal_payload));

  loom_value_id_t low_finite_payload = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp8_normal_or_zero_lane_payload(
      context, source_op, plan, lane_bits.low_source_no_sign,
      low_normal_payload, value_flags, vgpr_type, mask_type,
      &low_finite_payload));

  return loom_amdgpu_emit_fp8_signed_lane_payload(
      context, source_op, plan, low_byte, &lane_bits, low_finite_payload,
      value_flags, &kLoomAmdgpuFp8LaneF32ResultFormat, vgpr_type, mask_type,
      out_lane);
}

iree_status_t loom_amdgpu_emit_fp8_to_f32_lane(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fp8_decode_plan_t* plan, loom_value_id_t low_byte,
    loom_amdgpu_fp8_decode_value_flags_t value_flags, loom_type_t vgpr_type,
    loom_type_t sgpr_type, loom_type_t mask_type, loom_value_id_t* out_lane) {
  IREE_RETURN_IF_ERROR(loom_amdgpu_try_emit_fp8_not_subnormal_to_f32_lane(
      context, source_op, plan, low_byte, value_flags, vgpr_type, mask_type,
      out_lane));
  if (*out_lane != LOOM_VALUE_ID_INVALID) {
    return iree_ok_status();
  }

  loom_value_id_t bf16_lane = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp8_to_bf16_lane(
      context, source_op, plan, low_byte, value_flags, vgpr_type, sgpr_type,
      mask_type, &bf16_lane));
  return loom_amdgpu_emit_vgpr_shift(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32_LIT, 16,
      bf16_lane, vgpr_type, out_lane);
}
