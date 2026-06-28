// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/lower/fp8_decode.h"

#include <string.h>

#include "loom/ir/attribute.h"
#include "loom/ops/low/ops.h"
#include "loom/target/arch/amdgpu/lower/constants.h"
#include "loom/target/arch/amdgpu/lower/emit.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"

enum {
  LOOM_AMDGPU_FP8_DECODE_BF16_BYTE_PAIR_LOW_SELECTOR = 0x00050004u,
  LOOM_AMDGPU_FP8_DECODE_BF16_BYTE_PAIR_MIDDLE_SELECTOR = 0x00060005u,
  LOOM_AMDGPU_FP8_DECODE_BF16_BYTE_PAIR_HIGH_SELECTOR = 0x00070006u,
  LOOM_AMDGPU_FP8_DECODE_BF16_PAIR_NO_SIGN_MASK = 0x007F007Fu,
  LOOM_AMDGPU_FP8_DECODE_BF16_PAIR_SIGN_MASK = 0x00800080u,
  LOOM_AMDGPU_FP8_DECODE_BF16_PAIR_SIGN_INSERT_MASK = 0x80008000u,
  LOOM_AMDGPU_FP8_DECODE_BF16_SIGN_INSERT_MASK = 0x00008000u,
  LOOM_AMDGPU_FP8_DECODE_BF16_PAIR_ONE_MASK = 0x00010001u,
  LOOM_AMDGPU_FP8_DECODE_BF16_PAIR_ASHR_15_MASK = 0x000F000Fu,
  LOOM_AMDGPU_FP8_DECODE_BF16_PAIR_E4M3_SUBNORMAL_BIAS = 0xFFF8FFF8u,
  LOOM_AMDGPU_FP8_DECODE_BF16_PAIR_EQ_ZERO_BIAS = 0xFFFFFFFFu,
  LOOM_AMDGPU_FP8_DECODE_BF16_PAIR_E4M3_NAN_NO_SIGN = 0x007F007Fu,
  LOOM_AMDGPU_FP8_DECODE_BF16_PAIR_QUIET_NAN_BITS = 0x7FC07FC0u,
};

static const uint32_t kLoomAmdgpuFp8DecodeBf16BytePairSelectors[] = {
    LOOM_AMDGPU_FP8_DECODE_BF16_BYTE_PAIR_LOW_SELECTOR,
    LOOM_AMDGPU_FP8_DECODE_BF16_BYTE_PAIR_MIDDLE_SELECTOR,
    LOOM_AMDGPU_FP8_DECODE_BF16_BYTE_PAIR_HIGH_SELECTOR,
};

loom_amdgpu_fp8_decode_value_flags_t
loom_amdgpu_fp8_decode_value_flags_from_facts(loom_value_facts_t facts) {
  loom_amdgpu_fp8_decode_value_flags_t flags =
      LOOM_AMDGPU_FP8_DECODE_VALUE_FLAG_NONE;
  if (loom_value_facts_is_not_nan(facts)) {
    flags |= LOOM_AMDGPU_FP8_DECODE_VALUE_FLAG_NOT_NAN;
  }
  if (loom_value_facts_is_not_inf(facts)) {
    flags |= LOOM_AMDGPU_FP8_DECODE_VALUE_FLAG_NOT_INF;
  }
  if (loom_value_facts_is_not_subnormal(facts)) {
    flags |= LOOM_AMDGPU_FP8_DECODE_VALUE_FLAG_NOT_SUBNORMAL;
  }
  if (loom_value_facts_is_non_zero(facts)) {
    flags |= LOOM_AMDGPU_FP8_DECODE_VALUE_FLAG_NON_ZERO;
  }
  return flags;
}

bool loom_amdgpu_fp8_to_f32_descriptor_refs(
    loom_scalar_type_t source_element_type,
    loom_amdgpu_fp8_to_f32_descriptor_refs_t* out_refs) {
  *out_refs = (loom_amdgpu_fp8_to_f32_descriptor_refs_t){
      .lane = LOOM_AMDGPU_DESCRIPTOR_REF_NONE,
      .pair = LOOM_AMDGPU_DESCRIPTOR_REF_NONE,
  };
  switch (source_element_type) {
    case LOOM_SCALAR_TYPE_F8E4M3:
      out_refs->lane = LOOM_AMDGPU_DESCRIPTOR_REF_V_CVT_F32_FP8;
      out_refs->pair = LOOM_AMDGPU_DESCRIPTOR_REF_V_CVT_PK_F32_FP8;
      return true;
    case LOOM_SCALAR_TYPE_F8E5M2:
      out_refs->lane = LOOM_AMDGPU_DESCRIPTOR_REF_V_CVT_F32_BF8;
      out_refs->pair = LOOM_AMDGPU_DESCRIPTOR_REF_V_CVT_PK_F32_BF8;
      return true;
    default:
      return false;
  }
}

static const loom_low_lower_resolved_descriptor_t*
loom_amdgpu_fp8_decode_cmp_src1_inline_descriptor(
    const loom_amdgpu_fp8_decode_plan_t* plan,
    loom_amdgpu_descriptor_ref_t descriptor_ref, uint32_t immediate) {
  if (immediate > LOOM_AMDGPU_SOURCE_INLINE_U32_MAX) {
    return NULL;
  }
  switch (descriptor_ref) {
    case LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_EQ_I32:
      return plan->compare_eq_i32_src1_inline_descriptor.descriptor != NULL
                 ? &plan->compare_eq_i32_src1_inline_descriptor
                 : NULL;
    case LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_UGE_U32:
      return plan->compare_uge_u32_src1_inline_descriptor.descriptor != NULL
                 ? &plan->compare_uge_u32_src1_inline_descriptor
                 : NULL;
    case LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_ULT_U32:
      return plan->compare_ult_u32_src1_inline_descriptor.descriptor != NULL
                 ? &plan->compare_ult_u32_src1_inline_descriptor
                 : NULL;
    default:
      return NULL;
  }
}

static iree_status_t loom_amdgpu_emit_fp8_decode_cmp_u32_lit(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fp8_decode_plan_t* plan,
    loom_amdgpu_descriptor_ref_t descriptor_ref, loom_value_id_t low_value,
    uint32_t immediate, loom_type_t vgpr_type, loom_type_t mask_type,
    loom_value_id_t* out_mask) {
  *out_mask = LOOM_VALUE_ID_INVALID;
  const loom_low_lower_resolved_descriptor_t* src1_inline_descriptor =
      loom_amdgpu_fp8_decode_cmp_src1_inline_descriptor(plan, descriptor_ref,
                                                        immediate);
  if (src1_inline_descriptor != NULL) {
    loom_named_attr_t attrs[1] = {0};
    iree_host_size_t attr_count = 0;
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_append_i64_attr(context, IREE_SV("rhs"), immediate, attrs,
                                    IREE_ARRAYSIZE(attrs), &attr_count));
    const loom_value_id_t operands[] = {low_value};
    loom_op_t* compare_op = NULL;
    IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
        context, src1_inline_descriptor, operands, IREE_ARRAYSIZE(operands),
        loom_make_named_attr_slice(attrs, attr_count), &mask_type, 1,
        /*tied_results=*/NULL, /*tied_result_count=*/0, source_op->location,
        &compare_op));
    *out_mask = loom_value_slice_get(loom_low_op_results(compare_op), 0);
    return iree_ok_status();
  }

  loom_value_id_t low_immediate = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32, immediate,
      vgpr_type, &low_immediate));
  const loom_value_id_t operands[] = {low_value, low_immediate};
  loom_op_t* compare_op = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
      context, source_op, descriptor_ref, operands, IREE_ARRAYSIZE(operands),
      loom_named_attr_slice_empty(), &mask_type, 1, &compare_op));
  *out_mask = loom_value_slice_get(loom_low_op_results(compare_op), 0);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_fp8_decode_sgpr_const_u32(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    uint32_t value, loom_type_t sgpr_type, loom_value_id_t* out_value) {
  return loom_amdgpu_emit_const_u32(context, source_op,
                                    LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32, value,
                                    sgpr_type, out_value);
}

static iree_status_t loom_amdgpu_emit_fp8_decode_select_b32(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_false_value, loom_value_id_t low_true_value,
    loom_value_id_t low_condition, loom_type_t vgpr_type,
    loom_value_id_t* out_value) {
  *out_value = LOOM_VALUE_ID_INVALID;
  const loom_value_id_t operands[] = {
      low_false_value,
      low_true_value,
      low_condition,
  };
  loom_op_t* select_op = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_CNDMASK_B32, operands,
      IREE_ARRAYSIZE(operands), loom_named_attr_slice_empty(), &vgpr_type, 1,
      &select_op));
  *out_value = loom_value_slice_get(loom_low_op_results(select_op), 0);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_fp8_apply_special_values(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fp8_decode_plan_t* plan, loom_value_id_t low_finite_bits,
    loom_value_id_t low_sign_bits, loom_value_id_t low_source_no_sign,
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
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp8_decode_cmp_u32_lit(
        context, source_op, plan, LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_EQ_I32,
        low_source_no_sign, UINT32_C(0x7F), vgpr_type, mask_type, &low_is_nan));
    return loom_amdgpu_emit_fp8_decode_select_b32(
        context, source_op, low_finite_bits, low_quiet_nan, low_is_nan,
        vgpr_type, out_lane);
  }

  if (value_not_inf) {
    loom_value_id_t low_quiet_nan = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32,
        quiet_nan_bits, vgpr_type, &low_quiet_nan));
    loom_value_id_t low_is_top_exponent = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp8_decode_cmp_u32_lit(
        context, source_op, plan, LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_UGE_U32,
        low_source_no_sign, UINT32_C(0x7C), vgpr_type, mask_type,
        &low_is_top_exponent));
    return loom_amdgpu_emit_fp8_decode_select_b32(
        context, source_op, low_finite_bits, low_quiet_nan, low_is_top_exponent,
        vgpr_type, out_lane);
  }

  if (value_not_nan) {
    loom_value_id_t low_infinity_bits = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary_immediate(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_OR_B32_LIT,
        low_sign_bits, infinity_magnitude_bits, vgpr_type, &low_infinity_bits));
    loom_value_id_t low_is_infinity = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp8_decode_cmp_u32_lit(
        context, source_op, plan, LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_EQ_I32,
        low_source_no_sign, UINT32_C(0x7C), vgpr_type, mask_type,
        &low_is_infinity));
    return loom_amdgpu_emit_fp8_decode_select_b32(
        context, source_op, low_finite_bits, low_infinity_bits, low_is_infinity,
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
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp8_decode_cmp_u32_lit(
      context, source_op, plan, LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_EQ_I32,
      low_source_no_sign, UINT32_C(0x7C), vgpr_type, mask_type,
      &low_is_infinity));
  loom_value_id_t low_top_bits = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp8_decode_select_b32(
      context, source_op, low_quiet_nan, low_infinity_bits, low_is_infinity,
      vgpr_type, &low_top_bits));
  loom_value_id_t low_is_top_exponent = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp8_decode_cmp_u32_lit(
      context, source_op, plan, LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_UGE_U32,
      low_source_no_sign, UINT32_C(0x7C), vgpr_type, mask_type,
      &low_is_top_exponent));
  return loom_amdgpu_emit_fp8_decode_select_b32(
      context, source_op, low_finite_bits, low_top_bits, low_is_top_exponent,
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
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp8_decode_cmp_u32_lit(
        context, source_op, plan, LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_UGE_U32,
        low_source_no_sign, UINT32_C(1) << i, vgpr_type, mask_type,
        &low_has_leading_index));
    loom_value_id_t low_candidate = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32, i, vgpr_type,
        &low_candidate));
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp8_decode_select_b32(
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
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp8_decode_cmp_u32_lit(
      context, source_op, plan, LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_EQ_I32,
      low_source_no_sign, 0, vgpr_type, mask_type, &low_source_is_zero));
  return loom_amdgpu_emit_fp8_decode_select_b32(
      context, source_op, low_nonzero_bits, low_zero_bits, low_source_is_zero,
      vgpr_type, out_bf16_bits);
}

static uint32_t loom_amdgpu_fp8_subnormal_bf16_payload(
    const loom_scalar_type_fp8_format_t* format, uint32_t mantissa) {
  if (mantissa == 0) {
    return 0;
  }
  uint32_t leading_index = 0;
  for (uint32_t i = 1; i < format->mantissa_bits; ++i) {
    if ((mantissa & (UINT32_C(1) << i)) != 0) {
      leading_index = i;
    }
  }
  const uint32_t exponent =
      128u - format->exponent_bias - format->mantissa_bits + leading_index;
  const uint32_t fraction = (mantissa << (7u - leading_index)) & UINT32_C(0x7F);
  return (exponent << 7) | fraction;
}

static uint32_t loom_amdgpu_fp8_subnormal_table_word(
    const loom_scalar_type_fp8_format_t* format, uint32_t mantissa_base) {
  return loom_amdgpu_fp8_subnormal_bf16_payload(format, mantissa_base) |
         (loom_amdgpu_fp8_subnormal_bf16_payload(format, mantissa_base + 1u)
          << 16);
}

static uint32_t loom_amdgpu_fp8_subnormal_byte_table_word(
    const loom_scalar_type_fp8_format_t* format, uint32_t byte_index,
    uint32_t mantissa_base) {
  uint32_t table_word = 0;
  for (uint32_t i = 0; i < 4; ++i) {
    const uint32_t payload =
        loom_amdgpu_fp8_subnormal_bf16_payload(format, mantissa_base + i);
    table_word |= ((payload >> (byte_index * 8u)) & UINT32_C(0xFF)) << (i * 8u);
  }
  return table_word;
}

static void loom_amdgpu_initialize_fp8_decode_format(
    loom_scalar_type_t element_type, loom_amdgpu_fp8_decode_plan_t* plan) {
  if (!loom_scalar_type_fp8_format(element_type, &plan->format)) {
    IREE_ASSERT_UNREACHABLE("selected AMDGPU FP8 decode");
    IREE_BUILTIN_UNREACHABLE();
  }
  if (plan->format.mantissa_bits == 2) {
    plan->subnormal_bf16_table_words[0] =
        loom_amdgpu_fp8_subnormal_table_word(&plan->format, 0);
    plan->subnormal_bf16_table_words[1] =
        loom_amdgpu_fp8_subnormal_table_word(&plan->format, 2);
  } else if (plan->format.mantissa_bits == 3) {
    for (uint32_t byte_index = 0; byte_index < LOOM_AMDGPU_FP8_BF16_BYTE_COUNT;
         ++byte_index) {
      plan->subnormal_bf16_byte_table_words[byte_index][0] =
          loom_amdgpu_fp8_subnormal_byte_table_word(&plan->format, byte_index,
                                                    0);
      plan->subnormal_bf16_byte_table_words[byte_index][1] =
          loom_amdgpu_fp8_subnormal_byte_table_word(&plan->format, byte_index,
                                                    4);
    }
  }
}

iree_status_t loom_amdgpu_select_fp8_decode_plan(
    loom_low_lower_context_t* context, loom_scalar_type_t element_type,
    loom_amdgpu_fp8_decode_plan_t* out_plan) {
  memset(out_plan, 0, sizeof(*out_plan));

  bool has_bfe_u32 = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref_if_present(
      context, LOOM_AMDGPU_DESCRIPTOR_REF_V_BFE_U32_OFFSET_WIDTH_INLINE,
      &out_plan->bfe_u32_descriptor, &has_bfe_u32));
  if (has_bfe_u32) {
    out_plan->flags |= LOOM_AMDGPU_FP8_DECODE_PLAN_FLAG_HAS_BFE_U32;
  }

  bool has_inline_compare = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref_if_present(
      context, LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_EQ_I32_SRC1_INLINE,
      &out_plan->compare_eq_i32_src1_inline_descriptor, &has_inline_compare));
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref_if_present(
      context, LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_UGE_U32_SRC1_INLINE,
      &out_plan->compare_uge_u32_src1_inline_descriptor, &has_inline_compare));
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref_if_present(
      context, LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_ULT_U32_SRC1_INLINE,
      &out_plan->compare_ult_u32_src1_inline_descriptor, &has_inline_compare));

  bool has_pack_u16 = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref_if_present(
      context, LOOM_AMDGPU_DESCRIPTOR_REF_V_CVT_PK_U16_U32,
      &out_plan->pack_u16_descriptor, &has_pack_u16));
  if (has_pack_u16) {
    out_plan->flags |= LOOM_AMDGPU_FP8_DECODE_PLAN_FLAG_HAS_PACK_U16;
  }

  bool has_perm_b32 = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref_if_present(
      context, LOOM_AMDGPU_DESCRIPTOR_REF_V_PERM_B32,
      &out_plan->perm_b32_descriptor, &has_perm_b32));
  if (has_perm_b32) {
    out_plan->flags |= LOOM_AMDGPU_FP8_DECODE_PLAN_FLAG_HAS_PERM_B32;
  }

  loom_amdgpu_fp8_to_f32_descriptor_refs_t native_refs = {0};
  if (loom_amdgpu_fp8_to_f32_descriptor_refs(element_type, &native_refs)) {
    bool has_native_pair = false;
    IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref_if_present(
        context, native_refs.pair, &out_plan->native_f32_pair_descriptor,
        &has_native_pair));
    if (has_native_pair) {
      out_plan->flags |= LOOM_AMDGPU_FP8_DECODE_PLAN_FLAG_HAS_NATIVE_F32_PAIR;
    }
  }

  bool has_native_bf16_pack = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref_if_present(
      context, LOOM_AMDGPU_DESCRIPTOR_REF_V_CVT_PK_BF16_F32,
      &out_plan->native_bf16_pack_descriptor, &has_native_bf16_pack));
  if (has_native_bf16_pack) {
    out_plan->flags |= LOOM_AMDGPU_FP8_DECODE_PLAN_FLAG_HAS_NATIVE_BF16_PACK;
  }

  bool has_add3_src2_literal = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref_if_present(
      context, LOOM_AMDGPU_DESCRIPTOR_REF_V_ADD3_U32_SRC2_LIT,
      &out_plan->add3_src2_literal_descriptor, &has_add3_src2_literal));
  if (has_add3_src2_literal) {
    out_plan->flags |= LOOM_AMDGPU_FP8_DECODE_PLAN_FLAG_HAS_ADD3_SRC2_LITERAL;
  }

  bool has_lshl_add_u32_shift_imm = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref_if_present(
      context, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHL_ADD_U32_SHIFT_IMM,
      &out_plan->lshl_add_u32_shift_imm_descriptor,
      &has_lshl_add_u32_shift_imm));
  if (has_lshl_add_u32_shift_imm) {
    out_plan->flags |=
        LOOM_AMDGPU_FP8_DECODE_PLAN_FLAG_HAS_LSHL_ADD_U32_SHIFT_IMM;
  }

  bool has_perm_b32_src2_literal = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref_if_present(
      context, LOOM_AMDGPU_DESCRIPTOR_REF_V_PERM_B32_SRC2_LIT,
      &out_plan->perm_b32_src2_literal_descriptor, &has_perm_b32_src2_literal));
  if (has_perm_b32_src2_literal) {
    out_plan->flags |=
        LOOM_AMDGPU_FP8_DECODE_PLAN_FLAG_HAS_PERM_B32_SRC2_LITERAL;
  }

  bool has_pk_min_u16 = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref_if_present(
      context, LOOM_AMDGPU_DESCRIPTOR_REF_V_PK_MIN_U16,
      &out_plan->pk_min_u16_descriptor, &has_pk_min_u16));
  if (has_pk_min_u16) {
    out_plan->flags |= LOOM_AMDGPU_FP8_DECODE_PLAN_FLAG_HAS_PK_MIN_U16;
  }

  bool has_pk_mul_lo_u16 = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref_if_present(
      context, LOOM_AMDGPU_DESCRIPTOR_REF_V_PK_MUL_LO_U16,
      &out_plan->pk_mul_lo_u16_descriptor, &has_pk_mul_lo_u16));
  if (has_pk_mul_lo_u16) {
    out_plan->flags |= LOOM_AMDGPU_FP8_DECODE_PLAN_FLAG_HAS_PK_MUL_LO_U16;
  }

  bool has_pk_add_u16 = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref_if_present(
      context, LOOM_AMDGPU_DESCRIPTOR_REF_V_PK_ADD_U16,
      &out_plan->pk_add_u16_descriptor, &has_pk_add_u16));
  if (has_pk_add_u16) {
    out_plan->flags |= LOOM_AMDGPU_FP8_DECODE_PLAN_FLAG_HAS_PK_ADD_U16;
  }

  bool has_pk_ashrrev_i16 = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref_if_present(
      context, LOOM_AMDGPU_DESCRIPTOR_REF_V_PK_ASHRREV_I16,
      &out_plan->pk_ashrrev_i16_descriptor, &has_pk_ashrrev_i16));
  if (has_pk_ashrrev_i16) {
    out_plan->flags |= LOOM_AMDGPU_FP8_DECODE_PLAN_FLAG_HAS_PK_ASHRREV_I16;
  }

  bool has_bfi_b32_src0_literal = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref_if_present(
      context, LOOM_AMDGPU_DESCRIPTOR_REF_V_BFI_B32_SRC0_LIT,
      &out_plan->bfi_b32_src0_literal_descriptor, &has_bfi_b32_src0_literal));
  if (has_bfi_b32_src0_literal) {
    out_plan->flags |=
        LOOM_AMDGPU_FP8_DECODE_PLAN_FLAG_HAS_BFI_B32_SRC0_LITERAL;
  }

  loom_amdgpu_initialize_fp8_decode_format(element_type, out_plan);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_fp8_normal_bf16_payload(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fp8_decode_plan_t* plan,
    loom_value_id_t low_source_no_sign, loom_type_t vgpr_type,
    loom_value_id_t* out_payload) {
  const loom_scalar_type_fp8_format_t* format = &plan->format;
  const uint32_t payload_shift = 7u - format->mantissa_bits;
  const uint32_t exponent_bias = (127u - format->exponent_bias) << 7;
  if (iree_any_bit_set(
          plan->flags,
          LOOM_AMDGPU_FP8_DECODE_PLAN_FLAG_HAS_LSHL_ADD_U32_SHIFT_IMM)) {
    loom_value_id_t low_exponent_bias = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32, exponent_bias,
        vgpr_type, &low_exponent_bias));
    return loom_amdgpu_emit_resolved_vgpr_lshl_add_u32(
        context, source_op, &plan->lshl_add_u32_shift_imm_descriptor,
        low_source_no_sign, low_exponent_bias, payload_shift, vgpr_type,
        out_payload);
  }

  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_shift(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32_LIT,
      payload_shift, low_source_no_sign, vgpr_type, out_payload));
  return loom_amdgpu_emit_vgpr_binary_immediate(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_ADD_U32_LIT,
      *out_payload, exponent_bias, vgpr_type, out_payload);
}

static iree_status_t loom_amdgpu_emit_fp8_normal_packed_bf16_payload(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fp8_decode_plan_t* plan,
    loom_value_id_t low_source_no_sign_pair, loom_type_t vgpr_type,
    loom_value_id_t* out_payload) {
  const loom_scalar_type_fp8_format_t* format = &plan->format;
  const uint32_t payload_shift = 7u - format->mantissa_bits;
  const uint32_t exponent_bias = (127u - format->exponent_bias) << 7;
  const uint32_t packed_exponent_bias = exponent_bias | (exponent_bias << 16);
  if (iree_any_bit_set(
          plan->flags,
          LOOM_AMDGPU_FP8_DECODE_PLAN_FLAG_HAS_LSHL_ADD_U32_SHIFT_IMM)) {
    loom_value_id_t low_exponent_bias = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32,
        packed_exponent_bias, vgpr_type, &low_exponent_bias));
    return loom_amdgpu_emit_resolved_vgpr_lshl_add_u32(
        context, source_op, &plan->lshl_add_u32_shift_imm_descriptor,
        low_source_no_sign_pair, low_exponent_bias, payload_shift, vgpr_type,
        out_payload);
  }

  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_shift(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32_LIT,
      payload_shift, low_source_no_sign_pair, vgpr_type, out_payload));
  loom_value_id_t low_exponent_bias = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32,
      packed_exponent_bias, vgpr_type, &low_exponent_bias));
  return loom_amdgpu_emit_vgpr_binary(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_ADD_U32, *out_payload,
      low_exponent_bias, vgpr_type, out_payload);
}

static iree_status_t loom_amdgpu_emit_fp8_normal_f32_payload(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fp8_decode_plan_t* plan,
    loom_value_id_t low_source_no_sign, loom_type_t vgpr_type,
    loom_value_id_t* out_payload) {
  const loom_scalar_type_fp8_format_t* format = &plan->format;
  const uint32_t payload_shift = 23u - format->mantissa_bits;
  const uint32_t exponent_bias = (127u - format->exponent_bias) << 23;
  if (iree_any_bit_set(
          plan->flags,
          LOOM_AMDGPU_FP8_DECODE_PLAN_FLAG_HAS_LSHL_ADD_U32_SHIFT_IMM)) {
    loom_value_id_t low_exponent_bias = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32, exponent_bias,
        vgpr_type, &low_exponent_bias));
    return loom_amdgpu_emit_resolved_vgpr_lshl_add_u32(
        context, source_op, &plan->lshl_add_u32_shift_imm_descriptor,
        low_source_no_sign, low_exponent_bias, payload_shift, vgpr_type,
        out_payload);
  }

  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_shift(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32_LIT,
      payload_shift, low_source_no_sign, vgpr_type, out_payload));
  return loom_amdgpu_emit_vgpr_binary_immediate(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_ADD_U32_LIT,
      *out_payload, exponent_bias, vgpr_type, out_payload);
}

static iree_status_t loom_amdgpu_emit_fp8_decode_perm_b32(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_low_lower_resolved_descriptor_t* descriptor,
    loom_value_id_t low_src0, loom_value_id_t low_src1,
    loom_value_id_t low_selector, loom_type_t vgpr_type,
    loom_value_id_t* out_value) {
  *out_value = LOOM_VALUE_ID_INVALID;
  const loom_value_id_t operands[] = {low_src0, low_src1, low_selector};
  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
      context, descriptor, operands, IREE_ARRAYSIZE(operands),
      loom_named_attr_slice_empty(), &vgpr_type, 1,
      /*tied_results=*/NULL, /*tied_result_count=*/0, source_op->location,
      &low_op));
  *out_value = loom_value_slice_get(loom_low_op_results(low_op), 0);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_fp8_decode_resolved_vgpr_binary(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_low_lower_resolved_descriptor_t* descriptor, loom_value_id_t lhs,
    loom_value_id_t rhs, loom_type_t vgpr_type, loom_value_id_t* out_value) {
  *out_value = LOOM_VALUE_ID_INVALID;
  const loom_value_id_t operands[] = {lhs, rhs};
  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
      context, descriptor, operands, IREE_ARRAYSIZE(operands),
      loom_named_attr_slice_empty(), &vgpr_type, 1,
      /*tied_results=*/NULL, /*tied_result_count=*/0, source_op->location,
      &low_op));
  *out_value = loom_value_slice_get(loom_low_op_results(low_op), 0);
  return iree_ok_status();
}

static bool loom_amdgpu_fp8_decode_value_is_finite(
    loom_amdgpu_fp8_decode_value_flags_t value_flags) {
  return iree_all_bits_set(value_flags,
                           LOOM_AMDGPU_FP8_DECODE_VALUE_FLAG_NOT_NAN |
                               LOOM_AMDGPU_FP8_DECODE_VALUE_FLAG_NOT_INF);
}

static bool loom_amdgpu_fp8_decode_plan_has_packed_e4m3fn_repair(
    const loom_amdgpu_fp8_decode_plan_t* plan) {
  const loom_amdgpu_fp8_decode_plan_flags_t required_plan_flags =
      LOOM_AMDGPU_FP8_DECODE_PLAN_FLAG_HAS_PERM_B32 |
      LOOM_AMDGPU_FP8_DECODE_PLAN_FLAG_HAS_PERM_B32_SRC2_LITERAL |
      LOOM_AMDGPU_FP8_DECODE_PLAN_FLAG_HAS_PK_ADD_U16 |
      LOOM_AMDGPU_FP8_DECODE_PLAN_FLAG_HAS_PK_ASHRREV_I16;
  return plan->format.exponent_bits == 4 && plan->format.mantissa_bits == 3 &&
         plan->format.exponent_bias == 7 &&
         plan->format.special_policy ==
             LOOM_SCALAR_TYPE_FP8_SPECIAL_POLICY_FINITE_NAN &&
         iree_all_bits_set(plan->flags, required_plan_flags);
}

static bool loom_amdgpu_can_emit_fp8_pair_to_packed_bf16_normal_path(
    const loom_amdgpu_fp8_decode_plan_t* plan,
    loom_amdgpu_fp8_decode_value_flags_t value_flags) {
  const loom_amdgpu_fp8_decode_plan_flags_t required_plan_flags =
      LOOM_AMDGPU_FP8_DECODE_PLAN_FLAG_HAS_PERM_B32_SRC2_LITERAL;
  if (!iree_all_bits_set(plan->flags, required_plan_flags) ||
      !loom_amdgpu_fp8_decode_value_is_finite(value_flags) ||
      !iree_any_bit_set(value_flags,
                        LOOM_AMDGPU_FP8_DECODE_VALUE_FLAG_NOT_SUBNORMAL)) {
    return false;
  }
  if (iree_any_bit_set(value_flags,
                       LOOM_AMDGPU_FP8_DECODE_VALUE_FLAG_NON_ZERO)) {
    return true;
  }
  const loom_amdgpu_fp8_decode_plan_flags_t required_zero_repair_flags =
      LOOM_AMDGPU_FP8_DECODE_PLAN_FLAG_HAS_PK_MIN_U16 |
      LOOM_AMDGPU_FP8_DECODE_PLAN_FLAG_HAS_PK_MUL_LO_U16;
  return iree_all_bits_set(plan->flags, required_zero_repair_flags);
}

static iree_status_t loom_amdgpu_emit_fp8_subnormal_bf16_bits_permute(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fp8_decode_plan_t* plan,
    loom_value_id_t low_source_no_sign, loom_type_t vgpr_type,
    loom_type_t sgpr_type, loom_value_id_t* out_bf16_bits, bool* out_selected) {
  *out_bf16_bits = LOOM_VALUE_ID_INVALID;
  *out_selected = false;
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
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_shift(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32_LIT, 8,
      low_high_byte_offset, vgpr_type, &low_high_byte_offset));
  loom_value_id_t low_selector = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_OR_B32, low_byte_offset,
      low_high_byte_offset, vgpr_type, &low_selector));

  loom_value_id_t low_table_lo = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp8_decode_sgpr_const_u32(
      context, source_op, plan->subnormal_bf16_table_words[0], sgpr_type,
      &low_table_lo));
  loom_value_id_t low_table_hi = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp8_decode_sgpr_const_u32(
      context, source_op, plan->subnormal_bf16_table_words[1], sgpr_type,
      &low_table_hi));

  loom_value_id_t low_unsigned_bits = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp8_decode_perm_b32(
      context, source_op, &plan->perm_b32_descriptor, low_table_hi,
      low_table_lo, low_selector, vgpr_type, &low_unsigned_bits));

  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary_immediate(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT,
      low_unsigned_bits, UINT32_C(0xFFFF), vgpr_type, &low_unsigned_bits));
  *out_bf16_bits = low_unsigned_bits;
  *out_selected = true;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_fp8_subnormal_bf16_bits_byte_tables(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fp8_decode_plan_t* plan,
    loom_value_id_t low_source_no_sign, loom_type_t vgpr_type,
    loom_type_t sgpr_type, loom_value_id_t* out_bf16_bits, bool* out_selected) {
  *out_bf16_bits = LOOM_VALUE_ID_INVALID;
  *out_selected = false;
  if (!iree_any_bit_set(plan->flags,
                        LOOM_AMDGPU_FP8_DECODE_PLAN_FLAG_HAS_PERM_B32) ||
      plan->format.mantissa_bits != 3) {
    return iree_ok_status();
  }

  loom_value_id_t low_payload_bytes[LOOM_AMDGPU_FP8_BF16_BYTE_COUNT] = {0};
  for (uint32_t byte_index = 0; byte_index < LOOM_AMDGPU_FP8_BF16_BYTE_COUNT;
       ++byte_index) {
    loom_value_id_t low_table_lo = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp8_decode_sgpr_const_u32(
        context, source_op,
        plan->subnormal_bf16_byte_table_words[byte_index][0], sgpr_type,
        &low_table_lo));
    loom_value_id_t low_table_hi = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp8_decode_sgpr_const_u32(
        context, source_op,
        plan->subnormal_bf16_byte_table_words[byte_index][1], sgpr_type,
        &low_table_hi));

    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp8_decode_perm_b32(
        context, source_op, &plan->perm_b32_descriptor, low_table_hi,
        low_table_lo, low_source_no_sign, vgpr_type,
        &low_payload_bytes[byte_index]));
  }

  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_shift(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32_LIT, 8,
      low_payload_bytes[1], vgpr_type, &low_payload_bytes[1]));
  loom_value_id_t low_unsigned_bits = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_OR_B32,
      low_payload_bytes[0], low_payload_bytes[1], vgpr_type,
      &low_unsigned_bits));
  *out_bf16_bits = low_unsigned_bits;
  *out_selected = true;
  return iree_ok_status();
}

static iree_status_t
loom_amdgpu_emit_fp8_subnormal_packed_bf16_bits_byte_tables(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fp8_decode_plan_t* plan,
    loom_value_id_t low_source_no_sign_pair, loom_type_t vgpr_type,
    loom_type_t sgpr_type, loom_value_id_t* out_bf16_bits) {
  *out_bf16_bits = LOOM_VALUE_ID_INVALID;
  loom_value_id_t low_payload_bytes[LOOM_AMDGPU_FP8_BF16_BYTE_COUNT] = {0};
  for (uint32_t byte_index = 0; byte_index < LOOM_AMDGPU_FP8_BF16_BYTE_COUNT;
       ++byte_index) {
    loom_value_id_t low_table_lo = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp8_decode_sgpr_const_u32(
        context, source_op,
        plan->subnormal_bf16_byte_table_words[byte_index][0], sgpr_type,
        &low_table_lo));
    loom_value_id_t low_table_hi = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp8_decode_sgpr_const_u32(
        context, source_op,
        plan->subnormal_bf16_byte_table_words[byte_index][1], sgpr_type,
        &low_table_hi));

    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp8_decode_perm_b32(
        context, source_op, &plan->perm_b32_descriptor, low_table_hi,
        low_table_lo, low_source_no_sign_pair, vgpr_type,
        &low_payload_bytes[byte_index]));
  }

  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_shift(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32_LIT, 8,
      low_payload_bytes[1], vgpr_type, &low_payload_bytes[1]));
  return loom_amdgpu_emit_vgpr_binary(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_OR_B32,
      low_payload_bytes[0], low_payload_bytes[1], vgpr_type, out_bf16_bits);
}

static iree_status_t loom_amdgpu_emit_fp8_packed_bf16_pair_lt_mask(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fp8_decode_plan_t* plan, loom_value_id_t low_value_pair,
    uint32_t lane_bias, loom_type_t vgpr_type, loom_type_t sgpr_type,
    loom_value_id_t* out_mask) {
  *out_mask = LOOM_VALUE_ID_INVALID;
  loom_value_id_t low_bias = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp8_decode_sgpr_const_u32(
      context, source_op, lane_bias, sgpr_type, &low_bias));
  loom_value_id_t low_biased_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp8_decode_resolved_vgpr_binary(
      context, source_op, &plan->pk_add_u16_descriptor, low_value_pair,
      low_bias, vgpr_type, &low_biased_value));

  loom_value_id_t low_shift = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp8_decode_sgpr_const_u32(
      context, source_op, LOOM_AMDGPU_FP8_DECODE_BF16_PAIR_ASHR_15_MASK,
      sgpr_type, &low_shift));
  return loom_amdgpu_emit_fp8_decode_resolved_vgpr_binary(
      context, source_op, &plan->pk_ashrrev_i16_descriptor, low_shift,
      low_biased_value, vgpr_type, out_mask);
}

static iree_status_t loom_amdgpu_emit_fp8_packed_bf16_pair_eq_mask(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fp8_decode_plan_t* plan, loom_value_id_t low_value_pair,
    uint32_t lane_value, loom_type_t vgpr_type, loom_type_t sgpr_type,
    loom_value_id_t* out_mask) {
  *out_mask = LOOM_VALUE_ID_INVALID;
  loom_value_id_t low_delta = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary_immediate(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_XOR_B32_LIT,
      low_value_pair, lane_value, vgpr_type, &low_delta));
  return loom_amdgpu_emit_fp8_packed_bf16_pair_lt_mask(
      context, source_op, plan, low_delta,
      LOOM_AMDGPU_FP8_DECODE_BF16_PAIR_EQ_ZERO_BIAS, vgpr_type, sgpr_type,
      out_mask);
}

static iree_status_t loom_amdgpu_emit_fp8_packed_bf16_pair_select(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_false_value, loom_value_id_t low_true_value,
    loom_value_id_t low_mask, loom_type_t vgpr_type,
    loom_value_id_t* out_value) {
  *out_value = LOOM_VALUE_ID_INVALID;
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

iree_status_t loom_amdgpu_try_emit_fp8_pair_to_packed_bf16(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fp8_decode_plan_t* plan,
    loom_value_id_t low_source_register, uint32_t byte_offset,
    loom_amdgpu_fp8_decode_value_flags_t value_flags, loom_type_t vgpr_type,
    loom_type_t sgpr_type, loom_value_id_t* out_low_packet,
    bool* out_selected) {
  *out_low_packet = LOOM_VALUE_ID_INVALID;
  *out_selected = false;

  const bool can_use_normal_path =
      loom_amdgpu_can_emit_fp8_pair_to_packed_bf16_normal_path(plan,
                                                               value_flags);
  const bool can_use_exact_repair =
      loom_amdgpu_fp8_decode_plan_has_packed_e4m3fn_repair(plan);
  if ((!can_use_normal_path && !can_use_exact_repair) || byte_offset > 2) {
    return iree_ok_status();
  }

  const bool value_non_zero =
      iree_any_bit_set(value_flags, LOOM_AMDGPU_FP8_DECODE_VALUE_FLAG_NON_ZERO);
  const bool value_not_nan =
      iree_any_bit_set(value_flags, LOOM_AMDGPU_FP8_DECODE_VALUE_FLAG_NOT_NAN);
  const bool use_exact_repair = !can_use_normal_path;
  loom_value_id_t low_zero = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32, 0, vgpr_type,
      &low_zero));
  loom_value_id_t low_expanded_pair = LOOM_VALUE_ID_INVALID;
  const uint32_t selector =
      kLoomAmdgpuFp8DecodeBf16BytePairSelectors[byte_offset];
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_resolved_vgpr_binary_immediate(
      context, source_op, &plan->perm_b32_src2_literal_descriptor,
      low_source_register, low_zero, selector, vgpr_type, &low_expanded_pair));

  loom_value_id_t low_source_no_sign = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary_immediate(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT,
      low_expanded_pair, LOOM_AMDGPU_FP8_DECODE_BF16_PAIR_NO_SIGN_MASK,
      vgpr_type, &low_source_no_sign));

  loom_value_id_t low_normal_payload = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp8_normal_packed_bf16_payload(
      context, source_op, plan, low_source_no_sign, vgpr_type,
      &low_normal_payload));

  loom_value_id_t low_finite_payload = low_normal_payload;
  if (use_exact_repair) {
    loom_value_id_t low_subnormal_payload = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_emit_fp8_subnormal_packed_bf16_bits_byte_tables(
            context, source_op, plan, low_source_no_sign, vgpr_type, sgpr_type,
            &low_subnormal_payload));
    loom_value_id_t low_subnormal_mask = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp8_packed_bf16_pair_lt_mask(
        context, source_op, plan, low_source_no_sign,
        LOOM_AMDGPU_FP8_DECODE_BF16_PAIR_E4M3_SUBNORMAL_BIAS, vgpr_type,
        sgpr_type, &low_subnormal_mask));
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp8_packed_bf16_pair_select(
        context, source_op, low_normal_payload, low_subnormal_payload,
        low_subnormal_mask, vgpr_type, &low_finite_payload));
  } else if (!value_non_zero) {
    loom_value_id_t low_one_mask = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp8_decode_sgpr_const_u32(
        context, source_op, LOOM_AMDGPU_FP8_DECODE_BF16_PAIR_ONE_MASK,
        sgpr_type, &low_one_mask));
    loom_value_id_t low_nonzero_mask = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp8_decode_resolved_vgpr_binary(
        context, source_op, &plan->pk_min_u16_descriptor, low_source_no_sign,
        low_one_mask, vgpr_type, &low_nonzero_mask));
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp8_decode_resolved_vgpr_binary(
        context, source_op, &plan->pk_mul_lo_u16_descriptor, low_normal_payload,
        low_nonzero_mask, vgpr_type, &low_finite_payload));
  }

  if (iree_any_bit_set(
          plan->flags,
          LOOM_AMDGPU_FP8_DECODE_PLAN_FLAG_HAS_BFI_B32_SRC0_LITERAL)) {
    loom_value_id_t low_shifted_sign_source = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_shift(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32_LIT, 8,
        low_expanded_pair, vgpr_type, &low_shifted_sign_source));
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_resolved_vgpr_binary_immediate(
        context, source_op, &plan->bfi_b32_src0_literal_descriptor,
        low_shifted_sign_source, low_finite_payload,
        LOOM_AMDGPU_FP8_DECODE_BF16_PAIR_SIGN_INSERT_MASK, vgpr_type,
        out_low_packet));
  } else {
    loom_value_id_t low_sign_bits = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary_immediate(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT,
        low_expanded_pair, LOOM_AMDGPU_FP8_DECODE_BF16_PAIR_SIGN_MASK,
        vgpr_type, &low_sign_bits));
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_shift(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32_LIT, 8,
        low_sign_bits, vgpr_type, &low_sign_bits));
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_OR_B32, low_sign_bits,
        low_finite_payload, vgpr_type, out_low_packet));
  }
  if (use_exact_repair && !value_not_nan) {
    loom_value_id_t low_nan_mask = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp8_packed_bf16_pair_eq_mask(
        context, source_op, plan, low_source_no_sign,
        LOOM_AMDGPU_FP8_DECODE_BF16_PAIR_E4M3_NAN_NO_SIGN, vgpr_type, sgpr_type,
        &low_nan_mask));
    loom_value_id_t low_quiet_nan = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32,
        LOOM_AMDGPU_FP8_DECODE_BF16_PAIR_QUIET_NAN_BITS, vgpr_type,
        &low_quiet_nan));
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp8_packed_bf16_pair_select(
        context, source_op, *out_low_packet, low_quiet_nan, low_nan_mask,
        vgpr_type, out_low_packet));
  }
  *out_selected = true;
  return iree_ok_status();
}

bool loom_amdgpu_can_emit_fp8_pair_to_packed_bf16(
    const loom_amdgpu_fp8_decode_plan_t* plan,
    loom_amdgpu_fp8_decode_value_flags_t value_flags) {
  return loom_amdgpu_fp8_pair_to_packed_bf16_missing_requirements(
             plan, value_flags) ==
         LOOM_AMDGPU_FP8_PACKED_BF16_MISSING_REQUIREMENT_NONE;
}

loom_amdgpu_fp8_packed_bf16_missing_requirements_t
loom_amdgpu_fp8_pair_to_packed_bf16_missing_requirements(
    const loom_amdgpu_fp8_decode_plan_t* plan,
    loom_amdgpu_fp8_decode_value_flags_t value_flags) {
  loom_amdgpu_fp8_packed_bf16_missing_requirements_t missing_requirements =
      LOOM_AMDGPU_FP8_PACKED_BF16_MISSING_REQUIREMENT_NONE;
  const loom_amdgpu_fp8_decode_plan_flags_t required_plan_flags =
      LOOM_AMDGPU_FP8_DECODE_PLAN_FLAG_HAS_PERM_B32_SRC2_LITERAL;
  if (!iree_all_bits_set(plan->flags, required_plan_flags)) {
    missing_requirements |=
        LOOM_AMDGPU_FP8_PACKED_BF16_MISSING_REQUIREMENT_PERMUTE_PACKET;
  }

  const bool has_exact_repair =
      loom_amdgpu_fp8_decode_plan_has_packed_e4m3fn_repair(plan);
  if (!has_exact_repair) {
    const loom_amdgpu_fp8_decode_value_flags_t required_value_flags =
        LOOM_AMDGPU_FP8_DECODE_VALUE_FLAG_NOT_NAN |
        LOOM_AMDGPU_FP8_DECODE_VALUE_FLAG_NOT_INF;
    if (!iree_all_bits_set(value_flags, required_value_flags)) {
      missing_requirements |=
          LOOM_AMDGPU_FP8_PACKED_BF16_MISSING_REQUIREMENT_VALUE_FINITE;
    }
    if (!iree_all_bits_set(value_flags,
                           LOOM_AMDGPU_FP8_DECODE_VALUE_FLAG_NOT_SUBNORMAL)) {
      missing_requirements |=
          LOOM_AMDGPU_FP8_PACKED_BF16_MISSING_REQUIREMENT_VALUE_NOT_SUBNORMAL;
    }
  }

  if (has_exact_repair ||
      iree_any_bit_set(value_flags,
                       LOOM_AMDGPU_FP8_DECODE_VALUE_FLAG_NON_ZERO)) {
    return missing_requirements;
  }

  const loom_amdgpu_fp8_decode_plan_flags_t required_zero_repair_flags =
      LOOM_AMDGPU_FP8_DECODE_PLAN_FLAG_HAS_PK_MIN_U16 |
      LOOM_AMDGPU_FP8_DECODE_PLAN_FLAG_HAS_PK_MUL_LO_U16;
  if (!iree_all_bits_set(plan->flags, required_zero_repair_flags)) {
    missing_requirements |=
        LOOM_AMDGPU_FP8_PACKED_BF16_MISSING_REQUIREMENT_ZERO_REPAIR_PACKETS;
  }
  return missing_requirements;
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
  const bool use_finite_sign_insert =
      loom_amdgpu_fp8_decode_value_is_finite(value_flags) &&
      iree_any_bit_set(
          plan->flags,
          LOOM_AMDGPU_FP8_DECODE_PLAN_FLAG_HAS_BFI_B32_SRC0_LITERAL);

  loom_value_id_t low_sign_bits = LOOM_VALUE_ID_INVALID;
  if (!use_finite_sign_insert) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary_immediate(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT, low_byte,
        UINT32_C(0x80), vgpr_type, &low_sign_bits));
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_shift(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32_LIT, 8,
        low_sign_bits, vgpr_type, &low_sign_bits));
  }

  loom_value_id_t low_source_no_sign = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary_immediate(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT, low_byte,
      UINT32_C(0x7F), vgpr_type, &low_source_no_sign));

  loom_value_id_t low_normal_payload = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp8_normal_bf16_payload(
      context, source_op, plan, low_source_no_sign, vgpr_type,
      &low_normal_payload));

  loom_value_id_t low_finite_payload = LOOM_VALUE_ID_INVALID;
  if (value_not_subnormal && value_non_zero) {
    low_finite_payload = low_normal_payload;
  } else if (value_not_subnormal) {
    loom_value_id_t low_zero_payload = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32, 0, vgpr_type,
        &low_zero_payload));
    loom_value_id_t low_source_is_zero = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp8_decode_cmp_u32_lit(
        context, source_op, plan, LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_EQ_I32,
        low_source_no_sign, 0, vgpr_type, mask_type, &low_source_is_zero));
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp8_decode_select_b32(
        context, source_op, low_normal_payload, low_zero_payload,
        low_source_is_zero, vgpr_type, &low_finite_payload));
  } else {
    loom_value_id_t low_subnormal_payload = LOOM_VALUE_ID_INVALID;
    bool selected_subnormal_permute = false;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp8_subnormal_bf16_bits_permute(
        context, source_op, plan, low_source_no_sign, vgpr_type, sgpr_type,
        &low_subnormal_payload, &selected_subnormal_permute));
    if (!selected_subnormal_permute) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp8_subnormal_bf16_bits_byte_tables(
          context, source_op, plan, low_source_no_sign, vgpr_type, sgpr_type,
          &low_subnormal_payload, &selected_subnormal_permute));
    }
    if (!selected_subnormal_permute) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp8_subnormal_bf16_bits(
          context, source_op, plan, low_source_no_sign, vgpr_type, mask_type,
          &low_subnormal_payload));
    }
    loom_value_id_t low_exponent_is_zero = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp8_decode_cmp_u32_lit(
        context, source_op, plan, LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_ULT_U32,
        low_source_no_sign, UINT32_C(1) << format->mantissa_bits, vgpr_type,
        mask_type, &low_exponent_is_zero));
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp8_decode_select_b32(
        context, source_op, low_normal_payload, low_subnormal_payload,
        low_exponent_is_zero, vgpr_type, &low_finite_payload));
  }

  if (use_finite_sign_insert) {
    loom_value_id_t low_shifted_sign_source = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_shift(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32_LIT, 8,
        low_byte, vgpr_type, &low_shifted_sign_source));
    return loom_amdgpu_emit_resolved_vgpr_binary_immediate(
        context, source_op, &plan->bfi_b32_src0_literal_descriptor,
        low_shifted_sign_source, low_finite_payload,
        LOOM_AMDGPU_FP8_DECODE_BF16_SIGN_INSERT_MASK, vgpr_type, out_lane);
  }

  loom_value_id_t low_finite_bits = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_OR_B32, low_sign_bits,
      low_finite_payload, vgpr_type, &low_finite_bits));

  return loom_amdgpu_emit_fp8_apply_special_values(
      context, source_op, plan, low_finite_bits, low_sign_bits,
      low_source_no_sign, value_flags, /*quiet_nan_bits=*/0x7FC0,
      /*infinity_magnitude_bits=*/0x7F80, vgpr_type, mask_type, out_lane);
}

iree_status_t loom_amdgpu_try_emit_fp8_not_subnormal_to_f32_lane(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fp8_decode_plan_t* plan, loom_value_id_t low_byte,
    loom_amdgpu_fp8_decode_value_flags_t value_flags, loom_type_t vgpr_type,
    loom_type_t mask_type, loom_value_id_t* out_lane, bool* out_selected) {
  *out_lane = LOOM_VALUE_ID_INVALID;
  *out_selected = false;
  if (!iree_any_bit_set(value_flags,
                        LOOM_AMDGPU_FP8_DECODE_VALUE_FLAG_NOT_SUBNORMAL)) {
    return iree_ok_status();
  }
  const bool value_non_zero =
      iree_any_bit_set(value_flags, LOOM_AMDGPU_FP8_DECODE_VALUE_FLAG_NON_ZERO);

  loom_value_id_t low_sign_bits = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary_immediate(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT, low_byte,
      UINT32_C(0x80), vgpr_type, &low_sign_bits));
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_shift(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32_LIT, 24,
      low_sign_bits, vgpr_type, &low_sign_bits));

  loom_value_id_t low_source_no_sign = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary_immediate(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT, low_byte,
      UINT32_C(0x7F), vgpr_type, &low_source_no_sign));

  loom_value_id_t low_normal_payload = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp8_normal_f32_payload(
      context, source_op, plan, low_source_no_sign, vgpr_type,
      &low_normal_payload));

  loom_value_id_t low_finite_payload = LOOM_VALUE_ID_INVALID;
  if (value_non_zero) {
    low_finite_payload = low_normal_payload;
  } else {
    loom_value_id_t low_zero_payload = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32, 0, vgpr_type,
        &low_zero_payload));
    loom_value_id_t low_source_is_zero = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp8_decode_cmp_u32_lit(
        context, source_op, plan, LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_EQ_I32,
        low_source_no_sign, 0, vgpr_type, mask_type, &low_source_is_zero));
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp8_decode_select_b32(
        context, source_op, low_normal_payload, low_zero_payload,
        low_source_is_zero, vgpr_type, &low_finite_payload));
  }

  loom_value_id_t low_finite_bits = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_OR_B32, low_sign_bits,
      low_finite_payload, vgpr_type, &low_finite_bits));

  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp8_apply_special_values(
      context, source_op, plan, low_finite_bits, low_sign_bits,
      low_source_no_sign, value_flags, /*quiet_nan_bits=*/0x7FC00000,
      /*infinity_magnitude_bits=*/0x7F800000, vgpr_type, mask_type, out_lane));
  *out_selected = true;
  return iree_ok_status();
}

iree_status_t loom_amdgpu_emit_packed_bf16_pair(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fp8_decode_plan_t* plan, loom_value_id_t low_element,
    loom_value_id_t high_element, loom_type_t vgpr_type,
    loom_value_id_t* out_low_packet) {
  *out_low_packet = LOOM_VALUE_ID_INVALID;
  if (iree_any_bit_set(plan->flags,
                       LOOM_AMDGPU_FP8_DECODE_PLAN_FLAG_HAS_PACK_U16)) {
    const loom_value_id_t operands[] = {low_element, high_element};
    loom_op_t* low_op = NULL;
    IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
        context, &plan->pack_u16_descriptor, operands, IREE_ARRAYSIZE(operands),
        loom_named_attr_slice_empty(), &vgpr_type, 1,
        /*tied_results=*/NULL, /*tied_result_count=*/0, source_op->location,
        &low_op));
    *out_low_packet = loom_value_slice_get(loom_low_op_results(low_op), 0);
    return iree_ok_status();
  }

  loom_value_id_t high_bits = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_shift(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32_LIT, 16,
      high_element, vgpr_type, &high_bits));
  return loom_amdgpu_emit_vgpr_binary(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_OR_B32, low_element,
      high_bits, vgpr_type, out_low_packet);
}
