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

  loom_amdgpu_initialize_fp8_decode_format(element_type, out_plan);
  return iree_ok_status();
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

static iree_status_t loom_amdgpu_emit_fp8_subnormal_bf16_bits_permute(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fp8_decode_plan_t* plan,
    loom_value_id_t low_source_no_sign, loom_type_t vgpr_type,
    loom_value_id_t* out_bf16_bits, bool* out_selected) {
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
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32,
      plan->subnormal_bf16_table_words[0], vgpr_type, &low_table_lo));
  loom_value_id_t low_table_hi = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32,
      plan->subnormal_bf16_table_words[1], vgpr_type, &low_table_hi));

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
    loom_value_id_t* out_bf16_bits, bool* out_selected) {
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
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32,
        plan->subnormal_bf16_byte_table_words[byte_index][0], vgpr_type,
        &low_table_lo));
    loom_value_id_t low_table_hi = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32,
        plan->subnormal_bf16_byte_table_words[byte_index][1], vgpr_type,
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

iree_status_t loom_amdgpu_emit_fp8_to_bf16_lane(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fp8_decode_plan_t* plan, loom_value_id_t low_byte,
    loom_type_t vgpr_type, loom_type_t mask_type, loom_value_id_t* out_lane) {
  *out_lane = LOOM_VALUE_ID_INVALID;
  const loom_scalar_type_fp8_format_t* format = &plan->format;

  loom_value_id_t low_sign_bits = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary_immediate(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT, low_byte,
      UINT32_C(0x80), vgpr_type, &low_sign_bits));
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_shift(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32_LIT, 8,
      low_sign_bits, vgpr_type, &low_sign_bits));

  loom_value_id_t low_source_no_sign = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary_immediate(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT, low_byte,
      UINT32_C(0x7F), vgpr_type, &low_source_no_sign));

  loom_value_id_t low_normal_payload = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_shift(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32_LIT,
      7u - format->mantissa_bits, low_source_no_sign, vgpr_type,
      &low_normal_payload));
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary_immediate(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_ADD_U32_LIT,
      low_normal_payload, (127u - format->exponent_bias) << 7, vgpr_type,
      &low_normal_payload));

  loom_value_id_t low_subnormal_payload = LOOM_VALUE_ID_INVALID;
  bool selected_subnormal_permute = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp8_subnormal_bf16_bits_permute(
      context, source_op, plan, low_source_no_sign, vgpr_type,
      &low_subnormal_payload, &selected_subnormal_permute));
  if (!selected_subnormal_permute) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp8_subnormal_bf16_bits_byte_tables(
        context, source_op, plan, low_source_no_sign, vgpr_type,
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
  loom_value_id_t low_finite_payload = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp8_decode_select_b32(
      context, source_op, low_normal_payload, low_subnormal_payload,
      low_exponent_is_zero, vgpr_type, &low_finite_payload));
  loom_value_id_t low_finite_bits = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_OR_B32, low_sign_bits,
      low_finite_payload, vgpr_type, &low_finite_bits));

  loom_value_id_t low_quiet_nan = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32, 0x7FC0,
      vgpr_type, &low_quiet_nan));

  if (format->special_policy ==
      LOOM_SCALAR_TYPE_FP8_SPECIAL_POLICY_FINITE_NAN) {
    loom_value_id_t low_is_nan = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp8_decode_cmp_u32_lit(
        context, source_op, plan, LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_EQ_I32,
        low_source_no_sign, UINT32_C(0x7F), vgpr_type, mask_type, &low_is_nan));
    return loom_amdgpu_emit_fp8_decode_select_b32(
        context, source_op, low_finite_bits, low_quiet_nan, low_is_nan,
        vgpr_type, out_lane);
  }

  loom_value_id_t low_infinity_bits = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary_immediate(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_OR_B32_LIT,
      low_sign_bits, UINT32_C(0x7F80), vgpr_type, &low_infinity_bits));
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
