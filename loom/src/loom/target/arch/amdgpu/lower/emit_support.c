// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stddef.h>
#include <stdint.h>

#include "loom/codegen/low/builder.h"
#include "loom/ir/context.h"
#include "loom/target/arch/amdgpu/lower/constants.h"
#include "loom/target/arch/amdgpu/lower/emit.h"
#include "loom/target/arch/amdgpu/lower/memory.h"
#include "loom/target/arch/amdgpu/lower/types.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"
#include "loom/target/arch/amdgpu/target_info_defs.h"

enum {
  // Register-part mask for the low half of a 32-bit register.
  LOOM_AMDGPU_REGISTER_PART_MASK_LOW16 = 0x1u,
};

iree_status_t loom_amdgpu_intern(loom_low_lower_context_t* context,
                                 iree_string_view_t string,
                                 loom_string_id_t* out_string_id) {
  return loom_module_intern_string(loom_low_lower_context_module(context),
                                   string, out_string_id);
}

iree_status_t loom_amdgpu_append_i64_attr(loom_low_lower_context_t* context,
                                          iree_string_view_t name,
                                          int64_t value,
                                          loom_named_attr_t* attrs,
                                          iree_host_size_t attr_capacity,
                                          iree_host_size_t* inout_attr_count) {
  IREE_ASSERT_LT(*inout_attr_count, attr_capacity);
  loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_intern(context, name, &name_id));
  attrs[*inout_attr_count] = (loom_named_attr_t){
      .name_id = name_id,
      .value = loom_attr_i64(value),
  };
  *inout_attr_count += 1;
  return iree_ok_status();
}

iree_status_t loom_amdgpu_emit_sgpr_binary(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_descriptor_ref_t descriptor_ref, loom_value_id_t lhs,
    loom_value_id_t rhs, loom_type_t lane_type, loom_value_id_t* out_value) {
  *out_value = LOOM_VALUE_ID_INVALID;
  loom_value_id_t operands[] = {
      lhs,
      rhs,
  };
  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
      context, source_op, descriptor_ref, operands, IREE_ARRAYSIZE(operands),
      loom_make_named_attr_slice(NULL, 0), &lane_type, 1, &low_op));
  *out_value = loom_value_slice_get(loom_low_op_results(low_op), 0);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_sgpr_mul_u32(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_lhs, loom_value_id_t low_rhs, loom_type_t sgpr_type,
    loom_value_id_t* out_low_result) {
  return loom_amdgpu_emit_sgpr_binary(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_MUL_I32, low_lhs,
      low_rhs, sgpr_type, out_low_result);
}

static loom_amdgpu_descriptor_ref_t
loom_amdgpu_sgpr_binary_rhs_inline_descriptor_ref(
    loom_amdgpu_descriptor_ref_t descriptor_ref) {
  switch (descriptor_ref) {
    case LOOM_AMDGPU_DESCRIPTOR_REF_S_AND_B32:
      return LOOM_AMDGPU_DESCRIPTOR_REF_S_AND_B32_RHS_INLINE;
    case LOOM_AMDGPU_DESCRIPTOR_REF_S_ADD_U32:
      return LOOM_AMDGPU_DESCRIPTOR_REF_S_ADD_U32_RHS_INLINE;
    case LOOM_AMDGPU_DESCRIPTOR_REF_S_SUB_U32:
      return LOOM_AMDGPU_DESCRIPTOR_REF_S_SUB_U32_RHS_INLINE;
    case LOOM_AMDGPU_DESCRIPTOR_REF_S_MUL_I32:
      return LOOM_AMDGPU_DESCRIPTOR_REF_S_MUL_I32_RHS_INLINE;
    case LOOM_AMDGPU_DESCRIPTOR_REF_S_LSHL_B32:
      return LOOM_AMDGPU_DESCRIPTOR_REF_S_LSHL_B32_RHS_INLINE;
    case LOOM_AMDGPU_DESCRIPTOR_REF_S_LSHR_B32:
      return LOOM_AMDGPU_DESCRIPTOR_REF_S_LSHR_B32_RHS_INLINE;
    case LOOM_AMDGPU_DESCRIPTOR_REF_S_ASHR_I32:
      return LOOM_AMDGPU_DESCRIPTOR_REF_S_ASHR_I32_RHS_INLINE;
    default:
      return LOOM_AMDGPU_DESCRIPTOR_REF_NONE;
  }
}

static loom_amdgpu_descriptor_ref_t
loom_amdgpu_sgpr_binary_rhs_literal_descriptor_ref(
    loom_amdgpu_descriptor_ref_t descriptor_ref) {
  switch (descriptor_ref) {
    case LOOM_AMDGPU_DESCRIPTOR_REF_S_AND_B32:
      return LOOM_AMDGPU_DESCRIPTOR_REF_S_AND_B32_LIT;
    default:
      return LOOM_AMDGPU_DESCRIPTOR_REF_NONE;
  }
}

iree_status_t loom_amdgpu_emit_sgpr_binary_immediate(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_descriptor_ref_t descriptor_ref, loom_value_id_t value,
    uint32_t immediate, loom_type_t lane_type, loom_value_id_t* out_value) {
  *out_value = LOOM_VALUE_ID_INVALID;

  const loom_low_descriptor_set_t* descriptor_set =
      loom_low_lower_context_descriptor_set(context);
  const loom_amdgpu_descriptor_ref_t rhs_inline_descriptor_ref =
      loom_amdgpu_sgpr_binary_rhs_inline_descriptor_ref(descriptor_ref);
  if (immediate <= LOOM_AMDGPU_SOURCE_INLINE_U32_MAX &&
      loom_amdgpu_descriptor_ref_ordinal(descriptor_set,
                                         rhs_inline_descriptor_ref) !=
          LOOM_LOW_DESCRIPTOR_ORDINAL_NONE) {
    loom_named_attr_t attrs[1] = {0};
    iree_host_size_t attr_count = 0;
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_append_i64_attr(context, IREE_SV("imm32"), immediate, attrs,
                                    IREE_ARRAYSIZE(attrs), &attr_count));
    loom_value_id_t operands[] = {value};
    loom_op_t* low_op = NULL;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
        context, source_op, rhs_inline_descriptor_ref, operands,
        IREE_ARRAYSIZE(operands), loom_make_named_attr_slice(attrs, attr_count),
        &lane_type, 1, &low_op));
    *out_value = loom_value_slice_get(loom_low_op_results(low_op), 0);
    return iree_ok_status();
  }

  const loom_amdgpu_descriptor_ref_t rhs_literal_descriptor_ref =
      loom_amdgpu_sgpr_binary_rhs_literal_descriptor_ref(descriptor_ref);
  if (rhs_literal_descriptor_ref != LOOM_AMDGPU_DESCRIPTOR_REF_NONE &&
      loom_amdgpu_descriptor_ref_ordinal(descriptor_set,
                                         rhs_literal_descriptor_ref) !=
          LOOM_LOW_DESCRIPTOR_ORDINAL_NONE) {
    loom_named_attr_t attrs[1] = {0};
    iree_host_size_t attr_count = 0;
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_append_i64_attr(context, IREE_SV("imm32"), immediate, attrs,
                                    IREE_ARRAYSIZE(attrs), &attr_count));
    const loom_value_id_t operands[] = {value};
    loom_op_t* low_op = NULL;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
        context, source_op, rhs_literal_descriptor_ref, operands,
        IREE_ARRAYSIZE(operands), loom_make_named_attr_slice(attrs, attr_count),
        &lane_type, 1, &low_op));
    *out_value = loom_value_slice_get(loom_low_op_results(low_op), 0);
    return iree_ok_status();
  }

  loom_value_id_t low_immediate = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32, immediate,
      lane_type, &low_immediate));
  return loom_amdgpu_emit_sgpr_binary(context, source_op, descriptor_ref, value,
                                      low_immediate, lane_type, out_value);
}

iree_status_t loom_amdgpu_emit_sgpr_scale_u32(loom_low_lower_context_t* context,
                                              const loom_op_t* source_op,
                                              loom_value_id_t value,
                                              uint32_t scale,
                                              loom_type_t lane_type,
                                              loom_value_id_t* out_value) {
  *out_value = LOOM_VALUE_ID_INVALID;
  if (scale == 0) {
    return loom_amdgpu_emit_const_u32(context, source_op,
                                      LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32, 0,
                                      lane_type, out_value);
  }
  if (scale == 1) {
    *out_value = value;
    return iree_ok_status();
  }
  if (loom_amdgpu_u32_is_power_of_two(scale)) {
    uint32_t shift = 0;
    uint32_t remaining = scale;
    while (remaining > 1u) {
      remaining >>= 1u;
      ++shift;
    }
    return loom_amdgpu_emit_sgpr_binary_immediate(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_LSHL_B32, value, shift,
        lane_type, out_value);
  }

  return loom_amdgpu_emit_sgpr_binary_immediate(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_MUL_I32, value, scale,
      lane_type, out_value);
}

iree_status_t loom_amdgpu_emit_sgpr_scaled_add_u32(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t value, uint32_t scale, loom_value_id_t addend,
    loom_type_t lane_type, loom_value_id_t* out_value) {
  *out_value = LOOM_VALUE_ID_INVALID;
  loom_amdgpu_descriptor_ref_t fused_descriptor_ref =
      LOOM_AMDGPU_DESCRIPTOR_REF_NONE;
  switch (scale) {
    case 2:
      fused_descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_S_LSHL1_ADD_U32;
      break;
    case 4:
      fused_descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_S_LSHL2_ADD_U32;
      break;
    case 8:
      fused_descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_S_LSHL3_ADD_U32;
      break;
    case 16:
      fused_descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_S_LSHL4_ADD_U32;
      break;
    default:
      break;
  }
  const loom_low_descriptor_set_t* descriptor_set =
      loom_low_lower_context_descriptor_set(context);
  if (fused_descriptor_ref != LOOM_AMDGPU_DESCRIPTOR_REF_NONE &&
      loom_amdgpu_descriptor_ref_ordinal(descriptor_set,
                                         fused_descriptor_ref) !=
          LOOM_LOW_DESCRIPTOR_ORDINAL_NONE) {
    return loom_amdgpu_emit_sgpr_binary(context, source_op,
                                        fused_descriptor_ref, value, addend,
                                        lane_type, out_value);
  }

  loom_value_id_t scaled_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_sgpr_scale_u32(
      context, source_op, value, scale, lane_type, &scaled_value));
  return loom_amdgpu_emit_sgpr_binary(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_ADD_U32, scaled_value,
      addend, lane_type, out_value);
}

iree_status_t loom_amdgpu_emit_sgpr64_from_u32(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_value, loom_value_id_t* out_low_wide_value) {
  *out_low_wide_value = LOOM_VALUE_ID_INVALID;

  loom_type_t sgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_sgpr_type(context, &sgpr_type));
  loom_value_id_t low_zero = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32, 0, sgpr_type,
      &low_zero));

  loom_type_t sgpr_x2_type = loom_type_none();
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_make_sgpr_range_type(context, 2, &sgpr_x2_type));
  loom_value_id_t sources[] = {
      low_value,
      low_zero,
  };
  loom_op_t* concat_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_concat_build(
      loom_low_lower_context_builder(context), sources, IREE_ARRAYSIZE(sources),
      sgpr_x2_type, source_op->location, &concat_op));
  *out_low_wide_value = loom_low_concat_result(concat_op);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_sgpr64_constant_u32(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    uint32_t value, loom_value_id_t* out_low_wide_value) {
  *out_low_wide_value = LOOM_VALUE_ID_INVALID;
  loom_type_t sgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_sgpr_type(context, &sgpr_type));
  loom_value_id_t low_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32, value,
      sgpr_type, &low_value));
  return loom_amdgpu_emit_sgpr64_from_u32(context, source_op, low_value,
                                          out_low_wide_value);
}

iree_status_t loom_amdgpu_emit_sgpr64_constant_u64(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    uint64_t value, loom_value_id_t* out_low_wide_value) {
  *out_low_wide_value = LOOM_VALUE_ID_INVALID;
  if (value <= UINT32_MAX) {
    return loom_amdgpu_emit_sgpr64_constant_u32(
        context, source_op, (uint32_t)value, out_low_wide_value);
  }

  loom_type_t sgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_sgpr_type(context, &sgpr_type));
  loom_value_id_t low_value_lo = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32, (uint32_t)value,
      sgpr_type, &low_value_lo));
  loom_value_id_t low_value_hi = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32,
      (uint32_t)(value >> 32), sgpr_type, &low_value_hi));

  loom_type_t sgpr_x2_type = loom_type_none();
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_make_sgpr_range_type(context, 2, &sgpr_x2_type));
  loom_value_id_t sources[] = {
      low_value_lo,
      low_value_hi,
  };
  loom_op_t* concat_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_concat_build(
      loom_low_lower_context_builder(context), sources, IREE_ARRAYSIZE(sources),
      sgpr_x2_type, source_op->location, &concat_op));
  *out_low_wide_value = loom_low_concat_result(concat_op);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_lane_mask_low32(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_mask, loom_value_id_t* out_low_mask32) {
  *out_low_mask32 = LOOM_VALUE_ID_INVALID;
  loom_type_t sgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_sgpr_type(context, &sgpr_type));
  return loom_amdgpu_emit_low_slice(context, source_op, low_mask, /*offset=*/0,
                                    sgpr_type, out_low_mask32);
}

static iree_status_t loom_amdgpu_emit_sgpr32_nonzero_scc(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_value, loom_value_id_t* out_low_scc) {
  *out_low_scc = LOOM_VALUE_ID_INVALID;

  loom_type_t scc_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_scc_type(context, &scc_type));

  loom_named_attr_t attrs[1] = {0};
  iree_host_size_t attr_count = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_i64_attr(
      context, IREE_SV("rhs"), 0, attrs, IREE_ARRAYSIZE(attrs), &attr_count));
  const loom_value_id_t operands[] = {low_value};
  loom_op_t* compare_op = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_CMP_LG_I32_SRC1_INLINE,
      operands, IREE_ARRAYSIZE(operands),
      loom_make_named_attr_slice(attrs, attr_count), &scc_type, 1,
      &compare_op));
  *out_low_scc = loom_value_slice_get(loom_low_op_results(compare_op), 0);
  return iree_ok_status();
}

iree_status_t loom_amdgpu_emit_lane_mask_nonzero_scc(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_mask, uint32_t wavefront_size,
    loom_value_id_t* out_low_scc) {
  *out_low_scc = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT(loom_amdgpu_wavefront_size_is_valid(wavefront_size));

  if (wavefront_size == 32) {
    loom_value_id_t low_mask32 = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_lane_mask_low32(
        context, source_op, low_mask, &low_mask32));
    return loom_amdgpu_emit_sgpr32_nonzero_scc(context, source_op, low_mask32,
                                               out_low_scc);
  }
  IREE_ASSERT_EQ(wavefront_size, 64u);

  loom_type_t scc_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_scc_type(context, &scc_type));
  loom_value_id_t low_zero64 = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_emit_sgpr64_constant_u64(context, source_op, 0, &low_zero64));
  const loom_value_id_t operands[] = {low_mask, low_zero64};
  loom_op_t* compare_op = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_CMP_LG_U64, operands,
      IREE_ARRAYSIZE(operands), loom_named_attr_slice_empty(), &scc_type, 1,
      &compare_op));
  *out_low_scc = loom_value_slice_get(loom_low_op_results(compare_op), 0);
  return iree_ok_status();
}

iree_status_t loom_amdgpu_emit_lane_mask_equal_scc(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_lhs, loom_value_id_t low_rhs, uint32_t wavefront_size,
    loom_value_id_t* out_low_scc) {
  *out_low_scc = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT(loom_amdgpu_wavefront_size_is_valid(wavefront_size));

  loom_type_t scc_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_scc_type(context, &scc_type));
  if (wavefront_size == 32) {
    loom_value_id_t low_lhs32 = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_lane_mask_low32(context, source_op,
                                                          low_lhs, &low_lhs32));
    loom_value_id_t low_rhs32 = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_lane_mask_low32(context, source_op,
                                                          low_rhs, &low_rhs32));

    const loom_value_id_t operands[] = {low_lhs32, low_rhs32};
    loom_op_t* compare_op = NULL;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_CMP_EQ_I32, operands,
        IREE_ARRAYSIZE(operands), loom_named_attr_slice_empty(), &scc_type, 1,
        &compare_op));
    *out_low_scc = loom_value_slice_get(loom_low_op_results(compare_op), 0);
    return iree_ok_status();
  }
  IREE_ASSERT_EQ(wavefront_size, 64u);

  const loom_value_id_t operands[] = {low_lhs, low_rhs};
  loom_op_t* compare_op = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_CMP_EQ_U64, operands,
      IREE_ARRAYSIZE(operands), loom_named_attr_slice_empty(), &scc_type, 1,
      &compare_op));
  *out_low_scc = loom_value_slice_get(loom_low_op_results(compare_op), 0);
  return iree_ok_status();
}

iree_status_t loom_amdgpu_emit_sgpr64_add(loom_low_lower_context_t* context,
                                          const loom_op_t* source_op,
                                          loom_value_id_t low_lhs,
                                          loom_value_id_t low_rhs,
                                          loom_value_id_t* out_low_sum) {
  *out_low_sum = LOOM_VALUE_ID_INVALID;

  loom_type_t sgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_sgpr_type(context, &sgpr_type));
  loom_value_id_t low_lhs_lo = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(
      context, source_op, low_lhs, /*offset=*/0, sgpr_type, &low_lhs_lo));
  loom_value_id_t low_lhs_hi = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(
      context, source_op, low_lhs, /*offset=*/1, sgpr_type, &low_lhs_hi));
  loom_value_id_t low_rhs_lo = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(
      context, source_op, low_rhs, /*offset=*/0, sgpr_type, &low_rhs_lo));
  loom_value_id_t low_rhs_hi = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(
      context, source_op, low_rhs, /*offset=*/1, sgpr_type, &low_rhs_hi));

  loom_value_id_t low_add_lo_operands[] = {
      low_lhs_lo,
      low_rhs_lo,
  };
  loom_op_t* low_add_lo_op = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_ADD_U32,
      low_add_lo_operands, IREE_ARRAYSIZE(low_add_lo_operands),
      loom_make_named_attr_slice(NULL, 0), &sgpr_type, 1, &low_add_lo_op));
  const loom_value_id_t low_sum_lo =
      loom_value_slice_get(loom_low_op_results(low_add_lo_op), 0);

  loom_value_id_t low_add_hi_operands[] = {
      low_lhs_hi,
      low_rhs_hi,
  };
  loom_op_t* low_add_hi_op = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_ADDC_U32,
      low_add_hi_operands, IREE_ARRAYSIZE(low_add_hi_operands),
      loom_make_named_attr_slice(NULL, 0), &sgpr_type, 1, &low_add_hi_op));
  const loom_value_id_t low_sum_hi =
      loom_value_slice_get(loom_low_op_results(low_add_hi_op), 0);

  loom_type_t sgpr_x2_type = loom_type_none();
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_make_sgpr_range_type(context, 2, &sgpr_x2_type));
  loom_value_id_t sources[] = {
      low_sum_lo,
      low_sum_hi,
  };
  loom_op_t* concat_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_concat_build(
      loom_low_lower_context_builder(context), sources, IREE_ARRAYSIZE(sources),
      sgpr_x2_type, source_op->location, &concat_op));
  *out_low_sum = loom_low_concat_result(concat_op);
  return iree_ok_status();
}

iree_status_t loom_amdgpu_emit_sgpr64_add_u32_offset(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_base, loom_value_id_t low_offset,
    loom_value_id_t* out_low_sum) {
  *out_low_sum = LOOM_VALUE_ID_INVALID;

  loom_type_t sgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_sgpr_type(context, &sgpr_type));
  loom_value_id_t low_base_lo = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(
      context, source_op, low_base, /*offset=*/0, sgpr_type, &low_base_lo));
  loom_value_id_t low_base_hi = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(
      context, source_op, low_base, /*offset=*/1, sgpr_type, &low_base_hi));
  loom_value_id_t low_zero = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32, 0, sgpr_type,
      &low_zero));

  loom_value_id_t low_add_lo_operands[] = {
      low_base_lo,
      low_offset,
  };
  loom_op_t* low_add_lo_op = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_ADD_U32,
      low_add_lo_operands, IREE_ARRAYSIZE(low_add_lo_operands),
      loom_make_named_attr_slice(NULL, 0), &sgpr_type, 1, &low_add_lo_op));
  const loom_value_id_t low_sum_lo =
      loom_value_slice_get(loom_low_op_results(low_add_lo_op), 0);

  loom_value_id_t low_add_hi_operands[] = {
      low_base_hi,
      low_zero,
  };
  loom_op_t* low_add_hi_op = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_ADDC_U32,
      low_add_hi_operands, IREE_ARRAYSIZE(low_add_hi_operands),
      loom_make_named_attr_slice(NULL, 0), &sgpr_type, 1, &low_add_hi_op));
  const loom_value_id_t low_sum_hi =
      loom_value_slice_get(loom_low_op_results(low_add_hi_op), 0);

  loom_type_t sgpr_x2_type = loom_type_none();
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_make_sgpr_range_type(context, 2, &sgpr_x2_type));
  loom_value_id_t sources[] = {
      low_sum_lo,
      low_sum_hi,
  };
  loom_op_t* concat_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_concat_build(
      loom_low_lower_context_builder(context), sources, IREE_ARRAYSIZE(sources),
      sgpr_x2_type, source_op->location, &concat_op));
  *out_low_sum = loom_low_concat_result(concat_op);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_sgpr_scale_byte_offset_u32(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_unscaled_offset, int64_t byte_stride,
    uint32_t byte_shift, loom_type_t sgpr_type,
    loom_value_id_t* out_low_offset) {
  *out_low_offset = low_unscaled_offset;
  if (byte_stride == 1) {
    return iree_ok_status();
  }
  IREE_ASSERT(byte_stride >= 0 && byte_stride <= UINT32_MAX);
  if (byte_shift != LOOM_LOW_SOURCE_MEMORY_ACCESS_BYTE_SHIFT_NONE) {
    return loom_amdgpu_emit_sgpr_binary_immediate(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_LSHL_B32,
        low_unscaled_offset, byte_shift, sgpr_type, out_low_offset);
  }

  return loom_amdgpu_emit_sgpr_scale_u32(
      context, source_op, low_unscaled_offset, (uint32_t)byte_stride, sgpr_type,
      out_low_offset);
}

static iree_status_t loom_amdgpu_fit_memory_u32_soffset_term_operand(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_low_source_memory_dynamic_term_t* term,
    loom_value_id_t low_value, loom_value_id_t* out_low_value) {
  *out_low_value = low_value;
  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_type_t low_type = loom_module_value_type(module, low_value);
  IREE_ASSERT(loom_amdgpu_low_type_is_register_class(
      context, low_type, LOOM_AMDGPU_REG_CLASS_ID_SGPR));
  const uint32_t unit_count = loom_low_register_type_unit_count(low_type);
  if (unit_count == 1) {
    return iree_ok_status();
  }
  IREE_ASSERT_EQ(unit_count, 2u);
  IREE_ASSERT(
      loom_low_source_memory_dynamic_term_fits_unsigned_bit_count(term, 32));
  loom_type_t sgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_sgpr_type(context, &sgpr_type));
  return loom_amdgpu_emit_low_slice(context, source_op, low_value,
                                    /*offset=*/0, sgpr_type, out_low_value);
}

iree_status_t loom_amdgpu_emit_sgpr_byte_offset_term(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_low_source_memory_dynamic_term_t* term,
    loom_value_id_t* out_low_offset) {
  *out_low_offset = LOOM_VALUE_ID_INVALID;
  loom_type_t sgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_sgpr_type(context, &sgpr_type));

  loom_value_id_t low_offset = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, term->index, &low_offset));
  IREE_RETURN_IF_ERROR(loom_amdgpu_fit_memory_u32_soffset_term_operand(
      context, source_op, term, low_offset, &low_offset));
  for (uint8_t i = 0; i < term->stride_value_count; ++i) {
    loom_value_id_t low_stride = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(
        context, term->stride_values[i], &low_stride));
    IREE_RETURN_IF_ERROR(loom_amdgpu_fit_memory_u32_soffset_term_operand(
        context, source_op, term, low_stride, &low_stride));
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_sgpr_mul_u32(
        context, source_op, low_offset, low_stride, sgpr_type, &low_offset));
  }
  return loom_amdgpu_emit_sgpr_scale_byte_offset_u32(
      context, source_op, low_offset, term->byte_stride, term->byte_shift,
      sgpr_type, out_low_offset);
}

iree_status_t loom_amdgpu_emit_sgpr_byte_offset(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t dynamic_index, int64_t dynamic_index_byte_stride,
    uint32_t dynamic_index_byte_shift, uint32_t static_byte_offset,
    loom_value_id_t* out_low_offset) {
  *out_low_offset = LOOM_VALUE_ID_INVALID;
  loom_type_t sgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_sgpr_type(context, &sgpr_type));
  if (dynamic_index == LOOM_VALUE_ID_INVALID) {
    return loom_amdgpu_emit_const_u32(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32,
        static_byte_offset, sgpr_type, out_low_offset);
  }

  loom_value_id_t low_index = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, dynamic_index, &low_index));
  loom_value_id_t low_dynamic_offset = low_index;
  if (dynamic_index_byte_stride != 1) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_sgpr_scale_byte_offset_u32(
        context, source_op, low_index, dynamic_index_byte_stride,
        dynamic_index_byte_shift, sgpr_type, &low_dynamic_offset));
  }

  if (static_byte_offset == 0) {
    *out_low_offset = low_dynamic_offset;
    return iree_ok_status();
  }

  loom_value_id_t low_static_offset = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32,
      static_byte_offset, sgpr_type, &low_static_offset));
  return loom_amdgpu_emit_sgpr_binary(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_ADD_U32,
      low_dynamic_offset, low_static_offset, sgpr_type, out_low_offset);
}

iree_status_t loom_amdgpu_emit_sgpr_byte_offset_terms(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_low_source_memory_access_plan_t* source,
    const loom_amdgpu_memory_dynamic_index_kind_t* dynamic_term_kinds,
    uint32_t static_byte_offset, loom_value_id_t* out_low_offset) {
  *out_low_offset = LOOM_VALUE_ID_INVALID;

  loom_type_t sgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_sgpr_type(context, &sgpr_type));

  loom_amdgpu_memory_dynamic_term_sequence_t sequence = {0};
  loom_amdgpu_memory_access_resolve_dynamic_terms(
      context, source, dynamic_term_kinds, &sequence);
  loom_value_id_t low_accumulator = LOOM_VALUE_ID_INVALID;
  for (uint8_t i = 0; i < sequence.count; ++i) {
    switch (sequence.kinds[i]) {
      case LOOM_AMDGPU_MEMORY_DYNAMIC_INDEX_SOFFSET:
        break;
      case LOOM_AMDGPU_MEMORY_DYNAMIC_INDEX_VADDR:
        continue;
      case LOOM_AMDGPU_MEMORY_DYNAMIC_INDEX_NONE:
        IREE_ASSERT_UNREACHABLE("unknown AMDGPU memory dynamic index kind");
        IREE_BUILTIN_UNREACHABLE();
    }
    const loom_low_source_memory_dynamic_term_t* term = sequence.terms[i];
    loom_value_id_t low_term = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_sgpr_byte_offset_term(
        context, source_op, term, &low_term));
    if (low_accumulator == LOOM_VALUE_ID_INVALID) {
      low_accumulator = low_term;
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_sgpr_binary(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_ADD_U32,
        low_accumulator, low_term, sgpr_type, &low_accumulator));
  }

  if (low_accumulator == LOOM_VALUE_ID_INVALID) {
    return loom_amdgpu_emit_sgpr_byte_offset(
        context, source_op, LOOM_VALUE_ID_INVALID,
        /*dynamic_index_byte_stride=*/1,
        LOOM_LOW_SOURCE_MEMORY_ACCESS_BYTE_SHIFT_NONE, static_byte_offset,
        out_low_offset);
  }
  if (static_byte_offset == 0) {
    *out_low_offset = low_accumulator;
    return iree_ok_status();
  }

  loom_value_id_t low_static_offset = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32,
      static_byte_offset, sgpr_type, &low_static_offset));
  return loom_amdgpu_emit_sgpr_binary(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_ADD_U32, low_accumulator,
      low_static_offset, sgpr_type, out_low_offset);
}

iree_status_t loom_amdgpu_low_result_type(loom_low_lower_context_t* context,
                                          const loom_op_t* source_op,
                                          loom_value_id_t source_result,
                                          loom_type_t* out_low_type) {
  *out_low_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_lower_map_value(context, source_op,
                                                source_result, out_low_type));
  IREE_ASSERT(loom_low_type_is_register(*out_low_type),
              "AMDGPU selected lowering produced non-register result type");
  return iree_ok_status();
}

iree_status_t loom_amdgpu_build_low_register_range(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_value_id_t* low_registers, uint32_t register_count,
    loom_type_t result_type, loom_value_id_t* out_low_result) {
  IREE_ASSERT_GT(register_count, 0);
  IREE_ASSERT_LE(register_count, LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES);
  *out_low_result = LOOM_VALUE_ID_INVALID;
  if (register_count == 1) {
    *out_low_result = low_registers[0];
    return iree_ok_status();
  }

  loom_op_t* concat_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_concat_build(
      loom_low_lower_context_builder(context), low_registers, register_count,
      result_type, source_op->location, &concat_op));
  *out_low_result = loom_low_concat_result(concat_op);
  return iree_ok_status();
}

iree_status_t loom_amdgpu_extract_low_register_unit(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_source, uint32_t register_count,
    uint32_t register_offset, loom_type_t unit_type,
    loom_value_id_t* out_register_unit) {
  IREE_ASSERT_GT(register_count, 0);
  IREE_ASSERT_LT(register_offset, register_count);
  *out_register_unit = LOOM_VALUE_ID_INVALID;
  if (register_count == 1) {
    *out_register_unit = low_source;
    return iree_ok_status();
  }
  return loom_amdgpu_emit_low_slice(context, source_op, low_source,
                                    register_offset, unit_type,
                                    out_register_unit);
}

iree_status_t loom_amdgpu_bind_low_register_range(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source_result, const loom_value_id_t* low_registers,
    uint32_t register_count) {
  if (register_count == 1) {
    return loom_low_lower_bind_value(context, source_result, low_registers[0]);
  }

  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_result_type(
      context, source_op, source_result, &result_type));
  loom_value_id_t low_result = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_low_register_range(
      context, source_op, low_registers, register_count, result_type,
      &low_result));
  return loom_low_lower_bind_value(context, source_result, low_result);
}

iree_status_t loom_amdgpu_resolve_descriptor_ref_if_present(
    loom_low_lower_context_t* context,
    loom_amdgpu_descriptor_ref_t descriptor_ref,
    loom_low_lower_resolved_descriptor_t* out_descriptor, bool* out_present) {
  *out_descriptor = (loom_low_lower_resolved_descriptor_t){0};
  *out_present = false;
  const loom_low_descriptor_set_t* descriptor_set =
      loom_low_lower_context_descriptor_set(context);
  const uint32_t descriptor_ordinal =
      loom_amdgpu_descriptor_ref_ordinal(descriptor_set, descriptor_ref);
  if (descriptor_ordinal == LOOM_LOW_DESCRIPTOR_ORDINAL_NONE) {
    return iree_ok_status();
  }
  const loom_low_descriptor_t* descriptor =
      loom_low_descriptor_set_descriptor_at(descriptor_set, descriptor_ordinal);
  IREE_ASSERT(descriptor != NULL);
  out_descriptor->descriptor = descriptor;
  *out_present = true;
  return iree_ok_status();
}

iree_status_t loom_amdgpu_resolve_descriptor_refs_if_present(
    loom_low_lower_context_t* context,
    const loom_amdgpu_descriptor_resolution_t* resolutions,
    iree_host_size_t resolution_count, bool* out_present) {
  *out_present = false;
  for (iree_host_size_t i = 0; i < resolution_count; ++i) {
    bool present = false;
    IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref_if_present(
        context, resolutions[i].descriptor_ref, resolutions[i].out_descriptor,
        &present));
    if (!present) {
      return iree_ok_status();
    }
  }
  *out_present = true;
  return iree_ok_status();
}

typedef struct loom_amdgpu_cndmask_b32_descriptor_resolution_t {
  // Descriptor flag that requests this form.
  loom_amdgpu_cndmask_b32_descriptor_flags_t flag;
  // Descriptor ref resolved against the active descriptor set.
  loom_amdgpu_descriptor_ref_t descriptor_ref;
  // Offset of the destination field in loom_amdgpu_cndmask_b32_descriptors_t.
  iree_host_size_t destination_offset;
} loom_amdgpu_cndmask_b32_descriptor_resolution_t;

#define LOOM_AMDGPU_CNDMASK_B32_DESCRIPTOR_RESOLUTION(            \
    flag_value, descriptor_ref_value, field)                      \
  {                                                               \
      .flag = flag_value,                                         \
      .descriptor_ref = descriptor_ref_value,                     \
      .destination_offset =                                       \
          offsetof(loom_amdgpu_cndmask_b32_descriptors_t, field), \
  }

static const loom_amdgpu_cndmask_b32_descriptor_resolution_t
    kCndmaskB32DescriptorResolutions[] = {
        LOOM_AMDGPU_CNDMASK_B32_DESCRIPTOR_RESOLUTION(
            LOOM_AMDGPU_CNDMASK_B32_DESCRIPTOR_REGISTER,
            LOOM_AMDGPU_DESCRIPTOR_REF_V_CNDMASK_B32, register_descriptor),
        LOOM_AMDGPU_CNDMASK_B32_DESCRIPTOR_RESOLUTION(
            LOOM_AMDGPU_CNDMASK_B32_DESCRIPTOR_SRC0_INLINE,
            LOOM_AMDGPU_DESCRIPTOR_REF_V_CNDMASK_B32_SRC0_INLINE,
            src0_inline_descriptor),
        LOOM_AMDGPU_CNDMASK_B32_DESCRIPTOR_RESOLUTION(
            LOOM_AMDGPU_CNDMASK_B32_DESCRIPTOR_SRC1_INLINE,
            LOOM_AMDGPU_DESCRIPTOR_REF_V_CNDMASK_B32_SRC1_INLINE,
            src1_inline_descriptor),
        LOOM_AMDGPU_CNDMASK_B32_DESCRIPTOR_RESOLUTION(
            LOOM_AMDGPU_CNDMASK_B32_DESCRIPTOR_SRC0_LITERAL,
            LOOM_AMDGPU_DESCRIPTOR_REF_V_CNDMASK_B32_SRC0_LIT,
            src0_literal_descriptor),
        LOOM_AMDGPU_CNDMASK_B32_DESCRIPTOR_RESOLUTION(
            LOOM_AMDGPU_CNDMASK_B32_DESCRIPTOR_SRC1_LITERAL,
            LOOM_AMDGPU_DESCRIPTOR_REF_V_CNDMASK_B32_SRC1_LIT,
            src1_literal_descriptor),
        LOOM_AMDGPU_CNDMASK_B32_DESCRIPTOR_RESOLUTION(
            LOOM_AMDGPU_CNDMASK_B32_DESCRIPTOR_SRC0_LITERAL_SRC1_INLINE,
            LOOM_AMDGPU_DESCRIPTOR_REF_V_CNDMASK_B32_SRC0_LIT_SRC1_INLINE,
            src0_literal_src1_inline_descriptor),
        LOOM_AMDGPU_CNDMASK_B32_DESCRIPTOR_RESOLUTION(
            LOOM_AMDGPU_CNDMASK_B32_DESCRIPTOR_SRC1_LITERAL_SRC0_INLINE,
            LOOM_AMDGPU_DESCRIPTOR_REF_V_CNDMASK_B32_SRC1_LIT_SRC0_INLINE,
            src1_literal_src0_inline_descriptor),
};

#undef LOOM_AMDGPU_CNDMASK_B32_DESCRIPTOR_RESOLUTION

static iree_status_t loom_amdgpu_resolve_cndmask_b32_descriptor_set(
    loom_low_lower_context_t* context,
    loom_amdgpu_cndmask_b32_descriptor_flags_t requested_flags,
    loom_amdgpu_cndmask_b32_descriptors_t* out_descriptors,
    bool* out_all_present) {
  *out_all_present = true;
  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(kCndmaskB32DescriptorResolutions); ++i) {
    const loom_amdgpu_cndmask_b32_descriptor_resolution_t* resolution =
        &kCndmaskB32DescriptorResolutions[i];
    if (!iree_any_bit_set(requested_flags, resolution->flag)) continue;
    bool present = false;
    loom_low_lower_resolved_descriptor_t* destination =
        (loom_low_lower_resolved_descriptor_t*)((uint8_t*)out_descriptors +
                                                resolution->destination_offset);
    IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref_if_present(
        context, resolution->descriptor_ref, destination, &present));
    *out_all_present = *out_all_present && present;
  }
  return iree_ok_status();
}

iree_status_t loom_amdgpu_resolve_cndmask_b32_descriptors(
    loom_low_lower_context_t* context,
    loom_amdgpu_cndmask_b32_descriptor_flags_t required_flags,
    loom_amdgpu_cndmask_b32_descriptor_flags_t optional_flags,
    loom_amdgpu_cndmask_b32_descriptors_t* out_descriptors, bool* out_present) {
  IREE_ASSERT_EQ(required_flags & ~LOOM_AMDGPU_CNDMASK_B32_DESCRIPTOR_ALL, 0u);
  IREE_ASSERT_EQ(optional_flags & ~LOOM_AMDGPU_CNDMASK_B32_DESCRIPTOR_ALL, 0u);
  *out_descriptors = (loom_amdgpu_cndmask_b32_descriptors_t){0};
  *out_present = false;

  bool required_present = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_cndmask_b32_descriptor_set(
      context, required_flags, out_descriptors, &required_present));
  if (!required_present) {
    return iree_ok_status();
  }

  bool optional_present = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_cndmask_b32_descriptor_set(
      context, optional_flags & ~required_flags, out_descriptors,
      &optional_present));
  (void)optional_present;
  *out_present = true;
  return iree_ok_status();
}

iree_status_t loom_amdgpu_resolve_descriptor_ref(
    loom_low_lower_context_t* context,
    loom_amdgpu_descriptor_ref_t descriptor_ref,
    loom_low_lower_resolved_descriptor_t* out_descriptor) {
  bool present = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref_if_present(
      context, descriptor_ref, out_descriptor, &present));
  IREE_ASSERT(present,
              "generated AMDGPU lowering policy references missing descriptor "
              "ref");
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_populate_explicit_packet_plan(
    loom_low_lower_context_t* context,
    const loom_low_lower_resolved_descriptor_t* descriptor,
    const loom_amdgpu_explicit_packet_immediate_template_t* immediates,
    iree_host_size_t immediate_count,
    loom_amdgpu_explicit_packet_plan_t* out_plan) {
  IREE_ASSERT(immediate_count <=
              LOOM_AMDGPU_EXPLICIT_PACKET_IMMEDIATE_CAPACITY);
  IREE_ASSERT(immediate_count == 0 || immediates != NULL);

  *out_plan = (loom_amdgpu_explicit_packet_plan_t){
      .descriptor = *descriptor,
      .immediate_count = immediate_count,
  };
  for (iree_host_size_t i = 0; i < immediate_count; ++i) {
    loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_intern(context, immediates[i].name, &name_id));
    out_plan->immediates[i] = (loom_amdgpu_explicit_packet_immediate_t){
        .name_id = name_id,
        .value = immediates[i].value,
    };
  }
  return iree_ok_status();
}

iree_status_t loom_amdgpu_resolve_explicit_packet_plan(
    loom_low_lower_context_t* context,
    loom_amdgpu_descriptor_ref_t descriptor_ref,
    const loom_amdgpu_explicit_packet_immediate_template_t* immediates,
    iree_host_size_t immediate_count,
    loom_amdgpu_explicit_packet_plan_t* out_plan, bool* out_present) {
  *out_plan = (loom_amdgpu_explicit_packet_plan_t){0};
  *out_present = false;

  loom_low_lower_resolved_descriptor_t descriptor = {0};
  bool present = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref_if_present(
      context, descriptor_ref, &descriptor, &present));
  if (!present) {
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(loom_amdgpu_populate_explicit_packet_plan(
      context, &descriptor, immediates, immediate_count, out_plan));
  *out_present = true;
  return iree_ok_status();
}

iree_status_t loom_amdgpu_resolve_explicit_packet_row_plan(
    loom_low_lower_context_t* context, const loom_low_descriptor_t* descriptor,
    const loom_amdgpu_explicit_packet_immediate_template_t* immediates,
    iree_host_size_t immediate_count,
    loom_amdgpu_explicit_packet_plan_t* out_plan) {
  IREE_ASSERT(immediate_count <=
              LOOM_AMDGPU_EXPLICIT_PACKET_IMMEDIATE_CAPACITY);
  IREE_ASSERT(immediate_count == 0 || immediates != NULL);
  *out_plan = (loom_amdgpu_explicit_packet_plan_t){0};

  loom_low_lower_resolved_descriptor_t resolved_descriptor = {
      .descriptor = descriptor,
  };

  return loom_amdgpu_populate_explicit_packet_plan(
      context, &resolved_descriptor, immediates, immediate_count, out_plan);
}

iree_status_t loom_amdgpu_emit_low_op(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_descriptor_ref_t descriptor_ref,
    const loom_value_id_t* operands, iree_host_size_t operand_count,
    loom_named_attr_slice_t attrs, const loom_type_t* result_types,
    iree_host_size_t result_count, loom_op_t** out_op) {
  *out_op = NULL;
  loom_low_lower_resolved_descriptor_t descriptor = {0};
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_resolve_descriptor_ref(context, descriptor_ref, &descriptor));
  return loom_low_lower_emit_resolved_descriptor_op(
      context, &descriptor, operands, operand_count, attrs, result_types,
      result_count, /*tied_results=*/NULL, /*tied_result_count=*/0,
      source_op->location, out_op);
}

iree_status_t loom_amdgpu_emit_explicit_packet_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_explicit_packet_plan_t* plan) {
  if (plan->descriptor.descriptor == NULL) {
    return iree_ok_status();
  }

  loom_named_attr_t attrs[LOOM_AMDGPU_EXPLICIT_PACKET_IMMEDIATE_CAPACITY] = {0};
  for (iree_host_size_t i = 0; i < plan->immediate_count; ++i) {
    attrs[i].name_id = plan->immediates[i].name_id;
    attrs[i].value = loom_attr_i64(plan->immediates[i].value);
  }

  loom_op_t* low_op = NULL;
  return loom_low_lower_emit_resolved_descriptor_op(
      context, &plan->descriptor, /*operands=*/NULL, /*operand_count=*/0,
      loom_make_named_attr_slice(attrs, plan->immediate_count),
      /*result_types=*/NULL, /*result_count=*/0, /*tied_results=*/NULL,
      /*tied_result_count=*/0, source_op->location, &low_op);
}

iree_status_t loom_amdgpu_emit_const_u32(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_descriptor_ref_t descriptor_ref, uint32_t value,
    loom_type_t result_type, loom_value_id_t* out_value_id) {
  *out_value_id = LOOM_VALUE_ID_INVALID;
  loom_string_id_t value_name_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_intern(context, IREE_SV("imm32"), &value_name_id));
  loom_low_lower_resolved_descriptor_t descriptor = {0};
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_resolve_descriptor_ref(context, descriptor_ref, &descriptor));
  return loom_amdgpu_emit_resolved_const_u32(context, source_op, &descriptor,
                                             value_name_id, value, result_type,
                                             out_value_id);
}

iree_status_t loom_amdgpu_emit_resolved_const_u32(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_low_lower_resolved_descriptor_t* descriptor,
    loom_string_id_t imm32_attr_name_id, uint32_t value,
    loom_type_t result_type, loom_value_id_t* out_value_id) {
  IREE_ASSERT(imm32_attr_name_id != LOOM_STRING_ID_INVALID);
  *out_value_id = LOOM_VALUE_ID_INVALID;
  loom_named_attr_t attrs[] = {
      {
          .name_id = imm32_attr_name_id,
          .value = loom_attr_i64(value),
      },
  };
  loom_op_t* low_const = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_const(
      context, descriptor,
      loom_make_named_attr_slice(attrs, IREE_ARRAYSIZE(attrs)), result_type,
      source_op->location, &low_const));
  *out_value_id = loom_low_const_result(low_const);
  return iree_ok_status();
}

iree_status_t loom_amdgpu_emit_resolved_vgpr_binary_immediate(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_low_lower_resolved_descriptor_t* descriptor, loom_value_id_t lhs,
    loom_value_id_t rhs, uint32_t immediate, loom_type_t lane_type,
    loom_value_id_t* out_value) {
  IREE_ASSERT(descriptor->descriptor != NULL);
  *out_value = LOOM_VALUE_ID_INVALID;
  loom_named_attr_t attrs[1] = {0};
  iree_host_size_t attr_count = 0;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_append_i64_attr(context, IREE_SV("imm32"), immediate, attrs,
                                  IREE_ARRAYSIZE(attrs), &attr_count));
  const loom_value_id_t operands[] = {lhs, rhs};
  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
      context, descriptor, operands, IREE_ARRAYSIZE(operands),
      loom_make_named_attr_slice(attrs, attr_count), &lane_type, 1,
      /*tied_results=*/NULL, /*tied_result_count=*/0, source_op->location,
      &low_op));
  *out_value = loom_value_slice_get(loom_low_op_results(low_op), 0);
  return iree_ok_status();
}

iree_status_t loom_amdgpu_emit_resolved_vgpr_unary(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_low_lower_resolved_descriptor_t* descriptor,
    loom_value_id_t value, loom_type_t lane_type, loom_value_id_t* out_value) {
  IREE_ASSERT(descriptor->descriptor != NULL);
  *out_value = LOOM_VALUE_ID_INVALID;
  const loom_value_id_t operands[] = {value};
  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
      context, descriptor, operands, IREE_ARRAYSIZE(operands),
      loom_make_named_attr_slice(NULL, 0), &lane_type, 1,
      /*tied_results=*/NULL, /*tied_result_count=*/0, source_op->location,
      &low_op));
  *out_value = loom_value_slice_get(loom_low_op_results(low_op), 0);
  return iree_ok_status();
}

iree_status_t loom_amdgpu_emit_resolved_vgpr_binary(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_low_lower_resolved_descriptor_t* descriptor, loom_value_id_t lhs,
    loom_value_id_t rhs, loom_type_t lane_type, loom_value_id_t* out_value) {
  IREE_ASSERT(descriptor->descriptor != NULL);
  *out_value = LOOM_VALUE_ID_INVALID;
  const loom_value_id_t operands[] = {lhs, rhs};
  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
      context, descriptor, operands, IREE_ARRAYSIZE(operands),
      loom_named_attr_slice_empty(), &lane_type, 1,
      /*tied_results=*/NULL, /*tied_result_count=*/0, source_op->location,
      &low_op));
  *out_value = loom_value_slice_get(loom_low_op_results(low_op), 0);
  return iree_ok_status();
}

iree_status_t loom_amdgpu_emit_resolved_vgpr_ternary(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_low_lower_resolved_descriptor_t* descriptor, loom_value_id_t lhs,
    loom_value_id_t rhs, loom_value_id_t third, loom_type_t lane_type,
    loom_value_id_t* out_value) {
  IREE_ASSERT(descriptor->descriptor != NULL);
  *out_value = LOOM_VALUE_ID_INVALID;
  loom_value_id_t operands[] = {lhs, rhs, third};
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_legalize_vop3_scalar_sources(context, source_op, operands));
  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
      context, descriptor, operands, IREE_ARRAYSIZE(operands),
      loom_named_attr_slice_empty(), &lane_type, 1,
      /*tied_results=*/NULL, /*tied_result_count=*/0, source_op->location,
      &low_op));
  *out_value = loom_value_slice_get(loom_low_op_results(low_op), 0);
  return iree_ok_status();
}

iree_status_t loom_amdgpu_legalize_vop3_scalar_sources(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t sources[3]) {
  const loom_low_descriptor_set_t* descriptor_set =
      loom_low_lower_context_descriptor_set(context);
  const loom_amdgpu_descriptor_set_info_t* descriptor_set_info =
      loom_amdgpu_target_info_descriptor_set_at(
          descriptor_set->descriptor_set_ordinal);
  const uint32_t scalar_source_limit =
      loom_amdgpu_descriptor_set_info_has_flags(
          descriptor_set_info,
          LOOM_AMDGPU_DESCRIPTOR_SET_INFO_FLAG_VOP3_TWO_SCALAR_SOURCES)
          ? 2u
          : 1u;

  loom_value_id_t original_scalar_sources[3] = {
      LOOM_VALUE_ID_INVALID,
      LOOM_VALUE_ID_INVALID,
      LOOM_VALUE_ID_INVALID,
  };
  loom_value_id_t legal_scalar_sources[3] = {
      LOOM_VALUE_ID_INVALID,
      LOOM_VALUE_ID_INVALID,
      LOOM_VALUE_ID_INVALID,
  };
  uint32_t unique_scalar_source_count = 0;
  uint32_t retained_scalar_source_count = 0;
  const loom_module_t* module = loom_low_lower_context_module(context);
  for (uint32_t i = 0; i < 3; ++i) {
    const loom_type_t source_type = loom_module_value_type(module, sources[i]);
    if (!loom_amdgpu_low_type_is_register_class(
            context, source_type, LOOM_AMDGPU_REG_CLASS_ID_SGPR)) {
      continue;
    }

    uint32_t prior_index = 0;
    for (; prior_index < unique_scalar_source_count; ++prior_index) {
      if (original_scalar_sources[prior_index] == sources[i]) {
        sources[i] = legal_scalar_sources[prior_index];
        break;
      }
    }
    if (prior_index < unique_scalar_source_count) {
      continue;
    }

    original_scalar_sources[unique_scalar_source_count] = sources[i];
    if (retained_scalar_source_count < scalar_source_limit) {
      legal_scalar_sources[unique_scalar_source_count] = sources[i];
      ++retained_scalar_source_count;
    } else {
      IREE_RETURN_IF_ERROR(loom_amdgpu_materialize_low_vgpr_b32_registers(
          context, source_op, sources[i],
          &legal_scalar_sources[unique_scalar_source_count]));
      sources[i] = legal_scalar_sources[unique_scalar_source_count];
    }
    ++unique_scalar_source_count;
  }
  return iree_ok_status();
}

iree_status_t loom_amdgpu_emit_resolved_vgpr_unary_immediate(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_low_lower_resolved_descriptor_t* descriptor,
    loom_value_id_t value, uint32_t immediate, loom_type_t lane_type,
    loom_value_id_t* out_value) {
  IREE_ASSERT(descriptor->descriptor != NULL);
  *out_value = LOOM_VALUE_ID_INVALID;
  loom_named_attr_t attrs[1] = {0};
  iree_host_size_t attr_count = 0;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_append_i64_attr(context, IREE_SV("imm32"), immediate, attrs,
                                  IREE_ARRAYSIZE(attrs), &attr_count));
  const loom_value_id_t operands[] = {value};
  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
      context, descriptor, operands, IREE_ARRAYSIZE(operands),
      loom_make_named_attr_slice(attrs, attr_count), &lane_type, 1,
      /*tied_results=*/NULL, /*tied_result_count=*/0, source_op->location,
      &low_op));
  *out_value = loom_value_slice_get(loom_low_op_results(low_op), 0);
  return iree_ok_status();
}

iree_status_t loom_amdgpu_emit_m0_u32(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_low_lower_resolved_descriptor_t* consumer_descriptor,
    uint32_t value, loom_value_id_t* out_value_id) {
  IREE_ASSERT(consumer_descriptor->descriptor != NULL);
  *out_value_id = LOOM_VALUE_ID_INVALID;
  loom_type_t m0_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_build_descriptor_implicit_resource_type(
      loom_low_lower_context_descriptor_set(context),
      consumer_descriptor->descriptor, &m0_type));
  return loom_amdgpu_emit_const_u32(context, source_op,
                                    LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32_M0_IMM,
                                    value, m0_type, out_value_id);
}

iree_status_t loom_amdgpu_emit_vgpr_b32_copy(loom_low_lower_context_t* context,
                                             const loom_op_t* source_op,
                                             loom_value_id_t low_source,
                                             loom_value_id_t* out_value) {
  *out_value = LOOM_VALUE_ID_INVALID;
  loom_type_t vgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &vgpr_type));
  loom_value_id_t operands[] = {low_source};
  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32_COPY, operands,
      IREE_ARRAYSIZE(operands), loom_make_named_attr_slice(NULL, 0), &vgpr_type,
      1, &low_op));
  *out_value = loom_value_slice_get(loom_low_op_results(low_op), 0);
  return iree_ok_status();
}

bool loom_amdgpu_low_value_defines_vgpr_low16(loom_low_lower_context_t* context,
                                              loom_value_id_t low_value_id) {
  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_value_t* low_value = loom_module_value(module, low_value_id);
  if (low_value == NULL || loom_value_is_block_arg(low_value)) {
    return false;
  }
  const loom_op_t* low_op = loom_value_def_op(low_value);
  if (low_op == NULL || !loom_low_op_isa(low_op)) {
    return false;
  }
  const loom_low_descriptor_set_t* descriptor_set =
      loom_low_lower_context_descriptor_set(context);
  const loom_low_descriptor_t* descriptor =
      &descriptor_set->descriptors[loom_low_op_descriptor(low_op)];
  const uint16_t result_index = loom_value_def_index(low_value);
  if (result_index >= descriptor->result_count) {
    return false;
  }
  const uint32_t operand_index = descriptor->operand_start + result_index;
  if (operand_index >= descriptor_set->operand_count) {
    return false;
  }
  const loom_low_operand_t* result_operand =
      &descriptor_set->operands[operand_index];
  if (result_operand->register_part_id >= descriptor_set->register_part_count) {
    return false;
  }
  const loom_low_register_part_t* register_part =
      &descriptor_set->register_parts[result_operand->register_part_id];
  return register_part->reg_class_id == LOOM_AMDGPU_REG_CLASS_ID_VGPR &&
         register_part->mask == LOOM_AMDGPU_REGISTER_PART_MASK_LOW16;
}

iree_status_t loom_amdgpu_materialize_low_vgpr_b32(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_value, loom_value_id_t* out_low_value) {
  *out_low_value = low_value;
  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_type_t low_type = loom_module_value_type(module, low_value);
  const bool is_vgpr = loom_amdgpu_low_type_is_register_class(
      context, low_type, LOOM_AMDGPU_REG_CLASS_ID_VGPR);
  if (is_vgpr && loom_low_register_type_unit_count(low_type) == 1) {
    return iree_ok_status();
  }
  const bool is_sgpr = loom_amdgpu_low_type_is_register_class(
      context, low_type, LOOM_AMDGPU_REG_CLASS_ID_SGPR);
  if (!is_sgpr || loom_low_register_type_unit_count(low_type) != 1) {
    IREE_ASSERT_UNREACHABLE(
        "AMDGPU scalar VGPR materializer selected a non-scalar low value");
    IREE_BUILTIN_UNREACHABLE();
  }
  return loom_amdgpu_emit_vgpr_b32_copy(context, source_op, low_value,
                                        out_low_value);
}

iree_status_t loom_amdgpu_materialize_low_vgpr_b32_registers(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_value, loom_value_id_t* out_low_value) {
  *out_low_value = low_value;
  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_type_t low_type = loom_module_value_type(module, low_value);
  const uint32_t unit_count = loom_low_register_type_unit_count(low_type);
  const bool is_vgpr = loom_amdgpu_low_type_is_register_class(
      context, low_type, LOOM_AMDGPU_REG_CLASS_ID_VGPR);
  if (is_vgpr && unit_count != 0 &&
      unit_count <= LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES) {
    return iree_ok_status();
  }

  const bool is_sgpr = loom_amdgpu_low_type_is_register_class(
      context, low_type, LOOM_AMDGPU_REG_CLASS_ID_SGPR);
  if (!is_sgpr || unit_count == 0 ||
      unit_count > LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES) {
    IREE_ASSERT_UNREACHABLE(
        "AMDGPU VGPR materializer selected an unsupported low value");
    IREE_BUILTIN_UNREACHABLE();
  }
  if (unit_count == 1) {
    return loom_amdgpu_emit_vgpr_b32_copy(context, source_op, low_value,
                                          out_low_value);
  }

  loom_type_t sgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_sgpr_type(context, &sgpr_type));
  loom_type_t vgpr_range_type = loom_type_none();
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_make_vgpr_range_type(context, unit_count, &vgpr_range_type));

  loom_value_id_t lanes[LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES];
  for (uint32_t i = 0; i < unit_count; ++i) {
    loom_value_id_t sgpr_lane = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(
        context, source_op, low_value, i, sgpr_type, &sgpr_lane));
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_b32_copy(context, source_op,
                                                        sgpr_lane, &lanes[i]));
  }

  loom_op_t* concat_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_concat_build(
      loom_low_lower_context_builder(context), lanes, unit_count,
      vgpr_range_type, source_op->location, &concat_op));
  *out_low_value = loom_low_concat_result(concat_op);
  return iree_ok_status();
}

iree_status_t loom_amdgpu_materialize_full_low_vgpr_b32(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_value, loom_value_id_t* out_low_value) {
  *out_low_value = low_value;
  IREE_RETURN_IF_ERROR(loom_amdgpu_materialize_low_vgpr_b32(
      context, source_op, low_value, out_low_value));
  if (!loom_amdgpu_low_value_defines_vgpr_low16(context, *out_low_value)) {
    return iree_ok_status();
  }

  const loom_type_t low_type = loom_module_value_type(
      loom_low_lower_context_module(context), *out_low_value);
  const loom_value_id_t operands[] = {*out_low_value};
  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
      context, source_op,
      LOOM_AMDGPU_DESCRIPTOR_REF_V_BFE_U32_OFFSET_0_WIDTH_16_LOW16, operands,
      IREE_ARRAYSIZE(operands), loom_make_named_attr_slice(NULL, 0), &low_type,
      1, &low_op));
  *out_low_value = loom_value_slice_get(loom_low_op_results(low_op), 0);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_materialize_full_low_vgpr_b32_registers(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_value, loom_value_id_t* out_low_value) {
  *out_low_value = low_value;
  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_type_t low_type = loom_module_value_type(module, low_value);
  const uint32_t unit_count = loom_low_register_type_unit_count(low_type);
  const bool is_vgpr = loom_amdgpu_low_type_is_register_class(
      context, low_type, LOOM_AMDGPU_REG_CLASS_ID_VGPR);
  if (is_vgpr && unit_count == 1) {
    return loom_amdgpu_materialize_full_low_vgpr_b32(context, source_op,
                                                     low_value, out_low_value);
  }
  return loom_amdgpu_materialize_low_vgpr_b32_registers(
      context, source_op, low_value, out_low_value);
}

iree_status_t loom_amdgpu_materialize_structural_operand(
    void* user_data, loom_low_lower_context_t* context,
    const loom_op_t* source_op, iree_host_size_t operand_index,
    loom_value_id_t source_value_id, loom_value_id_t low_value_id,
    loom_type_t required_low_type, loom_value_id_t* out_low_value_id) {
  (void)user_data;
  (void)operand_index;
  (void)source_value_id;
  *out_low_value_id = low_value_id;

  const bool requires_vgpr = loom_amdgpu_low_type_is_register_class(
      context, required_low_type, LOOM_AMDGPU_REG_CLASS_ID_VGPR);
  if (!requires_vgpr) {
    return iree_ok_status();
  }
  return loom_amdgpu_materialize_full_low_vgpr_b32_registers(
      context, source_op, low_value_id, out_low_value_id);
}

iree_status_t loom_amdgpu_emit_vgpr_binary(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_descriptor_ref_t descriptor_ref, loom_value_id_t lhs,
    loom_value_id_t rhs, loom_type_t lane_type, loom_value_id_t* out_value) {
  *out_value = LOOM_VALUE_ID_INVALID;
  loom_value_id_t operands[] = {
      lhs,
      rhs,
  };
  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
      context, source_op, descriptor_ref, operands, IREE_ARRAYSIZE(operands),
      loom_make_named_attr_slice(NULL, 0), &lane_type, 1, &low_op));
  *out_value = loom_value_slice_get(loom_low_op_results(low_op), 0);
  return iree_ok_status();
}

iree_status_t loom_amdgpu_emit_vgpr_unary(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_descriptor_ref_t descriptor_ref, loom_value_id_t value,
    loom_type_t lane_type, loom_value_id_t* out_value) {
  *out_value = LOOM_VALUE_ID_INVALID;
  const loom_value_id_t operands[] = {value};
  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
      context, source_op, descriptor_ref, operands, IREE_ARRAYSIZE(operands),
      loom_make_named_attr_slice(NULL, 0), &lane_type, 1, &low_op));
  *out_value = loom_value_slice_get(loom_low_op_results(low_op), 0);
  return iree_ok_status();
}

iree_status_t loom_amdgpu_emit_resolved_vgpr_compare_immediate(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_low_lower_resolved_descriptor_t* src1_inline_descriptor,
    loom_amdgpu_descriptor_ref_t descriptor_ref, loom_value_id_t value,
    uint32_t immediate, loom_type_t vgpr_type, loom_type_t mask_type,
    loom_value_id_t* out_mask) {
  *out_mask = LOOM_VALUE_ID_INVALID;
  if (immediate <= LOOM_AMDGPU_SOURCE_INLINE_U32_MAX &&
      src1_inline_descriptor->descriptor != NULL) {
    loom_named_attr_t attrs[1] = {0};
    iree_host_size_t attr_count = 0;
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_append_i64_attr(context, IREE_SV("rhs"), immediate, attrs,
                                    IREE_ARRAYSIZE(attrs), &attr_count));
    const loom_value_id_t operands[] = {value};
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
  const loom_value_id_t operands[] = {value, low_immediate};
  loom_op_t* compare_op = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
      context, source_op, descriptor_ref, operands, IREE_ARRAYSIZE(operands),
      loom_named_attr_slice_empty(), &mask_type, 1, &compare_op));
  *out_mask = loom_value_slice_get(loom_low_op_results(compare_op), 0);
  return iree_ok_status();
}

iree_status_t loom_amdgpu_emit_vgpr_compare_immediate(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_descriptor_ref_t descriptor_ref,
    loom_amdgpu_descriptor_ref_t src1_inline_descriptor_ref,
    loom_value_id_t value, uint32_t immediate, loom_type_t vgpr_type,
    loom_type_t mask_type, loom_value_id_t* out_mask) {
  loom_low_lower_resolved_descriptor_t src1_inline_descriptor = {0};
  if (immediate <= LOOM_AMDGPU_SOURCE_INLINE_U32_MAX &&
      src1_inline_descriptor_ref != LOOM_AMDGPU_DESCRIPTOR_REF_NONE) {
    bool present = false;
    IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref_if_present(
        context, src1_inline_descriptor_ref, &src1_inline_descriptor,
        &present));
  }
  return loom_amdgpu_emit_resolved_vgpr_compare_immediate(
      context, source_op, &src1_inline_descriptor, descriptor_ref, value,
      immediate, vgpr_type, mask_type, out_mask);
}

iree_status_t loom_amdgpu_emit_vgpr_select(loom_low_lower_context_t* context,
                                           const loom_op_t* source_op,
                                           loom_value_id_t false_value,
                                           loom_value_id_t true_value,
                                           loom_value_id_t condition,
                                           loom_type_t vgpr_type,
                                           loom_value_id_t* out_value) {
  *out_value = LOOM_VALUE_ID_INVALID;
  const loom_value_id_t operands[] = {false_value, true_value, condition};
  loom_op_t* select_op = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_CNDMASK_B32, operands,
      IREE_ARRAYSIZE(operands), loom_named_attr_slice_empty(), &vgpr_type, 1,
      &select_op));
  *out_value = loom_value_slice_get(loom_low_op_results(select_op), 0);
  return iree_ok_status();
}

static loom_amdgpu_descriptor_ref_t
loom_amdgpu_vgpr_binary_src0_inline_descriptor_ref(
    loom_amdgpu_descriptor_ref_t descriptor_ref) {
  switch (descriptor_ref) {
    case LOOM_AMDGPU_DESCRIPTOR_REF_V_ADD_U32_LIT:
      return LOOM_AMDGPU_DESCRIPTOR_REF_V_ADD_U32_SRC0_INLINE;
    case LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT:
      return LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_SRC0_INLINE;
    case LOOM_AMDGPU_DESCRIPTOR_REF_V_OR_B32_LIT:
      return LOOM_AMDGPU_DESCRIPTOR_REF_V_OR_B32_SRC0_INLINE;
    case LOOM_AMDGPU_DESCRIPTOR_REF_V_XOR_B32_LIT:
      return LOOM_AMDGPU_DESCRIPTOR_REF_V_XOR_B32_SRC0_INLINE;
    case LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32_LIT:
      return LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32_SRC0_INLINE;
    case LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHRREV_B32_LIT:
      return LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHRREV_B32_SRC0_INLINE;
    case LOOM_AMDGPU_DESCRIPTOR_REF_V_ASHRREV_I32_LIT:
      return LOOM_AMDGPU_DESCRIPTOR_REF_V_ASHRREV_I32_SRC0_INLINE;
    case LOOM_AMDGPU_DESCRIPTOR_REF_V_MUL_U32_U24_LIT:
      return LOOM_AMDGPU_DESCRIPTOR_REF_V_MUL_U32_U24_SRC0_INLINE;
    default:
      return LOOM_AMDGPU_DESCRIPTOR_REF_NONE;
  }
}

loom_amdgpu_descriptor_ref_t
loom_amdgpu_select_vgpr_binary_immediate_descriptor_ref(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_descriptor_ref_t descriptor_ref, uint32_t immediate) {
  const loom_amdgpu_descriptor_ref_t src0_inline_descriptor_ref =
      loom_amdgpu_vgpr_binary_src0_inline_descriptor_ref(descriptor_ref);
  if (immediate <= LOOM_AMDGPU_SOURCE_INLINE_U32_MAX &&
      loom_amdgpu_descriptor_set_has_ref(descriptor_set,
                                         src0_inline_descriptor_ref)) {
    return src0_inline_descriptor_ref;
  }
  return loom_amdgpu_descriptor_set_has_ref(descriptor_set, descriptor_ref)
             ? descriptor_ref
             : LOOM_AMDGPU_DESCRIPTOR_REF_NONE;
}

bool loom_amdgpu_descriptor_set_can_emit_vgpr_binary_immediate(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_descriptor_ref_t descriptor_ref, uint32_t immediate) {
  return loom_amdgpu_select_vgpr_binary_immediate_descriptor_ref(
             descriptor_set, descriptor_ref, immediate) !=
         LOOM_AMDGPU_DESCRIPTOR_REF_NONE;
}

bool loom_amdgpu_descriptor_set_can_emit_vgpr_compare_immediate(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_descriptor_ref_t descriptor_ref,
    loom_amdgpu_descriptor_ref_t src1_inline_descriptor_ref,
    uint32_t immediate) {
  if (immediate <= LOOM_AMDGPU_SOURCE_INLINE_U32_MAX &&
      loom_amdgpu_descriptor_set_has_ref(descriptor_set,
                                         src1_inline_descriptor_ref)) {
    return true;
  }
  return loom_amdgpu_descriptor_set_has_ref(descriptor_set, descriptor_ref) &&
         loom_amdgpu_descriptor_set_has_ref(
             descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32);
}

iree_status_t loom_amdgpu_emit_vgpr_binary_immediate(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_descriptor_ref_t descriptor_ref, loom_value_id_t value,
    uint32_t immediate, loom_type_t lane_type, loom_value_id_t* out_value) {
  *out_value = LOOM_VALUE_ID_INVALID;
  const loom_low_descriptor_set_t* descriptor_set =
      loom_low_lower_context_descriptor_set(context);
  const loom_amdgpu_descriptor_ref_t selected_descriptor_ref =
      loom_amdgpu_select_vgpr_binary_immediate_descriptor_ref(
          descriptor_set, descriptor_ref, immediate);
  if (selected_descriptor_ref != LOOM_AMDGPU_DESCRIPTOR_REF_NONE) {
    descriptor_ref = selected_descriptor_ref;
  }

  loom_named_attr_t attrs[1] = {0};
  iree_host_size_t attr_count = 0;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_append_i64_attr(context, IREE_SV("imm32"), immediate, attrs,
                                  IREE_ARRAYSIZE(attrs), &attr_count));
  loom_value_id_t operands[] = {value};
  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
      context, source_op, descriptor_ref, operands, IREE_ARRAYSIZE(operands),
      loom_make_named_attr_slice(attrs, attr_count), &lane_type, 1, &low_op));
  *out_value = loom_value_slice_get(loom_low_op_results(low_op), 0);
  return iree_ok_status();
}

iree_status_t loom_amdgpu_emit_vgpr_shift(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_descriptor_ref_t descriptor_ref, uint32_t shift,
    loom_value_id_t value, loom_type_t lane_type, loom_value_id_t* out_value) {
  *out_value = LOOM_VALUE_ID_INVALID;
  if (shift == 0) {
    *out_value = value;
    return iree_ok_status();
  }

  return loom_amdgpu_emit_vgpr_binary_immediate(
      context, source_op, descriptor_ref, value, shift, lane_type, out_value);
}

iree_status_t loom_amdgpu_emit_resolved_vgpr_lshl_add_u32(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_low_lower_resolved_descriptor_t* descriptor,
    loom_value_id_t value, loom_value_id_t addend, uint32_t shift,
    loom_type_t lane_type, loom_value_id_t* out_value) {
  IREE_ASSERT(descriptor->descriptor != NULL);
  *out_value = LOOM_VALUE_ID_INVALID;
  loom_named_attr_t attrs[1] = {0};
  iree_host_size_t attr_count = 0;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_append_i64_attr(context, IREE_SV("shift"), shift, attrs,
                                  IREE_ARRAYSIZE(attrs), &attr_count));
  const loom_value_id_t operands[] = {value, addend};
  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
      context, descriptor, operands, IREE_ARRAYSIZE(operands),
      loom_make_named_attr_slice(attrs, attr_count), &lane_type, 1,
      /*tied_results=*/NULL, /*tied_result_count=*/0, source_op->location,
      &low_op));
  *out_value = loom_value_slice_get(loom_low_op_results(low_op), 0);
  return iree_ok_status();
}

iree_status_t loom_amdgpu_try_emit_vgpr_lshl_add_u32(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t value, loom_value_id_t addend, uint32_t shift,
    loom_type_t lane_type, loom_value_id_t* out_value, bool* out_selected) {
  *out_value = LOOM_VALUE_ID_INVALID;
  *out_selected = false;
  if (shift > 64) {
    return iree_ok_status();
  }

  loom_low_lower_resolved_descriptor_t descriptor = {0};
  bool descriptor_present = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref_if_present(
      context, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHL_ADD_U32_SHIFT_IMM, &descriptor,
      &descriptor_present));
  if (!descriptor_present) {
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_resolved_vgpr_lshl_add_u32(
      context, source_op, &descriptor, value, addend, shift, lane_type,
      out_value));
  *out_selected = true;
  return iree_ok_status();
}

iree_status_t loom_amdgpu_emit_vgpr64_from_u32(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_value, loom_value_id_t* out_low_wide_value) {
  *out_low_wide_value = LOOM_VALUE_ID_INVALID;

  loom_value_id_t low_lo = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_materialize_low_vgpr_b32(
      context, source_op, low_value, &low_lo));

  loom_type_t vgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &vgpr_type));
  loom_value_id_t low_hi = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32, 0, vgpr_type,
      &low_hi));

  loom_type_t vgpr_x2_type = loom_type_none();
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_make_vgpr_range_type(context, 2, &vgpr_x2_type));
  loom_value_id_t sources[] = {
      low_lo,
      low_hi,
  };
  loom_op_t* concat_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_concat_build(
      loom_low_lower_context_builder(context), sources, IREE_ARRAYSIZE(sources),
      vgpr_x2_type, source_op->location, &concat_op));
  *out_low_wide_value = loom_low_concat_result(concat_op);
  return iree_ok_status();
}

iree_status_t loom_amdgpu_emit_vgpr64_add(loom_low_lower_context_t* context,
                                          const loom_op_t* source_op,
                                          loom_value_id_t low_lhs,
                                          loom_value_id_t low_rhs,
                                          loom_value_id_t* out_low_sum) {
  *out_low_sum = LOOM_VALUE_ID_INVALID;

  loom_type_t vgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &vgpr_type));
  loom_type_t sgpr_x2_type = loom_type_none();
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_make_sgpr_range_type(context, 2, &sgpr_x2_type));

  loom_value_id_t low_lhs_lo = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(
      context, source_op, low_lhs, /*offset=*/0, vgpr_type, &low_lhs_lo));
  loom_value_id_t low_lhs_hi = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(
      context, source_op, low_lhs, /*offset=*/1, vgpr_type, &low_lhs_hi));
  loom_value_id_t low_rhs_lo = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(
      context, source_op, low_rhs, /*offset=*/0, vgpr_type, &low_rhs_lo));
  loom_value_id_t low_rhs_hi = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(
      context, source_op, low_rhs, /*offset=*/1, vgpr_type, &low_rhs_hi));

  loom_value_id_t add_lo_operands[] = {
      low_lhs_lo,
      low_rhs_lo,
  };
  loom_type_t add_result_types[] = {
      vgpr_type,
      sgpr_x2_type,
  };
  loom_op_t* low_add_lo_op = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_ADD_CO_U32,
      add_lo_operands, IREE_ARRAYSIZE(add_lo_operands),
      loom_make_named_attr_slice(NULL, 0), add_result_types,
      IREE_ARRAYSIZE(add_result_types), &low_add_lo_op));
  const loom_value_id_t low_sum_lo =
      loom_value_slice_get(loom_low_op_results(low_add_lo_op), 0);
  const loom_value_id_t low_carry =
      loom_value_slice_get(loom_low_op_results(low_add_lo_op), 1);

  loom_value_id_t add_hi_operands[] = {
      low_lhs_hi,
      low_rhs_hi,
      low_carry,
  };
  loom_op_t* low_add_hi_op = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_ADD_CO_CI_U32,
      add_hi_operands, IREE_ARRAYSIZE(add_hi_operands),
      loom_make_named_attr_slice(NULL, 0), add_result_types,
      IREE_ARRAYSIZE(add_result_types), &low_add_hi_op));
  const loom_value_id_t low_sum_hi =
      loom_value_slice_get(loom_low_op_results(low_add_hi_op), 0);

  loom_type_t vgpr_x2_type = loom_type_none();
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_make_vgpr_range_type(context, 2, &vgpr_x2_type));
  loom_value_id_t sources[] = {
      low_sum_lo,
      low_sum_hi,
  };
  loom_op_t* concat_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_concat_build(
      loom_low_lower_context_builder(context), sources, IREE_ARRAYSIZE(sources),
      vgpr_x2_type, source_op->location, &concat_op));
  *out_low_sum = loom_low_concat_result(concat_op);
  return iree_ok_status();
}

iree_status_t loom_amdgpu_emit_vgpr64_sub(loom_low_lower_context_t* context,
                                          const loom_op_t* source_op,
                                          loom_value_id_t low_lhs,
                                          loom_value_id_t low_rhs,
                                          loom_value_id_t* out_low_difference) {
  *out_low_difference = LOOM_VALUE_ID_INVALID;

  loom_type_t vgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &vgpr_type));
  loom_type_t sgpr_x2_type = loom_type_none();
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_make_sgpr_range_type(context, 2, &sgpr_x2_type));

  loom_value_id_t low_lhs_lo = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(
      context, source_op, low_lhs, /*offset=*/0, vgpr_type, &low_lhs_lo));
  loom_value_id_t low_lhs_hi = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(
      context, source_op, low_lhs, /*offset=*/1, vgpr_type, &low_lhs_hi));
  loom_value_id_t low_rhs_lo = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(
      context, source_op, low_rhs, /*offset=*/0, vgpr_type, &low_rhs_lo));
  loom_value_id_t low_rhs_hi = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(
      context, source_op, low_rhs, /*offset=*/1, vgpr_type, &low_rhs_hi));

  loom_value_id_t sub_lo_operands[] = {
      low_lhs_lo,
      low_rhs_lo,
  };
  loom_type_t sub_result_types[] = {
      vgpr_type,
      sgpr_x2_type,
  };
  loom_op_t* low_sub_lo_op = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_SUB_CO_U32,
      sub_lo_operands, IREE_ARRAYSIZE(sub_lo_operands),
      loom_make_named_attr_slice(NULL, 0), sub_result_types,
      IREE_ARRAYSIZE(sub_result_types), &low_sub_lo_op));
  const loom_value_id_t low_difference_lo =
      loom_value_slice_get(loom_low_op_results(low_sub_lo_op), 0);
  const loom_value_id_t low_borrow =
      loom_value_slice_get(loom_low_op_results(low_sub_lo_op), 1);

  loom_value_id_t sub_hi_operands[] = {
      low_lhs_hi,
      low_rhs_hi,
      low_borrow,
  };
  loom_op_t* low_sub_hi_op = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_SUB_CO_CI_U32,
      sub_hi_operands, IREE_ARRAYSIZE(sub_hi_operands),
      loom_make_named_attr_slice(NULL, 0), sub_result_types,
      IREE_ARRAYSIZE(sub_result_types), &low_sub_hi_op));
  const loom_value_id_t low_difference_hi =
      loom_value_slice_get(loom_low_op_results(low_sub_hi_op), 0);

  loom_type_t vgpr_x2_type = loom_type_none();
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_make_vgpr_range_type(context, 2, &vgpr_x2_type));
  loom_value_id_t sources[] = {
      low_difference_lo,
      low_difference_hi,
  };
  loom_op_t* concat_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_concat_build(
      loom_low_lower_context_builder(context), sources, IREE_ARRAYSIZE(sources),
      vgpr_x2_type, source_op->location, &concat_op));
  *out_low_difference = loom_low_concat_result(concat_op);
  return iree_ok_status();
}

iree_status_t loom_amdgpu_emit_vgpr64_mul_lo(loom_low_lower_context_t* context,
                                             const loom_op_t* source_op,
                                             loom_value_id_t low_lhs,
                                             loom_value_id_t low_rhs,
                                             loom_value_id_t* out_low_product) {
  *out_low_product = LOOM_VALUE_ID_INVALID;

  loom_type_t vgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &vgpr_type));

  loom_value_id_t low_lhs_lo = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(
      context, source_op, low_lhs, /*offset=*/0, vgpr_type, &low_lhs_lo));
  loom_value_id_t low_lhs_hi = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(
      context, source_op, low_lhs, /*offset=*/1, vgpr_type, &low_lhs_hi));
  loom_value_id_t low_rhs_lo = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(
      context, source_op, low_rhs, /*offset=*/0, vgpr_type, &low_rhs_lo));
  loom_value_id_t low_rhs_hi = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(
      context, source_op, low_rhs, /*offset=*/1, vgpr_type, &low_rhs_hi));

  loom_value_id_t product_lo = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MUL_LO_U32, low_lhs_lo,
      low_rhs_lo, vgpr_type, &product_lo));
  loom_value_id_t product_hi_from_lo = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MUL_HI_U32, low_lhs_lo,
      low_rhs_lo, vgpr_type, &product_hi_from_lo));
  loom_value_id_t lhs_lo_rhs_hi = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MUL_LO_U32, low_lhs_lo,
      low_rhs_hi, vgpr_type, &lhs_lo_rhs_hi));
  loom_value_id_t lhs_hi_rhs_lo = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MUL_LO_U32, low_lhs_hi,
      low_rhs_lo, vgpr_type, &lhs_hi_rhs_lo));

  loom_value_id_t high_partial = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_ADD_U32,
      product_hi_from_lo, lhs_lo_rhs_hi, vgpr_type, &high_partial));
  loom_value_id_t product_hi = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_ADD_U32, high_partial,
      lhs_hi_rhs_lo, vgpr_type, &product_hi));

  loom_type_t vgpr_x2_type = loom_type_none();
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_make_vgpr_range_type(context, 2, &vgpr_x2_type));
  loom_value_id_t sources[] = {
      product_lo,
      product_hi,
  };
  loom_op_t* concat_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_concat_build(
      loom_low_lower_context_builder(context), sources, IREE_ARRAYSIZE(sources),
      vgpr_x2_type, source_op->location, &concat_op));
  *out_low_product = loom_low_concat_result(concat_op);
  return iree_ok_status();
}

iree_status_t loom_amdgpu_emit_vgpr64_shl(loom_low_lower_context_t* context,
                                          const loom_op_t* source_op,
                                          loom_value_id_t low_value,
                                          loom_value_id_t low_shift,
                                          loom_value_id_t* out_low_shifted) {
  *out_low_shifted = LOOM_VALUE_ID_INVALID;

  loom_type_t vgpr_x2_type = loom_type_none();
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_make_vgpr_range_type(context, 2, &vgpr_x2_type));

  loom_value_id_t operands[] = {
      low_shift,
      low_value,
  };
  loom_op_t* low_shift_op = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B64, operands,
      IREE_ARRAYSIZE(operands), loom_make_named_attr_slice(NULL, 0),
      &vgpr_x2_type, 1, &low_shift_op));
  *out_low_shifted = loom_value_slice_get(loom_low_op_results(low_shift_op), 0);
  return iree_ok_status();
}

typedef enum loom_amdgpu_sdwa_selector_e {
  LOOM_AMDGPU_SDWA_SELECTOR_BYTE_0 = 0,
  LOOM_AMDGPU_SDWA_SELECTOR_BYTE_1 = 1,
  LOOM_AMDGPU_SDWA_SELECTOR_BYTE_2 = 2,
  LOOM_AMDGPU_SDWA_SELECTOR_BYTE_3 = 3,
  LOOM_AMDGPU_SDWA_SELECTOR_WORD_0 = 4,
  LOOM_AMDGPU_SDWA_SELECTOR_WORD_1 = 5,
  LOOM_AMDGPU_SDWA_SELECTOR_DWORD = 6,
} loom_amdgpu_sdwa_selector_t;

typedef enum loom_amdgpu_sdwa_dst_unused_e {
  LOOM_AMDGPU_SDWA_DST_UNUSED_PAD = 0,
} loom_amdgpu_sdwa_dst_unused_t;

static bool loom_amdgpu_select_sdwa_extract_selector(uint32_t bit_offset,
                                                     uint32_t bit_count,
                                                     uint32_t* out_selector) {
  *out_selector = 0;
  if (bit_count == 8 && (bit_offset % 8u) == 0 && bit_offset < 32u) {
    *out_selector = LOOM_AMDGPU_SDWA_SELECTOR_BYTE_0 + bit_offset / 8u;
    return true;
  }
  if (bit_count == 16 && (bit_offset % 16u) == 0 && bit_offset < 32u) {
    *out_selector = LOOM_AMDGPU_SDWA_SELECTOR_WORD_0 + bit_offset / 16u;
    return true;
  }
  return false;
}

iree_status_t loom_amdgpu_try_emit_vgpr_b32_sdwa_extract(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t value, uint32_t bit_offset, uint32_t bit_count,
    loom_amdgpu_vgpr_sdwa_extract_flags_t flags, loom_type_t lane_type,
    loom_value_id_t* out_value, bool* out_selected) {
  *out_value = LOOM_VALUE_ID_INVALID;
  *out_selected = false;
  IREE_ASSERT(
      !iree_any_bit_set(flags, ~LOOM_AMDGPU_VGPR_SDWA_EXTRACT_FLAG_SIGN_EXTEND),
      "unsupported AMDGPU SDWA extract flags");

  uint32_t source_selector = 0;
  if (!loom_amdgpu_select_sdwa_extract_selector(bit_offset, bit_count,
                                                &source_selector)) {
    return iree_ok_status();
  }

  loom_low_lower_resolved_descriptor_t descriptor = {0};
  bool present = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref_if_present(
      context, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32_SDWA, &descriptor,
      &present));
  if (!present) {
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(
      loom_amdgpu_materialize_low_vgpr_b32(context, source_op, value, &value));

  loom_named_attr_t attrs[4] = {0};
  iree_host_size_t attr_count = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_i64_attr(
      context, IREE_SV("dst_sel"), LOOM_AMDGPU_SDWA_SELECTOR_DWORD, attrs,
      IREE_ARRAYSIZE(attrs), &attr_count));
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_i64_attr(
      context, IREE_SV("dst_unused"), LOOM_AMDGPU_SDWA_DST_UNUSED_PAD, attrs,
      IREE_ARRAYSIZE(attrs), &attr_count));
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_append_i64_attr(context, IREE_SV("src0_sel"), source_selector,
                                  attrs, IREE_ARRAYSIZE(attrs), &attr_count));
  const int64_t sign_extend =
      iree_any_bit_set(flags, LOOM_AMDGPU_VGPR_SDWA_EXTRACT_FLAG_SIGN_EXTEND)
          ? 1
          : 0;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_append_i64_attr(context, IREE_SV("src0_sext"), sign_extend,
                                  attrs, IREE_ARRAYSIZE(attrs), &attr_count));

  loom_value_id_t operands[] = {value};
  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
      context, &descriptor, operands, IREE_ARRAYSIZE(operands),
      loom_make_named_attr_slice(attrs, attr_count), &lane_type, 1,
      /*tied_results=*/NULL, /*tied_result_count=*/0, source_op->location,
      &low_op));
  *out_value = loom_value_slice_get(loom_low_op_results(low_op), 0);
  *out_selected = true;
  return iree_ok_status();
}

iree_status_t loom_amdgpu_try_emit_vgpr_b32_bfe_extract(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t value, uint32_t bit_offset, uint32_t bit_count,
    loom_amdgpu_vgpr_bfe_extract_flags_t flags, loom_type_t lane_type,
    loom_value_id_t* out_value, bool* out_selected) {
  *out_value = LOOM_VALUE_ID_INVALID;
  *out_selected = false;
  IREE_ASSERT(
      !iree_any_bit_set(flags, ~LOOM_AMDGPU_VGPR_BFE_EXTRACT_FLAG_SIGN_EXTEND),
      "unsupported AMDGPU BFE extract flags");
  if (bit_offset > 31 || bit_count < 1 || bit_count > 32 ||
      bit_offset + bit_count > 32) {
    return iree_ok_status();
  }

  const bool sign_extend =
      iree_any_bit_set(flags, LOOM_AMDGPU_VGPR_BFE_EXTRACT_FLAG_SIGN_EXTEND);
  const loom_amdgpu_descriptor_ref_t descriptor_ref =
      sign_extend ? LOOM_AMDGPU_DESCRIPTOR_REF_V_BFE_I32_OFFSET_WIDTH_INLINE
                  : LOOM_AMDGPU_DESCRIPTOR_REF_V_BFE_U32_OFFSET_WIDTH_INLINE;
  loom_low_lower_resolved_descriptor_t descriptor = {0};
  bool present = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref_if_present(
      context, descriptor_ref, &descriptor, &present));
  if (!present) {
    return iree_ok_status();
  }

  loom_named_attr_t attrs[2] = {0};
  iree_host_size_t attr_count = 0;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_append_i64_attr(context, IREE_SV("offset"), bit_offset, attrs,
                                  IREE_ARRAYSIZE(attrs), &attr_count));
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_append_i64_attr(context, IREE_SV("width"), bit_count, attrs,
                                  IREE_ARRAYSIZE(attrs), &attr_count));

  loom_value_id_t operands[] = {value};
  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
      context, &descriptor, operands, IREE_ARRAYSIZE(operands),
      loom_make_named_attr_slice(attrs, attr_count), &lane_type, 1,
      /*tied_results=*/NULL, /*tied_result_count=*/0, source_op->location,
      &low_op));
  *out_value = loom_value_slice_get(loom_low_op_results(low_op), 0);
  *out_selected = true;
  return iree_ok_status();
}

iree_status_t loom_amdgpu_emit_vgpr_scale_u32(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t value, uint32_t scale,
    loom_amdgpu_vgpr_scale_u32_flags_t flags, loom_type_t lane_type,
    loom_value_id_t* out_value) {
  *out_value = LOOM_VALUE_ID_INVALID;
  if (scale == 0) {
    return loom_amdgpu_emit_const_u32(context, source_op,
                                      LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32, 0,
                                      lane_type, out_value);
  }
  if (scale == 1) {
    return loom_amdgpu_materialize_low_vgpr_b32(context, source_op, value,
                                                out_value);
  }
  if (loom_amdgpu_u32_is_power_of_two(scale)) {
    uint32_t shift = 0;
    uint32_t remaining = scale;
    while (remaining > 1u) {
      remaining >>= 1u;
      ++shift;
    }
    const loom_module_t* module = loom_low_lower_context_module(context);
    const loom_type_t low_type = loom_module_value_type(module, value);
    const bool is_sgpr = loom_amdgpu_low_type_is_register_class(
        context, low_type, LOOM_AMDGPU_REG_CLASS_ID_SGPR);
    if (is_sgpr && loom_low_register_type_unit_count(low_type) == 1) {
      return loom_amdgpu_emit_vgpr_binary_immediate(
          context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32_VOP3_IMM,
          value, shift, lane_type, out_value);
    }
    return loom_amdgpu_emit_vgpr_shift(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32_LIT, shift,
        value, lane_type, out_value);
  }

  IREE_RETURN_IF_ERROR(
      loom_amdgpu_materialize_low_vgpr_b32(context, source_op, value, &value));
  enum { LOOM_AMDGPU_U24_MAX = 0xFFFFFFu };
  if (iree_any_bit_set(flags,
                       LOOM_AMDGPU_VGPR_SCALE_U32_FLAG_VALUE_UNSIGNED_24) &&
      scale <= LOOM_AMDGPU_U24_MAX) {
    const loom_low_descriptor_set_t* descriptor_set =
        loom_low_lower_context_descriptor_set(context);
    if (loom_amdgpu_descriptor_ref_ordinal(
            descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_MUL_U32_U24_LIT) !=
        LOOM_LOW_DESCRIPTOR_ORDINAL_NONE) {
      return loom_amdgpu_emit_vgpr_binary_immediate(
          context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MUL_U32_U24_LIT,
          value, scale, lane_type, out_value);
    }
  }

  loom_value_id_t low_scale = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32, scale,
      lane_type, &low_scale));
  return loom_amdgpu_emit_vgpr_binary(context, source_op,
                                      LOOM_AMDGPU_DESCRIPTOR_REF_V_MUL_LO_U32,
                                      value, low_scale, lane_type, out_value);
}

iree_status_t loom_amdgpu_lookup_or_materialize_vgpr_i32(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source_value, loom_value_id_t* out_low_value) {
  *out_low_value = LOOM_VALUE_ID_INVALID;
  loom_value_id_t low_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, source_value, &low_value));

  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_type_t low_type = loom_module_value_type(module, low_value);
  const bool is_vgpr = loom_amdgpu_low_type_is_register_class(
      context, low_type, LOOM_AMDGPU_REG_CLASS_ID_VGPR);
  if (is_vgpr) {
    *out_low_value = low_value;
    return iree_ok_status();
  }

  int64_t value = 0;
  if (loom_amdgpu_value_as_i32_constant(context, source_value, &value)) {
    loom_type_t vgpr_type = loom_type_none();
    IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &vgpr_type));
    return loom_amdgpu_emit_const_u32(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32,
        (uint32_t)(int32_t)value, vgpr_type, out_low_value);
  }

  const bool is_sgpr = loom_amdgpu_low_type_is_register_class(
      context, low_type, LOOM_AMDGPU_REG_CLASS_ID_SGPR);
  if (is_sgpr) {
    return loom_amdgpu_materialize_low_vgpr_b32_registers(
        context, source_op, low_value, out_low_value);
  }

  IREE_ASSERT_UNREACHABLE(
      "AMDGPU i32 VGPR materializer selected an unsupported low value");
  IREE_BUILTIN_UNREACHABLE();
}

iree_status_t loom_amdgpu_lookup_or_materialize_vgpr_f32(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source_value, loom_value_id_t* out_low_value) {
  *out_low_value = LOOM_VALUE_ID_INVALID;
  loom_value_id_t low_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, source_value, &low_value));

  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_type_t low_type = loom_module_value_type(module, low_value);
  const bool is_vgpr = loom_amdgpu_low_type_is_register_class(
      context, low_type, LOOM_AMDGPU_REG_CLASS_ID_VGPR);
  if (is_vgpr) {
    *out_low_value = low_value;
    return iree_ok_status();
  }

  uint32_t bit_pattern = 0;
  if (loom_amdgpu_value_as_f32_constant(context, source_value, &bit_pattern)) {
    loom_type_t vgpr_type = loom_type_none();
    IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &vgpr_type));
    return loom_amdgpu_emit_const_u32(context, source_op,
                                      LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32,
                                      bit_pattern, vgpr_type, out_low_value);
  }

  const bool is_sgpr = loom_amdgpu_low_type_is_register_class(
      context, low_type, LOOM_AMDGPU_REG_CLASS_ID_SGPR);
  if (is_sgpr) {
    return loom_amdgpu_materialize_low_vgpr_b32_registers(
        context, source_op, low_value, out_low_value);
  }

  IREE_ASSERT_UNREACHABLE(
      "AMDGPU f32 VGPR materializer selected an unsupported low value");
  IREE_BUILTIN_UNREACHABLE();
}

iree_status_t loom_amdgpu_lookup_or_materialize_vgpr_i64(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source_value, loom_value_id_t* out_low_value) {
  *out_low_value = LOOM_VALUE_ID_INVALID;
  loom_value_id_t low_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, source_value, &low_value));

  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_type_t low_type = loom_module_value_type(module, low_value);
  if (!loom_low_type_is_register(low_type) ||
      loom_low_register_type_unit_count(low_type) != 2) {
    IREE_ASSERT_UNREACHABLE(
        "AMDGPU i64 VGPR materializer selected an unsupported low value");
    IREE_BUILTIN_UNREACHABLE();
  }

  const bool is_vgpr = loom_amdgpu_low_type_is_register_class(
      context, low_type, LOOM_AMDGPU_REG_CLASS_ID_VGPR);
  if (is_vgpr) {
    *out_low_value = low_value;
    return iree_ok_status();
  }

  const bool is_sgpr = loom_amdgpu_low_type_is_register_class(
      context, low_type, LOOM_AMDGPU_REG_CLASS_ID_SGPR);
  if (!is_sgpr) {
    IREE_ASSERT_UNREACHABLE(
        "AMDGPU i64 VGPR materializer selected an unsupported register class");
    IREE_BUILTIN_UNREACHABLE();
  }
  return loom_amdgpu_materialize_low_vgpr_b32_registers(
      context, source_op, low_value, out_low_value);
}

iree_status_t loom_amdgpu_lookup_or_materialize_vgpr_address(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source_value, loom_value_id_t* out_low_value) {
  *out_low_value = LOOM_VALUE_ID_INVALID;
  loom_value_id_t low_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, source_value, &low_value));

  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_type_t low_type = loom_module_value_type(module, low_value);
  const bool is_vgpr = loom_amdgpu_low_type_is_register_class(
      context, low_type, LOOM_AMDGPU_REG_CLASS_ID_VGPR);
  if (is_vgpr) {
    const uint32_t unit_count = loom_low_register_type_unit_count(low_type);
    if (unit_count == 1) {
      *out_low_value = low_value;
      return iree_ok_status();
    }
    if (unit_count == 2) {
      const loom_type_t lane_type =
          loom_amdgpu_low_register_lane_type(module, low_value);
      return loom_amdgpu_emit_low_slice(context, source_op, low_value,
                                        /*offset=*/0, lane_type, out_low_value);
    }
    IREE_ASSERT_UNREACHABLE(
        "AMDGPU address materializer selected a wrong-width VGPR value");
    IREE_BUILTIN_UNREACHABLE();
  }

  int64_t value = 0;
  if (!loom_amdgpu_value_as_address_constant(context, source_value, &value) ||
      value < 0 || value > UINT32_MAX) {
    const bool is_sgpr = loom_amdgpu_low_type_is_register_class(
        context, low_type, LOOM_AMDGPU_REG_CLASS_ID_SGPR);
    if (is_sgpr) {
      loom_value_id_t low_lane = low_value;
      if (loom_low_register_type_unit_count(low_type) == 2) {
        const loom_type_t lane_type =
            loom_amdgpu_low_register_lane_type(module, low_value);
        IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(
            context, source_op, low_value, /*offset=*/0, lane_type, &low_lane));
      }
      return loom_amdgpu_materialize_low_vgpr_b32_registers(
          context, source_op, low_lane, out_low_value);
    }
    IREE_ASSERT_UNREACHABLE(
        "AMDGPU address materializer selected an unsupported low value");
    IREE_BUILTIN_UNREACHABLE();
  }
  loom_type_t vgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &vgpr_type));
  return loom_amdgpu_emit_const_u32(context, source_op,
                                    LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32,
                                    (uint32_t)value, vgpr_type, out_low_value);
}

iree_status_t loom_amdgpu_lookup_or_materialize_sgpr_address(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source_value, loom_value_id_t* out_low_value) {
  *out_low_value = LOOM_VALUE_ID_INVALID;
  loom_value_id_t low_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, source_value, &low_value));

  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_type_t low_type = loom_module_value_type(module, low_value);
  const bool is_sgpr = loom_amdgpu_low_type_is_register_class(
      context, low_type, LOOM_AMDGPU_REG_CLASS_ID_SGPR);
  if (!is_sgpr) {
    IREE_ASSERT_UNREACHABLE(
        "AMDGPU address materializer selected an incompatible SGPR value");
    IREE_BUILTIN_UNREACHABLE();
  }
  const uint32_t unit_count = loom_low_register_type_unit_count(low_type);
  if (unit_count != 1 && unit_count != 2) {
    IREE_ASSERT_UNREACHABLE(
        "AMDGPU address materializer selected a wrong-width SGPR value");
    IREE_BUILTIN_UNREACHABLE();
  }
  if (unit_count == 1) {
    *out_low_value = low_value;
    return iree_ok_status();
  }
  const loom_type_t lane_type =
      loom_amdgpu_low_register_lane_type(module, low_value);
  return loom_amdgpu_emit_low_slice(context, source_op, low_value,
                                    /*offset=*/0, lane_type, out_low_value);
}

iree_status_t loom_amdgpu_lookup_or_materialize_native_i1_mask(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source_value, loom_value_id_t* out_low_value) {
  *out_low_value = LOOM_VALUE_ID_INVALID;
  loom_value_id_t low_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, source_value, &low_value));

  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_type_t low_type = loom_module_value_type(module, low_value);
  const bool is_sgpr = loom_amdgpu_low_type_is_register_class(
      context, low_type, LOOM_AMDGPU_REG_CLASS_ID_SGPR);
  if (is_sgpr && loom_low_register_type_unit_count(low_type) == 2) {
    *out_low_value = low_value;
    return iree_ok_status();
  }

  loom_type_t sgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_sgpr_type(context, &sgpr_type));

  loom_value_id_t condition = low_value;
  const bool is_scc = loom_amdgpu_low_type_is_register_class(
      context, low_type, LOOM_AMDGPU_REG_CLASS_ID_SCC);
  if (is_sgpr && loom_low_register_type_unit_count(low_type) == 1) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_sgpr32_nonzero_scc(
        context, source_op, low_value, &condition));
  } else if (!is_scc || loom_low_register_type_unit_count(low_type) != 1) {
    IREE_ASSERT_UNREACHABLE(
        "AMDGPU native mask materializer selected an unsupported low value");
    IREE_BUILTIN_UNREACHABLE();
  }

  loom_type_t mask_type = loom_type_none();
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_make_sgpr_range_type(context, 2, &mask_type));

  loom_op_t* exec_read_op = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B64_EXEC_READ,
      /*operands=*/NULL, /*operand_count=*/0, loom_named_attr_slice_empty(),
      &mask_type, 1, &exec_read_op));
  const loom_value_id_t exec_mask =
      loom_value_slice_get(loom_low_op_results(exec_read_op), 0);

  loom_value_id_t exec_lanes[2] = {
      LOOM_VALUE_ID_INVALID,
      LOOM_VALUE_ID_INVALID,
  };
  for (uint32_t i = 0; i < IREE_ARRAYSIZE(exec_lanes); ++i) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(
        context, source_op, exec_mask, i, sgpr_type, &exec_lanes[i]));
  }

  loom_value_id_t low_zero32 = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32, 0, sgpr_type,
      &low_zero32));

  loom_value_id_t mask_lanes[2] = {
      LOOM_VALUE_ID_INVALID,
      LOOM_VALUE_ID_INVALID,
  };
  for (uint32_t i = 0; i < IREE_ARRAYSIZE(mask_lanes); ++i) {
    const loom_value_id_t operands[] = {
        exec_lanes[i],
        low_zero32,
        condition,
    };
    loom_op_t* select_op = NULL;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_CSELECT_B32, operands,
        IREE_ARRAYSIZE(operands), loom_named_attr_slice_empty(), &sgpr_type, 1,
        &select_op));
    mask_lanes[i] = loom_value_slice_get(loom_low_op_results(select_op), 0);
  }

  loom_op_t* concat_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_concat_build(
      loom_low_lower_context_builder(context), mask_lanes,
      IREE_ARRAYSIZE(mask_lanes), mask_type, source_op->location, &concat_op));
  *out_low_value = loom_low_concat_result(concat_op);
  return iree_ok_status();
}

iree_status_t loom_amdgpu_emit_low_slice(loom_low_lower_context_t* context,
                                         const loom_op_t* source_op,
                                         loom_value_id_t low_source,
                                         uint32_t offset,
                                         loom_type_t result_type,
                                         loom_value_id_t* out_value_id) {
  *out_value_id = LOOM_VALUE_ID_INVALID;
  loom_op_t* slice_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_slice_build(
      loom_low_lower_context_builder(context), low_source, offset, result_type,
      source_op->location, &slice_op));
  *out_value_id = loom_low_slice_result(slice_op);
  return iree_ok_status();
}
