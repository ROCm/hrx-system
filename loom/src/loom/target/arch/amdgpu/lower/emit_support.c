// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <inttypes.h>
#include <stdint.h>

#include "loom/ir/context.h"
#include "loom/target/arch/amdgpu/lower/constants.h"
#include "loom/target/arch/amdgpu/lower/emit.h"
#include "loom/target/arch/amdgpu/lower/types.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"

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
  if (*inout_attr_count >= attr_capacity) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "AMDGPU low attr capacity exceeded");
  }
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

iree_status_t loom_amdgpu_emit_sgpr_binary_immediate(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_descriptor_ref_t descriptor_ref, loom_value_id_t value,
    uint32_t immediate, loom_type_t lane_type, loom_value_id_t* out_value) {
  *out_value = LOOM_VALUE_ID_INVALID;

  const loom_low_descriptor_set_t* descriptor_set =
      loom_low_lower_context_descriptor_set(context);
  const loom_amdgpu_descriptor_ref_t rhs_inline_descriptor_ref =
      loom_amdgpu_sgpr_binary_rhs_inline_descriptor_ref(descriptor_ref);
  if (immediate <= 64 && loom_amdgpu_descriptor_ref_ordinal(
                             descriptor_set, rhs_inline_descriptor_ref) !=
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
  for (uint8_t i = 0; i < term->stride_value_count; ++i) {
    loom_value_id_t low_stride = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(
        context, term->stride_values[i], &low_stride));
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

  loom_value_id_t low_accumulator = LOOM_VALUE_ID_INVALID;
  for (uint8_t i = 0; i < source->dynamic_term_count; ++i) {
    switch (dynamic_term_kinds[i]) {
      case LOOM_AMDGPU_MEMORY_DYNAMIC_INDEX_SOFFSET:
        break;
      case LOOM_AMDGPU_MEMORY_DYNAMIC_INDEX_VADDR:
        continue;
      case LOOM_AMDGPU_MEMORY_DYNAMIC_INDEX_NONE:
        IREE_ASSERT_UNREACHABLE("unknown AMDGPU memory dynamic index kind");
        IREE_BUILTIN_UNREACHABLE();
    }
    const loom_low_source_memory_dynamic_term_t* term =
        &source->dynamic_terms[i];
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
  if (!loom_low_type_is_register(*out_low_type)) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "AMDGPU source type did not map to a register");
  }
  return iree_ok_status();
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
  IREE_RETURN_IF_ERROR(loom_low_lower_resolve_descriptor_row(
      context, descriptor, out_descriptor));
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

iree_status_t loom_amdgpu_resolve_descriptor_ref(
    loom_low_lower_context_t* context,
    loom_amdgpu_descriptor_ref_t descriptor_ref,
    loom_low_lower_resolved_descriptor_t* out_descriptor) {
  bool present = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref_if_present(
      context, descriptor_ref, out_descriptor, &present));
  if (!present) {
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "generated AMDGPU lowering policy references missing descriptor ref "
        "%" PRIu16,
        descriptor_ref);
  }
  return iree_ok_status();
}

bool loom_amdgpu_descriptor_set_has_ref(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_descriptor_ref_t descriptor_ref) {
  if (descriptor_set == NULL) {
    return false;
  }
  return loom_amdgpu_descriptor_ref_ordinal(descriptor_set, descriptor_ref) !=
         LOOM_LOW_DESCRIPTOR_ORDINAL_NONE;
}

bool loom_amdgpu_descriptor_set_has_key(
    const loom_low_descriptor_set_t* descriptor_set, iree_string_view_t key) {
  if (descriptor_set == NULL) {
    return false;
  }
  return loom_low_descriptor_set_lookup_descriptor(descriptor_set, key) !=
         LOOM_LOW_DESCRIPTOR_ORDINAL_NONE;
}

bool loom_amdgpu_descriptor_has_implicit_resource_operand(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor) {
  IREE_ASSERT(descriptor_set != NULL);
  IREE_ASSERT(descriptor != NULL);
  IREE_ASSERT((uint64_t)descriptor->operand_start +
                  (uint64_t)descriptor->operand_count <=
              descriptor_set->operand_count);
  const loom_low_operand_t* operands =
      &descriptor_set->operands[descriptor->operand_start];
  for (uint16_t i = 0; i < descriptor->operand_count; ++i) {
    const loom_low_operand_t* operand = &operands[i];
    if (operand->role == LOOM_LOW_OPERAND_ROLE_RESOURCE &&
        iree_any_bit_set(operand->flags, LOOM_LOW_OPERAND_FLAG_IMPLICIT)) {
      return true;
    }
  }
  return false;
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

  loom_low_lower_resolved_descriptor_t resolved_descriptor = {0};
  IREE_RETURN_IF_ERROR(loom_low_lower_resolve_descriptor_row(
      context, descriptor, &resolved_descriptor));

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

iree_status_t loom_amdgpu_emit_m0_u32(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_low_lower_resolved_descriptor_t* consumer_descriptor,
    uint32_t value, loom_value_id_t* out_value_id) {
  IREE_ASSERT(consumer_descriptor->descriptor != NULL);
  *out_value_id = LOOM_VALUE_ID_INVALID;
  loom_type_t m0_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_descriptor_row_implicit_resource_type(
      context, consumer_descriptor->descriptor, &m0_type));
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

iree_status_t loom_amdgpu_materialize_low_vgpr_b32(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_value, loom_value_id_t* out_low_value) {
  *out_low_value = low_value;
  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_type_t low_type = loom_module_value_type(module, low_value);
  bool is_vgpr = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_type_register_class_is(
      context, low_type, LOOM_AMDGPU_REG_CLASS_ID_VGPR, &is_vgpr));
  if (is_vgpr && loom_low_register_type_unit_count(low_type) == 1) {
    return iree_ok_status();
  }
  bool is_sgpr = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_type_register_class_is(
      context, low_type, LOOM_AMDGPU_REG_CLASS_ID_SGPR, &is_sgpr));
  if (!is_sgpr || loom_low_register_type_unit_count(low_type) != 1) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "AMDGPU scalar VGPR materializer selected for a "
                            "non-scalar low value");
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
  bool is_vgpr = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_type_register_class_is(
      context, low_type, LOOM_AMDGPU_REG_CLASS_ID_VGPR, &is_vgpr));
  if (is_vgpr && unit_count != 0 &&
      unit_count <= LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES) {
    return iree_ok_status();
  }

  bool is_sgpr = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_type_register_class_is(
      context, low_type, LOOM_AMDGPU_REG_CLASS_ID_SGPR, &is_sgpr));
  if (!is_sgpr || unit_count == 0 ||
      unit_count > LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "AMDGPU VGPR materializer selected for an "
                            "unsupported low value");
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

iree_status_t loom_amdgpu_emit_vgpr_binary_immediate(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_descriptor_ref_t descriptor_ref, loom_value_id_t value,
    uint32_t immediate, loom_type_t lane_type, loom_value_id_t* out_value) {
  *out_value = LOOM_VALUE_ID_INVALID;
  const loom_low_descriptor_set_t* descriptor_set =
      loom_low_lower_context_descriptor_set(context);
  const loom_amdgpu_descriptor_ref_t src0_inline_descriptor_ref =
      loom_amdgpu_vgpr_binary_src0_inline_descriptor_ref(descriptor_ref);
  if (immediate <= 64 && loom_amdgpu_descriptor_ref_ordinal(
                             descriptor_set, src0_inline_descriptor_ref) !=
                             LOOM_LOW_DESCRIPTOR_ORDINAL_NONE) {
    descriptor_ref = src0_inline_descriptor_ref;
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

  loom_named_attr_t attrs[1] = {0};
  iree_host_size_t attr_count = 0;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_append_i64_attr(context, IREE_SV("shift"), shift, attrs,
                                  IREE_ARRAYSIZE(attrs), &attr_count));
  const loom_value_id_t operands[] = {value, addend};
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
  if (iree_any_bit_set(flags,
                       ~LOOM_AMDGPU_VGPR_SDWA_EXTRACT_FLAG_SIGN_EXTEND)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported AMDGPU SDWA extract flags");
  }

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
  if (iree_any_bit_set(flags, ~LOOM_AMDGPU_VGPR_BFE_EXTRACT_FLAG_SIGN_EXTEND)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported AMDGPU BFE extract flags");
  }
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
    bool is_sgpr = false;
    const loom_module_t* module = loom_low_lower_context_module(context);
    const loom_type_t low_type = loom_module_value_type(module, value);
    IREE_RETURN_IF_ERROR(loom_amdgpu_low_type_register_class_is(
        context, low_type, LOOM_AMDGPU_REG_CLASS_ID_SGPR, &is_sgpr));
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
  bool is_vgpr = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_type_register_class_is(
      context, low_type, LOOM_AMDGPU_REG_CLASS_ID_VGPR, &is_vgpr));
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

  bool is_sgpr = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_type_register_class_is(
      context, low_type, LOOM_AMDGPU_REG_CLASS_ID_SGPR, &is_sgpr));
  if (is_sgpr) {
    return loom_amdgpu_materialize_low_vgpr_b32_registers(
        context, source_op, low_value, out_low_value);
  }

  return iree_make_status(IREE_STATUS_INTERNAL,
                          "AMDGPU i32 VGPR materializer selected for an "
                          "unsupported low value");
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
  bool is_vgpr = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_type_register_class_is(
      context, low_type, LOOM_AMDGPU_REG_CLASS_ID_VGPR, &is_vgpr));
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

  bool is_sgpr = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_type_register_class_is(
      context, low_type, LOOM_AMDGPU_REG_CLASS_ID_SGPR, &is_sgpr));
  if (is_sgpr) {
    return loom_amdgpu_materialize_low_vgpr_b32_registers(
        context, source_op, low_value, out_low_value);
  }

  return iree_make_status(IREE_STATUS_INTERNAL,
                          "AMDGPU f32 VGPR materializer selected for an "
                          "unsupported low value");
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
  bool is_vgpr = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_type_register_class_is(
      context, low_type, LOOM_AMDGPU_REG_CLASS_ID_VGPR, &is_vgpr));
  if (is_vgpr) {
    *out_low_value = low_value;
    return iree_ok_status();
  }

  int64_t value = 0;
  if (!loom_amdgpu_value_as_address_constant(context, source_value, &value) ||
      value < 0 || value > UINT32_MAX) {
    bool is_sgpr = false;
    IREE_RETURN_IF_ERROR(loom_amdgpu_low_type_register_class_is(
        context, low_type, LOOM_AMDGPU_REG_CLASS_ID_SGPR, &is_sgpr));
    if (is_sgpr) {
      return loom_amdgpu_materialize_low_vgpr_b32_registers(
          context, source_op, low_value, out_low_value);
    }
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "AMDGPU address value cannot materialize as a VGPR operand");
  }
  loom_type_t vgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &vgpr_type));
  return loom_amdgpu_emit_const_u32(context, source_op,
                                    LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32,
                                    (uint32_t)value, vgpr_type, out_low_value);
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
  bool is_sgpr = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_type_register_class_is(
      context, low_type, LOOM_AMDGPU_REG_CLASS_ID_SGPR, &is_sgpr));
  if (is_sgpr && loom_low_register_type_unit_count(low_type) == 2) {
    *out_low_value = low_value;
    return iree_ok_status();
  }

  loom_type_t sgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_sgpr_type(context, &sgpr_type));

  loom_value_id_t zero = LOOM_VALUE_ID_INVALID;
  loom_value_id_t condition = low_value;
  bool is_scc = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_type_register_class_is(
      context, low_type, LOOM_AMDGPU_REG_CLASS_ID_SCC, &is_scc));
  if (is_sgpr && loom_low_register_type_unit_count(low_type) == 1) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32, 0, sgpr_type,
        &zero));
    loom_type_t scc_type = loom_type_none();
    IREE_RETURN_IF_ERROR(loom_amdgpu_make_scc_type(context, &scc_type));
    const loom_value_id_t operands[] = {low_value, zero};
    loom_op_t* compare_op = NULL;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_CMP_LG_I32, operands,
        IREE_ARRAYSIZE(operands), loom_named_attr_slice_empty(), &scc_type, 1,
        &compare_op));
    condition = loom_value_slice_get(loom_low_op_results(compare_op), 0);
  } else if (!is_scc || loom_low_register_type_unit_count(low_type) != 1) {
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "AMDGPU i1 value cannot materialize as a native mask operand");
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

  if (zero == LOOM_VALUE_ID_INVALID) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32, 0, sgpr_type,
        &zero));
  }

  loom_value_id_t mask_lanes[2] = {
      LOOM_VALUE_ID_INVALID,
      LOOM_VALUE_ID_INVALID,
  };
  for (uint32_t i = 0; i < IREE_ARRAYSIZE(mask_lanes); ++i) {
    const loom_value_id_t operands[] = {
        exec_lanes[i],
        zero,
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

bool loom_amdgpu_low_legality_bundle_is_amdgpu(
    const loom_target_bundle_t* bundle) {
  return bundle != NULL && bundle->config != NULL &&
         iree_string_view_starts_with(bundle->config->contract_set_key,
                                      IREE_SV("amdgpu."));
}
