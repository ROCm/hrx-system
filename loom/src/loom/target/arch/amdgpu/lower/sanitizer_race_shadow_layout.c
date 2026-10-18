// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/lower/sanitizer_race_shadow_layout.h"

#include <stdint.h>

#include "loom/codegen/low/builder.h"
#include "loom/target/arch/amdgpu/abi/tsan.h"
#include "loom/target/arch/amdgpu/lower/emit.h"
#include "loom/target/arch/amdgpu/lower/preamble.h"
#include "loom/target/arch/amdgpu/lower/types.h"
#include "loom/target/arch/amdgpu/lower/value/integer64.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"

static iree_status_t loom_amdgpu_normalize_bool(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_value, loom_value_id_t* out_low_bool) {
  loom_value_id_t low_scc = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_sgpr32_nonzero_scc(
      context, source_op, low_value, &low_scc));
  return loom_amdgpu_emit_scc_word(context, source_op, low_scc, out_low_bool);
}

static iree_status_t loom_amdgpu_bool_binary(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_descriptor_ref_t descriptor_ref, loom_value_id_t low_lhs,
    loom_value_id_t low_rhs, loom_value_id_t* out_low_value) {
  loom_type_t sgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_sgpr_type(context, &sgpr_type));
  return loom_amdgpu_emit_binary(context, source_op, descriptor_ref, low_lhs,
                                 low_rhs, sgpr_type, out_low_value);
}

static iree_status_t loom_amdgpu_bool_not(loom_low_lower_context_t* context,
                                          const loom_op_t* source_op,
                                          loom_value_id_t low_bool,
                                          loom_value_id_t* out_low_value) {
  loom_type_t sgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_sgpr_type(context, &sgpr_type));
  return loom_amdgpu_emit_sgpr_binary_immediate(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_XOR_B32, low_bool, 1,
      sgpr_type, out_low_value);
}

static iree_status_t loom_amdgpu_emit_sgpr64_equal_scc(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_lhs, loom_value_id_t low_rhs,
    loom_value_id_t* out_low_scc) {
  *out_low_scc = LOOM_VALUE_ID_INVALID;
  loom_type_t scc_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_scc_type(context, &scc_type));
  const loom_value_id_t operands[] = {low_lhs, low_rhs};
  loom_op_t* compare_op = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_CMP_EQ_U64, operands,
      IREE_ARRAYSIZE(operands), loom_named_attr_slice_empty(), &scc_type, 1,
      &compare_op));
  *out_low_scc = loom_value_slice_get(loom_low_op_results(compare_op), 0);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_select_sgpr64(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_condition_scc, loom_value_id_t low_true,
    loom_value_id_t low_false, loom_value_id_t* out_low_value) {
  *out_low_value = LOOM_VALUE_ID_INVALID;
  loom_type_t sgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_sgpr_type(context, &sgpr_type));
  loom_value_id_t selected[2] = {LOOM_VALUE_ID_INVALID, LOOM_VALUE_ID_INVALID};
  for (uint32_t i = 0; i < 2; ++i) {
    loom_value_id_t true_lane = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(
        context, source_op, low_true, i, sgpr_type, &true_lane));
    loom_value_id_t false_lane = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(
        context, source_op, low_false, i, sgpr_type, &false_lane));
    const loom_value_id_t operands[] = {true_lane, false_lane,
                                        low_condition_scc};
    loom_op_t* select_op = NULL;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_CSELECT_B32, operands,
        IREE_ARRAYSIZE(operands), loom_named_attr_slice_empty(), &sgpr_type, 1,
        &select_op));
    selected[i] = loom_value_slice_get(loom_low_op_results(select_op), 0);
  }
  loom_type_t sgpr_x2_type = loom_type_none();
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_make_sgpr_range_type(context, 2, &sgpr_x2_type));
  loom_op_t* concat_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_concat_build(
      loom_low_lower_context_builder(context), selected,
      IREE_ARRAYSIZE(selected), sgpr_x2_type, source_op->location, &concat_op));
  *out_low_value = loom_low_concat_result(concat_op);
  return iree_ok_status();
}

iree_status_t loom_amdgpu_build_sanitizer_race_shadow_layout(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_sanitizer_race_shadow_layout_config_t* config,
    loom_amdgpu_sanitizer_race_shadow_layout_t* out_layout) {
  *out_layout = (loom_amdgpu_sanitizer_race_shadow_layout_t){0};
  loom_amdgpu_sanitizer_race_shadow_layout_t layout = {0};

  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_current_group_segment_size(
      context, source_op, &layout.workgroup_local_memory_size));
  loom_value_id_t entry_count_u64 = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_sgpr64_from_u32(
      context, source_op, layout.workgroup_local_memory_size,
      &entry_count_u64));
  loom_value_id_t entry_size_u64 = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_sgpr64_constant_u64(
      context, source_op, LOOM_AMDGPU_TSAN_SHADOW_CELL_BYTE_LENGTH,
      &entry_size_u64));
  loom_value_id_t shadow_data_size = LOOM_VALUE_ID_INVALID;
  loom_value_id_t data_size_overflow = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_sgpr64_mul_checked(
      context, source_op, entry_count_u64, entry_size_u64, &shadow_data_size,
      &data_size_overflow));
  loom_value_id_t header_size = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_sgpr64_constant_u64(
      context, source_op, LOOM_AMDGPU_TSAN_WORKGROUP_SHADOW_HEADER_BYTE_LENGTH,
      &header_size));
  loom_value_id_t stride_overflow = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_sgpr64_add_checked(
      context, source_op, shadow_data_size, header_size,
      &layout.workgroup_shadow_stride, &stride_overflow));
  loom_value_id_t geometry_overflow = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_bool_binary(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_OR_B32,
      data_size_overflow, stride_overflow, &geometry_overflow));

  loom_value_id_t ordinal_overflow = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_current_workgroup_linear_id_u64_checked(
      context, source_op, &layout.workgroup_ordinal, &ordinal_overflow));
  loom_value_id_t offset_overflow = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_sgpr64_mul_checked(
      context, source_op, layout.workgroup_ordinal,
      layout.workgroup_shadow_stride, &layout.workgroup_shadow_offset,
      &offset_overflow));
  loom_value_id_t end_overflow = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_sgpr64_add_checked(
      context, source_op, layout.workgroup_shadow_offset,
      layout.workgroup_shadow_stride, &layout.required_shadow_end,
      &end_overflow));

  loom_value_id_t pre_end_overflow = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_bool_binary(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_OR_B32,
      geometry_overflow, ordinal_overflow, &pre_end_overflow));
  IREE_RETURN_IF_ERROR(loom_amdgpu_bool_binary(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_OR_B32, pre_end_overflow,
      offset_overflow, &pre_end_overflow));
  loom_value_id_t total_overflow = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_bool_binary(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_OR_B32, pre_end_overflow,
      end_overflow, &total_overflow));
  loom_value_id_t total_overflow_bool = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_normalize_bool(
      context, source_op, total_overflow, &total_overflow_bool));
  loom_value_id_t total_overflow_scc = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_sgpr32_nonzero_scc(
      context, source_op, total_overflow_bool, &total_overflow_scc));
  loom_value_id_t max_u64 = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_sgpr64_constant_u64(
      context, source_op, UINT64_MAX, &max_u64));
  IREE_RETURN_IF_ERROR(loom_amdgpu_select_sgpr64(
      context, source_op, total_overflow_scc, max_u64,
      layout.required_shadow_end, &layout.required_shadow_end));

  loom_value_id_t end_in_bounds_scc = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_sgpr64_ule_scc(
      context, source_op, layout.required_shadow_end, config->shadow_size,
      &end_in_bounds_scc));
  loom_value_id_t end_in_bounds = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_scc_word(
      context, source_op, end_in_bounds_scc, &end_in_bounds));
  loom_value_id_t no_overflow = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_bool_not(context, source_op,
                                            total_overflow_bool, &no_overflow));
  loom_value_id_t in_bounds = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_bool_binary(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_AND_B32, end_in_bounds,
      no_overflow, &in_bounds));
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_sgpr32_nonzero_scc(
      context, source_op, in_bounds, &layout.in_bounds_scc));

  loom_value_id_t offset_in_bounds_scc = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_sgpr64_ule_scc(
      context, source_op, layout.workgroup_shadow_offset, config->shadow_size,
      &offset_in_bounds_scc));
  loom_value_id_t offset_in_bounds = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_scc_word(
      context, source_op, offset_in_bounds_scc, &offset_in_bounds));
  loom_value_id_t pre_end_overflow_bool = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_normalize_bool(
      context, source_op, pre_end_overflow, &pre_end_overflow_bool));
  loom_value_id_t no_pre_end_overflow = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_bool_not(
      context, source_op, pre_end_overflow_bool, &no_pre_end_overflow));
  loom_value_id_t end_overflow_bool = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_normalize_bool(
      context, source_op, end_overflow, &end_overflow_bool));
  loom_value_id_t end_out_of_bounds = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_bool_not(context, source_op, end_in_bounds,
                                            &end_out_of_bounds));
  IREE_RETURN_IF_ERROR(loom_amdgpu_bool_binary(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_OR_B32,
      end_out_of_bounds, end_overflow_bool, &end_out_of_bounds));
  loom_value_id_t first_boundary = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_bool_binary(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_AND_B32,
      no_pre_end_overflow, offset_in_bounds, &first_boundary));
  IREE_RETURN_IF_ERROR(loom_amdgpu_bool_binary(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_AND_B32, first_boundary,
      end_out_of_bounds, &first_boundary));

  loom_value_id_t zero_u64 = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_emit_sgpr64_constant_u64(context, source_op, 0, &zero_u64));
  loom_value_id_t ordinal_zero_scc = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_sgpr64_equal_scc(
      context, source_op, layout.workgroup_ordinal, zero_u64,
      &ordinal_zero_scc));
  loom_value_id_t ordinal_zero = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_scc_word(
      context, source_op, ordinal_zero_scc, &ordinal_zero));
  loom_value_id_t geometry_overflow_bool = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_normalize_bool(
      context, source_op, geometry_overflow, &geometry_overflow_bool));
  loom_value_id_t invalid_geometry_first = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_bool_binary(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_AND_B32,
      geometry_overflow_bool, ordinal_zero, &invalid_geometry_first));
  loom_value_id_t first_invalid = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_bool_binary(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_OR_B32, first_boundary,
      invalid_geometry_first, &first_invalid));
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_sgpr32_nonzero_scc(
      context, source_op, first_invalid, &layout.first_invalid_scc));

  *out_layout = layout;
  return iree_ok_status();
}
