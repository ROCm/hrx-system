// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/view/load_coalescing.h"

#include "loom/ops/vector/memory.h"
#include "loom/ops/vector/ops.h"
#include "loom/ops/view/ops.h"

typedef struct loom_view_load_coalescing_axis_index_t {
  // Static logical index on this axis when value_id is invalid.
  int64_t static_index;
  // Dynamic logical index value on this axis, or invalid for static axes.
  loom_value_id_t value_id;
} loom_view_load_coalescing_axis_index_t;

static bool loom_view_load_coalescing_cache_policies_equal(
    const loom_module_t* module, const loom_op_t* left, const loom_op_t* right,
    loom_vector_memory_cache_policy_t* out_policy) {
  loom_vector_memory_cache_policy_t left_policy = {0};
  loom_vector_memory_cache_policy_t right_policy = {0};
  if (!loom_vector_memory_cache_policy_from_op(module, left, &left_policy) ||
      !loom_vector_memory_cache_policy_from_op(module, right, &right_policy)) {
    return false;
  }
  if (left_policy.build_flags != right_policy.build_flags ||
      left_policy.cache_scope != right_policy.cache_scope ||
      left_policy.cache_temporal != right_policy.cache_temporal) {
    return false;
  }
  *out_policy = left_policy;
  return true;
}

static void loom_view_load_coalescing_axis_indices(
    loom_attribute_t static_indices, loom_value_slice_t dynamic_indices,
    uint8_t view_rank, loom_view_load_coalescing_axis_index_t* out_indices) {
  uint16_t dynamic_index_ordinal = 0;
  for (uint8_t axis = 0; axis < view_rank; ++axis) {
    int64_t static_index = static_indices.i64_array[axis];
    if (static_index != INT64_MIN) {
      out_indices[axis] = (loom_view_load_coalescing_axis_index_t){
          .static_index = static_index,
          .value_id = LOOM_VALUE_ID_INVALID,
      };
      continue;
    }
    out_indices[axis] = (loom_view_load_coalescing_axis_index_t){
        .static_index = 0,
        .value_id = dynamic_indices.values[dynamic_index_ordinal++],
    };
  }
}

static iree_status_t loom_view_load_coalescing_axis_index_difference(
    loom_symbolic_expr_context_t* expression_context,
    const loom_view_load_coalescing_axis_index_t* left,
    const loom_view_load_coalescing_axis_index_t* right,
    int64_t* out_difference, bool* out_exact) {
  *out_difference = 0;
  *out_exact = false;
  if (left->value_id == LOOM_VALUE_ID_INVALID &&
      right->value_id == LOOM_VALUE_ID_INVALID) {
    if (!iree_checked_sub_i64(left->static_index, right->static_index,
                              out_difference)) {
      return iree_ok_status();
    }
    *out_exact = true;
    return iree_ok_status();
  }

  loom_symbolic_expr_t left_expression = {0};
  if (left->value_id == LOOM_VALUE_ID_INVALID) {
    loom_symbolic_expr_constant(left->static_index, &left_expression);
  } else {
    IREE_RETURN_IF_ERROR(loom_symbolic_expr_from_value(
        expression_context, left->value_id, &left_expression));
  }

  loom_symbolic_expr_t right_expression = {0};
  if (right->value_id == LOOM_VALUE_ID_INVALID) {
    loom_symbolic_expr_constant(right->static_index, &right_expression);
  } else {
    IREE_RETURN_IF_ERROR(loom_symbolic_expr_from_value(
        expression_context, right->value_id, &right_expression));
  }

  loom_symbolic_expr_t difference = {0};
  IREE_RETURN_IF_ERROR(loom_symbolic_expr_sub(
      expression_context, &left_expression, &right_expression, &difference));
  if (!loom_symbolic_expr_is_constant(&difference)) {
    return iree_ok_status();
  }
  *out_difference = difference.constant;
  *out_exact = true;
  return iree_ok_status();
}

iree_status_t loom_view_load_coalescing_rewrite(
    loom_rewriter_t* rewriter, loom_symbolic_expr_context_t* expression_context,
    loom_op_t* op, bool* out_changed) {
  *out_changed = false;
  loom_op_t* next_op = op->next_op;
  if (!next_op || !loom_view_load_isa(next_op)) {
    return iree_ok_status();
  }
  const loom_memory_access_flags_t memory_flags =
      loom_view_load_memory_flags(op);
  if (memory_flags != loom_view_load_memory_flags(next_op) ||
      (memory_flags != 0 &&
       memory_flags != LOOM_MEMORY_ACCESS_FLAG_ROOT_IN_BOUNDS)) {
    return iree_ok_status();
  }

  const loom_value_id_t view = loom_view_load_view(op);
  if (view != loom_view_load_view(next_op)) {
    return iree_ok_status();
  }

  const loom_value_id_t left_result = loom_view_load_result(op);
  const loom_value_id_t right_result = loom_view_load_result(next_op);
  const loom_type_t scalar_type =
      loom_module_value_type(rewriter->module, left_result);
  const int32_t scalar_bit_count =
      loom_scalar_type_bitwidth(loom_type_element_type(scalar_type));
  if (scalar_bit_count != 16) {
    return iree_ok_status();
  }

  const loom_type_t view_type = loom_module_value_type(rewriter->module, view);
  const uint8_t view_rank = loom_type_rank(view_type);
  if (view_rank == 0) {
    return iree_ok_status();
  }

  loom_vector_memory_cache_policy_t cache_policy = {0};
  if (!loom_view_load_coalescing_cache_policies_equal(rewriter->module, op,
                                                      next_op, &cache_policy)) {
    return iree_ok_status();
  }

  loom_view_load_coalescing_axis_index_t left_indices[LOOM_TYPE_MAX_RANK];
  loom_view_load_coalescing_axis_index_t right_indices[LOOM_TYPE_MAX_RANK];
  loom_view_load_coalescing_axis_indices(loom_view_load_static_indices(op),
                                         loom_view_load_indices(op), view_rank,
                                         left_indices);
  loom_view_load_coalescing_axis_indices(loom_view_load_static_indices(next_op),
                                         loom_view_load_indices(next_op),
                                         view_rank, right_indices);

  for (uint8_t axis = 0; axis < view_rank; ++axis) {
    int64_t difference = 0;
    bool exact = false;
    IREE_RETURN_IF_ERROR(loom_view_load_coalescing_axis_index_difference(
        expression_context, &right_indices[axis], &left_indices[axis],
        &difference, &exact));
    if (!exact) {
      return iree_ok_status();
    }
    const int64_t expected_difference = axis == view_rank - 1 ? 1 : 0;
    if (difference != expected_difference) {
      return iree_ok_status();
    }
  }

  const loom_type_t vector_type =
      loom_type_shaped_1d(LOOM_TYPE_VECTOR, loom_type_element_type(scalar_type),
                          loom_dim_pack_static(2), /*encoding_id=*/0);
  loom_vector_memory_access_t access = {0};
  const loom_fact_context_t* fact_context =
      rewriter->fact_table ? &rewriter->fact_table->context : NULL;
  if (!loom_vector_memory_access_describe(fact_context, rewriter->module,
                                          view_type, vector_type, &access)) {
    return iree_ok_status();
  }

  uint32_t vector_load_build_flags = 0;
  if (iree_any_bit_set(cache_policy.build_flags,
                       LOOM_VECTOR_MEMORY_CACHE_POLICY_BUILD_FLAG_SCOPE)) {
    vector_load_build_flags |= LOOM_VECTOR_LOAD_BUILD_FLAG_HAS_CACHE_SCOPE;
  }
  if (iree_any_bit_set(cache_policy.build_flags,
                       LOOM_VECTOR_MEMORY_CACHE_POLICY_BUILD_FLAG_TEMPORAL)) {
    vector_load_build_flags |= LOOM_VECTOR_LOAD_BUILD_FLAG_HAS_CACHE_TEMPORAL;
  }

  loom_builder_set_before(&rewriter->builder, op);
  loom_value_id_t value_checkpoint = loom_rewriter_value_checkpoint(rewriter);
  loom_op_t* vector_load_op = NULL;
  IREE_RETURN_IF_ERROR(loom_vector_load_build(
      &rewriter->builder, vector_load_build_flags, memory_flags, view,
      loom_view_load_indices(op).values, loom_view_load_indices(op).count,
      loom_view_load_static_indices(op).i64_array,
      loom_view_load_static_indices(op).count, cache_policy.cache_scope,
      cache_policy.cache_temporal, vector_type, op->location, &vector_load_op));
  const loom_value_id_t loaded_vector = loom_vector_load_result(vector_load_op);

  const int64_t left_lane_index[1] = {0};
  loom_op_t* left_extract_op = NULL;
  IREE_RETURN_IF_ERROR(loom_vector_extract_build(
      &rewriter->builder, loaded_vector, /*indices=*/NULL, 0, left_lane_index,
      IREE_ARRAYSIZE(left_lane_index), scalar_type, op->location,
      &left_extract_op));
  loom_value_id_t left_replacement =
      loom_vector_extract_result(left_extract_op);
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, &left_replacement, 1, value_checkpoint));

  const int64_t right_lane_index[1] = {1};
  loom_op_t* right_extract_op = NULL;
  IREE_RETURN_IF_ERROR(loom_vector_extract_build(
      &rewriter->builder, loaded_vector, /*indices=*/NULL, 0, right_lane_index,
      IREE_ARRAYSIZE(right_lane_index), scalar_type, next_op->location,
      &right_extract_op));
  loom_value_id_t right_replacement =
      loom_vector_extract_result(right_extract_op);
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, next_op, &right_replacement, 1, value_checkpoint));

  IREE_RETURN_IF_ERROR(loom_rewriter_replace_all_uses_with(
      rewriter, left_result, left_replacement));
  IREE_RETURN_IF_ERROR(loom_rewriter_replace_all_uses_with(
      rewriter, right_result, right_replacement));
  IREE_RETURN_IF_ERROR(loom_rewriter_erase(rewriter, next_op));
  IREE_RETURN_IF_ERROR(loom_rewriter_erase(rewriter, op));
  *out_changed = true;
  return iree_ok_status();
}
