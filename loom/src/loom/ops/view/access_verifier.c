// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ops/view/access_verifier.h"

#include "loom/error/error_catalog.h"
#include "loom/ir/context.h"

static iree_status_t loom_view_access_verify_emit(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op,
    const loom_error_def_t* error, const loom_diagnostic_param_t* params,
    iree_host_size_t param_count) {
  loom_diagnostic_emission_t emission = {
      .op = op,
      .error = error,
      .params = params,
      .param_count = param_count,
  };
  return iree_diagnostic_emit(emitter, &emission);
}

static uint16_t loom_view_access_dynamic_sentinel_count(
    loom_attribute_t values) {
  uint16_t dynamic_count = 0;
  for (uint16_t i = 0; i < values.count; ++i) {
    if (values.i64_array[i] == INT64_MIN) ++dynamic_count;
  }
  return dynamic_count;
}

static iree_status_t loom_view_access_verify_dynamic_index_count(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter, loom_attribute_t static_values,
    uint16_t dynamic_count) {
  uint16_t expected_dynamic_count =
      loom_view_access_dynamic_sentinel_count(static_values);
  if (dynamic_count == expected_dynamic_count) return iree_ok_status();

  loom_diagnostic_param_t params[] = {
      loom_param_string(loom_op_name(module, op)),
      loom_param_u32(dynamic_count),
      loom_param_u32(expected_dynamic_count),
  };
  return loom_view_access_verify_emit(emitter, op, LOOM_ERR_STRUCTURE_001,
                                      params, IREE_ARRAYSIZE(params));
}

static iree_status_t loom_view_access_verify_static_index_count_matches_rank(
    const loom_op_t* op, iree_diagnostic_emitter_t emitter,
    iree_string_view_t operand_name, loom_attribute_t static_values,
    loom_type_t shaped_type) {
  uint8_t rank = loom_type_rank(shaped_type);
  if (static_values.count == rank) return iree_ok_status();

  loom_diagnostic_param_t params[] = {
      loom_param_string(operand_name),
      loom_param_u32(static_values.count),
      loom_param_i64(rank),
  };
  return loom_view_access_verify_emit(emitter, op, LOOM_ERR_SUBRANGE_001,
                                      params, IREE_ARRAYSIZE(params));
}

static bool loom_view_access_static_index_in_bounds(loom_type_t view_type,
                                                    uint8_t axis,
                                                    int64_t static_index) {
  if (static_index < 0) return false;
  if (loom_type_dim_is_dynamic_at(view_type, axis)) return true;
  return static_index < loom_type_dim_static_size_at(view_type, axis);
}

static bool loom_view_access_find_static_index_out_of_bounds(
    loom_attribute_t static_indices, loom_type_t view_type, uint16_t* out_axis,
    int64_t* out_static_index, int64_t* out_bound) {
  for (uint16_t i = 0; i < static_indices.count; ++i) {
    uint8_t axis = (uint8_t)i;
    int64_t static_index = static_indices.i64_array[i];
    if (static_index == INT64_MIN) continue;
    if (loom_view_access_static_index_in_bounds(view_type, axis,
                                                static_index)) {
      continue;
    }
    *out_axis = i;
    *out_static_index = static_index;
    *out_bound = loom_type_dim_is_dynamic_at(view_type, axis)
                     ? -1
                     : loom_type_dim_static_size_at(view_type, axis);
    return true;
  }
  return false;
}

static iree_status_t loom_view_access_emit_static_index_out_of_bounds(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op, uint16_t axis,
    int64_t static_index, int64_t bound) {
  int64_t total = static_index == INT64_MAX ? INT64_MAX : static_index + 1;
  loom_diagnostic_param_t params[] = {
      loom_param_i64(axis),  loom_param_i64(static_index), loom_param_i64(1),
      loom_param_i64(total), loom_param_i64(bound),
  };
  return loom_view_access_verify_emit(emitter, op, LOOM_ERR_SUBRANGE_004,
                                      params, IREE_ARRAYSIZE(params));
}

iree_status_t loom_view_verify_index_list_rank(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter, iree_string_view_t view_field_name,
    loom_type_t view_type, loom_attribute_t static_indices,
    uint16_t dynamic_index_count) {
  IREE_RETURN_IF_ERROR(loom_view_access_verify_dynamic_index_count(
      module, op, emitter, static_indices, dynamic_index_count));
  if (!loom_type_is_view(view_type)) return iree_ok_status();
  return loom_view_access_verify_static_index_count_matches_rank(
      op, emitter, view_field_name, static_indices, view_type);
}

iree_status_t loom_view_verify_element_access(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter, iree_string_view_t view_field_name,
    loom_type_t view_type, loom_attribute_t static_indices,
    uint16_t dynamic_index_count) {
  IREE_RETURN_IF_ERROR(loom_view_verify_index_list_rank(
      module, op, emitter, view_field_name, view_type, static_indices,
      dynamic_index_count));
  if (!loom_type_is_view(view_type)) return iree_ok_status();

  uint16_t out_of_bounds_axis = 0;
  int64_t out_of_bounds_index = 0;
  int64_t out_of_bounds_bound = 0;
  if (loom_view_access_find_static_index_out_of_bounds(
          static_indices, view_type, &out_of_bounds_axis, &out_of_bounds_index,
          &out_of_bounds_bound)) {
    return loom_view_access_emit_static_index_out_of_bounds(
        emitter, op, out_of_bounds_axis, out_of_bounds_index,
        out_of_bounds_bound);
  }
  return iree_ok_status();
}
