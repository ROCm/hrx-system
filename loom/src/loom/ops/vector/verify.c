// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <math.h>

#include "loom/error/emitter.h"
#include "loom/error/error_catalog.h"
#include "loom/ir/attribute.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ir/scalar_type.h"
#include "loom/ops/atomic.h"
#include "loom/ops/cache.h"
#include "loom/ops/combining.h"
#include "loom/ops/encoding/numeric_transform.h"
#include "loom/ops/encoding/params.h"
#include "loom/ops/encoding/roles.h"
#include "loom/ops/encoding/storage.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/vector/encoding_auxiliary.h"
#include "loom/ops/vector/fragment.h"
#include "loom/ops/vector/memory.h"
#include "loom/ops/vector/ops.h"
#include "loom/util/fact_table.h"

static iree_status_t loom_vector_emit(iree_diagnostic_emitter_t emitter,
                                      const loom_op_t* op,
                                      const loom_error_def_t* error,
                                      const loom_diagnostic_param_t* params,
                                      iree_host_size_t param_count) {
  loom_diagnostic_emission_t emission = {
      .op = op,
      .error = error,
      .params = params,
      .param_count = param_count,
  };
  return iree_diagnostic_emit(emitter, &emission);
}

static uint32_t loom_vector_saturating_count(uint64_t count) {
  if (count > UINT32_MAX) return UINT32_MAX;
  return (uint32_t)count;
}

static iree_status_t loom_vector_emit_attribute_kind_mismatch(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op,
    iree_string_view_t attr_name, loom_attr_kind_t actual_kind,
    loom_attr_kind_t expected_kind) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(attr_name),
      loom_param_u32(actual_kind),
      loom_param_u32(expected_kind),
  };
  return loom_vector_emit(emitter, op, LOOM_ERR_TYPE_005, params,
                          IREE_ARRAYSIZE(params));
}

static iree_status_t loom_vector_emit_indexed_attribute_kind_mismatch(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op,
    iree_string_view_t attr_name, uint16_t attr_index,
    loom_attr_kind_t actual_kind, loom_attr_kind_t expected_kind) {
  loom_diagnostic_param_t params[] = {
      loom_param_with_field_ref(
          loom_param_string(attr_name),
          loom_diagnostic_field_ref(LOOM_DIAGNOSTIC_FIELD_ATTRIBUTE,
                                    attr_index)),
      loom_param_u32(actual_kind),
      loom_param_u32(expected_kind),
  };
  return loom_vector_emit(emitter, op, LOOM_ERR_TYPE_005, params,
                          IREE_ARRAYSIZE(params));
}

static iree_status_t loom_vector_emit_attribute_value_constraint(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op,
    iree_string_view_t attr_name, int64_t actual_value,
    iree_string_view_t expected_constraint) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(attr_name),
      loom_param_i64(actual_value),
      loom_param_string(expected_constraint),
  };
  return loom_vector_emit(emitter, op, LOOM_ERR_STRUCTURE_014, params,
                          IREE_ARRAYSIZE(params));
}

static iree_status_t loom_vector_emit_operand_constraint(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op,
    iree_string_view_t operand_name, loom_type_t actual_type,
    iree_string_view_t expected_constraint) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(operand_name),
      loom_param_type(actual_type),
      loom_param_string(expected_constraint),
  };
  return loom_vector_emit(emitter, op, LOOM_ERR_TYPE_003, params,
                          IREE_ARRAYSIZE(params));
}

static iree_status_t loom_vector_emit_result_constraint(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op,
    iree_string_view_t result_name, loom_type_t actual_type,
    iree_string_view_t expected_constraint) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(result_name),
      loom_param_type(actual_type),
      loom_param_string(expected_constraint),
  };
  return loom_vector_emit(emitter, op, LOOM_ERR_TYPE_004, params,
                          IREE_ARRAYSIZE(params));
}

static iree_status_t loom_vector_emit_result_code_capacity(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op,
    iree_string_view_t result_name, loom_type_t actual_type,
    int64_t threshold_count) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(result_name),
      loom_param_type(actual_type),
      loom_param_i64(threshold_count),
      loom_param_i64(threshold_count),
  };
  return loom_vector_emit(emitter, op, LOOM_ERR_TYPE_011, params,
                          IREE_ARRAYSIZE(params));
}

static iree_status_t loom_vector_emit_static_threshold_order(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op,
    iree_string_view_t field_name, uint32_t left_index, uint32_t right_index) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(field_name),
      loom_param_u32(left_index),
      loom_param_u32(right_index),
  };
  return loom_vector_emit(emitter, op, LOOM_ERR_STRUCTURE_015, params,
                          IREE_ARRAYSIZE(params));
}

static iree_status_t loom_vector_emit_field_constraint(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op,
    bool field_is_result, iree_string_view_t field_name,
    loom_type_t actual_type, iree_string_view_t expected_constraint) {
  if (field_is_result) {
    return loom_vector_emit_result_constraint(emitter, op, field_name,
                                              actual_type, expected_constraint);
  }
  return loom_vector_emit_operand_constraint(emitter, op, field_name,
                                             actual_type, expected_constraint);
}

static iree_status_t loom_vector_emit_rank_mismatch(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op,
    iree_string_view_t field_name, uint8_t actual_rank,
    iree_string_view_t expected_field_name, uint8_t expected_rank) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(field_name),
      loom_param_i64(actual_rank),
      loom_param_string(expected_field_name),
      loom_param_i64(expected_rank),
  };
  return loom_vector_emit(emitter, op, LOOM_ERR_SHAPE_001, params,
                          IREE_ARRAYSIZE(params));
}

static iree_status_t loom_vector_emit_shape_mismatch(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op,
    iree_string_view_t field_name, iree_string_view_t expected_field_name) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(field_name),
      loom_param_string(expected_field_name),
  };
  return loom_vector_emit(emitter, op, LOOM_ERR_SHAPE_002, params,
                          IREE_ARRAYSIZE(params));
}

static iree_status_t loom_vector_verify_float_instance_flags(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op,
    iree_string_view_t operand_name, loom_type_t operand_type) {
  if (op->instance_flags == 0 ||
      loom_scalar_type_is_float(loom_type_element_type(operand_type))) {
    return iree_ok_status();
  }
  return loom_vector_emit_operand_constraint(
      emitter, op, operand_name, operand_type,
      IREE_SV("floating-point element type when float flags are present"));
}

static iree_status_t loom_vector_emit_count_mismatch(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op,
    iree_string_view_t field_name, uint64_t actual_count,
    iree_string_view_t expected_field_name, uint64_t expected_count) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(field_name),
      loom_param_u32(loom_vector_saturating_count(actual_count)),
      loom_param_string(expected_field_name),
      loom_param_u32(loom_vector_saturating_count(expected_count)),
  };
  return loom_vector_emit(emitter, op, LOOM_ERR_STRUCTURE_013, params,
                          IREE_ARRAYSIZE(params));
}

static uint16_t loom_vector_dynamic_sentinel_count(loom_attribute_t values) {
  uint16_t dynamic_count = 0;
  for (uint16_t i = 0; i < values.count; ++i) {
    if (values.i64_array[i] == INT64_MIN) ++dynamic_count;
  }
  return dynamic_count;
}

static bool loom_vector_static_index_in_bounds(loom_type_t source_type,
                                               uint8_t source_axis,
                                               int64_t static_index) {
  if (static_index < 0) return false;
  if (loom_type_dim_is_dynamic_at(source_type, source_axis)) return true;
  return static_index < loom_type_dim_static_size_at(source_type, source_axis);
}

static bool loom_vector_find_static_index_out_of_bounds(
    loom_attribute_t static_indices, loom_type_t source_type,
    uint16_t* out_axis, int64_t* out_static_index, int64_t* out_bound) {
  for (uint16_t i = 0; i < static_indices.count; ++i) {
    int64_t static_index = static_indices.i64_array[i];
    if (static_index == INT64_MIN) continue;
    if (loom_vector_static_index_in_bounds(source_type, (uint8_t)i,
                                           static_index)) {
      continue;
    }
    *out_axis = i;
    *out_static_index = static_index;
    *out_bound = loom_type_dim_is_dynamic_at(source_type, i)
                     ? -1
                     : loom_type_dim_static_size_at(source_type, i);
    return true;
  }
  return false;
}

static iree_status_t loom_vector_emit_static_index_out_of_bounds(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op, uint16_t axis,
    int64_t static_index, int64_t bound) {
  int64_t total = static_index == INT64_MAX ? INT64_MAX : static_index + 1;
  loom_diagnostic_param_t params[] = {
      loom_param_i64(axis),  loom_param_i64(static_index), loom_param_i64(1),
      loom_param_i64(total), loom_param_i64(bound),
  };
  return loom_vector_emit(emitter, op, LOOM_ERR_SUBRANGE_004, params,
                          IREE_ARRAYSIZE(params));
}

static int64_t loom_vector_saturating_add_i64(int64_t lhs, int64_t rhs) {
  if (rhs > 0 && lhs > INT64_MAX - rhs) return INT64_MAX;
  if (rhs < 0 && lhs < INT64_MIN - rhs) return INT64_MIN;
  return lhs + rhs;
}

static iree_status_t loom_vector_emit_static_access_out_of_bounds(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op, uint16_t axis,
    int64_t offset, int64_t size, int64_t bound) {
  loom_diagnostic_param_t params[] = {
      loom_param_i64(axis),
      loom_param_i64(offset),
      loom_param_i64(size),
      loom_param_i64(loom_vector_saturating_add_i64(offset, size)),
      loom_param_i64(bound),
  };
  return loom_vector_emit(emitter, op, LOOM_ERR_SUBRANGE_004, params,
                          IREE_ARRAYSIZE(params));
}

static bool loom_vector_tail_shape_equals(loom_type_t source_type,
                                          uint8_t consumed_rank,
                                          loom_type_t tail_type) {
  uint8_t tail_rank = loom_type_rank(tail_type);
  for (uint8_t i = 0; i < tail_rank; ++i) {
    if (loom_type_dim(source_type, consumed_rank + i) !=
        loom_type_dim(tail_type, i)) {
      return false;
    }
  }
  return true;
}

static iree_status_t loom_vector_verify_subvalue_type(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op,
    iree_string_view_t value_name, loom_type_t value_type,
    iree_string_view_t expected_shape_name, loom_type_t source_type,
    uint8_t consumed_rank, bool value_is_result) {
  uint8_t source_rank = loom_type_rank(source_type);
  uint8_t expected_rank = source_rank - consumed_rank;
  if (expected_rank == 0) {
    if (loom_type_is_scalar(value_type)) return iree_ok_status();
    return loom_vector_emit_field_constraint(emitter, op, value_is_result,
                                             value_name, value_type,
                                             IREE_SV("scalar tail value"));
  }

  if (!loom_type_is_vector(value_type)) {
    return loom_vector_emit_field_constraint(emitter, op, value_is_result,
                                             value_name, value_type,
                                             IREE_SV("vector tail value"));
  }
  uint8_t value_rank = loom_type_rank(value_type);
  if (value_rank != expected_rank) {
    return loom_vector_emit_rank_mismatch(emitter, op, value_name, value_rank,
                                          IREE_SV("source tail"),
                                          expected_rank);
  }
  if (loom_vector_tail_shape_equals(source_type, consumed_rank, value_type)) {
    return iree_ok_status();
  }
  return loom_vector_emit_shape_mismatch(emitter, op, value_name,
                                         expected_shape_name);
}

static iree_status_t loom_vector_verify_atomic_kind(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op,
    iree_string_view_t value_name, loom_type_t value_type, uint8_t kind,
    bool allow_exchange) {
  if (!allow_exchange && loom_atomic_kind_is_exchange(kind)) {
    return loom_vector_emit_attribute_value_constraint(
        emitter, op, IREE_SV("kind"), kind,
        IREE_SV("non-exchange atomic reduce kind"));
  }
  if (!loom_atomic_kind_is_valid(kind)) return iree_ok_status();
  if (!loom_type_is_vector(value_type)) return iree_ok_status();

  loom_scalar_type_t element_type = loom_type_element_type(value_type);
  if (loom_scalar_type_is_integer(element_type) &&
      loom_atomic_kind_accepts_integer(kind)) {
    return iree_ok_status();
  }
  if (loom_scalar_type_is_float(element_type) &&
      loom_atomic_kind_accepts_float(kind)) {
    return iree_ok_status();
  }

  iree_string_view_t expected_constraint =
      loom_atomic_kind_accepts_integer(kind)
          ? IREE_SV("integer element type for atomic kind")
          : IREE_SV("floating-point element type for atomic kind");
  return loom_vector_emit_operand_constraint(emitter, op, value_name,
                                             value_type, expected_constraint);
}

static iree_status_t loom_vector_verify_atomic_cmpxchg_ordering(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op,
    uint8_t success_ordering, uint8_t failure_ordering) {
  loom_atomic_cmpxchg_ordering_error_t error =
      loom_atomic_cmpxchg_ordering_validate(success_ordering, failure_ordering);
  if (error == LOOM_ATOMIC_CMPXCHG_ORDERING_ERROR_NONE) {
    return iree_ok_status();
  }
  iree_string_view_t attr_name =
      loom_atomic_cmpxchg_ordering_error_attr_name(error);
  return loom_vector_emit_attribute_value_constraint(
      emitter, op, attr_name, failure_ordering,
      loom_atomic_cmpxchg_ordering_error_expected_constraint(error));
}

static iree_status_t loom_vector_emit_offset_count_mismatch(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op,
    iree_string_view_t operand_name, uint16_t offset_count, uint8_t rank) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(operand_name),
      loom_param_u32(offset_count),
      loom_param_i64(rank),
  };
  return loom_vector_emit(emitter, op, LOOM_ERR_SUBRANGE_001, params,
                          IREE_ARRAYSIZE(params));
}

static iree_status_t loom_vector_verify_dynamic_sentinel_count(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op,
    iree_string_view_t dynamic_field_name, loom_attribute_t static_values,
    uint16_t dynamic_value_count) {
  uint16_t expected_dynamic_count =
      loom_vector_dynamic_sentinel_count(static_values);
  if (dynamic_value_count == expected_dynamic_count) return iree_ok_status();
  return loom_vector_emit_count_mismatch(
      emitter, op, dynamic_field_name, dynamic_value_count,
      IREE_SV("dynamic sentinels"), expected_dynamic_count);
}

static iree_status_t loom_vector_verify_dynamic_index_count(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op,
    loom_attribute_t static_indices, uint16_t dynamic_index_count) {
  return loom_vector_verify_dynamic_sentinel_count(
      emitter, op, IREE_SV("indices"), static_indices, dynamic_index_count);
}

static bool loom_vector_find_static_slice_out_of_bounds(
    loom_attribute_t static_offsets, loom_type_t source_type,
    loom_type_t result_type, uint16_t* out_axis, int64_t* out_offset,
    int64_t* out_extent, int64_t* out_bound) {
  uint8_t rank = loom_type_rank(source_type);
  for (uint8_t axis = 0; axis < rank; ++axis) {
    int64_t offset = static_offsets.i64_array[axis];
    bool offset_is_static = offset != INT64_MIN;
    bool extent_is_static = !loom_type_dim_is_dynamic_at(result_type, axis);
    int64_t extent =
        extent_is_static ? loom_type_dim_static_size_at(result_type, axis) : 1;
    int64_t bound = loom_type_dim_is_dynamic_at(source_type, axis)
                        ? -1
                        : loom_type_dim_static_size_at(source_type, axis);

    if (offset_is_static && offset < 0) {
      *out_axis = axis;
      *out_offset = offset;
      *out_extent = extent;
      *out_bound = bound;
      return true;
    }

    if (bound < 0) continue;
    if (offset_is_static && offset > bound) {
      *out_axis = axis;
      *out_offset = offset;
      *out_extent = 0;
      *out_bound = bound;
      return true;
    }
    if (!extent_is_static) continue;
    if (extent > bound) {
      *out_axis = axis;
      *out_offset = offset_is_static ? offset : 0;
      *out_extent = extent;
      *out_bound = bound;
      return true;
    }
    if (offset_is_static && extent > bound - offset) {
      *out_axis = axis;
      *out_offset = offset;
      *out_extent = extent;
      *out_bound = bound;
      return true;
    }
  }
  return false;
}

static bool loom_vector_dim_equals(loom_type_t lhs_type, uint8_t lhs_axis,
                                   loom_type_t rhs_type, uint8_t rhs_axis) {
  return loom_type_dim(lhs_type, lhs_axis) == loom_type_dim(rhs_type, rhs_axis);
}

static bool loom_vector_shapes_match(loom_type_t lhs_type,
                                     loom_type_t rhs_type) {
  uint8_t rank = loom_type_rank(lhs_type);
  if (loom_type_rank(rhs_type) != rank) return false;
  for (uint8_t axis = 0; axis < rank; ++axis) {
    if (!loom_vector_dim_equals(lhs_type, axis, rhs_type, axis)) return false;
  }
  return true;
}

static bool loom_vector_find_static_memory_access_out_of_bounds(
    const loom_vector_memory_access_t* access, loom_attribute_t static_indices,
    uint16_t* out_axis, int64_t* out_offset, int64_t* out_extent,
    int64_t* out_bound) {
  uint8_t view_rank = access->view_rank;
  for (uint8_t axis = 0; axis < view_rank; ++axis) {
    int64_t offset = static_indices.i64_array[axis];
    if (offset == INT64_MIN) continue;

    int64_t extent = 1;
    bool extent_is_static =
        loom_vector_memory_access_static_axis_extent(access, axis, &extent);
    if (offset < 0) {
      *out_axis = axis;
      *out_offset = offset;
      *out_extent = extent_is_static ? extent : 1;
      *out_bound = loom_type_dim_is_dynamic_at(access->view_type, axis)
                       ? -1
                       : loom_type_dim_static_size_at(access->view_type, axis);
      return true;
    }
    if (loom_type_dim_is_dynamic_at(access->view_type, axis)) {
      continue;
    }

    int64_t bound = loom_type_dim_static_size_at(access->view_type, axis);
    if (offset > bound) {
      *out_axis = axis;
      *out_offset = offset;
      *out_extent = 0;
      *out_bound = bound;
      return true;
    }

    if (!extent_is_static) continue;
    if (extent < 0 || extent > bound - offset) {
      *out_axis = axis;
      *out_offset = offset;
      *out_extent = extent;
      *out_bound = bound;
      return true;
    }
  }
  return false;
}

static iree_status_t loom_vector_verify_memory_access(
    const loom_module_t* module, iree_diagnostic_emitter_t emitter,
    const loom_op_t* op, iree_string_view_t vector_name, bool vector_is_result,
    loom_type_t view_type, loom_type_t vector_type,
    loom_attribute_t static_indices, uint16_t dynamic_index_count) {
  if (!loom_type_is_view(view_type) || !loom_type_is_vector(vector_type)) {
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(loom_vector_verify_dynamic_index_count(
      emitter, op, static_indices, dynamic_index_count));

  uint8_t view_rank = loom_type_rank(view_type);
  if (view_rank == 0) {
    return loom_vector_emit_operand_constraint(
        emitter, op, IREE_SV("view"), view_type, IREE_SV("rank >= 1 view"));
  }
  if (static_indices.count != view_rank) {
    return loom_vector_emit_offset_count_mismatch(
        emitter, op, IREE_SV("view"), static_indices.count, view_rank);
  }

  uint8_t vector_rank = loom_type_rank(vector_type);
  if (vector_rank > view_rank) {
    return loom_vector_emit_field_constraint(
        emitter, op, vector_is_result, vector_name, vector_type,
        IREE_SV("rank no greater than view rank"));
  }

  loom_vector_memory_access_t access;
  if (!loom_vector_memory_access_describe(
          /*context=*/NULL, module, view_type, vector_type, &access)) {
    return iree_ok_status();
  }

  uint16_t out_of_bounds_axis = 0;
  int64_t out_of_bounds_offset = 0;
  int64_t out_of_bounds_extent = 0;
  int64_t out_of_bounds_bound = 0;
  if (loom_vector_find_static_memory_access_out_of_bounds(
          &access, static_indices, &out_of_bounds_axis, &out_of_bounds_offset,
          &out_of_bounds_extent, &out_of_bounds_bound)) {
    return loom_vector_emit_static_access_out_of_bounds(
        emitter, op, out_of_bounds_axis, out_of_bounds_offset,
        out_of_bounds_extent, out_of_bounds_bound);
  }
  return iree_ok_status();
}

static iree_status_t loom_vector_verify_fragment_memory_origin(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op,
    loom_type_t view_type, loom_attribute_t static_indices,
    uint16_t dynamic_index_count) {
  if (!loom_type_is_view(view_type)) {
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(loom_vector_verify_dynamic_index_count(
      emitter, op, static_indices, dynamic_index_count));

  uint8_t view_rank = loom_type_rank(view_type);
  if (view_rank == 0) {
    return loom_vector_emit_operand_constraint(
        emitter, op, IREE_SV("view"), view_type, IREE_SV("rank >= 1 view"));
  }
  if (static_indices.count != view_rank) {
    return loom_vector_emit_offset_count_mismatch(
        emitter, op, IREE_SV("view"), static_indices.count, view_rank);
  }

  uint16_t out_of_bounds_axis = 0;
  int64_t out_of_bounds_index = 0;
  int64_t out_of_bounds_bound = 0;
  if (loom_vector_find_static_index_out_of_bounds(
          static_indices, view_type, &out_of_bounds_axis, &out_of_bounds_index,
          &out_of_bounds_bound)) {
    return loom_vector_emit_static_index_out_of_bounds(
        emitter, op, out_of_bounds_axis, out_of_bounds_index,
        out_of_bounds_bound);
  }
  return iree_ok_status();
}

static iree_status_t loom_vector_verify_gather_scatter_access(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op,
    loom_type_t view_type, loom_type_t offsets_type,
    loom_attribute_t static_indices, uint16_t dynamic_index_count) {
  if (!loom_type_is_view(view_type) || !loom_type_is_vector(offsets_type)) {
    return iree_ok_status();
  }

  if (!loom_type_satisfies_constraint(
          offsets_type, LOOM_TYPE_CONSTRAINT_INDEX_OR_NON_I1_INTEGER_ELEMENT)) {
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(loom_vector_verify_dynamic_index_count(
      emitter, op, static_indices, dynamic_index_count));

  uint8_t view_rank = loom_type_rank(view_type);
  if (view_rank == 0) {
    return loom_vector_emit_operand_constraint(
        emitter, op, IREE_SV("view"), view_type, IREE_SV("rank >= 1 view"));
  }
  if (static_indices.count != view_rank) {
    return loom_vector_emit_offset_count_mismatch(
        emitter, op, IREE_SV("view"), static_indices.count, view_rank);
  }

  uint16_t out_of_bounds_axis = 0;
  int64_t out_of_bounds_index = 0;
  int64_t out_of_bounds_bound = 0;
  if (loom_vector_find_static_index_out_of_bounds(
          static_indices, view_type, &out_of_bounds_axis, &out_of_bounds_index,
          &out_of_bounds_bound)) {
    return loom_vector_emit_static_index_out_of_bounds(
        emitter, op, out_of_bounds_axis, out_of_bounds_index,
        out_of_bounds_bound);
  }
  return iree_ok_status();
}

static iree_status_t loom_vector_verify_optional_cache_policy(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op,
    uint16_t cache_scope_attr_index, uint16_t cache_temporal_attr_index,
    loom_cache_policy_access_t access) {
  loom_attribute_t cache_scope_attr = loom_op_attrs(op)[cache_scope_attr_index];
  loom_attribute_t cache_temporal_attr =
      loom_op_attrs(op)[cache_temporal_attr_index];
  bool has_cache_scope = !loom_attr_is_absent(cache_scope_attr);
  bool has_cache_temporal = !loom_attr_is_absent(cache_temporal_attr);
  if (!has_cache_scope && !has_cache_temporal) return iree_ok_status();
  if (!has_cache_scope) {
    return loom_vector_emit_attribute_value_constraint(
        emitter, op, IREE_SV("cache_scope"), 0,
        IREE_SV("present when cache_temporal is present"));
  }
  if (!has_cache_temporal) {
    return loom_vector_emit_attribute_value_constraint(
        emitter, op, IREE_SV("cache_temporal"), 0,
        IREE_SV("present when cache_scope is present"));
  }
  if (cache_scope_attr.kind != LOOM_ATTR_ENUM ||
      cache_temporal_attr.kind != LOOM_ATTR_ENUM) {
    return iree_ok_status();
  }

  uint8_t cache_scope = loom_attr_as_enum(cache_scope_attr);
  uint8_t cache_temporal = loom_attr_as_enum(cache_temporal_attr);
  loom_cache_policy_error_t error =
      loom_cache_policy_validate(cache_scope, cache_temporal, access);
  if (error == LOOM_CACHE_POLICY_ERROR_NONE) return iree_ok_status();
  iree_string_view_t attr_name = loom_cache_policy_error_attr_name(error);
  int64_t actual_value =
      iree_string_view_equal(attr_name, IREE_SV("cache_scope"))
          ? cache_scope
          : cache_temporal;
  return loom_vector_emit_attribute_value_constraint(
      emitter, op, attr_name, actual_value,
      loom_cache_policy_error_expected_constraint(error));
}

iree_status_t loom_vector_poison_verify(const loom_module_t* module,
                                        const loom_op_t* op,
                                        iree_diagnostic_emitter_t emitter) {
  loom_type_t result_type =
      loom_module_value_type(module, loom_vector_poison_result(op));
  if (!loom_type_is_vector(result_type)) return iree_ok_status();
  if (!loom_type_has_static_zero_extent(result_type)) return iree_ok_status();
  return loom_vector_emit_result_constraint(
      emitter, op, IREE_SV("result"), result_type,
      IREE_SV("non-empty vector type; use vector.empty for static zero-lane "
              "values"));
}

iree_status_t loom_vector_empty_verify(const loom_module_t* module,
                                       const loom_op_t* op,
                                       iree_diagnostic_emitter_t emitter) {
  loom_type_t result_type =
      loom_module_value_type(module, loom_vector_empty_result(op));
  if (!loom_type_is_vector(result_type)) return iree_ok_status();
  if (loom_type_has_static_zero_extent(result_type)) return iree_ok_status();
  return loom_vector_emit_result_constraint(
      emitter, op, IREE_SV("result"), result_type,
      IREE_SV("static zero-lane vector type"));
}

iree_status_t loom_vector_broadcast_verify(const loom_module_t* module,
                                           const loom_op_t* op,
                                           iree_diagnostic_emitter_t emitter) {
  loom_type_t source_type =
      loom_module_value_type(module, loom_vector_broadcast_source(op));
  loom_type_t result_type =
      loom_module_value_type(module, loom_vector_broadcast_result(op));
  if (!loom_type_is_vector(source_type) || !loom_type_is_vector(result_type)) {
    return iree_ok_status();
  }

  uint8_t source_rank = loom_type_rank(source_type);
  uint8_t result_rank = loom_type_rank(result_type);
  if (source_rank > result_rank) {
    return loom_vector_emit_rank_mismatch(emitter, op, IREE_SV("source"),
                                          source_rank, IREE_SV("result"),
                                          result_rank);
  }

  uint8_t result_axis_offset = result_rank - source_rank;
  for (uint8_t source_axis = 0; source_axis < source_rank; ++source_axis) {
    uint8_t result_axis = result_axis_offset + source_axis;
    if (loom_type_dim_is_dynamic_at(source_type, source_axis) ||
        loom_type_dim_is_dynamic_at(result_type, result_axis)) {
      continue;
    }
    int64_t source_size =
        loom_type_dim_static_size_at(source_type, source_axis);
    int64_t result_size =
        loom_type_dim_static_size_at(result_type, result_axis);
    if (source_size == 1 || source_size == result_size) continue;
    loom_diagnostic_param_t params[] = {
        loom_param_u32(source_axis),
        loom_param_i64(source_size),
        loom_param_u32(result_axis),
        loom_param_i64(result_size),
    };
    return loom_vector_emit(emitter, op, LOOM_ERR_SHAPE_004, params,
                            IREE_ARRAYSIZE(params));
  }
  return iree_ok_status();
}

iree_status_t loom_vector_load_verify(const loom_module_t* module,
                                      const loom_op_t* op,
                                      iree_diagnostic_emitter_t emitter) {
  loom_type_t view_type =
      loom_module_value_type(module, loom_vector_load_view(op));
  loom_type_t result_type =
      loom_module_value_type(module, loom_vector_load_result(op));
  IREE_RETURN_IF_ERROR(loom_vector_verify_memory_access(
      module, emitter, op, IREE_SV("result"), /*vector_is_result=*/true,
      view_type, result_type, loom_vector_load_static_indices(op),
      loom_vector_load_indices(op).count));
  return loom_vector_verify_optional_cache_policy(
      emitter, op, loom_vector_load_cache_scope_ATTR_INDEX,
      loom_vector_load_cache_temporal_ATTR_INDEX,
      LOOM_CACHE_POLICY_ACCESS_LOAD);
}

iree_status_t loom_vector_store_verify(const loom_module_t* module,
                                       const loom_op_t* op,
                                       iree_diagnostic_emitter_t emitter) {
  loom_type_t view_type =
      loom_module_value_type(module, loom_vector_store_view(op));
  loom_type_t value_type =
      loom_module_value_type(module, loom_vector_store_value(op));
  IREE_RETURN_IF_ERROR(loom_vector_verify_memory_access(
      module, emitter, op, IREE_SV("value"), /*vector_is_result=*/false,
      view_type, value_type, loom_vector_store_static_indices(op),
      loom_vector_store_indices(op).count));
  return loom_vector_verify_optional_cache_policy(
      emitter, op, loom_vector_store_cache_scope_ATTR_INDEX,
      loom_vector_store_cache_temporal_ATTR_INDEX,
      LOOM_CACHE_POLICY_ACCESS_STORE);
}

iree_status_t loom_vector_fragment_load_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter) {
  loom_type_t view_type =
      loom_module_value_type(module, loom_vector_fragment_load_view(op));
  IREE_RETURN_IF_ERROR(loom_vector_verify_fragment_memory_origin(
      emitter, op, view_type, loom_vector_fragment_load_static_indices(op),
      loom_vector_fragment_load_indices(op).count));
  return loom_vector_verify_optional_cache_policy(
      emitter, op, loom_vector_fragment_load_cache_scope_ATTR_INDEX,
      loom_vector_fragment_load_cache_temporal_ATTR_INDEX,
      LOOM_CACHE_POLICY_ACCESS_LOAD);
}

iree_status_t loom_vector_fragment_store_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter) {
  loom_type_t view_type =
      loom_module_value_type(module, loom_vector_fragment_store_view(op));
  IREE_RETURN_IF_ERROR(loom_vector_verify_fragment_memory_origin(
      emitter, op, view_type, loom_vector_fragment_store_static_indices(op),
      loom_vector_fragment_store_indices(op).count));
  return loom_vector_verify_optional_cache_policy(
      emitter, op, loom_vector_fragment_store_cache_scope_ATTR_INDEX,
      loom_vector_fragment_store_cache_temporal_ATTR_INDEX,
      LOOM_CACHE_POLICY_ACCESS_STORE);
}

iree_status_t loom_vector_load_mask_verify(const loom_module_t* module,
                                           const loom_op_t* op,
                                           iree_diagnostic_emitter_t emitter) {
  loom_type_t view_type =
      loom_module_value_type(module, loom_vector_load_mask_view(op));
  loom_type_t result_type =
      loom_module_value_type(module, loom_vector_load_mask_result(op));
  IREE_RETURN_IF_ERROR(loom_vector_verify_memory_access(
      module, emitter, op, IREE_SV("result"), /*vector_is_result=*/true,
      view_type, result_type, loom_vector_load_mask_static_indices(op),
      loom_vector_load_mask_indices(op).count));
  return loom_vector_verify_optional_cache_policy(
      emitter, op, loom_vector_load_mask_cache_scope_ATTR_INDEX,
      loom_vector_load_mask_cache_temporal_ATTR_INDEX,
      LOOM_CACHE_POLICY_ACCESS_LOAD);
}

iree_status_t loom_vector_store_mask_verify(const loom_module_t* module,
                                            const loom_op_t* op,
                                            iree_diagnostic_emitter_t emitter) {
  loom_type_t view_type =
      loom_module_value_type(module, loom_vector_store_mask_view(op));
  loom_type_t value_type =
      loom_module_value_type(module, loom_vector_store_mask_value(op));
  IREE_RETURN_IF_ERROR(loom_vector_verify_memory_access(
      module, emitter, op, IREE_SV("value"), /*vector_is_result=*/false,
      view_type, value_type, loom_vector_store_mask_static_indices(op),
      loom_vector_store_mask_indices(op).count));
  return loom_vector_verify_optional_cache_policy(
      emitter, op, loom_vector_store_mask_cache_scope_ATTR_INDEX,
      loom_vector_store_mask_cache_temporal_ATTR_INDEX,
      LOOM_CACHE_POLICY_ACCESS_STORE);
}

iree_status_t loom_vector_load_expand_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter) {
  loom_type_t view_type =
      loom_module_value_type(module, loom_vector_load_expand_view(op));
  loom_type_t result_type =
      loom_module_value_type(module, loom_vector_load_expand_result(op));
  if (!loom_type_satisfies_constraint(result_type,
                                      LOOM_TYPE_CONSTRAINT_RANK_ONE_VECTOR)) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_vector_verify_memory_access(
      module, emitter, op, IREE_SV("result"), /*vector_is_result=*/true,
      view_type, result_type, loom_vector_load_expand_static_indices(op),
      loom_vector_load_expand_indices(op).count));
  return loom_vector_verify_optional_cache_policy(
      emitter, op, loom_vector_load_expand_cache_scope_ATTR_INDEX,
      loom_vector_load_expand_cache_temporal_ATTR_INDEX,
      LOOM_CACHE_POLICY_ACCESS_LOAD);
}

iree_status_t loom_vector_store_compress_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter) {
  loom_type_t view_type =
      loom_module_value_type(module, loom_vector_store_compress_view(op));
  loom_type_t value_type =
      loom_module_value_type(module, loom_vector_store_compress_value(op));
  if (!loom_type_satisfies_constraint(value_type,
                                      LOOM_TYPE_CONSTRAINT_RANK_ONE_VECTOR)) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_vector_verify_memory_access(
      module, emitter, op, IREE_SV("value"), /*vector_is_result=*/false,
      view_type, value_type, loom_vector_store_compress_static_indices(op),
      loom_vector_store_compress_indices(op).count));
  return loom_vector_verify_optional_cache_policy(
      emitter, op, loom_vector_store_compress_cache_scope_ATTR_INDEX,
      loom_vector_store_compress_cache_temporal_ATTR_INDEX,
      LOOM_CACHE_POLICY_ACCESS_STORE);
}

iree_status_t loom_vector_gather_verify(const loom_module_t* module,
                                        const loom_op_t* op,
                                        iree_diagnostic_emitter_t emitter) {
  loom_type_t view_type =
      loom_module_value_type(module, loom_vector_gather_view(op));
  loom_type_t offsets_type =
      loom_module_value_type(module, loom_vector_gather_offsets(op));
  IREE_RETURN_IF_ERROR(loom_vector_verify_gather_scatter_access(
      emitter, op, view_type, offsets_type,
      loom_vector_gather_static_indices(op),
      loom_vector_gather_indices(op).count));
  return loom_vector_verify_optional_cache_policy(
      emitter, op, loom_vector_gather_cache_scope_ATTR_INDEX,
      loom_vector_gather_cache_temporal_ATTR_INDEX,
      LOOM_CACHE_POLICY_ACCESS_LOAD);
}

iree_status_t loom_vector_scatter_verify(const loom_module_t* module,
                                         const loom_op_t* op,
                                         iree_diagnostic_emitter_t emitter) {
  loom_type_t view_type =
      loom_module_value_type(module, loom_vector_scatter_view(op));
  loom_type_t offsets_type =
      loom_module_value_type(module, loom_vector_scatter_offsets(op));
  IREE_RETURN_IF_ERROR(loom_vector_verify_gather_scatter_access(
      emitter, op, view_type, offsets_type,
      loom_vector_scatter_static_indices(op),
      loom_vector_scatter_indices(op).count));
  return loom_vector_verify_optional_cache_policy(
      emitter, op, loom_vector_scatter_cache_scope_ATTR_INDEX,
      loom_vector_scatter_cache_temporal_ATTR_INDEX,
      LOOM_CACHE_POLICY_ACCESS_STORE);
}

iree_status_t loom_vector_gather_mask_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter) {
  loom_type_t view_type =
      loom_module_value_type(module, loom_vector_gather_mask_view(op));
  loom_type_t offsets_type =
      loom_module_value_type(module, loom_vector_gather_mask_offsets(op));
  IREE_RETURN_IF_ERROR(loom_vector_verify_gather_scatter_access(
      emitter, op, view_type, offsets_type,
      loom_vector_gather_mask_static_indices(op),
      loom_vector_gather_mask_indices(op).count));
  return loom_vector_verify_optional_cache_policy(
      emitter, op, loom_vector_gather_mask_cache_scope_ATTR_INDEX,
      loom_vector_gather_mask_cache_temporal_ATTR_INDEX,
      LOOM_CACHE_POLICY_ACCESS_LOAD);
}

iree_status_t loom_vector_scatter_mask_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter) {
  loom_type_t view_type =
      loom_module_value_type(module, loom_vector_scatter_mask_view(op));
  loom_type_t offsets_type =
      loom_module_value_type(module, loom_vector_scatter_mask_offsets(op));
  IREE_RETURN_IF_ERROR(loom_vector_verify_gather_scatter_access(
      emitter, op, view_type, offsets_type,
      loom_vector_scatter_mask_static_indices(op),
      loom_vector_scatter_mask_indices(op).count));
  return loom_vector_verify_optional_cache_policy(
      emitter, op, loom_vector_scatter_mask_cache_scope_ATTR_INDEX,
      loom_vector_scatter_mask_cache_temporal_ATTR_INDEX,
      LOOM_CACHE_POLICY_ACCESS_STORE);
}

iree_status_t loom_vector_atomic_reduce_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter) {
  loom_type_t value_type =
      loom_module_value_type(module, loom_vector_atomic_reduce_value(op));
  loom_type_t view_type =
      loom_module_value_type(module, loom_vector_atomic_reduce_view(op));
  loom_type_t offsets_type =
      loom_module_value_type(module, loom_vector_atomic_reduce_offsets(op));
  IREE_RETURN_IF_ERROR(loom_vector_verify_gather_scatter_access(
      emitter, op, view_type, offsets_type,
      loom_vector_atomic_reduce_static_indices(op),
      loom_vector_atomic_reduce_indices(op).count));
  IREE_RETURN_IF_ERROR(loom_vector_verify_atomic_kind(
      emitter, op, IREE_SV("value"), value_type,
      loom_vector_atomic_reduce_kind(op), /*allow_exchange=*/false));
  return loom_vector_verify_optional_cache_policy(
      emitter, op, loom_vector_atomic_reduce_cache_scope_ATTR_INDEX,
      loom_vector_atomic_reduce_cache_temporal_ATTR_INDEX,
      LOOM_CACHE_POLICY_ACCESS_ATOMIC);
}

iree_status_t loom_vector_atomic_reduce_mask_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter) {
  loom_type_t value_type =
      loom_module_value_type(module, loom_vector_atomic_reduce_mask_value(op));
  loom_type_t view_type =
      loom_module_value_type(module, loom_vector_atomic_reduce_mask_view(op));
  loom_type_t offsets_type = loom_module_value_type(
      module, loom_vector_atomic_reduce_mask_offsets(op));
  IREE_RETURN_IF_ERROR(loom_vector_verify_gather_scatter_access(
      emitter, op, view_type, offsets_type,
      loom_vector_atomic_reduce_mask_static_indices(op),
      loom_vector_atomic_reduce_mask_indices(op).count));
  IREE_RETURN_IF_ERROR(loom_vector_verify_atomic_kind(
      emitter, op, IREE_SV("value"), value_type,
      loom_vector_atomic_reduce_mask_kind(op), /*allow_exchange=*/false));
  return loom_vector_verify_optional_cache_policy(
      emitter, op, loom_vector_atomic_reduce_mask_cache_scope_ATTR_INDEX,
      loom_vector_atomic_reduce_mask_cache_temporal_ATTR_INDEX,
      LOOM_CACHE_POLICY_ACCESS_ATOMIC);
}

iree_status_t loom_vector_atomic_rmw_verify(const loom_module_t* module,
                                            const loom_op_t* op,
                                            iree_diagnostic_emitter_t emitter) {
  loom_type_t value_type =
      loom_module_value_type(module, loom_vector_atomic_rmw_value(op));
  loom_type_t view_type =
      loom_module_value_type(module, loom_vector_atomic_rmw_view(op));
  loom_type_t offsets_type =
      loom_module_value_type(module, loom_vector_atomic_rmw_offsets(op));
  IREE_RETURN_IF_ERROR(loom_vector_verify_gather_scatter_access(
      emitter, op, view_type, offsets_type,
      loom_vector_atomic_rmw_static_indices(op),
      loom_vector_atomic_rmw_indices(op).count));
  IREE_RETURN_IF_ERROR(loom_vector_verify_atomic_kind(
      emitter, op, IREE_SV("value"), value_type,
      loom_vector_atomic_rmw_kind(op), /*allow_exchange=*/true));
  return loom_vector_verify_optional_cache_policy(
      emitter, op, loom_vector_atomic_rmw_cache_scope_ATTR_INDEX,
      loom_vector_atomic_rmw_cache_temporal_ATTR_INDEX,
      LOOM_CACHE_POLICY_ACCESS_ATOMIC);
}

iree_status_t loom_vector_atomic_rmw_mask_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter) {
  loom_type_t value_type =
      loom_module_value_type(module, loom_vector_atomic_rmw_mask_value(op));
  loom_type_t view_type =
      loom_module_value_type(module, loom_vector_atomic_rmw_mask_view(op));
  loom_type_t offsets_type =
      loom_module_value_type(module, loom_vector_atomic_rmw_mask_offsets(op));
  IREE_RETURN_IF_ERROR(loom_vector_verify_gather_scatter_access(
      emitter, op, view_type, offsets_type,
      loom_vector_atomic_rmw_mask_static_indices(op),
      loom_vector_atomic_rmw_mask_indices(op).count));
  IREE_RETURN_IF_ERROR(loom_vector_verify_atomic_kind(
      emitter, op, IREE_SV("value"), value_type,
      loom_vector_atomic_rmw_mask_kind(op), /*allow_exchange=*/true));
  return loom_vector_verify_optional_cache_policy(
      emitter, op, loom_vector_atomic_rmw_mask_cache_scope_ATTR_INDEX,
      loom_vector_atomic_rmw_mask_cache_temporal_ATTR_INDEX,
      LOOM_CACHE_POLICY_ACCESS_ATOMIC);
}

iree_status_t loom_vector_atomic_cmpxchg_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter) {
  loom_type_t view_type =
      loom_module_value_type(module, loom_vector_atomic_cmpxchg_view(op));
  loom_type_t offsets_type =
      loom_module_value_type(module, loom_vector_atomic_cmpxchg_offsets(op));
  IREE_RETURN_IF_ERROR(loom_vector_verify_gather_scatter_access(
      emitter, op, view_type, offsets_type,
      loom_vector_atomic_cmpxchg_static_indices(op),
      loom_vector_atomic_cmpxchg_indices(op).count));
  IREE_RETURN_IF_ERROR(loom_vector_verify_atomic_cmpxchg_ordering(
      emitter, op, loom_vector_atomic_cmpxchg_success_ordering(op),
      loom_vector_atomic_cmpxchg_failure_ordering(op)));
  return loom_vector_verify_optional_cache_policy(
      emitter, op, loom_vector_atomic_cmpxchg_cache_scope_ATTR_INDEX,
      loom_vector_atomic_cmpxchg_cache_temporal_ATTR_INDEX,
      LOOM_CACHE_POLICY_ACCESS_ATOMIC);
}

iree_status_t loom_vector_extract_verify(const loom_module_t* module,
                                         const loom_op_t* op,
                                         iree_diagnostic_emitter_t emitter) {
  loom_type_t source_type =
      loom_module_value_type(module, loom_vector_extract_source(op));
  loom_type_t result_type =
      loom_module_value_type(module, loom_vector_extract_result(op));
  if (!loom_type_is_vector(source_type)) return iree_ok_status();

  loom_attribute_t static_indices = loom_vector_extract_static_indices(op);
  uint16_t dynamic_index_count = loom_vector_extract_indices(op).count;
  uint16_t expected_dynamic_count =
      loom_vector_dynamic_sentinel_count(static_indices);
  if (dynamic_index_count != expected_dynamic_count) {
    return loom_vector_emit_count_mismatch(
        emitter, op, IREE_SV("indices"), dynamic_index_count,
        IREE_SV("dynamic sentinels"), expected_dynamic_count);
  }

  uint8_t source_rank = loom_type_rank(source_type);
  if (static_indices.count > source_rank) {
    return loom_vector_emit_count_mismatch(
        emitter, op, IREE_SV("static_indices"), static_indices.count,
        IREE_SV("source rank"), source_rank);
  }

  uint16_t out_of_bounds_axis = 0;
  int64_t out_of_bounds_index = 0;
  int64_t out_of_bounds_bound = 0;
  if (loom_vector_find_static_index_out_of_bounds(
          static_indices, source_type, &out_of_bounds_axis,
          &out_of_bounds_index, &out_of_bounds_bound)) {
    return loom_vector_emit_static_index_out_of_bounds(
        emitter, op, out_of_bounds_axis, out_of_bounds_index,
        out_of_bounds_bound);
  }
  return loom_vector_verify_subvalue_type(
      emitter, op, IREE_SV("result"), result_type, IREE_SV("source tail"),
      source_type, (uint8_t)static_indices.count, /*value_is_result=*/true);
}

iree_status_t loom_vector_insert_verify(const loom_module_t* module,
                                        const loom_op_t* op,
                                        iree_diagnostic_emitter_t emitter) {
  loom_type_t value_type =
      loom_module_value_type(module, loom_vector_insert_value(op));
  loom_type_t dest_type =
      loom_module_value_type(module, loom_vector_insert_dest(op));
  if (!loom_type_is_vector(dest_type)) return iree_ok_status();

  loom_attribute_t static_indices = loom_vector_insert_static_indices(op);
  uint16_t dynamic_index_count = loom_vector_insert_indices(op).count;
  uint16_t expected_dynamic_count =
      loom_vector_dynamic_sentinel_count(static_indices);
  if (dynamic_index_count != expected_dynamic_count) {
    return loom_vector_emit_count_mismatch(
        emitter, op, IREE_SV("indices"), dynamic_index_count,
        IREE_SV("dynamic sentinels"), expected_dynamic_count);
  }

  uint8_t dest_rank = loom_type_rank(dest_type);
  if (static_indices.count > dest_rank) {
    return loom_vector_emit_count_mismatch(
        emitter, op, IREE_SV("static_indices"), static_indices.count,
        IREE_SV("dest rank"), dest_rank);
  }

  uint16_t out_of_bounds_axis = 0;
  int64_t out_of_bounds_index = 0;
  int64_t out_of_bounds_bound = 0;
  if (loom_vector_find_static_index_out_of_bounds(
          static_indices, dest_type, &out_of_bounds_axis, &out_of_bounds_index,
          &out_of_bounds_bound)) {
    return loom_vector_emit_static_index_out_of_bounds(
        emitter, op, out_of_bounds_axis, out_of_bounds_index,
        out_of_bounds_bound);
  }
  return loom_vector_verify_subvalue_type(
      emitter, op, IREE_SV("value"), value_type, IREE_SV("dest tail"),
      dest_type, (uint8_t)static_indices.count, /*value_is_result=*/false);
}

iree_status_t loom_vector_slice_verify(const loom_module_t* module,
                                       const loom_op_t* op,
                                       iree_diagnostic_emitter_t emitter) {
  loom_type_t source_type =
      loom_module_value_type(module, loom_vector_slice_source(op));
  loom_type_t result_type =
      loom_module_value_type(module, loom_vector_slice_result(op));
  if (!loom_type_is_vector(source_type) || !loom_type_is_vector(result_type)) {
    return iree_ok_status();
  }

  uint8_t source_rank = loom_type_rank(source_type);
  uint8_t result_rank = loom_type_rank(result_type);
  if (source_rank != result_rank) {
    return loom_vector_emit_rank_mismatch(emitter, op, IREE_SV("result"),
                                          result_rank, IREE_SV("source"),
                                          source_rank);
  }

  loom_attribute_t static_offsets = loom_vector_slice_static_offsets(op);
  uint16_t dynamic_offset_count = loom_vector_slice_offsets(op).count;
  IREE_RETURN_IF_ERROR(loom_vector_verify_dynamic_sentinel_count(
      emitter, op, IREE_SV("offsets"), static_offsets, dynamic_offset_count));

  if (static_offsets.count != source_rank) {
    return loom_vector_emit_offset_count_mismatch(
        emitter, op, IREE_SV("source"), static_offsets.count, source_rank);
  }

  uint16_t out_of_bounds_axis = 0;
  int64_t out_of_bounds_offset = 0;
  int64_t out_of_bounds_extent = 0;
  int64_t out_of_bounds_bound = 0;
  if (loom_vector_find_static_slice_out_of_bounds(
          static_offsets, source_type, result_type, &out_of_bounds_axis,
          &out_of_bounds_offset, &out_of_bounds_extent, &out_of_bounds_bound)) {
    return loom_vector_emit_static_access_out_of_bounds(
        emitter, op, out_of_bounds_axis, out_of_bounds_offset,
        out_of_bounds_extent, out_of_bounds_bound);
  }
  return iree_ok_status();
}

iree_status_t loom_vector_concat_verify(const loom_module_t* module,
                                        const loom_op_t* op,
                                        iree_diagnostic_emitter_t emitter) {
  loom_type_t result_type =
      loom_module_value_type(module, loom_vector_concat_result(op));
  if (!loom_type_is_vector(result_type)) return iree_ok_status();

  loom_value_slice_t inputs = loom_vector_concat_inputs(op);
  if (inputs.count == 0) {
    return loom_vector_emit_count_mismatch(emitter, op, IREE_SV("inputs"), 0,
                                           IREE_SV("minimum"), 1);
  }

  uint8_t result_rank = loom_type_rank(result_type);
  int64_t axis = loom_vector_concat_axis(op);
  if (axis < 0 || axis >= result_rank) return iree_ok_status();

  bool concat_axis_sum_is_static =
      !loom_type_dim_is_dynamic_at(result_type, (uint8_t)axis);
  int64_t concat_axis_sum = 0;
  for (uint16_t i = 0; i < inputs.count; ++i) {
    loom_type_t input_type = loom_module_value_type(module, inputs.values[i]);
    if (!loom_type_is_vector(input_type)) continue;

    uint8_t input_rank = loom_type_rank(input_type);
    if (input_rank != result_rank) {
      return loom_vector_emit_rank_mismatch(emitter, op, IREE_SV("inputs"),
                                            input_rank, IREE_SV("result"),
                                            result_rank);
    }

    for (uint8_t input_axis = 0; input_axis < input_rank; ++input_axis) {
      if (input_axis == (uint8_t)axis) continue;
      if (loom_vector_dim_equals(input_type, input_axis, result_type,
                                 input_axis)) {
        continue;
      }
      return loom_vector_emit_shape_mismatch(emitter, op, IREE_SV("inputs"),
                                             IREE_SV("result"));
    }

    if (loom_type_dim_is_dynamic_at(input_type, (uint8_t)axis)) {
      concat_axis_sum_is_static = false;
      continue;
    }
    if (!concat_axis_sum_is_static) continue;
    int64_t input_axis_size =
        loom_type_dim_static_size_at(input_type, (uint8_t)axis);
    if (!iree_checked_add_i64(concat_axis_sum, input_axis_size,
                              &concat_axis_sum)) {
      return loom_vector_emit_result_constraint(
          emitter, op, IREE_SV("result"), result_type,
          IREE_SV("representable static concat axis extent"));
    }
  }

  if (!concat_axis_sum_is_static) return iree_ok_status();
  int64_t result_axis_size =
      loom_type_dim_static_size_at(result_type, (uint8_t)axis);
  if (concat_axis_sum == result_axis_size) return iree_ok_status();
  return loom_vector_emit_result_constraint(
      emitter, op, IREE_SV("result"), result_type,
      IREE_SV("concat axis extent equal to sum of input extents"));
}

iree_status_t loom_vector_transpose_verify(const loom_module_t* module,
                                           const loom_op_t* op,
                                           iree_diagnostic_emitter_t emitter) {
  loom_type_t source_type =
      loom_module_value_type(module, loom_vector_transpose_source(op));
  loom_type_t result_type =
      loom_module_value_type(module, loom_vector_transpose_result(op));
  if (!loom_type_is_vector(source_type) || !loom_type_is_vector(result_type)) {
    return iree_ok_status();
  }

  uint8_t source_rank = loom_type_rank(source_type);
  uint8_t result_rank = loom_type_rank(result_type);
  if (source_rank != result_rank) {
    return loom_vector_emit_rank_mismatch(emitter, op, IREE_SV("result"),
                                          result_rank, IREE_SV("source"),
                                          source_rank);
  }

  loom_attribute_t permutation = loom_vector_transpose_permutation(op);
  if (permutation.count != source_rank) {
    return loom_vector_emit_count_mismatch(emitter, op, IREE_SV("permutation"),
                                           permutation.count,
                                           IREE_SV("source rank"), source_rank);
  }

  uint32_t seen_axes = 0;
  for (uint16_t result_axis = 0; result_axis < permutation.count;
       ++result_axis) {
    int64_t source_axis = permutation.i64_array[result_axis];
    if (source_axis < 0 || source_axis >= source_rank) {
      loom_diagnostic_param_t params[] = {
          loom_param_i64(source_axis),
          loom_param_i64(source_rank),
      };
      return loom_vector_emit(emitter, op, LOOM_ERR_SUBRANGE_002, params,
                              IREE_ARRAYSIZE(params));
    }

    uint32_t axis_bit = 1u << (uint32_t)source_axis;
    if (iree_all_bits_set(seen_axes, axis_bit)) {
      return loom_vector_emit_attribute_value_constraint(
          emitter, op, IREE_SV("permutation"), source_axis,
          IREE_SV("each source axis exactly once"));
    }
    seen_axes |= axis_bit;

    if (loom_vector_dim_equals(source_type, (uint8_t)source_axis, result_type,
                               (uint8_t)result_axis)) {
      continue;
    }
    return loom_vector_emit_shape_mismatch(emitter, op, IREE_SV("result"),
                                           IREE_SV("permutation(source)"));
  }
  return iree_ok_status();
}

iree_status_t loom_vector_shuffle_verify(const loom_module_t* module,
                                         const loom_op_t* op,
                                         iree_diagnostic_emitter_t emitter) {
  loom_type_t source_type =
      loom_module_value_type(module, loom_vector_shuffle_source(op));
  if (!loom_type_is_vector(source_type)) return iree_ok_status();
  if (!loom_type_satisfies_constraint(
          source_type, LOOM_TYPE_CONSTRAINT_ALL_STATIC_RANK_ONE_VECTOR)) {
    return iree_ok_status();
  }

  int64_t source_lane_count = loom_type_dim_static_size_at(source_type, 0);
  loom_attribute_t source_lanes = loom_vector_shuffle_source_lanes(op);
  if (source_lanes.count != (uint64_t)source_lane_count) {
    return loom_vector_emit_count_mismatch(
        emitter, op, IREE_SV("source_lanes"), source_lanes.count,
        IREE_SV("source lane count"), (uint64_t)source_lane_count);
  }

  for (uint16_t result_lane = 0; result_lane < source_lanes.count;
       ++result_lane) {
    int64_t source_lane = source_lanes.i64_array[result_lane];
    if (source_lane >= 0 && source_lane < source_lane_count) continue;
    return loom_vector_emit_static_access_out_of_bounds(
        emitter, op, result_lane, source_lane, 1, source_lane_count);
  }
  return iree_ok_status();
}

iree_status_t loom_vector_interleave_verify(const loom_module_t* module,
                                            const loom_op_t* op,
                                            iree_diagnostic_emitter_t emitter) {
  loom_type_t even_type =
      loom_module_value_type(module, loom_vector_interleave_even(op));
  loom_type_t result_type =
      loom_module_value_type(module, loom_vector_interleave_result(op));
  if (!loom_type_is_vector(even_type) || !loom_type_is_vector(result_type)) {
    return iree_ok_status();
  }

  uint8_t even_rank = loom_type_rank(even_type);
  uint8_t result_rank = loom_type_rank(result_type);
  if (result_rank != even_rank) {
    return loom_vector_emit_rank_mismatch(emitter, op, IREE_SV("result"),
                                          result_rank, IREE_SV("even"),
                                          even_rank);
  }

  int64_t axis = loom_vector_interleave_axis(op);
  if (axis < 0 || axis >= even_rank) return iree_ok_status();

  for (uint8_t i = 0; i < even_rank; ++i) {
    if (i == (uint8_t)axis) continue;
    if (loom_vector_dim_equals(even_type, i, result_type, i)) continue;
    return loom_vector_emit_shape_mismatch(emitter, op, IREE_SV("result"),
                                           IREE_SV("even"));
  }

  if (loom_type_dim_is_dynamic_at(even_type, (uint8_t)axis) ||
      loom_type_dim_is_dynamic_at(result_type, (uint8_t)axis)) {
    return iree_ok_status();
  }

  int64_t expected_result_axis_size = 0;
  int64_t even_axis_size =
      loom_type_dim_static_size_at(even_type, (uint8_t)axis);
  if (!iree_checked_mul_i64(even_axis_size, 2, &expected_result_axis_size)) {
    return loom_vector_emit_result_constraint(
        emitter, op, IREE_SV("result"), result_type,
        IREE_SV("representable interleave axis extent"));
  }

  int64_t result_axis_size =
      loom_type_dim_static_size_at(result_type, (uint8_t)axis);
  if (result_axis_size == expected_result_axis_size) return iree_ok_status();
  return loom_vector_emit_result_constraint(
      emitter, op, IREE_SV("result"), result_type,
      IREE_SV("interleave axis extent twice input axis extent"));
}

iree_status_t loom_vector_deinterleave_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter) {
  loom_type_t source_type =
      loom_module_value_type(module, loom_vector_deinterleave_source(op));
  loom_value_slice_t results = loom_vector_deinterleave_results(op);
  if (results.count != 2) {
    return loom_vector_emit_count_mismatch(emitter, op, IREE_SV("results"),
                                           results.count,
                                           IREE_SV("required result count"), 2);
  }
  loom_type_t even_type = loom_module_value_type(module, results.values[0]);
  if (!loom_type_is_vector(source_type) || !loom_type_is_vector(even_type)) {
    return iree_ok_status();
  }

  uint8_t source_rank = loom_type_rank(source_type);
  uint8_t even_rank = loom_type_rank(even_type);
  if (even_rank != source_rank) {
    return loom_vector_emit_rank_mismatch(emitter, op, IREE_SV("even"),
                                          even_rank, IREE_SV("source"),
                                          source_rank);
  }

  int64_t axis = loom_vector_deinterleave_axis(op);
  if (axis < 0 || axis >= source_rank) return iree_ok_status();

  for (uint8_t i = 0; i < source_rank; ++i) {
    if (i == (uint8_t)axis) continue;
    if (loom_vector_dim_equals(source_type, i, even_type, i)) continue;
    return loom_vector_emit_shape_mismatch(emitter, op, IREE_SV("even"),
                                           IREE_SV("source"));
  }

  bool source_axis_is_static =
      !loom_type_dim_is_dynamic_at(source_type, (uint8_t)axis);
  bool even_axis_is_static =
      !loom_type_dim_is_dynamic_at(even_type, (uint8_t)axis);

  int64_t source_axis_size =
      source_axis_is_static
          ? loom_type_dim_static_size_at(source_type, (uint8_t)axis)
          : 0;
  if (source_axis_is_static && (source_axis_size & 1) != 0) {
    return loom_vector_emit_operand_constraint(
        emitter, op, IREE_SV("source"), source_type,
        IREE_SV("even deinterleave axis extent"));
  }

  if (!even_axis_is_static) return iree_ok_status();
  int64_t expected_source_axis_size = 0;
  int64_t even_axis_size =
      loom_type_dim_static_size_at(even_type, (uint8_t)axis);
  if (!iree_checked_mul_i64(even_axis_size, 2, &expected_source_axis_size)) {
    return loom_vector_emit_result_constraint(
        emitter, op, IREE_SV("even"), even_type,
        IREE_SV("representable deinterleave source axis extent"));
  }

  if (!source_axis_is_static || source_axis_size == expected_source_axis_size) {
    return iree_ok_status();
  }
  return loom_vector_emit_result_constraint(
      emitter, op, IREE_SV("even"), even_type,
      IREE_SV("deinterleave axis extent half of source axis extent"));
}

static bool loom_vector_try_get_splat_i64_constant(const loom_module_t* module,
                                                   loom_value_id_t value_id,
                                                   int64_t* out_value) {
  const loom_value_t* value = loom_module_value(module, value_id);
  if (loom_value_is_block_arg(value)) return false;
  const loom_op_t* def_op = loom_value_def_op(value);
  if (!def_op || !loom_vector_constant_isa(def_op)) return false;
  loom_attribute_t attr = loom_vector_constant_value(def_op);
  if (attr.kind != LOOM_ATTR_I64) return false;
  *out_value = loom_attr_as_i64(attr);
  return true;
}

static bool loom_vector_try_get_scalar_f64_constant(const loom_module_t* module,
                                                    loom_value_id_t value_id,
                                                    double* out_value) {
  const loom_value_t* value = loom_module_value(module, value_id);
  if (loom_value_is_block_arg(value)) return false;
  const loom_op_t* def_op = loom_value_def_op(value);
  if (!def_op || !loom_scalar_constant_isa(def_op)) return false;
  loom_attribute_t attr = loom_scalar_constant_value(def_op);
  if (attr.kind != LOOM_ATTR_F64) return false;
  *out_value = loom_attr_as_f64(attr);
  return true;
}

static bool loom_vector_try_get_scalar_i64_constant(const loom_module_t* module,
                                                    loom_value_id_t value_id,
                                                    int64_t* out_value) {
  const loom_value_t* value = loom_module_value(module, value_id);
  if (loom_value_is_block_arg(value)) return false;
  const loom_op_t* def_op = loom_value_def_op(value);
  if (!def_op) return false;
  if (!loom_scalar_constant_isa(def_op)) return false;
  loom_attribute_t attr = loom_scalar_constant_value(def_op);
  if (attr.kind != LOOM_ATTR_I64) return false;
  *out_value = loom_attr_as_i64(attr);
  return true;
}

static bool loom_vector_unsigned_code_capacity_covers(int32_t bitwidth,
                                                      int64_t max_code) {
  if (bitwidth <= 0 || max_code < 0) return false;
  if (bitwidth >= 63) return true;
  return (uint64_t)max_code < (UINT64_C(1) << bitwidth);
}

static iree_status_t loom_vector_verify_quantize_result_capacity(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op,
    loom_type_t thresholds_type, loom_type_t result_type) {
  if (!loom_type_is_vector(thresholds_type) ||
      !loom_type_is_vector(result_type)) {
    return iree_ok_status();
  }
  if (loom_type_rank(thresholds_type) != 1 ||
      loom_type_dim_is_dynamic_at(thresholds_type, 0)) {
    return iree_ok_status();
  }

  loom_scalar_type_t result_element_type = loom_type_element_type(result_type);
  if (!loom_scalar_type_is_integer(result_element_type)) {
    return iree_ok_status();
  }

  int64_t threshold_count = loom_type_dim_static_size_at(thresholds_type, 0);
  int32_t result_bitwidth = loom_scalar_type_bitwidth(result_element_type);
  if (loom_vector_unsigned_code_capacity_covers(result_bitwidth,
                                                threshold_count)) {
    return iree_ok_status();
  }
  return loom_vector_emit_result_code_capacity(emitter, op, IREE_SV("result"),
                                               result_type, threshold_count);
}

static iree_status_t loom_vector_verify_static_threshold_order(
    const loom_module_t* module, iree_diagnostic_emitter_t emitter,
    const loom_op_t* op, loom_value_id_t thresholds_value) {
  const loom_value_t* value = loom_module_value(module, thresholds_value);
  if (loom_value_is_block_arg(value)) return iree_ok_status();
  const loom_op_t* def_op = loom_value_def_op(value);
  if (!def_op) return iree_ok_status();

  if (loom_vector_constant_isa(def_op)) {
    loom_attribute_t attr = loom_vector_constant_value(def_op);
    if (attr.kind != LOOM_ATTR_F64 || !isnan(loom_attr_as_f64(attr))) {
      return iree_ok_status();
    }
    return loom_vector_emit_static_threshold_order(
        emitter, op, IREE_SV("thresholds"), /*left_index=*/0,
        /*right_index=*/0);
  }

  if (!loom_vector_from_elements_isa(def_op)) return iree_ok_status();

  loom_value_slice_t elements = loom_vector_from_elements_elements(def_op);
  double previous_value = 0.0;
  for (uint32_t i = 0; i < elements.count; ++i) {
    double value = 0.0;
    if (!loom_vector_try_get_scalar_f64_constant(module, elements.values[i],
                                                 &value)) {
      return iree_ok_status();
    }
    if (isnan(value)) {
      return loom_vector_emit_static_threshold_order(
          emitter, op, IREE_SV("thresholds"), i, i);
    }
    if (i > 0 && previous_value > value) {
      return loom_vector_emit_static_threshold_order(
          emitter, op, IREE_SV("thresholds"), i - 1, i);
    }
    previous_value = value;
  }
  return iree_ok_status();
}

iree_status_t loom_vector_table_lookup_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter) {
  loom_value_id_t table_value = loom_vector_table_lookup_table(op);
  loom_value_id_t indices_value = loom_vector_table_lookup_indices(op);
  loom_type_t table_type = loom_module_value_type(module, table_value);
  loom_type_t indices_type = loom_module_value_type(module, indices_value);
  if (!loom_type_is_vector(table_type) || !loom_type_is_vector(indices_type)) {
    return iree_ok_status();
  }
  if (!loom_type_satisfies_constraint(table_type,
                                      LOOM_TYPE_CONSTRAINT_RANK_ONE_VECTOR)) {
    return iree_ok_status();
  }

  if (!loom_type_satisfies_constraint(
          indices_type, LOOM_TYPE_CONSTRAINT_INDEX_OR_NON_I1_INTEGER_ELEMENT)) {
    return iree_ok_status();
  }
  if (loom_type_dim_is_dynamic_at(table_type, 0)) return iree_ok_status();

  int64_t table_lane_count = loom_type_dim_static_size_at(table_type, 0);
  int64_t splat_index = 0;
  if (!loom_vector_try_get_splat_i64_constant(module, indices_value,
                                              &splat_index)) {
    return iree_ok_status();
  }
  if (splat_index >= 0 && splat_index < table_lane_count) {
    return iree_ok_status();
  }
  return loom_vector_emit_static_index_out_of_bounds(
      emitter, op, /*axis=*/0, splat_index, table_lane_count);
}

iree_status_t loom_vector_table_quantize_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter) {
  loom_value_id_t thresholds_value = loom_vector_table_quantize_thresholds(op);
  loom_type_t input_type =
      loom_module_value_type(module, loom_vector_table_quantize_input(op));
  loom_type_t thresholds_type =
      loom_module_value_type(module, thresholds_value);
  loom_type_t result_type =
      loom_module_value_type(module, loom_vector_table_quantize_result(op));
  if (!loom_type_is_vector(input_type) ||
      !loom_type_is_vector(thresholds_type) ||
      !loom_type_is_vector(result_type)) {
    return iree_ok_status();
  }
  if (!loom_type_satisfies_constraint(thresholds_type,
                                      LOOM_TYPE_CONSTRAINT_RANK_ONE_VECTOR)) {
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(loom_vector_verify_quantize_result_capacity(
      emitter, op, thresholds_type, result_type));
  return loom_vector_verify_static_threshold_order(module, emitter, op,
                                                   thresholds_value);
}

static iree_string_view_t loom_vector_transform_input_elems_param_name(void) {
  return IREE_SV("input_elems");
}

static iree_string_view_t loom_vector_transform_output_elems_param_name(void) {
  return IREE_SV("output_elems");
}

static iree_string_view_t loom_vector_transform_signs_param_name(void) {
  return IREE_SV("signs");
}

static iree_string_view_t loom_vector_transform_permutation_param_name(void) {
  return IREE_SV("permutation");
}

static iree_string_view_t loom_vector_transform_matrix_param_name(void) {
  return IREE_SV("matrix");
}

static bool loom_vector_string_id_equal(const loom_module_t* module,
                                        loom_string_id_t string_id,
                                        iree_string_view_t expected) {
  if (string_id == LOOM_STRING_ID_INVALID ||
      string_id >= module->strings.count) {
    return false;
  }
  return iree_string_view_equal(module->strings.entries[string_id], expected);
}

static const loom_named_attr_t* loom_vector_find_named_attr(
    const loom_module_t* module, loom_named_attr_slice_t attrs,
    iree_string_view_t name) {
  for (iree_host_size_t i = 0; i < attrs.count; ++i) {
    const loom_named_attr_t* entry = &attrs.entries[i];
    if (loom_vector_string_id_equal(module, entry->name_id, name)) {
      return entry;
    }
  }
  return NULL;
}

static iree_status_t loom_vector_emit_auxiliary_key_error(
    const loom_module_t* module, iree_diagnostic_emitter_t emitter,
    const loom_op_t* op, const loom_error_def_t* error,
    iree_string_view_t key_name) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(loom_op_name(module, op)),
      loom_param_string(key_name),
  };
  return loom_vector_emit(emitter, op, error, params, IREE_ARRAYSIZE(params));
}

static iree_status_t loom_vector_emit_unknown_auxiliary_key(
    const loom_module_t* module, iree_diagnostic_emitter_t emitter,
    const loom_op_t* op, iree_string_view_t key_name) {
  return loom_vector_emit_auxiliary_key_error(module, emitter, op,
                                              LOOM_ERR_ENCODING_015, key_name);
}

static iree_status_t loom_vector_emit_missing_auxiliary_key(
    const loom_module_t* module, iree_diagnostic_emitter_t emitter,
    const loom_op_t* op, iree_string_view_t key_name) {
  return loom_vector_emit_auxiliary_key_error(module, emitter, op,
                                              LOOM_ERR_ENCODING_016, key_name);
}

static bool loom_vector_query_static_storage_schema_for_schema_value(
    const loom_module_t* module, loom_value_id_t schema_value,
    loom_value_fact_storage_schema_t* out_schema) {
  *out_schema = (loom_value_fact_storage_schema_t){0};
  if (schema_value == LOOM_VALUE_ID_INVALID ||
      schema_value >= module->values.count) {
    return false;
  }
  const loom_value_t* value = loom_module_value(module, schema_value);
  if (loom_value_is_block_arg(value)) return false;
  const loom_op_t* defining_op = loom_value_def_op(value);
  if (!defining_op) return false;
  if (loom_encoding_define_isa(defining_op)) {
    return loom_encoding_query_static_storage_schema(
        module, loom_encoding_define_spec(defining_op), out_schema);
  }
  if (loom_encoding_assume_spec_isa(defining_op)) {
    return loom_encoding_query_static_storage_schema(
        module, loom_encoding_assume_spec_spec(defining_op), out_schema);
  }
  return false;
}

static iree_status_t loom_vector_verify_required_auxiliary_key(
    const loom_module_t* module, iree_diagnostic_emitter_t emitter,
    const loom_op_t* op,
    loom_vector_encoding_auxiliary_key_flags_t present_keys,
    loom_vector_encoding_auxiliary_key_t key) {
  loom_vector_encoding_auxiliary_key_flags_t required_key =
      loom_vector_encoding_auxiliary_key_flag(key);
  if (required_key != 0 && iree_any_bit_set(present_keys, required_key)) {
    return iree_ok_status();
  }
  iree_string_view_t key_name = loom_vector_encoding_auxiliary_key_name(key);
  return loom_vector_emit_missing_auxiliary_key(module, emitter, op, key_name);
}

static iree_status_t loom_vector_verify_required_auxiliary_keys(
    const loom_module_t* module, iree_diagnostic_emitter_t emitter,
    const loom_op_t* op,
    loom_vector_encoding_auxiliary_key_flags_t present_keys,
    loom_vector_encoding_auxiliary_key_flags_t required_keys) {
  for (uint8_t i = 0; i < LOOM_VECTOR_ENCODING_AUXILIARY_KEY_COUNT_; ++i) {
    loom_vector_encoding_auxiliary_key_t key =
        (loom_vector_encoding_auxiliary_key_t)i;
    loom_vector_encoding_auxiliary_key_flags_t key_flag =
        loom_vector_encoding_auxiliary_key_flag(key);
    if (!iree_any_bit_set(required_keys, key_flag)) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_vector_verify_required_auxiliary_key(
        module, emitter, op, present_keys, key));
  }
  return iree_ok_status();
}

static iree_status_t loom_vector_verify_encoded_auxiliary(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter, loom_value_id_t schema_value,
    loom_value_slice_t auxiliary_values,
    loom_named_attr_slice_t auxiliary_names) {
  loom_vector_encoding_auxiliary_view_t auxiliary_view;
  iree_string_view_t unknown_key = iree_string_view_empty();
  if (!loom_vector_encoding_auxiliary_view_resolve(
          module, auxiliary_values, auxiliary_names, &auxiliary_view,
          &unknown_key)) {
    return loom_vector_emit_unknown_auxiliary_key(module, emitter, op,
                                                  unknown_key);
  }

  loom_value_fact_storage_schema_t storage_schema = {0};
  if (!loom_vector_query_static_storage_schema_for_schema_value(
          module, schema_value, &storage_schema)) {
    return iree_ok_status();
  }
  if (loom_value_fact_encoded_operand_schema_is_unknown(
          storage_schema.encoded_operand)) {
    return iree_ok_status();
  }
  loom_vector_encoding_auxiliary_key_flags_t required_keys = 0;
  if (!loom_vector_encoding_auxiliary_required_keys_from_schema(
          storage_schema.encoded_operand, &required_keys, NULL)) {
    return loom_vector_emit_missing_auxiliary_key(module, emitter, op,
                                                  IREE_SV("scale"));
  }
  return loom_vector_verify_required_auxiliary_keys(
      module, emitter, op, auxiliary_view.present_keys, required_keys);
}

iree_status_t loom_vector_decode_verify(const loom_module_t* module,
                                        const loom_op_t* op,
                                        iree_diagnostic_emitter_t emitter) {
  return loom_vector_verify_encoded_auxiliary(
      module, op, emitter, loom_vector_decode_schema(op),
      loom_vector_decode_auxiliary(op), loom_vector_decode_auxiliary_names(op));
}

iree_status_t loom_vector_encode_verify(const loom_module_t* module,
                                        const loom_op_t* op,
                                        iree_diagnostic_emitter_t emitter) {
  return loom_vector_verify_encoded_auxiliary(
      module, op, emitter, loom_vector_encode_schema(op),
      loom_vector_encode_auxiliary(op), loom_vector_encode_auxiliary_names(op));
}

static iree_status_t loom_vector_verify_fragment_schema_type(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter, loom_value_id_t schema_value) {
  if (schema_value == LOOM_VALUE_ID_INVALID ||
      schema_value >= module->values.count) {
    return iree_ok_status();
  }
  loom_type_t schema_type = loom_module_value_type(module, schema_value);
  if (loom_type_satisfies_constraint(schema_type,
                                     LOOM_TYPE_CONSTRAINT_ENCODING_SCHEMA)) {
    return iree_ok_status();
  }
  return loom_vector_emit_operand_constraint(
      emitter, op, IREE_SV("schema"), schema_type,
      iree_make_cstring_view(
          loom_type_constraint_name(LOOM_TYPE_CONSTRAINT_ENCODING_SCHEMA)));
}

static iree_status_t loom_vector_verify_fragment_auxiliary_types(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter,
    loom_vector_encoding_auxiliary_view_t auxiliary) {
  for (uint8_t i = 0; i < LOOM_VECTOR_ENCODING_AUXILIARY_KEY_COUNT_; ++i) {
    loom_vector_encoding_auxiliary_key_t key =
        (loom_vector_encoding_auxiliary_key_t)i;
    loom_vector_encoding_auxiliary_key_flags_t key_flag =
        loom_vector_encoding_auxiliary_key_flag(key);
    if (!iree_any_bit_set(auxiliary.present_keys, key_flag)) {
      continue;
    }
    loom_value_id_t value = auxiliary.values[key];
    if (value == LOOM_VALUE_ID_INVALID || value >= module->values.count) {
      continue;
    }
    loom_type_t value_type = loom_module_value_type(module, value);
    if (loom_type_satisfies_constraint(value_type,
                                       LOOM_TYPE_CONSTRAINT_VECTOR)) {
      continue;
    }
    return loom_vector_emit_operand_constraint(
        emitter, op, loom_vector_encoding_auxiliary_key_name(key), value_type,
        iree_make_cstring_view(
            loom_type_constraint_name(LOOM_TYPE_CONSTRAINT_VECTOR)));
  }
  return iree_ok_status();
}

iree_status_t loom_vector_fragment_verify(const loom_module_t* module,
                                          const loom_op_t* op,
                                          iree_diagnostic_emitter_t emitter) {
  loom_vector_fragment_parameter_view_t parameters;
  iree_string_view_t unknown_key = iree_string_view_empty();
  if (!loom_vector_fragment_parameter_view_resolve(
          module, loom_vector_fragment_params(op),
          loom_vector_fragment_param_names(op), &parameters, &unknown_key)) {
    return loom_vector_emit_unknown_auxiliary_key(module, emitter, op,
                                                  unknown_key);
  }

  if (!parameters.has_schema && parameters.auxiliary.present_keys != 0) {
    return loom_vector_emit_missing_auxiliary_key(module, emitter, op,
                                                  IREE_SV("schema"));
  }

  if (!parameters.has_schema) {
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(loom_vector_verify_fragment_schema_type(
      module, op, emitter, parameters.schema_value_id));
  IREE_RETURN_IF_ERROR(loom_vector_verify_fragment_auxiliary_types(
      module, op, emitter, parameters.auxiliary));

  loom_value_fact_storage_schema_t storage_schema = {0};
  if (!loom_vector_query_static_storage_schema_for_schema_value(
          module, parameters.schema_value_id, &storage_schema)) {
    return iree_ok_status();
  }
  if (loom_value_fact_encoded_operand_schema_is_unknown(
          storage_schema.encoded_operand)) {
    return iree_ok_status();
  }

  loom_vector_encoding_auxiliary_key_flags_t required_keys = 0;
  if (!loom_vector_encoding_auxiliary_required_keys_from_schema(
          storage_schema.encoded_operand, &required_keys, NULL)) {
    return loom_vector_emit_missing_auxiliary_key(module, emitter, op,
                                                  IREE_SV("scale"));
  }
  return loom_vector_verify_required_auxiliary_keys(
      module, emitter, op, parameters.auxiliary.present_keys, required_keys);
}

static iree_status_t loom_vector_emit_encoding_dynamic_type_error(
    const loom_module_t* module, iree_diagnostic_emitter_t emitter,
    const loom_op_t* op, iree_string_view_t encoding_name,
    iree_string_view_t param_name, loom_value_id_t value_id,
    iree_string_view_t expected_type) {
  loom_type_t actual_type = loom_module_value_type(module, value_id);
  loom_diagnostic_param_t params[] = {
      loom_param_string(encoding_name),
      loom_param_string(param_name),
      loom_param_type(actual_type),
      loom_param_string(expected_type),
  };
  return loom_vector_emit(emitter, op, LOOM_ERR_ENCODING_009, params,
                          IREE_ARRAYSIZE(params));
}

static bool loom_vector_i64_is_power_of_two(int64_t value) {
  if (value <= 0) return false;
  return (value & (value - 1)) == 0;
}

static iree_status_t loom_vector_transform_verify_extent_param(
    const loom_module_t* module, iree_diagnostic_emitter_t emitter,
    const loom_op_t* op, const loom_encoding_define_param_view_t* params,
    iree_string_view_t param_name, iree_string_view_t field_name,
    loom_type_t field_type, uint8_t axis, bool field_is_result) {
  const loom_named_attr_t* static_param =
      loom_vector_find_named_attr(module, params->static_attrs, param_name);
  const loom_named_attr_t* dynamic_param =
      loom_vector_find_named_attr(module, params->dynamic_names, param_name);
  if (!static_param && !dynamic_param) return iree_ok_status();

  if (static_param) {
    if (static_param->value.kind != LOOM_ATTR_I64) {
      return iree_ok_status();
    }
    int64_t expected_extent = loom_attr_as_i64(static_param->value);
    if (expected_extent <= 0) {
      return iree_ok_status();
    }
    if (loom_type_dim_is_dynamic_at(field_type, axis)) {
      return loom_vector_emit_field_constraint(
          emitter, op, field_is_result, field_name, field_type,
          IREE_SV("last axis matching static transform extent"));
    }
    int64_t actual_extent = loom_type_dim_static_size_at(field_type, axis);
    if (actual_extent == expected_extent) return iree_ok_status();
    return loom_vector_emit_field_constraint(
        emitter, op, field_is_result, field_name, field_type,
        IREE_SV("last axis matching static transform extent"));
  }

  loom_value_id_t value_id = LOOM_VALUE_ID_INVALID;
  if (!loom_encoding_define_dynamic_param_value(params, dynamic_param,
                                                &value_id)) {
    return iree_ok_status();
  }
  if (!loom_type_equal(loom_module_value_type(module, value_id),
                       loom_type_scalar(LOOM_SCALAR_TYPE_INDEX))) {
    return iree_ok_status();
  }
  if (!loom_type_dim_is_dynamic_at(field_type, axis) ||
      loom_type_dim_value_id_at(field_type, axis) != value_id) {
    return loom_vector_emit_field_constraint(
        emitter, op, field_is_result, field_name, field_type,
        IREE_SV("last axis matching dynamic transform extent"));
  }
  return iree_ok_status();
}

static iree_status_t loom_vector_transform_verify_permutation_lane(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op, uint8_t source_axis,
    int64_t permutation_lane, int64_t source_lane_count) {
  if (permutation_lane >= 0 && permutation_lane < source_lane_count) {
    return iree_ok_status();
  }
  return loom_vector_emit_static_index_out_of_bounds(
      emitter, op, source_axis, permutation_lane, source_lane_count);
}

static iree_status_t loom_vector_transform_verify_static_permutation_values(
    const loom_module_t* module, iree_diagnostic_emitter_t emitter,
    const loom_op_t* op, loom_type_t source_type,
    loom_value_id_t permutation_value) {
  uint8_t last_axis = (uint8_t)(loom_type_rank(source_type) - 1);
  if (loom_type_dim_is_dynamic_at(source_type, last_axis)) {
    return iree_ok_status();
  }
  int64_t source_lane_count =
      loom_type_dim_static_size_at(source_type, last_axis);
  const loom_value_t* value = loom_module_value(module, permutation_value);
  if (loom_value_is_block_arg(value)) return iree_ok_status();
  const loom_op_t* def_op = loom_value_def_op(value);
  if (!def_op) return iree_ok_status();

  if (loom_vector_constant_isa(def_op)) {
    loom_attribute_t attr = loom_vector_constant_value(def_op);
    if (attr.kind != LOOM_ATTR_I64) return iree_ok_status();
    int64_t permutation_lane = loom_attr_as_i64(attr);
    IREE_RETURN_IF_ERROR(loom_vector_transform_verify_permutation_lane(
        emitter, op, last_axis, permutation_lane, source_lane_count));
    if (source_lane_count <= 1) return iree_ok_status();
    return loom_vector_emit_attribute_value_constraint(
        emitter, op, IREE_SV("permutation"), permutation_lane,
        IREE_SV("unique lane indices"));
  }

  if (!loom_vector_from_elements_isa(def_op)) return iree_ok_status();
  loom_value_slice_t elements = loom_vector_from_elements_elements(def_op);
  for (uint16_t i = 0; i < elements.count; ++i) {
    int64_t permutation_lane = 0;
    if (!loom_vector_try_get_scalar_i64_constant(module, elements.values[i],
                                                 &permutation_lane)) {
      return iree_ok_status();
    }
    IREE_RETURN_IF_ERROR(loom_vector_transform_verify_permutation_lane(
        emitter, op, last_axis, permutation_lane, source_lane_count));
    for (uint16_t j = 0; j < i; ++j) {
      int64_t previous_lane = 0;
      if (!loom_vector_try_get_scalar_i64_constant(module, elements.values[j],
                                                   &previous_lane)) {
        return iree_ok_status();
      }
      if (previous_lane != permutation_lane) continue;
      return loom_vector_emit_attribute_value_constraint(
          emitter, op, IREE_SV("permutation"), permutation_lane,
          IREE_SV("unique lane indices"));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_vector_transform_verify_dynamic_param_shapes(
    const loom_module_t* module, iree_diagnostic_emitter_t emitter,
    const loom_op_t* op, const loom_op_t* define_op,
    const loom_encoding_define_param_view_t* params,
    iree_string_view_t encoding_name, loom_type_t source_type,
    loom_type_t result_type) {
  const loom_named_attr_t* signs_param = loom_vector_find_named_attr(
      module, params->dynamic_names, loom_vector_transform_signs_param_name());
  if (signs_param) {
    loom_value_id_t signs_value = LOOM_VALUE_ID_INVALID;
    if (!loom_encoding_define_dynamic_param_value(params, signs_param,
                                                  &signs_value)) {
      return iree_ok_status();
    }
    loom_type_t signs_type = loom_module_value_type(module, signs_value);
    if (loom_type_is_vector(signs_type) &&
        loom_type_element_type(signs_type) == LOOM_SCALAR_TYPE_I1 &&
        !loom_vector_shapes_match(signs_type, source_type)) {
      return loom_vector_emit_encoding_dynamic_type_error(
          module, emitter, define_op, encoding_name,
          loom_vector_transform_signs_param_name(), signs_value,
          IREE_SV("i1 vector with source shape"));
    }
  }

  const loom_named_attr_t* permutation_param = loom_vector_find_named_attr(
      module, params->dynamic_names,
      loom_vector_transform_permutation_param_name());
  if (permutation_param) {
    loom_value_id_t permutation_value = LOOM_VALUE_ID_INVALID;
    if (!loom_encoding_define_dynamic_param_value(params, permutation_param,
                                                  &permutation_value)) {
      return iree_ok_status();
    }
    loom_type_t permutation_type =
        loom_module_value_type(module, permutation_value);
    if (loom_type_is_vector(permutation_type) &&
        loom_type_satisfies_constraint(
            permutation_type,
            LOOM_TYPE_CONSTRAINT_INDEX_OR_NON_I1_INTEGER_ELEMENT) &&
        !loom_vector_shapes_match(permutation_type, source_type)) {
      return loom_vector_emit_encoding_dynamic_type_error(
          module, emitter, define_op, encoding_name,
          loom_vector_transform_permutation_param_name(), permutation_value,
          IREE_SV("index or non-i1 integer vector with source shape"));
    }
    IREE_RETURN_IF_ERROR(loom_vector_transform_verify_static_permutation_values(
        module, emitter, op, source_type, permutation_value));
  }

  const loom_named_attr_t* matrix_param = loom_vector_find_named_attr(
      module, params->dynamic_names, loom_vector_transform_matrix_param_name());
  if (!matrix_param) return iree_ok_status();

  loom_value_id_t matrix_value = LOOM_VALUE_ID_INVALID;
  if (!loom_encoding_define_dynamic_param_value(params, matrix_param,
                                                &matrix_value)) {
    return iree_ok_status();
  }
  loom_type_t matrix_type = loom_module_value_type(module, matrix_value);
  if (!loom_type_is_vector(matrix_type) ||
      !loom_scalar_type_is_float(loom_type_element_type(matrix_type))) {
    return iree_ok_status();
  }

  loom_encoding_numeric_transform_read_t read =
      loom_encoding_numeric_transform_read_descriptor(
          module, loom_encoding_define_result(define_op));
  if (read.code != LOOM_ENCODING_NUMERIC_TRANSFORM_READ_OK) {
    return iree_ok_status();
  }
  if (read.descriptor.family !=
      LOOM_ENCODING_NUMERIC_TRANSFORM_FAMILY_JL_DENSE) {
    return iree_ok_status();
  }

  uint8_t matrix_rank = loom_type_rank(matrix_type);
  uint8_t source_last_axis = (uint8_t)(loom_type_rank(source_type) - 1);
  uint8_t result_last_axis = (uint8_t)(loom_type_rank(result_type) - 1);
  if (matrix_rank == 2 &&
      loom_vector_dim_equals(matrix_type, 0, result_type, result_last_axis) &&
      loom_vector_dim_equals(matrix_type, 1, source_type, source_last_axis)) {
    return iree_ok_status();
  }
  return loom_vector_emit_encoding_dynamic_type_error(
      module, emitter, define_op, encoding_name,
      loom_vector_transform_matrix_param_name(), matrix_value,
      IREE_SV("rank-2 floating-point matrix matching output x input extents"));
}

static iree_status_t loom_vector_transform_verify_family(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter, loom_type_t source_type,
    loom_type_t result_type) {
  loom_encoding_numeric_transform_read_t read =
      loom_encoding_numeric_transform_read_descriptor(
          module, loom_vector_transform_transform(op));
  if (read.code != LOOM_ENCODING_NUMERIC_TRANSFORM_READ_OK) {
    return iree_ok_status();
  }

  if (!loom_encoding_numeric_transform_family_is_hadamard_like(
          read.descriptor.family)) {
    return iree_ok_status();
  }
  if (!loom_vector_shapes_match(source_type, result_type)) {
    return loom_vector_emit_shape_mismatch(emitter, op, IREE_SV("result"),
                                           IREE_SV("source"));
  }

  uint8_t last_axis = (uint8_t)(loom_type_rank(source_type) - 1);
  if (loom_type_dim_is_dynamic_at(source_type, last_axis)) {
    return iree_ok_status();
  }
  int64_t last_axis_size = loom_type_dim_static_size_at(source_type, last_axis);
  if (loom_vector_i64_is_power_of_two(last_axis_size)) return iree_ok_status();
  return loom_vector_emit_operand_constraint(
      emitter, op, IREE_SV("source"), source_type,
      IREE_SV("power-of-two last axis extent for Hadamard transform"));
}

iree_status_t loom_vector_transform_verify(const loom_module_t* module,
                                           const loom_op_t* op,
                                           iree_diagnostic_emitter_t emitter) {
  loom_value_id_t source_value = loom_vector_transform_source(op);
  loom_value_id_t transform_value = loom_vector_transform_transform(op);
  loom_value_id_t result_value = loom_vector_transform_result(op);
  loom_type_t source_type = loom_module_value_type(module, source_value);
  loom_type_t transform_type = loom_module_value_type(module, transform_value);
  loom_type_t result_type = loom_module_value_type(module, result_value);
  if (!loom_type_is_vector(source_type) || !loom_type_is_vector(result_type) ||
      !loom_type_is_encoding(transform_type)) {
    return iree_ok_status();
  }

  if (loom_type_encoding_role(transform_type) !=
      LOOM_ENCODING_ROLE_NUMERIC_TRANSFORM) {
    return iree_ok_status();
  }

  uint8_t source_rank = loom_type_rank(source_type);
  uint8_t result_rank = loom_type_rank(result_type);
  if (source_rank == 0 || result_rank == 0) return iree_ok_status();
  if (source_rank != result_rank) {
    return loom_vector_emit_rank_mismatch(emitter, op, IREE_SV("result"),
                                          result_rank, IREE_SV("source"),
                                          source_rank);
  }

  uint8_t last_axis = (uint8_t)(source_rank - 1);
  for (uint8_t axis = 0; axis < last_axis; ++axis) {
    if (loom_vector_dim_equals(source_type, axis, result_type, axis)) continue;
    return loom_vector_emit_shape_mismatch(emitter, op, IREE_SV("result"),
                                           IREE_SV("source leading axes"));
  }

  const loom_value_t* transform = loom_module_value(module, transform_value);
  if (loom_value_is_block_arg(transform)) return iree_ok_status();
  const loom_op_t* define_op = loom_value_def_op(transform);
  if (!define_op || !loom_encoding_define_isa(define_op)) {
    return iree_ok_status();
  }

  loom_encoding_define_param_view_t params =
      loom_encoding_define_param_view(module, define_op);
  if (!params.spec || params.spec->name_id == LOOM_STRING_ID_INVALID ||
      params.spec->name_id >= module->strings.count) {
    return iree_ok_status();
  }
  iree_string_view_t encoding_name =
      module->strings.entries[params.spec->name_id];

  IREE_RETURN_IF_ERROR(loom_vector_transform_verify_extent_param(
      module, emitter, op, &params,
      loom_vector_transform_input_elems_param_name(), IREE_SV("source"),
      source_type, last_axis, /*field_is_result=*/false));
  IREE_RETURN_IF_ERROR(loom_vector_transform_verify_extent_param(
      module, emitter, op, &params,
      loom_vector_transform_output_elems_param_name(), IREE_SV("result"),
      result_type, last_axis, /*field_is_result=*/true));
  IREE_RETURN_IF_ERROR(loom_vector_transform_verify_dynamic_param_shapes(
      module, emitter, op, define_op, &params, encoding_name, source_type,
      result_type));

  return loom_vector_transform_verify_family(module, op, emitter, source_type,
                                             result_type);
}

iree_status_t loom_vector_geluf_verify(const loom_module_t* module,
                                       const loom_op_t* op,
                                       iree_diagnostic_emitter_t emitter) {
  loom_attribute_t scale_attr = loom_op_attrs(op)[1];
  bool has_scale = !loom_attr_is_absent(scale_attr);
  if (loom_vector_geluf_variant(op) == LOOM_VECTOR_GELUF_VARIANT_LOGISTIC) {
    if (has_scale) return iree_ok_status();
    return loom_vector_emit_indexed_attribute_kind_mismatch(
        emitter, op, IREE_SV("scale"), /*attr_index=*/1, LOOM_ATTR_ABSENT,
        LOOM_ATTR_F64);
  }
  if (!has_scale) return iree_ok_status();
  return loom_vector_emit_indexed_attribute_kind_mismatch(
      emitter, op, IREE_SV("scale"), /*attr_index=*/1, scale_attr.kind,
      LOOM_ATTR_ABSENT);
}

iree_status_t loom_vector_reduce_verify(const loom_module_t* module,
                                        const loom_op_t* op,
                                        iree_diagnostic_emitter_t emitter) {
  loom_type_t input_type =
      loom_module_value_type(module, loom_vector_reduce_input(op));
  if (!loom_type_is_vector(input_type)) return iree_ok_status();
  IREE_RETURN_IF_ERROR(loom_vector_verify_float_instance_flags(
      emitter, op, IREE_SV("input"), input_type));

  loom_combining_kind_t kind = loom_vector_reduce_kind(op);
  loom_scalar_type_t element_type = loom_type_element_type(input_type);
  if (loom_scalar_type_is_integer(element_type) &&
      loom_combining_kind_accepts_integer(kind)) {
    return iree_ok_status();
  }
  if (loom_scalar_type_is_float(element_type) &&
      loom_combining_kind_accepts_float(kind)) {
    return iree_ok_status();
  }
  if (!loom_combining_kind_is_valid(kind)) {
    return iree_ok_status();
  }
  iree_string_view_t expected_constraint =
      loom_combining_kind_accepts_integer(kind)
          ? IREE_SV("integer element type for reduce kind")
          : IREE_SV("floating-point element type for reduce kind");
  return loom_vector_emit_operand_constraint(emitter, op, IREE_SV("input"),
                                             input_type, expected_constraint);
}

static bool loom_vector_reduce_axes_contains(loom_attribute_t axes,
                                             uint8_t axis) {
  for (uint16_t i = 0; i < axes.count; ++i) {
    if (axes.i64_array[i] == axis) return true;
  }
  return false;
}

static iree_status_t loom_vector_reduce_axes_verify_shape(
    const loom_op_t* op, iree_diagnostic_emitter_t emitter,
    loom_type_t input_type, loom_type_t result_type, loom_attribute_t axes) {
  uint8_t input_rank = loom_type_rank(input_type);
  uint8_t expected_result_rank = (uint8_t)(input_rank - axes.count);
  if (expected_result_rank == 0) {
    if (loom_type_is_scalar(result_type)) return iree_ok_status();
    return loom_vector_emit_result_constraint(
        emitter, op, IREE_SV("result"), result_type,
        IREE_SV("scalar result when all axes are reduced"));
  }

  if (!loom_type_is_vector(result_type)) {
    return loom_vector_emit_result_constraint(
        emitter, op, IREE_SV("result"), result_type,
        IREE_SV("vector result when unreduced axes remain"));
  }
  uint8_t result_rank = loom_type_rank(result_type);
  if (result_rank != expected_result_rank) {
    return loom_vector_emit_rank_mismatch(
        emitter, op, IREE_SV("result"), result_rank,
        IREE_SV("input rank minus axes"), expected_result_rank);
  }

  uint8_t result_axis = 0;
  for (uint8_t input_axis = 0; input_axis < input_rank; ++input_axis) {
    if (loom_vector_reduce_axes_contains(axes, input_axis)) continue;
    if (!loom_vector_dim_equals(input_type, input_axis, result_type,
                                result_axis)) {
      return loom_vector_emit_shape_mismatch(emitter, op, IREE_SV("result"),
                                             IREE_SV("input without axes"));
    }
    ++result_axis;
  }
  return iree_ok_status();
}

iree_status_t loom_vector_reduce_axes_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter) {
  loom_type_t input_type =
      loom_module_value_type(module, loom_vector_reduce_axes_input(op));
  if (!loom_type_is_vector(input_type)) return iree_ok_status();
  IREE_RETURN_IF_ERROR(loom_vector_verify_float_instance_flags(
      emitter, op, IREE_SV("input"), input_type));

  loom_attribute_t axes = loom_vector_reduce_axes_axes(op);
  if (axes.kind != LOOM_ATTR_I64_ARRAY) {
    return loom_vector_emit_attribute_kind_mismatch(
        emitter, op, IREE_SV("axes"), axes.kind, LOOM_ATTR_I64_ARRAY);
  }
  if (axes.count == 0) {
    return loom_vector_emit_attribute_value_constraint(
        emitter, op, IREE_SV("axes"), 0, IREE_SV("non-empty axes list"));
  }

  uint8_t input_rank = loom_type_rank(input_type);
  if (axes.count > input_rank) {
    return loom_vector_emit_count_mismatch(emitter, op, IREE_SV("axes"),
                                           axes.count, IREE_SV("input rank"),
                                           input_rank);
  }
  int64_t previous_axis = -1;
  for (uint16_t i = 0; i < axes.count; ++i) {
    int64_t axis = axes.i64_array[i];
    if (axis < 0 || axis >= input_rank) {
      return loom_vector_emit_attribute_value_constraint(
          emitter, op, IREE_SV("axes"), axis,
          IREE_SV("axis in input rank bounds"));
    }
    if (axis <= previous_axis) {
      return loom_vector_emit_attribute_value_constraint(
          emitter, op, IREE_SV("axes"), axis,
          IREE_SV("strictly sorted unique axes"));
    }
    previous_axis = axis;
  }

  loom_type_t result_type =
      loom_module_value_type(module, loom_vector_reduce_axes_result(op));
  IREE_RETURN_IF_ERROR(loom_vector_reduce_axes_verify_shape(
      op, emitter, input_type, result_type, axes));

  loom_combining_kind_t kind = loom_vector_reduce_axes_kind(op);
  loom_scalar_type_t element_type = loom_type_element_type(input_type);
  if (loom_scalar_type_is_integer(element_type) &&
      loom_combining_kind_accepts_integer(kind)) {
    return iree_ok_status();
  }
  if (loom_scalar_type_is_float(element_type) &&
      loom_combining_kind_accepts_float(kind)) {
    return iree_ok_status();
  }
  if (!loom_combining_kind_is_valid(kind)) {
    return iree_ok_status();
  }
  iree_string_view_t expected_constraint =
      loom_combining_kind_accepts_integer(kind)
          ? IREE_SV("integer element type for reduce kind")
          : IREE_SV("floating-point element type for reduce kind");
  return loom_vector_emit_operand_constraint(emitter, op, IREE_SV("input"),
                                             input_type, expected_constraint);
}
