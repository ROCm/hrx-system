// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/source_memory_plan.h"

#include <stdint.h>

#include "iree/base/api.h"
#include "loom/codegen/low/memory_access.h"
#include "loom/ir/facts.h"
#include "loom/ir/module.h"
#include "loom/ops/buffer/ops.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/kernel/ops.h"
#include "loom/ops/vector/ops.h"
#include "loom/ops/view/ops.h"
#include "loom/util/math.h"

static bool loom_low_source_memory_static_view_vector_type(
    loom_type_t view_type, loom_type_t* out_vector_type,
    loom_low_source_memory_access_diagnostic_t* out_diagnostic) {
  *out_vector_type = loom_type_none();
  if (!loom_type_is_view(view_type) ||
      loom_type_rank(view_type) > LOOM_ENCODING_ADDRESS_LAYOUT_MAX_RANK) {
    out_diagnostic->rejection_bits |=
        LOOM_LOW_SOURCE_MEMORY_ACCESS_REJECTION_DESCRIBE_FAILED;
    return false;
  }

  const int32_t element_bit_count =
      loom_scalar_type_bitwidth(loom_type_element_type(view_type));
  if (element_bit_count <= 0 || (element_bit_count % 8) != 0) {
    out_diagnostic->rejection_bits |=
        LOOM_LOW_SOURCE_MEMORY_ACCESS_REJECTION_ELEMENT_WIDTH;
    return false;
  }

  int64_t lane_count = 1;
  const uint8_t rank = loom_type_rank(view_type);
  for (uint8_t i = 0; i < rank; ++i) {
    if (loom_type_dim_is_dynamic_at(view_type, i)) {
      out_diagnostic->rejection_bits |=
          LOOM_LOW_SOURCE_MEMORY_ACCESS_REJECTION_VECTOR_LANE_COUNT;
      return false;
    }
    const int64_t dim_size = loom_type_dim_static_size_at(view_type, i);
    if (dim_size <= 0 ||
        !iree_checked_mul_i64(lane_count, dim_size, &lane_count) ||
        lane_count > UINT32_MAX) {
      out_diagnostic->rejection_bits |=
          LOOM_LOW_SOURCE_MEMORY_ACCESS_REJECTION_VECTOR_LANE_COUNT;
      return false;
    }
  }

  *out_vector_type =
      loom_type_shaped_1d(LOOM_TYPE_VECTOR, loom_type_element_type(view_type),
                          loom_dim_pack_static(lane_count), /*encoding_id=*/0);
  return true;
}

static bool loom_low_source_memory_access_exact_i64(loom_value_facts_t facts,
                                                    int64_t* out_value) {
  *out_value = 0;
  if (!loom_value_facts_is_exact(facts) || loom_value_facts_is_float(facts)) {
    return false;
  }
  *out_value = facts.range_lo;
  return true;
}

static uint32_t loom_low_source_memory_clamp_alignment(uint64_t alignment) {
  if (alignment == 0) return 1;
  return alignment > UINT32_MAX ? UINT32_MAX : (uint32_t)alignment;
}

static uint32_t loom_low_source_memory_combine_alignment(uint32_t alignment,
                                                         int64_t byte_offset) {
  if (byte_offset == 0) return alignment == 0 ? 1 : alignment;
  return (uint32_t)loom_gcd_i64((int64_t)alignment, byte_offset);
}

static void loom_low_source_memory_access_finalize_alignment(
    loom_low_source_memory_access_plan_t* plan) {
  uint32_t alignment =
      plan->root_minimum_alignment == 0 ? 1 : plan->root_minimum_alignment;
  alignment = loom_low_source_memory_combine_alignment(
      alignment, plan->static_byte_offset);
  for (uint8_t i = 0; i < plan->dynamic_term_count; ++i) {
    const int64_t divisor =
        plan->dynamic_terms[i].byte_facts.known_divisor > 0
            ? plan->dynamic_terms[i].byte_facts.known_divisor
            : 1;
    alignment = loom_low_source_memory_combine_alignment(alignment, divisor);
  }
  plan->minimum_alignment = alignment == 0 ? 1 : alignment;
}

static loom_value_id_t loom_low_source_memory_access_identity_source_value(
    const loom_module_t* module, loom_value_id_t value_id) {
  while (value_id < module->values.count) {
    const loom_value_t* value = loom_module_value(module, value_id);
    if (loom_value_is_block_arg(value)) {
      return value_id;
    }
    const loom_op_t* defining_op = loom_value_def_op(value);
    if (!defining_op || !loom_index_assume_isa(defining_op)) {
      return value_id;
    }
    const uint16_t result_index = loom_value_def_index(value);
    if (result_index >= defining_op->operand_count) {
      return value_id;
    }
    const loom_value_id_t source_value_id =
        loom_op_const_operands(defining_op)[result_index];
    if (source_value_id == value_id) {
      return value_id;
    }
    value_id = source_value_id;
  }
  return value_id;
}

static bool loom_low_source_memory_access_value_as_workitem_id(
    const loom_module_t* module, loom_value_id_t value_id,
    loom_kernel_dimension_t* out_dimension) {
  *out_dimension = LOOM_KERNEL_DIMENSION_COUNT_;
  value_id =
      loom_low_source_memory_access_identity_source_value(module, value_id);
  if (value_id >= module->values.count) {
    return false;
  }
  const loom_value_t* value = loom_module_value(module, value_id);
  if (loom_value_is_block_arg(value)) {
    return false;
  }
  const loom_op_t* defining_op = loom_value_def_op(value);
  if (!defining_op || !loom_kernel_workitem_id_isa(defining_op)) {
    return false;
  }
  const loom_kernel_dimension_t dimension =
      loom_kernel_workitem_id_dimension(defining_op);
  if (dimension >= LOOM_KERNEL_DIMENSION_COUNT_) {
    return false;
  }
  *out_dimension = dimension;
  return true;
}

static bool loom_low_source_memory_access_value_as_workgroup_id(
    const loom_module_t* module, loom_value_id_t value_id,
    loom_kernel_dimension_t* out_dimension) {
  *out_dimension = LOOM_KERNEL_DIMENSION_COUNT_;
  value_id =
      loom_low_source_memory_access_identity_source_value(module, value_id);
  if (value_id >= module->values.count) {
    return false;
  }
  const loom_value_t* value = loom_module_value(module, value_id);
  if (loom_value_is_block_arg(value)) {
    return false;
  }
  const loom_op_t* defining_op = loom_value_def_op(value);
  if (!defining_op || !loom_kernel_workgroup_id_isa(defining_op)) {
    return false;
  }
  const loom_kernel_dimension_t dimension =
      loom_kernel_workgroup_id_dimension(defining_op);
  if (dimension >= LOOM_KERNEL_DIMENSION_COUNT_) {
    return false;
  }
  *out_dimension = dimension;
  return true;
}

static void loom_low_source_memory_access_dynamic_index_source(
    const loom_module_t* module, loom_value_id_t index,
    loom_low_source_memory_dynamic_index_source_t* out_source,
    loom_kernel_dimension_t* out_dimension) {
  *out_source = LOOM_LOW_SOURCE_MEMORY_DYNAMIC_INDEX_SOURCE_VALUE;
  *out_dimension = LOOM_KERNEL_DIMENSION_COUNT_;
  if (loom_low_source_memory_access_value_as_workitem_id(module, index,
                                                         out_dimension)) {
    *out_source = LOOM_LOW_SOURCE_MEMORY_DYNAMIC_INDEX_SOURCE_WORKITEM_ID;
  } else if (loom_low_source_memory_access_value_as_workgroup_id(
                 module, index, out_dimension)) {
    *out_source = LOOM_LOW_SOURCE_MEMORY_DYNAMIC_INDEX_SOURCE_WORKGROUP_ID;
  }
}

static bool loom_low_source_memory_access_collect_dynamic_axes(
    loom_attribute_t static_indices, uint8_t* out_dynamic_axes,
    uint8_t* out_dynamic_axis_count) {
  *out_dynamic_axis_count = 0;
  if (static_indices.kind != LOOM_ATTR_I64_ARRAY) {
    return false;
  }
  for (uint16_t i = 0; i < static_indices.count; ++i) {
    if (static_indices.i64_array[i] != INT64_MIN) {
      continue;
    }
    if (*out_dynamic_axis_count >=
            LOOM_LOW_SOURCE_MEMORY_DYNAMIC_TERM_CAPACITY ||
        i > UINT8_MAX) {
      return false;
    }
    out_dynamic_axes[*out_dynamic_axis_count] = (uint8_t)i;
    *out_dynamic_axis_count += 1;
  }
  return true;
}

static bool loom_low_source_memory_access_axis_is_dynamic(
    const uint8_t* dynamic_axes, uint8_t dynamic_axis_count, uint16_t axis) {
  for (uint8_t i = 0; i < dynamic_axis_count; ++i) {
    if (dynamic_axes[i] == axis) {
      return true;
    }
  }
  return false;
}

static bool loom_low_source_memory_access_static_byte_offset(
    const loom_vector_memory_access_t* vector_access,
    loom_attribute_t static_indices, const uint8_t* dynamic_axes,
    uint8_t dynamic_axis_count, int64_t* out_static_byte_offset) {
  *out_static_byte_offset = 0;
  if (static_indices.kind != LOOM_ATTR_I64_ARRAY ||
      static_indices.count > LOOM_ENCODING_ADDRESS_LAYOUT_MAX_RANK) {
    return false;
  }

  int64_t static_origin[LOOM_ENCODING_ADDRESS_LAYOUT_MAX_RANK] = {0};
  for (uint16_t i = 0; i < static_indices.count; ++i) {
    if (loom_low_source_memory_access_axis_is_dynamic(dynamic_axes,
                                                      dynamic_axis_count, i)) {
      if (static_indices.i64_array[i] != INT64_MIN) {
        return false;
      }
      continue;
    }
    if (static_indices.i64_array[i] == INT64_MIN) {
      return false;
    }
    static_origin[i] = static_indices.i64_array[i];
  }
  loom_attribute_t static_origin_attr =
      loom_attr_i64_array(static_origin, static_indices.count);
  int64_t lane_indices[] = {0};
  return loom_vector_memory_access_static_lane_byte_offset(
      vector_access, static_origin_attr, lane_indices,
      IREE_ARRAYSIZE(lane_indices), out_static_byte_offset);
}

static void loom_low_source_memory_access_fold_exact_dynamic_indices(
    const loom_value_fact_table_t* fact_table,
    loom_value_slice_t dynamic_indices, loom_attribute_t static_indices,
    int64_t* folded_static_indices, uint8_t* dynamic_axes,
    loom_value_id_t* dynamic_index_values, uint8_t* dynamic_axis_count) {
  for (uint16_t i = 0; i < static_indices.count; ++i) {
    folded_static_indices[i] = static_indices.i64_array[i];
  }

  uint8_t folded_dynamic_axis_count = 0;
  const uint8_t original_dynamic_axis_count = *dynamic_axis_count;
  for (uint8_t i = 0; i < original_dynamic_axis_count; ++i) {
    const uint8_t dynamic_axis = dynamic_axes[i];
    const loom_value_id_t dynamic_index = dynamic_indices.values[i];
    int64_t exact_index = 0;
    if (dynamic_axis < static_indices.count &&
        loom_low_source_memory_access_exact_i64(
            loom_value_fact_table_lookup(fact_table, dynamic_index),
            &exact_index) &&
        exact_index != INT64_MIN) {
      folded_static_indices[dynamic_axis] = exact_index;
      continue;
    }
    dynamic_axes[folded_dynamic_axis_count] = dynamic_axis;
    dynamic_index_values[folded_dynamic_axis_count] = dynamic_index;
    ++folded_dynamic_axis_count;
  }
  *dynamic_axis_count = folded_dynamic_axis_count;
}

static bool loom_low_source_memory_access_vector_lane_count(
    loom_type_t vector_type, uint32_t* out_lane_count) {
  *out_lane_count = 0;
  if (!loom_type_is_vector(vector_type) || loom_type_rank(vector_type) != 1 ||
      loom_type_dim_is_dynamic_at(vector_type, 0)) {
    return false;
  }
  const int64_t lane_count = loom_type_dim_static_size_at(vector_type, 0);
  if (lane_count <= 0 || lane_count > UINT32_MAX) {
    return false;
  }
  *out_lane_count = (uint32_t)lane_count;
  return true;
}

static bool loom_low_source_memory_access_power_of_two_shift(
    int64_t value, uint32_t* out_shift) {
  *out_shift = LOOM_LOW_SOURCE_MEMORY_ACCESS_BYTE_SHIFT_NONE;
  if (value <= 0 || value > UINT32_MAX) {
    return false;
  }
  uint32_t remaining_value = (uint32_t)value;
  if ((remaining_value & (remaining_value - 1)) != 0) {
    return false;
  }
  uint32_t shift = 0;
  while (remaining_value > 1) {
    remaining_value >>= 1;
    ++shift;
  }
  *out_shift = shift;
  return true;
}

static bool loom_low_source_memory_access_dynamic_dense_axis_byte_stride(
    const loom_vector_memory_access_t* vector_access, uint8_t view_axis,
    int64_t* out_byte_stride, loom_value_id_t* out_stride_values,
    uint8_t* out_stride_value_count) {
  *out_byte_stride = 0;
  *out_stride_value_count = 0;
  if (vector_access->layout_kind != LOOM_VECTOR_MEMORY_LAYOUT_DENSE ||
      view_axis >= vector_access->view_rank ||
      vector_access->static_element_byte_count <= 0) {
    return false;
  }

  int64_t byte_stride = vector_access->static_element_byte_count;
  for (uint8_t axis = (uint8_t)(view_axis + 1); axis < vector_access->view_rank;
       ++axis) {
    if (loom_type_dim_is_dynamic_at(vector_access->view_type, axis)) {
      if (*out_stride_value_count >=
          LOOM_LOW_SOURCE_MEMORY_DYNAMIC_TERM_CAPACITY) {
        return false;
      }
      out_stride_values[*out_stride_value_count] =
          loom_type_dim_value_id_at(vector_access->view_type, axis);
      *out_stride_value_count += 1;
      continue;
    }
    const int64_t dimension_size =
        loom_type_dim_static_size_at(vector_access->view_type, axis);
    if (dimension_size < 0 ||
        !iree_checked_mul_i64(byte_stride, dimension_size, &byte_stride)) {
      return false;
    }
  }
  *out_byte_stride = byte_stride;
  return true;
}

static bool loom_low_source_memory_access_dynamic_axis_byte_stride(
    const loom_vector_memory_access_t* vector_access, uint8_t dynamic_axis,
    int64_t* out_byte_stride, loom_value_id_t* out_stride_values,
    uint8_t* out_stride_value_count) {
  *out_byte_stride = 0;
  *out_stride_value_count = 0;
  switch (vector_access->layout_kind) {
    case LOOM_VECTOR_MEMORY_LAYOUT_DENSE:
      return loom_low_source_memory_access_dynamic_dense_axis_byte_stride(
          vector_access, dynamic_axis, out_byte_stride, out_stride_values,
          out_stride_value_count);
    case LOOM_VECTOR_MEMORY_LAYOUT_STRIDED: {
      int64_t axis_stride = 0;
      if (!loom_vector_memory_access_static_axis_stride(
              vector_access, dynamic_axis, &axis_stride)) {
        return false;
      }
      return iree_checked_mul_i64(axis_stride,
                                  vector_access->static_element_byte_count,
                                  out_byte_stride);
    }
    case LOOM_VECTOR_MEMORY_LAYOUT_UNKNOWN:
      return false;
  }
  return false;
}

static bool loom_low_source_memory_access_append_dynamic_term(
    loom_low_source_memory_access_plan_t* plan,
    const loom_low_source_memory_dynamic_term_t* term,
    loom_low_source_memory_access_diagnostic_t* diagnostic) {
  if (plan->dynamic_term_count >=
      LOOM_LOW_SOURCE_MEMORY_DYNAMIC_TERM_CAPACITY) {
    diagnostic->rejection_bits |=
        LOOM_LOW_SOURCE_MEMORY_ACCESS_REJECTION_DYNAMIC_INDEX_COUNT;
    return false;
  }
  plan->dynamic_terms[plan->dynamic_term_count] = *term;
  ++plan->dynamic_term_count;
  return true;
}

static loom_value_facts_t loom_low_source_memory_access_axis_dimension_facts(
    const loom_value_fact_table_t* fact_table, loom_type_t view_type,
    uint8_t axis) {
  if (!loom_type_dim_is_dynamic_at(view_type, axis)) {
    return loom_value_facts_exact_i64(
        loom_type_dim_static_size_at(view_type, axis));
  }
  const loom_value_id_t dimension_value_id =
      loom_type_dim_value_id_at(view_type, axis);
  return loom_value_facts_non_negative_extent(
      loom_value_fact_table_lookup(fact_table, dimension_value_id));
}

static bool loom_low_source_memory_access_origin_domain_facts(
    const loom_value_fact_table_t* fact_table,
    const loom_vector_memory_access_t* vector_access, uint8_t dynamic_axis,
    loom_value_facts_t* out_facts) {
  *out_facts = loom_value_facts_unknown();

  int64_t extent = 0;
  if (!loom_vector_memory_access_static_axis_extent(vector_access, dynamic_axis,
                                                    &extent) ||
      extent <= 0) {
    return false;
  }

  const loom_value_facts_t dimension_facts =
      loom_low_source_memory_access_axis_dimension_facts(
          fact_table, vector_access->view_type, dynamic_axis);
  if (loom_value_facts_is_float(dimension_facts) ||
      dimension_facts.range_hi == INT64_MAX || dimension_facts.range_hi < 0 ||
      dimension_facts.range_hi < extent) {
    return false;
  }
  *out_facts = loom_value_facts_make(0, dimension_facts.range_hi - extent, 1);
  return true;
}

static loom_value_facts_t loom_low_source_memory_access_intersect_index_facts(
    loom_value_facts_t index_facts, loom_value_facts_t domain_facts) {
  if (loom_value_facts_is_float(index_facts) ||
      loom_value_facts_is_float(domain_facts)) {
    return index_facts;
  }
  const int64_t lower_bound =
      loom_max_i64(index_facts.range_lo, domain_facts.range_lo);
  const int64_t upper_bound =
      loom_min_i64(index_facts.range_hi, domain_facts.range_hi);
  if (lower_bound > upper_bound) {
    return loom_value_facts_unknown();
  }
  return loom_value_facts_make(lower_bound, upper_bound,
                               index_facts.known_divisor);
}

static int64_t loom_low_source_memory_access_floor_div_i64(int64_t numerator,
                                                           int64_t divisor) {
  IREE_ASSERT_GT(divisor, 0);
  int64_t quotient = numerator / divisor;
  const int64_t remainder = numerator % divisor;
  if (remainder != 0 && numerator < 0) {
    --quotient;
  }
  return quotient;
}

static int64_t loom_low_source_memory_access_ceil_div_i64(int64_t numerator,
                                                          int64_t divisor) {
  IREE_ASSERT_GT(divisor, 0);
  int64_t quotient = numerator / divisor;
  const int64_t remainder = numerator % divisor;
  if (remainder != 0 && numerator > 0) {
    ++quotient;
  }
  return quotient;
}

static bool loom_low_source_memory_access_scale_domain_down(
    int64_t multiplier, int64_t offset,
    loom_value_facts_t* inout_domain_facts) {
  if (multiplier <= 0 || loom_value_facts_is_float(*inout_domain_facts) ||
      inout_domain_facts->range_lo < 0) {
    return false;
  }
  if (multiplier == 1 && offset == 0) {
    return true;
  }
  int64_t scaled_range_lo = 0;
  int64_t scaled_range_hi = 0;
  if (!loom_checked_sub_i64(inout_domain_facts->range_lo, offset,
                            &scaled_range_lo) ||
      !loom_checked_sub_i64(inout_domain_facts->range_hi, offset,
                            &scaled_range_hi)) {
    return false;
  }
  inout_domain_facts->range_lo =
      loom_low_source_memory_access_ceil_div_i64(scaled_range_lo, multiplier);
  inout_domain_facts->range_hi =
      loom_low_source_memory_access_floor_div_i64(scaled_range_hi, multiplier);
  inout_domain_facts->known_divisor = 1;
  return true;
}

static void loom_low_source_memory_dynamic_term_compute_index_byte_facts(
    const loom_value_fact_table_t* fact_table, loom_value_facts_t index_facts,
    int64_t byte_stride, const loom_value_id_t* stride_values,
    uint8_t stride_value_count, loom_value_facts_t* out_facts) {
  const loom_value_facts_t byte_stride_facts =
      loom_value_facts_exact_i64(byte_stride);
  loom_value_facts_muli(&index_facts, &byte_stride_facts, out_facts);
  for (uint8_t i = 0; i < stride_value_count; ++i) {
    loom_value_facts_t stride_facts = loom_value_facts_non_negative_extent(
        loom_value_fact_table_lookup(fact_table, stride_values[i]));
    loom_value_facts_muli(out_facts, &stride_facts, out_facts);
  }
}

static void loom_low_source_memory_dynamic_term_compute_scaled_byte_facts(
    const loom_value_fact_table_t* fact_table, loom_value_id_t index,
    int64_t byte_stride, const loom_value_id_t* stride_values,
    uint8_t stride_value_count, loom_value_facts_t* out_facts) {
  loom_low_source_memory_dynamic_term_compute_index_byte_facts(
      fact_table, loom_value_fact_table_lookup(fact_table, index), byte_stride,
      stride_values, stride_value_count, out_facts);
}

static void loom_low_source_memory_dynamic_term_compute_byte_facts(
    const loom_value_fact_table_t* fact_table,
    const loom_vector_memory_access_t* vector_access, loom_value_id_t index,
    uint8_t dynamic_axis, int64_t index_multiplier, int64_t index_offset,
    int64_t byte_stride, const loom_value_id_t* stride_values,
    uint8_t stride_value_count, loom_value_facts_t* out_facts) {
  loom_value_facts_t index_facts =
      loom_value_fact_table_lookup(fact_table, index);
  loom_value_facts_t domain_facts = loom_value_facts_unknown();
  if (loom_low_source_memory_access_origin_domain_facts(
          fact_table, vector_access, dynamic_axis, &domain_facts)) {
    (void)loom_low_source_memory_access_scale_domain_down(
        index_multiplier, index_offset, &domain_facts);
    index_facts = loom_low_source_memory_access_intersect_index_facts(
        index_facts, domain_facts);
  }
  loom_low_source_memory_dynamic_term_compute_index_byte_facts(
      fact_table, index_facts, byte_stride, stride_values, stride_value_count,
      out_facts);
}

bool loom_low_source_memory_dynamic_offset_fits_unsigned_bit_count(
    const loom_low_source_memory_access_plan_t* plan,
    int64_t static_byte_offset, uint8_t bit_count) {
  loom_value_facts_t offset_facts =
      loom_value_facts_exact_i64(static_byte_offset);
  for (uint8_t i = 0; i < plan->dynamic_term_count; ++i) {
    loom_value_facts_addi(&offset_facts, &plan->dynamic_terms[i].byte_facts,
                          &offset_facts);
  }
  return loom_value_facts_fit_unsigned_bit_count(offset_facts, bit_count);
}

static bool loom_low_source_memory_access_exact_positive_i64(
    const loom_value_fact_table_t* fact_table, loom_value_id_t value_id,
    int64_t* out_value) {
  *out_value = 0;
  int64_t value = 0;
  if (!loom_low_source_memory_access_exact_i64(
          loom_value_fact_table_lookup(fact_table, value_id), &value) ||
      value <= 0) {
    return false;
  }
  *out_value = value;
  return true;
}

typedef struct loom_low_source_memory_scaled_index_t {
  // Base SSA value used as the dynamic address term.
  loom_value_id_t index;
  // Positive multiplier applied to |index| in the original source expression.
  int64_t multiplier;
  // Constant offset added after the scaled index in the original expression.
  int64_t offset;
} loom_low_source_memory_scaled_index_t;

typedef struct loom_low_source_memory_affine_index_term_t {
  // Base SSA value used as one dynamic address term.
  loom_value_id_t index;
  // Positive multiplier applied to |index| in the affine index expression.
  int64_t multiplier;
} loom_low_source_memory_affine_index_term_t;

static bool loom_low_source_memory_access_can_extract_static_index_offset(
    int64_t index_offset) {
  // Static byte offsets are modeled as non-negative target-friendly addends.
  // Keep negative terms in SSA so the derived index carries its range facts.
  return index_offset >= 0;
}

static bool loom_low_source_memory_access_append_affine_index_term(
    loom_value_id_t index, int64_t multiplier,
    loom_low_source_memory_affine_index_term_t* terms, uint8_t* inout_count) {
  if (*inout_count >= LOOM_LOW_SOURCE_MEMORY_DYNAMIC_TERM_CAPACITY ||
      multiplier <= 0) {
    return false;
  }
  terms[*inout_count] = (loom_low_source_memory_affine_index_term_t){
      .index = index,
      .multiplier = multiplier,
  };
  *inout_count += 1;
  return true;
}

static bool loom_low_source_memory_access_value_has_affine_index_op(
    const loom_module_t* module, loom_value_id_t value_id) {
  if (value_id >= module->values.count) {
    return false;
  }
  const loom_value_t* value = loom_module_value(module, value_id);
  if (loom_value_is_block_arg(value)) {
    return false;
  }
  const loom_op_t* defining_op = loom_value_def_op(value);
  if (!defining_op) {
    return false;
  }
  return loom_index_add_isa(defining_op) || loom_index_mul_isa(defining_op) ||
         loom_index_scale_isa(defining_op) ||
         loom_index_madd_isa(defining_op) || loom_index_shli_isa(defining_op);
}

static bool loom_low_source_memory_access_affine_index_terms_from_value(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    loom_value_id_t value_id, int64_t multiplier, uint32_t recursion_depth,
    loom_low_source_memory_affine_index_term_t* terms, uint8_t* inout_count,
    int64_t* inout_offset) {
  if (recursion_depth > 8 || value_id >= module->values.count ||
      multiplier <= 0) {
    return false;
  }

  int64_t constant = 0;
  if (loom_low_source_memory_access_exact_i64(
          loom_value_fact_table_lookup(fact_table, value_id), &constant)) {
    int64_t offset_delta = 0;
    int64_t new_offset = 0;
    if (!loom_checked_mul_i64(constant, multiplier, &offset_delta) ||
        !loom_checked_add_i64(*inout_offset, offset_delta, &new_offset)) {
      return false;
    }
    *inout_offset = new_offset;
    return true;
  }

  const loom_value_t* value = loom_module_value(module, value_id);
  if (loom_value_is_block_arg(value)) {
    return loom_low_source_memory_access_append_affine_index_term(
        value_id, multiplier, terms, inout_count);
  }
  const loom_op_t* defining_op = loom_value_def_op(value);
  if (!defining_op) {
    return loom_low_source_memory_access_append_affine_index_term(
        value_id, multiplier, terms, inout_count);
  }

  if (loom_index_assume_isa(defining_op)) {
    const uint16_t result_index = loom_value_def_index(value);
    if (result_index >= defining_op->operand_count) {
      return loom_low_source_memory_access_append_affine_index_term(
          value_id, multiplier, terms, inout_count);
    }
    const loom_value_id_t source_value_id =
        loom_op_const_operands(defining_op)[result_index];
    if (source_value_id != value_id &&
        loom_low_source_memory_access_value_has_affine_index_op(
            module, source_value_id)) {
      return loom_low_source_memory_access_affine_index_terms_from_value(
          module, fact_table, source_value_id, multiplier, recursion_depth + 1,
          terms, inout_count, inout_offset);
    }
    return loom_low_source_memory_access_append_affine_index_term(
        value_id, multiplier, terms, inout_count);
  }

  if (loom_index_cast_isa(defining_op)) {
    const loom_type_t input_type =
        loom_module_value_type(module, loom_index_cast_input(defining_op));
    const loom_type_t result_type = loom_module_value_type(module, value_id);
    if (loom_type_is_scalar(input_type) && loom_type_is_scalar(result_type) &&
        loom_type_element_type(input_type) == LOOM_SCALAR_TYPE_INDEX &&
        loom_type_element_type(result_type) == LOOM_SCALAR_TYPE_OFFSET) {
      return loom_low_source_memory_access_affine_index_terms_from_value(
          module, fact_table, loom_index_cast_input(defining_op), multiplier,
          recursion_depth + 1, terms, inout_count, inout_offset);
    }
  }

  if (loom_index_add_isa(defining_op)) {
    return loom_low_source_memory_access_affine_index_terms_from_value(
               module, fact_table, loom_index_add_lhs(defining_op), multiplier,
               recursion_depth + 1, terms, inout_count, inout_offset) &&
           loom_low_source_memory_access_affine_index_terms_from_value(
               module, fact_table, loom_index_add_rhs(defining_op), multiplier,
               recursion_depth + 1, terms, inout_count, inout_offset);
  }

  if (loom_index_mul_isa(defining_op)) {
    loom_value_id_t lhs = loom_index_mul_lhs(defining_op);
    loom_value_id_t rhs = loom_index_mul_rhs(defining_op);
    int64_t static_multiplier = 0;
    loom_value_id_t scaled_value = LOOM_VALUE_ID_INVALID;
    if (loom_low_source_memory_access_exact_positive_i64(fact_table, lhs,
                                                         &static_multiplier)) {
      scaled_value = rhs;
    } else if (loom_low_source_memory_access_exact_positive_i64(
                   fact_table, rhs, &static_multiplier)) {
      scaled_value = lhs;
    } else {
      return loom_low_source_memory_access_append_affine_index_term(
          value_id, multiplier, terms, inout_count);
    }
    int64_t combined_multiplier = 0;
    if (!loom_checked_mul_i64(multiplier, static_multiplier,
                              &combined_multiplier)) {
      return false;
    }
    return loom_low_source_memory_access_affine_index_terms_from_value(
        module, fact_table, scaled_value, combined_multiplier,
        recursion_depth + 1, terms, inout_count, inout_offset);
  }

  if (loom_index_scale_isa(defining_op)) {
    int64_t byte_stride = 0;
    if (!loom_low_source_memory_access_exact_positive_i64(
            fact_table, loom_index_scale_stride(defining_op), &byte_stride)) {
      return loom_low_source_memory_access_append_affine_index_term(
          value_id, multiplier, terms, inout_count);
    }
    int64_t combined_multiplier = 0;
    if (!loom_checked_mul_i64(multiplier, byte_stride, &combined_multiplier)) {
      return false;
    }
    return loom_low_source_memory_access_affine_index_terms_from_value(
        module, fact_table, loom_index_scale_index(defining_op),
        combined_multiplier, recursion_depth + 1, terms, inout_count,
        inout_offset);
  }

  if (loom_index_madd_isa(defining_op)) {
    loom_value_id_t a = loom_index_madd_a(defining_op);
    loom_value_id_t b = loom_index_madd_b(defining_op);
    loom_value_id_t c = loom_index_madd_c(defining_op);
    int64_t static_multiplier = 0;
    loom_value_id_t scaled_value = LOOM_VALUE_ID_INVALID;
    if (loom_low_source_memory_access_exact_positive_i64(fact_table, a,
                                                         &static_multiplier)) {
      scaled_value = b;
    } else if (loom_low_source_memory_access_exact_positive_i64(
                   fact_table, b, &static_multiplier)) {
      scaled_value = a;
    } else {
      return loom_low_source_memory_access_append_affine_index_term(
          value_id, multiplier, terms, inout_count);
    }
    int64_t combined_multiplier = 0;
    if (!loom_checked_mul_i64(multiplier, static_multiplier,
                              &combined_multiplier)) {
      return false;
    }
    return loom_low_source_memory_access_affine_index_terms_from_value(
               module, fact_table, scaled_value, combined_multiplier,
               recursion_depth + 1, terms, inout_count, inout_offset) &&
           loom_low_source_memory_access_affine_index_terms_from_value(
               module, fact_table, c, multiplier, recursion_depth + 1, terms,
               inout_count, inout_offset);
  }

  if (loom_index_shli_isa(defining_op)) {
    int64_t shift_amount = 0;
    if (!loom_low_source_memory_access_exact_i64(
            loom_value_fact_table_lookup(fact_table,
                                         loom_index_shli_rhs(defining_op)),
            &shift_amount) ||
        shift_amount < 0 || shift_amount > 62) {
      return loom_low_source_memory_access_append_affine_index_term(
          value_id, multiplier, terms, inout_count);
    }
    int64_t combined_multiplier = 0;
    if (!loom_checked_mul_i64(multiplier, ((int64_t)1) << shift_amount,
                              &combined_multiplier)) {
      return false;
    }
    return loom_low_source_memory_access_affine_index_terms_from_value(
        module, fact_table, loom_index_shli_lhs(defining_op),
        combined_multiplier, recursion_depth + 1, terms, inout_count,
        inout_offset);
  }

  return loom_low_source_memory_access_append_affine_index_term(
      value_id, multiplier, terms, inout_count);
}

static bool
loom_low_source_memory_access_affine_terms_have_mixed_coordinate_sources(
    const loom_module_t* module,
    const loom_low_source_memory_affine_index_term_t* terms,
    uint8_t term_count) {
  bool has_workgroup = false;
  bool has_workitem = false;
  for (uint8_t i = 0; i < term_count; ++i) {
    loom_low_source_memory_dynamic_index_source_t source =
        LOOM_LOW_SOURCE_MEMORY_DYNAMIC_INDEX_SOURCE_NONE;
    loom_kernel_dimension_t dimension = LOOM_KERNEL_DIMENSION_COUNT_;
    loom_low_source_memory_access_dynamic_index_source(module, terms[i].index,
                                                       &source, &dimension);
    has_workgroup |=
        source == LOOM_LOW_SOURCE_MEMORY_DYNAMIC_INDEX_SOURCE_WORKGROUP_ID;
    has_workitem |=
        source == LOOM_LOW_SOURCE_MEMORY_DYNAMIC_INDEX_SOURCE_WORKITEM_ID;
  }
  return has_workgroup && has_workitem;
}

static bool loom_low_source_memory_access_scaled_index_from_value(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    loom_value_id_t value_id, uint32_t recursion_depth,
    loom_low_source_memory_scaled_index_t* out_scaled_index) {
  *out_scaled_index = (loom_low_source_memory_scaled_index_t){
      .index = value_id,
      .multiplier = 1,
      .offset = 0,
  };
  if (recursion_depth > 8 || value_id >= module->values.count) {
    return false;
  }
  const loom_value_t* value = loom_module_value(module, value_id);
  if (loom_value_is_block_arg(value)) {
    return true;
  }
  const loom_op_t* defining_op = loom_value_def_op(value);
  if (!defining_op) {
    return true;
  }

  if (loom_index_cast_isa(defining_op)) {
    const loom_type_t input_type =
        loom_module_value_type(module, loom_index_cast_input(defining_op));
    const loom_type_t result_type = loom_module_value_type(module, value_id);
    if (loom_type_is_scalar(input_type) && loom_type_is_scalar(result_type) &&
        loom_type_element_type(input_type) == LOOM_SCALAR_TYPE_INDEX &&
        loom_type_element_type(result_type) == LOOM_SCALAR_TYPE_OFFSET) {
      return loom_low_source_memory_access_scaled_index_from_value(
          module, fact_table, loom_index_cast_input(defining_op),
          recursion_depth + 1, out_scaled_index);
    }
  }

  if (loom_index_mul_isa(defining_op)) {
    loom_value_id_t lhs = loom_index_mul_lhs(defining_op);
    loom_value_id_t rhs = loom_index_mul_rhs(defining_op);
    int64_t multiplier = 0;
    loom_value_id_t scaled_value = LOOM_VALUE_ID_INVALID;
    if (loom_low_source_memory_access_exact_positive_i64(fact_table, lhs,
                                                         &multiplier)) {
      scaled_value = rhs;
    } else if (loom_low_source_memory_access_exact_positive_i64(fact_table, rhs,
                                                                &multiplier)) {
      scaled_value = lhs;
    } else {
      return true;
    }

    loom_low_source_memory_scaled_index_t inner_scaled_index = {0};
    if (!loom_low_source_memory_access_scaled_index_from_value(
            module, fact_table, scaled_value, recursion_depth + 1,
            &inner_scaled_index)) {
      return false;
    }
    int64_t scaled_multiplier = 0;
    int64_t scaled_offset = 0;
    if (!loom_checked_mul_i64(inner_scaled_index.multiplier, multiplier,
                              &scaled_multiplier) ||
        !loom_checked_mul_i64(inner_scaled_index.offset, multiplier,
                              &scaled_offset)) {
      return false;
    }
    *out_scaled_index = (loom_low_source_memory_scaled_index_t){
        .index = inner_scaled_index.index,
        .multiplier = scaled_multiplier,
        .offset = scaled_offset,
    };
    return true;
  }

  if (loom_index_scale_isa(defining_op)) {
    loom_value_id_t index = loom_index_scale_index(defining_op);
    loom_value_id_t stride = loom_index_scale_stride(defining_op);
    int64_t multiplier = 0;
    if (!loom_low_source_memory_access_exact_positive_i64(fact_table, stride,
                                                          &multiplier)) {
      return true;
    }

    loom_low_source_memory_scaled_index_t inner_scaled_index = {0};
    if (!loom_low_source_memory_access_scaled_index_from_value(
            module, fact_table, index, recursion_depth + 1,
            &inner_scaled_index)) {
      return false;
    }
    int64_t scaled_multiplier = 0;
    int64_t scaled_offset = 0;
    if (!loom_checked_mul_i64(inner_scaled_index.multiplier, multiplier,
                              &scaled_multiplier) ||
        !loom_checked_mul_i64(inner_scaled_index.offset, multiplier,
                              &scaled_offset)) {
      return false;
    }
    *out_scaled_index = (loom_low_source_memory_scaled_index_t){
        .index = inner_scaled_index.index,
        .multiplier = scaled_multiplier,
        .offset = scaled_offset,
    };
    return true;
  }

  if (loom_index_shli_isa(defining_op)) {
    loom_value_id_t lhs = loom_index_shli_lhs(defining_op);
    loom_value_id_t rhs = loom_index_shli_rhs(defining_op);
    int64_t shift_amount = 0;
    if (!loom_low_source_memory_access_exact_i64(
            loom_value_fact_table_lookup(fact_table, rhs), &shift_amount) ||
        shift_amount < 0 || shift_amount > 62) {
      return true;
    }

    loom_low_source_memory_scaled_index_t inner_scaled_index = {0};
    if (!loom_low_source_memory_access_scaled_index_from_value(
            module, fact_table, lhs, recursion_depth + 1,
            &inner_scaled_index)) {
      return false;
    }
    const int64_t multiplier = ((int64_t)1) << shift_amount;
    int64_t scaled_multiplier = 0;
    int64_t scaled_offset = 0;
    if (!loom_checked_mul_i64(inner_scaled_index.multiplier, multiplier,
                              &scaled_multiplier) ||
        !loom_checked_mul_i64(inner_scaled_index.offset, multiplier,
                              &scaled_offset)) {
      return false;
    }
    *out_scaled_index = (loom_low_source_memory_scaled_index_t){
        .index = inner_scaled_index.index,
        .multiplier = scaled_multiplier,
        .offset = scaled_offset,
    };
    return true;
  }

  if (loom_index_madd_isa(defining_op)) {
    loom_value_id_t a = loom_index_madd_a(defining_op);
    loom_value_id_t b = loom_index_madd_b(defining_op);
    loom_value_id_t c = loom_index_madd_c(defining_op);
    int64_t multiplier = 0;
    loom_value_id_t scaled_value = LOOM_VALUE_ID_INVALID;
    if (loom_low_source_memory_access_exact_positive_i64(fact_table, a,
                                                         &multiplier)) {
      scaled_value = b;
    } else if (loom_low_source_memory_access_exact_positive_i64(fact_table, b,
                                                                &multiplier)) {
      scaled_value = a;
    } else {
      return true;
    }

    int64_t constant = 0;
    if (!loom_low_source_memory_access_exact_i64(
            loom_value_fact_table_lookup(fact_table, c), &constant)) {
      return true;
    }
    if (!loom_low_source_memory_access_can_extract_static_index_offset(
            constant)) {
      return true;
    }

    loom_low_source_memory_scaled_index_t inner_scaled_index = {0};
    if (!loom_low_source_memory_access_scaled_index_from_value(
            module, fact_table, scaled_value, recursion_depth + 1,
            &inner_scaled_index)) {
      return false;
    }
    int64_t scaled_multiplier = 0;
    int64_t scaled_offset = 0;
    int64_t offset = 0;
    if (!loom_checked_mul_i64(inner_scaled_index.multiplier, multiplier,
                              &scaled_multiplier) ||
        !loom_checked_mul_i64(inner_scaled_index.offset, multiplier,
                              &scaled_offset) ||
        !loom_checked_add_i64(scaled_offset, constant, &offset)) {
      return false;
    }
    *out_scaled_index = (loom_low_source_memory_scaled_index_t){
        .index = inner_scaled_index.index,
        .multiplier = scaled_multiplier,
        .offset = offset,
    };
    return true;
  }

  if (loom_index_add_isa(defining_op) || loom_index_sub_isa(defining_op)) {
    loom_value_id_t lhs = loom_index_add_isa(defining_op)
                              ? loom_index_add_lhs(defining_op)
                              : loom_index_sub_lhs(defining_op);
    loom_value_id_t rhs = loom_index_add_isa(defining_op)
                              ? loom_index_add_rhs(defining_op)
                              : loom_index_sub_rhs(defining_op);
    int64_t constant = 0;
    loom_value_id_t scaled_value = LOOM_VALUE_ID_INVALID;
    if (loom_low_source_memory_access_exact_i64(
            loom_value_fact_table_lookup(fact_table, rhs), &constant)) {
      scaled_value = lhs;
      if (loom_index_sub_isa(defining_op) &&
          !loom_checked_sub_i64(0, constant, &constant)) {
        return false;
      }
    } else if (loom_index_add_isa(defining_op) &&
               loom_low_source_memory_access_exact_i64(
                   loom_value_fact_table_lookup(fact_table, lhs), &constant)) {
      scaled_value = rhs;
    } else {
      return true;
    }
    if (!loom_low_source_memory_access_can_extract_static_index_offset(
            constant)) {
      return true;
    }

    loom_low_source_memory_scaled_index_t inner_scaled_index = {0};
    if (!loom_low_source_memory_access_scaled_index_from_value(
            module, fact_table, scaled_value, recursion_depth + 1,
            &inner_scaled_index)) {
      return false;
    }
    int64_t offset = 0;
    if (!loom_checked_add_i64(inner_scaled_index.offset, constant, &offset)) {
      return false;
    }
    *out_scaled_index = (loom_low_source_memory_scaled_index_t){
        .index = inner_scaled_index.index,
        .multiplier = inner_scaled_index.multiplier,
        .offset = offset,
    };
    return true;
  }

  return true;
}

static bool loom_low_source_memory_access_scaled_dynamic_index(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    loom_value_id_t dynamic_index, int64_t byte_stride,
    int64_t static_byte_offset, loom_value_id_t* out_scaled_index,
    int64_t* out_index_multiplier, int64_t* out_index_offset,
    int64_t* out_byte_stride, int64_t* out_static_byte_offset) {
  *out_scaled_index = dynamic_index;
  *out_index_multiplier = 1;
  *out_index_offset = 0;
  *out_byte_stride = byte_stride;
  *out_static_byte_offset = static_byte_offset;

  loom_low_source_memory_scaled_index_t scaled_index = {0};
  if (!loom_low_source_memory_access_scaled_index_from_value(
          module, fact_table, dynamic_index, 0, &scaled_index) ||
      (scaled_index.index == dynamic_index && scaled_index.multiplier == 1 &&
       scaled_index.offset == 0)) {
    return false;
  }

  int64_t scaled_byte_stride = 0;
  int64_t static_offset_delta = 0;
  int64_t new_static_byte_offset = 0;
  if (!iree_checked_mul_i64(byte_stride, scaled_index.multiplier,
                            &scaled_byte_stride) ||
      !loom_checked_mul_i64(byte_stride, scaled_index.offset,
                            &static_offset_delta) ||
      !loom_checked_add_i64(static_byte_offset, static_offset_delta,
                            &new_static_byte_offset)) {
    return false;
  }
  // Leave the original dynamic expression intact if extracting its affine
  // offset would move a non-negative static byte offset outside that domain.
  if (static_byte_offset >= 0 && new_static_byte_offset < 0) {
    return false;
  }
  *out_scaled_index = scaled_index.index;
  *out_index_multiplier = scaled_index.multiplier;
  *out_index_offset = scaled_index.offset;
  *out_byte_stride = scaled_byte_stride;
  *out_static_byte_offset = new_static_byte_offset;
  return true;
}

static bool loom_low_source_memory_access_view_base_value(
    const loom_module_t* module, loom_value_id_t view_value_id,
    loom_value_id_t* out_byte_offset) {
  *out_byte_offset = LOOM_VALUE_ID_INVALID;
  while (view_value_id < module->values.count) {
    const loom_value_t* view_value = loom_module_value(module, view_value_id);
    if (loom_value_is_block_arg(view_value)) {
      return false;
    }
    const loom_op_t* defining_op = loom_value_def_op(view_value);
    if (!defining_op) {
      return false;
    }
    if (loom_view_refine_isa(defining_op)) {
      view_value_id = loom_view_refine_source(defining_op);
      continue;
    }
    if (loom_buffer_view_isa(defining_op)) {
      *out_byte_offset = loom_buffer_view_byte_offset(defining_op);
      return true;
    }
    return false;
  }
  return false;
}

static bool loom_low_source_memory_access_add_dynamic_view_base_byte_offset(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    loom_value_id_t view_value_id, loom_low_source_memory_access_plan_t* plan,
    loom_low_source_memory_access_diagnostic_t* diagnostic,
    int64_t* inout_static_byte_offset) {
  loom_value_id_t byte_offset = LOOM_VALUE_ID_INVALID;
  if (!loom_low_source_memory_access_view_base_value(module, view_value_id,
                                                     &byte_offset)) {
    diagnostic->rejection_bits |=
        LOOM_LOW_SOURCE_MEMORY_ACCESS_REJECTION_VIEW_BASE;
    return false;
  }

  loom_value_id_t dynamic_index = byte_offset;
  int64_t index_multiplier = 1;
  int64_t index_offset = 0;
  int64_t byte_stride = 1;
  const int64_t original_static_byte_offset = *inout_static_byte_offset;
  int64_t static_byte_offset = original_static_byte_offset;
  (void)loom_low_source_memory_access_scaled_dynamic_index(
      module, fact_table, byte_offset, byte_stride, static_byte_offset,
      &dynamic_index, &index_multiplier, &index_offset, &byte_stride,
      &static_byte_offset);
  int64_t static_view_base_byte_offset = 0;
  if (!loom_checked_sub_i64(static_byte_offset, original_static_byte_offset,
                            &static_view_base_byte_offset)) {
    diagnostic->rejection_bits |=
        LOOM_LOW_SOURCE_MEMORY_ACCESS_REJECTION_VIEW_BASE_OVERFLOW;
    return false;
  }
  *inout_static_byte_offset = static_byte_offset;

  loom_low_source_memory_dynamic_term_t term = {
      .index = dynamic_index,
      .axis = LOOM_LOW_SOURCE_MEMORY_DYNAMIC_TERM_AXIS_NONE,
      .byte_stride = byte_stride,
      .byte_shift = LOOM_LOW_SOURCE_MEMORY_ACCESS_BYTE_SHIFT_NONE,
  };
  loom_low_source_memory_access_dynamic_index_source(
      module, dynamic_index, &term.source, &term.dimension);
  (void)loom_low_source_memory_access_power_of_two_shift(byte_stride,
                                                         &term.byte_shift);
  loom_low_source_memory_dynamic_term_compute_scaled_byte_facts(
      fact_table, dynamic_index, byte_stride, NULL, 0, &term.byte_facts);
  IREE_ASSERT_EQ(plan->dynamic_term_count, 0u);
  if (!loom_low_source_memory_access_append_dynamic_term(plan, &term,
                                                         diagnostic)) {
    return false;
  }
  plan->static_view_base_byte_offset = static_view_base_byte_offset;
  plan->dynamic_view_base_term_count = 1;
  return true;
}

static bool loom_low_source_memory_access_add_view_base_byte_offset(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    loom_value_id_t view_value_id, loom_low_source_memory_access_plan_t* plan,
    loom_low_source_memory_access_diagnostic_t* diagnostic,
    int64_t* inout_static_byte_offset) {
  loom_value_fact_view_reference_t view_reference = {0};
  if (!loom_value_facts_query_view_reference(
          &fact_table->context,
          loom_value_fact_table_lookup(fact_table, view_value_id),
          &view_reference)) {
    diagnostic->rejection_bits |=
        LOOM_LOW_SOURCE_MEMORY_ACCESS_REJECTION_VIEW_SOURCE;
    return false;
  }

  int64_t view_base_byte_offset = 0;
  if (loom_low_source_memory_access_exact_i64(view_reference.base_byte_offset,
                                              &view_base_byte_offset)) {
    int64_t static_byte_offset = 0;
    if (!iree_checked_add_i64(*inout_static_byte_offset, view_base_byte_offset,
                              &static_byte_offset)) {
      diagnostic->rejection_bits |=
          LOOM_LOW_SOURCE_MEMORY_ACCESS_REJECTION_VIEW_BASE_OVERFLOW;
      return false;
    }
    plan->static_view_base_byte_offset = view_base_byte_offset;
    *inout_static_byte_offset = static_byte_offset;
  } else if (!loom_low_source_memory_access_add_dynamic_view_base_byte_offset(
                 module, fact_table, view_value_id, plan, diagnostic,
                 inout_static_byte_offset)) {
    return false;
  }

  plan->memory_space = view_reference.memory_space;
  plan->root_value_id = view_reference.root_value_id;
  plan->root_minimum_alignment = loom_low_source_memory_clamp_alignment(
      view_reference.root_minimum_alignment);
  plan->alias_scope_id = view_reference.alias_scope_id;
  return true;
}

static loom_low_memory_space_t loom_low_source_memory_access_space(
    loom_value_fact_memory_space_t memory_space) {
  switch (memory_space) {
    case LOOM_VALUE_FACT_MEMORY_SPACE_GLOBAL:
    case LOOM_VALUE_FACT_MEMORY_SPACE_CONSTANT:
    case LOOM_VALUE_FACT_MEMORY_SPACE_DESCRIPTOR:
      return LOOM_LOW_MEMORY_SPACE_GLOBAL;
    case LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP:
      return LOOM_LOW_MEMORY_SPACE_WORKGROUP;
    case LOOM_VALUE_FACT_MEMORY_SPACE_PRIVATE:
      return LOOM_LOW_MEMORY_SPACE_STACK;
    case LOOM_VALUE_FACT_MEMORY_SPACE_HOST:
    case LOOM_VALUE_FACT_MEMORY_SPACE_GENERIC:
    case LOOM_VALUE_FACT_MEMORY_SPACE_UNKNOWN:
    default:
      return LOOM_LOW_MEMORY_SPACE_GENERIC;
  }
}

static bool loom_low_source_memory_access_lane_byte_envelope(
    const loom_low_source_memory_access_plan_t* plan, int64_t* out_begin_offset,
    int64_t* out_end_offset) {
  *out_begin_offset = 0;
  *out_end_offset = 0;
  if (plan->vector_lane_count == 0 || plan->element_byte_count == 0) {
    return false;
  }

  int64_t last_lane_offset = 0;
  if (!iree_checked_mul_i64((int64_t)(plan->vector_lane_count - 1),
                            plan->vector_lane_byte_stride, &last_lane_offset)) {
    return false;
  }
  int64_t begin_offset = loom_min_i64(0, last_lane_offset);
  int64_t end_offset = 0;
  if (!iree_checked_add_i64(loom_max_i64(0, last_lane_offset),
                            (int64_t)plan->element_byte_count, &end_offset)) {
    return false;
  }
  *out_begin_offset = begin_offset;
  *out_end_offset = end_offset;
  return true;
}

void loom_low_source_memory_access_plan_make_summary(
    const loom_low_source_memory_access_plan_t* plan,
    loom_low_byte_interval_t* out_interval,
    loom_low_memory_access_summary_t* out_summary) {
  const loom_low_memory_space_t memory_space =
      loom_low_memory_access_normalize_space(
          loom_low_source_memory_access_space(plan->memory_space));
  loom_low_memory_access_precision_flags_t precision_flags = 0;
  if (memory_space != LOOM_LOW_MEMORY_SPACE_GENERIC) {
    precision_flags |= LOOM_LOW_MEMORY_ACCESS_PRECISION_SPACE;
  }
  uint32_t alias_root_id = LOOM_LOW_MEMORY_ALIAS_ID_NONE;
  if (plan->alias_scope_id != LOOM_VALUE_FACT_ALIAS_SCOPE_ID_NONE) {
    alias_root_id = plan->alias_scope_id;
    precision_flags |= LOOM_LOW_MEMORY_ACCESS_PRECISION_ROOT;
  }

  *out_interval = (loom_low_byte_interval_t){0};
  const loom_low_byte_interval_t* interval = NULL;
  int64_t lane_begin_offset = 0;
  int64_t lane_end_offset = 0;
  if (loom_low_source_memory_access_lane_byte_envelope(plan, &lane_begin_offset,
                                                       &lane_end_offset)) {
    loom_value_facts_t begin_facts =
        loom_value_facts_exact_i64(plan->static_byte_offset);
    for (uint8_t i = 0; i < plan->dynamic_term_count; ++i) {
      loom_value_facts_addi(&begin_facts, &plan->dynamic_terms[i].byte_facts,
                            &begin_facts);
    }
    loom_value_facts_t end_facts = begin_facts;
    const loom_value_facts_t begin_adjustment =
        loom_value_facts_exact_i64(lane_begin_offset);
    const loom_value_facts_t end_adjustment =
        loom_value_facts_exact_i64(lane_end_offset);
    loom_value_facts_addi(&begin_facts, &begin_adjustment, &begin_facts);
    loom_value_facts_addi(&end_facts, &end_adjustment, &end_facts);
    *out_interval = (loom_low_byte_interval_t){
        .begin_facts = begin_facts,
        .end_facts = end_facts,
        .begin_expr_id = LOOM_LOW_MEMORY_EXPR_ID_NONE,
        .end_expr_id = LOOM_LOW_MEMORY_EXPR_ID_NONE,
        .precision_flags = LOOM_LOW_BYTE_INTERVAL_PRECISION_BEGIN_RANGE |
                           LOOM_LOW_BYTE_INTERVAL_PRECISION_END_RANGE |
                           LOOM_LOW_BYTE_INTERVAL_PRECISION_EXACT_LENGTH,
    };
    precision_flags |= LOOM_LOW_MEMORY_ACCESS_PRECISION_INTERVAL;
    interval = out_interval;
  }

  *out_summary = (loom_low_memory_access_summary_t){
      .memory_space = memory_space,
      .alias_root_id = alias_root_id,
      .alias_group_id = LOOM_LOW_MEMORY_ALIAS_ID_NONE,
      .precision_flags = precision_flags,
      .byte_interval = interval,
  };
}

static bool loom_low_source_memory_operation_kind_from_op(
    const loom_op_t* source_op,
    loom_low_source_memory_operation_kind_t* out_operation_kind) {
  switch (source_op->kind) {
    case LOOM_OP_VECTOR_FRAGMENT_LOAD:
    case LOOM_OP_VECTOR_LOAD:
    case LOOM_OP_VIEW_LOAD:
      *out_operation_kind = LOOM_LOW_SOURCE_MEMORY_OPERATION_LOAD;
      return true;
    case LOOM_OP_VECTOR_FRAGMENT_STORE:
    case LOOM_OP_VECTOR_STORE:
    case LOOM_OP_VIEW_STORE:
      *out_operation_kind = LOOM_LOW_SOURCE_MEMORY_OPERATION_STORE;
      return true;
    case LOOM_OP_VIEW_ATOMIC_REDUCE:
      *out_operation_kind = LOOM_LOW_SOURCE_MEMORY_OPERATION_ATOMIC_REDUCE;
      return true;
    case LOOM_OP_VIEW_ATOMIC_RMW:
      *out_operation_kind = LOOM_LOW_SOURCE_MEMORY_OPERATION_ATOMIC_RMW;
      return true;
    case LOOM_OP_VIEW_ATOMIC_CMPXCHG:
      *out_operation_kind = LOOM_LOW_SOURCE_MEMORY_OPERATION_ATOMIC_CMPXCHG;
      return true;
    case LOOM_OP_VIEW_PREFETCH:
      *out_operation_kind = LOOM_LOW_SOURCE_MEMORY_OPERATION_PREFETCH;
      return true;
    default:
      *out_operation_kind = LOOM_LOW_SOURCE_MEMORY_OPERATION_LOAD;
      return false;
  }
}

static loom_type_t loom_low_source_memory_element_vector_type(
    loom_type_t view_type) {
  return loom_type_shaped_1d(LOOM_TYPE_VECTOR,
                             loom_type_element_type(view_type),
                             loom_dim_pack_static(1), /*encoding_id=*/0);
}

static loom_type_t loom_low_source_memory_access_payload_vector_type(
    const loom_module_t* module, const loom_op_t* source_op,
    loom_memory_access_t access, loom_type_t view_type) {
  const loom_value_id_t value_id = loom_memory_access_value(access);
  if (value_id != LOOM_VALUE_ID_INVALID && value_id < module->values.count) {
    const loom_type_t value_type = loom_module_value_type(module, value_id);
    if (loom_type_is_vector(value_type)) {
      return value_type;
    }
  }
  if (source_op->result_count == 1) {
    const loom_value_id_t result_id = loom_op_const_results(source_op)[0];
    if (result_id < module->values.count) {
      const loom_type_t result_type = loom_module_value_type(module, result_id);
      if (loom_type_is_vector(result_type)) {
        return result_type;
      }
    }
  }
  return loom_low_source_memory_element_vector_type(view_type);
}

static bool loom_low_source_memory_access_plan_from_components(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    loom_low_source_memory_operation_kind_t operation_kind,
    loom_value_id_t view_value_id, loom_value_slice_t dynamic_indices,
    loom_attribute_t static_indices, loom_type_t view_type,
    loom_type_t vector_type, loom_vector_memory_cache_policy_t cache_policy,
    loom_low_source_memory_access_plan_t* out_plan,
    loom_low_source_memory_access_diagnostic_t* out_diagnostic) {
  *out_plan = (loom_low_source_memory_access_plan_t){
      .operation_kind = operation_kind,
      .view_value_id = view_value_id,
      .memory_space = LOOM_VALUE_FACT_MEMORY_SPACE_UNKNOWN,
      .root_value_id = LOOM_VALUE_ID_INVALID,
      .root_minimum_alignment = 1,
      .alias_scope_id = LOOM_VALUE_FACT_ALIAS_SCOPE_ID_NONE,
      .minimum_alignment = 1,
      .cache_policy = cache_policy,
  };
  *out_diagnostic = (loom_low_source_memory_access_diagnostic_t){0};

  loom_vector_memory_access_t vector_access;
  const loom_fact_context_t* fact_context =
      fact_table ? &fact_table->context : NULL;
  if (!loom_vector_memory_access_describe(fact_context, module, view_type,
                                          vector_type, &vector_access)) {
    out_diagnostic->rejection_bits |=
        LOOM_LOW_SOURCE_MEMORY_ACCESS_REJECTION_DESCRIBE_FAILED;
    return false;
  }
  switch (vector_access.layout_kind) {
    case LOOM_VECTOR_MEMORY_LAYOUT_DENSE:
    case LOOM_VECTOR_MEMORY_LAYOUT_STRIDED:
      break;
    case LOOM_VECTOR_MEMORY_LAYOUT_UNKNOWN:
      out_diagnostic->rejection_bits |=
          LOOM_LOW_SOURCE_MEMORY_ACCESS_REJECTION_LAYOUT;
      return false;
  }
  if (vector_access.static_element_byte_count <= 0 ||
      vector_access.static_element_byte_count > UINT32_MAX) {
    out_diagnostic->rejection_bits |=
        LOOM_LOW_SOURCE_MEMORY_ACCESS_REJECTION_ELEMENT_WIDTH;
    return false;
  }
  out_plan->element_byte_count =
      (uint32_t)vector_access.static_element_byte_count;
  if (vector_access.vector_rank != 1) {
    out_diagnostic->rejection_bits |=
        LOOM_LOW_SOURCE_MEMORY_ACCESS_REJECTION_VECTOR_RANK;
    return false;
  }
  if (!loom_low_source_memory_access_vector_lane_count(
          vector_type, &out_plan->vector_lane_count)) {
    out_diagnostic->rejection_bits |=
        LOOM_LOW_SOURCE_MEMORY_ACCESS_REJECTION_VECTOR_LANE_COUNT;
    return false;
  }

  int64_t vector_axis_stride = 0;
  if (!loom_vector_memory_access_static_axis_stride(
          &vector_access, vector_access.first_vector_axis,
          &vector_axis_stride) ||
      !iree_checked_mul_i64(vector_axis_stride,
                            vector_access.static_element_byte_count,
                            &out_plan->vector_lane_byte_stride)) {
    out_diagnostic->rejection_bits |=
        LOOM_LOW_SOURCE_MEMORY_ACCESS_REJECTION_VECTOR_AXIS_STRIDE;
    return false;
  }

  uint8_t dynamic_axes[LOOM_LOW_SOURCE_MEMORY_DYNAMIC_TERM_CAPACITY] = {0};
  uint8_t dynamic_axis_count = 0;
  if (!loom_low_source_memory_access_collect_dynamic_axes(
          static_indices, dynamic_axes, &dynamic_axis_count) ||
      dynamic_axis_count != dynamic_indices.count) {
    out_diagnostic->rejection_bits |=
        LOOM_LOW_SOURCE_MEMORY_ACCESS_REJECTION_DYNAMIC_INDEX_COUNT;
    return false;
  }
  int64_t folded_static_indices[LOOM_ENCODING_ADDRESS_LAYOUT_MAX_RANK] = {0};
  loom_value_id_t
      dynamic_index_values[LOOM_LOW_SOURCE_MEMORY_DYNAMIC_TERM_CAPACITY] = {0};
  loom_low_source_memory_access_fold_exact_dynamic_indices(
      fact_table, dynamic_indices, static_indices, folded_static_indices,
      dynamic_axes, dynamic_index_values, &dynamic_axis_count);
  static_indices =
      loom_attr_i64_array(folded_static_indices, static_indices.count);

  int64_t static_byte_offset = 0;
  if (!loom_low_source_memory_access_static_byte_offset(
          &vector_access, static_indices, dynamic_axes, dynamic_axis_count,
          &static_byte_offset)) {
    out_diagnostic->rejection_bits |=
        LOOM_LOW_SOURCE_MEMORY_ACCESS_REJECTION_STATIC_OFFSET;
    return false;
  }
  if (!loom_low_source_memory_access_add_view_base_byte_offset(
          module, fact_table, view_value_id, out_plan, out_diagnostic,
          &static_byte_offset)) {
    return false;
  }

  for (uint8_t i = 0; i < dynamic_axis_count; ++i) {
    const uint8_t dynamic_axis = dynamic_axes[i];
    if (dynamic_axis >= vector_access.view_rank) {
      out_diagnostic->rejection_bits |=
          LOOM_LOW_SOURCE_MEMORY_ACCESS_REJECTION_DYNAMIC_AXIS;
      return false;
    }

    int64_t byte_stride = 0;
    loom_value_id_t
        stride_values[LOOM_LOW_SOURCE_MEMORY_DYNAMIC_TERM_CAPACITY] = {0};
    uint8_t stride_value_count = 0;
    if (!loom_low_source_memory_access_dynamic_axis_byte_stride(
            &vector_access, dynamic_axis, &byte_stride, stride_values,
            &stride_value_count)) {
      out_diagnostic->rejection_bits |=
          LOOM_LOW_SOURCE_MEMORY_ACCESS_REJECTION_DYNAMIC_STRIDE;
      return false;
    }

    loom_value_id_t dynamic_index = dynamic_index_values[i];
    loom_low_source_memory_affine_index_term_t
        affine_terms[LOOM_LOW_SOURCE_MEMORY_DYNAMIC_TERM_CAPACITY] = {0};
    uint8_t affine_term_count = 0;
    int64_t affine_index_offset = 0;
    bool affine_expanded =
        loom_low_source_memory_access_affine_index_terms_from_value(
            module, fact_table, dynamic_index, /*multiplier=*/1,
            /*recursion_depth=*/0, affine_terms, &affine_term_count,
            &affine_index_offset) &&
        affine_term_count > 1 &&
        loom_low_source_memory_access_affine_terms_have_mixed_coordinate_sources(
            module, affine_terms, affine_term_count) &&
        loom_low_source_memory_access_can_extract_static_index_offset(
            affine_index_offset);
    int64_t affine_static_byte_offset = static_byte_offset;
    if (affine_expanded) {
      int64_t static_offset_delta = 0;
      if (!loom_checked_mul_i64(byte_stride, affine_index_offset,
                                &static_offset_delta) ||
          !loom_checked_add_i64(static_byte_offset, static_offset_delta,
                                &affine_static_byte_offset) ||
          (static_byte_offset >= 0 && affine_static_byte_offset < 0)) {
        affine_expanded = false;
      }
    }
    if (affine_expanded) {
      static_byte_offset = affine_static_byte_offset;
      for (uint8_t term_ordinal = 0; term_ordinal < affine_term_count;
           ++term_ordinal) {
        const loom_low_source_memory_affine_index_term_t* affine_term =
            &affine_terms[term_ordinal];
        int64_t term_byte_stride = 0;
        if (!iree_checked_mul_i64(byte_stride, affine_term->multiplier,
                                  &term_byte_stride)) {
          out_diagnostic->rejection_bits |=
              LOOM_LOW_SOURCE_MEMORY_ACCESS_REJECTION_DYNAMIC_STRIDE;
          return false;
        }
        uint32_t byte_shift = LOOM_LOW_SOURCE_MEMORY_ACCESS_BYTE_SHIFT_NONE;
        (void)loom_low_source_memory_access_power_of_two_shift(term_byte_stride,
                                                               &byte_shift);
        loom_value_facts_t byte_facts = loom_value_facts_unknown();
        loom_low_source_memory_dynamic_term_compute_scaled_byte_facts(
            fact_table, affine_term->index, term_byte_stride, stride_values,
            stride_value_count, &byte_facts);
        loom_low_source_memory_dynamic_term_t term = {
            .index = affine_term->index,
            .axis = LOOM_LOW_SOURCE_MEMORY_DYNAMIC_TERM_AXIS_NONE,
            .byte_stride = term_byte_stride,
            .byte_facts = byte_facts,
            .byte_shift = byte_shift,
            .stride_value_count = stride_value_count,
        };
        loom_low_source_memory_access_dynamic_index_source(
            module, affine_term->index, &term.source, &term.dimension);
        for (uint8_t stride_ordinal = 0; stride_ordinal < stride_value_count;
             ++stride_ordinal) {
          term.stride_values[stride_ordinal] = stride_values[stride_ordinal];
        }
        if (!loom_low_source_memory_access_append_dynamic_term(
                out_plan, &term, out_diagnostic)) {
          return false;
        }
      }
      continue;
    }

    int64_t dynamic_index_multiplier = 1;
    int64_t dynamic_index_offset = 0;
    // Keep the exact source coordinate in the access plan when a dynamic index
    // is a simple affine expression of another value. Targets then see the
    // combined byte stride/static offset and do not need to materialize the
    // intermediate index.
    (void)loom_low_source_memory_access_scaled_dynamic_index(
        module, fact_table, dynamic_index, byte_stride, static_byte_offset,
        &dynamic_index, &dynamic_index_multiplier, &dynamic_index_offset,
        &byte_stride, &static_byte_offset);
    uint32_t byte_shift = LOOM_LOW_SOURCE_MEMORY_ACCESS_BYTE_SHIFT_NONE;
    (void)loom_low_source_memory_access_power_of_two_shift(byte_stride,
                                                           &byte_shift);
    loom_value_facts_t byte_facts = loom_value_facts_unknown();
    loom_low_source_memory_dynamic_term_compute_byte_facts(
        fact_table, &vector_access, dynamic_index, dynamic_axis,
        dynamic_index_multiplier, dynamic_index_offset, byte_stride,
        stride_values, stride_value_count, &byte_facts);
    loom_low_source_memory_dynamic_term_t term = {
        .index = dynamic_index,
        .axis = dynamic_axis,
        .byte_stride = byte_stride,
        .byte_facts = byte_facts,
        .byte_shift = byte_shift,
        .stride_value_count = stride_value_count,
    };
    loom_low_source_memory_access_dynamic_index_source(
        module, dynamic_index, &term.source, &term.dimension);
    for (uint8_t stride_ordinal = 0; stride_ordinal < stride_value_count;
         ++stride_ordinal) {
      term.stride_values[stride_ordinal] = stride_values[stride_ordinal];
    }
    if (!loom_low_source_memory_access_append_dynamic_term(out_plan, &term,
                                                           out_diagnostic)) {
      return false;
    }
  }
  out_plan->static_byte_offset = static_byte_offset;
  loom_low_source_memory_access_finalize_alignment(out_plan);
  return true;
}

bool loom_low_source_memory_access_plan_build(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    const loom_op_t* source_op, loom_low_source_memory_access_plan_t* out_plan,
    loom_low_source_memory_access_diagnostic_t* out_diagnostic) {
  *out_plan = (loom_low_source_memory_access_plan_t){0};
  *out_diagnostic = (loom_low_source_memory_access_diagnostic_t){0};

  loom_low_source_memory_operation_kind_t operation_kind =
      LOOM_LOW_SOURCE_MEMORY_OPERATION_LOAD;
  if (!loom_low_source_memory_operation_kind_from_op(source_op,
                                                     &operation_kind)) {
    out_diagnostic->rejection_bits |=
        LOOM_LOW_SOURCE_MEMORY_ACCESS_REJECTION_UNSUPPORTED_OP;
    return false;
  }

  loom_memory_access_t access = loom_memory_access_cast(module, source_op);
  if (!loom_memory_access_isa(access)) {
    out_diagnostic->rejection_bits |=
        LOOM_LOW_SOURCE_MEMORY_ACCESS_REJECTION_UNSUPPORTED_OP;
    return false;
  }
  const loom_value_id_t view_value_id = loom_memory_access_view(access);
  if (view_value_id >= module->values.count) {
    out_diagnostic->rejection_bits |=
        LOOM_LOW_SOURCE_MEMORY_ACCESS_REJECTION_VIEW_SOURCE;
    return false;
  }

  loom_vector_memory_cache_policy_t cache_policy = {0};
  if (!loom_vector_memory_cache_policy_from_attrs(
          loom_memory_access_cache_scope(access),
          loom_memory_access_cache_temporal(access), &cache_policy)) {
    out_diagnostic->rejection_bits |=
        LOOM_LOW_SOURCE_MEMORY_ACCESS_REJECTION_CACHE_POLICY;
    return false;
  }

  const loom_type_t view_type = loom_module_value_type(module, view_value_id);
  const loom_type_t vector_type =
      loom_low_source_memory_access_payload_vector_type(module, source_op,
                                                        access, view_type);
  return loom_low_source_memory_access_plan_from_components(
      module, fact_table, operation_kind, view_value_id,
      loom_memory_access_dynamic_indices(access),
      loom_memory_access_static_indices(access), view_type, vector_type,
      cache_policy, out_plan, out_diagnostic);
}

bool loom_low_source_memory_access_plan_build_view(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    loom_low_source_memory_operation_kind_t operation_kind,
    loom_value_id_t view_value_id,
    loom_vector_memory_cache_policy_t cache_policy,
    loom_low_source_memory_access_plan_t* out_plan,
    loom_low_source_memory_access_diagnostic_t* out_diagnostic) {
  *out_plan = (loom_low_source_memory_access_plan_t){0};
  *out_diagnostic = (loom_low_source_memory_access_diagnostic_t){0};
  if (view_value_id >= module->values.count) {
    out_diagnostic->rejection_bits |=
        LOOM_LOW_SOURCE_MEMORY_ACCESS_REJECTION_VIEW_SOURCE;
    return false;
  }

  const loom_type_t result_view_type =
      loom_module_value_type(module, view_value_id);
  loom_type_t vector_type = loom_type_none();
  if (!loom_low_source_memory_static_view_vector_type(
          result_view_type, &vector_type, out_diagnostic)) {
    return false;
  }

  loom_value_id_t access_view_id = view_value_id;
  loom_value_slice_t dynamic_indices = {0};
  int64_t zero_indices[LOOM_ENCODING_ADDRESS_LAYOUT_MAX_RANK] = {0};
  loom_attribute_t static_indices =
      loom_attr_i64_array(zero_indices, loom_type_rank(result_view_type));
  loom_type_t access_view_type = result_view_type;

  loom_value_id_t projection_view_id = view_value_id;
  const loom_value_t* view_value = loom_module_value(module, view_value_id);
  if (!loom_value_is_block_arg(view_value)) {
    const loom_op_t* defining_op = loom_value_def_op(view_value);
    if (defining_op && loom_view_refine_isa(defining_op)) {
      projection_view_id = loom_view_refine_source(defining_op);
    }
  }

  const loom_value_t* projection_view =
      loom_module_value(module, projection_view_id);
  if (!loom_value_is_block_arg(projection_view)) {
    const loom_op_t* defining_op = loom_value_def_op(projection_view);
    if (defining_op && loom_view_subview_isa(defining_op)) {
      access_view_id = loom_view_subview_source(defining_op);
      access_view_type = loom_module_value_type(module, access_view_id);
      dynamic_indices = loom_view_subview_offsets(defining_op);
      static_indices = loom_view_subview_static_offsets(defining_op);
    }
  }

  return loom_low_source_memory_access_plan_from_components(
      module, fact_table, operation_kind, access_view_id, dynamic_indices,
      static_indices, access_view_type, vector_type, cache_policy, out_plan,
      out_diagnostic);
}

iree_string_view_t loom_low_source_memory_access_rejection_key(
    loom_low_source_memory_access_rejection_flags_t rejection_bits) {
  if (iree_any_bit_set(
          rejection_bits,
          LOOM_LOW_SOURCE_MEMORY_ACCESS_REJECTION_UNSUPPORTED_OP)) {
    return IREE_SV("source_memory.unsupported_op");
  }
  if (iree_any_bit_set(
          rejection_bits,
          LOOM_LOW_SOURCE_MEMORY_ACCESS_REJECTION_DESCRIBE_FAILED)) {
    return IREE_SV("source_memory.describe_failed");
  }
  if (iree_any_bit_set(rejection_bits,
                       LOOM_LOW_SOURCE_MEMORY_ACCESS_REJECTION_LAYOUT)) {
    return IREE_SV("source_memory.layout");
  }
  if (iree_any_bit_set(rejection_bits,
                       LOOM_LOW_SOURCE_MEMORY_ACCESS_REJECTION_ELEMENT_WIDTH)) {
    return IREE_SV("source_memory.element_width");
  }
  if (iree_any_bit_set(rejection_bits,
                       LOOM_LOW_SOURCE_MEMORY_ACCESS_REJECTION_VECTOR_RANK)) {
    return IREE_SV("source_memory.vector_rank");
  }
  if (iree_any_bit_set(
          rejection_bits,
          LOOM_LOW_SOURCE_MEMORY_ACCESS_REJECTION_VECTOR_LANE_COUNT)) {
    return IREE_SV("source_memory.vector_lane_count");
  }
  if (iree_any_bit_set(
          rejection_bits,
          LOOM_LOW_SOURCE_MEMORY_ACCESS_REJECTION_VECTOR_AXIS_STRIDE)) {
    return IREE_SV("source_memory.vector_axis_stride");
  }
  if (iree_any_bit_set(rejection_bits,
                       LOOM_LOW_SOURCE_MEMORY_ACCESS_REJECTION_STATIC_OFFSET)) {
    return IREE_SV("source_memory.static_offset");
  }
  if (iree_any_bit_set(
          rejection_bits,
          LOOM_LOW_SOURCE_MEMORY_ACCESS_REJECTION_DYNAMIC_INDEX_COUNT)) {
    return IREE_SV("source_memory.dynamic_index_count");
  }
  if (iree_any_bit_set(rejection_bits,
                       LOOM_LOW_SOURCE_MEMORY_ACCESS_REJECTION_DYNAMIC_AXIS)) {
    return IREE_SV("source_memory.dynamic_axis");
  }
  if (iree_any_bit_set(
          rejection_bits,
          LOOM_LOW_SOURCE_MEMORY_ACCESS_REJECTION_DYNAMIC_STRIDE)) {
    return IREE_SV("source_memory.dynamic_stride");
  }
  if (iree_any_bit_set(rejection_bits,
                       LOOM_LOW_SOURCE_MEMORY_ACCESS_REJECTION_VIEW_SOURCE)) {
    return IREE_SV("source_memory.view_source");
  }
  if (iree_any_bit_set(rejection_bits,
                       LOOM_LOW_SOURCE_MEMORY_ACCESS_REJECTION_VIEW_BASE)) {
    return IREE_SV("source_memory.view_base");
  }
  if (iree_any_bit_set(
          rejection_bits,
          LOOM_LOW_SOURCE_MEMORY_ACCESS_REJECTION_VIEW_BASE_OVERFLOW)) {
    return IREE_SV("source_memory.view_base_overflow");
  }
  if (iree_any_bit_set(rejection_bits,
                       LOOM_LOW_SOURCE_MEMORY_ACCESS_REJECTION_CACHE_POLICY)) {
    return IREE_SV("source_memory.cache_policy");
  }
  return IREE_SV("source_memory.representability");
}
