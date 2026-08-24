// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/source_memory_plan.h"

#include <stdint.h>

#include "iree/base/api.h"
#include "iree/base/internal/math.h"
#include "loom/analysis/view_regions.h"
#include "loom/codegen/low/memory_access.h"
#include "loom/ir/facts.h"
#include "loom/ir/module.h"
#include "loom/ops/kernel/ops.h"
#include "loom/ops/vector/ops.h"
#include "loom/ops/view/ops.h"

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

static bool loom_low_source_memory_access_exact_i64_at(
    const loom_value_facts_t* facts, iree_host_size_t count,
    iree_host_size_t ordinal, int64_t* out_value) {
  *out_value = 0;
  if (ordinal >= count) {
    return false;
  }
  return loom_low_source_memory_access_exact_i64(facts[ordinal], out_value);
}

static bool loom_low_source_memory_access_offset_facts_are_identity_iota(
    const loom_fact_context_t* context, loom_value_facts_t facts) {
  loom_value_fact_vector_iota_t iota = {0};
  if (loom_value_facts_query_vector_iota(context, facts, &iota)) {
    int64_t base = 0;
    int64_t step = 0;
    return loom_low_source_memory_access_exact_i64(iota.base, &base) &&
           loom_low_source_memory_access_exact_i64(iota.step, &step) &&
           base == 0 && step == 1;
  }

  loom_value_fact_small_static_lanes_t lanes = {0};
  if (!loom_value_facts_query_small_static_lanes(context, facts, &lanes)) {
    return false;
  }
  for (iree_host_size_t i = 0; i < lanes.count; ++i) {
    int64_t lane_offset = 0;
    if (!loom_low_source_memory_access_exact_i64_at(lanes.lanes, lanes.count, i,
                                                    &lane_offset) ||
        lane_offset != (int64_t)i) {
      return false;
    }
  }
  return true;
}

static loom_low_source_memory_vector_offset_kind_t
loom_low_source_memory_access_vector_offset_kind(
    const loom_value_fact_table_t* fact_table, loom_value_id_t offset_value) {
  if (offset_value == LOOM_VALUE_ID_INVALID) {
    return LOOM_LOW_SOURCE_MEMORY_VECTOR_OFFSET_NONE;
  }
  const loom_value_facts_t facts =
      loom_value_fact_table_lookup(fact_table, offset_value);
  return loom_low_source_memory_access_offset_facts_are_identity_iota(
             &fact_table->context, facts)
             ? LOOM_LOW_SOURCE_MEMORY_VECTOR_OFFSET_IDENTITY_IOTA
             : LOOM_LOW_SOURCE_MEMORY_VECTOR_OFFSET_OTHER;
}

static uint32_t loom_low_source_memory_clamp_alignment(uint64_t alignment) {
  if (alignment == 0) return 1;
  return alignment > UINT32_MAX ? UINT32_MAX : (uint32_t)alignment;
}

static uint32_t loom_low_source_memory_combine_alignment(uint32_t alignment,
                                                         int64_t byte_offset) {
  if (byte_offset == 0) return alignment == 0 ? 1 : alignment;
  return (uint32_t)iree_math_gcd_i64((int64_t)alignment, byte_offset);
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

static void loom_low_source_memory_access_dynamic_index_source(
    const loom_value_fact_table_t* fact_table, loom_value_id_t index,
    loom_low_source_memory_dynamic_index_source_t* out_source,
    loom_kernel_dimension_t* out_dimension) {
  static const loom_low_source_memory_dynamic_index_source_t
      kSources[LOOM_VALUE_FACT_TOPOLOGY_VALUE_COUNT_] = {
          [LOOM_VALUE_FACT_TOPOLOGY_VALUE_NONE] =
              LOOM_LOW_SOURCE_MEMORY_DYNAMIC_INDEX_SOURCE_VALUE,
          [LOOM_VALUE_FACT_TOPOLOGY_VALUE_WORKITEM_ID] =
              LOOM_LOW_SOURCE_MEMORY_DYNAMIC_INDEX_SOURCE_WORKITEM_ID,
          [LOOM_VALUE_FACT_TOPOLOGY_VALUE_WORKGROUP_ID] =
              LOOM_LOW_SOURCE_MEMORY_DYNAMIC_INDEX_SOURCE_WORKGROUP_ID,
          [LOOM_VALUE_FACT_TOPOLOGY_VALUE_SUBGROUP_LANE_ID] =
              LOOM_LOW_SOURCE_MEMORY_DYNAMIC_INDEX_SOURCE_VALUE,
          [LOOM_VALUE_FACT_TOPOLOGY_VALUE_CLUSTER_ID] =
              LOOM_LOW_SOURCE_MEMORY_DYNAMIC_INDEX_SOURCE_VALUE,
          [LOOM_VALUE_FACT_TOPOLOGY_VALUE_CLUSTER_WORKGROUP_ID] =
              LOOM_LOW_SOURCE_MEMORY_DYNAMIC_INDEX_SOURCE_VALUE,
      };
  static const loom_kernel_dimension_t
      kDimensions[LOOM_VALUE_FACT_TOPOLOGY_AXIS_COUNT_] = {
          [LOOM_VALUE_FACT_TOPOLOGY_AXIS_X] = LOOM_KERNEL_DIMENSION_X,
          [LOOM_VALUE_FACT_TOPOLOGY_AXIS_Y] = LOOM_KERNEL_DIMENSION_Y,
          [LOOM_VALUE_FACT_TOPOLOGY_AXIS_Z] = LOOM_KERNEL_DIMENSION_Z,
          [LOOM_VALUE_FACT_TOPOLOGY_AXIS_LANE] = LOOM_KERNEL_DIMENSION_COUNT_,
      };
  *out_source = LOOM_LOW_SOURCE_MEMORY_DYNAMIC_INDEX_SOURCE_VALUE;
  *out_dimension = LOOM_KERNEL_DIMENSION_COUNT_;
  if (index == LOOM_VALUE_ID_INVALID ||
      !loom_value_fact_table_has_entry(fact_table, index)) {
    return;
  }
  const loom_value_fact_topology_domain_t* domain =
      loom_value_facts_topology_domain(
          loom_value_fact_table_lookup(fact_table, index));
  if (!domain) {
    return;
  }
  *out_source = kSources[domain->value_kind];
  *out_dimension = kDimensions[domain->axis];
  if (*out_dimension == LOOM_KERNEL_DIMENSION_COUNT_) {
    *out_source = LOOM_LOW_SOURCE_MEMORY_DYNAMIC_INDEX_SOURCE_VALUE;
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
    const loom_value_fact_table_t* fact_table,
    const loom_vector_memory_access_t* vector_access,
    loom_attribute_t static_indices, const uint8_t* dynamic_axes,
    uint8_t dynamic_axis_count, uint8_t* out_dynamic_stride_axes,
    uint8_t* out_dynamic_stride_axis_count, int64_t* out_static_byte_offset) {
  *out_static_byte_offset = 0;
  *out_dynamic_stride_axis_count = 0;
  if (static_indices.kind != LOOM_ATTR_I64_ARRAY ||
      static_indices.count != vector_access->view_rank) {
    return false;
  }

  int64_t static_byte_offset = 0;
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
    const int64_t coordinate = static_indices.i64_array[i];
    if (coordinate == 0) continue;
    if (coordinate < 0) return false;
    loom_low_source_memory_axis_byte_stride_t axis_stride;
    loom_low_source_memory_query_axis_byte_stride(fact_table, vector_access,
                                                  (uint8_t)i, &axis_stride);
    if (axis_stride.kind == LOOM_LOW_SOURCE_MEMORY_AXIS_BYTE_STRIDE_STATIC) {
      if (!iree_checked_mul_add_i64(static_byte_offset, coordinate,
                                    axis_stride.static_byte_coefficient,
                                    &static_byte_offset)) {
        return false;
      }
      continue;
    }
    if (axis_stride.kind != LOOM_LOW_SOURCE_MEMORY_AXIS_BYTE_STRIDE_DYNAMIC ||
        *out_dynamic_stride_axis_count >= vector_access->view_rank) {
      return false;
    }
    out_dynamic_stride_axes[*out_dynamic_stride_axis_count] = (uint8_t)i;
    ++*out_dynamic_stride_axis_count;
  }
  *out_static_byte_offset = static_byte_offset;
  return true;
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

static bool loom_low_source_memory_access_explicit_stride_value(
    const loom_vector_memory_access_t* vector_access, uint8_t view_axis,
    loom_value_id_t* out_value) {
  *out_value = LOOM_VALUE_ID_INVALID;
  const loom_encoding_address_layout_operands_t* operands =
      &vector_access->layout_operands;
  if (operands->static_strides.kind != LOOM_ATTR_I64_ARRAY ||
      operands->static_strides.count != vector_access->view_rank ||
      view_axis >= operands->static_strides.count) {
    return false;
  }
  uint16_t dynamic_ordinal = 0;
  for (uint16_t axis = 0; axis < operands->static_strides.count; ++axis) {
    if (operands->static_strides.i64_array[axis] != INT64_MIN) continue;
    if (axis == view_axis) {
      if (dynamic_ordinal >= operands->dynamic_stride_count) return false;
      *out_value = operands->dynamic_stride_values[dynamic_ordinal];
      return true;
    }
    ++dynamic_ordinal;
  }
  return false;
}

void loom_low_source_memory_query_axis_byte_stride(
    const loom_value_fact_table_t* fact_table,
    const loom_vector_memory_access_t* vector_access, uint8_t view_axis,
    loom_low_source_memory_axis_byte_stride_t* out_stride) {
  *out_stride = (loom_low_source_memory_axis_byte_stride_t){
      .kind = LOOM_LOW_SOURCE_MEMORY_AXIS_BYTE_STRIDE_UNAVAILABLE,
      .byte_facts = loom_value_facts_unknown(),
      .static_byte_shift = LOOM_LOW_SOURCE_MEMORY_ACCESS_BYTE_SHIFT_NONE,
  };
  if (view_axis >= vector_access->view_rank ||
      vector_access->static_element_byte_count < 0) {
    return;
  }

  int64_t static_byte_coefficient = vector_access->static_element_byte_count;
  loom_value_facts_t byte_facts =
      loom_value_facts_exact_i64(static_byte_coefficient);
  if (vector_access->layout_kind == LOOM_VECTOR_MEMORY_LAYOUT_DENSE) {
    for (uint8_t axis = (uint8_t)(view_axis + 1);
         axis < vector_access->view_rank; ++axis) {
      loom_value_facts_t dimension_facts = loom_value_facts_unknown();
      if (loom_type_dim_is_dynamic_at(vector_access->view_type, axis)) {
        const loom_value_id_t dimension_value =
            loom_type_dim_value_id_at(vector_access->view_type, axis);
        dimension_facts = loom_value_facts_non_negative_extent(
            loom_value_fact_table_lookup(fact_table, dimension_value));
        int64_t exact_dimension = 0;
        if (loom_low_source_memory_access_exact_i64(dimension_facts,
                                                    &exact_dimension)) {
          if (!iree_checked_mul_i64(static_byte_coefficient, exact_dimension,
                                    &static_byte_coefficient)) {
            out_stride->kind = LOOM_LOW_SOURCE_MEMORY_AXIS_BYTE_STRIDE_INVALID;
            return;
          }
        } else {
          out_stride->dynamic_factors[out_stride->dynamic_factor_count++] =
              dimension_value;
        }
      } else {
        const int64_t dimension_size =
            loom_type_dim_static_size_at(vector_access->view_type, axis);
        dimension_facts = loom_value_facts_exact_i64(dimension_size);
        if (dimension_size < 0 ||
            !iree_checked_mul_i64(static_byte_coefficient, dimension_size,
                                  &static_byte_coefficient)) {
          out_stride->kind = LOOM_LOW_SOURCE_MEMORY_AXIS_BYTE_STRIDE_INVALID;
          return;
        }
      }
      loom_value_facts_muli(&byte_facts, &dimension_facts, &byte_facts);
    }
  } else if (vector_access->layout_kind == LOOM_VECTOR_MEMORY_LAYOUT_STRIDED) {
    const loom_value_fact_address_layout_t layout =
        vector_access->layout_summary;
    if (layout.kind != LOOM_VALUE_FACT_ADDRESS_LAYOUT_STRIDED ||
        view_axis >= layout.rank || !layout.strides) {
      return;
    }
    const loom_value_facts_t element_stride_facts = layout.strides[view_axis];
    if (loom_value_facts_is_float(element_stride_facts)) {
      out_stride->kind = LOOM_LOW_SOURCE_MEMORY_AXIS_BYTE_STRIDE_INVALID;
      return;
    }
    loom_value_facts_muli(&byte_facts, &element_stride_facts, &byte_facts);
    int64_t exact_element_stride = 0;
    if (loom_low_source_memory_access_exact_i64(element_stride_facts,
                                                &exact_element_stride)) {
      if (!iree_checked_mul_i64(static_byte_coefficient, exact_element_stride,
                                &static_byte_coefficient)) {
        out_stride->kind = LOOM_LOW_SOURCE_MEMORY_AXIS_BYTE_STRIDE_INVALID;
        return;
      }
    } else {
      loom_value_id_t stride_value = LOOM_VALUE_ID_INVALID;
      if (!loom_low_source_memory_access_explicit_stride_value(
              vector_access, view_axis, &stride_value)) {
        out_stride->kind =
            LOOM_LOW_SOURCE_MEMORY_AXIS_BYTE_STRIDE_UNMATERIALIZED;
        out_stride->byte_facts = byte_facts;
        return;
      }
      out_stride->dynamic_factors[out_stride->dynamic_factor_count++] =
          stride_value;
    }
  } else {
    return;
  }

  // A zero static factor makes the complete product exact even when a suffix
  // dimension is dynamic.
  if (static_byte_coefficient == 0) {
    out_stride->dynamic_factor_count = 0;
    byte_facts = loom_value_facts_exact_i64(0);
  }
  out_stride->kind = out_stride->dynamic_factor_count == 0
                         ? LOOM_LOW_SOURCE_MEMORY_AXIS_BYTE_STRIDE_STATIC
                         : LOOM_LOW_SOURCE_MEMORY_AXIS_BYTE_STRIDE_DYNAMIC;
  out_stride->static_byte_coefficient = static_byte_coefficient;
  out_stride->byte_facts = byte_facts;
  (void)loom_low_source_memory_access_power_of_two_shift(
      static_byte_coefficient, &out_stride->static_byte_shift);
}

static uint8_t loom_low_source_memory_access_axis_from_byte_stride(
    const loom_value_fact_table_t* fact_table,
    const loom_vector_memory_access_t* vector_access, int64_t byte_stride) {
  for (uint8_t axis = 0; axis < vector_access->view_rank; ++axis) {
    loom_low_source_memory_axis_byte_stride_t axis_stride;
    loom_low_source_memory_query_axis_byte_stride(fact_table, vector_access,
                                                  axis, &axis_stride);
    if (axis_stride.kind != LOOM_LOW_SOURCE_MEMORY_AXIS_BYTE_STRIDE_STATIC ||
        axis_stride.static_byte_coefficient != byte_stride) {
      continue;
    }
    return axis;
  }
  return LOOM_LOW_SOURCE_MEMORY_DYNAMIC_TERM_AXIS_NONE;
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

static void loom_low_source_memory_access_append_dynamic_realization(
    loom_low_source_memory_access_plan_t* plan,
    const loom_low_source_memory_dynamic_term_t* term, uint8_t first_term,
    uint8_t term_count) {
  IREE_ASSERT_GE(term_count, 2u);
  IREE_ASSERT_LT(plan->dynamic_realization_count,
                 LOOM_LOW_SOURCE_MEMORY_DYNAMIC_REALIZATION_CAPACITY);
  plan->dynamic_realizations[plan->dynamic_realization_count++] =
      (loom_low_source_memory_dynamic_realization_t){
          .term = *term,
          .first_term = first_term,
          .term_count = term_count,
      };
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
      iree_max(index_facts.range_lo, domain_facts.range_lo);
  const int64_t upper_bound =
      iree_min(index_facts.range_hi, domain_facts.range_hi);
  if (lower_bound > upper_bound) {
    return loom_value_facts_unknown();
  }
  return loom_value_facts_clamp_domain(index_facts, domain_facts.range_lo,
                                       domain_facts.range_hi);
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
  if (!iree_checked_sub_i64(inout_domain_facts->range_lo, offset,
                            &scaled_range_lo) ||
      !iree_checked_sub_i64(inout_domain_facts->range_hi, offset,
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

static void loom_low_source_memory_dynamic_term_compute_expression_byte_facts(
    const loom_value_fact_table_t* fact_table,
    const loom_vector_memory_access_t* vector_access,
    loom_value_facts_t expression_facts, uint8_t dynamic_axis,
    int64_t index_offset, int64_t byte_stride,
    const loom_value_id_t* stride_values, uint8_t stride_value_count,
    loom_value_facts_t* out_facts) {
  loom_value_facts_t index_facts = expression_facts;
  loom_value_facts_t domain_facts = loom_value_facts_unknown();
  if (dynamic_axis != LOOM_LOW_SOURCE_MEMORY_DYNAMIC_TERM_AXIS_NONE &&
      loom_low_source_memory_access_origin_domain_facts(
          fact_table, vector_access, dynamic_axis, &domain_facts)) {
    index_facts = loom_low_source_memory_access_intersect_index_facts(
        index_facts, domain_facts);
  }
  if (index_offset != 0) {
    const loom_value_facts_t offset_facts =
        loom_value_facts_exact_i64(index_offset);
    loom_value_facts_subi(&index_facts, &offset_facts, &index_facts);
  }
  loom_low_source_memory_dynamic_term_compute_index_byte_facts(
      fact_table, index_facts, byte_stride, stride_values, stride_value_count,
      out_facts);
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
  if (dynamic_axis != LOOM_LOW_SOURCE_MEMORY_DYNAMIC_TERM_AXIS_NONE &&
      loom_low_source_memory_access_origin_domain_facts(
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

static bool loom_low_source_memory_access_single_term_dynamic_facts(
    const loom_symbolic_expr_t* expression, loom_value_facts_t* out_facts) {
  *out_facts = loom_value_facts_unknown();
  if (expression->term_count != 1 ||
      loom_value_facts_is_unknown(expression->facts) ||
      loom_value_facts_is_float(expression->facts)) {
    return false;
  }

  loom_value_facts_t byte_facts = expression->facts;
  if (expression->constant != 0) {
    const loom_value_facts_t offset_facts =
        loom_value_facts_exact_i64(expression->constant);
    loom_value_facts_subi(&byte_facts, &offset_facts, &byte_facts);
  }

  *out_facts = byte_facts;
  return true;
}

static void loom_low_source_memory_access_refine_projection_term_byte_facts(
    const loom_value_fact_table_t* fact_table,
    const loom_vector_memory_access_t* vector_access,
    const loom_symbolic_term_t* expression_term, uint8_t dynamic_axis,
    loom_value_facts_t* inout_facts) {
  if (dynamic_axis == LOOM_LOW_SOURCE_MEMORY_DYNAMIC_TERM_AXIS_NONE) return;
  loom_value_facts_t domain_facts = loom_value_facts_unknown();
  if (!loom_low_source_memory_access_origin_domain_facts(
          fact_table, vector_access, dynamic_axis, &domain_facts)) {
    return;
  }
  loom_value_facts_t domain_byte_facts = loom_value_facts_unknown();
  loom_low_source_memory_dynamic_term_compute_index_byte_facts(
      fact_table, domain_facts, expression_term->coefficient,
      /*stride_values=*/NULL, /*stride_value_count=*/0, &domain_byte_facts);
  *inout_facts = loom_low_source_memory_access_intersect_index_facts(
      *inout_facts, domain_byte_facts);
}

loom_value_facts_t loom_low_source_memory_dynamic_offset_facts(
    const loom_low_source_memory_access_plan_t* plan,
    int64_t static_byte_offset) {
  loom_value_facts_t offset_facts =
      loom_value_facts_exact_i64(static_byte_offset);
  uint8_t term_ordinal = 0;
  if (plan->dynamic_view_base_term_count != 0 &&
      !loom_value_facts_is_unknown(plan->dynamic_view_base_byte_facts)) {
    loom_value_facts_addi(&offset_facts, &plan->dynamic_view_base_byte_facts,
                          &offset_facts);
    term_ordinal = plan->dynamic_view_base_term_count;
  }
  uint8_t realization_ordinal = 0;
  while (term_ordinal < plan->dynamic_term_count) {
    const loom_low_source_memory_dynamic_realization_t* realization = NULL;
    if (realization_ordinal < plan->dynamic_realization_count &&
        plan->dynamic_realizations[realization_ordinal].first_term ==
            term_ordinal) {
      realization = &plan->dynamic_realizations[realization_ordinal++];
    }
    if (realization != NULL) {
      loom_value_facts_addi(&offset_facts, &realization->term.byte_facts,
                            &offset_facts);
      term_ordinal = (uint8_t)(term_ordinal + realization->term_count);
    } else {
      loom_value_facts_addi(&offset_facts,
                            &plan->dynamic_terms[term_ordinal].byte_facts,
                            &offset_facts);
      ++term_ordinal;
    }
  }
  return offset_facts;
}

bool loom_low_source_memory_dynamic_offset_fits_unsigned_bit_count(
    const loom_low_source_memory_access_plan_t* plan,
    int64_t static_byte_offset, uint8_t bit_count) {
  return loom_value_facts_fit_unsigned_bit_count(
      loom_low_source_memory_dynamic_offset_facts(plan, static_byte_offset),
      bit_count);
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

static bool loom_low_source_memory_access_can_extract_static_index_offset(
    int64_t index_offset) {
  // Static byte offsets are modeled as non-negative target-friendly addends.
  // Keep negative terms in SSA so the derived index carries its range facts.
  return index_offset >= 0;
}

static bool loom_low_source_memory_access_apply_static_index_offset(
    int64_t index_offset, int64_t byte_stride, int64_t static_byte_offset,
    int64_t* out_static_byte_offset) {
  *out_static_byte_offset = static_byte_offset;

  int64_t static_offset_delta = 0;
  int64_t new_static_byte_offset = 0;
  if (!iree_checked_mul_i64(byte_stride, index_offset, &static_offset_delta) ||
      !iree_checked_add_i64(static_byte_offset, static_offset_delta,
                            &new_static_byte_offset)) {
    return false;
  }
  // Leave the original dynamic expression intact if extracting its affine
  // offset would move a non-negative static byte offset outside that domain.
  if (static_byte_offset >= 0 && new_static_byte_offset < 0) {
    return false;
  }
  *out_static_byte_offset = new_static_byte_offset;
  return true;
}

static bool loom_low_source_memory_access_can_expand_index_expression(
    const loom_symbolic_expr_t* expression,
    iree_host_size_t available_term_count) {
  if (!loom_symbolic_expr_is_linear(expression) ||
      expression->term_count <= 1 ||
      expression->term_count > available_term_count ||
      !loom_low_source_memory_access_can_extract_static_index_offset(
          expression->constant)) {
    return false;
  }
  for (iree_host_size_t i = 0; i < expression->term_count; ++i) {
    if (expression->terms[i].coefficient <= 0) return false;
  }
  return true;
}

static bool loom_low_source_memory_facts_are_stronger(
    loom_value_facts_t candidate, loom_value_facts_t baseline) {
  if (loom_value_facts_is_float(candidate) !=
      loom_value_facts_is_float(baseline)) {
    return false;
  }
  const bool range_no_weaker = candidate.range_lo >= baseline.range_lo &&
                               candidate.range_hi <= baseline.range_hi;
  const bool range_stronger = candidate.range_lo > baseline.range_lo ||
                              candidate.range_hi < baseline.range_hi;
  const bool divisor_no_weaker =
      baseline.known_divisor <= 1 ||
      (candidate.known_divisor >= baseline.known_divisor &&
       (candidate.known_divisor % baseline.known_divisor) == 0);
  const bool divisor_stronger =
      divisor_no_weaker && candidate.known_divisor > baseline.known_divisor;
  return range_no_weaker && divisor_no_weaker &&
         (range_stronger || divisor_stronger);
}

static loom_value_id_t loom_low_source_memory_symbolic_term_materialized_value(
    const loom_value_fact_table_t* fact_table,
    const loom_symbolic_term_t* expression_term) {
  if (expression_term->relation_value_id == LOOM_VALUE_ID_INVALID ||
      expression_term->relation_value_id == expression_term->value_id ||
      !loom_value_fact_table_has_entry(fact_table,
                                       expression_term->relation_value_id) ||
      !loom_value_fact_table_has_entry(fact_table, expression_term->value_id)) {
    return expression_term->value_id;
  }
  return loom_low_source_memory_facts_are_stronger(
             loom_value_fact_table_lookup(fact_table,
                                          expression_term->relation_value_id),
             loom_value_fact_table_lookup(fact_table,
                                          expression_term->value_id))
             ? expression_term->relation_value_id
             : expression_term->value_id;
}

typedef enum loom_low_source_memory_view_expression_role_e {
  // Root-relative byte terms have no logical view axis.
  LOOM_LOW_SOURCE_MEMORY_VIEW_EXPRESSION_ROOT_BASE = 0,
  // Projection-relative byte terms may inherit a logical view axis.
  LOOM_LOW_SOURCE_MEMORY_VIEW_EXPRESSION_PROJECTION = 1,
} loom_low_source_memory_view_expression_role_t;

// Appends canonical terms for one analyzed view expression. When provided,
// |inout_combined_byte_facts| accumulates independent term facts during the
// append so aggregate refinement requires no later term traversal.
static bool loom_low_source_memory_access_add_view_region_expression_terms(
    const loom_value_fact_table_t* fact_table,
    const loom_vector_memory_access_t* vector_access,
    loom_low_source_memory_view_expression_role_t expression_role,
    const loom_symbolic_expr_t* expression,
    loom_low_source_memory_access_plan_t* plan,
    loom_low_source_memory_access_diagnostic_t* diagnostic,
    loom_value_facts_t* inout_combined_byte_facts) {
  if (!loom_symbolic_expr_is_linear(expression)) {
    diagnostic->rejection_bits |=
        LOOM_LOW_SOURCE_MEMORY_ACCESS_REJECTION_VIEW_BASE;
    return false;
  }

  for (iree_host_size_t i = 0; i < expression->term_count; ++i) {
    const loom_symbolic_term_t* expression_term = &expression->terms[i];
    if (expression_term->coefficient <= 0) {
      diagnostic->rejection_bits |=
          LOOM_LOW_SOURCE_MEMORY_ACCESS_REJECTION_VIEW_BASE;
      return false;
    }
    loom_value_id_t materialized_value =
        loom_low_source_memory_symbolic_term_materialized_value(
            fact_table, expression_term);
    const uint8_t dynamic_axis =
        expression_role == LOOM_LOW_SOURCE_MEMORY_VIEW_EXPRESSION_PROJECTION
            ? loom_low_source_memory_access_axis_from_byte_stride(
                  fact_table, vector_access, expression_term->coefficient)
            : LOOM_LOW_SOURCE_MEMORY_DYNAMIC_TERM_AXIS_NONE;
    loom_low_source_memory_dynamic_term_t term = {
        .index = materialized_value,
        .axis = dynamic_axis,
        .byte_stride = expression_term->coefficient,
        .byte_shift = LOOM_LOW_SOURCE_MEMORY_ACCESS_BYTE_SHIFT_NONE,
    };
    loom_low_source_memory_access_dynamic_index_source(
        fact_table, materialized_value, &term.source, &term.dimension);
    (void)loom_low_source_memory_access_power_of_two_shift(term.byte_stride,
                                                           &term.byte_shift);
    loom_value_facts_t byte_facts = loom_value_facts_unknown();
    if (loom_low_source_memory_access_single_term_dynamic_facts(expression,
                                                                &byte_facts)) {
      if (expression_role ==
          LOOM_LOW_SOURCE_MEMORY_VIEW_EXPRESSION_PROJECTION) {
        loom_low_source_memory_access_refine_projection_term_byte_facts(
            fact_table, vector_access, expression_term, term.axis, &byte_facts);
      }
    } else if (expression_role ==
               LOOM_LOW_SOURCE_MEMORY_VIEW_EXPRESSION_PROJECTION) {
      loom_low_source_memory_dynamic_term_compute_byte_facts(
          fact_table, vector_access, materialized_value, term.axis,
          /*index_multiplier=*/1, /*index_offset=*/0, term.byte_stride,
          /*stride_values=*/NULL, /*stride_value_count=*/0, &byte_facts);
    } else {
      loom_low_source_memory_dynamic_term_compute_scaled_byte_facts(
          fact_table, materialized_value, term.byte_stride,
          /*stride_values=*/NULL, /*stride_value_count=*/0, &byte_facts);
    }
    term.byte_facts = byte_facts;
    if (!loom_low_source_memory_access_append_dynamic_term(plan, &term,
                                                           diagnostic)) {
      return false;
    }
    if (inout_combined_byte_facts != NULL) {
      loom_value_facts_addi(inout_combined_byte_facts, &term.byte_facts,
                            inout_combined_byte_facts);
    }
  }
  return true;
}

static bool loom_low_source_memory_access_add_view_region_byte_offset(
    const loom_value_fact_table_t* fact_table,
    const loom_vector_memory_access_t* vector_access,
    const loom_view_region_t* view_region,
    loom_low_source_memory_access_plan_t* plan,
    loom_low_source_memory_access_diagnostic_t* diagnostic,
    int64_t* inout_static_byte_offset) {
  if (!loom_symbolic_expr_is_linear(&view_region->begin_byte_offset)) {
    diagnostic->rejection_bits |=
        LOOM_LOW_SOURCE_MEMORY_ACCESS_REJECTION_VIEW_BASE;
    return false;
  }
  int64_t static_byte_offset = 0;
  if (!iree_checked_add_i64(*inout_static_byte_offset,
                            view_region->begin_byte_offset.constant,
                            &static_byte_offset)) {
    diagnostic->rejection_bits |=
        LOOM_LOW_SOURCE_MEMORY_ACCESS_REJECTION_VIEW_BASE_OVERFLOW;
    return false;
  }
  plan->static_view_base_byte_offset = view_region->begin_byte_offset.constant;
  *inout_static_byte_offset = static_byte_offset;

  const uint8_t dynamic_view_base_begin = plan->dynamic_term_count;
  // Multi-term bases can carry relational bounds in the analyzed complete
  // expression that independent canonical terms cannot represent. Accumulate
  // canonical facts during construction so taking the tighter range requires
  // neither source rediscovery nor another term traversal.
  const bool can_refine_dynamic_view_base_facts =
      view_region->base_begin_byte_offset.term_count +
              view_region->projection_byte_offset.term_count >=
          2 &&
      !loom_value_facts_is_unknown(view_region->begin_byte_offset.facts);
  loom_value_facts_t canonical_view_base_byte_facts;
  loom_value_facts_t* combined_view_base_byte_facts = NULL;
  if (can_refine_dynamic_view_base_facts) {
    canonical_view_base_byte_facts = loom_value_facts_exact_i64(0);
    combined_view_base_byte_facts = &canonical_view_base_byte_facts;
  }
  if (!loom_low_source_memory_access_add_view_region_expression_terms(
          fact_table, vector_access,
          LOOM_LOW_SOURCE_MEMORY_VIEW_EXPRESSION_ROOT_BASE,
          &view_region->base_begin_byte_offset, plan, diagnostic,
          combined_view_base_byte_facts) ||
      !loom_low_source_memory_access_add_view_region_expression_terms(
          fact_table, vector_access,
          LOOM_LOW_SOURCE_MEMORY_VIEW_EXPRESSION_PROJECTION,
          &view_region->projection_byte_offset, plan, diagnostic,
          combined_view_base_byte_facts)) {
    return false;
  }
  const uint8_t dynamic_view_base_count =
      (uint8_t)(plan->dynamic_term_count - dynamic_view_base_begin);
  plan->dynamic_view_base_term_count = dynamic_view_base_count;
  if (view_region->base_view_value_id != LOOM_VALUE_ID_INVALID) {
    plan->base_view_value_id = view_region->base_view_value_id;
  }
  if (can_refine_dynamic_view_base_facts) {
    // The canonical static contribution is already represented separately in
    // the access plan, so retain only the analyzed dynamic contribution.
    loom_value_facts_t analyzed_view_base_byte_facts =
        view_region->begin_byte_offset.facts;
    const loom_value_facts_t static_view_base_byte_facts =
        loom_value_facts_exact_i64(plan->static_view_base_byte_offset);
    loom_value_facts_subi(&analyzed_view_base_byte_facts,
                          &static_view_base_byte_facts,
                          &analyzed_view_base_byte_facts);
    const loom_value_facts_t refined_view_base_byte_facts =
        loom_low_source_memory_access_intersect_index_facts(
            canonical_view_base_byte_facts, analyzed_view_base_byte_facts);
    if (loom_low_source_memory_facts_are_stronger(
            refined_view_base_byte_facts, canonical_view_base_byte_facts)) {
      plan->dynamic_view_base_byte_facts = refined_view_base_byte_facts;
    }
  }
  if (dynamic_view_base_count != 0 &&
      view_region->begin_value_id != LOOM_VALUE_ID_INVALID) {
    plan->dynamic_view_base_value_id = view_region->begin_value_id;
    plan->dynamic_view_base_value_static_byte_offset =
        plan->static_view_base_byte_offset;
  } else if (dynamic_view_base_count == 1 &&
             plan->dynamic_terms[dynamic_view_base_begin].byte_stride == 1) {
    plan->dynamic_view_base_value_id =
        plan->dynamic_terms[dynamic_view_base_begin].index;
  }
  return true;
}

static bool loom_low_source_memory_access_add_view_base_byte_offset(
    const loom_view_region_table_t* view_regions,
    const loom_vector_memory_access_t* vector_access,
    loom_value_id_t view_value_id, loom_low_source_memory_access_plan_t* plan,
    loom_low_source_memory_access_diagnostic_t* diagnostic,
    int64_t* inout_static_byte_offset) {
  const loom_module_t* module = view_regions->expression_context->module;
  const loom_value_fact_table_t* fact_table =
      view_regions->expression_context->fact_table;
  const loom_view_region_t* view_region = NULL;
  if (!loom_view_region_table_try_lookup(view_regions, view_value_id,
                                         &view_region)) {
    diagnostic->rejection_bits |=
        LOOM_LOW_SOURCE_MEMORY_ACCESS_REJECTION_VIEW_SOURCE;
    return false;
  }

  loom_vector_memory_access_t base_vector_access = {0};
  const loom_vector_memory_access_t* view_base_access = vector_access;
  if (view_region->base_view_value_id != LOOM_VALUE_ID_INVALID) {
    const loom_type_t base_view_type =
        loom_module_value_type(module, view_region->base_view_value_id);
    if (!loom_vector_memory_access_describe(
            &fact_table->context, module, base_view_type,
            vector_access->vector_type, &base_vector_access)) {
      diagnostic->rejection_bits |=
          LOOM_LOW_SOURCE_MEMORY_ACCESS_REJECTION_VIEW_BASE;
      return false;
    }
    view_base_access = &base_vector_access;
  }
  if (!loom_low_source_memory_access_add_view_region_byte_offset(
          fact_table, view_base_access, view_region, plan, diagnostic,
          inout_static_byte_offset)) {
    return false;
  }

  plan->memory_space = view_region->memory_space;
  plan->root_value_id = view_region->root_value_id;
  plan->root_minimum_alignment = loom_low_source_memory_clamp_alignment(
      view_region->root_minimum_alignment);
  plan->alias_scope_id = view_region->alias_scope_id;
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

bool loom_low_source_memory_access_plan_lane_byte_envelope(
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
  int64_t begin_offset = iree_min(0, last_lane_offset);
  int64_t end_offset = 0;
  if (!iree_checked_add_i64(iree_max(0, last_lane_offset),
                            (int64_t)plan->element_byte_count, &end_offset)) {
    return false;
  }
  *out_begin_offset = begin_offset;
  *out_end_offset = end_offset;
  return true;
}

static bool loom_low_source_memory_access_plan_strided_interval(
    const loom_low_source_memory_access_plan_t* plan, int64_t lane_begin_offset,
    int64_t lane_end_offset, loom_low_strided_byte_interval_t* out_interval) {
  *out_interval = (loom_low_strided_byte_interval_t){0};
  if (plan->dynamic_term_count == 0) {
    return false;
  }
  uint64_t stride_bytes = 0;
  for (uint8_t i = 0; i < plan->dynamic_term_count; ++i) {
    const loom_low_source_memory_dynamic_term_t* term = &plan->dynamic_terms[i];
    const int64_t signed_stride = term->byte_stride;
    if (term->stride_value_count != 0 || signed_stride == 0 ||
        signed_stride == INT64_MIN) {
      return false;
    }
    const uint64_t term_stride_bytes =
        (uint64_t)(signed_stride < 0 ? -signed_stride : signed_stride);
    stride_bytes = stride_bytes == 0
                       ? term_stride_bytes
                       : iree_math_gcd_u64(stride_bytes, term_stride_bytes);
  }
  int64_t access_begin = 0;
  int64_t access_end = 0;
  if (!iree_checked_add_i64(plan->static_byte_offset, lane_begin_offset,
                            &access_begin) ||
      !iree_checked_add_i64(plan->static_byte_offset, lane_end_offset,
                            &access_end) ||
      access_end <= access_begin) {
    return false;
  }
  int64_t signed_length_bytes = 0;
  if (!iree_checked_sub_i64(access_end, access_begin, &signed_length_bytes) ||
      signed_length_bytes <= 0) {
    return false;
  }
  const uint64_t length_bytes = (uint64_t)signed_length_bytes;
  if (length_bytes > stride_bytes) {
    return false;
  }
  int64_t signed_begin_residue = access_begin % (int64_t)stride_bytes;
  if (signed_begin_residue < 0) {
    signed_begin_residue += (int64_t)stride_bytes;
  }
  const uint64_t begin_bytes = (uint64_t)signed_begin_residue;
  if (begin_bytes > stride_bytes - length_bytes) {
    return false;
  }
  *out_interval = (loom_low_strided_byte_interval_t){
      .stride_bytes = stride_bytes,
      .begin_bytes = begin_bytes,
      .end_bytes = begin_bytes + length_bytes,
  };
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
  loom_low_strided_byte_interval_t strided_interval = {0};
  int64_t lane_begin_offset = 0;
  int64_t lane_end_offset = 0;
  if (loom_low_source_memory_access_plan_lane_byte_envelope(
          plan, &lane_begin_offset, &lane_end_offset)) {
    loom_value_facts_t begin_facts =
        loom_low_source_memory_dynamic_offset_facts(plan,
                                                    plan->static_byte_offset);
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
    if (iree_any_bit_set(precision_flags,
                         LOOM_LOW_MEMORY_ACCESS_PRECISION_ROOT) &&
        loom_low_source_memory_access_plan_strided_interval(
            plan, lane_begin_offset, lane_end_offset, &strided_interval)) {
      precision_flags |= LOOM_LOW_MEMORY_ACCESS_PRECISION_STRIDED_INTERVAL;
    }
  }

  *out_summary = (loom_low_memory_access_summary_t){
      .memory_space = memory_space,
      .alias_root_id = alias_root_id,
      .alias_group_id = LOOM_LOW_MEMORY_ALIAS_ID_NONE,
      .precision_flags = precision_flags,
      .strided_interval = strided_interval,
      .byte_interval = interval,
  };
}

bool loom_low_source_memory_operation_kind_from_access(
    loom_memory_access_t access,
    loom_low_source_memory_operation_kind_t* out_operation_kind) {
  *out_operation_kind = (loom_low_source_memory_operation_kind_t)
      loom_memory_access_operation_kind(access);
  return *out_operation_kind != LOOM_MEMORY_ACCESS_OPERATION_COUNT_;
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

static loom_low_source_memory_address_layout_t
loom_low_source_memory_classify_address_layout(
    const loom_vector_memory_access_t* vector_access) {
  if (vector_access->layout_kind == LOOM_VECTOR_MEMORY_LAYOUT_DENSE) {
    return LOOM_LOW_SOURCE_MEMORY_ADDRESS_LAYOUT_COMPACT_ROW_MAJOR;
  }
  if (vector_access->layout_kind != LOOM_VECTOR_MEMORY_LAYOUT_STRIDED) {
    return LOOM_LOW_SOURCE_MEMORY_ADDRESS_LAYOUT_UNPROVEN;
  }

  int64_t expected_stride = 1;
  for (int16_t axis = (int16_t)vector_access->view_rank - 1; axis >= 0;
       --axis) {
    int64_t actual_stride = 0;
    if (!loom_vector_memory_access_static_axis_stride(
            vector_access, (uint8_t)axis, &actual_stride) ||
        actual_stride != expected_stride) {
      return LOOM_LOW_SOURCE_MEMORY_ADDRESS_LAYOUT_UNPROVEN;
    }
    if (axis != 0) {
      if (loom_type_dim_is_dynamic_at(vector_access->view_type,
                                      (uint8_t)axis) ||
          !iree_checked_mul_i64(expected_stride,
                                loom_type_dim_static_size_at(
                                    vector_access->view_type, (uint8_t)axis),
                                &expected_stride)) {
        return LOOM_LOW_SOURCE_MEMORY_ADDRESS_LAYOUT_UNPROVEN;
      }
    }
  }
  return LOOM_LOW_SOURCE_MEMORY_ADDRESS_LAYOUT_COMPACT_ROW_MAJOR;
}

static bool loom_low_source_memory_access_plan_from_components(
    const loom_view_region_table_t* view_regions,
    loom_low_source_memory_operation_kind_t operation_kind,
    loom_value_id_t view_value_id, loom_value_slice_t dynamic_indices,
    loom_attribute_t static_indices, loom_type_t view_type,
    loom_type_t vector_type, loom_vector_memory_cache_policy_t cache_policy,
    loom_low_source_memory_access_plan_t* out_plan,
    loom_low_source_memory_access_diagnostic_t* out_diagnostic) {
  out_plan->operation_kind = operation_kind;
  out_plan->view_value_id = view_value_id;
  out_plan->base_view_value_id = view_value_id;
  out_plan->root_value_id = LOOM_VALUE_ID_INVALID;
  out_plan->root_minimum_alignment = 1;
  out_plan->alias_scope_id = LOOM_VALUE_FACT_ALIAS_SCOPE_ID_NONE;
  out_plan->dynamic_view_base_value_id = LOOM_VALUE_ID_INVALID;
  out_plan->dynamic_view_base_byte_facts = loom_value_facts_unknown();
  out_plan->minimum_alignment = 1;
  out_plan->cache_policy = cache_policy;
  const loom_module_t* module = view_regions->expression_context->module;
  const loom_value_fact_table_t* fact_table =
      view_regions->expression_context->fact_table;

  loom_vector_memory_access_t vector_access;
  if (!loom_vector_memory_access_describe(&fact_table->context, module,
                                          view_type, vector_type,
                                          &vector_access)) {
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
  out_plan->address_layout =
      loom_low_source_memory_classify_address_layout(&vector_access);
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

  loom_low_source_memory_axis_byte_stride_t vector_axis_stride;
  loom_low_source_memory_query_axis_byte_stride(fact_table, &vector_access,
                                                vector_access.first_vector_axis,
                                                &vector_axis_stride);
  if (vector_axis_stride.kind ==
      LOOM_LOW_SOURCE_MEMORY_AXIS_BYTE_STRIDE_STATIC) {
    out_plan->vector_lane_byte_stride =
        vector_axis_stride.static_byte_coefficient;
  } else if (out_plan->vector_lane_count == 1) {
    // A one-lane access has no adjacent-lane address delta to materialize.
    out_plan->vector_lane_byte_stride = 0;
  } else {
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

  uint8_t dynamic_stride_axes[LOOM_ENCODING_ADDRESS_LAYOUT_MAX_RANK] = {0};
  uint8_t dynamic_stride_axis_count = 0;
  int64_t static_byte_offset = 0;
  if (!loom_low_source_memory_access_static_byte_offset(
          fact_table, &vector_access, static_indices, dynamic_axes,
          dynamic_axis_count, dynamic_stride_axes, &dynamic_stride_axis_count,
          &static_byte_offset)) {
    out_diagnostic->rejection_bits |=
        LOOM_LOW_SOURCE_MEMORY_ACCESS_REJECTION_STATIC_OFFSET;
    return false;
  }
  if (!loom_low_source_memory_access_add_view_base_byte_offset(
          view_regions, &vector_access, view_value_id, out_plan, out_diagnostic,
          &static_byte_offset)) {
    return false;
  }

  // A nonzero static coordinate crossing a runtime physical stride is still a
  // dynamic byte-address term. Use the first stride factor as the term source
  // and retain the remaining factors on its byte stride.
  for (uint8_t i = 0; i < dynamic_stride_axis_count; ++i) {
    const uint8_t axis = dynamic_stride_axes[i];
    const int64_t coordinate = static_indices.i64_array[axis];
    loom_low_source_memory_axis_byte_stride_t axis_stride;
    loom_low_source_memory_query_axis_byte_stride(fact_table, &vector_access,
                                                  axis, &axis_stride);
    IREE_ASSERT_GT(axis_stride.dynamic_factor_count, 0u);
    int64_t byte_stride = 0;
    if (!iree_checked_mul_i64(axis_stride.static_byte_coefficient, coordinate,
                              &byte_stride)) {
      out_diagnostic->rejection_bits |=
          LOOM_LOW_SOURCE_MEMORY_ACCESS_REJECTION_DYNAMIC_STRIDE;
      return false;
    }
    loom_low_source_memory_dynamic_term_t term = {
        .index = axis_stride.dynamic_factors[0],
        .source = LOOM_LOW_SOURCE_MEMORY_DYNAMIC_INDEX_SOURCE_VALUE,
        .dimension = LOOM_KERNEL_DIMENSION_COUNT_,
        .axis = LOOM_LOW_SOURCE_MEMORY_DYNAMIC_TERM_AXIS_NONE,
        .byte_stride = byte_stride,
        .byte_facts = loom_value_facts_unknown(),
        .byte_shift = LOOM_LOW_SOURCE_MEMORY_ACCESS_BYTE_SHIFT_NONE,
        .stride_value_count = (uint8_t)(axis_stride.dynamic_factor_count - 1),
    };
    loom_low_source_memory_access_dynamic_index_source(
        fact_table, term.index, &term.source, &term.dimension);
    for (uint8_t factor = 1; factor < axis_stride.dynamic_factor_count;
         ++factor) {
      term.stride_values[factor - 1] = axis_stride.dynamic_factors[factor];
    }
    (void)loom_low_source_memory_access_power_of_two_shift(term.byte_stride,
                                                           &term.byte_shift);
    loom_low_source_memory_dynamic_term_compute_scaled_byte_facts(
        fact_table, term.index, term.byte_stride, term.stride_values,
        term.stride_value_count, &term.byte_facts);
    if (!loom_low_source_memory_access_append_dynamic_term(out_plan, &term,
                                                           out_diagnostic)) {
      return false;
    }
  }

  for (uint8_t i = 0; i < dynamic_axis_count; ++i) {
    const uint8_t dynamic_axis = dynamic_axes[i];
    if (dynamic_axis >= vector_access.view_rank) {
      out_diagnostic->rejection_bits |=
          LOOM_LOW_SOURCE_MEMORY_ACCESS_REJECTION_DYNAMIC_AXIS;
      return false;
    }

    loom_low_source_memory_axis_byte_stride_t axis_stride;
    loom_low_source_memory_query_axis_byte_stride(fact_table, &vector_access,
                                                  dynamic_axis, &axis_stride);
    if (axis_stride.kind != LOOM_LOW_SOURCE_MEMORY_AXIS_BYTE_STRIDE_STATIC &&
        axis_stride.kind != LOOM_LOW_SOURCE_MEMORY_AXIS_BYTE_STRIDE_DYNAMIC) {
      out_diagnostic->rejection_bits |=
          LOOM_LOW_SOURCE_MEMORY_ACCESS_REJECTION_DYNAMIC_STRIDE;
      return false;
    }
    int64_t byte_stride = axis_stride.static_byte_coefficient;
    loom_value_id_t
        stride_values[LOOM_LOW_SOURCE_MEMORY_DYNAMIC_TERM_CAPACITY] = {0};
    const uint8_t stride_value_count = axis_stride.dynamic_factor_count;
    for (uint8_t stride_ordinal = 0; stride_ordinal < stride_value_count;
         ++stride_ordinal) {
      stride_values[stride_ordinal] =
          axis_stride.dynamic_factors[stride_ordinal];
    }

    const loom_value_id_t source_index = dynamic_index_values[i];
    loom_symbolic_term_t source_term = {
        .coefficient = 1,
        .value_id = source_index,
        .relation_value_id = source_index,
    };
    loom_symbolic_expr_summary_t index_summary = {
        .expression =
            {
                .constant = 0,
                .terms = &source_term,
                .term_count = 1,
                .facts = loom_value_fact_table_lookup(fact_table, source_index),
                .flags = LOOM_SYMBOLIC_EXPR_FLAG_LINEAR,
            },
        .materialized_dynamic_value_id = source_index,
    };
    loom_symbolic_expr_summary_t analyzed_summary = {0};
    if (loom_symbolic_expr_context_try_lookup_summary(
            view_regions->expression_context, source_index,
            &analyzed_summary)) {
      index_summary = analyzed_summary;
    }

    const loom_symbolic_expr_t* index_expression = &index_summary.expression;
    loom_value_id_t dynamic_index = source_index;
    int64_t dynamic_index_multiplier = 1;
    int64_t dynamic_index_offset = 0;
    const int64_t expression_byte_stride = byte_stride;
    loom_value_facts_t expression_facts =
        loom_value_fact_table_lookup(fact_table, source_index);

    // A coordinate suffix cannot become a static byte offset when its axis
    // stride contains a dynamic extent: the suffix is multiplied by that
    // extent and remains part of the dynamic address. A multiplier-only
    // decomposition remains safe because every dynamic stride value stays on
    // the term.
    const int64_t unscaled_static_byte_offset = static_byte_offset;
    if (loom_symbolic_expr_is_linear(index_expression) &&
        index_expression->term_count == 1 &&
        index_expression->terms[0].coefficient > 0 &&
        (index_expression->constant != 0 ||
         index_expression->terms[0].coefficient != 1) &&
        loom_low_source_memory_access_can_extract_static_index_offset(
            index_expression->constant) &&
        (index_expression->constant == 0 || stride_value_count == 0)) {
      int64_t scaled_byte_stride = 0;
      int64_t scaled_static_byte_offset = static_byte_offset;
      if (iree_checked_mul_i64(byte_stride,
                               index_expression->terms[0].coefficient,
                               &scaled_byte_stride) &&
          loom_low_source_memory_access_apply_static_index_offset(
              index_expression->constant, byte_stride, static_byte_offset,
              &scaled_static_byte_offset)) {
        dynamic_index = loom_low_source_memory_symbolic_term_materialized_value(
            fact_table, &index_expression->terms[0]);
        dynamic_index_multiplier = index_expression->terms[0].coefficient;
        dynamic_index_offset = index_expression->constant;
        expression_facts = index_expression->facts;
        byte_stride = scaled_byte_stride;
        static_byte_offset = scaled_static_byte_offset;
      }
      if (static_byte_offset != unscaled_static_byte_offset) {
        out_plan->source_index_static_offset_extracted = true;
      }
    }

    const iree_host_size_t available_term_count =
        LOOM_LOW_SOURCE_MEMORY_DYNAMIC_TERM_CAPACITY -
        out_plan->dynamic_term_count;
    bool expression_expanded =
        loom_low_source_memory_access_can_expand_index_expression(
            index_expression, available_term_count) &&
        (index_expression->constant == 0 || stride_value_count == 0);
    int64_t expanded_static_byte_offset = unscaled_static_byte_offset;
    if (expression_expanded) {
      if (!loom_low_source_memory_access_apply_static_index_offset(
              index_expression->constant, expression_byte_stride,
              unscaled_static_byte_offset, &expanded_static_byte_offset)) {
        expression_expanded = false;
      }
    }
    if (expression_expanded) {
      const uint8_t first_expression_term = out_plan->dynamic_term_count;
      if (expanded_static_byte_offset != unscaled_static_byte_offset) {
        out_plan->source_index_static_offset_extracted = true;
      }
      static_byte_offset = expanded_static_byte_offset;
      for (iree_host_size_t term_ordinal = 0;
           term_ordinal < index_expression->term_count; ++term_ordinal) {
        const loom_symbolic_term_t* expression_term =
            &index_expression->terms[term_ordinal];
        const loom_value_id_t materialized_value =
            loom_low_source_memory_symbolic_term_materialized_value(
                fact_table, expression_term);
        int64_t term_byte_stride = 0;
        if (!iree_checked_mul_i64(expression_byte_stride,
                                  expression_term->coefficient,
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
            fact_table, materialized_value, term_byte_stride, stride_values,
            stride_value_count, &byte_facts);
        loom_low_source_memory_dynamic_term_t term = {
            .index = materialized_value,
            .axis = LOOM_LOW_SOURCE_MEMORY_DYNAMIC_TERM_AXIS_NONE,
            .byte_stride = term_byte_stride,
            .byte_facts = byte_facts,
            .byte_shift = byte_shift,
            .stride_value_count = stride_value_count,
        };
        loom_low_source_memory_access_dynamic_index_source(
            fact_table, materialized_value, &term.source, &term.dimension);
        for (uint8_t stride_ordinal = 0; stride_ordinal < stride_value_count;
             ++stride_ordinal) {
          term.stride_values[stride_ordinal] = stride_values[stride_ordinal];
        }
        if (!loom_low_source_memory_access_append_dynamic_term(
                out_plan, &term, out_diagnostic)) {
          return false;
        }
      }
      // Preserve the analyzed SSA value that exactly materializes the
      // canonical dynamic terms. This may be the source index itself or a
      // constant-free prefix retained while symbolic analysis folded a static
      // suffix.
      if (index_summary.materialized_dynamic_value_id !=
          LOOM_VALUE_ID_INVALID) {
        const loom_value_id_t realization_value =
            index_summary.materialized_dynamic_value_id;
        uint32_t byte_shift = LOOM_LOW_SOURCE_MEMORY_ACCESS_BYTE_SHIFT_NONE;
        (void)loom_low_source_memory_access_power_of_two_shift(
            expression_byte_stride, &byte_shift);
        loom_value_facts_t canonical_byte_facts = loom_value_facts_exact_i64(0);
        for (uint8_t term_ordinal = first_expression_term;
             term_ordinal < out_plan->dynamic_term_count; ++term_ordinal) {
          loom_value_facts_addi(
              &canonical_byte_facts,
              &out_plan->dynamic_terms[term_ordinal].byte_facts,
              &canonical_byte_facts);
        }
        loom_value_facts_t expression_byte_facts = loom_value_facts_unknown();
        loom_low_source_memory_dynamic_term_compute_expression_byte_facts(
            fact_table, &vector_access, index_expression->facts, dynamic_axis,
            index_expression->constant, expression_byte_stride, stride_values,
            stride_value_count, &expression_byte_facts);
        loom_value_facts_t byte_facts =
            loom_low_source_memory_access_intersect_index_facts(
                canonical_byte_facts, expression_byte_facts);
        loom_value_facts_t realization_byte_facts = loom_value_facts_unknown();
        loom_low_source_memory_dynamic_term_compute_scaled_byte_facts(
            fact_table, realization_value, expression_byte_stride,
            stride_values, stride_value_count, &realization_byte_facts);
        byte_facts = loom_low_source_memory_access_intersect_index_facts(
            byte_facts, realization_byte_facts);
        loom_low_source_memory_dynamic_term_t realization_term = {
            .index = realization_value,
            .axis = dynamic_axis,
            .byte_stride = expression_byte_stride,
            .byte_facts = byte_facts,
            .byte_shift = byte_shift,
            .stride_value_count = stride_value_count,
        };
        loom_low_source_memory_access_dynamic_index_source(
            fact_table, realization_value, &realization_term.source,
            &realization_term.dimension);
        for (uint8_t stride_ordinal = 0; stride_ordinal < stride_value_count;
             ++stride_ordinal) {
          realization_term.stride_values[stride_ordinal] =
              stride_values[stride_ordinal];
        }
        loom_low_source_memory_access_append_dynamic_realization(
            out_plan, &realization_term, first_expression_term,
            (uint8_t)(out_plan->dynamic_term_count - first_expression_term));
      }
      continue;
    }

    // Keep the exact source coordinate in the access plan when a dynamic index
    // is a simple affine expression of another value. Targets then see the
    // combined byte stride/static offset and do not need to materialize the
    // intermediate index.
    uint32_t byte_shift = LOOM_LOW_SOURCE_MEMORY_ACCESS_BYTE_SHIFT_NONE;
    (void)loom_low_source_memory_access_power_of_two_shift(byte_stride,
                                                           &byte_shift);
    loom_value_facts_t base_byte_facts = loom_value_facts_unknown();
    loom_low_source_memory_dynamic_term_compute_byte_facts(
        fact_table, &vector_access, dynamic_index, dynamic_axis,
        dynamic_index_multiplier, dynamic_index_offset, byte_stride,
        stride_values, stride_value_count, &base_byte_facts);
    loom_value_facts_t expression_byte_facts = loom_value_facts_unknown();
    loom_low_source_memory_dynamic_term_compute_expression_byte_facts(
        fact_table, &vector_access, expression_facts, dynamic_axis,
        dynamic_index_offset, expression_byte_stride, stride_values,
        stride_value_count, &expression_byte_facts);
    loom_value_facts_t byte_facts =
        loom_low_source_memory_access_intersect_index_facts(
            base_byte_facts, expression_byte_facts);
    loom_low_source_memory_dynamic_term_t term = {
        .index = dynamic_index,
        .axis = dynamic_axis,
        .byte_stride = byte_stride,
        .byte_facts = byte_facts,
        .byte_shift = byte_shift,
        .stride_value_count = stride_value_count,
    };
    loom_low_source_memory_access_dynamic_index_source(
        fact_table, dynamic_index, &term.source, &term.dimension);
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

static bool loom_low_source_memory_access_plan_build_indexed_impl(
    const loom_view_region_table_t* view_regions,
    loom_low_source_memory_operation_kind_t operation_kind,
    loom_value_id_t view_value_id, loom_value_slice_t dynamic_indices,
    loom_attribute_t static_indices, loom_type_t vector_type,
    loom_vector_memory_cache_policy_t cache_policy,
    loom_low_source_memory_access_plan_t* out_plan,
    loom_low_source_memory_access_diagnostic_t* out_diagnostic) {
  const loom_module_t* module = view_regions->expression_context->module;
  if (view_value_id >= module->values.count) {
    out_diagnostic->rejection_bits |=
        LOOM_LOW_SOURCE_MEMORY_ACCESS_REJECTION_VIEW_SOURCE;
    return false;
  }
  const loom_type_t view_type = loom_module_value_type(module, view_value_id);
  return loom_low_source_memory_access_plan_from_components(
      view_regions, operation_kind, view_value_id, dynamic_indices,
      static_indices, view_type, vector_type, cache_policy, out_plan,
      out_diagnostic);
}

static bool loom_low_source_memory_access_plan_build_byte_offset_impl(
    const loom_view_region_table_t* view_regions,
    loom_low_source_memory_operation_kind_t operation_kind,
    loom_value_id_t memory_value_id, loom_value_id_t byte_offset_value_id,
    loom_vector_memory_cache_policy_t cache_policy,
    loom_low_source_memory_access_plan_t* out_plan,
    loom_low_source_memory_access_diagnostic_t* out_diagnostic) {
  const loom_module_t* module = view_regions->expression_context->module;
  const loom_value_fact_table_t* fact_table =
      view_regions->expression_context->fact_table;
  if (memory_value_id >= module->values.count ||
      byte_offset_value_id >= module->values.count) {
    out_diagnostic->rejection_bits |=
        LOOM_LOW_SOURCE_MEMORY_ACCESS_REJECTION_VIEW_SOURCE;
    return false;
  }

  loom_value_fact_buffer_reference_t reference = {0};
  if (!loom_value_facts_query_buffer_reference(
          &fact_table->context,
          loom_value_fact_table_lookup(fact_table, memory_value_id),
          &reference)) {
    out_diagnostic->rejection_bits |=
        LOOM_LOW_SOURCE_MEMORY_ACCESS_REJECTION_VIEW_SOURCE;
    return false;
  }

  *out_plan = (loom_low_source_memory_access_plan_t){
      .operation_kind = operation_kind,
      .view_value_id = memory_value_id,
      .base_view_value_id = memory_value_id,
      .memory_space = reference.memory_space,
      .address_layout = LOOM_LOW_SOURCE_MEMORY_ADDRESS_LAYOUT_COMPACT_ROW_MAJOR,
      .root_value_id = loom_value_fact_buffer_reference_resolve_root_value(
          reference, memory_value_id),
      .root_minimum_alignment =
          loom_low_source_memory_clamp_alignment(reference.minimum_alignment),
      .alias_scope_id = reference.alias_scope_id,
      .element_byte_count = 1,
      .vector_lane_count = 1,
      .vector_lane_byte_stride = 1,
      .vector_offset_kind = LOOM_LOW_SOURCE_MEMORY_VECTOR_OFFSET_NONE,
      .dynamic_view_base_value_id = LOOM_VALUE_ID_INVALID,
      .dynamic_view_base_byte_facts = loom_value_facts_unknown(),
      .minimum_alignment = 1,
      .cache_policy = cache_policy,
  };

  const loom_value_facts_t byte_offset_facts =
      loom_value_fact_table_lookup(fact_table, byte_offset_value_id);
  int64_t static_byte_offset = 0;
  if (loom_value_facts_as_exact_i64(byte_offset_facts, &static_byte_offset)) {
    out_plan->static_byte_offset = static_byte_offset;
  } else {
    loom_low_source_memory_dynamic_term_t term = {
        .index = byte_offset_value_id,
        .source = LOOM_LOW_SOURCE_MEMORY_DYNAMIC_INDEX_SOURCE_VALUE,
        .dimension = LOOM_KERNEL_DIMENSION_COUNT_,
        .axis = LOOM_LOW_SOURCE_MEMORY_DYNAMIC_TERM_AXIS_NONE,
        .byte_stride = 1,
        .byte_facts = byte_offset_facts,
        .byte_shift = 0,
    };
    loom_low_source_memory_access_dynamic_index_source(
        fact_table, byte_offset_value_id, &term.source, &term.dimension);
    if (!loom_low_source_memory_access_append_dynamic_term(out_plan, &term,
                                                           out_diagnostic)) {
      return false;
    }
  }
  loom_low_source_memory_access_finalize_alignment(out_plan);
  return true;
}

bool loom_low_source_memory_access_plan_build(
    const loom_view_region_table_t* view_regions, const loom_op_t* source_op,
    loom_low_source_memory_access_plan_t* out_plan,
    loom_low_source_memory_access_diagnostic_t* out_diagnostic) {
  *out_plan = (loom_low_source_memory_access_plan_t){0};
  *out_diagnostic = (loom_low_source_memory_access_diagnostic_t){0};
  const loom_module_t* module = view_regions->expression_context->module;
  const loom_value_fact_table_t* fact_table =
      view_regions->expression_context->fact_table;

  loom_memory_access_t access = loom_memory_access_cast(module, source_op);
  if (!loom_memory_access_isa(access)) {
    out_diagnostic->rejection_bits |=
        LOOM_LOW_SOURCE_MEMORY_ACCESS_REJECTION_UNSUPPORTED_OP;
    return false;
  }

  loom_low_source_memory_operation_kind_t operation_kind =
      LOOM_LOW_SOURCE_MEMORY_OPERATION_LOAD;
  if (!loom_low_source_memory_operation_kind_from_access(access,
                                                         &operation_kind)) {
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

  const loom_value_id_t byte_offset_value_id =
      loom_memory_access_byte_offset(access);
  if (byte_offset_value_id != LOOM_VALUE_ID_INVALID) {
    return loom_low_source_memory_access_plan_build_byte_offset_impl(
        view_regions, operation_kind, view_value_id, byte_offset_value_id,
        cache_policy, out_plan, out_diagnostic);
  }

  const loom_type_t view_type = loom_module_value_type(module, view_value_id);
  const loom_type_t vector_type =
      loom_low_source_memory_access_payload_vector_type(module, source_op,
                                                        access, view_type);
  const bool built = loom_low_source_memory_access_plan_build_indexed_impl(
      view_regions, operation_kind, view_value_id,
      loom_memory_access_dynamic_indices(access),
      loom_memory_access_static_indices(access), vector_type, cache_policy,
      out_plan, out_diagnostic);
  if (built) {
    out_plan->vector_offset_kind =
        loom_low_source_memory_access_vector_offset_kind(
            fact_table, loom_memory_access_offsets(access));
  }
  return built;
}

bool loom_low_source_memory_access_plan_build_indexed(
    const loom_view_region_table_t* view_regions,
    loom_low_source_memory_operation_kind_t operation_kind,
    loom_value_id_t view_value_id, loom_value_slice_t dynamic_indices,
    loom_attribute_t static_indices, loom_type_t vector_type,
    loom_vector_memory_cache_policy_t cache_policy,
    loom_low_source_memory_access_plan_t* out_plan,
    loom_low_source_memory_access_diagnostic_t* out_diagnostic) {
  *out_plan = (loom_low_source_memory_access_plan_t){0};
  *out_diagnostic = (loom_low_source_memory_access_diagnostic_t){0};
  return loom_low_source_memory_access_plan_build_indexed_impl(
      view_regions, operation_kind, view_value_id, dynamic_indices,
      static_indices, vector_type, cache_policy, out_plan, out_diagnostic);
}

bool loom_low_source_memory_access_plan_build_view(
    const loom_view_region_table_t* view_regions,
    loom_low_source_memory_operation_kind_t operation_kind,
    loom_value_id_t view_value_id,
    loom_vector_memory_cache_policy_t cache_policy,
    loom_low_source_memory_access_plan_t* out_plan,
    loom_low_source_memory_access_diagnostic_t* out_diagnostic) {
  *out_plan = (loom_low_source_memory_access_plan_t){0};
  *out_diagnostic = (loom_low_source_memory_access_diagnostic_t){0};
  const loom_module_t* module = view_regions->expression_context->module;
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

  return loom_low_source_memory_access_plan_build_indexed_impl(
      view_regions, operation_kind, access_view_id, dynamic_indices,
      static_indices, vector_type, cache_policy, out_plan, out_diagnostic);
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
