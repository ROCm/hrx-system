// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ops/view/reference.h"

#include "loom/ops/encoding/storage.h"
#include "loom/util/math.h"

static loom_value_facts_t loom_view_nonnegative_unknown_facts(void) {
  return loom_value_facts_make(0, INT64_MAX, 1);
}

static bool loom_view_facts_are_integer(loom_value_facts_t facts) {
  return !loom_value_facts_is_float(facts);
}

static loom_value_facts_t loom_view_clamp_nonnegative(
    loom_value_facts_t facts) {
  if (!loom_view_facts_are_integer(facts)) {
    return loom_view_nonnegative_unknown_facts();
  }
  if (facts.range_hi < 0) return loom_view_nonnegative_unknown_facts();
  int64_t lower_bound = facts.range_lo < 0 ? 0 : facts.range_lo;
  int64_t upper_bound = facts.range_hi < 0 ? 0 : facts.range_hi;
  int64_t divisor = facts.known_divisor > 0 ? facts.known_divisor : 1;
  return loom_value_facts_make(lower_bound, upper_bound, divisor);
}

static loom_value_facts_t loom_view_muli_nonnegative(loom_value_facts_t lhs,
                                                     loom_value_facts_t rhs) {
  lhs = loom_view_clamp_nonnegative(lhs);
  rhs = loom_view_clamp_nonnegative(rhs);
  loom_value_facts_t result = loom_value_facts_unknown();
  loom_value_facts_muli(&lhs, &rhs, &result);
  return loom_view_clamp_nonnegative(result);
}

static loom_value_facts_t loom_view_addi_nonnegative(loom_value_facts_t lhs,
                                                     loom_value_facts_t rhs) {
  lhs = loom_view_clamp_nonnegative(lhs);
  rhs = loom_view_clamp_nonnegative(rhs);
  loom_value_facts_t result = loom_value_facts_unknown();
  loom_value_facts_addi(&lhs, &rhs, &result);
  return loom_view_clamp_nonnegative(result);
}

static loom_value_facts_t loom_view_scale_by_element_bytes(
    loom_value_facts_t element_facts, int64_t static_element_byte_count) {
  if (static_element_byte_count < 0)
    return loom_view_nonnegative_unknown_facts();
  return loom_view_muli_nonnegative(
      element_facts, loom_value_facts_exact_i64(static_element_byte_count));
}

static uint64_t loom_view_power_of_two_factor(int64_t value) {
  if (value <= 1) return 1;
  uint64_t unsigned_value = (uint64_t)value;
  return unsigned_value & (~unsigned_value + 1);
}

static uint64_t loom_view_alignment_from_offset_facts(
    loom_value_facts_t offset_facts) {
  if (loom_value_facts_is_exact(offset_facts) &&
      !loom_value_facts_is_float(offset_facts)) {
    return loom_view_power_of_two_factor(offset_facts.range_lo);
  }
  return loom_view_power_of_two_factor(offset_facts.known_divisor);
}

static int64_t loom_view_static_element_byte_count(loom_type_t type) {
  int32_t bit_count = loom_scalar_type_bitwidth(loom_type_element_type(type));
  if (bit_count <= 0 || (bit_count % 8) != 0) return -1;
  return bit_count / 8;
}

static loom_value_facts_t loom_view_dim_facts(
    const loom_fact_context_t* context, loom_type_t type, uint8_t axis) {
  if (!loom_type_dim_is_dynamic_at(type, axis)) {
    return loom_value_facts_exact_i64(loom_type_dim_static_size_at(type, axis));
  }
  loom_value_id_t dim_value_id = loom_type_dim_value_id_at(type, axis);
  if (!context || !context->table) return loom_view_nonnegative_unknown_facts();
  return loom_view_clamp_nonnegative(
      loom_value_fact_table_lookup(context->table, dim_value_id));
}

static loom_value_facts_t loom_view_element_count_facts(
    const loom_fact_context_t* context, loom_type_t type) {
  loom_value_facts_t element_count = loom_value_facts_exact_i64(1);
  uint8_t rank = loom_type_rank(type);
  for (uint8_t axis = 0; axis < rank; ++axis) {
    loom_value_facts_t dim = loom_view_dim_facts(context, type, axis);
    if (loom_value_facts_is_exact(dim) && dim.range_lo == 0) {
      return loom_value_facts_exact_i64(0);
    }
    element_count = loom_view_muli_nonnegative(element_count, dim);
  }
  return element_count;
}

typedef struct loom_view_address_layout_t {
  // Fact summary for static encodings or SSA encodings with analysis facts.
  loom_value_fact_address_layout_t summary;

  // Caller-owned stride facts used when decoding static strided encodings.
  loom_value_facts_t static_strides[LOOM_ENCODING_ADDRESS_LAYOUT_MAX_RANK];
} loom_view_address_layout_t;

static bool loom_view_address_layout(const loom_fact_context_t* context,
                                     const loom_module_t* module,
                                     loom_type_t view_type,
                                     loom_view_address_layout_t* out_layout) {
  *out_layout = (loom_view_address_layout_t){0};
  return loom_encoding_query_type_address_layout(
      context, module, view_type, out_layout->static_strides,
      IREE_ARRAYSIZE(out_layout->static_strides), &out_layout->summary);
}

static bool loom_view_dense_axis_stride_facts(
    const loom_fact_context_t* context, loom_type_t view_type, uint8_t axis,
    loom_value_facts_t* out_stride) {
  if (axis >= loom_type_rank(view_type)) return false;
  loom_value_facts_t stride = loom_value_facts_exact_i64(1);
  uint8_t rank = loom_type_rank(view_type);
  for (uint8_t suffix_axis = (uint8_t)(axis + 1); suffix_axis < rank;
       ++suffix_axis) {
    stride = loom_view_muli_nonnegative(
        stride, loom_view_dim_facts(context, view_type, suffix_axis));
  }
  *out_stride = stride;
  return true;
}

static bool loom_view_strided_summary_axis_stride_facts(
    loom_value_fact_address_layout_t layout, uint8_t axis,
    loom_value_facts_t* out_stride) {
  if (layout.kind != LOOM_VALUE_FACT_ADDRESS_LAYOUT_STRIDED ||
      axis >= layout.rank || !layout.strides) {
    return false;
  }
  *out_stride = loom_view_clamp_nonnegative(layout.strides[axis]);
  return true;
}

static bool loom_view_axis_stride_facts(const loom_fact_context_t* context,
                                        const loom_module_t* module,
                                        loom_type_t view_type, uint8_t axis,
                                        loom_value_facts_t* out_stride) {
  loom_view_address_layout_t layout = {0};
  if (!loom_view_address_layout(context, module, view_type, &layout)) {
    return false;
  }
  if (layout.summary.kind == LOOM_VALUE_FACT_ADDRESS_LAYOUT_DENSE) {
    return loom_view_dense_axis_stride_facts(context, view_type, axis,
                                             out_stride);
  }
  if (layout.summary.kind == LOOM_VALUE_FACT_ADDRESS_LAYOUT_STRIDED) {
    return loom_view_strided_summary_axis_stride_facts(layout.summary, axis,
                                                       out_stride);
  }
  return false;
}

static loom_value_facts_t loom_view_dense_footprint_facts(
    const loom_fact_context_t* context, loom_type_t view_type,
    int64_t static_element_byte_count) {
  return loom_view_scale_by_element_bytes(
      loom_view_element_count_facts(context, view_type),
      static_element_byte_count);
}

static loom_value_facts_t loom_view_extent_max_index_facts(
    loom_value_facts_t extent_facts) {
  extent_facts = loom_view_clamp_nonnegative(extent_facts);
  int64_t lower_bound =
      extent_facts.range_lo > 0 ? extent_facts.range_lo - 1 : 0;
  int64_t upper_bound =
      extent_facts.range_hi > 0 ? extent_facts.range_hi - 1 : 0;
  return loom_value_facts_make(lower_bound, upper_bound, 1);
}

static loom_value_facts_t loom_view_strided_footprint_facts(
    const loom_fact_context_t* context, const loom_module_t* module,
    loom_type_t view_type, int64_t static_element_byte_count) {
  if (static_element_byte_count < 0)
    return loom_view_nonnegative_unknown_facts();

  uint8_t rank = loom_type_rank(view_type);
  loom_value_facts_t max_element_offset = loom_value_facts_exact_i64(0);
  bool may_be_empty = false;
  for (uint8_t axis = 0; axis < rank; ++axis) {
    loom_value_facts_t extent = loom_view_dim_facts(context, view_type, axis);
    if (loom_value_facts_is_exact(extent) && extent.range_lo == 0) {
      return loom_value_facts_exact_i64(0);
    }
    if (extent.range_lo == 0) may_be_empty = true;

    loom_value_facts_t stride = loom_value_facts_unknown();
    if (!loom_view_axis_stride_facts(context, module, view_type, axis,
                                     &stride)) {
      return loom_view_nonnegative_unknown_facts();
    }
    loom_value_facts_t contribution = loom_view_muli_nonnegative(
        loom_view_extent_max_index_facts(extent), stride);
    max_element_offset =
        loom_view_addi_nonnegative(max_element_offset, contribution);
  }

  loom_value_facts_t element_span = loom_view_addi_nonnegative(
      max_element_offset, loom_value_facts_exact_i64(1));
  loom_value_facts_t byte_span =
      loom_view_scale_by_element_bytes(element_span, static_element_byte_count);
  if (may_be_empty && byte_span.range_lo > 0) {
    byte_span.range_lo = 0;
    loom_value_facts_recompute_flags(&byte_span);
  }
  return byte_span;
}

static loom_value_facts_t loom_view_footprint_facts(
    const loom_fact_context_t* context, const loom_module_t* module,
    loom_type_t view_type, int64_t static_element_byte_count) {
  loom_view_address_layout_t layout = {0};
  if (!loom_view_address_layout(context, module, view_type, &layout)) {
    return loom_view_nonnegative_unknown_facts();
  }
  if (layout.summary.kind == LOOM_VALUE_FACT_ADDRESS_LAYOUT_DENSE) {
    return loom_view_dense_footprint_facts(context, view_type,
                                           static_element_byte_count);
  }
  if (layout.summary.kind == LOOM_VALUE_FACT_ADDRESS_LAYOUT_STRIDED) {
    return loom_view_strided_footprint_facts(context, module, view_type,
                                             static_element_byte_count);
  }
  return loom_view_nonnegative_unknown_facts();
}

static bool loom_view_static_or_dynamic_index_facts(
    const loom_fact_context_t* context, loom_attribute_t static_indices,
    loom_value_slice_t dynamic_indices, uint8_t axis,
    loom_value_facts_t* out_index) {
  if (static_indices.kind != LOOM_ATTR_I64_ARRAY ||
      axis >= static_indices.count) {
    return false;
  }
  uint16_t dynamic_ordinal = 0;
  for (uint8_t i = 0; i <= axis; ++i) {
    int64_t static_index = static_indices.i64_array[i];
    if (static_index != INT64_MIN) {
      if (i == axis) {
        if (static_index < 0) return false;
        *out_index = loom_value_facts_exact_i64(static_index);
        return true;
      }
      continue;
    }
    if (i == axis) {
      if (dynamic_ordinal >= dynamic_indices.count || !context ||
          !context->table) {
        return false;
      }
      loom_value_id_t index_value_id = dynamic_indices.values[dynamic_ordinal];
      *out_index = loom_view_clamp_nonnegative(
          loom_value_fact_table_lookup(context->table, index_value_id));
      return true;
    }
    ++dynamic_ordinal;
  }
  return false;
}

static bool loom_view_offsets_are_exact_zero(
    const loom_fact_context_t* context, loom_attribute_t static_offsets,
    loom_value_slice_t dynamic_offsets) {
  if (static_offsets.kind != LOOM_ATTR_I64_ARRAY) return false;
  for (uint16_t axis = 0; axis < static_offsets.count; ++axis) {
    loom_value_facts_t offset = loom_value_facts_unknown();
    if (!loom_view_static_or_dynamic_index_facts(
            context, static_offsets, dynamic_offsets, (uint8_t)axis, &offset)) {
      return false;
    }
    if (!loom_value_facts_is_exact(offset) || offset.range_lo != 0) {
      return false;
    }
  }
  return true;
}

static bool loom_view_subview_additional_byte_offset_facts(
    const loom_fact_context_t* context, const loom_module_t* module,
    loom_type_t source_type, loom_attribute_t static_offsets,
    loom_value_slice_t dynamic_offsets, int64_t static_element_byte_count,
    loom_value_facts_t* out_offset) {
  if (static_offsets.kind != LOOM_ATTR_I64_ARRAY ||
      static_offsets.count != loom_type_rank(source_type)) {
    return false;
  }
  if (loom_view_offsets_are_exact_zero(context, static_offsets,
                                       dynamic_offsets)) {
    *out_offset = loom_value_facts_exact_i64(0);
    return true;
  }
  if (static_element_byte_count < 0) return false;

  loom_value_facts_t element_offset = loom_value_facts_exact_i64(0);
  uint8_t rank = loom_type_rank(source_type);
  for (uint8_t axis = 0; axis < rank; ++axis) {
    loom_value_facts_t index = loom_value_facts_unknown();
    if (!loom_view_static_or_dynamic_index_facts(
            context, static_offsets, dynamic_offsets, axis, &index)) {
      return false;
    }
    if (loom_value_facts_is_exact(index) && index.range_lo == 0) continue;

    loom_value_facts_t stride = loom_value_facts_unknown();
    if (!loom_view_axis_stride_facts(context, module, source_type, axis,
                                     &stride)) {
      return false;
    }
    loom_value_facts_t contribution = loom_view_muli_nonnegative(index, stride);
    element_offset = loom_view_addi_nonnegative(element_offset, contribution);
  }

  *out_offset = loom_view_scale_by_element_bytes(element_offset,
                                                 static_element_byte_count);
  return true;
}

static loom_value_fact_buffer_reference_t loom_view_default_buffer_reference(
    loom_value_id_t buffer_value_id) {
  return (loom_value_fact_buffer_reference_t){
      .maximum_byte_extent = loom_view_nonnegative_unknown_facts(),
      .minimum_alignment = 1,
      .memory_space = LOOM_VALUE_FACT_MEMORY_SPACE_UNKNOWN,
      .root_value_id = buffer_value_id,
      .alias_scope_id = LOOM_VALUE_FACT_ALIAS_SCOPE_ID_NONE,
      .nullability = LOOM_VALUE_FACT_REFERENCE_NULLABILITY_UNKNOWN,
  };
}

static loom_value_fact_view_reference_t loom_view_default_view_reference(
    loom_value_id_t source_value_id, loom_type_t source_type) {
  return (loom_value_fact_view_reference_t){
      .base_byte_offset = loom_value_facts_exact_i64(0),
      .footprint_byte_length = loom_view_nonnegative_unknown_facts(),
      .minimum_alignment = 1,
      .root_minimum_alignment = 1,
      .static_element_byte_count =
          loom_view_static_element_byte_count(source_type),
      .memory_space = LOOM_VALUE_FACT_MEMORY_SPACE_UNKNOWN,
      .root_value_id = source_value_id,
      .alias_scope_id = LOOM_VALUE_FACT_ALIAS_SCOPE_ID_NONE,
      .nullability = LOOM_VALUE_FACT_REFERENCE_NULLABILITY_UNKNOWN,
  };
}

iree_status_t loom_view_reference_make_buffer_view(
    loom_fact_context_t* context, const loom_module_t* module,
    loom_value_id_t buffer_value_id, loom_value_facts_t buffer_facts,
    loom_value_facts_t byte_offset_facts, loom_type_t result_type,
    loom_value_facts_t* out) {
  loom_value_fact_buffer_reference_t buffer_reference =
      loom_view_default_buffer_reference(buffer_value_id);
  (void)loom_value_facts_query_buffer_reference(context, buffer_facts,
                                                &buffer_reference);

  loom_value_fact_view_reference_t view_reference = {0};
  view_reference.base_byte_offset =
      loom_view_clamp_nonnegative(byte_offset_facts);
  view_reference.static_element_byte_count =
      loom_view_static_element_byte_count(result_type);
  view_reference.footprint_byte_length = loom_view_footprint_facts(
      context, module, result_type, view_reference.static_element_byte_count);
  view_reference.minimum_alignment =
      loom_view_alignment_from_offset_facts(view_reference.base_byte_offset);
  view_reference.root_minimum_alignment = buffer_reference.minimum_alignment;
  view_reference.memory_space = buffer_reference.memory_space;
  view_reference.root_value_id = buffer_reference.root_value_id;
  view_reference.alias_scope_id = buffer_reference.alias_scope_id;
  view_reference.nullability = buffer_reference.nullability;
  return loom_value_facts_make_view_reference(context, view_reference, out);
}

iree_status_t loom_view_reference_make_subview(
    loom_fact_context_t* context, const loom_module_t* module,
    loom_value_id_t source_value_id, loom_value_facts_t source_facts,
    loom_attribute_t static_offsets, loom_value_slice_t dynamic_offsets,
    loom_type_t source_type, loom_type_t result_type, loom_value_facts_t* out) {
  loom_value_fact_view_reference_t source_reference =
      loom_view_default_view_reference(source_value_id, source_type);
  (void)loom_value_facts_query_view_reference(context, source_facts,
                                              &source_reference);

  loom_value_fact_view_reference_t view_reference = {0};
  view_reference.static_element_byte_count =
      loom_view_static_element_byte_count(result_type);
  loom_value_facts_t additional_byte_offset = loom_value_facts_unknown();
  if (!loom_view_subview_additional_byte_offset_facts(
          context, module, source_type, static_offsets, dynamic_offsets,
          view_reference.static_element_byte_count, &additional_byte_offset)) {
    additional_byte_offset = loom_view_nonnegative_unknown_facts();
  }
  view_reference.base_byte_offset = loom_view_addi_nonnegative(
      source_reference.base_byte_offset, additional_byte_offset);
  view_reference.footprint_byte_length = loom_view_footprint_facts(
      context, module, result_type, view_reference.static_element_byte_count);
  view_reference.minimum_alignment =
      loom_view_alignment_from_offset_facts(view_reference.base_byte_offset);
  view_reference.root_minimum_alignment =
      source_reference.root_minimum_alignment;
  view_reference.memory_space = source_reference.memory_space;
  view_reference.root_value_id = source_reference.root_value_id;
  view_reference.alias_scope_id = source_reference.alias_scope_id;
  view_reference.nullability = source_reference.nullability;
  return loom_value_facts_make_view_reference(context, view_reference, out);
}

iree_status_t loom_view_reference_make_refine(
    loom_fact_context_t* context, const loom_module_t* module,
    loom_value_id_t source_value_id, loom_value_facts_t source_facts,
    loom_type_t source_type, loom_type_t result_type, loom_value_facts_t* out) {
  loom_value_fact_view_reference_t source_reference =
      loom_view_default_view_reference(source_value_id, source_type);
  (void)loom_value_facts_query_view_reference(context, source_facts,
                                              &source_reference);

  loom_value_fact_view_reference_t view_reference = source_reference;
  view_reference.static_element_byte_count =
      loom_view_static_element_byte_count(result_type);
  view_reference.footprint_byte_length = loom_view_footprint_facts(
      context, module, result_type, view_reference.static_element_byte_count);
  return loom_value_facts_make_view_reference(context, view_reference, out);
}
