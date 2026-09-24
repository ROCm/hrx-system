// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/view_regions.h"

#include <string.h>

#include "iree/base/internal/math.h"
#include "loom/analysis/symbolic_congruence.h"
#include "loom/analysis/symbolic_expr_proof.h"
#include "loom/ir/attribute.h"
#include "loom/ir/context.h"
#include "loom/ops/buffer/ops.h"
#include "loom/ops/encoding/storage.h"
#include "loom/ops/kernel/ops.h"
#include "loom/ops/view/ops.h"
#include "loom/util/fact_cfg.h"

//===----------------------------------------------------------------------===//
// Storage
//===----------------------------------------------------------------------===//

enum loom_view_region_value_state_e {
  LOOM_VIEW_REGION_VALUE_EMPTY = 0,
  LOOM_VIEW_REGION_VALUE_VISITING = 1,
  LOOM_VIEW_REGION_VALUE_READY = 2,
};

static iree_status_t loom_view_region_table_append_region(
    loom_view_region_table_t* table, const loom_view_region_t* region,
    loom_view_region_t** out_region) {
  if (table->region_count >= table->region_capacity) {
    void* regions = table->regions;
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        table->expression_context->arena, table->region_count,
        table->region_count + 1, sizeof(*table->regions),
        &table->region_capacity, &regions));
    table->regions = (loom_view_region_t*)regions;
  }
  loom_view_region_t* stored_region = &table->regions[table->region_count];
  *stored_region = *region;
  stored_region->region_id = (loom_view_region_id_t)table->region_count;
  ++table->region_count;
  *out_region = stored_region;
  return iree_ok_status();
}

iree_status_t loom_view_region_table_initialize(
    loom_local_value_domain_t* value_domain,
    loom_symbolic_expr_context_t* expression_context,
    loom_view_region_table_t* out_table) {
  IREE_ASSERT(loom_local_value_domain_is_acquired(value_domain));
  memset(out_table, 0, sizeof(*out_table));
  out_table->expression_context = expression_context;
  out_table->value_domain = value_domain;
  // Captured views can name backing roots absent from the source operands.
  // Retain those fact-owned identities before allocating ordinal-keyed tables
  // so read/write aggregation covers the captured storage as well as its views.
  const loom_value_fact_table_t* fact_table = expression_context->fact_table;
  for (loom_value_ordinal_t i = 0; i < value_domain->value_count; ++i) {
    const loom_value_id_t value_id = value_domain->value_ids[i];
    const loom_value_facts_t facts =
        loom_value_fact_table_lookup(fact_table, value_id);
    loom_value_id_t root = LOOM_VALUE_ID_INVALID;
    loom_value_fact_buffer_reference_t buffer;
    loom_value_fact_view_reference_t view;
    if (loom_value_facts_query_buffer_reference(&fact_table->context, facts,
                                                &buffer)) {
      root =
          loom_value_fact_buffer_reference_resolve_root_value(buffer, value_id);
    } else if (loom_value_facts_query_view_reference(&fact_table->context,
                                                     facts, &view)) {
      root = view.root_value_id;
    }
    if (root != LOOM_VALUE_ID_INVALID) {
      loom_value_ordinal_t root_ordinal;
      IREE_RETURN_IF_ERROR(loom_local_value_domain_register_value(
          value_domain, expression_context->arena, root, &root_ordinal));
    }
  }
  const iree_host_size_t value_count = value_domain->value_count;
  if (value_count == 0) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      expression_context->arena, value_count,
      sizeof(*out_table->region_ids_by_value_ordinal),
      (void**)&out_table->region_ids_by_value_ordinal));
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(expression_context->arena, value_count,
                                sizeof(*out_table->states_by_value_ordinal),
                                (void**)&out_table->states_by_value_ordinal));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      expression_context->arena, value_count,
      sizeof(*out_table->storage_flags_by_value_ordinal),
      (void**)&out_table->storage_flags_by_value_ordinal));
  for (iree_host_size_t i = 0; i < value_count; ++i) {
    out_table->region_ids_by_value_ordinal[i] = LOOM_VIEW_REGION_ID_INVALID;
  }
  memset(out_table->states_by_value_ordinal, 0,
         value_count * sizeof(*out_table->states_by_value_ordinal));
  memset(out_table->storage_flags_by_value_ordinal, 0,
         value_count * sizeof(*out_table->storage_flags_by_value_ordinal));
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Expression helpers
//===----------------------------------------------------------------------===//

static loom_value_facts_t loom_view_region_lookup_facts(
    const loom_view_region_table_t* table, loom_value_id_t value_id) {
  return loom_value_fact_table_lookup(table->expression_context->fact_table,
                                      value_id);
}

static bool loom_view_region_facts_exact_i64(loom_value_facts_t facts,
                                             int64_t* out_value) {
  if (!loom_value_facts_is_exact(facts) || loom_value_facts_is_float(facts)) {
    return false;
  }
  *out_value = facts.range_lo;
  return true;
}

static void loom_view_region_expression_refine_facts(
    loom_symbolic_expr_t* expression, loom_value_facts_t facts) {
  if (loom_value_facts_is_unknown(facts)) {
    return;
  }
  int64_t exact_value = 0;
  if (loom_symbolic_expr_is_constant(expression) &&
      loom_view_region_facts_exact_i64(facts, &exact_value)) {
    if (expression->constant == exact_value) {
      expression->facts = facts;
    }
    return;
  }
  if (loom_symbolic_expr_is_constant(expression)) {
    return;
  }
  if (loom_value_facts_is_unknown(expression->facts) ||
      loom_value_facts_is_exact(facts)) {
    expression->facts = facts;
  }
}

static void loom_view_region_expression_from_facts(
    loom_value_facts_t facts, loom_symbolic_expr_t* out_expression) {
  int64_t exact_value = 0;
  if (loom_view_region_facts_exact_i64(facts, &exact_value)) {
    loom_symbolic_expr_constant(exact_value, out_expression);
    out_expression->facts = facts;
    return;
  }
  loom_symbolic_expr_unknown(facts, out_expression);
}

static bool loom_view_region_expr_is_constant(
    const loom_symbolic_expr_t* expression, int64_t* out_value) {
  if (!loom_symbolic_expr_is_constant(expression)) {
    return false;
  }
  *out_value = expression->constant;
  return true;
}

static iree_status_t loom_view_region_expr_add(
    loom_view_region_table_t* table, const loom_symbolic_expr_t* left,
    const loom_symbolic_expr_t* right, loom_symbolic_expr_t* out_expression) {
  return loom_symbolic_expr_add(table->expression_context, left, right,
                                out_expression);
}

static iree_status_t loom_view_region_expr_sub(
    loom_view_region_table_t* table, const loom_symbolic_expr_t* left,
    const loom_symbolic_expr_t* right, loom_symbolic_expr_t* out_expression) {
  return loom_symbolic_expr_sub(table->expression_context, left, right,
                                out_expression);
}

static iree_status_t loom_view_region_expr_mul(
    loom_view_region_table_t* table, const loom_symbolic_expr_t* left,
    const loom_symbolic_expr_t* right, loom_symbolic_expr_t* out_expression) {
  int64_t left_constant = 0;
  int64_t right_constant = 0;
  if (loom_view_region_expr_is_constant(left, &left_constant)) {
    return loom_symbolic_expr_mul_i64(table->expression_context, right,
                                      left_constant, out_expression);
  }
  if (loom_view_region_expr_is_constant(right, &right_constant)) {
    return loom_symbolic_expr_mul_i64(table->expression_context, left,
                                      right_constant, out_expression);
  }
  loom_value_facts_t facts = loom_value_facts_unknown();
  loom_value_facts_muli(&left->facts, &right->facts, &facts);
  loom_symbolic_expr_unknown(facts, out_expression);
  return iree_ok_status();
}

static iree_status_t loom_view_region_dim_expr(
    loom_view_region_table_t* table, loom_type_t type, uint8_t axis,
    loom_symbolic_expr_t* out_expression) {
  if (!loom_type_dim_is_dynamic_at(type, axis)) {
    loom_symbolic_expr_constant(loom_type_dim_static_size_at(type, axis),
                                out_expression);
    return iree_ok_status();
  }
  return loom_symbolic_expr_from_value(table->expression_context,
                                       loom_type_dim_value_id_at(type, axis),
                                       out_expression);
}

static int64_t loom_view_region_static_element_byte_count(loom_type_t type) {
  int32_t bit_count = loom_scalar_type_bitwidth(loom_type_element_type(type));
  if (bit_count <= 0 || (bit_count % 8) != 0) {
    return -1;
  }
  return bit_count / 8;
}

//===----------------------------------------------------------------------===//
// Layout queries
//===----------------------------------------------------------------------===//

typedef struct loom_view_region_address_layout_t {
  // Address-layout facts decoded from the view type encoding.
  loom_value_fact_address_layout_t summary;

  // Inline stride facts backing summary.strides when decoded locally.
  loom_value_facts_t static_strides[LOOM_ENCODING_ADDRESS_LAYOUT_MAX_RANK];
} loom_view_region_address_layout_t;

static bool loom_view_region_address_layout(
    loom_view_region_table_t* table, loom_type_t view_type,
    loom_view_region_address_layout_t* out_layout) {
  *out_layout = (loom_view_region_address_layout_t){0};
  return loom_encoding_query_type_address_layout(
      &table->expression_context->fact_table->context,
      table->expression_context->module, view_type, out_layout->static_strides,
      IREE_ARRAYSIZE(out_layout->static_strides), &out_layout->summary);
}

static iree_status_t loom_view_region_static_or_dynamic_expr(
    loom_view_region_table_t* table, loom_attribute_t static_values,
    loom_value_slice_t dynamic_values, uint8_t axis,
    loom_symbolic_expr_t* out_expression, bool* out_known) {
  *out_known = false;
  if (static_values.kind != LOOM_ATTR_I64_ARRAY ||
      axis >= static_values.count) {
    return iree_ok_status();
  }
  uint16_t dynamic_ordinal = 0;
  for (uint8_t i = 0; i <= axis; ++i) {
    int64_t static_value = static_values.i64_array[i];
    if (static_value != INT64_MIN) {
      if (i == axis) {
        loom_symbolic_expr_constant(static_value, out_expression);
        *out_known = true;
        return iree_ok_status();
      }
      continue;
    }
    if (i == axis) {
      if (dynamic_ordinal >= dynamic_values.count) {
        return iree_ok_status();
      }
      IREE_RETURN_IF_ERROR(loom_symbolic_expr_from_value(
          table->expression_context, dynamic_values.values[dynamic_ordinal],
          out_expression));
      *out_known = true;
      return iree_ok_status();
    }
    ++dynamic_ordinal;
  }
  return iree_ok_status();
}

static iree_status_t loom_view_region_dense_axis_stride_expr(
    loom_view_region_table_t* table, loom_type_t view_type, uint8_t axis,
    loom_symbolic_expr_t* out_expression, bool* out_known) {
  *out_known = false;
  if (axis >= loom_type_rank(view_type)) {
    return iree_ok_status();
  }
  loom_symbolic_expr_constant(1, out_expression);
  uint8_t rank = loom_type_rank(view_type);
  for (uint8_t suffix_axis = (uint8_t)(axis + 1); suffix_axis < rank;
       ++suffix_axis) {
    loom_symbolic_expr_t dim = {0};
    IREE_RETURN_IF_ERROR(
        loom_view_region_dim_expr(table, view_type, suffix_axis, &dim));
    loom_symbolic_expr_t product = {0};
    IREE_RETURN_IF_ERROR(
        loom_view_region_expr_mul(table, out_expression, &dim, &product));
    *out_expression = product;
    if (!loom_symbolic_expr_is_linear(out_expression)) {
      *out_known = true;
      return iree_ok_status();
    }
  }
  *out_known = true;
  return iree_ok_status();
}

static iree_status_t loom_view_region_axis_stride_expr(
    loom_view_region_table_t* table, loom_type_t view_type, uint8_t axis,
    loom_symbolic_expr_t* out_expression, bool* out_known) {
  *out_known = false;
  loom_view_region_address_layout_t layout = {0};
  if (!loom_view_region_address_layout(table, view_type, &layout)) {
    loom_symbolic_expr_unknown(loom_value_facts_unknown(), out_expression);
    return iree_ok_status();
  }
  if (layout.summary.kind == LOOM_VALUE_FACT_ADDRESS_LAYOUT_DENSE) {
    return loom_view_region_dense_axis_stride_expr(table, view_type, axis,
                                                   out_expression, out_known);
  }
  if (layout.summary.kind != LOOM_VALUE_FACT_ADDRESS_LAYOUT_STRIDED ||
      axis >= layout.summary.rank || !layout.summary.strides) {
    loom_symbolic_expr_unknown(loom_value_facts_unknown(), out_expression);
    return iree_ok_status();
  }
  loom_value_facts_t stride_facts = layout.summary.strides[axis];
  if (loom_value_facts_is_exact(stride_facts) &&
      !loom_value_facts_is_float(stride_facts)) {
    loom_symbolic_expr_constant(stride_facts.range_lo, out_expression);
    out_expression->facts = stride_facts;
    *out_known = true;
    return iree_ok_status();
  }
  const loom_value_fact_layout_strides_t stride_values =
      loom_encoding_query_type_layout_strides(
          &table->expression_context->fact_table->context, view_type);
  if (axis < stride_values.count &&
      stride_values.values[axis] != LOOM_VALUE_ID_INVALID) {
    IREE_RETURN_IF_ERROR(loom_symbolic_expr_from_value(
        table->expression_context, stride_values.values[axis], out_expression));
    loom_view_region_expression_refine_facts(out_expression, stride_facts);
    *out_known = true;
    return iree_ok_status();
  }
  loom_symbolic_expr_unknown(stride_facts, out_expression);
  *out_known = true;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Footprints and offsets
//===----------------------------------------------------------------------===//

static bool loom_view_region_extent_is_statically_zero(
    loom_symbolic_expr_t extent) {
  return loom_symbolic_expr_is_constant(&extent) && extent.constant == 0;
}

static bool loom_view_region_extent_is_positive(loom_symbolic_expr_t extent) {
  return !loom_value_facts_is_float(extent.facts) && extent.facts.range_lo > 0;
}

static iree_status_t loom_view_region_dense_footprint_expr(
    loom_view_region_table_t* table, loom_type_t view_type,
    int64_t static_element_byte_count, loom_symbolic_expr_t* out_expression) {
  if (static_element_byte_count < 0) {
    loom_symbolic_expr_unknown(loom_value_facts_make(0, INT64_MAX, 1),
                               out_expression);
    return iree_ok_status();
  }
  loom_symbolic_expr_constant(static_element_byte_count, out_expression);
  uint8_t rank = loom_type_rank(view_type);
  for (uint8_t axis = 0; axis < rank; ++axis) {
    loom_symbolic_expr_t dim = {0};
    IREE_RETURN_IF_ERROR(
        loom_view_region_dim_expr(table, view_type, axis, &dim));
    if (loom_view_region_extent_is_statically_zero(dim)) {
      loom_symbolic_expr_constant(0, out_expression);
      return iree_ok_status();
    }
    loom_symbolic_expr_t product = {0};
    IREE_RETURN_IF_ERROR(
        loom_view_region_expr_mul(table, out_expression, &dim, &product));
    *out_expression = product;
    if (!loom_symbolic_expr_is_linear(out_expression)) {
      return iree_ok_status();
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_view_region_strided_footprint_expr(
    loom_view_region_table_t* table, loom_type_t view_type,
    int64_t static_element_byte_count, loom_symbolic_expr_t* out_expression) {
  if (static_element_byte_count < 0) {
    loom_symbolic_expr_unknown(loom_value_facts_make(0, INT64_MAX, 1),
                               out_expression);
    return iree_ok_status();
  }

  loom_symbolic_expr_t max_element_offset = {0};
  loom_symbolic_expr_constant(0, &max_element_offset);
  uint8_t rank = loom_type_rank(view_type);
  for (uint8_t axis = 0; axis < rank; ++axis) {
    loom_symbolic_expr_t extent = {0};
    IREE_RETURN_IF_ERROR(
        loom_view_region_dim_expr(table, view_type, axis, &extent));
    if (loom_view_region_extent_is_statically_zero(extent)) {
      loom_symbolic_expr_constant(0, out_expression);
      return iree_ok_status();
    }
    if (!loom_view_region_extent_is_positive(extent)) {
      loom_symbolic_expr_unknown(loom_value_facts_make(0, INT64_MAX, 1),
                                 out_expression);
      return iree_ok_status();
    }

    loom_symbolic_expr_t one = {0};
    loom_symbolic_expr_constant(1, &one);
    loom_symbolic_expr_t extent_minus_one = {0};
    IREE_RETURN_IF_ERROR(
        loom_view_region_expr_sub(table, &extent, &one, &extent_minus_one));

    loom_symbolic_expr_t stride = {0};
    bool stride_known = false;
    IREE_RETURN_IF_ERROR(loom_view_region_axis_stride_expr(
        table, view_type, axis, &stride, &stride_known));
    if (!stride_known) {
      loom_symbolic_expr_unknown(loom_value_facts_make(0, INT64_MAX, 1),
                                 out_expression);
      return iree_ok_status();
    }
    loom_symbolic_expr_t contribution = {0};
    IREE_RETURN_IF_ERROR(loom_view_region_expr_mul(table, &extent_minus_one,
                                                   &stride, &contribution));
    loom_symbolic_expr_t new_offset = {0};
    IREE_RETURN_IF_ERROR(loom_view_region_expr_add(table, &max_element_offset,
                                                   &contribution, &new_offset));
    max_element_offset = new_offset;
    if (!loom_symbolic_expr_is_linear(&max_element_offset)) {
      loom_symbolic_expr_unknown(loom_value_facts_make(0, INT64_MAX, 1),
                                 out_expression);
      return iree_ok_status();
    }
  }

  loom_symbolic_expr_t one = {0};
  loom_symbolic_expr_constant(1, &one);
  loom_symbolic_expr_t element_span = {0};
  IREE_RETURN_IF_ERROR(loom_view_region_expr_add(table, &max_element_offset,
                                                 &one, &element_span));
  loom_symbolic_expr_t element_bytes = {0};
  loom_symbolic_expr_constant(static_element_byte_count, &element_bytes);
  return loom_view_region_expr_mul(table, &element_span, &element_bytes,
                                   out_expression);
}

static iree_status_t loom_view_region_footprint_expr(
    loom_view_region_table_t* table, loom_type_t view_type,
    int64_t static_element_byte_count, loom_symbolic_expr_t* out_expression) {
  loom_view_region_address_layout_t layout = {0};
  if (!loom_view_region_address_layout(table, view_type, &layout)) {
    loom_symbolic_expr_unknown(loom_value_facts_make(0, INT64_MAX, 1),
                               out_expression);
    return iree_ok_status();
  }
  if (layout.summary.kind == LOOM_VALUE_FACT_ADDRESS_LAYOUT_DENSE) {
    return loom_view_region_dense_footprint_expr(
        table, view_type, static_element_byte_count, out_expression);
  }
  if (layout.summary.kind == LOOM_VALUE_FACT_ADDRESS_LAYOUT_STRIDED) {
    return loom_view_region_strided_footprint_expr(
        table, view_type, static_element_byte_count, out_expression);
  }
  loom_symbolic_expr_unknown(loom_value_facts_make(0, INT64_MAX, 1),
                             out_expression);
  return iree_ok_status();
}

static iree_status_t loom_view_region_additional_byte_offset_expr(
    loom_view_region_table_t* table, loom_type_t source_type,
    loom_attribute_t static_offsets, loom_value_slice_t dynamic_offsets,
    int64_t static_element_byte_count, loom_symbolic_expr_t* out_expression) {
  if (static_element_byte_count < 0 ||
      static_offsets.kind != LOOM_ATTR_I64_ARRAY ||
      static_offsets.count != loom_type_rank(source_type)) {
    loom_symbolic_expr_unknown(loom_value_facts_unknown(), out_expression);
    return iree_ok_status();
  }

  loom_symbolic_expr_t element_offset = {0};
  loom_symbolic_expr_constant(0, &element_offset);
  uint8_t rank = loom_type_rank(source_type);
  for (uint8_t axis = 0; axis < rank; ++axis) {
    loom_symbolic_expr_t index = {0};
    bool index_known = false;
    IREE_RETURN_IF_ERROR(loom_view_region_static_or_dynamic_expr(
        table, static_offsets, dynamic_offsets, axis, &index, &index_known));
    if (!index_known) {
      loom_symbolic_expr_unknown(loom_value_facts_unknown(), out_expression);
      return iree_ok_status();
    }
    int64_t static_index = 0;
    if (loom_view_region_expr_is_constant(&index, &static_index) &&
        static_index == 0) {
      continue;
    }

    loom_symbolic_expr_t stride = {0};
    bool stride_known = false;
    IREE_RETURN_IF_ERROR(loom_view_region_axis_stride_expr(
        table, source_type, axis, &stride, &stride_known));
    if (!stride_known) {
      loom_symbolic_expr_unknown(loom_value_facts_unknown(), out_expression);
      return iree_ok_status();
    }
    loom_symbolic_expr_t contribution = {0};
    IREE_RETURN_IF_ERROR(
        loom_view_region_expr_mul(table, &index, &stride, &contribution));
    loom_symbolic_expr_t new_offset = {0};
    IREE_RETURN_IF_ERROR(loom_view_region_expr_add(table, &element_offset,
                                                   &contribution, &new_offset));
    element_offset = new_offset;
    if (!loom_symbolic_expr_is_linear(&element_offset)) {
      loom_symbolic_expr_unknown(loom_value_facts_unknown(), out_expression);
      return iree_ok_status();
    }
  }

  loom_symbolic_expr_t element_bytes = {0};
  loom_symbolic_expr_constant(static_element_byte_count, &element_bytes);
  return loom_view_region_expr_mul(table, &element_offset, &element_bytes,
                                   out_expression);
}

//===----------------------------------------------------------------------===//
// Region construction
//===----------------------------------------------------------------------===//

static loom_value_fact_view_reference_t loom_view_region_default_reference(
    loom_value_id_t value_id, loom_type_t view_type) {
  return (loom_value_fact_view_reference_t){
      .base_byte_offset = loom_value_facts_exact_i64(0),
      .footprint_byte_length = loom_value_facts_make(0, INT64_MAX, 1),
      .minimum_alignment = 1,
      .root_minimum_alignment = 1,
      .static_element_byte_count =
          loom_view_region_static_element_byte_count(view_type),
      .memory_space = LOOM_VALUE_FACT_MEMORY_SPACE_UNKNOWN,
      .root_value_id = value_id,
      .buffer_value_id = LOOM_VALUE_ID_INVALID,
      .alias_scope_id = LOOM_VALUE_FACT_ALIAS_SCOPE_ID_NONE,
      .nullability = LOOM_VALUE_FACT_REFERENCE_NULLABILITY_UNKNOWN,
  };
}

static iree_status_t loom_view_region_build_default(
    loom_view_region_table_t* table, loom_value_id_t value_id,
    loom_type_t view_type, loom_value_fact_view_reference_t reference,
    loom_view_region_t* out_region) {
  loom_symbolic_expr_t begin = {0};
  loom_view_region_expression_from_facts(reference.base_byte_offset, &begin);

  loom_symbolic_expr_t length = {0};
  IREE_RETURN_IF_ERROR(loom_view_region_footprint_expr(
      table, view_type, reference.static_element_byte_count, &length));
  loom_view_region_expression_refine_facts(&length,
                                           reference.footprint_byte_length);

  loom_symbolic_expr_t end = {0};
  IREE_RETURN_IF_ERROR(loom_view_region_expr_add(table, &begin, &length, &end));
  loom_symbolic_expr_t zero = {0};
  loom_symbolic_expr_constant(0, &zero);

  *out_region = (loom_view_region_t){
      .region_id = LOOM_VIEW_REGION_ID_INVALID,
      .view_value_id = value_id,
      .base_view_value_id = value_id,
      .root_value_id = reference.root_value_id,
      .alias_scope_id = reference.alias_scope_id,
      .nullability = reference.nullability,
      .address_bitwidth = reference.address_bitwidth,
      .origin = reference.origin,
      .begin_byte_offset = begin,
      .base_begin_byte_offset = begin,
      .projection_byte_offset = zero,
      .begin_value_id = LOOM_VALUE_ID_INVALID,
      .byte_length = length,
      .end_byte_offset = end,
      .minimum_alignment = reference.minimum_alignment,
      .root_minimum_alignment = reference.root_minimum_alignment,
      .static_element_byte_count = reference.static_element_byte_count,
      .memory_space = reference.memory_space,
      .access_flags = 0,
      .precision_flags = 0,
  };
  return iree_ok_status();
}

static void loom_view_region_refresh_precision(loom_view_region_t* region) {
  region->precision_flags = 0;
  if (region->root_value_id != LOOM_VALUE_ID_INVALID) {
    region->precision_flags |= LOOM_VIEW_REGION_PRECISION_ROOT;
  }
  if (loom_symbolic_expr_is_linear(&region->begin_byte_offset)) {
    region->precision_flags |= LOOM_VIEW_REGION_PRECISION_BEGIN;
  }
  if (loom_symbolic_expr_is_linear(&region->byte_length)) {
    region->precision_flags |= LOOM_VIEW_REGION_PRECISION_LENGTH;
  }
  if (loom_symbolic_expr_is_linear(&region->end_byte_offset)) {
    region->precision_flags |= LOOM_VIEW_REGION_PRECISION_END;
  }
}

static iree_status_t loom_view_region_build_for_value(
    loom_view_region_table_t* table, loom_value_id_t value_id,
    loom_view_region_t* out_region);

static iree_status_t loom_view_region_get_source(
    loom_view_region_table_t* table, loom_value_id_t source_value_id,
    const loom_view_region_t** out_region) {
  return loom_view_region_table_get(table, source_value_id, out_region);
}

static iree_status_t loom_view_region_build_buffer_view(
    loom_view_region_table_t* table, loom_value_id_t value_id,
    const loom_op_t* op, loom_type_t view_type,
    loom_value_fact_view_reference_t reference,
    loom_view_region_t* out_region) {
  IREE_RETURN_IF_ERROR(loom_view_region_build_default(
      table, value_id, view_type, reference, out_region));
  out_region->root_value_id = reference.root_value_id;
  out_region->base_view_value_id = value_id;
  out_region->begin_value_id = loom_buffer_view_byte_offset(op);
  IREE_RETURN_IF_ERROR(loom_symbolic_expr_from_value(
      table->expression_context, loom_buffer_view_byte_offset(op),
      &out_region->begin_byte_offset));
  loom_view_region_expression_refine_facts(&out_region->begin_byte_offset,
                                           reference.base_byte_offset);
  out_region->base_begin_byte_offset = out_region->begin_byte_offset;
  loom_symbolic_expr_constant(0, &out_region->projection_byte_offset);
  IREE_RETURN_IF_ERROR(loom_view_region_expr_add(
      table, &out_region->begin_byte_offset, &out_region->byte_length,
      &out_region->end_byte_offset));
  return iree_ok_status();
}

static iree_status_t loom_view_region_build_subview(
    loom_view_region_table_t* table, loom_value_id_t value_id,
    const loom_op_t* op, loom_type_t view_type,
    loom_value_fact_view_reference_t reference,
    loom_view_region_t* out_region) {
  const loom_view_region_t* source_region = NULL;
  IREE_RETURN_IF_ERROR(loom_view_region_get_source(
      table, loom_view_subview_source(op), &source_region));
  IREE_RETURN_IF_ERROR(loom_view_region_build_default(
      table, value_id, view_type, reference, out_region));
  if (!source_region) {
    return iree_ok_status();
  }

  loom_type_t source_type = loom_module_value_type(
      table->expression_context->module, loom_view_subview_source(op));
  loom_symbolic_expr_t additional_offset = {0};
  IREE_RETURN_IF_ERROR(loom_view_region_additional_byte_offset_expr(
      table, source_type, loom_view_subview_static_offsets(op),
      loom_view_subview_offsets(op), reference.static_element_byte_count,
      &additional_offset));
  loom_symbolic_expr_t begin = {0};
  IREE_RETURN_IF_ERROR(loom_view_region_expr_add(
      table, &source_region->begin_byte_offset, &additional_offset, &begin));
  out_region->root_value_id = source_region->root_value_id;
  out_region->alias_scope_id = source_region->alias_scope_id;
  out_region->nullability = source_region->nullability;
  out_region->origin = source_region->origin;
  out_region->base_view_value_id = source_region->base_view_value_id;
  out_region->base_begin_byte_offset = source_region->base_begin_byte_offset;
  IREE_RETURN_IF_ERROR(loom_view_region_expr_add(
      table, &source_region->projection_byte_offset, &additional_offset,
      &out_region->projection_byte_offset));
  int64_t static_additional_offset = 0;
  if (loom_view_region_expr_is_constant(&additional_offset,
                                        &static_additional_offset) &&
      static_additional_offset == 0) {
    out_region->begin_value_id = source_region->begin_value_id;
  }
  out_region->begin_byte_offset = begin;
  loom_view_region_expression_refine_facts(&out_region->begin_byte_offset,
                                           reference.base_byte_offset);
  IREE_RETURN_IF_ERROR(loom_view_region_expr_add(
      table, &out_region->begin_byte_offset, &out_region->byte_length,
      &out_region->end_byte_offset));
  return iree_ok_status();
}

static iree_status_t loom_view_region_build_refine(
    loom_view_region_table_t* table, loom_value_id_t value_id,
    const loom_op_t* op, loom_type_t view_type,
    loom_value_fact_view_reference_t reference,
    loom_view_region_t* out_region) {
  const loom_view_region_t* source_region = NULL;
  IREE_RETURN_IF_ERROR(loom_view_region_get_source(
      table, loom_view_refine_source(op), &source_region));
  IREE_RETURN_IF_ERROR(loom_view_region_build_default(
      table, value_id, view_type, reference, out_region));
  if (!source_region) {
    return iree_ok_status();
  }
  out_region->root_value_id = source_region->root_value_id;
  out_region->alias_scope_id = source_region->alias_scope_id;
  out_region->nullability = source_region->nullability;
  out_region->origin = source_region->origin;
  out_region->base_view_value_id = source_region->base_view_value_id;
  out_region->base_begin_byte_offset = source_region->base_begin_byte_offset;
  out_region->projection_byte_offset = source_region->projection_byte_offset;
  out_region->begin_value_id = source_region->begin_value_id;
  out_region->begin_byte_offset = source_region->begin_byte_offset;
  loom_view_region_expression_refine_facts(&out_region->begin_byte_offset,
                                           reference.base_byte_offset);
  IREE_RETURN_IF_ERROR(loom_view_region_expr_add(
      table, &out_region->begin_byte_offset, &out_region->byte_length,
      &out_region->end_byte_offset));
  return iree_ok_status();
}

static iree_status_t loom_view_region_build_for_value(
    loom_view_region_table_t* table, loom_value_id_t value_id,
    loom_view_region_t* out_region) {
  const loom_module_t* module = table->expression_context->module;
  loom_type_t view_type = loom_module_value_type(module, value_id);
  loom_value_fact_view_reference_t reference =
      loom_view_region_default_reference(value_id, view_type);
  loom_value_facts_t facts = loom_view_region_lookup_facts(table, value_id);
  (void)loom_value_facts_query_view_reference(
      &table->expression_context->fact_table->context, facts, &reference);
  reference.root_value_id =
      loom_value_fact_view_reference_resolve_root_value(reference, value_id);

  const loom_value_t* value = loom_module_value(module, value_id);
  if (loom_value_is_block_arg(value)) {
    return loom_view_region_build_default(table, value_id, view_type, reference,
                                          out_region);
  }
  const loom_op_t* defining_op = loom_value_def_op(value);
  if (!defining_op) {
    return loom_view_region_build_default(table, value_id, view_type, reference,
                                          out_region);
  }
  if (loom_buffer_view_isa(defining_op)) {
    return loom_view_region_build_buffer_view(table, value_id, defining_op,
                                              view_type, reference, out_region);
  }
  if (loom_view_subview_isa(defining_op)) {
    return loom_view_region_build_subview(table, value_id, defining_op,
                                          view_type, reference, out_region);
  }
  if (loom_view_refine_isa(defining_op)) {
    return loom_view_region_build_refine(table, value_id, defining_op,
                                         view_type, reference, out_region);
  }
  return loom_view_region_build_default(table, value_id, view_type, reference,
                                        out_region);
}

iree_status_t loom_view_region_table_get(
    loom_view_region_table_t* table, loom_value_id_t value_id,
    const loom_view_region_t** out_region) {
  const loom_module_t* module = table->expression_context->module;
  IREE_ASSERT(value_id < module->values.count);
  *out_region = NULL;
  const loom_value_ordinal_t value_ordinal =
      loom_local_value_domain_ordinal(table->value_domain, value_id);
  loom_type_t type = loom_module_value_type(module, value_id);
  if (!loom_type_is_view(type)) {
    return iree_ok_status();
  }

  uint8_t state = table->states_by_value_ordinal[value_ordinal];
  if (state == LOOM_VIEW_REGION_VALUE_READY) {
    loom_view_region_id_t region_id =
        table->region_ids_by_value_ordinal[value_ordinal];
    if (region_id != LOOM_VIEW_REGION_ID_INVALID &&
        region_id < table->region_count) {
      *out_region = &table->regions[region_id];
    }
    return iree_ok_status();
  }
  if (state == LOOM_VIEW_REGION_VALUE_VISITING) {
    return iree_ok_status();
  }

  table->states_by_value_ordinal[value_ordinal] =
      LOOM_VIEW_REGION_VALUE_VISITING;
  loom_view_region_t region = {0};
  iree_status_t status =
      loom_view_region_build_for_value(table, value_id, &region);
  if (iree_status_is_ok(status)) {
    loom_view_region_refresh_precision(&region);
    loom_view_region_t* stored_region = NULL;
    status =
        loom_view_region_table_append_region(table, &region, &stored_region);
    if (iree_status_is_ok(status)) {
      table->region_ids_by_value_ordinal[value_ordinal] =
          stored_region->region_id;
      table->states_by_value_ordinal[value_ordinal] =
          LOOM_VIEW_REGION_VALUE_READY;
      *out_region = stored_region;
    }
  }
  if (!iree_status_is_ok(status)) {
    table->states_by_value_ordinal[value_ordinal] =
        LOOM_VIEW_REGION_VALUE_EMPTY;
  }
  return status;
}

bool loom_view_region_table_try_lookup(const loom_view_region_table_t* table,
                                       loom_value_id_t value_id,
                                       const loom_view_region_t** out_region) {
  *out_region = NULL;
  if (value_id >= table->expression_context->module->values.count) {
    return false;
  }
  const loom_value_ordinal_t value_ordinal =
      loom_local_value_domain_try_ordinal(table->value_domain, value_id);
  if (value_ordinal == LOOM_VALUE_ORDINAL_INVALID) {
    return false;
  }
  if (table->states_by_value_ordinal[value_ordinal] !=
      LOOM_VIEW_REGION_VALUE_READY) {
    return false;
  }
  const loom_view_region_id_t region_id =
      table->region_ids_by_value_ordinal[value_ordinal];
  if (region_id == LOOM_VIEW_REGION_ID_INVALID ||
      region_id >= table->region_count) {
    return false;
  }
  *out_region = &table->regions[region_id];
  return true;
}

iree_status_t loom_view_region_table_derive_element_region(
    loom_view_region_table_t* table, loom_value_id_t view_value_id,
    loom_attribute_t static_indices, loom_value_slice_t dynamic_indices,
    loom_view_region_t* out_region, bool* out_derived) {
  *out_region = (loom_view_region_t){0};
  *out_derived = false;
  if (view_value_id >= table->expression_context->module->values.count) {
    return iree_ok_status();
  }

  const loom_view_region_t* source_region = NULL;
  IREE_RETURN_IF_ERROR(
      loom_view_region_table_get(table, view_value_id, &source_region));
  if (!source_region || source_region->static_element_byte_count <= 0) {
    return iree_ok_status();
  }
  const loom_type_t view_type =
      loom_module_value_type(table->expression_context->module, view_value_id);
  if (static_indices.kind != LOOM_ATTR_I64_ARRAY ||
      static_indices.count != loom_type_rank(view_type)) {
    return iree_ok_status();
  }
  iree_host_size_t expected_dynamic_count = 0;
  for (uint16_t i = 0; i < static_indices.count; ++i) {
    expected_dynamic_count += static_indices.i64_array[i] == INT64_MIN;
  }
  if (dynamic_indices.count != expected_dynamic_count) {
    return iree_ok_status();
  }

  loom_symbolic_expr_t additional_offset = {0};
  IREE_RETURN_IF_ERROR(loom_view_region_additional_byte_offset_expr(
      table, view_type, static_indices, dynamic_indices,
      source_region->static_element_byte_count, &additional_offset));
  loom_symbolic_expr_t begin = {0};
  IREE_RETURN_IF_ERROR(loom_view_region_expr_add(
      table, &source_region->begin_byte_offset, &additional_offset, &begin));
  loom_symbolic_expr_t byte_length = {0};
  loom_symbolic_expr_constant(source_region->static_element_byte_count,
                              &byte_length);
  loom_symbolic_expr_t end = {0};
  IREE_RETURN_IF_ERROR(
      loom_view_region_expr_add(table, &begin, &byte_length, &end));
  loom_symbolic_expr_t projection = {0};
  IREE_RETURN_IF_ERROR(
      loom_view_region_expr_add(table, &source_region->projection_byte_offset,
                                &additional_offset, &projection));

  *out_region = *source_region;
  out_region->region_id = LOOM_VIEW_REGION_ID_INVALID;
  out_region->projection_byte_offset = projection;
  out_region->begin_value_id = LOOM_VALUE_ID_INVALID;
  out_region->begin_byte_offset = begin;
  out_region->byte_length = byte_length;
  out_region->end_byte_offset = end;
  out_region->minimum_alignment =
      iree_math_gcd_i64((int64_t)source_region->minimum_alignment,
                        additional_offset.facts.known_divisor);
  out_region->access_flags = 0;
  loom_view_region_refresh_precision(out_region);
  *out_derived = true;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Access derivation
//===----------------------------------------------------------------------===//

static loom_view_access_flags_t loom_view_region_access_flags(
    loom_operand_flags_t operand_flags) {
  loom_view_access_flags_t flags = 0;
  if (iree_any_bit_set(operand_flags, LOOM_OPERAND_READS)) {
    flags |= LOOM_VIEW_ACCESS_READ;
  }
  if (iree_any_bit_set(operand_flags, LOOM_OPERAND_WRITES)) {
    flags |= LOOM_VIEW_ACCESS_WRITE;
  }
  return flags;
}

static uint32_t loom_view_region_overlapping_memory_spaces(
    loom_value_fact_memory_space_t memory_space) {
  uint32_t spaces = 0;
  for (uint32_t i = LOOM_VALUE_FACT_MEMORY_SPACE_UNKNOWN;
       i <= LOOM_VALUE_FACT_MEMORY_SPACE_GENERIC; ++i) {
    if (!loom_view_memory_spaces_are_disjoint(
            memory_space, (loom_value_fact_memory_space_t)i)) {
      spaces |= 1u << i;
    }
  }
  return spaces;
}

static bool loom_view_region_ordering_acquires(loom_attribute_t attribute) {
  if (loom_attr_is_absent(attribute)) {
    return false;
  }
  const loom_atomic_ordering_t ordering = loom_attr_as_enum(attribute);
  return ordering == LOOM_ATOMIC_ORDERING_ACQUIRE ||
         ordering == LOOM_ATOMIC_ORDERING_ACQ_REL ||
         ordering == LOOM_ATOMIC_ORDERING_SEQ_CST;
}

static void loom_view_region_analyze_interference(
    loom_view_region_table_t* table, const loom_op_t* op,
    const loom_op_vtable_t* vtable, loom_trait_flags_t traits) {
  if (iree_any_bit_set(traits, LOOM_TRAIT_UNKNOWN_EFFECTS)) {
    table->interference_memory_spaces = UINT32_MAX;
  }
  if (vtable->memory_access && loom_memory_access_operation_kind_is_atomic(
                                   vtable->memory_access->operation_kind)) {
    const loom_memory_access_t access = {.op = op, .op_vtable = vtable};
    if (loom_attr_as_enum(loom_memory_access_atomic_scope(access)) !=
            LOOM_ATOMIC_SCOPE_THREAD &&
        (loom_view_region_ordering_acquires(
             loom_memory_access_atomic_ordering(access)) ||
         loom_view_region_ordering_acquires(
             loom_memory_access_atomic_success_ordering(access)) ||
         loom_view_region_ordering_acquires(
             loom_memory_access_atomic_failure_ordering(access)))) {
      // Acquisition can import writes to any shared storage, regardless of the
      // address space holding the synchronization token.
      table->interference_memory_spaces |=
          UINT32_MAX & ~(1u << LOOM_VALUE_FACT_MEMORY_SPACE_PRIVATE);
    }
  } else if (iree_any_bit_set(traits, LOOM_TRAIT_MEMORY_FENCE)) {
    if (loom_buffer_fence_isa(op)) {
      if (loom_buffer_fence_scope(op) != LOOM_ATOMIC_SCOPE_THREAD &&
          loom_view_region_ordering_acquires(
              loom_attr_enum(loom_buffer_fence_ordering(op)))) {
        table->interference_memory_spaces |=
            UINT32_MAX & ~(1u << LOOM_VALUE_FACT_MEMORY_SPACE_PRIVATE);
      }
    } else if (loom_kernel_barrier_isa(op)) {
      if (loom_view_region_ordering_acquires(
              loom_attr_enum(loom_kernel_barrier_ordering(op)))) {
        table->interference_memory_spaces |=
            loom_view_region_overlapping_memory_spaces(
                loom_kernel_barrier_memory_space(op));
      }
    } else {
      table->interference_memory_spaces = UINT32_MAX;
    }
  }
}

// Access bits share the existing byte with the value's execution
// correspondence.
enum {
  LOOM_VIEW_STORAGE_INVARIANT = 1u << 2,
  // Temporary incoming-payload proof, consumed before type dependencies refine
  // it.
  LOOM_VIEW_STORAGE_JOIN_INVARIANT = 1u << 3,
};

typedef enum loom_view_region_analysis_phase_e {
  LOOM_VIEW_REGION_ANALYZE_MEMORY,
  LOOM_VIEW_REGION_ANALYZE_STABILITY,
} loom_view_region_analysis_phase_t;

static bool loom_view_region_value_is_invariant(
    const loom_view_region_table_t* table, loom_value_id_t value_id) {
  const loom_value_ordinal_t ordinal =
      loom_local_value_domain_try_ordinal(table->value_domain, value_id);
  return ordinal >= table->value_domain->definition_count ||
         iree_any_bit_set(table->storage_flags_by_value_ordinal[ordinal],
                          LOOM_VIEW_STORAGE_INVARIANT);
}

loom_view_access_flags_t loom_view_region_table_root_access_flags(
    const loom_view_region_table_t* table, loom_value_id_t root_value_id) {
  const loom_value_ordinal_t ordinal =
      loom_local_value_domain_try_ordinal(table->value_domain, root_value_id);
  return ordinal == LOOM_VALUE_ORDINAL_INVALID
             ? 0
             : table->storage_flags_by_value_ordinal[ordinal] &
                   (LOOM_VIEW_ACCESS_READ | LOOM_VIEW_ACCESS_WRITE);
}

static void loom_view_region_add_root_access(
    loom_view_region_table_t* table, loom_value_id_t root_value_id,
    loom_value_fact_alias_scope_id_t alias_scope_id,
    loom_value_fact_memory_space_t memory_space,
    loom_view_access_flags_t flags) {
  const loom_value_ordinal_t ordinal =
      loom_local_value_domain_try_ordinal(table->value_domain, root_value_id);
  if (ordinal != LOOM_VALUE_ORDINAL_INVALID) {
    table->storage_flags_by_value_ordinal[ordinal] |= flags;
  }
  if (iree_any_bit_set(flags, LOOM_VIEW_ACCESS_WRITE) &&
      (ordinal == LOOM_VALUE_ORDINAL_INVALID ||
       alias_scope_id == LOOM_VALUE_FACT_ALIAS_SCOPE_ID_NONE)) {
    table->interference_memory_spaces |=
        loom_view_region_overlapping_memory_spaces(memory_space);
  } else if (iree_any_bit_set(flags, LOOM_VIEW_ACCESS_WRITE) &&
             !loom_view_region_value_is_invariant(table, root_value_id)) {
    table->varying_root_write_memory_spaces |=
        loom_view_region_overlapping_memory_spaces(memory_space);
  }
}

static iree_status_t loom_view_region_analyze_operand_access(
    loom_view_region_table_t* table, loom_value_id_t operand,
    loom_view_access_flags_t flags) {
  const loom_view_region_t* region = NULL;
  IREE_RETURN_IF_ERROR(loom_view_region_table_get(table, operand, &region));
  if (region) {
    table->regions[region->region_id].access_flags |= flags;
    loom_view_region_add_root_access(table, region->root_value_id,
                                     region->alias_scope_id,
                                     region->memory_space, flags);
  } else {
    // Raw byte accesses and buffer aliases participate in the same root proof.
    loom_value_fact_buffer_reference_t reference = {
        .root_value_id = operand,
        .alias_scope_id = LOOM_VALUE_FACT_ALIAS_SCOPE_ID_NONE,
        .memory_space = LOOM_VALUE_FACT_MEMORY_SPACE_UNKNOWN,
    };
    (void)loom_value_facts_query_buffer_reference(
        &table->expression_context->fact_table->context,
        loom_view_region_lookup_facts(table, operand), &reference);
    loom_view_region_add_root_access(
        table,
        loom_value_fact_buffer_reference_resolve_root_value(reference, operand),
        reference.alias_scope_id, reference.memory_space, flags);
  }
  return iree_ok_status();
}

static iree_status_t loom_view_region_table_analyze_op_memory(
    loom_view_region_table_t* table, const loom_op_t* op,
    const loom_op_vtable_t* vtable, loom_trait_flags_t traits,
    loom_view_region_analysis_phase_t phase) {
  if (!vtable) {
    table->interference_memory_spaces = UINT32_MAX;
    return iree_ok_status();
  }
  if (phase == LOOM_VIEW_REGION_ANALYZE_MEMORY) {
    loom_view_region_analyze_interference(table, op, vtable, traits);
  }

  if (phase == LOOM_VIEW_REGION_ANALYZE_MEMORY && vtable->memory_access) {
    const loom_memory_access_t access = {
        .op = op,
        .op_vtable = vtable,
    };
    const loom_value_slice_t dynamic_indices =
        loom_memory_access_dynamic_indices(access);
    for (uint16_t i = 0; i < dynamic_indices.count; ++i) {
      const loom_value_id_t value_id = dynamic_indices.values[i];
      int64_t exact_value = 0;
      if (loom_view_region_facts_exact_i64(
              loom_view_region_lookup_facts(table, value_id), &exact_value)) {
        continue;
      }
      loom_symbolic_expr_t expression = {0};
      IREE_RETURN_IF_ERROR(loom_symbolic_expr_from_value(
          table->expression_context, value_id, &expression));
    }
  }

  bool has_described_write = false;
  if (vtable->operand_descriptors) {
    const uint8_t descriptor_count =
        loom_op_vtable_operand_descriptor_count(vtable);
    for (uint8_t i = 0; i < descriptor_count; ++i) {
      const loom_view_access_flags_t flags =
          loom_view_region_access_flags(vtable->operand_descriptors[i].flags) &
          (phase == LOOM_VIEW_REGION_ANALYZE_MEMORY
               ? LOOM_VIEW_ACCESS_READ | LOOM_VIEW_ACCESS_WRITE
               : LOOM_VIEW_ACCESS_WRITE);
      if (!flags) {
        continue;
      }
      has_described_write |= iree_any_bit_set(flags, LOOM_VIEW_ACCESS_WRITE);
      const loom_value_slice_t operands =
          loom_op_operand_field_span(vtable, op, i);
      for (uint16_t j = 0; j < operands.count; ++j) {
        IREE_RETURN_IF_ERROR(loom_view_region_analyze_operand_access(
            table, operands.values[j], flags));
      }
    }
  }
  if (!has_described_write &&
      iree_any_bit_set(traits, LOOM_TRAIT_WRITES_MEMORY)) {
    table->interference_memory_spaces = UINT32_MAX;
  }
  return iree_ok_status();
}

static bool loom_view_region_repeated_op_is_invariant(
    loom_view_region_table_t* table, const loom_op_t* op,
    const loom_op_vtable_t* vtable, loom_trait_flags_t traits) {
  if (op->region_count ||
      iree_any_bit_set(traits, LOOM_TRAIT_CONVERGENT |
                                   LOOM_TRAIT_NON_DETERMINISTIC |
                                   LOOM_TRAIT_UNIQUE_IDENTITY |
                                   LOOM_TRAIT_UNKNOWN_EFFECTS)) {
    return false;
  }
  if (!iree_any_bit_set(traits, LOOM_TRAIT_PURE)) {
    if (!vtable || !vtable->memory_access ||
        vtable->memory_access->operation_kind !=
            LOOM_MEMORY_ACCESS_OPERATION_LOAD) {
      return false;
    }
    const loom_memory_access_t access = {.op = op, .op_vtable = vtable};
    const loom_value_id_t source_value = loom_memory_access_view(access);
    const loom_view_region_t* source = NULL;
    loom_value_id_t root;
    loom_value_fact_alias_scope_id_t alias_scope;
    loom_value_fact_memory_space_t memory_space;
    if (loom_view_region_table_try_lookup(table, source_value, &source)) {
      root = source->root_value_id;
      alias_scope = source->alias_scope_id;
      memory_space = source->memory_space;
    } else {
      loom_value_fact_buffer_reference_t reference;
      if (!loom_value_facts_query_buffer_reference(
              &table->expression_context->fact_table->context,
              loom_view_region_lookup_facts(table, source_value), &reference)) {
        return false;
      }
      root = loom_value_fact_buffer_reference_resolve_root_value(reference,
                                                                 source_value);
      alias_scope = reference.alias_scope_id;
      memory_space = reference.memory_space;
    }
    // Access aggregation is complete. A fixed root can use same-execution
    // noalias against every written root, including roots that still vary.
    if (!loom_view_region_value_is_invariant(table, root) ||
        !loom_view_region_table_root_is_stable(table, root, alias_scope,
                                               memory_space)) {
      return false;
    }
  }
  const loom_value_id_t* operands = loom_op_const_operands(op);
  for (uint16_t i = 0; i < op->operand_count; ++i) {
    if (!loom_view_region_value_is_invariant(table, operands[i])) {
      return false;
    }
  }
  const loom_module_t* module = table->expression_context->module;
  const uint32_t* attribute_owners = loom_op_attribute_owners(op);
  for (uint8_t i = 0; i < op->attribute_count; ++i) {
    if (!attribute_owners[i]) {
      continue;
    }
    loom_type_use_iterator_t dependencies;
    loom_attribute_dependencies_begin(&module->type_uses, op, i, &dependencies);
    for (loom_value_id_t provider = loom_type_dependencies_next(&dependencies);
         provider != LOOM_VALUE_ID_INVALID;
         provider = loom_type_dependencies_next(&dependencies)) {
      if (!loom_view_region_value_is_invariant(table, provider)) {
        return false;
      }
    }
  }
  return true;
}

static void loom_view_region_mark_invariant(loom_view_region_table_t* table,
                                            loom_value_id_t value_id) {
  table->storage_flags_by_value_ordinal[loom_local_value_domain_ordinal(
      table->value_domain, value_id)] |= LOOM_VIEW_STORAGE_INVARIANT;
}

static void loom_view_region_classify_repeated_value(
    loom_view_region_table_t* table, loom_value_id_t value_id, bool invariant) {
  const loom_module_t* module = table->expression_context->module;
  const loom_value_fact_table_t* facts = table->expression_context->fact_table;
  if (!invariant) {
    const loom_value_id_t identity =
        loom_value_fact_table_query_identity(facts, value_id);
    int64_t exact_value = 0;
    invariant =
        (identity != value_id &&
         loom_view_region_value_is_invariant(table, identity)) ||
        loom_value_facts_as_exact_i64(
            loom_value_fact_table_lookup(facts, value_id), &exact_value);
  }
  if (invariant) {
    loom_type_use_iterator_t dependencies;
    loom_module_value_type_dependencies(module, value_id, &dependencies);
    for (loom_value_id_t provider = loom_type_dependencies_next(&dependencies);
         provider != LOOM_VALUE_ID_INVALID;
         provider = loom_type_dependencies_next(&dependencies)) {
      if (provider != value_id &&
          !loom_view_region_value_is_invariant(table, provider)) {
        invariant = false;
        break;
      }
    }
  }
  if (invariant) {
    loom_view_region_mark_invariant(table, value_id);
  }
}

// Refines the temporary proof for each argument over a contiguous incoming
// range. A varying choice is harmless when every alternative carries the same
// invariant value. Fixed choices may carry distinct invariant values.
static void loom_view_region_refine_join_arguments(
    loom_view_region_table_t* table, const loom_cfg_graph_t* graph,
    const loom_block_t* block, loom_cfg_edge_index_span_t incoming) {
  const loom_value_id_t* first_arguments = NULL;
  uint16_t first_argument_count = 0;
  const bool first_known = loom_cfg_terminator_payload_for_successor(
      graph->edges[incoming.values[0]].terminator, block, &first_arguments,
      &first_argument_count);
  for (uint16_t i = 0; i < block->arg_count; ++i) {
    uint8_t* flags =
        &table->storage_flags_by_value_ordinal[loom_local_value_domain_ordinal(
            table->value_domain, loom_block_arg_id(block, i))];
    if (!iree_any_bit_set(*flags, LOOM_VIEW_STORAGE_JOIN_INVARIANT)) {
      continue;
    }
    bool invariant = first_known;
    loom_value_id_t first_identity = LOOM_VALUE_ID_INVALID;
    for (iree_host_size_t j = 0; invariant && j < incoming.count; ++j) {
      const loom_value_id_t* arguments = first_arguments;
      uint16_t argument_count = first_argument_count;
      if (j && !loom_cfg_terminator_payload_for_successor(
                   graph->edges[incoming.values[j]].terminator, block,
                   &arguments, &argument_count)) {
        invariant = false;
        break;
      }
      invariant = loom_view_region_value_is_invariant(table, arguments[i]);
      if (invariant && incoming.count > 1) {
        const loom_value_id_t identity = loom_value_fact_table_query_identity(
            table->expression_context->fact_table, arguments[i]);
        if (!j) {
          first_identity = identity;
        } else {
          invariant = identity == first_identity;
        }
      }
    }
    if (!invariant) {
      *flags &= ~LOOM_VIEW_STORAGE_JOIN_INVARIANT;
    }
  }
}

// Incoming sources are ordered by dominance preorder. Regions on their paths
// to the common dominator belong to this join's proof only when it is their
// unique continuation. Already checked prefixes and complete varying-choice
// groups are skipped, so qualifying regions and incoming edges are visited
// once.
static bool loom_view_region_classify_join_arguments(
    loom_view_region_table_t* table,
    const loom_value_fact_cfg_region_t* structure, const loom_block_t* block) {
  const loom_cfg_graph_t* graph = &structure->graph;
  const uint16_t block_index = block->region_index;
  const iree_host_size_t incoming_start =
      structure->regions.incoming_offsets[block_index];
  const iree_host_size_t incoming_count =
      structure->regions.incoming_offsets[block_index + 1] - incoming_start;
  if (!incoming_count || graph->blocks[block_index].is_dfs_backedge_target) {
    return false;
  }
  const uint16_t dominator =
      structure->dominance.immediate_dominators[block_index];
  if (dominator == LOOM_CFG_DOMINATOR_INVALID || dominator == block_index) {
    return false;
  }
  const loom_cfg_edge_index_t* incoming_edges =
      structure->regions.incoming_edges + incoming_start;
  for (uint16_t i = 0; i < block->arg_count; ++i) {
    table->storage_flags_by_value_ordinal[loom_local_value_domain_ordinal(
        table->value_domain, loom_block_arg_id(block, i))] |=
        LOOM_VIEW_STORAGE_JOIN_INVARIANT;
  }
  uint16_t previous_source = LOOM_CFG_DOMINATOR_INVALID;
  for (iree_host_size_t i = 0; i < incoming_count;) {
    const loom_cfg_edge_info_t* edge = &graph->edges[incoming_edges[i]];
    uint16_t varying_choice =
        edge->selector_value_id != LOOM_VALUE_ID_INVALID &&
                !loom_view_region_value_is_invariant(table,
                                                     edge->selector_value_id)
            ? edge->source_block_index
            : LOOM_CFG_DOMINATOR_INVALID;
    uint16_t current = edge->source_block_index;
    while (current != dominator &&
           (previous_source == LOOM_CFG_DOMINATOR_INVALID ||
            !loom_cfg_dominance_block_dominates(&structure->dominance, current,
                                                previous_source))) {
      if (structure->regions.blocks[current].continuation_index !=
          block_index) {
        return false;
      }
      const uint16_t parent =
          structure->dominance.immediate_dominators[current];
      // Reconvergence executes after either parent alternative. Otherwise the
      // unique entry predecessor owns the decision to enter this subtree.
      if (structure->control_structure.postdominance.nodes[parent]
              .immediate_postdominator != current) {
        if (structure->dominance.entry_predecessors[current] != parent) {
          varying_choice = parent;
        } else {
          const loom_cfg_edge_index_span_t entries =
              loom_cfg_graph_predecessor_edges(graph, current);
          for (iree_host_size_t j = 0; j < entries.count; ++j) {
            const loom_cfg_edge_info_t* entry =
                &graph->edges[entries.values[j]];
            if (entry->source_block_index == parent &&
                entry->selector_value_id != LOOM_VALUE_ID_INVALID &&
                !loom_view_region_value_is_invariant(
                    table, entry->selector_value_id)) {
              varying_choice = parent;
            }
          }
        }
      }
      current = parent;
    }
    loom_cfg_edge_index_span_t group = {
        .values = incoming_edges + i,
        .count = 1,
    };
    if (varying_choice != LOOM_CFG_DOMINATOR_INVALID) {
      group = varying_choice == dominator
                  ? (loom_cfg_edge_index_span_t){.values = incoming_edges,
                                                 .count = incoming_count}
                  : structure->regions.blocks[varying_choice].exit_edges;
      if (group.values != incoming_edges + i) {
        return false;
      }
    }
    loom_view_region_refine_join_arguments(table, graph, block, group);
    i += group.count;
    previous_source = graph->edges[incoming_edges[i - 1]].source_block_index;
  }
  return true;
}

static iree_status_t loom_view_region_table_analyze_region(
    loom_view_region_table_t* table, const loom_region_t* region,
    loom_view_region_analysis_phase_t phase, bool repeated);

static iree_status_t loom_view_region_table_analyze_op_tree(
    loom_view_region_table_t* table, const loom_op_t* op,
    loom_view_region_analysis_phase_t phase, bool repeated) {
  const loom_op_vtable_t* vtable =
      loom_op_vtable(table->expression_context->module, op);
  const loom_trait_flags_t traits =
      loom_op_effective_traits(table->expression_context->module, op);
  const bool refine = repeated && op->result_count &&
                      phase == LOOM_VIEW_REGION_ANALYZE_STABILITY;
  const bool invariant = refine && loom_view_region_repeated_op_is_invariant(
                                       table, op, vtable, traits);
  const loom_value_id_t* results = loom_op_const_results(op);
  for (uint16_t i = 0; i < op->result_count; ++i) {
    if (!repeated) {
      loom_view_region_mark_invariant(table, results[i]);
    } else if (refine) {
      loom_view_region_classify_repeated_value(table, results[i], invariant);
    }
    if (phase == LOOM_VIEW_REGION_ANALYZE_MEMORY) {
      const loom_view_region_t* region = NULL;
      IREE_RETURN_IF_ERROR(
          loom_view_region_table_get(table, results[i], &region));
    }
  }
  IREE_RETURN_IF_ERROR(loom_view_region_table_analyze_op_memory(
      table, op, vtable, traits, phase));
  loom_region_t* const* regions = loom_op_regions(op);
  for (uint8_t i = 0; i < op->region_count; ++i) {
    if (regions[i]) {
      IREE_RETURN_IF_ERROR(loom_view_region_table_analyze_region(
          table, regions[i], phase,
          repeated || !vtable || !vtable->region_branch));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_view_region_table_analyze_region(
    loom_view_region_table_t* table, const loom_region_t* region,
    loom_view_region_analysis_phase_t phase, bool repeated) {
  if (!region) {
    return iree_ok_status();
  }
  const loom_value_fact_cfg_region_t* structure =
      loom_value_fact_table_lookup_cfg_region(
          table->expression_context->fact_table, region);
  const loom_cfg_graph_t* graph = structure ? &structure->graph : NULL;
  uint16_t unreachable_index = 0;
  for (uint16_t i = 0; i < region->block_count; ++i) {
    uint16_t block_index = i;
    if (graph && i < graph->reverse_postorder.count) {
      block_index = graph->reverse_postorder.values[i];
    } else if (graph) {
      while (graph->blocks[unreachable_index].reachable) {
        ++unreachable_index;
      }
      block_index = unreachable_index++;
    }
    const loom_block_t* block = loom_region_const_block(region, block_index);
    const bool block_repeated =
        repeated || (graph && (!graph->blocks[block_index].reachable ||
                               graph->blocks[block_index].component_is_cyclic));
    const bool join_analyzed =
        block_repeated && block->arg_count && structure &&
        phase == LOOM_VIEW_REGION_ANALYZE_STABILITY &&
        loom_view_region_classify_join_arguments(table, structure, block);
    for (uint16_t j = 0; j < block->arg_count; ++j) {
      const loom_value_id_t argument = loom_block_arg_id(block, j);
      if (!block_repeated) {
        loom_view_region_mark_invariant(table, argument);
      } else if (phase == LOOM_VIEW_REGION_ANALYZE_STABILITY) {
        uint8_t* flags = &table->storage_flags_by_value_ordinal
                              [loom_local_value_domain_ordinal(
                                  table->value_domain, argument)];
        const bool invariant =
            join_analyzed &&
            iree_any_bit_set(*flags, LOOM_VIEW_STORAGE_JOIN_INVARIANT);
        *flags &= ~LOOM_VIEW_STORAGE_JOIN_INVARIANT;
        loom_view_region_classify_repeated_value(table, argument, invariant);
      }
      if (phase == LOOM_VIEW_REGION_ANALYZE_MEMORY) {
        const loom_view_region_t* view_region = NULL;
        IREE_RETURN_IF_ERROR(
            loom_view_region_table_get(table, argument, &view_region));
      }
    }
    const loom_op_t* op = NULL;
    loom_block_for_each_op(block, op) {
      IREE_RETURN_IF_ERROR(loom_view_region_table_analyze_op_tree(
          table, op, phase, block_repeated));
    }
  }
  return iree_ok_status();
}

iree_status_t loom_view_region_table_analyze(loom_view_region_table_t* table) {
  IREE_RETURN_IF_ERROR(loom_view_region_table_analyze_region(
      table, table->value_domain->region, LOOM_VIEW_REGION_ANALYZE_MEMORY,
      false));
  if (table->varying_root_write_memory_spaces) {
    table->varying_root_write_memory_spaces = 0;
    IREE_RETURN_IF_ERROR(loom_view_region_table_analyze_region(
        table, table->value_domain->region, LOOM_VIEW_REGION_ANALYZE_STABILITY,
        false));
  }
  return iree_ok_status();
}

bool loom_view_region_table_root_is_stable(
    const loom_view_region_table_t* table, loom_value_id_t root_value_id,
    loom_value_fact_alias_scope_id_t alias_scope_id,
    loom_value_fact_memory_space_t memory_space) {
  const loom_value_ordinal_t ordinal =
      loom_local_value_domain_try_ordinal(table->value_domain, root_value_id);
  const uint8_t flags = ordinal == LOOM_VALUE_ORDINAL_INVALID
                            ? 0
                            : table->storage_flags_by_value_ordinal[ordinal];
  if (!iree_any_bit_set(flags, LOOM_VIEW_ACCESS_READ) ||
      iree_any_bit_set(flags, LOOM_VIEW_ACCESS_WRITE)) {
    return false;
  }
  return memory_space == LOOM_VALUE_FACT_MEMORY_SPACE_CONSTANT ||
         (alias_scope_id != LOOM_VALUE_FACT_ALIAS_SCOPE_ID_NONE &&
          !(table->interference_memory_spaces & (1u << memory_space)) &&
          (ordinal >= table->value_domain->definition_count ||
           iree_any_bit_set(flags, LOOM_VIEW_STORAGE_INVARIANT) ||
           !(table->varying_root_write_memory_spaces & (1u << memory_space))));
}

bool loom_view_memory_spaces_are_disjoint(
    loom_value_fact_memory_space_t left, loom_value_fact_memory_space_t right) {
  if (left == right || left == LOOM_VALUE_FACT_MEMORY_SPACE_UNKNOWN ||
      right == LOOM_VALUE_FACT_MEMORY_SPACE_UNKNOWN ||
      left == LOOM_VALUE_FACT_MEMORY_SPACE_GENERIC ||
      right == LOOM_VALUE_FACT_MEMORY_SPACE_GENERIC) {
    return false;
  }
  return left == LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP ||
         right == LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP ||
         left == LOOM_VALUE_FACT_MEMORY_SPACE_PRIVATE ||
         right == LOOM_VALUE_FACT_MEMORY_SPACE_PRIVATE;
}

iree_status_t loom_view_regions_prove_no_overlap(
    loom_view_region_table_t* table, const loom_view_region_t* left_region,
    const loom_view_region_t* right_region, bool* out_no_overlap) {
  *out_no_overlap = false;
  if (!left_region || !right_region) {
    return iree_ok_status();
  }
  if (loom_value_fact_reference_origins_are_disjoint(left_region->origin,
                                                     right_region->origin)) {
    *out_no_overlap = true;
    return iree_ok_status();
  }
  if (left_region->root_value_id == LOOM_VALUE_ID_INVALID ||
      right_region->root_value_id == LOOM_VALUE_ID_INVALID) {
    return iree_ok_status();
  }
  if (left_region->root_value_id != right_region->root_value_id) {
    if (loom_view_memory_spaces_are_disjoint(left_region->memory_space,
                                             right_region->memory_space) ||
        (left_region->alias_scope_id != LOOM_VALUE_FACT_ALIAS_SCOPE_ID_NONE &&
         right_region->alias_scope_id != LOOM_VALUE_FACT_ALIAS_SCOPE_ID_NONE &&
         left_region->alias_scope_id != right_region->alias_scope_id)) {
      *out_no_overlap = true;
    }
    return iree_ok_status();
  }

  loom_symbolic_proof_result_t proof = LOOM_SYMBOLIC_PROOF_UNKNOWN;
  IREE_RETURN_IF_ERROR(loom_symbolic_expr_prove_le(
      table->expression_context, &left_region->end_byte_offset,
      &right_region->begin_byte_offset, &proof));
  if (proof == LOOM_SYMBOLIC_PROOF_TRUE) {
    *out_no_overlap = true;
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_symbolic_expr_prove_le(
      table->expression_context, &right_region->end_byte_offset,
      &left_region->begin_byte_offset, &proof));
  if (proof == LOOM_SYMBOLIC_PROOF_TRUE) {
    *out_no_overlap = true;
  } else if (loom_symbolic_expr_is_constant(&left_region->byte_length) &&
             loom_symbolic_expr_is_constant(&right_region->byte_length) &&
             left_region->byte_length.constant > 0 &&
             right_region->byte_length.constant > 0) {
    // Periodic placement can separate intervals without fixing their order:
    // complementary banks exchange positions on each iteration. Overlap would
    // require right.begin-left.begin in [1-right.length, left.length-1].
    *out_no_overlap = loom_symbolic_congruence_excludes_difference(
        &right_region->begin_byte_offset, &left_region->begin_byte_offset,
        1 - right_region->byte_length.constant,
        left_region->byte_length.constant - 1);
  }
  return iree_ok_status();
}
