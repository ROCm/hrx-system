// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/view_regions.h"

#include <string.h>

#include "loom/ir/attribute.h"
#include "loom/ir/context.h"
#include "loom/ops/buffer/ops.h"
#include "loom/ops/encoding/ops.h"
#include "loom/ops/encoding/storage.h"
#include "loom/ops/view/ops.h"
#include "loom/util/math.h"

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
        table->arena, table->region_count, table->region_count + 1,
        sizeof(*table->regions), &table->region_capacity, &regions));
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
    loom_value_fact_table_t* fact_table,
    const loom_local_value_domain_t* value_domain,
    iree_arena_allocator_t* arena, loom_view_region_table_t* out_table) {
  IREE_ASSERT(loom_local_value_domain_is_acquired(value_domain));
  memset(out_table, 0, sizeof(*out_table));
  out_table->module = value_domain->module;
  out_table->fact_table = fact_table;
  out_table->fact_context.table = fact_table;
  out_table->arena = arena;
  out_table->value_domain = value_domain;
  loom_symbolic_expr_context_initialize(value_domain->module, fact_table, arena,
                                        &out_table->expression_context);
  const iree_host_size_t value_count = value_domain->value_count;
  if (value_count == 0) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, value_count, sizeof(*out_table->region_ids_by_value_ordinal),
      (void**)&out_table->region_ids_by_value_ordinal));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, value_count, sizeof(*out_table->states_by_value_ordinal),
      (void**)&out_table->states_by_value_ordinal));
  for (iree_host_size_t i = 0; i < value_count; ++i) {
    out_table->region_ids_by_value_ordinal[i] = LOOM_VIEW_REGION_ID_INVALID;
  }
  memset(out_table->states_by_value_ordinal, 0,
         value_count * sizeof(*out_table->states_by_value_ordinal));
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Expression helpers
//===----------------------------------------------------------------------===//

static loom_value_facts_t loom_view_region_lookup_facts(
    const loom_view_region_table_t* table, loom_value_id_t value_id) {
  if (!table->fact_table) return loom_value_facts_unknown();
  return loom_value_fact_table_lookup(table->fact_table, value_id);
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
  if (loom_value_facts_is_unknown(facts)) return;
  int64_t exact_value = 0;
  if (loom_symbolic_expr_is_constant(expression) &&
      loom_view_region_facts_exact_i64(facts, &exact_value)) {
    if (expression->constant == exact_value) expression->facts = facts;
    return;
  }
  if (loom_symbolic_expr_is_constant(expression)) return;
  if (loom_value_facts_is_unknown(expression->facts) ||
      loom_value_facts_is_exact(facts)) {
    expression->facts = facts;
  }
}

static void loom_view_region_constant_from_facts_or_value(
    loom_value_facts_t facts, int64_t fallback_value,
    loom_symbolic_expr_t* out_expression) {
  int64_t exact_value = 0;
  if (loom_view_region_facts_exact_i64(facts, &exact_value)) {
    loom_symbolic_expr_constant(exact_value, out_expression);
    out_expression->facts = facts;
    return;
  }
  loom_symbolic_expr_constant(fallback_value, out_expression);
  loom_view_region_expression_refine_facts(out_expression, facts);
}

static bool loom_view_region_expr_is_constant(
    const loom_symbolic_expr_t* expression, int64_t* out_value) {
  if (!loom_symbolic_expr_is_constant(expression)) return false;
  *out_value = expression->constant;
  return true;
}

static iree_status_t loom_view_region_expr_add(
    loom_view_region_table_t* table, const loom_symbolic_expr_t* left,
    const loom_symbolic_expr_t* right, loom_symbolic_expr_t* out_expression) {
  return loom_symbolic_expr_add(&table->expression_context, left, right,
                                out_expression);
}

static iree_status_t loom_view_region_expr_sub(
    loom_view_region_table_t* table, const loom_symbolic_expr_t* left,
    const loom_symbolic_expr_t* right, loom_symbolic_expr_t* out_expression) {
  return loom_symbolic_expr_sub(&table->expression_context, left, right,
                                out_expression);
}

static iree_status_t loom_view_region_expr_mul(
    loom_view_region_table_t* table, const loom_symbolic_expr_t* left,
    const loom_symbolic_expr_t* right, loom_symbolic_expr_t* out_expression) {
  int64_t left_constant = 0;
  int64_t right_constant = 0;
  if (loom_view_region_expr_is_constant(left, &left_constant)) {
    return loom_symbolic_expr_mul_i64(&table->expression_context, right,
                                      left_constant, out_expression);
  }
  if (loom_view_region_expr_is_constant(right, &right_constant)) {
    return loom_symbolic_expr_mul_i64(&table->expression_context, left,
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
  return loom_symbolic_expr_from_value(&table->expression_context,
                                       loom_type_dim_value_id_at(type, axis),
                                       out_expression);
}

static int64_t loom_view_region_static_element_byte_count(loom_type_t type) {
  int32_t bit_count = loom_scalar_type_bitwidth(loom_type_element_type(type));
  if (bit_count <= 0 || (bit_count % 8) != 0) return -1;
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
      &table->fact_context, table->module, view_type,
      out_layout->static_strides, IREE_ARRAYSIZE(out_layout->static_strides),
      &out_layout->summary);
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
      if (dynamic_ordinal >= dynamic_values.count) return iree_ok_status();
      IREE_RETURN_IF_ERROR(loom_symbolic_expr_from_value(
          &table->expression_context, dynamic_values.values[dynamic_ordinal],
          out_expression));
      *out_known = true;
      return iree_ok_status();
    }
    ++dynamic_ordinal;
  }
  return iree_ok_status();
}

static iree_status_t loom_view_region_direct_strided_stride_expr(
    loom_view_region_table_t* table, loom_type_t view_type, uint8_t axis,
    loom_symbolic_expr_t* out_expression, bool* out_known) {
  *out_known = false;
  if (!loom_type_has_ssa_encoding(view_type)) return iree_ok_status();
  loom_value_id_t layout_value_id =
      (loom_value_id_t)loom_type_encoding_value_id(view_type);
  if (layout_value_id >= table->module->values.count) return iree_ok_status();
  const loom_value_t* layout_value =
      loom_module_value(table->module, layout_value_id);
  if (loom_value_is_block_arg(layout_value)) return iree_ok_status();
  const loom_op_t* op = loom_value_def_op(layout_value);
  if (!op || !loom_encoding_layout_strided_isa(op)) return iree_ok_status();
  return loom_view_region_static_or_dynamic_expr(
      table, loom_encoding_layout_strided_static_strides(op),
      loom_encoding_layout_strided_strides(op), axis, out_expression,
      out_known);
}

static iree_status_t loom_view_region_dense_axis_stride_expr(
    loom_view_region_table_t* table, loom_type_t view_type, uint8_t axis,
    loom_symbolic_expr_t* out_expression, bool* out_known) {
  *out_known = false;
  if (axis >= loom_type_rank(view_type)) return iree_ok_status();
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
  bool direct_known = false;
  IREE_RETURN_IF_ERROR(loom_view_region_direct_strided_stride_expr(
      table, view_type, axis, out_expression, &direct_known));
  if (direct_known) {
    loom_view_region_expression_refine_facts(out_expression,
                                             layout.summary.strides[axis]);
    *out_known = true;
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
    if (!loom_symbolic_expr_is_linear(out_expression)) return iree_ok_status();
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
      .alias_scope_id = LOOM_VALUE_FACT_ALIAS_SCOPE_ID_NONE,
      .nullability = LOOM_VALUE_FACT_REFERENCE_NULLABILITY_UNKNOWN,
  };
}

static iree_status_t loom_view_region_build_default(
    loom_view_region_table_t* table, loom_value_id_t value_id,
    loom_type_t view_type, loom_value_fact_view_reference_t reference,
    loom_view_region_t* out_region) {
  loom_symbolic_expr_t begin = {0};
  loom_view_region_constant_from_facts_or_value(reference.base_byte_offset, 0,
                                                &begin);

  loom_symbolic_expr_t length = {0};
  IREE_RETURN_IF_ERROR(loom_view_region_footprint_expr(
      table, view_type, reference.static_element_byte_count, &length));
  loom_view_region_expression_refine_facts(&length,
                                           reference.footprint_byte_length);

  loom_symbolic_expr_t end = {0};
  IREE_RETURN_IF_ERROR(loom_view_region_expr_add(table, &begin, &length, &end));

  *out_region = (loom_view_region_t){
      .region_id = LOOM_VIEW_REGION_ID_INVALID,
      .view_value_id = value_id,
      .root_value_id = reference.root_value_id,
      .begin_byte_offset = begin,
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
  IREE_RETURN_IF_ERROR(loom_symbolic_expr_from_value(
      &table->expression_context, loom_buffer_view_byte_offset(op),
      &out_region->begin_byte_offset));
  loom_view_region_expression_refine_facts(&out_region->begin_byte_offset,
                                           reference.base_byte_offset);
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
  if (!source_region) return iree_ok_status();

  loom_type_t source_type =
      loom_module_value_type(table->module, loom_view_subview_source(op));
  loom_symbolic_expr_t additional_offset = {0};
  IREE_RETURN_IF_ERROR(loom_view_region_additional_byte_offset_expr(
      table, source_type, loom_view_subview_static_offsets(op),
      loom_view_subview_offsets(op), reference.static_element_byte_count,
      &additional_offset));
  loom_symbolic_expr_t begin = {0};
  IREE_RETURN_IF_ERROR(loom_view_region_expr_add(
      table, &source_region->begin_byte_offset, &additional_offset, &begin));
  out_region->root_value_id = source_region->root_value_id;
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
  if (!source_region) return iree_ok_status();
  out_region->root_value_id = source_region->root_value_id;
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
  loom_type_t view_type = loom_module_value_type(table->module, value_id);
  loom_value_fact_view_reference_t reference =
      loom_view_region_default_reference(value_id, view_type);
  loom_value_facts_t facts = loom_view_region_lookup_facts(table, value_id);
  (void)loom_value_facts_query_view_reference(&table->fact_context, facts,
                                              &reference);

  const loom_value_t* value = loom_module_value(table->module, value_id);
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
  IREE_ASSERT(value_id < table->module->values.count);
  *out_region = NULL;
  const loom_value_ordinal_t value_ordinal =
      loom_local_value_domain_ordinal(table->value_domain, value_id);
  loom_type_t type = loom_module_value_type(table->module, value_id);
  if (!loom_type_is_view(type)) return iree_ok_status();

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
  if (state == LOOM_VIEW_REGION_VALUE_VISITING) return iree_ok_status();

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

//===----------------------------------------------------------------------===//
// Access derivation
//===----------------------------------------------------------------------===//

static void loom_view_region_add_access(loom_view_region_t* region,
                                        loom_operand_flags_t operand_flags) {
  if (iree_any_bit_set(operand_flags, LOOM_OPERAND_READS)) {
    region->access_flags |= LOOM_VIEW_ACCESS_READ;
  }
  if (iree_any_bit_set(operand_flags, LOOM_OPERAND_WRITES)) {
    region->access_flags |= LOOM_VIEW_ACCESS_WRITE;
  }
}

static iree_status_t loom_view_region_table_record_op_accesses(
    loom_view_region_table_t* table, const loom_op_t* op) {
  const loom_op_vtable_t* vtable = loom_op_vtable(table->module, op);
  if (!vtable || !vtable->operand_descriptors) return iree_ok_status();
  uint16_t descriptor_count = op->operand_count < vtable->fixed_operand_count
                                  ? op->operand_count
                                  : vtable->fixed_operand_count;
  const loom_value_id_t* operands = loom_op_const_operands(op);
  for (uint16_t i = 0; i < descriptor_count; ++i) {
    loom_operand_flags_t flags = vtable->operand_descriptors[i].flags;
    if (!iree_any_bit_set(flags, LOOM_OPERAND_READS | LOOM_OPERAND_WRITES)) {
      continue;
    }
    const loom_view_region_t* const_region = NULL;
    IREE_RETURN_IF_ERROR(
        loom_view_region_table_get(table, operands[i], &const_region));
    if (!const_region) continue;
    loom_view_region_t* region = &table->regions[const_region->region_id];
    loom_view_region_add_access(region, flags);
  }
  return iree_ok_status();
}

static iree_status_t loom_view_region_table_analyze_region(
    loom_view_region_table_t* table, const loom_region_t* region);

static iree_status_t loom_view_region_table_analyze_op_tree(
    loom_view_region_table_t* table, const loom_op_t* op) {
  IREE_RETURN_IF_ERROR(loom_view_region_table_record_op_accesses(table, op));
  const loom_value_id_t* results = loom_op_const_results(op);
  for (uint16_t i = 0; i < op->result_count; ++i) {
    const loom_view_region_t* region = NULL;
    IREE_RETURN_IF_ERROR(
        loom_view_region_table_get(table, results[i], &region));
  }
  loom_region_t* const* regions = loom_op_regions(op);
  for (uint8_t i = 0; i < op->region_count; ++i) {
    if (regions[i]) {
      IREE_RETURN_IF_ERROR(
          loom_view_region_table_analyze_region(table, regions[i]));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_view_region_table_analyze_region(
    loom_view_region_table_t* table, const loom_region_t* region) {
  if (!region) return iree_ok_status();
  const loom_block_t* block = NULL;
  loom_region_for_each_block(region, block) {
    for (uint16_t i = 0; i < block->arg_count; ++i) {
      const loom_view_region_t* view_region = NULL;
      IREE_RETURN_IF_ERROR(loom_view_region_table_get(
          table, loom_block_arg_id(block, i), &view_region));
    }
    const loom_op_t* op = NULL;
    loom_block_for_each_op(block, op) {
      IREE_RETURN_IF_ERROR(loom_view_region_table_analyze_op_tree(table, op));
    }
  }
  return iree_ok_status();
}

iree_status_t loom_view_region_table_analyze(loom_view_region_table_t* table) {
  return loom_view_region_table_analyze_region(table,
                                               table->value_domain->region);
}

loom_view_access_flags_t loom_view_region_table_root_access_flags(
    const loom_view_region_table_t* table, loom_value_id_t root_value_id) {
  loom_view_access_flags_t access_flags = 0;
  for (iree_host_size_t i = 0; i < table->region_count; ++i) {
    const loom_view_region_t* region = &table->regions[i];
    if (region->root_value_id == root_value_id) {
      access_flags |= region->access_flags;
    }
  }
  return access_flags;
}

iree_status_t loom_view_regions_prove_no_overlap(
    loom_view_region_table_t* table, const loom_view_region_t* left_region,
    const loom_view_region_t* right_region, bool* out_no_overlap) {
  *out_no_overlap = false;
  if (!left_region || !right_region) return iree_ok_status();
  if (left_region->root_value_id == LOOM_VALUE_ID_INVALID ||
      right_region->root_value_id == LOOM_VALUE_ID_INVALID) {
    return iree_ok_status();
  }
  if (left_region->root_value_id != right_region->root_value_id) {
    return iree_ok_status();
  }

  loom_symbolic_proof_result_t proof = LOOM_SYMBOLIC_PROOF_UNKNOWN;
  IREE_RETURN_IF_ERROR(loom_symbolic_expr_prove_le(
      &table->expression_context, &left_region->end_byte_offset,
      &right_region->begin_byte_offset, &proof));
  if (proof == LOOM_SYMBOLIC_PROOF_TRUE) {
    *out_no_overlap = true;
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_symbolic_expr_prove_le(
      &table->expression_context, &right_region->end_byte_offset,
      &left_region->begin_byte_offset, &proof));
  if (proof == LOOM_SYMBOLIC_PROOF_TRUE) {
    *out_no_overlap = true;
  }
  return iree_ok_status();
}
