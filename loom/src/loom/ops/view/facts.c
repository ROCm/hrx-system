// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Fact implementations for the view dialect.

#include <string.h>

#include "loom/ir/module.h"
#include "loom/ops/encoding/roles.h"
#include "loom/ops/encoding/storage.h"
#include "loom/ops/view/ops.h"
#include "loom/ops/view/reference.h"
#include "loom/util/fact_extensions.h"
#include "loom/util/fact_table.h"

static loom_value_facts_t loom_view_subview_dimension_facts(
    const loom_fact_context_t* context, loom_type_t type, uint8_t axis) {
  if (!loom_type_dim_is_dynamic_at(type, axis)) {
    return loom_value_facts_exact_i64(loom_type_dim_static_size_at(type, axis));
  }
  return loom_value_facts_non_negative_extent(loom_value_fact_table_lookup(
      context->table, loom_type_dim_value_id_at(type, axis)));
}

static void loom_view_subview_dense_strides(const loom_fact_context_t* context,
                                            loom_type_t source_type,
                                            loom_value_facts_t* out_strides) {
  loom_value_facts_t stride = loom_value_facts_exact_i64(1);
  for (uint8_t reverse_axis = 0; reverse_axis < loom_type_rank(source_type);
       ++reverse_axis) {
    const uint8_t axis =
        (uint8_t)(loom_type_rank(source_type) - reverse_axis - 1);
    out_strides[axis] = stride;
    const loom_value_facts_t dimension =
        loom_view_subview_dimension_facts(context, source_type, axis);
    loom_value_facts_muli(&stride, &dimension, &stride);
    stride = loom_value_facts_non_negative_extent(stride);
  }
}

static iree_status_t loom_view_subview_capture_layout_origin(
    loom_fact_context_t* context, const loom_module_t* module,
    loom_type_t source_type,
    loom_value_fact_address_layout_kind_t source_layout_kind,
    loom_value_id_t result_value_id) {
  if (!context->publication) {
    return iree_ok_status();
  }
  const bool has_source_encoding = loom_type_has_ssa_encoding(source_type);
  const loom_value_id_t source_encoding =
      has_source_encoding ? loom_type_encoding_value_id(source_type)
                          : LOOM_VALUE_ID_INVALID;
  const bool has_physical_encoding =
      has_source_encoding &&
      loom_encoding_query_type_role(context, module, source_type) ==
          LOOM_ENCODING_ROLE_PHYSICAL_STORAGE;
  const loom_value_fact_layout_origin_t source_origin =
      has_source_encoding ? loom_value_fact_table_query_layout_origin(
                                context->table, source_encoding)
                          : (loom_value_fact_layout_origin_t){0};
  const loom_value_id_t physical_encoding =
      has_physical_encoding && source_origin.has_physical_encoding_value
          ? source_origin.physical_encoding_value_id
          : source_encoding;
  if (source_layout_kind == LOOM_VALUE_FACT_ADDRESS_LAYOUT_DENSE) {
    bool has_dynamic_stride = false;
    for (uint8_t axis = 1; axis < loom_type_rank(source_type); ++axis) {
      has_dynamic_stride |= loom_type_dim_is_dynamic_at(source_type, axis);
    }
    if (!has_dynamic_stride && !has_physical_encoding &&
        !context->table->layout_origins) {
      return iree_ok_status();
    }
    uint64_t dimensions[LOOM_TYPE_MAX_RANK] = {0};
    for (uint8_t axis = 0; axis < loom_type_rank(source_type); ++axis) {
      dimensions[axis] = loom_type_dim(source_type, axis);
    }
    return loom_value_fact_table_define_layout_origin(
        context->table, result_value_id,
        (loom_value_fact_layout_origin_t){
            .values.dense_dimensions = dimensions,
            .physical_encoding_value_id = physical_encoding,
            .count = loom_type_rank(source_type),
            .kind = LOOM_VALUE_FACT_LAYOUT_ORIGIN_DENSE_DIMENSIONS,
            .has_physical_encoding_value = has_physical_encoding,
        });
  }

  if (source_layout_kind == LOOM_VALUE_FACT_ADDRESS_LAYOUT_STRIDED &&
      has_source_encoding && context->table->layout_origins &&
      (!has_physical_encoding || source_origin.has_physical_encoding_value)) {
    return loom_value_fact_table_forward_layout_origin(
        context->table, source_encoding, result_value_id);
  }
  if (has_physical_encoding) {
    loom_value_fact_layout_origin_t result_origin = source_origin;
    result_origin.has_physical_encoding_value = true;
    result_origin.physical_encoding_value_id = physical_encoding;
    return loom_value_fact_table_define_layout_origin(
        context->table, result_value_id, result_origin);
  }
  if (context->table->layout_origins) {
    loom_value_fact_table_clear_layout_origin(context->table, result_value_id);
  }
  return iree_ok_status();
}

static iree_status_t loom_view_subview_capture_physical_encoding(
    loom_fact_context_t* context, const loom_module_t* module,
    loom_type_t source_type, loom_value_id_t result_value_id,
    loom_value_facts_t* captured_strides,
    loom_value_fact_address_layout_t* out_address_layout,
    loom_value_facts_t* out_facts) {
  loom_value_facts_t source_stride_storage[LOOM_TYPE_MAX_RANK] = {0};
  loom_value_fact_address_layout_t source_layout = {0};
  (void)loom_encoding_query_type_address_layout(
      context, module, source_type, source_stride_storage,
      IREE_ARRAYSIZE(source_stride_storage), &source_layout);

  if (source_layout.kind == LOOM_VALUE_FACT_ADDRESS_LAYOUT_DENSE) {
    loom_view_subview_dense_strides(context, source_type, captured_strides);
    *out_address_layout = (loom_value_fact_address_layout_t){
        .kind = LOOM_VALUE_FACT_ADDRESS_LAYOUT_STRIDED,
        .rank = loom_type_rank(source_type),
        .strides = captured_strides,
    };
  } else if (source_layout.kind == LOOM_VALUE_FACT_ADDRESS_LAYOUT_STRIDED &&
             source_layout.rank == loom_type_rank(source_type)) {
    memcpy(captured_strides, source_layout.strides,
           source_layout.rank * sizeof(*captured_strides));
    *out_address_layout = (loom_value_fact_address_layout_t){
        .kind = LOOM_VALUE_FACT_ADDRESS_LAYOUT_STRIDED,
        .rank = source_layout.rank,
        .strides = captured_strides,
    };
  } else {
    *out_address_layout = (loom_value_fact_address_layout_t){0};
  }

  IREE_RETURN_IF_ERROR(loom_view_subview_capture_layout_origin(
      context, module, source_type, source_layout.kind, result_value_id));

  loom_value_fact_storage_schema_t storage_schema = {0};
  (void)loom_encoding_query_type_storage_schema(context, module, source_type,
                                                &storage_schema);
  loom_value_fact_encoding_summary_t summary = {
      .role = loom_encoding_query_type_role(context, module, source_type),
      .address_layout = *out_address_layout,
      .storage_schema = storage_schema,
  };
  return loom_value_facts_make_encoding_summary(context, summary, out_facts);
}

iree_status_t loom_view_load_facts(loom_fact_context_t* context,
                                   const loom_module_t* module,
                                   const loom_op_t* op,
                                   const loom_value_facts_t* operand_facts,
                                   loom_value_facts_t* result_facts) {
  (void)operand_facts;
  loom_type_t view_type =
      loom_module_value_type(module, loom_view_load_view(op));
  if (!loom_encoding_query_type_storage_content_facts(
          context, module, view_type, &result_facts[0])) {
    result_facts[0] = loom_value_facts_unknown();
  }
  return iree_ok_status();
}

iree_status_t loom_view_atomic_observation_facts(
    loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts) {
  (void)context;
  (void)module;
  (void)op;
  (void)operand_facts;
  // Loads, read-modify-writes, and compare-exchanges each return one observed
  // scalar. Concurrent publication can change it between lane observations,
  // even when every lane addresses the same element.
  result_facts[0] = loom_value_facts_unknown();
  loom_value_facts_mark_lane_varying(&result_facts[0]);
  return iree_ok_status();
}

iree_status_t loom_view_subview_facts(loom_fact_context_t* context,
                                      const loom_module_t* module,
                                      const loom_op_t* op,
                                      const loom_value_facts_t* operand_facts,
                                      loom_value_facts_t* result_facts) {
  loom_type_t source_type =
      loom_module_value_type(module, loom_view_subview_source(op));
  loom_type_t result_type =
      loom_module_value_type(module, loom_view_subview_result(op));
  const loom_value_slice_t physical_encoding =
      loom_view_subview_physical_encoding(op);
  loom_value_facts_t captured_strides[LOOM_TYPE_MAX_RANK] = {0};
  loom_value_fact_address_layout_t captured_layout = {0};
  const loom_value_fact_address_layout_t* result_layout = NULL;
  if (physical_encoding.count == 1) {
    IREE_RETURN_IF_ERROR(loom_view_subview_capture_physical_encoding(
        context, module, source_type, physical_encoding.values[0],
        captured_strides, &captured_layout, &result_facts[1]));
    result_layout = &captured_layout;
  }
  return loom_view_reference_make_subview(
      context, module, loom_view_subview_source(op), operand_facts[0],
      loom_view_subview_static_offsets(op), loom_view_subview_offsets(op),
      source_type, result_type, result_layout, &result_facts[0]);
}

iree_status_t loom_view_reinterpret_facts(
    loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts) {
  loom_value_id_t source = loom_op_const_operands(op)[0];
  loom_type_t source_type = loom_module_value_type(module, source);
  loom_type_t result_type =
      loom_module_value_type(module, loom_op_const_results(op)[0]);
  return loom_view_reference_make_reinterpret(context, module, source,
                                              operand_facts[0], source_type,
                                              result_type, &result_facts[0]);
}
