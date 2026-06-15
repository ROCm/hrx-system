// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/movement.h"

#include "loom/ir/facts.h"
#include "loom/ops/kernel/ops.h"
#include "loom/ops/vector/ops.h"
#include "loom/util/math.h"

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

static bool loom_movement_exact_i64(loom_symbolic_expr_t expression,
                                    int64_t* out_value) {
  *out_value = 0;
  if (loom_symbolic_expr_is_constant(&expression)) {
    *out_value = expression.constant;
    return true;
  }
  if (!loom_value_facts_is_exact(expression.facts) ||
      loom_value_facts_is_float(expression.facts)) {
    return false;
  }
  *out_value = expression.facts.range_lo;
  return true;
}

static void loom_movement_endpoint_initialize_none(
    loom_movement_endpoint_t* out_endpoint) {
  *out_endpoint = (loom_movement_endpoint_t){
      .kind = LOOM_MOVEMENT_ENDPOINT_NONE,
      .value_id = LOOM_VALUE_ID_INVALID,
      .root_value_id = LOOM_VALUE_ID_INVALID,
      .memory_space = LOOM_VALUE_FACT_MEMORY_SPACE_UNKNOWN,
  };
}

static void loom_movement_request_initialize(
    const loom_op_t* op, loom_movement_request_t* out_request,
    loom_movement_diagnostic_t* out_diagnostic) {
  *out_request = (loom_movement_request_t){
      .op = op,
      .kind = LOOM_MOVEMENT_KIND_UNKNOWN,
      .layout_kind = LOOM_MOVEMENT_LAYOUT_UNKNOWN,
      .schema_kind = LOOM_MOVEMENT_SCHEMA_UNKNOWN,
      .mask_value_id = LOOM_VALUE_ID_INVALID,
      .offsets_value_id = LOOM_VALUE_ID_INVALID,
      .descriptor_value_id = LOOM_VALUE_ID_INVALID,
      .cluster_mask_value_id = LOOM_VALUE_ID_INVALID,
      .direction = UINT8_MAX,
  };
  loom_movement_endpoint_initialize_none(&out_request->source);
  loom_movement_endpoint_initialize_none(&out_request->dest);
  *out_diagnostic = (loom_movement_diagnostic_t){0};
}

static iree_status_t loom_movement_describe_result(bool described,
                                                   bool* out_described) {
  *out_described = described;
  return iree_ok_status();
}

static bool loom_movement_static_vector_lane_count(loom_type_t vector_type,
                                                   int64_t* out_lane_count) {
  *out_lane_count = 0;
  if (!loom_type_is_vector(vector_type)) {
    return false;
  }
  int64_t lane_count = 1;
  const uint8_t rank = loom_type_rank(vector_type);
  for (uint8_t i = 0; i < rank; ++i) {
    if (loom_type_dim_is_dynamic_at(vector_type, i)) {
      return false;
    }
    const int64_t dim_size = loom_type_dim_static_size_at(vector_type, i);
    if (dim_size <= 0 ||
        !iree_checked_mul_i64(lane_count, dim_size, &lane_count)) {
      return false;
    }
  }
  *out_lane_count = lane_count;
  return true;
}

static bool loom_movement_vector_transfer_byte_count(
    const loom_vector_memory_access_t* access, int64_t* out_byte_count) {
  *out_byte_count = 0;
  if (access->static_element_byte_count <= 0) {
    return false;
  }
  int64_t lane_count = 0;
  if (!loom_movement_static_vector_lane_count(access->vector_type,
                                              &lane_count)) {
    return false;
  }
  return iree_checked_mul_i64(lane_count, access->static_element_byte_count,
                              out_byte_count);
}

static bool loom_movement_vector_footprint_byte_length(
    const loom_vector_memory_access_t* access, int64_t* out_byte_length) {
  *out_byte_length = 0;
  if (access->static_element_byte_count <= 0) {
    return false;
  }

  int64_t maximum_element_offset = 0;
  for (uint8_t view_axis = access->first_vector_axis;
       view_axis < access->view_rank; ++view_axis) {
    const uint8_t vector_axis = view_axis - access->first_vector_axis;
    if (loom_type_dim_is_dynamic_at(access->vector_type, vector_axis)) {
      return false;
    }
    const int64_t extent =
        loom_type_dim_static_size_at(access->vector_type, vector_axis);
    if (extent <= 0) {
      return false;
    }
    int64_t axis_stride = 0;
    if (!loom_vector_memory_access_static_axis_stride(access, view_axis,
                                                      &axis_stride)) {
      return false;
    }
    int64_t contribution = 0;
    if (!iree_checked_mul_i64(extent - 1, axis_stride, &contribution) ||
        !iree_checked_add_i64(maximum_element_offset, contribution,
                              &maximum_element_offset)) {
      return false;
    }
  }

  int64_t element_span = 0;
  if (!iree_checked_add_i64(maximum_element_offset, 1, &element_span)) {
    return false;
  }
  return iree_checked_mul_i64(element_span, access->static_element_byte_count,
                              out_byte_length);
}

static bool loom_movement_vector_static_begin_byte_offset(
    const loom_vector_memory_access_t* access, loom_attribute_t static_indices,
    int64_t* out_byte_offset) {
  *out_byte_offset = 0;
  if (static_indices.kind != LOOM_ATTR_I64_ARRAY ||
      static_indices.count != access->view_rank) {
    return false;
  }
  int64_t lane_indices[LOOM_ENCODING_ADDRESS_LAYOUT_MAX_RANK] = {0};
  return loom_vector_memory_access_static_lane_byte_offset(
      access, static_indices, lane_indices, access->vector_rank,
      out_byte_offset);
}

static loom_movement_layout_kind_t loom_movement_direct_layout_kind(
    const loom_vector_memory_access_t* access) {
  switch (access->layout_kind) {
    case LOOM_VECTOR_MEMORY_LAYOUT_DENSE:
      return LOOM_MOVEMENT_LAYOUT_DENSE;
    case LOOM_VECTOR_MEMORY_LAYOUT_STRIDED:
      return LOOM_MOVEMENT_LAYOUT_STATIC_STRIDED;
    case LOOM_VECTOR_MEMORY_LAYOUT_UNKNOWN:
      return LOOM_MOVEMENT_LAYOUT_UNKNOWN;
  }
  return LOOM_MOVEMENT_LAYOUT_UNKNOWN;
}

static iree_status_t loom_movement_endpoint_for_view(
    loom_movement_analysis_t* analysis, loom_value_id_t view_value_id,
    loom_movement_endpoint_flags_t access_flags,
    loom_movement_endpoint_t* out_endpoint) {
  loom_movement_endpoint_initialize_none(out_endpoint);
  if (view_value_id >= analysis->module->values.count) {
    return iree_ok_status();
  }

  const loom_view_region_t* region = NULL;
  IREE_RETURN_IF_ERROR(loom_view_region_table_get(&analysis->view_regions,
                                                  view_value_id, &region));
  if (!region) {
    return iree_ok_status();
  }

  *out_endpoint = (loom_movement_endpoint_t){
      .kind = LOOM_MOVEMENT_ENDPOINT_VIEW,
      .flags = access_flags,
      .value_id = view_value_id,
      .type = loom_module_value_type(analysis->module, view_value_id),
      .memory_space = region->memory_space,
      .root_value_id = region->root_value_id,
      .begin_byte_offset = region->begin_byte_offset,
      .byte_length = region->byte_length,
      .end_byte_offset = region->end_byte_offset,
      .minimum_alignment = region->minimum_alignment,
      .root_minimum_alignment = region->root_minimum_alignment,
      .precision_flags = region->precision_flags,
  };
  if (loom_movement_exact_i64(region->begin_byte_offset,
                              &out_endpoint->static_begin_byte_offset)) {
    out_endpoint->flags |= LOOM_MOVEMENT_ENDPOINT_STATIC_BEGIN;
  }
  if (loom_movement_exact_i64(region->byte_length,
                              &out_endpoint->static_byte_length)) {
    out_endpoint->flags |= LOOM_MOVEMENT_ENDPOINT_STATIC_LENGTH;
  }
  return iree_ok_status();
}

static void loom_movement_endpoint_for_register(
    const loom_module_t* module, loom_value_id_t value_id,
    loom_movement_endpoint_flags_t access_flags,
    loom_movement_endpoint_t* out_endpoint) {
  *out_endpoint = (loom_movement_endpoint_t){
      .kind = LOOM_MOVEMENT_ENDPOINT_REGISTER,
      .flags = access_flags,
      .value_id = value_id,
      .type = loom_module_value_type(module, value_id),
      .memory_space = LOOM_VALUE_FACT_MEMORY_SPACE_UNKNOWN,
      .root_value_id = LOOM_VALUE_ID_INVALID,
  };
}

static bool loom_movement_apply_vector_access(
    loom_movement_analysis_t* analysis, loom_value_id_t view_value_id,
    loom_type_t vector_type, loom_attribute_t static_indices,
    loom_movement_layout_kind_t fallback_layout_kind,
    loom_movement_request_t* request, loom_movement_diagnostic_t* diagnostic) {
  const loom_type_t view_type =
      loom_module_value_type(analysis->module, view_value_id);
  const loom_fact_context_t* fact_context =
      analysis->fact_table ? &analysis->fact_table->context : NULL;
  if (!loom_vector_memory_access_describe(fact_context, analysis->module,
                                          view_type, vector_type,
                                          &request->vector_access)) {
    diagnostic->rejection_bits |= LOOM_MOVEMENT_REJECTION_VECTOR_ACCESS;
    return false;
  }

  if (request->vector_access.layout_kind == LOOM_VECTOR_MEMORY_LAYOUT_UNKNOWN) {
    diagnostic->rejection_bits |= LOOM_MOVEMENT_REJECTION_LAYOUT;
    return false;
  }
  if (request->vector_access.static_element_byte_count <= 0) {
    diagnostic->rejection_bits |= LOOM_MOVEMENT_REJECTION_ELEMENT_WIDTH;
    return false;
  }

  int64_t transferred_byte_count = 0;
  if (!loom_movement_vector_transfer_byte_count(&request->vector_access,
                                                &transferred_byte_count)) {
    diagnostic->rejection_bits |= LOOM_MOVEMENT_REJECTION_LANE_COUNT;
    return false;
  }
  request->flags |= LOOM_MOVEMENT_REQUEST_STATIC_TRANSFER;
  request->transferred_byte_count = transferred_byte_count;

  request->layout_kind =
      fallback_layout_kind == LOOM_MOVEMENT_LAYOUT_UNKNOWN
          ? loom_movement_direct_layout_kind(&request->vector_access)
          : fallback_layout_kind;

  int64_t endpoint_byte_length = 0;
  if (loom_movement_vector_footprint_byte_length(&request->vector_access,
                                                 &endpoint_byte_length)) {
    loom_symbolic_expr_constant(endpoint_byte_length,
                                &request->source.byte_length);
    loom_symbolic_expr_constant(endpoint_byte_length,
                                &request->dest.byte_length);
    request->source.flags |= LOOM_MOVEMENT_ENDPOINT_STATIC_LENGTH;
    request->dest.flags |= LOOM_MOVEMENT_ENDPOINT_STATIC_LENGTH;
    request->source.static_byte_length = endpoint_byte_length;
    request->dest.static_byte_length = endpoint_byte_length;
  }

  int64_t static_begin = 0;
  if (loom_movement_vector_static_begin_byte_offset(
          &request->vector_access, static_indices, &static_begin)) {
    if (request->source.kind == LOOM_MOVEMENT_ENDPOINT_VIEW &&
        iree_all_bits_set(request->source.flags,
                          LOOM_MOVEMENT_ENDPOINT_STATIC_BEGIN |
                              LOOM_MOVEMENT_ENDPOINT_STATIC_LENGTH)) {
      if (!iree_checked_add_i64(request->source.static_begin_byte_offset,
                                static_begin,
                                &request->source.static_begin_byte_offset)) {
        diagnostic->rejection_bits |= LOOM_MOVEMENT_REJECTION_FOOTPRINT;
        return false;
      }
      int64_t static_end = 0;
      if (!iree_checked_add_i64(request->source.static_begin_byte_offset,
                                request->source.static_byte_length,
                                &static_end)) {
        diagnostic->rejection_bits |= LOOM_MOVEMENT_REJECTION_FOOTPRINT;
        return false;
      }
      loom_symbolic_expr_constant(request->source.static_begin_byte_offset,
                                  &request->source.begin_byte_offset);
      loom_symbolic_expr_constant(static_end, &request->source.end_byte_offset);
    }
    if (request->dest.kind == LOOM_MOVEMENT_ENDPOINT_VIEW &&
        iree_all_bits_set(request->dest.flags,
                          LOOM_MOVEMENT_ENDPOINT_STATIC_BEGIN |
                              LOOM_MOVEMENT_ENDPOINT_STATIC_LENGTH)) {
      if (!iree_checked_add_i64(request->dest.static_begin_byte_offset,
                                static_begin,
                                &request->dest.static_begin_byte_offset)) {
        diagnostic->rejection_bits |= LOOM_MOVEMENT_REJECTION_FOOTPRINT;
        return false;
      }
      int64_t static_end = 0;
      if (!iree_checked_add_i64(request->dest.static_begin_byte_offset,
                                request->dest.static_byte_length,
                                &static_end)) {
        diagnostic->rejection_bits |= LOOM_MOVEMENT_REJECTION_FOOTPRINT;
        return false;
      }
      loom_symbolic_expr_constant(request->dest.static_begin_byte_offset,
                                  &request->dest.begin_byte_offset);
      loom_symbolic_expr_constant(static_end, &request->dest.end_byte_offset);
    }
  }

  return true;
}

static bool loom_movement_cache_policy_from_attrs(
    loom_attribute_t cache_scope_attr, loom_attribute_t cache_temporal_attr,
    loom_vector_memory_cache_policy_t* out_policy,
    loom_movement_diagnostic_t* diagnostic) {
  if (!loom_vector_memory_cache_policy_from_attrs(
          cache_scope_attr, cache_temporal_attr, out_policy)) {
    diagnostic->rejection_bits |= LOOM_MOVEMENT_REJECTION_CACHE_POLICY;
    return false;
  }
  return true;
}

enum {
  // Operand/result index sentinel used by descriptor rows.
  LOOM_MOVEMENT_ABSENT_INDEX = UINT8_MAX,
};

typedef struct loom_movement_vector_descriptor_t {
  // Operation kind accepted by this row.
  loom_op_kind_t op_kind;

  // Movement category assigned to matching operations.
  loom_movement_kind_t movement_kind;

  // Request flags applied before access decomposition.
  loom_movement_request_flags_t request_flags;

  // Fixed movement layout, or UNKNOWN to derive the direct view layout.
  loom_movement_layout_kind_t layout_kind;

  // Operand index of the accessed view.
  uint8_t view_operand_index;

  // Operand index of the source/destination register value, when present.
  uint8_t register_operand_index;

  // Result index of the destination register value, when present.
  uint8_t register_result_index;

  // Operand index of the optional mask value.
  uint8_t mask_operand_index;

  // Operand index of the optional per-lane offset value.
  uint8_t offsets_operand_index;

  // True when the view endpoint is written rather than read.
  bool writes_view;
} loom_movement_vector_descriptor_t;

#define LOOM_MOVEMENT_VECTOR_DESCRIPTOR(                                       \
    op_kind_value, movement_kind_value, request_flags_value,                   \
    layout_kind_value, view_operand_index_value, register_operand_index_value, \
    register_result_index_value, mask_operand_index_value,                     \
    offsets_operand_index_value, writes_view_value)                            \
  {                                                                            \
      .op_kind = (op_kind_value),                                              \
      .movement_kind = (movement_kind_value),                                  \
      .request_flags = (request_flags_value),                                  \
      .layout_kind = (layout_kind_value),                                      \
      .view_operand_index = (view_operand_index_value),                        \
      .register_operand_index = (register_operand_index_value),                \
      .register_result_index = (register_result_index_value),                  \
      .mask_operand_index = (mask_operand_index_value),                        \
      .offsets_operand_index = (offsets_operand_index_value),                  \
      .writes_view = (writes_view_value),                                      \
  }

static const loom_movement_vector_descriptor_t
    loom_movement_vector_descriptors[] = {
        LOOM_MOVEMENT_VECTOR_DESCRIPTOR(
            LOOM_OP_VECTOR_LOAD, LOOM_MOVEMENT_KIND_VECTOR_LOAD, 0,
            LOOM_MOVEMENT_LAYOUT_UNKNOWN, 0, LOOM_MOVEMENT_ABSENT_INDEX, 0,
            LOOM_MOVEMENT_ABSENT_INDEX, LOOM_MOVEMENT_ABSENT_INDEX, false),
        LOOM_MOVEMENT_VECTOR_DESCRIPTOR(
            LOOM_OP_VECTOR_STORE, LOOM_MOVEMENT_KIND_VECTOR_STORE, 0,
            LOOM_MOVEMENT_LAYOUT_UNKNOWN, 1, 0, LOOM_MOVEMENT_ABSENT_INDEX,
            LOOM_MOVEMENT_ABSENT_INDEX, LOOM_MOVEMENT_ABSENT_INDEX, true),
        LOOM_MOVEMENT_VECTOR_DESCRIPTOR(
            LOOM_OP_VECTOR_LOAD_MASK, LOOM_MOVEMENT_KIND_VECTOR_LOAD_MASK,
            LOOM_MOVEMENT_REQUEST_MASKED, LOOM_MOVEMENT_LAYOUT_UNKNOWN, 0,
            LOOM_MOVEMENT_ABSENT_INDEX, 0, 1, LOOM_MOVEMENT_ABSENT_INDEX,
            false),
        LOOM_MOVEMENT_VECTOR_DESCRIPTOR(
            LOOM_OP_VECTOR_STORE_MASK, LOOM_MOVEMENT_KIND_VECTOR_STORE_MASK,
            LOOM_MOVEMENT_REQUEST_MASKED, LOOM_MOVEMENT_LAYOUT_UNKNOWN, 1, 0,
            LOOM_MOVEMENT_ABSENT_INDEX, 2, LOOM_MOVEMENT_ABSENT_INDEX, true),
        LOOM_MOVEMENT_VECTOR_DESCRIPTOR(
            LOOM_OP_VECTOR_LOAD_EXPAND, LOOM_MOVEMENT_KIND_VECTOR_LOAD_EXPAND,
            LOOM_MOVEMENT_REQUEST_MASKED, LOOM_MOVEMENT_LAYOUT_COMPRESS_EXPAND,
            0, LOOM_MOVEMENT_ABSENT_INDEX, 0, 1, LOOM_MOVEMENT_ABSENT_INDEX,
            false),
        LOOM_MOVEMENT_VECTOR_DESCRIPTOR(
            LOOM_OP_VECTOR_STORE_COMPRESS,
            LOOM_MOVEMENT_KIND_VECTOR_STORE_COMPRESS,
            LOOM_MOVEMENT_REQUEST_MASKED, LOOM_MOVEMENT_LAYOUT_COMPRESS_EXPAND,
            1, 0, LOOM_MOVEMENT_ABSENT_INDEX, 2, LOOM_MOVEMENT_ABSENT_INDEX,
            true),
        LOOM_MOVEMENT_VECTOR_DESCRIPTOR(
            LOOM_OP_VECTOR_GATHER, LOOM_MOVEMENT_KIND_VECTOR_GATHER,
            LOOM_MOVEMENT_REQUEST_IRREGULAR_OFFSETS,
            LOOM_MOVEMENT_LAYOUT_GATHER_SCATTER, 0, LOOM_MOVEMENT_ABSENT_INDEX,
            0, LOOM_MOVEMENT_ABSENT_INDEX, 1, false),
        LOOM_MOVEMENT_VECTOR_DESCRIPTOR(
            LOOM_OP_VECTOR_SCATTER, LOOM_MOVEMENT_KIND_VECTOR_SCATTER,
            LOOM_MOVEMENT_REQUEST_IRREGULAR_OFFSETS,
            LOOM_MOVEMENT_LAYOUT_GATHER_SCATTER, 1, 0,
            LOOM_MOVEMENT_ABSENT_INDEX, LOOM_MOVEMENT_ABSENT_INDEX, 2, true),
        LOOM_MOVEMENT_VECTOR_DESCRIPTOR(
            LOOM_OP_VECTOR_GATHER_MASK, LOOM_MOVEMENT_KIND_VECTOR_GATHER_MASK,
            LOOM_MOVEMENT_REQUEST_IRREGULAR_OFFSETS |
                LOOM_MOVEMENT_REQUEST_MASKED,
            LOOM_MOVEMENT_LAYOUT_GATHER_SCATTER, 0, LOOM_MOVEMENT_ABSENT_INDEX,
            0, 2, 1, false),
        LOOM_MOVEMENT_VECTOR_DESCRIPTOR(
            LOOM_OP_VECTOR_SCATTER_MASK, LOOM_MOVEMENT_KIND_VECTOR_SCATTER_MASK,
            LOOM_MOVEMENT_REQUEST_IRREGULAR_OFFSETS |
                LOOM_MOVEMENT_REQUEST_MASKED,
            LOOM_MOVEMENT_LAYOUT_GATHER_SCATTER, 1, 0,
            LOOM_MOVEMENT_ABSENT_INDEX, 3, 2, true),
};

#undef LOOM_MOVEMENT_VECTOR_DESCRIPTOR

static const loom_movement_vector_descriptor_t*
loom_movement_vector_descriptor_for(loom_op_kind_t op_kind) {
  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(loom_movement_vector_descriptors); ++i) {
    const loom_movement_vector_descriptor_t* descriptor =
        &loom_movement_vector_descriptors[i];
    if (descriptor->op_kind == op_kind) {
      return descriptor;
    }
  }
  return NULL;
}

static iree_status_t loom_movement_describe_vector(
    loom_movement_analysis_t* analysis, const loom_op_t* op,
    const loom_movement_vector_descriptor_t* descriptor,
    loom_movement_request_t* request, loom_movement_diagnostic_t* diagnostic,
    bool* out_described) {
  request->kind = descriptor->movement_kind;
  request->flags |= descriptor->request_flags;
  request->layout_kind = descriptor->layout_kind;
  request->schema_kind = LOOM_MOVEMENT_SCHEMA_TYPED_ELEMENT;

  const loom_value_id_t* operands = loom_op_const_operands(op);
  const loom_value_id_t* results = loom_op_const_results(op);
  const loom_value_id_t view_value_id =
      operands[descriptor->view_operand_index];
  const bool register_is_result =
      descriptor->register_result_index != LOOM_MOVEMENT_ABSENT_INDEX;
  const loom_value_id_t register_value_id =
      register_is_result ? results[descriptor->register_result_index]
                         : operands[descriptor->register_operand_index];
  if (descriptor->mask_operand_index != LOOM_MOVEMENT_ABSENT_INDEX) {
    request->mask_value_id = operands[descriptor->mask_operand_index];
  }
  if (descriptor->offsets_operand_index != LOOM_MOVEMENT_ABSENT_INDEX) {
    request->offsets_value_id = operands[descriptor->offsets_operand_index];
  }

  if (!loom_vector_memory_cache_policy_from_op(analysis->module, op,
                                               &request->cache_policy)) {
    diagnostic->rejection_bits |= LOOM_MOVEMENT_REJECTION_CACHE_POLICY;
    return loom_movement_describe_result(false, out_described);
  }
  if (descriptor->writes_view) {
    loom_movement_endpoint_for_register(analysis->module, register_value_id,
                                        LOOM_MOVEMENT_ENDPOINT_READ,
                                        &request->source);
    IREE_RETURN_IF_ERROR(loom_movement_endpoint_for_view(
        analysis, view_value_id, LOOM_MOVEMENT_ENDPOINT_WRITE, &request->dest));
    if (request->dest.kind != LOOM_MOVEMENT_ENDPOINT_VIEW) {
      diagnostic->rejection_bits |= LOOM_MOVEMENT_REJECTION_ENDPOINT;
      return loom_movement_describe_result(false, out_described);
    }
  } else {
    IREE_RETURN_IF_ERROR(loom_movement_endpoint_for_view(
        analysis, view_value_id, LOOM_MOVEMENT_ENDPOINT_READ,
        &request->source));
    if (request->source.kind != LOOM_MOVEMENT_ENDPOINT_VIEW) {
      diagnostic->rejection_bits |= LOOM_MOVEMENT_REJECTION_ENDPOINT;
      return loom_movement_describe_result(false, out_described);
    }
    loom_movement_endpoint_for_register(analysis->module, register_value_id,
                                        LOOM_MOVEMENT_ENDPOINT_WRITE,
                                        &request->dest);
  }

  if (!loom_movement_apply_vector_access(
          analysis, view_value_id,
          loom_module_value_type(analysis->module, register_value_id),
          loom_op_const_attrs(op)[2], descriptor->layout_kind, request,
          diagnostic)) {
    return loom_movement_describe_result(false, out_described);
  }
  return loom_movement_describe_result(true, out_described);
}

typedef enum loom_movement_async_transfer_mode_e {
  // Source and destination byte footprints must be statically equal.
  LOOM_MOVEMENT_ASYNC_TRANSFER_EQUAL_ENDPOINTS = 0,

  // The source byte footprint is the transferred payload; destination padding
  // is legal.
  LOOM_MOVEMENT_ASYNC_TRANSFER_SOURCE_LENGTH = 1,
} loom_movement_async_transfer_mode_t;

typedef struct loom_movement_async_descriptor_t {
  // Operation kind accepted by this row.
  loom_op_kind_t op_kind;

  // Movement category assigned to matching operations.
  loom_movement_kind_t movement_kind;

  // Request flags applied before endpoint decomposition.
  loom_movement_request_flags_t request_flags;

  // Fixed movement layout.
  loom_movement_layout_kind_t layout_kind;

  // Static transfer proof mode for this operation.
  loom_movement_async_transfer_mode_t transfer_mode;

  // Operand index of the source view.
  uint8_t source_operand_index;

  // Operand index of the destination view.
  uint8_t dest_operand_index;

  // Operand index of the optional mask value.
  uint8_t mask_operand_index;

  // Operand index of the optional hardware descriptor value.
  uint8_t descriptor_operand_index;

  // Operand index of the optional cluster mask value.
  uint8_t cluster_mask_operand_index;

  // Attribute index of the optional direction enum.
  uint8_t direction_attr_index;

  // Constant direction enum when no direction attribute is present.
  uint8_t direction_value;
} loom_movement_async_descriptor_t;

#define LOOM_MOVEMENT_ASYNC_DESCRIPTOR(                                 \
    op_kind_value, movement_kind_value, request_flags_value,            \
    layout_kind_value, transfer_mode_value, source_operand_index_value, \
    dest_operand_index_value, mask_operand_index_value,                 \
    descriptor_operand_index_value, cluster_mask_operand_index_value,   \
    direction_attr_index_value, direction_value_value)                  \
  {                                                                     \
      .op_kind = (op_kind_value),                                       \
      .movement_kind = (movement_kind_value),                           \
      .request_flags = (request_flags_value),                           \
      .layout_kind = (layout_kind_value),                               \
      .transfer_mode = (transfer_mode_value),                           \
      .source_operand_index = (source_operand_index_value),             \
      .dest_operand_index = (dest_operand_index_value),                 \
      .mask_operand_index = (mask_operand_index_value),                 \
      .descriptor_operand_index = (descriptor_operand_index_value),     \
      .cluster_mask_operand_index = (cluster_mask_operand_index_value), \
      .direction_attr_index = (direction_attr_index_value),             \
      .direction_value = (direction_value_value),                       \
  }

static const loom_movement_async_descriptor_t
    loom_movement_async_descriptors[] = {
        LOOM_MOVEMENT_ASYNC_DESCRIPTOR(
            LOOM_OP_KERNEL_ASYNC_COPY, LOOM_MOVEMENT_KIND_KERNEL_ASYNC_COPY, 0,
            LOOM_MOVEMENT_LAYOUT_BYTE_RANGE,
            LOOM_MOVEMENT_ASYNC_TRANSFER_EQUAL_ENDPOINTS, 0, 1,
            LOOM_MOVEMENT_ABSENT_INDEX, LOOM_MOVEMENT_ABSENT_INDEX,
            LOOM_MOVEMENT_ABSENT_INDEX, 2, UINT8_MAX),
        LOOM_MOVEMENT_ASYNC_DESCRIPTOR(
            LOOM_OP_KERNEL_ASYNC_COPY_MASK,
            LOOM_MOVEMENT_KIND_KERNEL_ASYNC_COPY_MASK,
            LOOM_MOVEMENT_REQUEST_MASKED, LOOM_MOVEMENT_LAYOUT_BYTE_RANGE,
            LOOM_MOVEMENT_ASYNC_TRANSFER_EQUAL_ENDPOINTS, 0, 1, 2,
            LOOM_MOVEMENT_ABSENT_INDEX, LOOM_MOVEMENT_ABSENT_INDEX, 2,
            UINT8_MAX),
        LOOM_MOVEMENT_ASYNC_DESCRIPTOR(
            LOOM_OP_KERNEL_ASYNC_GATHER, LOOM_MOVEMENT_KIND_KERNEL_ASYNC_GATHER,
            0, LOOM_MOVEMENT_LAYOUT_SUBGROUP_GATHER,
            LOOM_MOVEMENT_ASYNC_TRANSFER_SOURCE_LENGTH, 0, 1,
            LOOM_MOVEMENT_ABSENT_INDEX, LOOM_MOVEMENT_ABSENT_INDEX,
            LOOM_MOVEMENT_ABSENT_INDEX, LOOM_MOVEMENT_ABSENT_INDEX, UINT8_MAX),
        LOOM_MOVEMENT_ASYNC_DESCRIPTOR(
            LOOM_OP_KERNEL_ASYNC_GATHER_MASK,
            LOOM_MOVEMENT_KIND_KERNEL_ASYNC_GATHER_MASK,
            LOOM_MOVEMENT_REQUEST_MASKED, LOOM_MOVEMENT_LAYOUT_SUBGROUP_GATHER,
            LOOM_MOVEMENT_ASYNC_TRANSFER_SOURCE_LENGTH, 0, 1, 2,
            LOOM_MOVEMENT_ABSENT_INDEX, LOOM_MOVEMENT_ABSENT_INDEX,
            LOOM_MOVEMENT_ABSENT_INDEX, UINT8_MAX),
        LOOM_MOVEMENT_ASYNC_DESCRIPTOR(
            LOOM_OP_KERNEL_ASYNC_CLUSTER_GATHER,
            LOOM_MOVEMENT_KIND_KERNEL_ASYNC_CLUSTER_GATHER, 0,
            LOOM_MOVEMENT_LAYOUT_CLUSTER_GATHER,
            LOOM_MOVEMENT_ASYNC_TRANSFER_EQUAL_ENDPOINTS, 0, 1,
            LOOM_MOVEMENT_ABSENT_INDEX, LOOM_MOVEMENT_ABSENT_INDEX, 2,
            LOOM_MOVEMENT_ABSENT_INDEX, UINT8_MAX),
        LOOM_MOVEMENT_ASYNC_DESCRIPTOR(
            LOOM_OP_KERNEL_ASYNC_CLUSTER_GATHER_MASK,
            LOOM_MOVEMENT_KIND_KERNEL_ASYNC_CLUSTER_GATHER_MASK,
            LOOM_MOVEMENT_REQUEST_MASKED, LOOM_MOVEMENT_LAYOUT_CLUSTER_GATHER,
            LOOM_MOVEMENT_ASYNC_TRANSFER_EQUAL_ENDPOINTS, 0, 1, 3,
            LOOM_MOVEMENT_ABSENT_INDEX, 2, LOOM_MOVEMENT_ABSENT_INDEX,
            UINT8_MAX),
        LOOM_MOVEMENT_ASYNC_DESCRIPTOR(
            LOOM_OP_KERNEL_ASYNC_TENSOR_LOAD_TO_LDS,
            LOOM_MOVEMENT_KIND_KERNEL_ASYNC_TENSOR_LOAD_TO_LDS, 0,
            LOOM_MOVEMENT_LAYOUT_TENSOR_TILE,
            LOOM_MOVEMENT_ASYNC_TRANSFER_EQUAL_ENDPOINTS, 0, 1,
            LOOM_MOVEMENT_ABSENT_INDEX, 2, LOOM_MOVEMENT_ABSENT_INDEX,
            LOOM_MOVEMENT_ABSENT_INDEX,
            LOOM_KERNEL_DIRECTION_GLOBAL_TO_WORKGROUP),
        LOOM_MOVEMENT_ASYNC_DESCRIPTOR(
            LOOM_OP_KERNEL_ASYNC_TENSOR_STORE_FROM_LDS,
            LOOM_MOVEMENT_KIND_KERNEL_ASYNC_TENSOR_STORE_FROM_LDS, 0,
            LOOM_MOVEMENT_LAYOUT_TENSOR_TILE,
            LOOM_MOVEMENT_ASYNC_TRANSFER_EQUAL_ENDPOINTS, 0, 1,
            LOOM_MOVEMENT_ABSENT_INDEX, 2, LOOM_MOVEMENT_ABSENT_INDEX,
            LOOM_MOVEMENT_ABSENT_INDEX,
            LOOM_KERNEL_DIRECTION_WORKGROUP_TO_GLOBAL),
};

#undef LOOM_MOVEMENT_ASYNC_DESCRIPTOR

static const loom_movement_async_descriptor_t*
loom_movement_async_descriptor_for(loom_op_kind_t op_kind) {
  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(loom_movement_async_descriptors); ++i) {
    const loom_movement_async_descriptor_t* descriptor =
        &loom_movement_async_descriptors[i];
    if (descriptor->op_kind == op_kind) {
      return descriptor;
    }
  }
  return NULL;
}

static bool loom_movement_set_equal_endpoint_transfer(
    loom_movement_request_t* request) {
  if (!iree_any_bit_set(request->source.flags,
                        LOOM_MOVEMENT_ENDPOINT_STATIC_LENGTH) ||
      !iree_any_bit_set(request->dest.flags,
                        LOOM_MOVEMENT_ENDPOINT_STATIC_LENGTH)) {
    return false;
  }
  if (request->source.static_byte_length != request->dest.static_byte_length) {
    return false;
  }
  request->flags |= LOOM_MOVEMENT_REQUEST_STATIC_TRANSFER;
  request->transferred_byte_count = request->source.static_byte_length;
  return true;
}

static bool loom_movement_set_source_length_transfer(
    loom_movement_request_t* request) {
  if (!iree_any_bit_set(request->source.flags,
                        LOOM_MOVEMENT_ENDPOINT_STATIC_LENGTH)) {
    return false;
  }
  request->flags |= LOOM_MOVEMENT_REQUEST_STATIC_TRANSFER;
  request->transferred_byte_count = request->source.static_byte_length;
  return true;
}

static bool loom_movement_apply_async_transfer(
    const loom_movement_async_descriptor_t* descriptor,
    loom_movement_request_t* request) {
  switch (descriptor->transfer_mode) {
    case LOOM_MOVEMENT_ASYNC_TRANSFER_EQUAL_ENDPOINTS:
      return loom_movement_set_equal_endpoint_transfer(request);
    case LOOM_MOVEMENT_ASYNC_TRANSFER_SOURCE_LENGTH:
      return loom_movement_set_source_length_transfer(request);
  }
  return false;
}

static iree_status_t loom_movement_describe_async(
    loom_movement_analysis_t* analysis, const loom_op_t* op,
    const loom_movement_async_descriptor_t* descriptor,
    loom_movement_request_t* request, loom_movement_diagnostic_t* diagnostic,
    bool* out_described) {
  request->kind = descriptor->movement_kind;
  request->flags |= descriptor->request_flags | LOOM_MOVEMENT_REQUEST_ASYNC;
  request->schema_kind = LOOM_MOVEMENT_SCHEMA_BYTE_PRESERVING;
  request->layout_kind = descriptor->layout_kind;

  const loom_value_id_t* operands = loom_op_const_operands(op);
  request->source.value_id = operands[descriptor->source_operand_index];
  request->dest.value_id = operands[descriptor->dest_operand_index];
  if (descriptor->mask_operand_index != LOOM_MOVEMENT_ABSENT_INDEX) {
    request->mask_value_id = operands[descriptor->mask_operand_index];
  }
  if (descriptor->descriptor_operand_index != LOOM_MOVEMENT_ABSENT_INDEX) {
    request->descriptor_value_id =
        operands[descriptor->descriptor_operand_index];
  }
  if (descriptor->cluster_mask_operand_index != LOOM_MOVEMENT_ABSENT_INDEX) {
    request->cluster_mask_value_id =
        operands[descriptor->cluster_mask_operand_index];
  }

  const loom_attribute_t* attrs = loom_op_const_attrs(op);
  if (!loom_movement_cache_policy_from_attrs(
          attrs[0], attrs[1], &request->cache_policy, diagnostic)) {
    return loom_movement_describe_result(false, out_described);
  }
  if (descriptor->direction_attr_index != LOOM_MOVEMENT_ABSENT_INDEX) {
    request->flags |= LOOM_MOVEMENT_REQUEST_HAS_DIRECTION;
    request->direction =
        (uint8_t)loom_attr_as_enum(attrs[descriptor->direction_attr_index]);
  } else if (descriptor->direction_value != UINT8_MAX) {
    request->flags |= LOOM_MOVEMENT_REQUEST_HAS_DIRECTION;
    request->direction = descriptor->direction_value;
  }

  IREE_RETURN_IF_ERROR(loom_movement_endpoint_for_view(
      analysis, request->source.value_id, LOOM_MOVEMENT_ENDPOINT_READ,
      &request->source));
  IREE_RETURN_IF_ERROR(loom_movement_endpoint_for_view(
      analysis, request->dest.value_id, LOOM_MOVEMENT_ENDPOINT_WRITE,
      &request->dest));
  if (request->source.kind != LOOM_MOVEMENT_ENDPOINT_VIEW ||
      request->dest.kind != LOOM_MOVEMENT_ENDPOINT_VIEW) {
    diagnostic->rejection_bits |= LOOM_MOVEMENT_REJECTION_ENDPOINT;
    return loom_movement_describe_result(false, out_described);
  }
  if (!loom_movement_apply_async_transfer(descriptor, request)) {
    diagnostic->rejection_bits |= LOOM_MOVEMENT_REJECTION_FOOTPRINT;
    return loom_movement_describe_result(false, out_described);
  }
  request->flags |= LOOM_MOVEMENT_REQUEST_ASYNC_ELIGIBLE;
  return loom_movement_describe_result(true, out_described);
}

//===----------------------------------------------------------------------===//
// Public API
//===----------------------------------------------------------------------===//

iree_status_t loom_movement_analysis_initialize(
    loom_value_fact_table_t* fact_table,
    const loom_local_value_domain_t* value_domain,
    iree_arena_allocator_t* arena, loom_movement_analysis_t* out_analysis) {
  *out_analysis = (loom_movement_analysis_t){0};
  out_analysis->module = value_domain->module;
  out_analysis->fact_table = fact_table;
  out_analysis->arena = arena;
  return loom_view_region_table_initialize(fact_table, value_domain, arena,
                                           &out_analysis->view_regions);
}

iree_status_t loom_movement_analysis_analyze(
    loom_movement_analysis_t* analysis) {
  return loom_view_region_table_analyze(&analysis->view_regions);
}

bool loom_movement_op_kind_is_async(loom_op_kind_t op_kind) {
  return loom_movement_async_descriptor_for(op_kind) != NULL;
}

uint64_t loom_movement_endpoint_minimum_byte_alignment(
    const loom_movement_endpoint_t* endpoint) {
  if (endpoint->kind != LOOM_MOVEMENT_ENDPOINT_VIEW ||
      endpoint->root_minimum_alignment == 0) {
    return 0;
  }
  if (iree_any_bit_set(endpoint->flags, LOOM_MOVEMENT_ENDPOINT_STATIC_BEGIN) &&
      endpoint->static_begin_byte_offset == 0) {
    return endpoint->root_minimum_alignment;
  }
  if (endpoint->minimum_alignment == 0) {
    return 0;
  }
  return endpoint->root_minimum_alignment < endpoint->minimum_alignment
             ? endpoint->root_minimum_alignment
             : endpoint->minimum_alignment;
}

iree_status_t loom_movement_request_describe_op(
    loom_movement_analysis_t* analysis, const loom_op_t* op,
    loom_movement_request_t* out_request,
    loom_movement_diagnostic_t* out_diagnostic, bool* out_described) {
  *out_described = false;
  loom_movement_request_initialize(op, out_request, out_diagnostic);

  const loom_movement_vector_descriptor_t* vector_descriptor =
      loom_movement_vector_descriptor_for(op->kind);
  if (vector_descriptor) {
    return loom_movement_describe_vector(analysis, op, vector_descriptor,
                                         out_request, out_diagnostic,
                                         out_described);
  }

  const loom_movement_async_descriptor_t* async_descriptor =
      loom_movement_async_descriptor_for(op->kind);
  if (async_descriptor) {
    return loom_movement_describe_async(analysis, op, async_descriptor,
                                        out_request, out_diagnostic,
                                        out_described);
  }

  out_diagnostic->rejection_bits |= LOOM_MOVEMENT_REJECTION_UNSUPPORTED_OP;
  return loom_movement_describe_result(false, out_described);
}

iree_string_view_t loom_movement_rejection_detail(
    loom_movement_rejection_flags_t rejection_bits) {
  if (iree_any_bit_set(rejection_bits,
                       LOOM_MOVEMENT_REJECTION_UNSUPPORTED_OP)) {
    return IREE_SV("source op is not a supported movement operation");
  }
  if (iree_any_bit_set(rejection_bits, LOOM_MOVEMENT_REJECTION_ENDPOINT)) {
    return IREE_SV("movement endpoint is not a known view or register");
  }
  if (iree_any_bit_set(rejection_bits, LOOM_MOVEMENT_REJECTION_VECTOR_ACCESS)) {
    return IREE_SV("vector view and payload types do not form an access");
  }
  if (iree_any_bit_set(rejection_bits, LOOM_MOVEMENT_REJECTION_LAYOUT)) {
    return IREE_SV("movement memory layout is unknown");
  }
  if (iree_any_bit_set(rejection_bits, LOOM_MOVEMENT_REJECTION_ELEMENT_WIDTH)) {
    return IREE_SV("movement element width is not byte-addressable");
  }
  if (iree_any_bit_set(rejection_bits, LOOM_MOVEMENT_REJECTION_LANE_COUNT)) {
    return IREE_SV("movement vector lane count is not static");
  }
  if (iree_any_bit_set(rejection_bits, LOOM_MOVEMENT_REJECTION_FOOTPRINT)) {
    return IREE_SV("movement byte footprint is not statically proven");
  }
  if (iree_any_bit_set(rejection_bits, LOOM_MOVEMENT_REJECTION_CACHE_POLICY)) {
    return IREE_SV("movement cache policy is malformed");
  }
  return IREE_SV("movement request is not representable");
}
