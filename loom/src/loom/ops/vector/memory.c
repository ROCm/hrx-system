// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ops/vector/memory.h"

#include "loom/ir/scalar_type.h"
#include "loom/ops/encoding/storage.h"
#include "loom/ops/vector/fragment.h"
#include "loom/ops/vector/ops.h"
#include "loom/util/fact_table.h"

static loom_vector_memory_layout_kind_t loom_vector_memory_layout_kind(
    loom_value_fact_address_layout_t layout) {
  switch (layout.kind) {
    case LOOM_VALUE_FACT_ADDRESS_LAYOUT_DENSE:
      return LOOM_VECTOR_MEMORY_LAYOUT_DENSE;
    case LOOM_VALUE_FACT_ADDRESS_LAYOUT_STRIDED:
      return LOOM_VECTOR_MEMORY_LAYOUT_STRIDED;
    case LOOM_VALUE_FACT_ADDRESS_LAYOUT_UNKNOWN:
      return LOOM_VECTOR_MEMORY_LAYOUT_UNKNOWN;
  }
  return LOOM_VECTOR_MEMORY_LAYOUT_UNKNOWN;
}

static bool loom_vector_memory_query_layout(
    const loom_fact_context_t* context, const loom_module_t* module,
    loom_type_t view_type, loom_value_facts_t* stride_storage,
    iree_host_size_t stride_capacity,
    loom_value_fact_address_layout_t* out_layout) {
  return loom_encoding_query_type_address_layout(
      context, module, view_type, stride_storage, stride_capacity, out_layout);
}

bool loom_vector_memory_access_describe(
    const loom_fact_context_t* context, const loom_module_t* module,
    loom_type_t view_type, loom_type_t vector_type,
    loom_vector_memory_access_t* out_access) {
  if (!out_access) return false;
  *out_access = (loom_vector_memory_access_t){0};

  if (!loom_type_is_view(view_type) || !loom_type_is_vector(vector_type)) {
    return false;
  }

  uint8_t view_rank = loom_type_rank(view_type);
  uint8_t vector_rank = loom_type_rank(vector_type);
  if (vector_rank == 0 || vector_rank > view_rank) return false;

  int32_t element_bit_count =
      loom_scalar_type_bitwidth(loom_type_element_type(view_type));
  int64_t static_element_byte_count = -1;
  if (element_bit_count > 0 && (element_bit_count % 8) == 0) {
    static_element_byte_count = element_bit_count / 8;
  }

  loom_value_facts_t layout_strides[LOOM_ENCODING_ADDRESS_LAYOUT_MAX_RANK] = {
      0};
  loom_value_fact_address_layout_t layout_summary = {0};
  (void)loom_vector_memory_query_layout(
      context, module, view_type, layout_strides,
      IREE_ARRAYSIZE(layout_strides), &layout_summary);
  loom_encoding_address_layout_operands_t layout_operands = {0};
  bool layout_has_non_exact_stride = false;
  if (layout_summary.kind == LOOM_VALUE_FACT_ADDRESS_LAYOUT_STRIDED &&
      layout_summary.strides != NULL) {
    for (uint8_t i = 0; i < layout_summary.rank; ++i) {
      if (!loom_value_facts_is_exact(layout_summary.strides[i])) {
        layout_has_non_exact_stride = true;
        break;
      }
    }
  }
  if (context != NULL && layout_has_non_exact_stride) {
    (void)loom_encoding_query_type_address_layout_operands(module, view_type,
                                                           &layout_operands);
  }
  *out_access = (loom_vector_memory_access_t){
      .view_type = view_type,
      .vector_type = vector_type,
      .view_rank = view_rank,
      .vector_rank = vector_rank,
      .first_vector_axis = (uint8_t)(view_rank - vector_rank),
      .element_bit_count = element_bit_count,
      .static_element_byte_count = static_element_byte_count,
      .layout_kind = loom_vector_memory_layout_kind(layout_summary),
      .layout_summary = layout_summary,
      .layout_operands = layout_operands,
  };
  if (layout_summary.strides == layout_strides) {
    for (uint8_t i = 0; i < layout_summary.rank; ++i) {
      out_access->layout_strides[i] = layout_summary.strides[i];
    }
    out_access->layout_summary.strides = out_access->layout_strides;
  }
  return true;
}

static bool loom_vector_memory_value_type(const loom_module_t* module,
                                          const loom_op_t* op,
                                          loom_memory_access_t access,
                                          loom_type_t* out_type) {
  *out_type = loom_type_none();
  loom_value_id_t value = loom_memory_access_value(access);
  if (value != LOOM_VALUE_ID_INVALID) {
    *out_type = loom_module_value_type(module, value);
    return true;
  }
  if (op->result_count == 0) return false;
  const loom_value_id_t result = loom_op_const_results(op)[0];
  if (result == LOOM_VALUE_ID_INVALID) return false;
  *out_type = loom_module_value_type(module, result);
  return true;
}

static bool loom_vector_memory_fragment_footprint_type(
    const loom_module_t* module, loom_type_t view_type, const loom_op_t* op,
    loom_overflow_dim_t dimension_storage[3], loom_type_t* out_type) {
  *out_type = loom_type_none();
  if (loom_type_rank(view_type) < 2) return false;

  loom_value_id_t blocks = LOOM_VALUE_ID_INVALID;
  loom_value_id_t rows = LOOM_VALUE_ID_INVALID;
  loom_value_id_t columns = LOOM_VALUE_ID_INVALID;
  switch (op->kind) {
    case LOOM_OP_VECTOR_FRAGMENT_LOAD:
      blocks = loom_vector_fragment_load_blocks(op);
      rows = loom_vector_fragment_load_rows(op);
      columns = loom_vector_fragment_load_columns(op);
      break;
    case LOOM_OP_VECTOR_FRAGMENT_STORE:
      blocks = loom_vector_fragment_store_blocks(op);
      rows = loom_vector_fragment_store_rows(op);
      columns = loom_vector_fragment_store_columns(op);
      break;
    default:
      return false;
  }

  const uint8_t shape_rank = blocks == LOOM_VALUE_ID_INVALID ? 2 : 3;
  loom_value_id_t shape[3] = {blocks, rows, columns};
  const uint8_t shape_offset = (uint8_t)(3 - shape_rank);
  if (loom_type_rank(view_type) < shape_rank) {
    return false;
  }
  for (uint8_t i = shape_offset; i < 3; ++i) {
    if (shape[i] == LOOM_VALUE_ID_INVALID || shape[i] >= module->values.count) {
      return false;
    }
  }
  if (shape_rank == 2) {
    *out_type = loom_type_shaped_2d(
        LOOM_TYPE_VECTOR, loom_type_element_type(view_type),
        loom_dim_pack_dynamic(rows), loom_dim_pack_dynamic(columns),
        /*encoding_id=*/0);
    return true;
  }
  for (uint8_t i = 0; i < shape_rank; ++i) {
    dimension_storage[i] = loom_dim_pack_dynamic(shape[i]);
  }
  out_type->header =
      loom_type_make_header(LOOM_TYPE_VECTOR, loom_type_element_type(view_type),
                            shape_rank, /*flags=*/0);
  out_type->dims[0] = (uint64_t)(uintptr_t)dimension_storage;
  return true;
}

static loom_vector_memory_footprint_axis_scale_t
loom_vector_memory_fragment_footprint_axis_scale(
    const loom_fact_context_t* context, const loom_module_t* module,
    loom_type_t view_type, const loom_op_t* op) {
  loom_vector_memory_footprint_axis_scale_t scale = {
      .vector_axis = UINT8_MAX,
  };
  loom_value_fact_storage_schema_t storage_schema = {0};
  if (!loom_encoding_query_type_storage_schema(context, module, view_type,
                                               &storage_schema)) {
    return scale;
  }
  const loom_value_fact_encoded_operand_schema_t operand =
      storage_schema.encoded_operand;
  if (operand.sparsity_policy !=
          LOOM_VALUE_FACT_SPARSITY_POLICY_N_M_STRUCTURED ||
      !loom_value_fact_encoded_operand_schema_sparsity_is_complete(operand)) {
    return scale;
  }

  loom_vector_role_t role = LOOM_VECTOR_ROLE_COUNT_;
  bool has_blocks = false;
  switch (op->kind) {
    case LOOM_OP_VECTOR_FRAGMENT_LOAD:
      role = loom_vector_fragment_load_role(op);
      has_blocks = loom_vector_fragment_load_blocks_is_present(op);
      break;
    case LOOM_OP_VECTOR_FRAGMENT_STORE:
      role = loom_vector_fragment_store_role(op);
      has_blocks = loom_vector_fragment_store_blocks_is_present(op);
      break;
    default:
      return scale;
  }
  static const uint8_t kReductionVectorAxes[LOOM_VECTOR_ROLE_COUNT_] = {
      [LOOM_VECTOR_ROLE_LHS] = 1,
      [LOOM_VECTOR_ROLE_RHS] = 0,
      [LOOM_VECTOR_ROLE_INIT] = UINT8_MAX,
      [LOOM_VECTOR_ROLE_RESULT] = UINT8_MAX,
  };
  if ((iree_host_size_t)role >= IREE_ARRAYSIZE(kReductionVectorAxes)) {
    return scale;
  }
  scale.vector_axis = kReductionVectorAxes[role];
  if (scale.vector_axis == UINT8_MAX) {
    return scale;
  }
  if (has_blocks) {
    ++scale.vector_axis;
  }
  scale.storage_element_count = operand.sparsity_group.nonzero_element_count;
  scale.logical_element_count = operand.sparsity_group.element_count;
  return scale;
}

static loom_vector_memory_footprint_kind_t
loom_vector_memory_classify_footprint_kind(const loom_op_t* op,
                                           loom_memory_access_t access) {
  switch (op->kind) {
    case LOOM_OP_VECTOR_FRAGMENT_LOAD:
    case LOOM_OP_VECTOR_FRAGMENT_STORE:
      return LOOM_VECTOR_MEMORY_FOOTPRINT_FRAGMENT;
    case LOOM_OP_VECTOR_LOAD_EXPAND:
    case LOOM_OP_VECTOR_STORE_COMPRESS:
      return LOOM_VECTOR_MEMORY_FOOTPRINT_COMPRESS_EXPAND;
    default:
      break;
  }

  const bool has_mask =
      loom_memory_access_mask(access) != LOOM_VALUE_ID_INVALID;
  const bool has_offsets =
      loom_memory_access_offsets(access) != LOOM_VALUE_ID_INVALID;
  const bool has_atomic = loom_memory_access_operation_kind_is_atomic(
      loom_memory_access_operation_kind(access));
  if (has_atomic) {
    return has_mask ? LOOM_VECTOR_MEMORY_FOOTPRINT_MASKED_ATOMIC_PER_LANE
                    : LOOM_VECTOR_MEMORY_FOOTPRINT_ATOMIC_PER_LANE;
  }
  if (has_offsets) {
    return has_mask ? LOOM_VECTOR_MEMORY_FOOTPRINT_MASKED_PER_LANE_OFFSET
                    : LOOM_VECTOR_MEMORY_FOOTPRINT_PER_LANE_OFFSET;
  }
  return has_mask ? LOOM_VECTOR_MEMORY_FOOTPRINT_MASKED_DENSE
                  : LOOM_VECTOR_MEMORY_FOOTPRINT_DENSE;
}

loom_vector_memory_footprint_kind_t loom_vector_memory_op_footprint_kind(
    const loom_module_t* module, const loom_op_t* op) {
  if (!module || !op || loom_op_dialect_id(op->kind) != LOOM_DIALECT_VECTOR) {
    return LOOM_VECTOR_MEMORY_FOOTPRINT_NONE;
  }
  loom_memory_access_t access = loom_memory_access_cast(module, op);
  if (!loom_memory_access_isa(access)) {
    return LOOM_VECTOR_MEMORY_FOOTPRINT_NONE;
  }
  return loom_vector_memory_classify_footprint_kind(op, access);
}

bool loom_vector_memory_footprint_describe(
    const loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, loom_vector_memory_footprint_t* out_footprint) {
  if (!out_footprint) return false;
  *out_footprint = (loom_vector_memory_footprint_t){
      .kind = LOOM_VECTOR_MEMORY_FOOTPRINT_NONE,
      .view = LOOM_VALUE_ID_INVALID,
      .value = LOOM_VALUE_ID_INVALID,
      .mask = LOOM_VALUE_ID_INVALID,
      .passthrough = LOOM_VALUE_ID_INVALID,
      .offsets = LOOM_VALUE_ID_INVALID,
      .axis_scale.vector_axis = UINT8_MAX,
  };
  if (!module || !op || loom_op_dialect_id(op->kind) != LOOM_DIALECT_VECTOR) {
    return false;
  }

  loom_memory_access_t access = loom_memory_access_cast(module, op);
  if (!loom_memory_access_isa(access)) return false;

  const loom_value_id_t view = loom_memory_access_view(access);
  if (view >= module->values.count) return false;
  const loom_type_t view_type = loom_module_value_type(module, view);

  *out_footprint = (loom_vector_memory_footprint_t){
      .kind = loom_vector_memory_classify_footprint_kind(op, access),
      .access = access,
      .view = view,
      .value = loom_memory_access_value(access),
      .mask = loom_memory_access_mask(access),
      .passthrough = loom_memory_access_passthrough(access),
      .offsets = loom_memory_access_offsets(access),
      .dynamic_indices = loom_memory_access_dynamic_indices(access),
      .static_indices = loom_memory_access_static_indices(access),
      .view_type = view_type,
      .vector_type = loom_type_none(),
      .axis_scale.vector_axis = UINT8_MAX,
  };
  loom_vector_memory_footprint_t* footprint = out_footprint;

  const loom_trait_flags_t traits = loom_op_effective_traits(module, op);
  if (loom_traits_may_read(traits)) {
    footprint->flags |= LOOM_VECTOR_MEMORY_FOOTPRINT_READS;
  }
  if (loom_traits_may_write(traits)) {
    footprint->flags |= LOOM_VECTOR_MEMORY_FOOTPRINT_WRITES;
  }

  if (footprint->kind == LOOM_VECTOR_MEMORY_FOOTPRINT_FRAGMENT) {
    if (!loom_vector_memory_fragment_footprint_type(
            module, view_type, op, footprint->fragment_dimensions,
            &footprint->vector_type)) {
      return false;
    }
    footprint->axis_scale = loom_vector_memory_fragment_footprint_axis_scale(
        context, module, view_type, op);
  } else if (!loom_vector_memory_value_type(module, op, access,
                                            &footprint->vector_type)) {
    return false;
  }
  if (!loom_vector_memory_access_describe(context, module, view_type,
                                          footprint->vector_type,
                                          &footprint->vector_access)) {
    return false;
  }
  return true;
}

bool loom_vector_memory_footprint_static_extents(
    const loom_vector_memory_footprint_t* footprint, int64_t* out_extents,
    iree_host_size_t capacity) {
  if (!footprint || !out_extents ||
      footprint->kind == LOOM_VECTOR_MEMORY_FOOTPRINT_NONE ||
      footprint->kind == LOOM_VECTOR_MEMORY_FOOTPRINT_FRAGMENT ||
      capacity < footprint->vector_access.view_rank) {
    return false;
  }
  for (uint8_t axis = 0; axis < footprint->vector_access.view_rank; ++axis) {
    if (!loom_vector_memory_access_static_axis_extent(
            &footprint->vector_access, axis, &out_extents[axis])) {
      return false;
    }
  }
  return true;
}

bool loom_vector_memory_cache_policy_from_attrs(
    loom_attribute_t cache_scope_attr, loom_attribute_t cache_temporal_attr,
    loom_vector_memory_cache_policy_t* out_policy) {
  *out_policy = (loom_vector_memory_cache_policy_t){0};
  if (!loom_attr_is_absent(cache_scope_attr)) {
    if (cache_scope_attr.kind != LOOM_ATTR_ENUM) return false;
    out_policy->build_flags |= LOOM_VECTOR_MEMORY_CACHE_POLICY_BUILD_FLAG_SCOPE;
    out_policy->cache_scope = loom_attr_as_enum(cache_scope_attr);
  }
  if (!loom_attr_is_absent(cache_temporal_attr)) {
    if (cache_temporal_attr.kind != LOOM_ATTR_ENUM) return false;
    out_policy->build_flags |=
        LOOM_VECTOR_MEMORY_CACHE_POLICY_BUILD_FLAG_TEMPORAL;
    out_policy->cache_temporal = loom_attr_as_enum(cache_temporal_attr);
  }
  return true;
}

bool loom_vector_memory_cache_policy_from_op(
    const loom_module_t* module, const loom_op_t* op,
    loom_vector_memory_cache_policy_t* out_policy) {
  *out_policy = (loom_vector_memory_cache_policy_t){0};
  if (!module || !op) {
    return false;
  }
  loom_memory_access_t access = loom_memory_access_cast(module, op);
  if (!loom_memory_access_isa(access)) {
    return false;
  }
  return loom_vector_memory_cache_policy_from_attrs(
      loom_memory_access_cache_scope(access),
      loom_memory_access_cache_temporal(access), out_policy);
}

bool loom_vector_memory_access_static_axis_extent(
    const loom_vector_memory_access_t* access, uint8_t view_axis,
    int64_t* out_extent) {
  if (!access || !out_extent || view_axis >= access->view_rank) return false;
  if (view_axis < access->first_vector_axis) {
    *out_extent = 1;
    return true;
  }

  uint8_t vector_axis = view_axis - access->first_vector_axis;
  if (loom_type_dim_is_dynamic_at(access->vector_type, vector_axis)) {
    return false;
  }
  *out_extent = loom_type_dim_static_size_at(access->vector_type, vector_axis);
  return true;
}

static bool loom_vector_memory_access_static_dense_axis_stride(
    const loom_vector_memory_access_t* access, uint8_t view_axis,
    int64_t* out_stride) {
  int64_t stride = 1;
  for (int16_t axis = (int16_t)access->view_rank - 1; axis > view_axis;
       --axis) {
    if (loom_type_dim_is_dynamic_at(access->view_type, (uint8_t)axis)) {
      return false;
    }
    int64_t dimension_size =
        loom_type_dim_static_size_at(access->view_type, (uint8_t)axis);
    if (dimension_size < 0 ||
        !iree_checked_mul_i64(stride, dimension_size, &stride)) {
      return false;
    }
  }
  *out_stride = stride;
  return true;
}

bool loom_vector_memory_access_static_axis_stride(
    const loom_vector_memory_access_t* access, uint8_t view_axis,
    int64_t* out_stride) {
  if (!access || !out_stride || view_axis >= access->view_rank) return false;

  switch (access->layout_kind) {
    case LOOM_VECTOR_MEMORY_LAYOUT_DENSE:
      return loom_vector_memory_access_static_dense_axis_stride(
          access, view_axis, out_stride);
    case LOOM_VECTOR_MEMORY_LAYOUT_STRIDED: {
      loom_value_fact_address_layout_t layout = access->layout_summary;
      if (layout.kind != LOOM_VALUE_FACT_ADDRESS_LAYOUT_STRIDED ||
          view_axis >= layout.rank || !layout.strides) {
        return false;
      }
      loom_value_facts_t stride = layout.strides[view_axis];
      if (!loom_value_facts_is_exact(stride) ||
          loom_value_facts_is_float(stride)) {
        return false;
      }
      *out_stride = stride.range_lo;
      return true;
    }
    case LOOM_VECTOR_MEMORY_LAYOUT_UNKNOWN:
      return false;
  }
  return false;
}

bool loom_vector_memory_access_static_lane_element_offset(
    const loom_vector_memory_access_t* access, loom_attribute_t static_indices,
    const int64_t* lane_indices, uint8_t lane_index_count,
    int64_t* out_element_offset) {
  if (!access || !out_element_offset) return false;
  if (static_indices.kind != LOOM_ATTR_I64_ARRAY) return false;
  if (static_indices.count != access->view_rank) return false;
  if (lane_index_count != access->vector_rank) return false;
  if (lane_index_count > 0 && !lane_indices) return false;

  int64_t element_offset = 0;
  for (uint8_t view_axis = 0; view_axis < access->view_rank; ++view_axis) {
    int64_t origin = static_indices.i64_array[view_axis];
    if (origin == INT64_MIN) return false;

    int64_t lane_index = 0;
    if (view_axis >= access->first_vector_axis) {
      uint8_t vector_axis = view_axis - access->first_vector_axis;
      lane_index = lane_indices[vector_axis];
      if (lane_index < 0) return false;
      if (!loom_type_dim_is_dynamic_at(access->vector_type, vector_axis)) {
        int64_t lane_bound =
            loom_type_dim_static_size_at(access->vector_type, vector_axis);
        if (lane_bound < 0 || lane_index >= lane_bound) return false;
      }
    }

    int64_t logical_index = 0;
    if (!iree_checked_add_i64(origin, lane_index, &logical_index)) {
      return false;
    }
    if (logical_index == 0) continue;

    int64_t stride = 0;
    if (!loom_vector_memory_access_static_axis_stride(access, view_axis,
                                                      &stride)) {
      return false;
    }

    if (!iree_checked_mul_add_i64(element_offset, logical_index, stride,
                                  &element_offset)) {
      return false;
    }
  }

  *out_element_offset = element_offset;
  return true;
}

bool loom_vector_memory_access_static_lane_byte_offset(
    const loom_vector_memory_access_t* access, loom_attribute_t static_indices,
    const int64_t* lane_indices, uint8_t lane_index_count,
    int64_t* out_byte_offset) {
  if (!access || !out_byte_offset || access->static_element_byte_count < 0) {
    return false;
  }

  int64_t element_offset = 0;
  if (!loom_vector_memory_access_static_lane_element_offset(
          access, static_indices, lane_indices, lane_index_count,
          &element_offset)) {
    return false;
  }
  return iree_checked_mul_i64(element_offset, access->static_element_byte_count,
                              out_byte_offset);
}

bool loom_vector_memory_access_linear_element_end_facts(
    const loom_vector_memory_access_t* access,
    const loom_value_facts_t* axis_exclusive_end_facts,
    loom_value_facts_t* out_element_end_facts) {
  loom_value_facts_t maximum_element_offset = loom_value_facts_exact_i64(0);
  const loom_value_facts_t one = loom_value_facts_exact_i64(1);
  for (uint8_t axis = 0; axis < access->view_rank; ++axis) {
    int64_t stride = 0;
    if (!loom_vector_memory_access_static_axis_stride(access, axis, &stride)) {
      return false;
    }
    loom_value_facts_t maximum_index = loom_value_facts_unknown();
    loom_value_facts_subi(&axis_exclusive_end_facts[axis], &one,
                          &maximum_index);
    const loom_value_facts_t stride_facts = loom_value_facts_exact_i64(stride);
    loom_value_facts_t contribution = loom_value_facts_unknown();
    loom_value_facts_muli(&maximum_index, &stride_facts, &contribution);
    loom_value_facts_addi(&maximum_element_offset, &contribution,
                          &maximum_element_offset);
  }
  loom_value_facts_addi(&maximum_element_offset, &one, out_element_end_facts);
  return true;
}
