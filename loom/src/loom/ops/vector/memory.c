// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ops/vector/memory.h"

#include "loom/ir/scalar_type.h"
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
  if (!module || !loom_type_has_encoding(view_type)) return false;
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
  };
  if (layout_summary.strides == layout_strides) {
    for (uint8_t i = 0; i < layout_summary.rank; ++i) {
      out_access->layout_strides[i] = layout_summary.strides[i];
    }
    out_access->layout_summary.strides = out_access->layout_strides;
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
