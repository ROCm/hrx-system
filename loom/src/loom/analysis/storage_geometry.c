// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/storage_geometry.h"

#include "loom/ops/encoding/storage.h"

bool loom_storage_geometry_query(const loom_fact_context_t* context,
                                 const loom_module_t* module, loom_type_t type,
                                 loom_storage_geometry_t* out_geometry) {
  loom_value_facts_t stride_storage[LOOM_TYPE_MAX_RANK];
  loom_value_fact_address_layout_t layout = {0};
  if (!loom_encoding_query_type_address_layout(
          context, module, type, stride_storage, IREE_ARRAYSIZE(stride_storage),
          &layout)) {
    return false;
  }
  const int32_t element_bit_count =
      loom_fact_context_element_bitwidth(context, type);
  if (element_bit_count <= 0) {
    return false;
  }
  loom_storage_geometry_t geometry = {
      .layout_kind = layout.kind,
      .element_bit_count = (uint16_t)element_bit_count,
      .rank = loom_type_rank(type),
  };
  uint64_t dense_stride = 1;
  for (uint8_t i = geometry.rank; i-- > 0;) {
    int64_t extent = 0;
    if (loom_type_dim_is_dynamic_at(type, i)) {
      loom_value_facts_t element_facts = loom_value_facts_unknown();
      if (!context || !context->table ||
          !loom_value_facts_query_all_equal_element(
              context,
              loom_value_fact_table_lookup(context->table,
                                           loom_type_dim_value_id_at(type, i)),
              &element_facts) ||
          !loom_value_facts_as_exact_i64(element_facts, &extent)) {
        return false;
      }
    } else {
      extent = loom_type_dim_static_size_at(type, i);
    }
    if (extent < 0) {
      return false;
    }
    geometry.axes[i].extent = (uint64_t)extent;
    if (layout.kind == LOOM_VALUE_FACT_ADDRESS_LAYOUT_DENSE) {
      geometry.axes[i].element_stride = dense_stride;
      if (i > 0 && !iree_checked_mul_u64(dense_stride, (uint64_t)extent,
                                         &dense_stride)) {
        return false;
      }
    } else {
      int64_t stride = 0;
      if (!loom_value_facts_as_exact_i64(layout.strides[i], &stride) ||
          stride < 0) {
        return false;
      }
      geometry.axes[i].element_stride = (uint64_t)stride;
    }
  }
  *out_geometry = geometry;
  return true;
}

bool loom_storage_geometry_span_prepend(loom_storage_geometry_axis_t axis,
                                        loom_storage_geometry_span_t* span) {
  if (axis.extent == 0 || span->element_count == 0) {
    *span = (loom_storage_geometry_span_t){.dense = true};
    return true;
  }
  loom_storage_geometry_span_t result = {
      .dense = span->dense &&
               (axis.extent == 1 || axis.element_stride == span->element_count),
  };
  uint64_t contribution = 0;
  if (!iree_checked_mul_u64(span->element_count, axis.extent,
                            &result.element_count) ||
      !iree_checked_mul_u64(axis.extent - 1, axis.element_stride,
                            &contribution) ||
      !iree_checked_add_u64(span->element_span, contribution,
                            &result.element_span)) {
    return false;
  }
  *span = result;
  return true;
}

bool loom_storage_geometry_measure(const loom_storage_geometry_t* geometry,
                                   uint8_t first_axis,
                                   loom_storage_geometry_span_t* out_span) {
  for (uint8_t i = first_axis; i < geometry->rank; ++i) {
    if (geometry->axes[i].extent == 0) {
      *out_span = (loom_storage_geometry_span_t){.dense = true};
      return true;
    }
  }
  loom_storage_geometry_span_t span = {
      .element_count = 1, .element_span = 1, .dense = true};
  for (uint8_t i = geometry->rank; i-- > first_axis;) {
    if (!loom_storage_geometry_span_prepend(geometry->axes[i], &span)) {
      return false;
    }
  }
  *out_span = span;
  return true;
}
