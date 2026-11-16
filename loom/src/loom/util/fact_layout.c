// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/util/fact_layout.h"

#include <string.h>

#include "iree/base/internal/math.h"
#include "loom/util/fact_table.h"

struct loom_value_fact_layout_origins_t {
  // Immutable recipes indexed relative to range.first_value_id; NULL is absent.
  const loom_value_fact_layout_origin_t** entries;
  // Value-ID range covered by the direct entries.
  loom_value_fact_range_t range;
};

static iree_host_size_t loom_value_fact_layout_origin_element_size(
    loom_value_fact_layout_origin_kind_t kind) {
  switch (kind) {
    case LOOM_VALUE_FACT_LAYOUT_ORIGIN_EXPLICIT_STRIDES:
      return sizeof(loom_value_id_t);
    case LOOM_VALUE_FACT_LAYOUT_ORIGIN_DENSE_DIMENSIONS:
      return sizeof(uint64_t);
    case LOOM_VALUE_FACT_LAYOUT_ORIGIN_NONE:
      return 0;
  }
  IREE_ASSERT_UNREACHABLE("invalid layout origin kind");
  return 0;
}

bool loom_value_fact_layout_origin_equal(loom_value_fact_layout_origin_t lhs,
                                         loom_value_fact_layout_origin_t rhs) {
  if (lhs.kind != rhs.kind || lhs.count != rhs.count ||
      lhs.has_physical_encoding_value != rhs.has_physical_encoding_value ||
      (lhs.has_physical_encoding_value &&
       lhs.physical_encoding_value_id != rhs.physical_encoding_value_id)) {
    return false;
  }
  if (lhs.kind == LOOM_VALUE_FACT_LAYOUT_ORIGIN_NONE) {
    return true;
  }
  const iree_host_size_t byte_length =
      lhs.count * loom_value_fact_layout_origin_element_size(lhs.kind);
  return memcmp(lhs.values.data, rhs.values.data, byte_length) == 0;
}

static iree_status_t loom_value_fact_layout_origins_ensure_capacity(
    loom_value_fact_table_t* table, loom_value_id_t value_id) {
  IREE_RETURN_IF_ERROR(loom_value_fact_table_touch(table, value_id));
  if (!table->layout_origins) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate(table->transient_arena,
                                             sizeof(*table->layout_origins),
                                             (void**)&table->layout_origins));
    *table->layout_origins = (loom_value_fact_layout_origins_t){0};
  }
  loom_value_fact_layout_origins_t* origins = table->layout_origins;
  return loom_value_fact_range_grow(table->transient_arena, value_id,
                                    sizeof(*origins->entries), NULL,
                                    &origins->range, (void**)&origins->entries);
}

loom_value_fact_layout_origin_t loom_value_fact_table_query_layout_origin(
    const loom_value_fact_table_t* table, loom_value_id_t value_id) {
  loom_value_fact_table_record_read(table, value_id);
  const loom_value_fact_layout_origins_t* origins = table->layout_origins;
  if (!origins || !loom_value_fact_range_contains(origins->range, value_id) ||
      !origins
           ->entries[loom_value_fact_range_offset(origins->range, value_id)]) {
    return (loom_value_fact_layout_origin_t){0};
  }
  return *origins
              ->entries[loom_value_fact_range_offset(origins->range, value_id)];
}

void loom_value_fact_table_clear_layout_origin(loom_value_fact_table_t* table,
                                               loom_value_id_t value_id) {
  loom_value_fact_layout_origins_t* origins = table->layout_origins;
  bool changed = false;
  if (origins && loom_value_fact_range_contains(origins->range, value_id)) {
    changed = origins->entries[loom_value_fact_range_offset(origins->range,
                                                            value_id)] != NULL;
    origins->entries[loom_value_fact_range_offset(origins->range, value_id)] =
        NULL;
  }
  loom_value_fact_table_record_origin(table, value_id,
                                      LOOM_VALUE_FACT_ORIGIN_LAYOUT, changed);
}

static iree_status_t loom_value_fact_table_store_layout_origin(
    loom_value_fact_table_t* table, loom_value_id_t value_id,
    loom_value_fact_layout_origin_t origin) {
  if (origin.kind == LOOM_VALUE_FACT_LAYOUT_ORIGIN_NONE) {
    origin.count = 0;
    origin.values.data = NULL;
  }
  if (origin.count == 0 && !origin.has_physical_encoding_value) {
    loom_value_fact_table_clear_layout_origin(table, value_id);
    return iree_ok_status();
  }
  const loom_value_fact_layout_origin_t previous =
      loom_value_fact_table_query_layout_origin(table, value_id);
  if (loom_value_fact_layout_origin_equal(previous, origin)) {
    loom_value_fact_table_record_origin(table, value_id,
                                        LOOM_VALUE_FACT_ORIGIN_LAYOUT, false);
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(
      loom_value_fact_layout_origins_ensure_capacity(table, value_id));
  const iree_host_size_t byte_length =
      origin.count * loom_value_fact_layout_origin_element_size(origin.kind);
  loom_value_fact_layout_origin_t* storage = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(table->transient_arena,
                                           sizeof(*storage) + byte_length,
                                           (void**)&storage));
  void* payload = storage + 1;
  if (byte_length != 0) {
    memcpy(payload, origin.values.data, byte_length);
  }
  *storage = origin;
  storage->values.data = byte_length != 0 ? payload : NULL;
  table->layout_origins->entries[loom_value_fact_range_offset(
      table->layout_origins->range, value_id)] = storage;
  loom_value_fact_table_record_origin(table, value_id,
                                      LOOM_VALUE_FACT_ORIGIN_LAYOUT, true);
  return iree_ok_status();
}

iree_status_t loom_value_fact_table_define_layout_origin(
    loom_value_fact_table_t* table, loom_value_id_t value_id,
    loom_value_fact_layout_origin_t origin) {
  return loom_value_fact_table_store_layout_origin(table, value_id, origin);
}

iree_status_t loom_value_fact_table_define_layout_explicit_strides(
    loom_value_fact_table_t* table, loom_value_id_t value_id,
    const loom_value_id_t* strides, uint16_t count) {
  loom_value_fact_layout_origin_t origin =
      loom_value_fact_table_query_layout_origin(table, value_id);
  origin.kind = count ? LOOM_VALUE_FACT_LAYOUT_ORIGIN_EXPLICIT_STRIDES
                      : LOOM_VALUE_FACT_LAYOUT_ORIGIN_NONE;
  origin.count = count;
  origin.values.explicit_strides = strides;
  return loom_value_fact_table_store_layout_origin(table, value_id, origin);
}

iree_status_t loom_value_fact_table_define_layout_dense_dimensions(
    loom_value_fact_table_t* table, loom_value_id_t value_id,
    const uint64_t* dimensions, uint16_t count) {
  loom_value_fact_layout_origin_t origin =
      loom_value_fact_table_query_layout_origin(table, value_id);
  origin.kind = count ? LOOM_VALUE_FACT_LAYOUT_ORIGIN_DENSE_DIMENSIONS
                      : LOOM_VALUE_FACT_LAYOUT_ORIGIN_NONE;
  origin.count = count;
  origin.values.dense_dimensions = dimensions;
  return loom_value_fact_table_store_layout_origin(table, value_id, origin);
}

iree_status_t loom_value_fact_table_define_layout_physical_encoding_value(
    loom_value_fact_table_t* table, loom_value_id_t value_id,
    loom_value_id_t physical_encoding_value_id) {
  loom_value_fact_layout_origin_t origin =
      loom_value_fact_table_query_layout_origin(table, value_id);
  origin.has_physical_encoding_value =
      physical_encoding_value_id != LOOM_VALUE_ID_INVALID;
  origin.physical_encoding_value_id =
      origin.has_physical_encoding_value ? physical_encoding_value_id : 0;
  return loom_value_fact_table_store_layout_origin(table, value_id, origin);
}

iree_status_t loom_value_fact_table_forward_layout_origin(
    loom_value_fact_table_t* table, loom_value_id_t source_value_id,
    loom_value_id_t result_value_id) {
  loom_value_fact_layout_origins_t* origins = table->layout_origins;
  if (!origins ||
      !loom_value_fact_range_contains(origins->range, source_value_id) ||
      !origins->entries[loom_value_fact_range_offset(origins->range,
                                                     source_value_id)]) {
    loom_value_fact_table_clear_layout_origin(table, result_value_id);
    return iree_ok_status();
  }
  const loom_value_fact_layout_origin_t* origin =
      origins->entries[loom_value_fact_range_offset(origins->range,
                                                    source_value_id)];
  const loom_value_fact_layout_origin_t previous =
      loom_value_fact_table_query_layout_origin(table, result_value_id);
  const bool changed = !loom_value_fact_layout_origin_equal(previous, *origin);
  IREE_RETURN_IF_ERROR(
      loom_value_fact_layout_origins_ensure_capacity(table, result_value_id));
  origins
      ->entries[loom_value_fact_range_offset(origins->range, result_value_id)] =
      origin;
  loom_value_fact_table_record_origin(table, result_value_id,
                                      LOOM_VALUE_FACT_ORIGIN_LAYOUT, changed);
  return iree_ok_status();
}

static bool loom_value_fact_layout_accumulate_dimension(
    const loom_value_fact_table_t* table, uint64_t packed_dimension,
    int64_t* coefficient, loom_value_fact_layout_axis_t* out_axis) {
  if (!loom_dim_is_dynamic(packed_dimension)) {
    return iree_checked_mul_i64(
        *coefficient, loom_dim_static_size(packed_dimension), coefficient);
  }

  const loom_value_id_t value_id = loom_dim_value_id(packed_dimension);
  const loom_value_facts_t dimension_facts =
      loom_value_facts_non_negative_extent(
          loom_value_fact_table_lookup(table, value_id));
  int64_t exact_dimension = 0;
  if (loom_value_facts_as_exact_i64(dimension_facts, &exact_dimension)) {
    return iree_checked_mul_i64(*coefficient, exact_dimension, coefficient);
  }
  out_axis->dynamic_factors[out_axis->dynamic_factor_count++] = value_id;
  return true;
}

void loom_value_fact_layout_query_axis(
    const loom_value_fact_table_t* table,
    loom_value_fact_layout_origin_t origin, uint8_t axis,
    loom_value_facts_t numeric_stride_facts,
    loom_value_fact_layout_axis_t* out_axis) {
  *out_axis = (loom_value_fact_layout_axis_t){
      .kind = LOOM_VALUE_FACT_LAYOUT_AXIS_UNAVAILABLE,
  };
  int64_t exact_stride = 0;
  if (loom_value_facts_as_exact_i64(numeric_stride_facts, &exact_stride)) {
    out_axis->kind = exact_stride < 0 ? LOOM_VALUE_FACT_LAYOUT_AXIS_INVALID
                                      : LOOM_VALUE_FACT_LAYOUT_AXIS_STATIC;
    out_axis->static_coefficient = exact_stride;
    return;
  }
  if (axis >= origin.count) {
    return;
  }

  int64_t coefficient = 1;
  switch (origin.kind) {
    case LOOM_VALUE_FACT_LAYOUT_ORIGIN_EXPLICIT_STRIDES: {
      const loom_value_id_t value_id = origin.values.explicit_strides[axis];
      if (value_id == LOOM_VALUE_ID_INVALID) {
        return;
      }
      const loom_value_facts_t value_facts =
          loom_value_fact_table_lookup(table, value_id);
      int64_t exact_value = 0;
      if (loom_value_facts_as_exact_i64(value_facts, &exact_value)) {
        if (exact_value < 0) {
          out_axis->kind = LOOM_VALUE_FACT_LAYOUT_AXIS_INVALID;
          return;
        }
        coefficient = exact_value;
      } else {
        out_axis->dynamic_factors[out_axis->dynamic_factor_count++] = value_id;
      }
      break;
    }
    case LOOM_VALUE_FACT_LAYOUT_ORIGIN_DENSE_DIMENSIONS:
      for (uint8_t i = (uint8_t)(axis + 1); i < origin.count; ++i) {
        if (!loom_value_fact_layout_accumulate_dimension(
                table, origin.values.dense_dimensions[i], &coefficient,
                out_axis)) {
          out_axis->kind = LOOM_VALUE_FACT_LAYOUT_AXIS_INVALID;
          return;
        }
      }
      break;
    case LOOM_VALUE_FACT_LAYOUT_ORIGIN_NONE:
      return;
  }

  if (coefficient == 0) {
    out_axis->dynamic_factor_count = 0;
  }
  out_axis->static_coefficient = coefficient;
  out_axis->kind = out_axis->dynamic_factor_count == 0
                       ? LOOM_VALUE_FACT_LAYOUT_AXIS_STATIC
                       : LOOM_VALUE_FACT_LAYOUT_AXIS_DYNAMIC;
}
