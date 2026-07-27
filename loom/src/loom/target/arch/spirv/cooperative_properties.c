// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/spirv/cooperative_properties.h"

#include <string.h>

#include "loom/target/arch/spirv/descriptors/descriptors.h"

#define MATRIX_LAYOUT_ANY                               \
  (LOOM_SPIRV_COOPERATIVE_MATRIX_LAYOUT_ROW_MAJOR_BIT | \
   LOOM_SPIRV_COOPERATIVE_MATRIX_LAYOUT_COLUMN_MAJOR_BIT)

#define VECTOR_LAYOUT_INFERENCE_ANY                               \
  (LOOM_SPIRV_COOPERATIVE_VECTOR_MATRIX_LAYOUT_ROW_MAJOR_BIT |    \
   LOOM_SPIRV_COOPERATIVE_VECTOR_MATRIX_LAYOUT_COLUMN_MAJOR_BIT | \
   LOOM_SPIRV_COOPERATIVE_VECTOR_MATRIX_LAYOUT_INFERENCING_OPTIMAL_BIT)

#define MATRIX_STORAGE_ANY                       \
  (LOOM_SPIRV_STORAGE_CLASS_BIT_WORKGROUP |      \
   LOOM_SPIRV_STORAGE_CLASS_BIT_STORAGE_BUFFER | \
   LOOM_SPIRV_STORAGE_CLASS_BIT_PHYSICAL_STORAGE_BUFFER)

#define STORAGE_BUFFER_OR_BDA                    \
  (LOOM_SPIRV_STORAGE_CLASS_BIT_STORAGE_BUFFER | \
   LOOM_SPIRV_STORAGE_CLASS_BIT_PHYSICAL_STORAGE_BUFFER)

#include "loom/target/arch/spirv/cooperative_properties_tables.inl"

iree_string_view_t loom_spirv_component_type_name(
    loom_spirv_component_type_t component_type) {
  switch (component_type) {
    case LOOM_SPIRV_COMPONENT_TYPE_FLOAT16_NV:
      return IREE_SV("f16");
    case LOOM_SPIRV_COMPONENT_TYPE_FLOAT32_NV:
      return IREE_SV("f32");
    case LOOM_SPIRV_COMPONENT_TYPE_FLOAT64_NV:
      return IREE_SV("f64");
    case LOOM_SPIRV_COMPONENT_TYPE_SIGNED_INT8_NV:
      return IREE_SV("s8");
    case LOOM_SPIRV_COMPONENT_TYPE_SIGNED_INT16_NV:
      return IREE_SV("s16");
    case LOOM_SPIRV_COMPONENT_TYPE_SIGNED_INT32_NV:
      return IREE_SV("s32");
    case LOOM_SPIRV_COMPONENT_TYPE_SIGNED_INT64_NV:
      return IREE_SV("s64");
    case LOOM_SPIRV_COMPONENT_TYPE_UNSIGNED_INT8_NV:
      return IREE_SV("u8");
    case LOOM_SPIRV_COMPONENT_TYPE_UNSIGNED_INT16_NV:
      return IREE_SV("u16");
    case LOOM_SPIRV_COMPONENT_TYPE_UNSIGNED_INT32_NV:
      return IREE_SV("u32");
    case LOOM_SPIRV_COMPONENT_TYPE_UNSIGNED_INT64_NV:
      return IREE_SV("u64");
    case LOOM_SPIRV_COMPONENT_TYPE_SIGNED_INT8_PACKED_NV:
      return IREE_SV("s8_packed");
    case LOOM_SPIRV_COMPONENT_TYPE_UNSIGNED_INT8_PACKED_NV:
      return IREE_SV("u8_packed");
    case LOOM_SPIRV_COMPONENT_TYPE_FLOAT_E4_M3_NV:
      return IREE_SV("f8e4m3");
    case LOOM_SPIRV_COMPONENT_TYPE_FLOAT_E5_M2_NV:
      return IREE_SV("f8e5m2");
    case LOOM_SPIRV_COMPONENT_TYPE_MAX:
    default:
      return IREE_SV("unknown");
  }
}

bool loom_spirv_component_type_from_numeric_format(
    loom_value_fact_numeric_format_flags_t numeric_format,
    loom_spirv_component_type_t* out_component_type) {
  IREE_ASSERT_ARGUMENT(out_component_type);
  *out_component_type = LOOM_SPIRV_COMPONENT_TYPE_MAX;
  switch (numeric_format) {
    case LOOM_VALUE_FACT_NUMERIC_FORMAT_F64:
      *out_component_type = LOOM_SPIRV_COMPONENT_TYPE_FLOAT64_NV;
      return true;
    case LOOM_VALUE_FACT_NUMERIC_FORMAT_F32:
      *out_component_type = LOOM_SPIRV_COMPONENT_TYPE_FLOAT32_NV;
      return true;
    case LOOM_VALUE_FACT_NUMERIC_FORMAT_F16:
      *out_component_type = LOOM_SPIRV_COMPONENT_TYPE_FLOAT16_NV;
      return true;
    case LOOM_VALUE_FACT_NUMERIC_FORMAT_I32:
      *out_component_type = LOOM_SPIRV_COMPONENT_TYPE_SIGNED_INT32_NV;
      return true;
    case LOOM_VALUE_FACT_NUMERIC_FORMAT_I16:
      *out_component_type = LOOM_SPIRV_COMPONENT_TYPE_SIGNED_INT16_NV;
      return true;
    case LOOM_VALUE_FACT_NUMERIC_FORMAT_I8:
    case LOOM_VALUE_FACT_NUMERIC_FORMAT_QUANT_I8:
      *out_component_type = LOOM_SPIRV_COMPONENT_TYPE_SIGNED_INT8_NV;
      return true;
    case LOOM_VALUE_FACT_NUMERIC_FORMAT_U32:
      *out_component_type = LOOM_SPIRV_COMPONENT_TYPE_UNSIGNED_INT32_NV;
      return true;
    case LOOM_VALUE_FACT_NUMERIC_FORMAT_U16:
      *out_component_type = LOOM_SPIRV_COMPONENT_TYPE_UNSIGNED_INT16_NV;
      return true;
    case LOOM_VALUE_FACT_NUMERIC_FORMAT_U8:
      *out_component_type = LOOM_SPIRV_COMPONENT_TYPE_UNSIGNED_INT8_NV;
      return true;
    case LOOM_VALUE_FACT_NUMERIC_FORMAT_F8_E4M3:
    case LOOM_VALUE_FACT_NUMERIC_FORMAT_F8_E4M3FN:
    case LOOM_VALUE_FACT_NUMERIC_FORMAT_F8_E4M3FNUZ:
      *out_component_type = LOOM_SPIRV_COMPONENT_TYPE_FLOAT_E4_M3_NV;
      return true;
    case LOOM_VALUE_FACT_NUMERIC_FORMAT_F8_E5M2:
    case LOOM_VALUE_FACT_NUMERIC_FORMAT_F8_E5M2FNUZ:
    case LOOM_VALUE_FACT_NUMERIC_FORMAT_BF8:
      *out_component_type = LOOM_SPIRV_COMPONENT_TYPE_FLOAT_E5_M2_NV;
      return true;
    default:
      return false;
  }
}

iree_string_view_t loom_spirv_scope_name(loom_spirv_scope_t scope) {
  switch (scope) {
    case LOOM_SPIRV_SCOPE_DEVICE:
      return IREE_SV("device");
    case LOOM_SPIRV_SCOPE_WORKGROUP:
      return IREE_SV("workgroup");
    case LOOM_SPIRV_SCOPE_SUBGROUP:
      return IREE_SV("subgroup");
    case LOOM_SPIRV_SCOPE_QUEUE_FAMILY:
      return IREE_SV("queue_family");
    case LOOM_SPIRV_SCOPE_MAX:
    default:
      return IREE_SV("unknown");
  }
}

iree_string_view_t loom_spirv_cooperative_matrix_layout_name(
    loom_spirv_cooperative_matrix_layout_t layout) {
  switch (layout) {
    case LOOM_SPIRV_COOPERATIVE_MATRIX_LAYOUT_ROW_MAJOR_KHR:
      return IREE_SV("row_major");
    case LOOM_SPIRV_COOPERATIVE_MATRIX_LAYOUT_COLUMN_MAJOR_KHR:
      return IREE_SV("column_major");
    case LOOM_SPIRV_COOPERATIVE_MATRIX_LAYOUT_MAX:
    default:
      return IREE_SV("unknown");
  }
}

iree_string_view_t loom_spirv_cooperative_vector_matrix_layout_name(
    loom_spirv_cooperative_vector_matrix_layout_t layout) {
  switch (layout) {
    case LOOM_SPIRV_COOPERATIVE_VECTOR_MATRIX_LAYOUT_ROW_MAJOR_NV:
      return IREE_SV("row_major");
    case LOOM_SPIRV_COOPERATIVE_VECTOR_MATRIX_LAYOUT_COLUMN_MAJOR_NV:
      return IREE_SV("column_major");
    case LOOM_SPIRV_COOPERATIVE_VECTOR_MATRIX_LAYOUT_INFERENCING_OPTIMAL_NV:
      return IREE_SV("inferencing_optimal");
    case LOOM_SPIRV_COOPERATIVE_VECTOR_MATRIX_LAYOUT_TRAINING_OPTIMAL_NV:
      return IREE_SV("training_optimal");
    case LOOM_SPIRV_COOPERATIVE_VECTOR_MATRIX_LAYOUT_MAX:
    default:
      return IREE_SV("unknown");
  }
}

iree_string_view_t loom_spirv_storage_class_name(
    loom_spirv_storage_class_t storage_class) {
  switch (storage_class) {
    case LOOM_SPIRV_STORAGE_CLASS_WORKGROUP:
      return IREE_SV("Workgroup");
    case LOOM_SPIRV_STORAGE_CLASS_STORAGE_BUFFER:
      return IREE_SV("StorageBuffer");
    case LOOM_SPIRV_STORAGE_CLASS_PHYSICAL_STORAGE_BUFFER:
      return IREE_SV("PhysicalStorageBuffer");
    case LOOM_SPIRV_STORAGE_CLASS_UNIFORM_CONSTANT:
      return IREE_SV("UniformConstant");
    case LOOM_SPIRV_STORAGE_CLASS_UNIFORM:
      return IREE_SV("Uniform");
    case LOOM_SPIRV_STORAGE_CLASS_CROSS_WORKGROUP:
      return IREE_SV("CrossWorkgroup");
    case LOOM_SPIRV_STORAGE_CLASS_FUNCTION:
      return IREE_SV("Function");
    case LOOM_SPIRV_STORAGE_CLASS_GENERIC:
      return IREE_SV("Generic");
    default:
      return IREE_SV("unknown");
  }
}

loom_spirv_storage_class_flags_t loom_spirv_storage_class_bit(
    loom_spirv_storage_class_t storage_class) {
  switch (storage_class) {
    case LOOM_SPIRV_STORAGE_CLASS_WORKGROUP:
      return LOOM_SPIRV_STORAGE_CLASS_BIT_WORKGROUP;
    case LOOM_SPIRV_STORAGE_CLASS_STORAGE_BUFFER:
      return LOOM_SPIRV_STORAGE_CLASS_BIT_STORAGE_BUFFER;
    case LOOM_SPIRV_STORAGE_CLASS_PHYSICAL_STORAGE_BUFFER:
      return LOOM_SPIRV_STORAGE_CLASS_BIT_PHYSICAL_STORAGE_BUFFER;
    default:
      return 0;
  }
}

uint64_t loom_spirv_cooperative_matrix_shape_key(uint16_t m_size,
                                                 uint16_t n_size,
                                                 uint16_t k_size) {
  return ((uint64_t)m_size << 32) | ((uint64_t)n_size << 16) | k_size;
}

static uint64_t loom_spirv_vector_shape_key(uint16_t m_size, uint16_t k_size) {
  return ((uint64_t)m_size << 16) | k_size;
}

static const loom_spirv_cooperative_property_span_t*
loom_spirv_property_span_find(
    const loom_spirv_cooperative_property_span_t* spans, uint16_t span_count,
    uint64_t shape_key) {
  uint16_t lo = 0;
  uint16_t hi = span_count;
  while (lo < hi) {
    uint16_t mid = (uint16_t)(lo + (hi - lo) / 2u);
    if (spans[mid].shape_key < shape_key) {
      lo = (uint16_t)(mid + 1u);
    } else {
      hi = mid;
    }
  }
  if (lo < span_count && spans[lo].shape_key == shape_key) {
    return &spans[lo];
  }
  return NULL;
}

static bool loom_spirv_policy_requires_target_primitive(
    loom_lowering_policy_t policy) {
  return policy == LOOM_LOWERING_POLICY_OPTIMIZED_REQUIRED ||
         policy == LOOM_LOWERING_POLICY_TARGET_PRIMITIVE_REQUIRED;
}

static loom_spirv_feature_atom_t loom_spirv_first_missing_feature_atom(
    loom_spirv_feature_bits_t missing_feature_bits) {
  for (uint32_t i = LOOM_SPIRV_FEATURE_ATOM_UNKNOWN + 1;
       i < LOOM_SPIRV_FEATURE_ATOM_COUNT; ++i) {
    const loom_spirv_feature_atom_t atom = (loom_spirv_feature_atom_t)i;
    if (iree_any_bit_set(missing_feature_bits,
                         loom_spirv_feature_atom_bit(atom))) {
      return atom;
    }
  }
  return LOOM_SPIRV_FEATURE_ATOM_UNKNOWN;
}

static void loom_spirv_cooperative_diagnostic_fail(
    loom_spirv_cooperative_selection_status_t status,
    loom_spirv_feature_atom_t feature_atom,
    loom_spirv_cooperative_rejection_flags_t rejection_flags,
    loom_spirv_cooperative_diagnostic_t* out_diagnostic) {
  if (out_diagnostic) {
    *out_diagnostic = (loom_spirv_cooperative_diagnostic_t){
        .status = status,
        .feature_atom = feature_atom,
        .rejection_flags = rejection_flags,
    };
  }
}

static void loom_spirv_cooperative_diagnostic_match(
    loom_spirv_feature_atom_t feature_atom,
    loom_spirv_cooperative_diagnostic_t* out_diagnostic) {
  if (out_diagnostic) {
    *out_diagnostic = (loom_spirv_cooperative_diagnostic_t){
        .status = LOOM_SPIRV_COOPERATIVE_SELECTION_MATCHED,
        .feature_atom = feature_atom,
    };
  }
}

void loom_spirv_cooperative_property_set_prepare(
    const loom_spirv_feature_set_t* feature_set,
    loom_spirv_cooperative_property_set_t* out_property_set) {
  IREE_ASSERT_ARGUMENT(feature_set);
  IREE_ASSERT_ARGUMENT(out_property_set);
  *out_property_set = (loom_spirv_cooperative_property_set_t){
      .feature_bits = feature_set->atom_bits,
      .matrix_properties = kCooperativeMatrixProperties,
      .matrix_property_count = IREE_ARRAYSIZE(kCooperativeMatrixProperties),
      .matrix_shape_spans = kCooperativeMatrixShapeSpans,
      .matrix_shape_span_count = IREE_ARRAYSIZE(kCooperativeMatrixShapeSpans),
      .vector_properties = kCooperativeVectorProperties,
      .vector_property_count = IREE_ARRAYSIZE(kCooperativeVectorProperties),
      .vector_shape_spans = kCooperativeVectorShapeSpans,
      .vector_shape_span_count = IREE_ARRAYSIZE(kCooperativeVectorShapeSpans),
  };
}

const loom_spirv_cooperative_matrix_property_t*
loom_spirv_cooperative_matrix_model_properties(iree_host_size_t* out_count) {
  IREE_ASSERT_ARGUMENT(out_count);
  *out_count = IREE_ARRAYSIZE(kCooperativeMatrixProperties);
  return kCooperativeMatrixProperties;
}

const loom_spirv_cooperative_vector_property_t*
loom_spirv_cooperative_vector_model_properties(iree_host_size_t* out_count) {
  IREE_ASSERT_ARGUMENT(out_count);
  *out_count = IREE_ARRAYSIZE(kCooperativeVectorProperties);
  return kCooperativeVectorProperties;
}

static uint64_t loom_spirv_cooperative_matrix_property_shape_key(
    const loom_spirv_cooperative_matrix_property_t* row) {
  return loom_spirv_cooperative_matrix_shape_key(row->m_size, row->n_size,
                                                 row->k_size);
}

static uint64_t loom_spirv_cooperative_vector_property_shape_key(
    const loom_spirv_cooperative_vector_property_t* row) {
  return loom_spirv_vector_shape_key(row->m_size, row->k_size);
}

static void loom_spirv_cooperative_property_storage_sort_matrix_rows(
    loom_spirv_cooperative_matrix_property_t* rows,
    iree_host_size_t row_count) {
  for (iree_host_size_t i = 1; i < row_count; ++i) {
    const loom_spirv_cooperative_matrix_property_t row = rows[i];
    const uint64_t shape_key =
        loom_spirv_cooperative_matrix_property_shape_key(&row);
    iree_host_size_t j = i;
    while (j > 0 && loom_spirv_cooperative_matrix_property_shape_key(
                        &rows[j - 1]) > shape_key) {
      rows[j] = rows[j - 1];
      --j;
    }
    rows[j] = row;
  }
}

static void loom_spirv_cooperative_property_storage_sort_vector_rows(
    loom_spirv_cooperative_vector_property_t* rows,
    iree_host_size_t row_count) {
  for (iree_host_size_t i = 1; i < row_count; ++i) {
    const loom_spirv_cooperative_vector_property_t row = rows[i];
    const uint64_t shape_key =
        loom_spirv_cooperative_vector_property_shape_key(&row);
    iree_host_size_t j = i;
    while (j > 0 && loom_spirv_cooperative_vector_property_shape_key(
                        &rows[j - 1]) > shape_key) {
      rows[j] = rows[j - 1];
      --j;
    }
    rows[j] = row;
  }
}

static iree_status_t loom_spirv_cooperative_property_storage_copy_matrix_rows(
    const loom_spirv_cooperative_matrix_property_t* matrix_properties,
    iree_host_size_t matrix_property_count, iree_allocator_t allocator,
    loom_spirv_cooperative_property_storage_t* storage) {
  if (matrix_property_count == 0) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      allocator, matrix_property_count * sizeof(*storage->matrix_properties),
      (void**)&storage->matrix_properties));
  memcpy(storage->matrix_properties, matrix_properties,
         matrix_property_count * sizeof(*storage->matrix_properties));
  loom_spirv_cooperative_property_storage_sort_matrix_rows(
      storage->matrix_properties, matrix_property_count);
  storage->set.matrix_properties = storage->matrix_properties;
  storage->set.matrix_property_count = (uint16_t)matrix_property_count;
  return iree_ok_status();
}

static iree_status_t loom_spirv_cooperative_property_storage_copy_vector_rows(
    const loom_spirv_cooperative_vector_property_t* vector_properties,
    iree_host_size_t vector_property_count, iree_allocator_t allocator,
    loom_spirv_cooperative_property_storage_t* storage) {
  if (vector_property_count == 0) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      allocator, vector_property_count * sizeof(*storage->vector_properties),
      (void**)&storage->vector_properties));
  memcpy(storage->vector_properties, vector_properties,
         vector_property_count * sizeof(*storage->vector_properties));
  loom_spirv_cooperative_property_storage_sort_vector_rows(
      storage->vector_properties, vector_property_count);
  storage->set.vector_properties = storage->vector_properties;
  storage->set.vector_property_count = (uint16_t)vector_property_count;
  return iree_ok_status();
}

static iree_status_t loom_spirv_cooperative_property_storage_build_matrix_spans(
    iree_host_size_t matrix_property_count, iree_allocator_t allocator,
    loom_spirv_cooperative_property_storage_t* storage) {
  if (matrix_property_count == 0) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      allocator, matrix_property_count * sizeof(*storage->matrix_shape_spans),
      (void**)&storage->matrix_shape_spans));
  uint64_t previous_shape_key = 0;
  for (iree_host_size_t i = 0; i < matrix_property_count; ++i) {
    const loom_spirv_cooperative_matrix_property_t* row =
        &storage->matrix_properties[i];
    const uint64_t shape_key = loom_spirv_cooperative_matrix_shape_key(
        row->m_size, row->n_size, row->k_size);
    if (i != 0 && shape_key < previous_shape_key) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "SPIR-V cooperative matrix rows must be sorted by shape key");
    }
    if (i == 0 || shape_key != previous_shape_key) {
      storage->matrix_shape_spans[storage->set.matrix_shape_span_count++] =
          (loom_spirv_cooperative_property_span_t){
              .shape_key = shape_key,
              .start = (uint16_t)i,
              .count = 1,
          };
    } else {
      const uint16_t span_index = storage->set.matrix_shape_span_count - 1;
      loom_spirv_cooperative_property_span_t* span =
          &storage->matrix_shape_spans[span_index];
      ++span->count;
    }
    previous_shape_key = shape_key;
  }
  storage->set.matrix_shape_spans = storage->matrix_shape_spans;
  return iree_ok_status();
}

static iree_status_t loom_spirv_cooperative_property_storage_build_vector_spans(
    iree_host_size_t vector_property_count, iree_allocator_t allocator,
    loom_spirv_cooperative_property_storage_t* storage) {
  if (vector_property_count == 0) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      allocator, vector_property_count * sizeof(*storage->vector_shape_spans),
      (void**)&storage->vector_shape_spans));
  uint64_t previous_shape_key = 0;
  for (iree_host_size_t i = 0; i < vector_property_count; ++i) {
    const loom_spirv_cooperative_vector_property_t* row =
        &storage->vector_properties[i];
    const uint64_t shape_key =
        loom_spirv_cooperative_vector_property_shape_key(row);
    if (i == 0 || shape_key != previous_shape_key) {
      storage->vector_shape_spans[storage->set.vector_shape_span_count++] =
          (loom_spirv_cooperative_property_span_t){
              .shape_key = shape_key,
              .start = (uint16_t)i,
              .count = 1,
          };
    } else {
      const uint16_t span_index = storage->set.vector_shape_span_count - 1;
      loom_spirv_cooperative_property_span_t* span =
          &storage->vector_shape_spans[span_index];
      ++span->count;
    }
    previous_shape_key = shape_key;
  }
  storage->set.vector_shape_spans = storage->vector_shape_spans;
  return iree_ok_status();
}

iree_status_t loom_spirv_cooperative_property_storage_initialize(
    loom_spirv_feature_bits_t feature_bits,
    const loom_spirv_cooperative_matrix_property_t* matrix_properties,
    iree_host_size_t matrix_property_count,
    const loom_spirv_cooperative_vector_property_t* vector_properties,
    iree_host_size_t vector_property_count, iree_allocator_t allocator,
    loom_spirv_cooperative_property_storage_t* out_storage) {
  IREE_ASSERT_ARGUMENT(out_storage);
  if (matrix_property_count != 0) {
    IREE_ASSERT_ARGUMENT(matrix_properties);
  }
  if (vector_property_count != 0) {
    IREE_ASSERT_ARGUMENT(vector_properties);
  }
  memset(out_storage, 0, sizeof(*out_storage));
  if (matrix_property_count > UINT16_MAX) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "SPIR-V cooperative matrix property count exceeds uint16_t capacity");
  }
  if (vector_property_count > UINT16_MAX) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "SPIR-V cooperative vector property count exceeds uint16_t capacity");
  }
  out_storage->set.feature_bits = feature_bits;
  iree_status_t status =
      loom_spirv_cooperative_property_storage_copy_matrix_rows(
          matrix_properties, matrix_property_count, allocator, out_storage);
  if (iree_status_is_ok(status)) {
    status = loom_spirv_cooperative_property_storage_copy_vector_rows(
        vector_properties, vector_property_count, allocator, out_storage);
  }
  if (iree_status_is_ok(status)) {
    status = loom_spirv_cooperative_property_storage_build_matrix_spans(
        matrix_property_count, allocator, out_storage);
  }
  if (iree_status_is_ok(status)) {
    status = loom_spirv_cooperative_property_storage_build_vector_spans(
        vector_property_count, allocator, out_storage);
  }
  if (!iree_status_is_ok(status)) {
    loom_spirv_cooperative_property_storage_deinitialize(out_storage,
                                                         allocator);
  }
  return status;
}

void loom_spirv_cooperative_property_storage_deinitialize(
    loom_spirv_cooperative_property_storage_t* storage,
    iree_allocator_t allocator) {
  if (storage == NULL) {
    return;
  }
  iree_allocator_free(allocator, storage->vector_shape_spans);
  iree_allocator_free(allocator, storage->vector_properties);
  iree_allocator_free(allocator, storage->matrix_shape_spans);
  iree_allocator_free(allocator, storage->matrix_properties);
  *storage = (loom_spirv_cooperative_property_storage_t){0};
}

static loom_spirv_cooperative_matrix_layout_flags_t
loom_spirv_matrix_layout_bit(loom_spirv_cooperative_matrix_layout_t layout) {
  switch (layout) {
    case LOOM_SPIRV_COOPERATIVE_MATRIX_LAYOUT_ROW_MAJOR_KHR:
      return LOOM_SPIRV_COOPERATIVE_MATRIX_LAYOUT_ROW_MAJOR_BIT;
    case LOOM_SPIRV_COOPERATIVE_MATRIX_LAYOUT_COLUMN_MAJOR_KHR:
      return LOOM_SPIRV_COOPERATIVE_MATRIX_LAYOUT_COLUMN_MAJOR_BIT;
    default:
      return 0;
  }
}

const loom_spirv_cooperative_matrix_property_t*
loom_spirv_cooperative_matrix_property_select(
    const loom_spirv_cooperative_property_set_t* property_set,
    const loom_spirv_cooperative_matrix_query_t* query,
    loom_spirv_cooperative_diagnostic_t* out_diagnostic) {
  IREE_ASSERT_ARGUMENT(property_set);
  IREE_ASSERT_ARGUMENT(query);
  const loom_spirv_feature_bits_t required_feature_bits =
      LOOM_SPIRV_FEATURE_COOPERATIVE_MATRIX_KHR;
  if (!iree_all_bits_set(property_set->feature_bits, required_feature_bits)) {
    const loom_spirv_cooperative_selection_status_t status =
        !loom_spirv_policy_requires_target_primitive(query->policy)
            ? LOOM_SPIRV_COOPERATIVE_SELECTION_FALLBACK_PERMITTED
            : LOOM_SPIRV_COOPERATIVE_SELECTION_REQUIRED_FEATURE_MISSING;
    const loom_spirv_cooperative_rejection_flags_t rejection_flags =
        LOOM_SPIRV_COOPERATIVE_REJECTION_FEATURE |
        (status == LOOM_SPIRV_COOPERATIVE_SELECTION_FALLBACK_PERMITTED
             ? LOOM_SPIRV_COOPERATIVE_REJECTION_POLICY_FALLBACK
             : 0);
    loom_spirv_cooperative_diagnostic_fail(
        status, LOOM_SPIRV_FEATURE_ATOM_COOPERATIVE_MATRIX_KHR, rejection_flags,
        out_diagnostic);
    return NULL;
  }

  const uint64_t shape_key = loom_spirv_cooperative_matrix_shape_key(
      query->m_size, query->n_size, query->k_size);
  const loom_spirv_cooperative_property_span_t* span =
      loom_spirv_property_span_find(property_set->matrix_shape_spans,
                                    property_set->matrix_shape_span_count,
                                    shape_key);
  loom_spirv_cooperative_diagnostic_t diagnostic = {
      .status = LOOM_SPIRV_COOPERATIVE_SELECTION_REQUIRED_PROPERTY_MISSING,
      .feature_atom = LOOM_SPIRV_FEATURE_ATOM_COOPERATIVE_MATRIX_KHR,
      .rejection_flags = LOOM_SPIRV_COOPERATIVE_REJECTION_SHAPE,
  };
  if (span == NULL) {
    if (!loom_spirv_policy_requires_target_primitive(query->policy)) {
      diagnostic.status = LOOM_SPIRV_COOPERATIVE_SELECTION_FALLBACK_PERMITTED;
      diagnostic.rejection_flags |=
          LOOM_SPIRV_COOPERATIVE_REJECTION_POLICY_FALLBACK;
    }
    if (out_diagnostic) {
      *out_diagnostic = diagnostic;
    }
    return NULL;
  }

  diagnostic.shape_candidate_count = span->count;
  const loom_spirv_cooperative_matrix_layout_flags_t layout_bit =
      loom_spirv_matrix_layout_bit(query->layout);
  const loom_spirv_storage_class_flags_t storage_class_bit =
      loom_spirv_storage_class_bit(query->storage_class);
  diagnostic.rejection_flags = LOOM_SPIRV_COOPERATIVE_REJECTION_COMPONENT_TYPE;
  loom_spirv_feature_bits_t missing_row_feature_bits = 0;
  uint16_t enabled_type_candidate_count = 0;
  for (uint16_t i = 0; i < span->count; ++i) {
    const loom_spirv_cooperative_matrix_property_t* row =
        &property_set->matrix_properties[span->start + i];
    if (row->lhs_type != query->lhs_type || row->rhs_type != query->rhs_type ||
        row->accumulator_type != query->accumulator_type ||
        row->result_type != query->result_type) {
      continue;
    }
    ++diagnostic.type_candidate_count;
    const loom_spirv_feature_bits_t row_missing_bits =
        row->required_feature_bits & ~property_set->feature_bits;
    if (row_missing_bits != 0) {
      missing_row_feature_bits |= row_missing_bits;
      continue;
    }
    ++enabled_type_candidate_count;
    diagnostic.rejection_flags = LOOM_SPIRV_COOPERATIVE_REJECTION_SCOPE;
    if (row->scope != query->scope) {
      continue;
    }
    diagnostic.rejection_flags = LOOM_SPIRV_COOPERATIVE_REJECTION_LAYOUT;
    if (layout_bit == 0 || !iree_any_bit_set(row->layout_flags, layout_bit)) {
      continue;
    }
    ++diagnostic.layout_candidate_count;
    diagnostic.rejection_flags = LOOM_SPIRV_COOPERATIVE_REJECTION_STORAGE_CLASS;
    if (storage_class_bit == 0 ||
        !iree_any_bit_set(row->storage_class_flags, storage_class_bit)) {
      continue;
    }
    ++diagnostic.storage_candidate_count;
    diagnostic.rejection_flags = LOOM_SPIRV_COOPERATIVE_REJECTION_OPERANDS;
    if (row->operand_flags != query->operand_flags) {
      continue;
    }
    loom_spirv_cooperative_diagnostic_match(
        LOOM_SPIRV_FEATURE_ATOM_COOPERATIVE_MATRIX_KHR, out_diagnostic);
    return row;
  }

  if (diagnostic.type_candidate_count != 0 &&
      enabled_type_candidate_count == 0 && missing_row_feature_bits != 0) {
    diagnostic.status =
        !loom_spirv_policy_requires_target_primitive(query->policy)
            ? LOOM_SPIRV_COOPERATIVE_SELECTION_FALLBACK_PERMITTED
            : LOOM_SPIRV_COOPERATIVE_SELECTION_REQUIRED_FEATURE_MISSING;
    diagnostic.feature_atom =
        loom_spirv_first_missing_feature_atom(missing_row_feature_bits);
    diagnostic.rejection_flags = LOOM_SPIRV_COOPERATIVE_REJECTION_FEATURE;
    if (diagnostic.status ==
        LOOM_SPIRV_COOPERATIVE_SELECTION_FALLBACK_PERMITTED) {
      diagnostic.rejection_flags |=
          LOOM_SPIRV_COOPERATIVE_REJECTION_POLICY_FALLBACK;
    }
    if (out_diagnostic) {
      *out_diagnostic = diagnostic;
    }
    return NULL;
  }

  if (!loom_spirv_policy_requires_target_primitive(query->policy)) {
    diagnostic.status = LOOM_SPIRV_COOPERATIVE_SELECTION_FALLBACK_PERMITTED;
    diagnostic.rejection_flags |=
        LOOM_SPIRV_COOPERATIVE_REJECTION_POLICY_FALLBACK;
  }
  if (out_diagnostic) {
    *out_diagnostic = diagnostic;
  }
  return NULL;
}

static loom_spirv_cooperative_vector_matrix_layout_flags_t
loom_spirv_vector_matrix_layout_bit(
    loom_spirv_cooperative_vector_matrix_layout_t layout) {
  switch (layout) {
    case LOOM_SPIRV_COOPERATIVE_VECTOR_MATRIX_LAYOUT_ROW_MAJOR_NV:
      return LOOM_SPIRV_COOPERATIVE_VECTOR_MATRIX_LAYOUT_ROW_MAJOR_BIT;
    case LOOM_SPIRV_COOPERATIVE_VECTOR_MATRIX_LAYOUT_COLUMN_MAJOR_NV:
      return LOOM_SPIRV_COOPERATIVE_VECTOR_MATRIX_LAYOUT_COLUMN_MAJOR_BIT;
    case LOOM_SPIRV_COOPERATIVE_VECTOR_MATRIX_LAYOUT_INFERENCING_OPTIMAL_NV:
      return LOOM_SPIRV_COOPERATIVE_VECTOR_MATRIX_LAYOUT_INFERENCING_OPTIMAL_BIT;
    case LOOM_SPIRV_COOPERATIVE_VECTOR_MATRIX_LAYOUT_TRAINING_OPTIMAL_NV:
      return LOOM_SPIRV_COOPERATIVE_VECTOR_MATRIX_LAYOUT_TRAINING_OPTIMAL_BIT;
    default:
      return 0;
  }
}

static bool loom_spirv_cooperative_vector_flags_match(
    loom_spirv_cooperative_vector_flags_t row_flags,
    loom_spirv_cooperative_vector_flags_t query_flags) {
  const bool row_is_training =
      iree_any_bit_set(row_flags, LOOM_SPIRV_COOPERATIVE_VECTOR_FLAG_TRAINING);
  const bool query_is_training = iree_any_bit_set(
      query_flags, LOOM_SPIRV_COOPERATIVE_VECTOR_FLAG_TRAINING);
  if (row_is_training != query_is_training) {
    return false;
  }
  return iree_all_bits_set(row_flags, query_flags);
}

static loom_spirv_cooperative_rejection_flags_t
loom_spirv_cooperative_vector_component_rejection_flags(
    const loom_spirv_cooperative_vector_property_t* row,
    const loom_spirv_cooperative_vector_query_t* query) {
  loom_spirv_cooperative_rejection_flags_t rejection_flags =
      LOOM_SPIRV_COOPERATIVE_REJECTION_NONE;
  if (row->input_type != query->input_type) {
    rejection_flags |= LOOM_SPIRV_COOPERATIVE_REJECTION_INPUT_TYPE;
  }
  if (row->input_interpretation != query->input_interpretation) {
    rejection_flags |= LOOM_SPIRV_COOPERATIVE_REJECTION_INPUT_INTERPRETATION;
  }
  if (row->matrix_interpretation != query->matrix_interpretation) {
    rejection_flags |= LOOM_SPIRV_COOPERATIVE_REJECTION_MATRIX_INTERPRETATION;
  }
  if (row->bias_interpretation != query->bias_interpretation) {
    rejection_flags |= LOOM_SPIRV_COOPERATIVE_REJECTION_BIAS_INTERPRETATION;
  }
  if (row->result_type != query->result_type) {
    rejection_flags |= LOOM_SPIRV_COOPERATIVE_REJECTION_RESULT_TYPE;
  }
  return rejection_flags;
}

const loom_spirv_cooperative_vector_property_t*
loom_spirv_cooperative_vector_property_select(
    const loom_spirv_cooperative_property_set_t* property_set,
    const loom_spirv_cooperative_vector_query_t* query,
    loom_spirv_cooperative_diagnostic_t* out_diagnostic) {
  IREE_ASSERT_ARGUMENT(property_set);
  IREE_ASSERT_ARGUMENT(query);
  const loom_spirv_feature_bits_t required_feature_bits =
      iree_any_bit_set(query->flags,
                       LOOM_SPIRV_COOPERATIVE_VECTOR_FLAG_TRAINING)
          ? LOOM_SPIRV_FEATURE_COOPERATIVE_VECTOR_NV |
                LOOM_SPIRV_FEATURE_COOPERATIVE_VECTOR_TRAINING_NV
          : LOOM_SPIRV_FEATURE_COOPERATIVE_VECTOR_NV;
  const loom_spirv_feature_atom_t feature_atom =
      iree_any_bit_set(required_feature_bits,
                       LOOM_SPIRV_FEATURE_COOPERATIVE_VECTOR_TRAINING_NV)
          ? LOOM_SPIRV_FEATURE_ATOM_COOPERATIVE_VECTOR_TRAINING_NV
          : LOOM_SPIRV_FEATURE_ATOM_COOPERATIVE_VECTOR_NV;
  if (!iree_all_bits_set(property_set->feature_bits, required_feature_bits)) {
    const loom_spirv_cooperative_selection_status_t status =
        !loom_spirv_policy_requires_target_primitive(query->policy)
            ? LOOM_SPIRV_COOPERATIVE_SELECTION_FALLBACK_PERMITTED
            : LOOM_SPIRV_COOPERATIVE_SELECTION_REQUIRED_FEATURE_MISSING;
    const loom_spirv_cooperative_rejection_flags_t rejection_flags =
        LOOM_SPIRV_COOPERATIVE_REJECTION_FEATURE |
        (status == LOOM_SPIRV_COOPERATIVE_SELECTION_FALLBACK_PERMITTED
             ? LOOM_SPIRV_COOPERATIVE_REJECTION_POLICY_FALLBACK
             : 0);
    loom_spirv_cooperative_diagnostic_fail(status, feature_atom,
                                           rejection_flags, out_diagnostic);
    return NULL;
  }

  const uint64_t shape_key =
      loom_spirv_vector_shape_key(query->m_size, query->k_size);
  const loom_spirv_cooperative_property_span_t* span =
      loom_spirv_property_span_find(property_set->vector_shape_spans,
                                    property_set->vector_shape_span_count,
                                    shape_key);
  loom_spirv_cooperative_diagnostic_t diagnostic = {
      .status = LOOM_SPIRV_COOPERATIVE_SELECTION_REQUIRED_PROPERTY_MISSING,
      .feature_atom = feature_atom,
      .rejection_flags = LOOM_SPIRV_COOPERATIVE_REJECTION_SHAPE,
  };
  if (span == NULL) {
    if (!loom_spirv_policy_requires_target_primitive(query->policy)) {
      diagnostic.status = LOOM_SPIRV_COOPERATIVE_SELECTION_FALLBACK_PERMITTED;
      diagnostic.rejection_flags |=
          LOOM_SPIRV_COOPERATIVE_REJECTION_POLICY_FALLBACK;
    }
    if (out_diagnostic) {
      *out_diagnostic = diagnostic;
    }
    return NULL;
  }

  diagnostic.shape_candidate_count = span->count;
  const loom_spirv_cooperative_vector_matrix_layout_flags_t layout_bit =
      loom_spirv_vector_matrix_layout_bit(query->matrix_layout);
  const loom_spirv_storage_class_flags_t storage_class_bit =
      loom_spirv_storage_class_bit(query->storage_class);
  diagnostic.rejection_flags = LOOM_SPIRV_COOPERATIVE_REJECTION_COMPONENT_TYPE;
  loom_spirv_cooperative_rejection_flags_t type_rejection_flags =
      LOOM_SPIRV_COOPERATIVE_REJECTION_NONE;
  for (uint16_t i = 0; i < span->count; ++i) {
    const loom_spirv_cooperative_vector_property_t* row =
        &property_set->vector_properties[span->start + i];
    if (!iree_all_bits_set(property_set->feature_bits,
                           row->required_feature_bits)) {
      continue;
    }
    const loom_spirv_cooperative_rejection_flags_t row_type_rejection_flags =
        loom_spirv_cooperative_vector_component_rejection_flags(row, query);
    if (row_type_rejection_flags != LOOM_SPIRV_COOPERATIVE_REJECTION_NONE) {
      type_rejection_flags |= row_type_rejection_flags;
      continue;
    }
    ++diagnostic.type_candidate_count;
    diagnostic.rejection_flags = LOOM_SPIRV_COOPERATIVE_REJECTION_LAYOUT;
    if (layout_bit == 0 ||
        !iree_any_bit_set(row->matrix_layout_flags, layout_bit)) {
      continue;
    }
    ++diagnostic.layout_candidate_count;
    diagnostic.rejection_flags = LOOM_SPIRV_COOPERATIVE_REJECTION_STORAGE_CLASS;
    if (storage_class_bit == 0 ||
        !iree_any_bit_set(row->storage_class_flags, storage_class_bit)) {
      continue;
    }
    ++diagnostic.storage_candidate_count;
    diagnostic.rejection_flags = LOOM_SPIRV_COOPERATIVE_REJECTION_OPERANDS;
    if (!loom_spirv_cooperative_vector_flags_match(row->flags, query->flags)) {
      continue;
    }
    loom_spirv_cooperative_diagnostic_match(feature_atom, out_diagnostic);
    return row;
  }

  if (!loom_spirv_policy_requires_target_primitive(query->policy)) {
    diagnostic.status = LOOM_SPIRV_COOPERATIVE_SELECTION_FALLBACK_PERMITTED;
    diagnostic.rejection_flags |=
        LOOM_SPIRV_COOPERATIVE_REJECTION_POLICY_FALLBACK;
  }
  if (diagnostic.type_candidate_count == 0) {
    diagnostic.rejection_flags |= type_rejection_flags;
  }
  if (out_diagnostic) {
    *out_diagnostic = diagnostic;
  }
  return NULL;
}
