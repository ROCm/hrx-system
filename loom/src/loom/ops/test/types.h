// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// GENERATED FILE: DO NOT EDIT.
// Generator: loom.gen.ops.c_tables.
// clang-format off
#ifndef LOOM_OPS_TEST_TYPES_H_
#define LOOM_OPS_TEST_TYPES_H_

#include "loom/ops/type_registry.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_module_t loom_module_t;

// Synthetic scope for parameterized value coverage.
typedef enum loom_test_scope_type_scope_e {
  LOOM_TEST_SCOPE_TYPE_SCOPE_WORKGROUP = 1,
  LOOM_TEST_SCOPE_TYPE_SCOPE_SUBGROUP = 2,
  LOOM_TEST_SCOPE_TYPE_SCOPE_COUNT_ = 3,
} loom_test_scope_type_scope_t;

extern const loom_parameterized_type_descriptor_t loom_test_scope_type_parameterized_descriptor;
static inline bool loom_test_scope_type_isa(loom_type_t type) {
  return loom_type_is_parameterized(type) && loom_type_parameterized_descriptor(type) == &loom_test_scope_type_parameterized_descriptor;
}
enum { LOOM_TEST_SCOPE_TYPE_SCOPE_PARAMETER_INDEX = 0 };
static inline loom_test_scope_type_scope_t loom_test_scope_type_scope(loom_type_t type) {
  return (loom_test_scope_type_scope_t)loom_attr_as_enum(loom_type_parameterized_parameters(type)[LOOM_TEST_SCOPE_TYPE_SCOPE_PARAMETER_INDEX]);
}
iree_status_t loom_test_scope_type_make(
    loom_module_t* module,
    loom_test_scope_type_scope_t scope,
    loom_type_t* out_type);

// Synthetic scope for parameterized value coverage.
typedef enum loom_test_matrix_type_scope_e {
  LOOM_TEST_MATRIX_TYPE_SCOPE_WORKGROUP = 1,
  LOOM_TEST_MATRIX_TYPE_SCOPE_SUBGROUP = 2,
  LOOM_TEST_MATRIX_TYPE_SCOPE_COUNT_ = 3,
} loom_test_matrix_type_scope_t;

extern const loom_parameterized_type_descriptor_t loom_test_matrix_type_parameterized_descriptor;
enum loom_test_matrix_type_build_flag_bits_e {
  LOOM_TEST_MATRIX_TYPE_BUILD_FLAG_HAS_TARGET = 1u << 0,
};
typedef uint32_t loom_test_matrix_type_build_flags_t;

static inline bool loom_test_matrix_type_isa(loom_type_t type) {
  return loom_type_is_parameterized(type) && loom_type_parameterized_descriptor(type) == &loom_test_matrix_type_parameterized_descriptor;
}
enum { LOOM_TEST_MATRIX_TYPE_ELEMENT_TYPE_PARAMETER_INDEX = 0 };
static inline loom_type_id_t loom_test_matrix_type_element_type(loom_type_t type) {
  return loom_attr_as_type_id(loom_type_parameterized_parameters(type)[LOOM_TEST_MATRIX_TYPE_ELEMENT_TYPE_PARAMETER_INDEX]);
}
enum { LOOM_TEST_MATRIX_TYPE_SCOPE_PARAMETER_INDEX = 1 };
static inline loom_test_matrix_type_scope_t loom_test_matrix_type_scope(loom_type_t type) {
  return (loom_test_matrix_type_scope_t)loom_attr_as_enum(loom_type_parameterized_parameters(type)[LOOM_TEST_MATRIX_TYPE_SCOPE_PARAMETER_INDEX]);
}
enum { LOOM_TEST_MATRIX_TYPE_ROWS_PARAMETER_INDEX = 2 };
static inline int64_t loom_test_matrix_type_rows(loom_type_t type) {
  return loom_attr_as_i64(loom_type_parameterized_parameters(type)[LOOM_TEST_MATRIX_TYPE_ROWS_PARAMETER_INDEX]);
}
enum { LOOM_TEST_MATRIX_TYPE_TARGET_PARAMETER_INDEX = 3 };
static inline bool loom_test_matrix_type_has_target(loom_type_t type) {
  return !loom_attr_is_absent(loom_type_parameterized_parameters(type)[LOOM_TEST_MATRIX_TYPE_TARGET_PARAMETER_INDEX]);
}
static inline loom_symbol_ref_t loom_test_matrix_type_target(loom_type_t type) {
  return loom_attr_as_symbol(loom_type_parameterized_parameters(type)[LOOM_TEST_MATRIX_TYPE_TARGET_PARAMETER_INDEX]);
}
iree_status_t loom_test_matrix_type_make(
    loom_module_t* module,
    loom_test_matrix_type_build_flags_t build_flags,
    loom_type_id_t element_type,
    loom_test_matrix_type_scope_t scope,
    int64_t rows,
    loom_symbol_ref_t target,
    loom_type_t* out_type);

// Synthetic scope for parameterized value coverage.
typedef enum loom_test_compact_matrix_type_scope_e {
  LOOM_TEST_COMPACT_MATRIX_TYPE_SCOPE_WORKGROUP = 1,
  LOOM_TEST_COMPACT_MATRIX_TYPE_SCOPE_SUBGROUP = 2,
  LOOM_TEST_COMPACT_MATRIX_TYPE_SCOPE_COUNT_ = 3,
} loom_test_compact_matrix_type_scope_t;

extern const loom_parameterized_type_descriptor_t loom_test_compact_matrix_type_parameterized_descriptor;
static inline bool loom_test_compact_matrix_type_isa(loom_type_t type) {
  return loom_type_is_parameterized(type) && loom_type_parameterized_descriptor(type) == &loom_test_compact_matrix_type_parameterized_descriptor;
}
enum { LOOM_TEST_COMPACT_MATRIX_TYPE_ROWS_PARAMETER_INDEX = 0 };
static inline int64_t loom_test_compact_matrix_type_rows(loom_type_t type) {
  return loom_attr_as_i64(loom_type_parameterized_parameters(type)[LOOM_TEST_COMPACT_MATRIX_TYPE_ROWS_PARAMETER_INDEX]);
}
enum { LOOM_TEST_COMPACT_MATRIX_TYPE_COLUMNS_PARAMETER_INDEX = 1 };
static inline int64_t loom_test_compact_matrix_type_columns(loom_type_t type) {
  return loom_attr_as_i64(loom_type_parameterized_parameters(type)[LOOM_TEST_COMPACT_MATRIX_TYPE_COLUMNS_PARAMETER_INDEX]);
}
enum { LOOM_TEST_COMPACT_MATRIX_TYPE_ELEMENT_TYPE_PARAMETER_INDEX = 2 };
static inline loom_type_id_t loom_test_compact_matrix_type_element_type(loom_type_t type) {
  return loom_attr_as_type_id(loom_type_parameterized_parameters(type)[LOOM_TEST_COMPACT_MATRIX_TYPE_ELEMENT_TYPE_PARAMETER_INDEX]);
}
enum { LOOM_TEST_COMPACT_MATRIX_TYPE_SCOPE_PARAMETER_INDEX = 3 };
static inline loom_test_compact_matrix_type_scope_t loom_test_compact_matrix_type_scope(loom_type_t type) {
  return (loom_test_compact_matrix_type_scope_t)loom_attr_as_enum(loom_type_parameterized_parameters(type)[LOOM_TEST_COMPACT_MATRIX_TYPE_SCOPE_PARAMETER_INDEX]);
}
iree_status_t loom_test_compact_matrix_type_make(
    loom_module_t* module,
    int64_t rows,
    int64_t columns,
    loom_type_id_t element_type,
    loom_test_compact_matrix_type_scope_t scope,
    loom_type_t* out_type);

extern const loom_parameterized_type_descriptor_t loom_test_array_type_parameterized_descriptor;
enum loom_test_array_type_build_flag_bits_e {
  LOOM_TEST_ARRAY_TYPE_BUILD_FLAG_HAS_ALIGNMENT = 1u << 0,
  LOOM_TEST_ARRAY_TYPE_BUILD_FLAG_HAS_METADATA = 1u << 1,
};
typedef uint32_t loom_test_array_type_build_flags_t;

static inline bool loom_test_array_type_isa(loom_type_t type) {
  return loom_type_is_parameterized(type) && loom_type_parameterized_descriptor(type) == &loom_test_array_type_parameterized_descriptor;
}
enum { LOOM_TEST_ARRAY_TYPE_ELEMENT_TYPE_PARAMETER_INDEX = 0 };
static inline loom_type_id_t loom_test_array_type_element_type(loom_type_t type) {
  return loom_attr_as_type_id(loom_type_parameterized_parameters(type)[LOOM_TEST_ARRAY_TYPE_ELEMENT_TYPE_PARAMETER_INDEX]);
}
enum { LOOM_TEST_ARRAY_TYPE_ALIGNMENT_PARAMETER_INDEX = 1 };
static inline bool loom_test_array_type_has_alignment(loom_type_t type) {
  return !loom_attr_is_absent(loom_type_parameterized_parameters(type)[LOOM_TEST_ARRAY_TYPE_ALIGNMENT_PARAMETER_INDEX]);
}
static inline int64_t loom_test_array_type_alignment(loom_type_t type) {
  return loom_attr_as_i64(loom_type_parameterized_parameters(type)[LOOM_TEST_ARRAY_TYPE_ALIGNMENT_PARAMETER_INDEX]);
}
enum { LOOM_TEST_ARRAY_TYPE_METADATA_PARAMETER_INDEX = 2 };
static inline bool loom_test_array_type_has_metadata(loom_type_t type) {
  return !loom_attr_is_absent(loom_type_parameterized_parameters(type)[LOOM_TEST_ARRAY_TYPE_METADATA_PARAMETER_INDEX]);
}
static inline loom_named_attr_slice_t loom_test_array_type_metadata(loom_type_t type) {
  return loom_attr_as_dict(loom_type_parameterized_parameters(type)[LOOM_TEST_ARRAY_TYPE_METADATA_PARAMETER_INDEX]);
}
iree_status_t loom_test_array_type_make(
    loom_module_t* module,
    loom_test_array_type_build_flags_t build_flags,
    loom_type_id_t element_type,
    int64_t alignment,
    loom_named_attr_slice_t metadata,
    loom_type_t* out_type);

extern const loom_parameterized_type_descriptor_t loom_test_variant_set_type_parameterized_descriptor;
enum loom_test_variant_set_type_build_flag_bits_e {
  LOOM_TEST_VARIANT_SET_TYPE_BUILD_FLAG_HAS_ALTERNATIVES = 1u << 0,
};
typedef uint32_t loom_test_variant_set_type_build_flags_t;

static inline bool loom_test_variant_set_type_isa(loom_type_t type) {
  return loom_type_is_parameterized(type) && loom_type_parameterized_descriptor(type) == &loom_test_variant_set_type_parameterized_descriptor;
}
enum { LOOM_TEST_VARIANT_SET_TYPE_VALUES_PARAMETER_INDEX = 0 };
static inline loom_parameterized_attr_array_t loom_test_variant_set_type_values(loom_type_t type) {
  return loom_attr_as_parameterized_array(loom_type_parameterized_parameters(type)[LOOM_TEST_VARIANT_SET_TYPE_VALUES_PARAMETER_INDEX]);
}
enum { LOOM_TEST_VARIANT_SET_TYPE_ALTERNATIVES_PARAMETER_INDEX = 1 };
static inline bool loom_test_variant_set_type_has_alternatives(loom_type_t type) {
  return !loom_attr_is_absent(loom_type_parameterized_parameters(type)[LOOM_TEST_VARIANT_SET_TYPE_ALTERNATIVES_PARAMETER_INDEX]);
}
static inline loom_parameterized_attr_array_t loom_test_variant_set_type_alternatives(loom_type_t type) {
  return loom_attr_as_parameterized_array(loom_type_parameterized_parameters(type)[LOOM_TEST_VARIANT_SET_TYPE_ALTERNATIVES_PARAMETER_INDEX]);
}
iree_status_t loom_test_variant_set_type_make(
    loom_module_t* module,
    loom_test_variant_set_type_build_flags_t build_flags,
    loom_parameterized_attr_array_t values,
    loom_parameterized_attr_array_t alternatives,
    loom_type_t* out_type);

// Generated type descriptors owned by this dialect.
extern const loom_type_registry_entry_t loom_test_type_registry_entries[];
extern const iree_host_size_t loom_test_type_registry_entry_count;

#ifdef __cplusplus
}
#endif

#endif  // LOOM_OPS_TEST_TYPES_H_
