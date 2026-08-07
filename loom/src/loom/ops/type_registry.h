// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// GENERATED FILE: DO NOT EDIT.
// Generator: loom.gen.ops.c_tables.
// clang-format off
#ifndef LOOM_OPS_TYPE_REGISTRY_H_
#define LOOM_OPS_TYPE_REGISTRY_H_

#include "iree/base/api.h"
#include "loom/ir/parameterized_type.h"
#include "loom/ir/types.h"
#include "loom/ops/op_defs.h"
#include "loom/ir/types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_value_fact_domain_t loom_value_fact_domain_t;
typedef struct loom_module_t loom_module_t;

// Format element kinds for type interiors (inside <...>).
// These are separate from op format elements because type
// interiors have different semantics (shape dims, element
// types, encodings) than op bodies (operand refs, attr values).
typedef enum loom_type_format_kind_e {
  LOOM_TYPE_FMT_SHAPE = 0,      // Dimension list: 4x[%M]x...
  LOOM_TYPE_FMT_SCALAR = 1,      // Element type keyword: f32, i8.
  LOOM_TYPE_FMT_ENCODING = 2,    // Encoding ref: #q8_0 or %enc.
  LOOM_TYPE_FMT_TYPE = 3,         // Recursive type: vm.ref<T>.
  LOOM_TYPE_FMT_ATTR = 4,         // Bare identifier attribute.
  LOOM_TYPE_FMT_KEYWORD = 5,      // Literal punctuation/word.
  LOOM_TYPE_FMT_OPTIONAL = 6,     // Conditional elements.
  LOOM_TYPE_FMT_GLUE = 7,         // Suppress space.
  LOOM_TYPE_FMT_PARAM = 8,        // Descriptor-backed attribute value.
  LOOM_TYPE_FMT_PARAM_KEY = 9,    // Descriptor-backed parameter name.
} loom_type_format_kind_t;

// A 4-byte format element for type interiors. Same layout
// as loom_format_element_t for consistent handling.
typedef struct loom_type_format_element_t {
  // Format opcode, encoded as loom_type_format_kind_t.
  uint8_t kind;

  // Parameter index consumed by this element.
  uint8_t field_index;

  // Kind-specific payload such as a keyword ID or skip count.
  uint16_t data;
} loom_type_format_element_t;

// Descriptor for a registered type. Contains the name,
// the IR type kind to construct, parameter count, and
// format elements describing the type interior syntax.
typedef struct loom_type_descriptor_t {
  // B-string name with a trailing '<' outside its declared length.
  const uint8_t* name;

  // What IR type kind to construct when parsing.
  loom_type_kind_t ir_kind;

  // Number of declared parameters.
  uint8_t param_count;

  // Optional type-owned value fact domain. NULL means the type only has generic
  // scalar facts or uses the domain-free extension behavior.
  const loom_value_fact_domain_t* fact_domain;

  // Semantic role and target-contract families for this type.
  loom_type_semantics_t semantics;

  // Format element array for the type interior (inside <...>).
  // NULL for opaque types (no angle brackets).
  const loom_type_format_element_t* format_elements;

  // Number of entries in |format_elements|.
  uint8_t format_element_count;

  // Descriptor-backed parameter schema, or NULL when not declared.
  const loom_parameterized_type_descriptor_t* parameterized;
} loom_type_descriptor_t;

extern const loom_parameterized_type_descriptor_t loom_encoding_type_parameterized_descriptor;
static inline iree_string_view_t loom_encoding_type_role_name(loom_encoding_role_t value) {
  loom_bstring_t name = loom_attr_descriptor_enum_case_name(
      &loom_encoding_type_parameterized_descriptor.parameter_descriptors[0], (uint8_t)value);
  return name ? loom_bstring_view(name) : iree_string_view_empty();
}
static inline bool loom_encoding_type_role_parse(iree_string_view_t name, loom_encoding_role_t* out_value) {
  uint8_t value = 0;
  if (!loom_attr_descriptor_find_enum_case(
          &loom_encoding_type_parameterized_descriptor.parameter_descriptors[0], name, &value)) {
    return false;
  }
  *out_value = (loom_encoding_role_t)value;
  return true;
}

extern const loom_parameterized_type_descriptor_t loom_low_storage_type_parameterized_descriptor;
static inline iree_string_view_t loom_low_storage_type_space_name(loom_storage_space_t value) {
  loom_bstring_t name = loom_attr_descriptor_enum_case_name(
      &loom_low_storage_type_parameterized_descriptor.parameter_descriptors[0], (uint8_t)value);
  return name ? loom_bstring_view(name) : iree_string_view_empty();
}
static inline bool loom_low_storage_type_space_parse(iree_string_view_t name, loom_storage_space_t* out_value) {
  uint8_t value = 0;
  if (!loom_attr_descriptor_find_enum_case(
          &loom_low_storage_type_parameterized_descriptor.parameter_descriptors[0], name, &value)) {
    return false;
  }
  *out_value = (loom_storage_space_t)value;
  return true;
}

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

// Entry in the sorted type registry.
typedef struct loom_type_registry_entry_t {
  iree_string_view_t name;
  const loom_type_descriptor_t* descriptor;
} loom_type_registry_entry_t;

// Returns the number of entries in the global type registry.
iree_host_size_t loom_type_registry_count(void);

// Returns the sorted registry array (for iteration/testing).
const loom_type_registry_entry_t* loom_type_registry_entries(void);

// Looks up a type descriptor by name (e.g., "tile", "hal.buffer").
// Returns the descriptor on success, NULL if not found.
const loom_type_descriptor_t* loom_type_registry_lookup(
    iree_string_view_t name);

// Looks up a registered built-in descriptor by runtime type kind.
// Returns NULL for dialect, generic parameterized, or invalid kinds.
const loom_type_descriptor_t* loom_type_registry_lookup_builtin(
    loom_type_kind_t kind);

// Resolves the type-owned value fact domain for |type|, or NULL if the
// registered type has no extension fact domain.
const loom_value_fact_domain_t* loom_type_registry_resolve_fact_domain(
    void* user_data, const loom_fact_context_t* context,
    const loom_module_t* module, loom_type_t type);

// Installs the generated type-registry fact-domain resolver on |context|.
void loom_type_registry_configure_fact_context(
    loom_fact_context_t* context);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_OPS_TYPE_REGISTRY_H_
