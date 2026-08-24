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
#include "loom/ir/type_descriptor.h"
#include "loom/ops/op_defs.h"
#include "loom/ir/types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_module_t loom_module_t;
typedef struct loom_context_t loom_context_t;

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

// Function-reference yieldability. Absent (0) guarantees synchronous execution.
typedef enum loom_func_ref_type_yieldability_e {
  LOOM_FUNC_REF_TYPE_YIELDABILITY_YIELDABLE = 1,
  LOOM_FUNC_REF_TYPE_YIELDABILITY_COUNT_ = 2,
} loom_func_ref_type_yieldability_t;

extern const loom_parameterized_type_descriptor_t loom_func_ref_type_parameterized_descriptor;
enum loom_func_ref_type_build_flag_bits_e {
  LOOM_FUNC_REF_TYPE_BUILD_FLAG_HAS_YIELDABILITY = 1u << 0,
};
typedef uint32_t loom_func_ref_type_build_flags_t;

static inline bool loom_func_ref_type_isa(loom_type_t type) {
  return loom_type_is_parameterized(type) && loom_type_parameterized_descriptor(type) == &loom_func_ref_type_parameterized_descriptor;
}
enum { LOOM_FUNC_REF_TYPE_YIELDABILITY_PARAMETER_INDEX = 0 };
static inline bool loom_func_ref_type_has_yieldability(loom_type_t type) {
  return !loom_attr_is_absent(loom_type_parameterized_parameters(type)[LOOM_FUNC_REF_TYPE_YIELDABILITY_PARAMETER_INDEX]);
}
static inline loom_func_ref_type_yieldability_t loom_func_ref_type_yieldability(loom_type_t type) {
  return (loom_func_ref_type_yieldability_t)loom_attr_as_enum(loom_type_parameterized_parameters(type)[LOOM_FUNC_REF_TYPE_YIELDABILITY_PARAMETER_INDEX]);
}
enum { LOOM_FUNC_REF_TYPE_SIGNATURE_PARAMETER_INDEX = 1 };
static inline loom_type_id_t loom_func_ref_type_signature(loom_type_t type) {
  return loom_attr_as_type_id(loom_type_parameterized_parameters(type)[LOOM_FUNC_REF_TYPE_SIGNATURE_PARAMETER_INDEX]);
}
iree_status_t loom_func_ref_type_make(
    loom_module_t* module,
    loom_func_ref_type_build_flags_t build_flags,
    loom_func_ref_type_yieldability_t yieldability,
    loom_type_id_t signature,
    loom_type_t* out_type);

// Returns the number of entries in the common type registry.
iree_host_size_t loom_type_registry_count(void);

// Returns the sorted common registry array (for iteration/testing).
const loom_type_registry_entry_t* loom_type_registry_entries(void);

// Registers a generated dialect-owned type table with |context|.
// Common and previously registered names cannot be replaced.
iree_status_t loom_type_registry_register_types(
    loom_context_t* context, const loom_type_registry_entry_t* entries,
    iree_host_size_t entry_count);

// Looks up a common or context-registered type by name.
// Returns the descriptor on success, NULL if not found.
const loom_type_descriptor_t* loom_type_registry_lookup(
    const loom_context_t* context, iree_string_view_t name);

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
