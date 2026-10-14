// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Pass descriptor registry core.
//
// The registry is the shared dispatch surface between pass-pipeline tools and
// pass implementations. The verifier and program compiler resolve pass.run keys
// through registries before the interpreter executes descriptor callbacks.

#ifndef LOOM_PASS_REGISTRY_H_
#define LOOM_PASS_REGISTRY_H_

#include "iree/base/api.h"
#include "loom/pass/types.h"

#ifdef __cplusplus
extern "C" {
#endif

// Bitset of loom_pass_descriptor_flag_bits_e values.
typedef uint32_t loom_pass_descriptor_flags_t;

// Bitset of loom_pass_option_schema_flag_bits_e values.
typedef uint32_t loom_pass_option_schema_flags_t;

// Returns static pass metadata for one descriptor.
typedef const loom_pass_info_t* (*loom_pass_info_fn_t)(void);
typedef struct loom_pass_environment_capability_type_t
    loom_pass_environment_capability_type_t;

enum loom_pass_descriptor_flag_bits_e {
  // Descriptor is known to the registry but unavailable in this build/config.
  LOOM_PASS_DESCRIPTOR_UNAVAILABLE = 1u << 0,
};

enum loom_pass_option_schema_flag_bits_e {
  // Option must be present in the pass option dictionary.
  LOOM_PASS_OPTION_SCHEMA_REQUIRED = 1u << 0,
};

// Kind-specific validation applied to one textual pass option.
typedef enum loom_pass_option_schema_kind_e {
  // Opaque non-empty string validated by the pass create callback.
  LOOM_PASS_OPTION_SCHEMA_STRING = 0,
  // Unsigned 32-bit integer validated against an inclusive range.
  LOOM_PASS_OPTION_SCHEMA_UINT32 = 1,
  // String selected from a descriptor-owned value table.
  LOOM_PASS_OPTION_SCHEMA_ENUM = 2,
} loom_pass_option_schema_kind_t;

// One allowed value for an enum pass option.
typedef struct loom_pass_option_enum_value_t {
  // Accepted textual value.
  iree_string_view_t value;
} loom_pass_option_enum_value_t;

// Typed schema for one named pass option.
typedef struct loom_pass_option_schema_t {
  // Option name matching a loom_pass_info_t option_defs entry.
  iree_string_view_t name;
  // Option value kind.
  loom_pass_option_schema_kind_t kind;
  // Option presence and validation flags.
  loom_pass_option_schema_flags_t flags;
  // Inclusive lower bound when |kind| is LOOM_PASS_OPTION_SCHEMA_UINT32.
  uint32_t minimum_uint32;
  // Inclusive upper bound when |kind| is LOOM_PASS_OPTION_SCHEMA_UINT32.
  uint32_t maximum_uint32;
  // Sorted allowed values when |kind| is LOOM_PASS_OPTION_SCHEMA_ENUM.
  const loom_pass_option_enum_value_t* enum_values;
  // Number of entries in |enum_values|.
  uint16_t enum_value_count;
} loom_pass_option_schema_t;

// Static requirement declared by a pass descriptor.
typedef struct loom_pass_requirement_def_t {
  // Capability type that owns and can satisfy this requirement.
  const loom_pass_environment_capability_type_t* capability_type;
  // Stable requirement key.
  iree_string_view_t key;
  // Human-readable description of the requirement.
  iree_string_view_t description;
} loom_pass_requirement_def_t;

// Static descriptor for one pass implementation.
typedef struct loom_pass_descriptor_t {
  // Canonical pass key used by textual pipelines and pass.run ops.
  iree_string_view_t key;
  // Returns static pass metadata.
  loom_pass_info_fn_t info;
  union {
    // Module-pass callback when info->kind is LOOM_PASS_MODULE.
    loom_module_pass_fn_t module_run;
    // Function-pass callback when info->kind is LOOM_PASS_FUNCTION.
    loom_function_pass_fn_t function_run;
  };
  // Optional pass instance creation callback.
  loom_pass_create_fn_t create;
  // Optional pass instance destruction callback.
  loom_pass_destroy_fn_t destroy;
  // Descriptor availability flags.
  loom_pass_descriptor_flags_t flags;
  // Explanation used when the descriptor is unavailable.
  iree_string_view_t unavailable_reason;
  // Typed option schemas matching the pass info option definitions.
  const loom_pass_option_schema_t* option_schema;
  // Number of entries in |option_schema|.
  uint16_t option_schema_count;
  // Requirement metadata needed before the pass can run.
  const loom_pass_requirement_def_t* requirement_defs;
  // Number of entries in |requirement_defs|.
  uint16_t requirement_count;
} loom_pass_descriptor_t;

// A sorted registry of pass descriptors.
typedef struct loom_pass_registry_t {
  // Optional contiguous descriptors sorted by canonical key. Static registries
  // use this representation; composed registries leave it NULL.
  const loom_pass_descriptor_t* descriptors;
  // Number of descriptors in the active representation.
  iree_host_size_t descriptor_count;
  // Optional sorted references to canonical descriptors. Composed registries
  // use this representation; static registries leave it NULL.
  const loom_pass_descriptor_t* const* descriptor_refs;
} loom_pass_registry_t;

// Returns descriptor |index| from either registry representation.
static inline const loom_pass_descriptor_t* loom_pass_registry_at(
    const loom_pass_registry_t* registry, iree_host_size_t index) {
  return registry->descriptor_refs != NULL ? registry->descriptor_refs[index]
                                           : &registry->descriptors[index];
}

enum {
  LOOM_PASS_REGISTRY_STORAGE_DESCRIPTOR_CAPACITY = 256,
};

// Storage for a composed pass registry assembled from static registries.
typedef struct loom_pass_registry_storage_t {
  // Sorted references to canonical descriptors owned by source registries.
  const loom_pass_descriptor_t*
      descriptor_refs[LOOM_PASS_REGISTRY_STORAGE_DESCRIPTOR_CAPACITY];
  // Registry view over |descriptor_refs|.
  loom_pass_registry_t registry;
} loom_pass_registry_storage_t;

// Looks up |key| by exact canonical spelling, returning NULL when unknown.
const loom_pass_descriptor_t* loom_pass_registry_lookup(
    const loom_pass_registry_t* registry, iree_string_view_t key);

// Builds a sorted registry from zero or more sorted source registries.
//
// Descriptor references remain owned by the source registry packages, which
// must outlive |out_storage| and all consumers of descriptors returned from it.
// Duplicate pass keys are rejected so every textual pass.run key has a single
// owner.
iree_status_t loom_pass_registry_storage_initialize_from_registries(
    const loom_pass_registry_t* const* registries,
    iree_host_size_t registry_count, loom_pass_registry_storage_t* out_storage);

// Returns the immutable registry view stored in |storage|.
const loom_pass_registry_t* loom_pass_registry_storage_registry(
    const loom_pass_registry_storage_t* storage);

// Returns true when |descriptor| can be instantiated in the current build.
bool loom_pass_descriptor_is_available(
    const loom_pass_descriptor_t* descriptor);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_PASS_REGISTRY_H_
