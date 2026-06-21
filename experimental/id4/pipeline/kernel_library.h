// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXPERIMENTAL_ID4_PIPELINE_KERNEL_LIBRARY_H_
#define EXPERIMENTAL_ID4_PIPELINE_KERNEL_LIBRARY_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// In-memory library of Loom modules addressable by stable module paths.
typedef struct id4_pipeline_kernel_library_t id4_pipeline_kernel_library_t;

// Compile-time configuration binding passed to Loom config resolution.
typedef struct id4_pipeline_kernel_config_binding_t {
  // Config symbol key, with or without a leading at-sign.
  iree_string_view_t key;
  // Config value spelling.
  iree_string_view_t value;
} id4_pipeline_kernel_config_binding_t;

// Exported Loom kernel function selected by model authoring code.
typedef struct id4_pipeline_kernel_ref_t {
  // Stable module path such as "model_family/kernel_family".
  iree_string_view_t module_path;
  // Exported kernel function name inside the module.
  iree_string_view_t function_name;
} id4_pipeline_kernel_ref_t;

// Returns a compile-time configuration binding value.
static inline id4_pipeline_kernel_config_binding_t
id4_pipeline_make_kernel_config_binding(iree_string_view_t key,
                                        iree_string_view_t value) {
  id4_pipeline_kernel_config_binding_t binding;
  binding.key = key;
  binding.value = value;
  return binding;
}

// Returns an exported Loom kernel reference value.
static inline id4_pipeline_kernel_ref_t id4_pipeline_make_kernel_ref(
    iree_string_view_t module_path, iree_string_view_t function_name) {
  id4_pipeline_kernel_ref_t ref;
  ref.module_path = module_path;
  ref.function_name = function_name;
  return ref;
}

// Borrowed module source entry used to create a kernel library.
typedef struct id4_pipeline_kernel_module_t {
  // Stable module path used by plans, dispatches, and source lookup.
  iree_string_view_t module_path;
  // Diagnostic source identifier, usually "kernels/<module_path>.loom".
  iree_string_view_t source_identifier;
  // In-memory Loom source contents.
  iree_const_byte_span_t source_contents;
} id4_pipeline_kernel_module_t;

// Borrowed Loom source file entry used to create a kernel library.
typedef struct id4_pipeline_kernel_source_file_t {
  // Source identifier formatted as "<module_path>.loom".
  iree_string_view_t source_identifier;
  // In-memory Loom source contents.
  iree_const_byte_span_t source_contents;
} id4_pipeline_kernel_source_file_t;

// Options for creating an in-memory kernel library.
typedef struct id4_pipeline_kernel_library_create_options_t {
  // Size of this structure for versioning.
  iree_host_size_t structure_size;
  // Extension structure chain; must be NULL for now.
  const void* next;
  // Number of module entries to copy.
  iree_host_size_t module_count;
  // Module entries copied into the library.
  const id4_pipeline_kernel_module_t* modules;
} id4_pipeline_kernel_library_create_options_t;

// Creates a kernel library by copying module metadata and source contents.
iree_status_t id4_pipeline_kernel_library_create(
    const id4_pipeline_kernel_library_create_options_t* options,
    iree_allocator_t host_allocator,
    id4_pipeline_kernel_library_t** out_library);

// Creates a kernel library from source files named "<module_path>.loom".
iree_status_t id4_pipeline_kernel_library_create_from_source_files(
    iree_host_size_t source_file_count,
    const id4_pipeline_kernel_source_file_t* source_files,
    iree_allocator_t host_allocator,
    id4_pipeline_kernel_library_t** out_library);

// Retains |library| for the caller.
void id4_pipeline_kernel_library_retain(id4_pipeline_kernel_library_t* library);

// Releases |library| from the caller.
void id4_pipeline_kernel_library_release(
    id4_pipeline_kernel_library_t* library);

// Returns the number of modules in |library|.
iree_host_size_t id4_pipeline_kernel_library_module_count(
    const id4_pipeline_kernel_library_t* library);

// Returns module |index| or NULL when out of range.
const id4_pipeline_kernel_module_t* id4_pipeline_kernel_library_module_at(
    const id4_pipeline_kernel_library_t* library, iree_host_size_t index);

// Finds the module with |module_path|.
iree_status_t id4_pipeline_kernel_library_lookup(
    const id4_pipeline_kernel_library_t* library,
    iree_string_view_t module_path,
    const id4_pipeline_kernel_module_t** out_module);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // EXPERIMENTAL_ID4_PIPELINE_KERNEL_LIBRARY_H_
