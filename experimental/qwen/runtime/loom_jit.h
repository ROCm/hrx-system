// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXPERIMENTAL_QWEN_RUNTIME_LOOM_JIT_H_
#define EXPERIMENTAL_QWEN_RUNTIME_LOOM_JIT_H_

#include <stdint.h>

#include "experimental/qwen/runtime/loom_source.h"
#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "loomc/sanitizer.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Default number of exact prepared executables retained by one JIT.
#define QWEN_LOOM_JIT_DEFAULT_ENTRY_LIMIT 64

// Reusable device-specific Loom compiler and executable cache.
typedef struct qwen_loom_jit_t qwen_loom_jit_t;

// Prepared executable for one exact source/config/workload specialization.
typedef struct qwen_loom_executable_t qwen_loom_executable_t;

// Compile-time configuration binding passed to Loom selective linking.
typedef struct qwen_loom_config_binding_t {
  // Config symbol key, with or without a leading at-sign.
  iree_string_view_t key;
  // Config value spelling.
  iree_string_view_t value;
} qwen_loom_config_binding_t;

// Options for creating a device-specific Loom JIT.
typedef struct qwen_loom_jit_options_t {
  // Size of this structure for versioning.
  iree_host_size_t structure_size;
  // Extension structure chain; must be NULL.
  const void* next;
  // HAL device used for target discovery and executable loading.
  iree_hal_device_t* device;
  // Queue affinity on which prepared executables may run.
  iree_hal_queue_affinity_t queue_affinity;
  // Maximum exact prepared executables retained by the cache.
  iree_host_size_t entry_limit;
  // Loom sanitizer assertions compiled into prepared executables.
  loomc_sanitizer_checks_t sanitizer_checks;
} qwen_loom_jit_options_t;

// Options for preparing one exact exported kernel executable.
typedef struct qwen_loom_jit_prepare_options_t {
  // Size of this structure for versioning.
  iree_host_size_t structure_size;
  // Extension structure chain; must be NULL.
  const void* next;
  // Embedded linked module containing the selected root.
  const qwen_loom_source_module_t* source_module;
  // Exported Loom function selected as the only link root.
  iree_string_view_t function_name;
  // Number of compile-time config bindings.
  iree_host_size_t config_binding_count;
  // Config bindings borrowed for the prepare call.
  const qwen_loom_config_binding_t* config_bindings;
  // Number of positional signed workload arguments.
  iree_host_size_t workload_argument_count;
  // Workload arguments borrowed for the prepare call.
  const int64_t* workload_arguments;
} qwen_loom_jit_prepare_options_t;

// Creates a JIT specialized to one HAL device and queue affinity.
iree_status_t qwen_loom_jit_create(const qwen_loom_jit_options_t* options,
                                   iree_allocator_t host_allocator,
                                   qwen_loom_jit_t** out_jit);

// Retains |jit| for the caller.
void qwen_loom_jit_retain(qwen_loom_jit_t* jit);

// Releases |jit| from the caller.
void qwen_loom_jit_release(qwen_loom_jit_t* jit);

// Prepares or reuses one exact exported executable.
//
// Source identity and bytes, function name, config bindings, and workload
// arguments all participate in cache identity. Concurrent cache misses are
// serialized by the JIT.
iree_status_t qwen_loom_jit_prepare(
    qwen_loom_jit_t* jit, const qwen_loom_jit_prepare_options_t* options,
    qwen_loom_executable_t** out_executable);

// Returns the current number of retained exact cache entries.
iree_host_size_t qwen_loom_jit_entry_count(qwen_loom_jit_t* jit);

// Retains |executable| for the caller.
void qwen_loom_executable_retain(qwen_loom_executable_t* executable);

// Releases |executable| from the caller.
void qwen_loom_executable_release(qwen_loom_executable_t* executable);

// Returns the retained HAL executable. The returned handle is borrowed.
iree_hal_executable_t* qwen_loom_executable_hal_executable(
    const qwen_loom_executable_t* executable);

// Returns the resolved exported HAL function.
iree_hal_executable_function_t qwen_loom_executable_function(
    const qwen_loom_executable_t* executable);

// Returns the resolved static workgroup count.
//
// The loaded executable function owns the matching static workgroup size, so
// the returned dispatch configuration carries no redundant size override.
iree_hal_dispatch_config_t qwen_loom_executable_dispatch_config(
    const qwen_loom_executable_t* executable);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // EXPERIMENTAL_QWEN_RUNTIME_LOOM_JIT_H_
