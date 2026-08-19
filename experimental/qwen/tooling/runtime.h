// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXPERIMENTAL_QWEN_TOOLING_RUNTIME_H_
#define EXPERIMENTAL_QWEN_TOOLING_RUNTIME_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_async_frontier_tracker_t iree_async_frontier_tracker_t;
typedef struct iree_async_proactor_pool_t iree_async_proactor_pool_t;
typedef struct iree_io_parameter_index_t iree_io_parameter_index_t;
typedef struct iree_io_parameter_provider_t iree_io_parameter_provider_t;
typedef struct qwen_tooling_compile_pool_t qwen_tooling_compile_pool_t;

// Process-local services created from the standard IREE tooling flags.
//
// A context owns exactly one HAL device group and one anonymous parameter
// index/provider pair. Programs and requests using the context must be released
// before the context is deinitialized.
typedef struct qwen_tooling_runtime_context_t {
  // Allocator used for context-owned host allocations.
  iree_allocator_t host_allocator;
  // Shared task workers used for independent Loom compilation units.
  qwen_tooling_compile_pool_t* compile_pool;
  // Proactor pool passed to the selected HAL device.
  iree_async_proactor_pool_t* proactor_pool;
  // Frontier tracker retained by the single-device group.
  iree_async_frontier_tracker_t* frontier_tracker;
  // Single-device group created from the required --device flag.
  iree_hal_device_group_t* device_group;
  // Command-buffer mode derived from the standard profiling flags.
  iree_hal_command_buffer_mode_t command_buffer_mode;
  // Anonymous parameter index used for schema and source queries.
  iree_io_parameter_index_t* parameter_index;
  // Anonymous provider serving parameters from |parameter_index|.
  iree_io_parameter_provider_t* parameter_provider;
} qwen_tooling_runtime_context_t;

// Initializes |out_context| from one --device and one anonymous --parameters
// scope. Multiple parameter files may contribute to the anonymous index, but
// named or additional scopes are rejected.
iree_status_t qwen_tooling_runtime_context_initialize_from_flags(
    iree_allocator_t host_allocator,
    qwen_tooling_runtime_context_t* out_context);

// Releases all resources owned by |context|.
void qwen_tooling_runtime_context_deinitialize(
    qwen_tooling_runtime_context_t* context);

// Returns the sole device in |context|'s group.
iree_hal_device_t* qwen_tooling_runtime_context_device(
    const qwen_tooling_runtime_context_t* context);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // EXPERIMENTAL_QWEN_TOOLING_RUNTIME_H_
