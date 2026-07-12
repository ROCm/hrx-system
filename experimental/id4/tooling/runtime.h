// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXPERIMENTAL_ID4_TOOLING_RUNTIME_H_
#define EXPERIMENTAL_ID4_TOOLING_RUNTIME_H_

#include "experimental/id4/pipeline/kernel_cache.h"
#include "experimental/id4/pipeline/kernel_library.h"
#include "experimental/id4/pipeline/stage.h"
#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/io/parameter_provider.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_async_frontier_tracker_t iree_async_frontier_tracker_t;
typedef struct iree_async_proactor_pool_t iree_async_proactor_pool_t;

// Process-local HAL and Loom services created from standard IREE tooling flags.
typedef struct id4_tooling_runtime_context_t {
  // Allocator used for runtime-context storage.
  iree_allocator_t host_allocator;
  // Proactor pool passed to HAL devices that support asynchronous host I/O.
  iree_async_proactor_pool_t* proactor_pool;
  // Frontier tracker retained by the HAL device group.
  iree_async_frontier_tracker_t* frontier_tracker;
  // Device group created from every --device= flag value.
  iree_hal_device_group_t* device_group;
  // HAL executable cache for executables prepared against device index zero.
  iree_hal_executable_cache_t* executable_cache;
  // Loom kernel cache shared by prepared stages.
  id4_pipeline_kernel_cache_t* kernel_cache;
  // Command-buffer mode selected from HAL profiling flags.
  // Profile-retained modes preserve command operation and dispatch attribution.
  iree_hal_command_buffer_mode_t command_buffer_mode;
} id4_tooling_runtime_context_t;

// Options for creating a runtime context from parsed flags.
typedef struct id4_tooling_runtime_context_options_t {
  // Size of this structure for versioning.
  iree_host_size_t structure_size;
  // Extension structure chain; must be NULL for now.
  const void* next;
  // HAL executable cache identifier used for prepared kernels.
  iree_string_view_t executable_cache_identifier;
} id4_tooling_runtime_context_options_t;

// Initializes |out_context| from standard --device= and profiling flags.
iree_status_t id4_tooling_runtime_context_initialize_from_flags(
    const id4_tooling_runtime_context_options_t* options,
    iree_allocator_t host_allocator,
    id4_tooling_runtime_context_t* out_context);

// Releases all resources owned by |context|.
void id4_tooling_runtime_context_deinitialize(
    id4_tooling_runtime_context_t* context);

// Returns the primary device, currently device index zero in |context|.
iree_hal_device_t* id4_tooling_runtime_context_primary_device(
    const id4_tooling_runtime_context_t* context);

// Returns stage services backed by |context|.
id4_pipeline_stage_services_t id4_tooling_runtime_context_stage_services(
    const id4_tooling_runtime_context_t* context);

// Creates a kernel library from embedded ID4 Loom source files.
iree_status_t id4_tooling_create_embedded_kernel_library(
    iree_allocator_t host_allocator,
    id4_pipeline_kernel_library_t** out_library);

// One scoped parameter provider requested from standard --parameters= flags.
typedef struct id4_tooling_parameter_provider_request_t {
  // Parameter scope name required by the caller.
  iree_string_view_t scope;
  // Receives a newly-created provider for |scope| on success.
  iree_io_parameter_provider_t** out_provider;
  // Optionally receives the retained index exposed by |out_provider|.
  iree_io_parameter_index_t** out_index;
} id4_tooling_parameter_provider_request_t;

// One scoped child provider retained by a parameter provider set.
typedef struct id4_tooling_parameter_provider_set_entry_t {
  // Parameter scope routed to |provider|.
  iree_string_view_t scope;
  // Provider that must support |scope|.
  iree_io_parameter_provider_t* provider;
} id4_tooling_parameter_provider_set_entry_t;

// Creates one provider that routes operations to scoped child providers.
iree_status_t id4_tooling_create_parameter_provider_set(
    iree_host_size_t entry_count,
    const id4_tooling_parameter_provider_set_entry_t* entries,
    iree_allocator_t host_allocator,
    iree_io_parameter_provider_t** out_provider);

// Creates one provider for |scopes| from standard --parameters= flags.
iree_status_t id4_tooling_create_parameter_provider_set_from_flags(
    iree_host_size_t scope_count, const iree_string_view_t* scopes,
    iree_allocator_t host_allocator,
    iree_io_parameter_provider_t** out_provider);

// Creates scoped parameter providers from one shared parse of --parameters=.
iree_status_t id4_tooling_create_parameter_providers_from_flags(
    iree_host_size_t request_count,
    const id4_tooling_parameter_provider_request_t* requests,
    iree_allocator_t host_allocator);

// Creates a parameter provider for |scope| from standard --parameters= flags.
iree_status_t id4_tooling_create_parameter_provider_from_flags(
    iree_string_view_t scope, iree_allocator_t host_allocator,
    iree_io_parameter_provider_t** out_provider);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // EXPERIMENTAL_ID4_TOOLING_RUNTIME_H_
