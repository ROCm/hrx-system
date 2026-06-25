// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXPERIMENTAL_ID4_TOOLING_RUNTIME_H_
#define EXPERIMENTAL_ID4_TOOLING_RUNTIME_H_

#include "experimental/id4/pipeline/kernel_cache.h"
#include "experimental/id4/pipeline/kernel_library.h"
#include "experimental/id4/pipeline/plan.h"
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

// Owned set of standalone HAL buffers and binding table entries.
typedef struct id4_tooling_buffer_binding_set_t {
  // Allocator used for arrays in this set.
  iree_allocator_t host_allocator;
  // Number of bindings in plan order.
  iree_host_size_t count;
  // Owned buffers backing each binding.
  iree_hal_buffer_t** buffers;
  // Binding table entries in plan order.
  iree_hal_buffer_binding_t* bindings;
} id4_tooling_buffer_binding_set_t;

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

// Creates a parameter provider for |scope| from standard --parameters= flags.
iree_status_t id4_tooling_create_parameter_provider_from_flags(
    iree_string_view_t scope, iree_allocator_t host_allocator,
    iree_io_parameter_provider_t** out_provider);

// Releases all buffers and arrays owned by |binding_set|.
void id4_tooling_buffer_binding_set_deinitialize(
    id4_tooling_buffer_binding_set_t* binding_set);

// Allocates one device-local buffer for each planned boundary tensor.
iree_status_t id4_tooling_allocate_boundary_bindings(
    iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
    const id4_pipeline_plan_t* plan, iree_allocator_t host_allocator,
    id4_tooling_buffer_binding_set_t* out_binding_set);

// Allocates one device-local buffer for each planned diagnostic tap.
iree_status_t id4_tooling_allocate_diagnostic_tap_bindings(
    iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
    const id4_pipeline_plan_t* plan, iree_allocator_t host_allocator,
    id4_tooling_buffer_binding_set_t* out_binding_set);

// Finds a boundary binding by planned boundary tensor name.
iree_status_t id4_tooling_find_boundary_binding(
    const id4_pipeline_plan_t* plan,
    const id4_tooling_buffer_binding_set_t* binding_set,
    iree_string_view_t name, iree_hal_buffer_binding_t* out_binding);

// Finds a diagnostic tap binding by planned tap name.
iree_status_t id4_tooling_find_diagnostic_tap_binding(
    const id4_pipeline_plan_t* plan,
    const id4_tooling_buffer_binding_set_t* binding_set,
    iree_string_view_t name, iree_hal_buffer_binding_t* out_binding);

// Queues chunked host-to-device updates into |binding| on |semaphore|.
iree_status_t id4_tooling_queue_update_binding(
    iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_buffer_binding_t* binding, const void* source_data,
    iree_host_size_t source_length, iree_hal_semaphore_t* semaphore,
    uint64_t* inout_payload_value);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // EXPERIMENTAL_ID4_TOOLING_RUNTIME_H_
