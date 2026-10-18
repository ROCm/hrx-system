// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_MEMORY_BACKEND_H_
#define IREE_HAL_DRIVERS_AMDGPU_MEMORY_BACKEND_H_

#include "iree/hal/allocator.h"
#include "iree/hal/drivers/amdgpu/util/libhsa.h"
#include "iree/hal/memory_backend.h"

typedef struct iree_hal_amdgpu_topology_t iree_hal_amdgpu_topology_t;
typedef struct iree_hal_amdgpu_physical_device_t
    iree_hal_amdgpu_physical_device_t;
typedef struct iree_hal_amdgpu_asan_state_t iree_hal_amdgpu_asan_state_t;

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Borrowed native resources established by device creation and group sealing.
// Only cold factories inspect these owners; created pools capture their used
// services and access facts without retaining either topology or this view.
typedef struct iree_hal_amdgpu_memory_backend_t {
  // Generic ROCr owner schema and shared construction factory.
  iree_hal_memory_backend_t base;
  // Existing HAL owner for native buffer placement and profiling.
  iree_hal_device_t* device;
  // Device-owned allocator borrowed as the legacy ROCr attachment service.
  iree_hal_allocator_t* allocator;
  // Device-owned native runtime dispatch table.
  const iree_hal_amdgpu_libhsa_t* libhsa;
  // Device-owned native agent snapshot, borrowed only during qualification.
  const iree_hal_amdgpu_topology_t* topology;
  // Existing per-family native memory, wrapper and progress owners.
  iree_hal_amdgpu_physical_device_t* const* physical_devices;
  // Device-owned sanitizer namespace and advice services.
  iree_hal_amdgpu_asan_state_t* asan_state;
  // Existing completion probe for the sealed group's shared tracker.
  iree_hal_pool_epoch_query_t epoch_query;
} iree_hal_amdgpu_memory_backend_t;

// Publishes the factory after the caller supplies existing native services.
// Creates no payload, native object, execution queue or allocation policy.
void iree_hal_amdgpu_memory_backend_initialize(
    iree_hal_amdgpu_memory_backend_t* backend);

// Native address qualification/construction used by the backend descriptor.
iree_status_t iree_hal_amdgpu_virtual_buffer_query_capabilities(
    const iree_hal_memory_backend_t* backend, iree_hal_device_group_t* group,
    iree_hal_pool_scope_t scope, iree_allocator_t host_allocator,
    iree_hal_virtual_buffer_capabilities_t* out_capabilities);
iree_status_t iree_hal_amdgpu_virtual_buffer_create(
    const iree_hal_memory_backend_t* backend, iree_hal_device_group_t* group,
    iree_hal_pool_scope_t scope,
    const iree_hal_virtual_buffer_options_t* options,
    iree_allocator_t host_allocator, iree_hal_buffer_t** out_buffer);

// Uses the legacy queue's host-call owner and publishes the complete mapping
// batch before its signal semaphores become ready.
iree_status_t iree_hal_amdgpu_memory_mapping_queue_map(
    iree_hal_queue_t* queue, iree_hal_semaphore_list_t waits,
    iree_hal_semaphore_list_t signals, iree_host_size_t mapping_count,
    const iree_hal_memory_mapping_t* mappings,
    iree_hal_queue_map_flags_t flags);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_MEMORY_BACKEND_H_
