// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_VIRTUAL_MEMORY_H_
#define IREE_HAL_DRIVERS_AMDGPU_VIRTUAL_MEMORY_H_

#include "iree/base/api.h"
#include "iree/hal/allocator.h"
#include "iree/hal/drivers/amdgpu/util/libhsa.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_hal_amdgpu_topology_t iree_hal_amdgpu_topology_t;
typedef struct iree_hal_amdgpu_virtual_memory_state_t
    iree_hal_amdgpu_virtual_memory_state_t;

// Placement and capability information selected by the AMDGPU allocator.
// The virtual-memory component consumes this representation without owning the
// allocator's broader pool-selection policy.
typedef struct iree_hal_amdgpu_virtual_memory_placement_t {
  // HSA memory pool used for the physical allocation.
  hsa_amd_memory_pool_t memory_pool;

  // HAL device that owns virtual-buffer placement metadata.
  iree_hal_device_t* device;

  // HAL queues permitted to access the virtual address range.
  iree_hal_queue_affinity_t queue_affinity;

  // HAL memory type exposed by the virtual buffer and physical allocation.
  iree_hal_memory_type_t memory_type;

  // HAL buffer usage exposed by the virtual buffer.
  iree_hal_buffer_usage_t buffer_usage;

  // Smallest legal HSA VMM allocation and mapping multiple.
  iree_device_size_t minimum_granule;

  // Preferred HSA VMM allocation multiple for reducing fragmentation.
  iree_device_size_t recommended_granule;

  // Maximum physical allocation size accepted by this memory pool.
  iree_device_size_t max_allocation_size;
} iree_hal_amdgpu_virtual_memory_placement_t;

// Creates the virtual-memory context for one AMDGPU allocator.
iree_status_t iree_hal_amdgpu_virtual_memory_state_create(
    const iree_hal_amdgpu_libhsa_t* libhsa,
    const iree_hal_amdgpu_topology_t* topology, iree_hal_device_t* device,
    iree_hal_queue_affinity_t supported_queue_affinity,
    iree_hal_allocator_statistics_t* statistics,
    iree_allocator_t host_allocator,
    iree_hal_amdgpu_virtual_memory_state_t** out_state);

// Destroys an allocator-owned VMM context after all handles are released.
void iree_hal_amdgpu_virtual_memory_state_destroy(
    iree_hal_amdgpu_virtual_memory_state_t* state);

// Reserves a virtual buffer with no physical backing or access permissions.
iree_status_t iree_hal_amdgpu_virtual_memory_reserve(
    iree_hal_amdgpu_virtual_memory_state_t* state,
    iree_hal_amdgpu_virtual_memory_placement_t placement,
    iree_device_size_t size, iree_hal_buffer_t** out_virtual_buffer);

// Releases an unmapped virtual buffer returned by virtual_memory_reserve.
iree_status_t iree_hal_amdgpu_virtual_memory_release(
    iree_hal_amdgpu_virtual_memory_state_t* state,
    iree_hal_buffer_t* virtual_buffer);

// Creates a physical allocation that can be mapped into one or more virtual
// buffers associated with this state.
iree_status_t iree_hal_amdgpu_physical_memory_allocate(
    iree_hal_amdgpu_virtual_memory_state_t* state,
    iree_hal_amdgpu_virtual_memory_placement_t placement,
    iree_device_size_t size, iree_allocator_t host_allocator,
    iree_hal_physical_memory_t** out_physical_memory);

// Releases an unmapped physical allocation.
iree_status_t iree_hal_amdgpu_physical_memory_free(
    iree_hal_amdgpu_virtual_memory_state_t* state,
    iree_hal_physical_memory_t* physical_memory);

// Maps physical memory into an unmapped virtual address range.
iree_status_t iree_hal_amdgpu_virtual_memory_map(
    iree_hal_amdgpu_virtual_memory_state_t* state,
    iree_hal_buffer_t* virtual_buffer, iree_device_size_t virtual_offset,
    iree_hal_physical_memory_t* physical_memory,
    iree_device_size_t physical_offset, iree_device_size_t size);

// Unmaps a fully mapped virtual address range.
iree_status_t iree_hal_amdgpu_virtual_memory_unmap(
    iree_hal_amdgpu_virtual_memory_state_t* state,
    iree_hal_buffer_t* virtual_buffer, iree_device_size_t virtual_offset,
    iree_device_size_t size);

// Sets the requested access permissions on a mapped virtual address range.
iree_status_t iree_hal_amdgpu_virtual_memory_protect(
    iree_hal_amdgpu_virtual_memory_state_t* state,
    iree_hal_buffer_t* virtual_buffer, iree_device_size_t virtual_offset,
    iree_device_size_t size, iree_hal_queue_affinity_t queue_affinity,
    iree_hal_virtual_memory_access_scope_t access_scope,
    iree_hal_memory_protection_t protection);

// Validates an advisory virtual-address range and otherwise performs no work.
iree_status_t iree_hal_amdgpu_virtual_memory_advise(
    iree_hal_amdgpu_virtual_memory_state_t* state,
    iree_hal_buffer_t* virtual_buffer, iree_device_size_t virtual_offset,
    iree_device_size_t size, iree_hal_queue_affinity_t queue_affinity,
    iree_hal_memory_advice_t advice);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_VIRTUAL_MEMORY_H_
