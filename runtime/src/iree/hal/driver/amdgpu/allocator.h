// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVER_AMDGPU_ALLOCATOR_H_
#define IREE_HAL_DRIVER_AMDGPU_ALLOCATOR_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/driver/amdgpu/util/libhsa.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_hal_amdgpu_logical_device_t
    iree_hal_amdgpu_logical_device_t;
typedef struct iree_hal_amdgpu_topology_t iree_hal_amdgpu_topology_t;

// Exact native properties used to allocate or attach one AMDGPU buffer.
//
// This is private allocation policy consumed by the retained ROCr allocator
// and slab providers. Public pools fix these properties at construction time;
// individual reservations carry only their size and alignment.
typedef struct iree_hal_amdgpu_buffer_spec_t {
  // Memory properties required or preferred during HSA pool selection.
  iree_hal_memory_type_t type;
  // Access permissions published by the resulting HAL buffer.
  iree_hal_memory_access_t access;
  // Native usage supported by the selected allocation or attachment.
  iree_hal_buffer_usage_t usage;
  // HAL queue families permitted to access the native storage.
  iree_hal_queue_family_affinity_t queue_family_affinity;
  // Minimum required native address alignment, or zero for the native default.
  iree_device_size_t min_alignment;
} iree_hal_amdgpu_buffer_spec_t;

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_allocator_t
//===----------------------------------------------------------------------===//

// Creates a buffer allocator that allocates from HSA memory pools.
//
// Direct allocations use HSA memory pools and publish sanitizer shadow state
// when enabled. Public allocations use explicit pools so redzone and quarantine
// policy stays attached to the caller-selected owner.
//
// |logical_device| is unretained and must outlive the allocator.
iree_status_t iree_hal_amdgpu_allocator_create(
    iree_hal_amdgpu_logical_device_t* logical_device,
    const iree_hal_amdgpu_libhsa_t* libhsa,
    const iree_hal_amdgpu_topology_t* topology, iree_allocator_t host_allocator,
    iree_hal_allocator_t** out_allocator);

// Returns the native GPU-local pool and allocation flags that implement
// uncached device access for |physical_device_ordinal|. Returns false when the
// native runtime exposes no such route. The returned pool is borrowed from the
// allocator and remains valid for the allocator's lifetime.
bool iree_hal_amdgpu_allocator_query_uncached_memory_pool(
    iree_hal_allocator_t* allocator, uint32_t physical_device_ordinal,
    hsa_amd_memory_pool_t* out_memory_pool, uint32_t* out_allocation_flags);

// Allocates one AMDGPU buffer using exact backend-private native policy.
//
// This is used for driver-owned resources and AMDGPU implementation tests.
// Public allocation owners construct a slab pool and issue geometry-only
// reservations through it.
iree_status_t iree_hal_amdgpu_allocator_allocate_direct_buffer(
    iree_hal_allocator_t* allocator, iree_hal_amdgpu_buffer_spec_t spec,
    iree_device_size_t allocation_size, iree_hal_buffer_t** out_buffer);

// Allocates private reusable-command-buffer kernarg template storage for one
// physical device. The returned buffer is device-local fine-grained memory
// with direct coherent host access and queue-family reach restricted to the
// selected physical device.
iree_status_t iree_hal_amdgpu_allocator_allocate_prepublished_kernarg_buffer(
    iree_hal_allocator_t* allocator, uint32_t physical_device_ordinal,
    iree_device_size_t allocation_size, iree_hal_buffer_t** out_buffer);

// Attaches typed external storage as one complete AMDGPU root buffer.
//
// The returned buffer owns only the native ROCr attachment and never consumes
// or releases |storage|'s transport owner. |storage| must remain alive until
// the returned buffer is released. Explicit pool providers use this primitive
// to preserve their separate transport and native attachment lifetimes.
// |access_agents| is the exact group-qualified native access set and is copied
// by the synchronous attachment calls.
iree_status_t iree_hal_amdgpu_allocator_import_direct_buffer(
    iree_hal_allocator_t* allocator, const iree_hal_amdgpu_buffer_spec_t* spec,
    const iree_hal_external_storage_t* storage,
    iree_host_size_t access_agent_count, const hsa_agent_t* access_agents,
    iree_hal_buffer_t** out_buffer);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVER_AMDGPU_ALLOCATOR_H_
