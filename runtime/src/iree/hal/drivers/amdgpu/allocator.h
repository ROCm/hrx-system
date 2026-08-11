// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_ALLOCATOR_H_
#define IREE_HAL_DRIVERS_AMDGPU_ALLOCATOR_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"

typedef struct iree_hal_amdgpu_libhsa_t iree_hal_amdgpu_libhsa_t;
typedef struct iree_hal_amdgpu_logical_device_t
    iree_hal_amdgpu_logical_device_t;
typedef struct iree_hal_amdgpu_topology_t iree_hal_amdgpu_topology_t;

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_allocator_t
//===----------------------------------------------------------------------===//

// Creates a buffer allocator that allocates from HSA memory pools.
//
// Normal allocations are direct HSA memory-pool allocations. When the logical
// device has HAL ASAN enabled, device-local persistent allocations route
// through the device default pool set so redzones, shadow publication, release
// poisoning, and quarantine follow the same policy as queue allocations.
//
// |logical_device| is unretained and must outlive the allocator.
iree_status_t iree_hal_amdgpu_allocator_create(
    iree_hal_amdgpu_logical_device_t* logical_device,
    const iree_hal_amdgpu_libhsa_t* libhsa,
    const iree_hal_amdgpu_topology_t* topology, iree_allocator_t host_allocator,
    iree_hal_allocator_t** out_allocator);

// Bitfield selecting AMDGPU agent classes updated by virtual-memory
// protection operations.
typedef uint32_t iree_hal_amdgpu_memory_agent_classes_t;
enum iree_hal_amdgpu_memory_agent_class_bits_e {
  IREE_HAL_AMDGPU_MEMORY_AGENT_CLASS_NONE = 0u,
  IREE_HAL_AMDGPU_MEMORY_AGENT_CLASS_HOST = 1u << 0,
  IREE_HAL_AMDGPU_MEMORY_AGENT_CLASS_DEVICE = 1u << 1,
  IREE_HAL_AMDGPU_MEMORY_AGENT_CLASS_ALL =
      IREE_HAL_AMDGPU_MEMORY_AGENT_CLASS_HOST |
      IREE_HAL_AMDGPU_MEMORY_AGENT_CLASS_DEVICE,
};

// Sets virtual-memory permissions for only the selected AMDGPU agent classes.
// The reservation may originate from another AMDGPU allocator in the same
// process because ROCr virtual addresses are process-global.
IREE_API_EXPORT iree_status_t
iree_hal_amdgpu_allocator_virtual_memory_protect_agents(
    iree_hal_allocator_t* allocator, iree_hal_buffer_t* virtual_buffer,
    iree_device_size_t virtual_offset, iree_device_size_t size,
    iree_hal_queue_affinity_t queue_affinity,
    iree_hal_amdgpu_memory_agent_classes_t agent_classes,
    iree_hal_memory_protection_t protection);

#endif  // IREE_HAL_DRIVERS_AMDGPU_ALLOCATOR_H_
