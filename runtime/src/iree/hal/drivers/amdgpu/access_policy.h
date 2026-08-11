// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_ACCESS_POLICY_H_
#define IREE_HAL_DRIVERS_AMDGPU_ACCESS_POLICY_H_

#include "iree/base/api.h"
#include "iree/hal/allocator.h"
#include "iree/hal/drivers/amdgpu/queue_affinity.h"
#include "iree/hal/drivers/amdgpu/util/libhsa.h"
#include "iree/hal/drivers/amdgpu/util/topology.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_access_agent_list_t
//===----------------------------------------------------------------------===//

// Fixed-capacity list of HSA agents that should be granted access to memory.
//
// HSA access calls are cold allocation/pinning paths. Keeping this list in
// caller storage avoids heap allocations while still making access policy an
// explicit, inspectable decision instead of ad-hoc all-agent grants.
typedef struct iree_hal_amdgpu_access_agent_list_t {
  // Number of initialized entries in |values|.
  uint32_t count;

  // HSA agents to pass to hsa_amd_agents_allow_access or hsa_amd_memory_lock.
  hsa_agent_t
      values[IREE_HAL_AMDGPU_MAX_CPU_AGENT + IREE_HAL_AMDGPU_MAX_GPU_AGENT];
} iree_hal_amdgpu_access_agent_list_t;

// Resolves the HSA agents that may access memory placed for |queue_affinity|.
//
// |access_scope| selects whether the result contains each selected GPU agent,
// its nearest CPU agent, or both.
// IREE_HAL_QUEUE_AFFINITY_ANY selects the entire logical device topology,
// while physical-device-local affinities stay scoped to that physical device.
iree_status_t iree_hal_amdgpu_access_agent_list_resolve(
    const iree_hal_amdgpu_topology_t* topology,
    iree_hal_amdgpu_queue_affinity_domain_t queue_affinity_domain,
    iree_hal_queue_affinity_t queue_affinity,
    iree_hal_virtual_memory_access_scope_t access_scope,
    iree_hal_amdgpu_access_agent_list_t* out_agent_list);

// Grants |agent_list| access to an HSA memory pool allocation.
iree_status_t iree_hal_amdgpu_access_allow_agent_list(
    const iree_hal_amdgpu_libhsa_t* libhsa,
    const iree_hal_amdgpu_access_agent_list_t* agent_list, const void* ptr);

// Pins |host_ptr| for the HSA agents in |agent_list| using the access
// properties of the CPU-owned |memory_pool|.
iree_status_t iree_hal_amdgpu_access_lock_host_allocation_to_pool(
    const iree_hal_amdgpu_libhsa_t* libhsa,
    const iree_hal_amdgpu_access_agent_list_t* agent_list,
    hsa_amd_memory_pool_t memory_pool, void* host_ptr,
    iree_device_size_t length, void** out_agent_ptr);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_ACCESS_POLICY_H_
