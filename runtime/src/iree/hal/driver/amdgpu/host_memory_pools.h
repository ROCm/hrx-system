// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVER_AMDGPU_HOST_MEMORY_POOLS_H_
#define IREE_HAL_DRIVER_AMDGPU_HOST_MEMORY_POOLS_H_

#include "iree/base/api.h"
#include "iree/hal/driver/amdgpu/util/libhsa.h"

#ifdef __cplusplus
extern "C" {
#endif

// A set of queried HSA regions and memory pools for a particular CPU agent.
// In a NUMA topology each CPU agent has memory that is faster to access and
// GPU agents attached to a particular CPU agent will require fewer hops to
// access the CPU memory. We try to allocate resources that may be used
// frequently on a particular cluster of CPU/GPU agents closest to the agents.
typedef struct iree_hal_amdgpu_host_memory_pools_t {
  // Coarse-grained shared host memory pool used for write-once/read-many data.
  hsa_amd_memory_pool_t coarse_pool;
  // Shared host memory pool suitable for kernel argument storage.
  // Allocations from this pool must have
  // HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT so dispatch packets can use
  // them as kernarg segment storage.
  hsa_amd_memory_pool_t kernarg_pool;
  // Memory pool used for various system-level resources.
  // Allocations from this pool must be accessible to all agents. Must have the
  // HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED flag as memory within this
  // pool will be used cross-agent with atomics. This is likely to be located
  // in host memory and we take the hit for device accesses for the ability to
  // share access with the host. For resources we expect to only be accessed on
  // a particular agent we instead allocate from there.
  hsa_amd_memory_pool_t fine_pool;
  // Optional extended fine-grained system pool with an uncached device mapping.
  // This is the native backing for concurrent host/device control data when
  // ordinary fine-grained memory does not bypass the device cache hierarchy.
  hsa_amd_memory_pool_t uncached_pool;
  // Memory region used for various system-level resources.
  // Allocations from this pool must be accessible to all agents.
  // The more modern memory pool API and `fine_pool` should be used
  // unless the particular HSA API requires a region.
  hsa_region_t fine_region;
} iree_hal_amdgpu_host_memory_pools_t;

// Selects allocatable host pools and a legacy fine-grained region for an agent.
// Pool selection requires global, accessible-by-all memory, prefers a
// non-kernarg ordinary fine pool, and records an extended fine-grained pool
// when present. Region enumeration may include every CPU NUMA node.
// Handles are borrowed from libhsa and remain valid for that runtime's
// lifetime. On failure out_pools is zeroed and the native or missing-capability
// error is propagated.
iree_status_t iree_hal_amdgpu_host_memory_pools_query(
    const iree_hal_amdgpu_libhsa_t* libhsa, hsa_agent_t host_agent,
    iree_hal_amdgpu_host_memory_pools_t* out_pools);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_HAL_DRIVER_AMDGPU_HOST_MEMORY_POOLS_H_
