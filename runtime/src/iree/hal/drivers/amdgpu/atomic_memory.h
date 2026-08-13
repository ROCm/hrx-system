// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_ATOMIC_MEMORY_H_
#define IREE_HAL_DRIVERS_AMDGPU_ATOMIC_MEMORY_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/amdgpu/util/libhsa.h"
#include "iree/hal/drivers/amdgpu/util/topology.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Atomic memory width and coherence-domain cells supported by an allocation.
typedef uint32_t iree_hal_amdgpu_atomic_memory_cell_flags_t;
typedef enum iree_hal_amdgpu_atomic_memory_cell_flag_bits_e {
  IREE_HAL_AMDGPU_ATOMIC_MEMORY_CELL_FLAG_NONE = 0u,
  // 32-bit atomic operations in the device coherence domain.
  IREE_HAL_AMDGPU_ATOMIC_MEMORY_CELL_FLAG_DEVICE_SCOPE_32 = 1u << 0,
  // 64-bit atomic operations in the device coherence domain.
  IREE_HAL_AMDGPU_ATOMIC_MEMORY_CELL_FLAG_DEVICE_SCOPE_64 = 1u << 1,
  // 32-bit atomic operations in the system coherence domain.
  IREE_HAL_AMDGPU_ATOMIC_MEMORY_CELL_FLAG_SYSTEM_SCOPE_32 = 1u << 2,
  // 64-bit atomic operations in the system coherence domain.
  IREE_HAL_AMDGPU_ATOMIC_MEMORY_CELL_FLAG_SYSTEM_SCOPE_64 = 1u << 3,
} iree_hal_amdgpu_atomic_memory_cell_flag_bits_t;

#define IREE_HAL_AMDGPU_ATOMIC_MEMORY_CELL_FLAGS_ALL                                                      \
  ((iree_hal_amdgpu_atomic_memory_cell_flags_t)(IREE_HAL_AMDGPU_ATOMIC_MEMORY_CELL_FLAG_DEVICE_SCOPE_32 | \
                                                IREE_HAL_AMDGPU_ATOMIC_MEMORY_CELL_FLAG_DEVICE_SCOPE_64 | \
                                                IREE_HAL_AMDGPU_ATOMIC_MEMORY_CELL_FLAG_SYSTEM_SCOPE_32 | \
                                                IREE_HAL_AMDGPU_ATOMIC_MEMORY_CELL_FLAG_SYSTEM_SCOPE_64))

// Source GPU masks supporting each atomic memory cell in one HSA memory pool.
//
// A bit at physical device ordinal N means that GPU may issue the corresponding
// width and scope of atomic transaction against allocations from the pool.
typedef struct iree_hal_amdgpu_atomic_memory_source_masks_t {
  // GPUs supporting 32-bit device-scope atomics.
  iree_hal_amdgpu_gpu_agent_mask_t device_scope_32;
  // GPUs supporting 64-bit device-scope atomics.
  iree_hal_amdgpu_gpu_agent_mask_t device_scope_64;
  // GPUs supporting 32-bit system-scope atomics.
  iree_hal_amdgpu_gpu_agent_mask_t system_scope_32;
  // GPUs supporting 64-bit system-scope atomics.
  iree_hal_amdgpu_gpu_agent_mask_t system_scope_64;
} iree_hal_amdgpu_atomic_memory_source_masks_t;

// Already-queried HSA facts for one source GPU accessing one memory pool.
typedef struct iree_hal_amdgpu_atomic_memory_source_selection_t {
  // HSA memory-pool global flags describing granularity and scope.
  uint32_t global_flags;
  // Flags supplied when allocating from the HSA memory pool.
  uint32_t allocation_flags;
  // Source GPU access mode for the memory pool.
  hsa_amd_memory_pool_access_t access;
  // Number of entries in |link_hops|.
  iree_host_size_t link_hop_count;
  // HSA link records from the source GPU to the memory pool.
  const hsa_amd_memory_pool_link_info_t* link_hops;
} iree_hal_amdgpu_atomic_memory_source_selection_t;

// Selects the atomic memory cells supported by one source GPU.
//
// This validates HSA-reported access and pool-granularity facts at the external
// runtime boundary. A zero-hop path is local to the pool owner and supports
// both widths; peer paths intersect every HSA link hop.
iree_status_t iree_hal_amdgpu_atomic_memory_select_source_cells(
    const iree_hal_amdgpu_atomic_memory_source_selection_t* selection,
    iree_hal_amdgpu_atomic_memory_cell_flags_t* out_cell_flags);

// Queries immutable source GPU masks for allocations from |memory_pool|.
iree_status_t iree_hal_amdgpu_atomic_memory_query_source_masks(
    const iree_hal_amdgpu_libhsa_t* libhsa,
    const iree_hal_amdgpu_topology_t* topology,
    hsa_amd_memory_pool_t memory_pool, uint32_t allocation_flags,
    iree_hal_amdgpu_atomic_memory_source_masks_t* out_source_masks);

// Intersects |source_masks| across all physical devices in |device_mask|.
// Returns no cells when |device_mask| is empty.
iree_hal_amdgpu_atomic_memory_cell_flags_t
iree_hal_amdgpu_atomic_memory_select_device_cells(
    const iree_hal_amdgpu_atomic_memory_source_masks_t* source_masks,
    iree_hal_amdgpu_gpu_agent_mask_t device_mask);

// Expands compact memory cells into the public all-operations capability
// matrix.
iree_hal_atomic_operation_capabilities_t
iree_hal_amdgpu_atomic_memory_expand_capabilities(
    iree_hal_amdgpu_atomic_memory_cell_flags_t cell_flags);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_ATOMIC_MEMORY_H_
