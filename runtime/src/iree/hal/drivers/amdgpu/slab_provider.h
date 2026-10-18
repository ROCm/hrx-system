// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_SLAB_PROVIDER_H_
#define IREE_HAL_DRIVERS_AMDGPU_SLAB_PROVIDER_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/amdgpu/buffer.h"
#include "iree/hal/drivers/amdgpu/util/libhsa.h"
#include "iree/hal/drivers/amdgpu/util/vmem.h"
#include "iree/hal/memory/slab_provider.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_hal_amdgpu_asan_state_t iree_hal_amdgpu_asan_state_t;

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_slab_provider_t
//===----------------------------------------------------------------------===//

// HSA memory-pool properties needed to configure slab-backed HAL pools.
typedef struct iree_hal_amdgpu_slab_provider_memory_pool_properties_t {
  // Smallest allocation-size multiple accepted by hsa_amd_memory_pool_allocate.
  iree_device_size_t allocation_granule;

  // Base-pointer alignment guaranteed by hsa_amd_memory_pool_allocate.
  iree_device_size_t allocation_alignment;

  // HAL memory type bits provided by this HSA memory pool.
  iree_hal_memory_type_t memory_type;

  // HAL buffer usage bits supported by buffers materialized from this pool.
  iree_hal_buffer_usage_t supported_usage;
} iree_hal_amdgpu_slab_provider_memory_pool_properties_t;

typedef uint32_t iree_hal_amdgpu_slab_provider_flags_t;
enum iree_hal_amdgpu_slab_provider_flag_bits_e {
  IREE_HAL_AMDGPU_SLAB_PROVIDER_FLAG_NONE = 0u,
  // Slabs precommit ASAN shadow at acquisition time and publish precise
  // suballocation shadow state through advise_asan_range().
  IREE_HAL_AMDGPU_SLAB_PROVIDER_FLAG_ASAN_SHADOW = 1u << 0,
  // Slabs are acquired with HSA VMM and mapped into the ASAN-owned application
  // virtual address window.
  IREE_HAL_AMDGPU_SLAB_PROVIDER_FLAG_ASAN_VMM = 1u << 1,
  // Slabs own ordinary VMM addresses and expose their complete physical handle
  // for explicit aliases. This route is independent of ASAN's address window.
  IREE_HAL_AMDGPU_SLAB_PROVIDER_FLAG_MAPPING_SOURCE = 1u << 2,
};

typedef struct iree_hal_amdgpu_allocation_queue_t
    iree_hal_amdgpu_allocation_queue_t;

// HSA memory pool, mapping policy, and HAL capabilities exposed by a slab
// provider.
typedef struct iree_hal_amdgpu_slab_provider_options_t {
  // Native owner retained by this provider, or NULL for explicit inline use.
  iree_hal_amdgpu_allocation_queue_t* allocation_queue;

  // Flags selecting provider behavior.
  iree_hal_amdgpu_slab_provider_flags_t flags;

  // HSA memory pool used for slab allocations.
  hsa_amd_memory_pool_t memory_pool;

  // HSA VMM memory type used when ASAN VMM placement is enabled.
  iree_hal_amdgpu_vmem_memory_type_t vmem_memory_type;

  // ASAN state used when ASAN shadow support is enabled.
  iree_hal_amdgpu_asan_state_t* asan_state;

  // Facts for slabs allocated from |memory_pool|.
  struct {
    // HAL memory type bits of provider-owned HSA allocations.
    iree_hal_memory_type_t memory_type;
    // HAL buffer usage bits supported by provider-owned HSA allocations.
    iree_hal_buffer_usage_t supported_usage;
    // Flags passed to every HSA memory-pool allocation.
    uint32_t allocation_flags;
  } owned;

  // Common facts guaranteed by every owned or imported slab route.
  struct {
    // HAL memory type bits common to every admitted route.
    iree_hal_memory_type_t memory_type;
    // HAL buffer usage bits common to every admitted route.
    iree_hal_buffer_usage_t supported_usage;
    // Atomic operations common to every admitted route.
    iree_hal_atomic_operation_capabilities_t atomic_operations;
    // Guaranteed base alignment common to every admitted route. Zero inherits
    // the alignment of provider-owned HSA allocations.
    iree_device_size_t allocation_alignment;
  } exposed;

  // Qualified native access captured before provider construction.
  struct {
    // Device-local family mask used by legacy buffer placement metadata.
    iree_hal_queue_family_affinity_t queue_family_affinity;
    // Number of unique native agents granted access to every slab.
    uint32_t agent_count;
    // Borrowed during creation; the provider copies exactly |agent_count|.
    const hsa_agent_t* agents;
    // Per-family atomic capabilities already qualified for this memory pool.
    iree_hal_amdgpu_atomic_memory_source_masks_t atomic_source_masks;
  } access;
} iree_hal_amdgpu_slab_provider_options_t;

// Queries HSA memory-pool properties used by AMDGPU slab providers and pools.
iree_status_t iree_hal_amdgpu_slab_provider_query_memory_pool_properties(
    const iree_hal_amdgpu_libhsa_t* libhsa, hsa_amd_memory_pool_t memory_pool,
    iree_hal_amdgpu_slab_provider_memory_pool_properties_t* out_properties);

// Creates a slab provider backed by an HSA memory pool on one GPU agent.
//
// The provider acquires whole slabs with hsa_amd_memory_pool_allocate() by
// default. When IREE_HAL_AMDGPU_SLAB_PROVIDER_FLAG_ASAN_SHADOW is set, the
// provider commits ASAN shadow coverage for each whole slab before returning it
// and later publishes suballocation poison state through advise_asan_range().
// When IREE_HAL_AMDGPU_SLAB_PROVIDER_FLAG_ASAN_VMM is also set, slabs are HSA
// VMM mappings inside |options.asan_state|'s application window instead of
// ordinary HSA memory-pool allocations. In all allocation modes the provider
// grants every captured |options.access| agent access and wraps slab slices as
// iree_hal_amdgpu_buffer_t views. Materialized buffers preserve the requested
// queue-family accessibility independently of the physical GPU memory pool
// owning the slab. View wrappers are allocated from |buffer_pool|, which must
// be in the same physical-device lifetime domain as the backing HSA memory.
// No topology is retained or queried during acquisition.
// |import_allocator| is an optional borrowed native attachment service. It is
// required only when the provider admits external storage. The provider
// borrows |device|, |import_allocator|, |libhsa|, |options.asan_state|, and
// |buffer_pool|; the owning physical/logical device must outlive the provider
// and every pool/buffer created from it.
iree_status_t iree_hal_amdgpu_slab_provider_create(
    iree_hal_device_t* device, iree_hal_allocator_t* import_allocator,
    const iree_hal_amdgpu_libhsa_t* libhsa,
    iree_hal_amdgpu_slab_provider_options_t options,
    iree_host_size_t physical_device_ordinal,
    iree_hal_amdgpu_buffer_pool_t* buffer_pool, iree_string_view_t trace_name,
    iree_allocator_t host_allocator, iree_hal_slab_provider_t** out_provider);

// Executes one native slab batch on the owning allocation worker.
void iree_hal_amdgpu_slab_provider_execute(
    iree_hal_slab_operation_t* operation);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_SLAB_PROVIDER_H_
