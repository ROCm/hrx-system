// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVER_AMD_GPU_QUEUE_SDMA_QUEUE_H_
#define IREE_HAL_DRIVER_AMD_GPU_QUEUE_SDMA_QUEUE_H_

#include "iree/hal/driver/amd/gpu/queue/allocation.h"
#include "iree/hal/driver/amd/gpu/queue/file_staging.h"
#include "iree/hal/driver/amd/gpu/queue/membership.h"
#include "iree/hal/driver/amd/gpu/queue/memory_mapping.h"
#include "iree/hal/driver/amd/gpu/queue/sdma/barrier.h"

#ifdef __cplusplus
extern "C" {
#endif

// Native dependency representation shared with this device's compute queues.
// SDMA accesses the value inside the native operand; AQL consumers require the
// descriptor address while PM4 consumers use the value address itself.
typedef struct iree_hal_amd_gpu_sdma_signal_layout_t {
  // Bytes per native operand, including any firmware descriptor.
  uint32_t stride;
  // Value displacement from a signal's device_address, in bytes.
  uint32_t value_offset;
  // Coherent binary value before completion; the complete value is its inverse.
  uint32_t pending_value;
  // Attaches acquired native backing to the existing independent-reader pool.
  iree_status_t (*initialize)(iree_hal_amd_gpu_context_t* context,
                              iree_hal_amd_gpu_signal_pool_t* pool,
                              iree_hal_amd_gpu_signal_storage_t* storage);
} iree_hal_amd_gpu_sdma_signal_layout_t;

// Cold construction facts. Services and policies belong to the device and
// outlive the queue; no worker, public pool, or auxiliary compute queue is
// owned here. Membership assigns stable causal coordinates independently of
// families.
typedef struct iree_hal_amd_gpu_sdma_queue_options_t {
  // Selected native transfer family ordinal.
  uint32_t family_ordinal;
  // Stable native-domain causal coordinate.
  uint32_t queue_ordinal;
  // Device publication/completion source ordinal.
  uint32_t source_ordinal;
  // Installed extent of every native-domain causal vector.
  uint32_t queue_count;
  // Completed epoch from the prior owner of this coordinate.
  uint64_t initial_epoch;
  // Exactly one qualified USER or KERNEL carrier.
  amdf_queue_publication_modes_t publication_mode;
  // Immutable native cache and execution boundary encoding.
  iree_hal_amd_gpu_sdma_barrier_profile_t barriers;
  // Qualified binary-signal layout, independent of SDMA packet format.
  iree_hal_amd_gpu_sdma_signal_layout_t signal_layout;
  // Device-owned native cells, retained beyond this queue's last reader.
  iree_hal_amd_gpu_signal_cache_t* signals;
  // Coherent host/device control and staging placement.
  const iree_hal_amd_fabric_memory_options_t* memory;
  // Existing device publisher; allocation callbacks only enqueue work here.
  iree_hal_amd_gpu_publication_t* publication;
  // Shared pool transaction owner.
  iree_hal_amd_gpu_allocation_context_t* allocations;
  // Shared mapping admission and result owner.
  iree_hal_amd_gpu_memory_mapping_context_t* mappings;
  // Family-owned file staging and registration owner.
  iree_hal_amd_gpu_file_staging_t* file_staging;
  // Bounded host transfer window per native span, in bytes, at least 64.
  uint32_t staging_capacity;
  // Reusable CPU capture block size, independent of payload staging.
  iree_host_size_t capture_block_size;
  // Number of CPU capture blocks to preallocate during construction.
  iree_host_size_t capture_block_count;
} iree_hal_amd_gpu_sdma_queue_options_t;

// Creates one explicit TRANSFER queue. The parent installs both worker sources
// and release ownership before exposing it. Construction performs cold native
// acquisition on this caller; all later growth uses existing shared services.
iree_status_t iree_hal_amd_gpu_sdma_queue_create(
    const iree_hal_queue_family_t* family, iree_hal_amd_gpu_context_t* context,
    const iree_hal_amd_gpu_sdma_queue_options_t* options,
    const iree_hal_queue_params_t* params, iree_allocator_t host_allocator,
    iree_hal_queue_t** out_queue);

// Cold device membership operations; public execution stays on the queue
// vtable.
extern const iree_hal_amd_gpu_queue_lifecycle_t
    iree_hal_amd_gpu_sdma_queue_lifecycle;

#ifdef __cplusplus
}  // extern "C"
#endif
#endif  // IREE_HAL_DRIVER_AMD_GPU_QUEUE_SDMA_QUEUE_H_
