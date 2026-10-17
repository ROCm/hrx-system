// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMD_GPU_QUEUE_MEMORY_MAPPING_H_
#define IREE_HAL_DRIVERS_AMD_GPU_QUEUE_MEMORY_MAPPING_H_

#include "iree/base/internal/arena.h"
#include "iree/base/internal/atomic_slist.h"
#include "iree/hal/drivers/amd/gpu/context.h"
#include "iree/hal/drivers/amd/gpu/profile.h"
#include "iree/hal/drivers/amd/gpu/publication.h"
#include "iree/hal/drivers/amd/gpu/queue/host_wait.h"

#ifdef __cplusplus
extern "C" {
#endif

// Device-owned mapping captures and completed native results. Dependencies
// register inline; only a ready typed mapping enters the native allocation
// domain. That domain publishes a result here without signaling dependent work.
// No worker is created by this component.
typedef struct iree_hal_amd_gpu_memory_mapping_context_t {
  // Borrowed device, retained by every accepted mapping capture.
  iree_hal_device_t* device;
  // Borrowed native domain; teardown joins all mapping callbacks.
  iree_hal_amd_gpu_context_t* context;
  // Borrowed device publisher, independent of blocking native memory calls.
  iree_hal_amd_gpu_publication_t* publication;
  // Fixed source index for completed mapping results.
  uint32_t publication_ordinal;
  // Completed native transactions, pushed without executing their consumers.
  iree_atomic_slist_t ready;
  // Sole publisher's detached prefix, processed in bounded turns.
  iree_atomic_slist_entry_t* ready_head;
  // Reusable CPU storage for captures, mapping vectors, and waits.
  iree_arena_block_pool_t blocks;
  // Dependency associations, joined before their captures are reclaimed.
  iree_async_semaphore_wait_context_t waits;
  // Optional session, changed only across idle profiling boundaries.
  iree_hal_amd_gpu_profile_t* profile;
  // Optional terminal device status, borrowed through device ownership.
  const iree_atomic_int32_t* failure;
} iree_hal_amd_gpu_memory_mapping_context_t;

void iree_hal_amd_gpu_memory_mapping_context_initialize(
    iree_hal_device_t* device, iree_hal_amd_gpu_context_t* context,
    iree_hal_amd_gpu_publication_t* publication, uint32_t publication_ordinal,
    iree_allocator_t host_allocator,
    iree_hal_amd_gpu_memory_mapping_context_t* out_context);

// All accepted operations, native callbacks, and publications have joined.
void iree_hal_amd_gpu_memory_mapping_context_deinitialize(
    iree_hal_amd_gpu_memory_mapping_context_t* context);

// Publishes only completed native results and their public HAL signals.
iree_hal_amd_gpu_publication_source_t
iree_hal_amd_gpu_memory_mapping_context_publication_source(
    iree_hal_amd_gpu_memory_mapping_context_t* context);

// Captures logical mappings before acceptance, then resolves after
// prerequisites complete. Native failures fail public signals; no success is
// published before native translation visibility. Captures retain views through
// publication but never extend an explicitly ended allocation epoch or an
// installed alias.
iree_status_t iree_hal_amd_gpu_queue_map(
    iree_hal_amd_gpu_memory_mapping_context_t* context, iree_hal_queue_t* queue,
    uint32_t queue_ordinal, iree_hal_semaphore_list_t waits,
    iree_hal_semaphore_list_t signals, iree_host_size_t mapping_count,
    const iree_hal_memory_mapping_t* mappings,
    iree_hal_queue_map_flags_t flags);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_HAL_DRIVERS_AMD_GPU_QUEUE_MEMORY_MAPPING_H_
