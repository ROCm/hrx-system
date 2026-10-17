// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMD_GPU_QUEUE_ALLOCATION_H_
#define IREE_HAL_DRIVERS_AMD_GPU_QUEUE_ALLOCATION_H_

#include "iree/base/internal/arena.h"
#include "iree/base/internal/atomic_slist.h"
#include "iree/hal/drivers/amd/gpu/context.h"
#include "iree/hal/drivers/amd/gpu/profile.h"
#include "iree/hal/drivers/amd/gpu/publication.h"
#include "iree/hal/drivers/amd/gpu/queue/host_wait.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Device-owned queue allocation metadata and dependency registration. Native
// backing work belongs to the pool's native allocation domain. Warm metadata
// work completes inline; deferred acquisitions publish a result to the device.
// Accepted operations and returned transient roots retain the device, keeping
// this state alive independently of the caller's device reference.
typedef struct iree_hal_amd_gpu_allocation_context_t {
  // Borrowed parent, retained by individual operations and transient roots.
  iree_hal_device_t* device;
  // Borrowed diagnostic owner, dominated by device.
  const iree_hal_amd_gpu_context_t* context;
  // Borrowed device publisher, used only after a deferred pool acquisition.
  iree_hal_amd_gpu_publication_t* publication;
  // Fixed source ordinal for typed acquisition results in publication.
  uint32_t publication_ordinal;
  // Native callbacks publish only completed acquisition descriptors here.
  iree_atomic_slist_t ready;
  // Sole publisher's detached batch, consumed in bounded turns.
  iree_atomic_slist_entry_t* ready_head;
  // Reusable CPU storage for captures, wait records, and transient roots.
  iree_arena_block_pool_t blocks;
  // Dependency associations shared by this device's allocation operations.
  iree_async_semaphore_wait_context_t waits;
  // Optional device-owned session, stable while operations are in flight.
  iree_hal_amd_gpu_profile_t* profile;
  // Optional device failure code, borrowed through the retained device owner.
  const iree_atomic_int32_t* failure;
} iree_hal_amd_gpu_allocation_context_t;

// Initializes embedded state without creating a worker or native allocation.
void iree_hal_amd_gpu_allocation_context_initialize(
    iree_hal_device_t* device, const iree_hal_amd_gpu_context_t* context,
    iree_hal_amd_gpu_publication_t* publication, uint32_t publication_ordinal,
    iree_allocator_t host_allocator,
    iree_hal_amd_gpu_allocation_context_t* out_context);

// Releases idle metadata after every operation, root, and publication joins.
void iree_hal_amd_gpu_allocation_context_deinitialize(
    iree_hal_amd_gpu_allocation_context_t* context);

// Fixed device source: commits completed acquisitions and publishes their HAL
// signals. It never performs native allocation or waits for backing or reuse.
iree_hal_amd_gpu_publication_source_t
iree_hal_amd_gpu_allocation_context_publication_source(
    iree_hal_amd_gpu_allocation_context_t* context);

// Captures the complete transaction and returns uncommitted roots. The public
// queue boundary has validated lists, family grants, and request shape. Pool
// limits are admitted here. Failure leaves every output untouched; success
// transfers one root reference per request and completes through signals.
iree_status_t iree_hal_amd_gpu_queue_alloca(
    iree_hal_amd_gpu_allocation_context_t* context, uint32_t queue_ordinal,
    iree_hal_semaphore_list_t waits, iree_hal_semaphore_list_t signals,
    iree_hal_pool_t* pool, iree_host_size_t count,
    const iree_hal_pool_reservation_request_t* requests,
    iree_hal_buffer_t** out_buffers);

// Captures roots transactionally, then decommits and returns their reservations
// after host-completed dependencies. Roots may originate from different pools
// or cooperating devices in the same sealed group. Pool/root grants and root
// capability are already checked by the public queue boundary.
iree_status_t iree_hal_amd_gpu_queue_dealloca(
    iree_hal_amd_gpu_allocation_context_t* context, uint32_t queue_ordinal,
    iree_hal_semaphore_list_t waits, iree_hal_semaphore_list_t signals,
    iree_host_size_t count, iree_hal_buffer_t* const* buffers);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMD_GPU_QUEUE_ALLOCATION_H_
