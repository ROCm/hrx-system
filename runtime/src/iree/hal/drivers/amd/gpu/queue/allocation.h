// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMD_GPU_QUEUE_ALLOCATION_H_
#define IREE_HAL_DRIVERS_AMD_GPU_QUEUE_ALLOCATION_H_

#include "iree/base/internal/arena.h"
#include "iree/base/internal/atomic_slist.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/amd/gpu/publication.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Device-owned pool transaction metadata. Queue publishers own dependency
// admission, native acceptance, and retirement. Native backing work belongs to
// the pool's allocation domain; only deferred backing needs this result source.
// The application keeps the device alive through operations and transient
// roots.
typedef struct iree_hal_amd_gpu_allocation_context_t {
  // Borrowed parent, dominated by the application's device ownership.
  iree_hal_device_t* device;
  // Borrowed device publisher, used only after a deferred pool acquisition.
  iree_hal_amd_gpu_publication_t* publication;
  // Fixed source ordinal for typed acquisition results in publication.
  uint32_t publication_ordinal;
  // Native callbacks publish only completed acquisition descriptors here.
  iree_atomic_slist_t ready;
  // Sole publisher's detached batch, consumed in bounded turns.
  iree_atomic_slist_entry_t* ready_head;
  // Reusable CPU storage for transactions, pool waits, and transient roots.
  iree_arena_block_pool_t blocks;
} iree_hal_amd_gpu_allocation_context_t;

// Initializes embedded state without creating a worker or native allocation.
void iree_hal_amd_gpu_allocation_context_initialize(
    iree_hal_device_t* device, iree_hal_amd_gpu_publication_t* publication,
    uint32_t publication_ordinal, iree_allocator_t host_allocator,
    iree_hal_amd_gpu_allocation_context_t* out_context);

// Releases idle metadata after every operation, root, and publication joins.
void iree_hal_amd_gpu_allocation_context_deinitialize(
    iree_hal_amd_gpu_allocation_context_t* context);

// Fixed device source: consumes completed acquisition results and resumes the
// exact queue's publication owner. It never blocks for backing or reuse.
iree_hal_amd_gpu_publication_source_t
iree_hal_amd_gpu_allocation_context_publication_source(
    iree_hal_amd_gpu_allocation_context_t* context);

// One captured pool transaction, owned by a native queue submission through
// acceptance and terminal retirement (or rejection before acceptance).
typedef struct iree_hal_amd_gpu_allocation_t iree_hal_amd_gpu_allocation_t;

// Captures requested extents and returns uncommitted roots. The public queue
// boundary validates family grants and request shape; this boundary checks pool
// limits. Failure leaves buffer outputs untouched and out_operation NULL.
// Success transfers one root reference per request and the transaction owner.
iree_status_t iree_hal_amd_gpu_allocation_capture_alloca(
    iree_hal_amd_gpu_allocation_context_t* context, iree_hal_pool_t* pool,
    iree_host_size_t count, const iree_hal_pool_reservation_request_t* requests,
    iree_hal_buffer_t** out_buffers,
    iree_hal_amd_gpu_allocation_t** out_operation);

// Exclusively captures deallocation roots from one source pool. Roots may be
// uncommitted until queue dependency admission. Rejection rolls back every
// captured deallocation mark without changing the backing or reservations.
iree_status_t iree_hal_amd_gpu_allocation_capture_dealloca(
    iree_hal_amd_gpu_allocation_context_t* context, iree_host_size_t count,
    iree_hal_buffer_t* const* buffers,
    iree_hal_amd_gpu_allocation_t** out_operation);

// Host sanitizer transitions require completed accesses, not an enqueued wait.
// Unguarded transactions use ordinary native queue admission.
bool iree_hal_amd_gpu_allocation_requires_host_wait(
    const iree_hal_amd_gpu_allocation_t* operation);

// Prepares backing after input dependency admission, using the exact queue's
// prospective causal prefix. Copies the dense native-domain epochs into owned
// canonical history before any asynchronous acquisition can retain it. axis is
// the queue's canonical group coordinate, or zero without a group tracker.
//
// Warm preparation completes inline with out_pending false and no callback.
// Otherwise out_pending is true and ready receives the terminal status exactly
// once, possibly before this call returns. The queue joins that callback's tail
// independently from native retirement. A prepared operation may be retried
// after native capacity deferral without repeating acquisition or staging.
iree_status_t iree_hal_amd_gpu_allocation_prepare(
    iree_hal_amd_gpu_allocation_t* operation, iree_async_axis_t axis,
    uint32_t queue_count, const uint64_t* frontier,
    void (*ready)(void* user_data, iree_status_t status), void* user_data,
    bool* out_pending);

// Commits prepared roots or returns reservations after irrevocable native
// acceptance and before publishing native output dependencies. A deallocation
// records the accepted queue death point; its old backing remains committed
// until destroy at retirement. No allocation or fallible work occurs here.
void iree_hal_amd_gpu_allocation_accept(
    iree_hal_amd_gpu_allocation_t* operation,
    const iree_async_frontier_t* death_frontier);

// Releases the transaction before publishing terminal completion. Accepted
// returns decommit their old backing; rejected transactions roll back unused
// reservations or deallocation marks. Preparation callbacks must have joined.
void iree_hal_amd_gpu_allocation_destroy(
    iree_hal_amd_gpu_allocation_t* operation);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMD_GPU_QUEUE_ALLOCATION_H_
