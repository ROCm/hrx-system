// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMD_GPU_QUEUE_BACKING_H_
#define IREE_HAL_DRIVERS_AMD_GPU_QUEUE_BACKING_H_

#include "iree/base/threading/notification.h"
#include "iree/hal/drivers/amd/fabric/memory.h"
#include "iree/hal/drivers/amd/gpu/context.h"
#include "iree/hal/drivers/amd/gpu/publication.h"
#include "iree/hal/drivers/amd/gpu/queue/signal.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Argument generations are FIFO by publication, including alignment and wrap.
// A replaced generation is released only after its final byte position has
// retired. Native readers of signals have a separate lifetime below.
typedef struct iree_hal_amd_gpu_argument_storage_t {
  // Older generation, or NULL at the end of the cold cleanup list.
  struct iree_hal_amd_gpu_argument_storage_t* next;
  // Native storage and persistent host view.
  iree_hal_amd_fabric_memory_t memory;
  // Address of byte zero in this queue's native domain.
  uint64_t address;
  // Monotonic first position belonging to this generation.
  uint64_t begin;
  // Exclusive final reserved position in this generation.
  uint64_t end;
  // Power-of-two usable byte capacity.
  uint64_t capacity;
  // Release descriptor acquired with this generation, never at retirement.
  iree_hal_amd_gpu_memory_release_t release;
  // Host allocator owning this metadata through native release completion.
  iree_allocator_t host_allocator;
} iree_hal_amd_gpu_argument_storage_t;

typedef struct iree_hal_amd_gpu_signal_storage_t {
  // Next backing, retained while active or referenced by native readers.
  struct iree_hal_amd_gpu_signal_storage_t* next;
  // Native cells and coherent CPU mapping.
  iree_hal_amd_fabric_memory_t memory;
  // Finite extent attached to the publisher's available signal pool.
  iree_hal_amd_gpu_signal_block_t block;
  // Number of native cells and corresponding host signal records.
  uint32_t capacity;
  // Native-domain frontier width for each signal record.
  uint32_t queue_count;
  // Prepared inline metadata, initialized after native backing arrives.
  iree_hal_amd_gpu_signal_t* signals;
  // Initial inline frontier table header and rows. Expanded tables stay linked
  // to this initial allocation until all block readers have joined.
  iree_hal_amd_gpu_signal_frontiers_t* frontiers;
  // Release descriptor acquired with this storage, never at retirement.
  iree_hal_amd_gpu_memory_release_t release;
  // Host allocator owning this metadata through native release completion.
  iree_allocator_t host_allocator;
} iree_hal_amd_gpu_signal_storage_t;

// Device-owned native signal capacity borrowed by one queue publisher. Native
// readers may outlive that queue; neither their cells nor their return wake
// target belongs to the queue body. The device joins readers before final
// deinitialization. Acquisition, attachment, and trim remain publisher-owned.
typedef struct iree_hal_amd_gpu_signal_cache_t {
  // Available cells and independently retained native readers.
  iree_hal_amd_gpu_signal_pool_t pool;
  // Native blocks, including blocks still borrowed by other queues.
  iree_hal_amd_gpu_signal_storage_t* storage;
  // Stable device publication owner, live through all return callbacks.
  iree_hal_amd_gpu_publication_t* publication;
  // Stable source ordinal; a vacant source discards stale readiness hints.
  uint32_t source_ordinal;
  // One while the current queue has enrolled demand for returned capacity.
  iree_atomic_int32_t waiting;
  // Coalesced return hint, consumed only by the source's deferred publisher.
  iree_atomic_int32_t ready;
} iree_hal_amd_gpu_signal_cache_t;

// Initializes host bookkeeping without allocating native storage. The device
// supplies the stable wake target; no return callback borrows a queue body.
void iree_hal_amd_gpu_signal_cache_initialize(
    iree_hal_amd_gpu_publication_t* publication, uint32_t source_ordinal,
    iree_hal_amd_gpu_signal_cache_t* out_cache);

// Expands every attached block under exclusive publication ownership. The
// publisher vector must already cover queue_count; a partial failure can leave
// wider zero-filled tables while the cold owner rejects the new queue. Native
// addresses and retained readers are unchanged.
iree_status_t iree_hal_amd_gpu_signal_cache_reserve_frontiers(
    iree_hal_amd_gpu_signal_cache_t* cache, uint32_t queue_count,
    iree_allocator_t host_allocator);

// Cold queue-membership request embedded in a protocol's typed FIFO entry.
// The caller owns this stack storage through await. Its notification belongs
// to the device and outlives the callback tail after completed is published.
typedef struct iree_hal_amd_gpu_frontier_request_t {
  // Requested native coordinate extent; never shrinks installed metadata.
  uint32_t queue_count;
  // Borrowed device-owned notification, shared by serialized cold requests.
  iree_notification_t* notification;
  // Set after status and all metadata writes; the callback then stops using
  // this request and only posts the captured device notification.
  iree_atomic_int32_t completed;
  // Owned allocation result transferred to the awaiting cold caller.
  iree_status_t status;
} iree_hal_amd_gpu_frontier_request_t;

void iree_hal_amd_gpu_frontier_request_complete(
    iree_hal_amd_gpu_frontier_request_t* request, iree_status_t status);
iree_status_t iree_hal_amd_gpu_frontier_request_await(
    iree_hal_amd_gpu_frontier_request_t* request);

// Consumes an actual capacity-return hint before reserving publication. An
// enrolled but still empty cache does not request another worker turn.
static inline bool iree_hal_amd_gpu_signal_cache_take_ready(
    iree_hal_amd_gpu_signal_cache_t* cache) {
  return iree_atomic_load(&cache->ready, iree_memory_order_acquire) &&
         iree_atomic_exchange(&cache->ready, 0, iree_memory_order_acq_rel);
}

// Releases unused blocks while retaining at least minimum_block_count blocks.
// Called by the exclusive publisher, or the cold owner after joining it.
// Native readers keep their blocks attached; release uses embedded descriptors
// and never waits or performs native calls on this caller.
void iree_hal_amd_gpu_signal_cache_trim(iree_hal_amd_gpu_context_t* context,
                                        uint32_t minimum_block_count,
                                        iree_hal_amd_gpu_signal_cache_t* cache);

// The caller joins all readers before this call. Transfers native blocks to
// the context's release owner and deinitializes host bookkeeping. The context
// separately joins those native releases before its own destruction.
void iree_hal_amd_gpu_signal_cache_deinitialize(
    iree_hal_amd_gpu_context_t* context,
    iree_hal_amd_gpu_signal_cache_t* cache);

// Queries the GPU address of an acquired coherent write-back mapping. The
// admitted allocation policy already proves coherence; this validates the
// achieved host cacheability. A NULL output checks only that mapping property.
iree_status_t iree_hal_amd_gpu_backing_query_address(
    const iree_hal_amd_gpu_context_t* context,
    const iree_hal_amd_fabric_memory_t* memory, uint64_t* out_address);

// Allocates coherent write-back storage for commands, arguments, or control
// cells under the caller's admitted memory policy. A non-NULL out_address asks
// for its GPU address at access ordinal zero. Failure may leave partial memory;
// the caller always deinitializes it. Native allocation runs on this
// constructor's caller, never on publication or retirement.
iree_status_t iree_hal_amd_gpu_backing_allocate(
    iree_hal_amd_gpu_context_t* context,
    const iree_hal_amd_fabric_memory_options_t* options, uint64_t length,
    uint64_t alignment, iree_hal_amd_fabric_memory_t* memory,
    uint64_t* out_address);

// Allocates only host ownership and the native release descriptor. The caller
// acquires storage->memory asynchronously, then initializes its address and
// monotonic extent on the publication owner. Failure returns NULL storage.
iree_status_t iree_hal_amd_gpu_argument_storage_prepare(
    uint64_t capacity, iree_allocator_t host_allocator,
    iree_hal_amd_gpu_argument_storage_t** out_storage);
iree_status_t iree_hal_amd_gpu_argument_storage_initialize(
    iree_hal_amd_gpu_context_t* context, uint64_t begin,
    iree_hal_amd_gpu_argument_storage_t* storage);

// Allocates an independent generation at the admitted monotonic begin position.
// The caller links and retires generations by their final reserved byte extent.
// Success returns the unique owner; failure leaves out_storage NULL.
iree_status_t iree_hal_amd_gpu_argument_storage_allocate(
    iree_hal_amd_gpu_context_t* context,
    const iree_hal_amd_fabric_memory_options_t* options, uint64_t capacity,
    uint64_t begin, iree_allocator_t host_allocator,
    iree_hal_amd_gpu_argument_storage_t** out_storage);

// Transfers a retired or partially acquired generation to native release.
// List ownership and the exact consumed-byte proof stay with the queue. This
// performs no native call or wait, and storage cannot be accessed afterward.
void iree_hal_amd_gpu_argument_storage_free(
    iree_hal_amd_gpu_context_t* context,
    iree_hal_amd_gpu_argument_storage_t* storage);

// Prepares host metadata and release ownership without acquiring native cells.
// After typed acquisition fills storage->memory, the protocol initializes its
// native descriptors and attaches the block to its finite signal pool.
iree_status_t iree_hal_amd_gpu_signal_storage_prepare(
    uint32_t capacity, uint32_t queue_count, iree_allocator_t host_allocator,
    iree_hal_amd_gpu_signal_storage_t** out_storage);

// Transfers inactive backing after every cell reader has released its
// reference. Also accepts partially acquired storage. No native call or wait
// occurs here.
void iree_hal_amd_gpu_signal_storage_free(
    iree_hal_amd_gpu_context_t* context,
    iree_hal_amd_gpu_signal_storage_t* storage);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMD_GPU_QUEUE_BACKING_H_
