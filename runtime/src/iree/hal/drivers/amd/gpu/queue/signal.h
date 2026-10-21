// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMD_GPU_QUEUE_SIGNAL_H_
#define IREE_HAL_DRIVERS_AMD_GPU_QUEUE_SIGNAL_H_

#include "iree/base/api.h"
#include "iree/base/internal/atomics.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_hal_amd_gpu_signal_pool_t iree_hal_amd_gpu_signal_pool_t;
typedef struct iree_hal_amd_gpu_signal_block_t iree_hal_amd_gpu_signal_block_t;

// A host table of signal-major native publication epochs. The capacity *
// queue_count uint64_t values immediately follow this header. Each published
// row is immutable until its last signal reader returns. Cold expansion copies
// rows under publisher ownership and retains previous tables for readers that
// already borrowed them; native signal addresses never change.
typedef struct iree_alignas(8) iree_hal_amd_gpu_signal_frontiers_t {
  // Previous narrower table, retained until all block readers have joined.
  struct iree_hal_amd_gpu_signal_frontiers_t* previous;
  // Number of signal rows, equal to the native block's capacity.
  uint32_t capacity;
  // Immutable coordinate extent of each row in this table.
  uint32_t queue_count;
} iree_hal_amd_gpu_signal_frontiers_t;

// Frees expanded tables preceding initial after every block reader has joined.
// The initial table belongs to the enclosing metadata allocation and remains
// untouched. NULL initial releases an entirely separately allocated chain.
void iree_hal_amd_gpu_signal_frontiers_free(
    iree_hal_amd_gpu_signal_frontiers_t* frontiers,
    iree_hal_amd_gpu_signal_frontiers_t* initial,
    iree_allocator_t host_allocator);

// One native binary signal, independently retained by its publisher,
// producer, caches, and dependent executions. A producer completing does not
// retire the native readers. This metadata stays in ordinary host memory;
// only the separately backed value is accessed by the GPU.
typedef struct iree_hal_amd_gpu_signal_t {
  // Exclusive free-list state or shared live-use state, never both.
  union {
    // Live ownership and producer retirement, sharing the free-link footprint.
    struct {
      // Independent uses; new references require an existing live borrow.
      iree_atomic_ref_count_t ref_count;
      // Zero until producer payloads and public completion have joined, then
      // one plus the terminal status code. Independent of the GPU-written cell.
      iree_atomic_int32_t retirement;
    } live;
    // Link owned by the final releaser, then by the pool's exclusive publisher.
    struct iree_hal_amd_gpu_signal_t* next;
  } state;
  // Borrowed block owning this slot, live until every reference is returned.
  iree_hal_amd_gpu_signal_block_t* block;
  // Native operand address: an eight-byte PM4 cell or 64-byte AQL descriptor.
  uint64_t device_address;
  // Borrowed host value, initialized only during exclusive acquisition. The
  // protocol supplies its initial value and completion predicate.
  uint64_t* host_value;
} iree_hal_amd_gpu_signal_t;

// Coalescible capacity hint, invoked after publishing returned capacity. The
// callback notifies the enclosing owner's existing progress mechanism; it
// does not wait or destroy the pool. Another acquisition may
// consume the available slot before the callback runs.
typedef struct iree_hal_amd_gpu_signal_pool_notification_t {
  // Optional callback when the shared return list becomes nonempty.
  void (*fn)(void* user_data);
  // Borrowed callback state, live through all external pool calls.
  void* user_data;
} iree_hal_amd_gpu_signal_pool_notification_t;

// Available cells across independently allocated blocks. The queue's exclusive
// publisher owns acquisition, block attachment/detachment, and capacity
// queries. Arbitrary readers may return cells concurrently. Acquisition pops
// one local cell, detaching a batch of returned cells only when local capacity
// is empty. Pending native allocation is independent of returned capacity. This
// object performs no allocation or native call and owns no worker.
typedef struct iree_hal_amd_gpu_signal_pool_t {
  // Blocks with at least one free cell, linked without an availability scan.
  iree_hal_amd_gpu_signal_block_t* available;
  // Returned cells detached by the publisher; still counted as live by blocks.
  iree_hal_amd_gpu_signal_t* recycled;
  // Number of attached blocks, including exhausted and zero-capacity blocks.
  uint32_t block_count;
  // Concurrent last releases push here. Separate from publisher-local metadata
  // so returning cells does not invalidate the ordinary acquisition cache line.
  iree_alignas(iree_hardware_destructive_interference_size)
      iree_atomic_intptr_t returned;
  // Immutable wake target, sharing the return line and copied before a final
  // release exposes its cell. Final releases never access publisher-local
  // state.
  iree_hal_amd_gpu_signal_pool_notification_t notification;
} iree_hal_amd_gpu_signal_pool_t;

// Finite bookkeeping borrowing one native extent and its host metadata. The
// enclosing owner joins every native writer, reader, and cache reference before
// detaching a block. The pool and notification target outlive those releases.
typedef struct iree_hal_amd_gpu_signal_block_t {
  // Borrowed pool, or NULL after successful detachment.
  iree_hal_amd_gpu_signal_pool_t* pool;
  // Checked-out or pending-return cells; modified only by the publisher.
  uint32_t live_count;
  // Available metadata slots, owned by the publisher.
  iree_hal_amd_gpu_signal_t* free_head;
  // Previous available block, or NULL at the list head or while exhausted.
  iree_hal_amd_gpu_signal_block_t* available_previous;
  // Next available block, or NULL at the list tail or while exhausted.
  iree_hal_amd_gpu_signal_block_t* available_next;
  // Borrowed qualified address/visibility/atomic/protocol domain. Matching
  // identities permit native waits on these operands with the same completion
  // predicate; NULL permits only host waits.
  const void* domain;
  // Borrowed metadata base for constant-time signal-to-frontier indexing.
  iree_hal_amd_gpu_signal_t* signals;
  // Current host table, release-published by cold expansion under exclusive
  // publication ownership. Independent readers acquire this pointer once.
  iree_atomic_intptr_t frontiers;
} iree_hal_amd_gpu_signal_block_t;

// Initializes an empty owner. Native blocks are attached only after their
// allocation completes. All blocks must be detached before deinitialization.
void iree_hal_amd_gpu_signal_pool_initialize(
    iree_hal_amd_gpu_signal_pool_notification_t notification,
    iree_hal_amd_gpu_signal_pool_t* out_pool);
void iree_hal_amd_gpu_signal_pool_deinitialize(
    iree_hal_amd_gpu_signal_pool_t* pool);

// Links capacity native entries and zeroes only their eight-byte value fields.
// Each native operand begins stride bytes after the previous; value_offset
// locates its naturally aligned value. Construction has qualified the complete
// extents, host/device visibility, atomicity, and native protocol. All other
// native fields are initialized by the protocol owner before publication.
// Metadata, storage, and frontier_storage do not overlap. Frontier storage is
// aligned to eight bytes and holds two uint64_t header words followed by
// capacity * queue_count values. This initializes the header and zeroes the
// rows; publication fills a complete vector before exposing a signal. The
// initial table may share the block's enclosing host allocation. A zero
// capacity attaches an empty block.
// The exclusive publisher attaches the block and already owns its waiting
// work; attachment does not invoke the notification.
void iree_hal_amd_gpu_signal_block_initialize(
    iree_hal_amd_gpu_signal_pool_t* pool, const void* domain, void* storage,
    uint64_t device_address, uint32_t stride, uint32_t value_offset,
    iree_hal_amd_gpu_signal_t* signals, uint32_t capacity, uint32_t queue_count,
    void* frontier_storage, iree_hal_amd_gpu_signal_block_t* out_block);

// Returns one table borrow covered by an existing signal reference, or by the
// block owner's exclusive publication/lifecycle ownership.
static inline iree_hal_amd_gpu_signal_frontiers_t*
iree_hal_amd_gpu_signal_block_frontiers(
    const iree_hal_amd_gpu_signal_block_t* block) {
  return (iree_hal_amd_gpu_signal_frontiers_t*)iree_atomic_load(
      &block->frontiers, iree_memory_order_acquire);
}

// Expands host proof metadata without replacing native cells or waiting for
// readers. Called only under exclusive publication ownership. New coordinates
// start at zero. Failure leaves the current table and every reader unchanged.
// The cold membership owner grows widths geometrically and finishes expanding
// all participating queues before publishing work on a new coordinate.
iree_status_t iree_hal_amd_gpu_signal_block_reserve_frontiers(
    iree_hal_amd_gpu_signal_block_t* block, uint32_t queue_count,
    iree_allocator_t host_allocator);

// Reconciles pending returns, then detaches an entirely unused block. False
// leaves it attached with live readers. True joins the last release's block
// access and permits reclamation of its metadata and native extent. This cold
// publisher operation walks only returned cells, never live cells. A copied
// notification may still run; its target belongs to the enclosing owner.
bool iree_hal_amd_gpu_signal_block_try_deinitialize(
    iree_hal_amd_gpu_signal_block_t* block);

// Detaches a block after the enclosing owner has joined all signal references.
void iree_hal_amd_gpu_signal_block_deinitialize(
    iree_hal_amd_gpu_signal_block_t* block);

// Constant-time publisher query used after capacity enrollment. An empty local
// pool exchanges the shared return head even when empty, ordering enrollment
// against final-return notifications. Returned cells become publisher-local.
bool iree_hal_amd_gpu_signal_pool_has_available(
    iree_hal_amd_gpu_signal_pool_t* pool);

// Returns one owning reference, or NULL for ordinary capacity pressure. Writes
// initial_value to the exclusively owned native value without shared metadata
// access. The protocol's publication barrier orders this store before GPU use.
// PM4 supplies zero and later writes one; AQL supplies one and decrements to
// zero. An abandoned unpublished signal can return without a write; there is no
// generation wrap or dependency on a previous producer having executed. The
// caller transfers or retains a reference before exposing the signal to another
// owner.
iree_hal_amd_gpu_signal_t* iree_hal_amd_gpu_signal_pool_try_acquire(
    iree_hal_amd_gpu_signal_pool_t* pool, uint64_t initial_value);

// Captures an accepted producer's complete native-domain frontier before the
// signal is exposed to readers or completion. The vector covers the current
// table width and includes the producer's own accepted publication epoch.
// Exactly one publisher writes it; every subsequent retained reader sees that
// immutable snapshot, even after the producer queue advances or disappears.
void iree_hal_amd_gpu_signal_publish_frontier(iree_hal_amd_gpu_signal_t* signal,
                                              const uint64_t* frontier);

// Joins the retained producer's immutable snapshot into an invocation's
// candidate frontier of queue_count entries. Cold membership expansion can
// temporarily expose a wider table with zeroes in coordinates the consumer has
// not installed yet. The merge bounds that zero suffix to its current extent;
// new coordinates become usable only after every peer has installed them.
// Rejection discards the candidate without claiming its native waits executed.
void iree_hal_amd_gpu_signal_join_frontier(
    const iree_hal_amd_gpu_signal_t* signal, uint32_t queue_count,
    uint64_t* frontier);

// Adds a use from an existing live reference. A cache must pin the reference
// under the same lock that protects removal; raw pointer loads cannot retain a
// signal racing its last release and reuse. NULL is ignored.
void iree_hal_amd_gpu_signal_retain(iree_hal_amd_gpu_signal_t* signal);

// Publishes the terminal outcome after all producer payload references and
// public completion effects have joined. Native execution alone does not
// establish this fact. Only the producer's completion owner calls this, once;
// final publication-bookkeeping release cannot overwrite the outcome.
static inline void iree_hal_amd_gpu_signal_retire(
    iree_hal_amd_gpu_signal_t* signal, iree_status_code_t status_code) {
  if (signal) {
    iree_atomic_store(&signal->state.live.retirement, (int32_t)status_code + 1,
                      iree_memory_order_release);
  }
}

// Queries a retained native dependency without visiting its original queue,
// capture, semaphore, or global frontier tracker. A true result publishes the
// terminal status code; false leaves out_status_code unchanged.
static inline bool iree_hal_amd_gpu_signal_query_retirement(
    const iree_hal_amd_gpu_signal_t* signal,
    iree_status_code_t* out_status_code) {
  const int32_t retirement = iree_atomic_load(&signal->state.live.retirement,
                                              iree_memory_order_acquire);
  if (!retirement) {
    return false;
  }
  *out_status_code = (iree_status_code_t)(retirement - 1);
  return true;
}

// Returns one use, making the cell reusable only after the last use joins.
// A native use ends at proven execution completion or native quiescence, never
// from failure alone. All accesses finish before this call; even its return
// does not permit another read of the signal. Last release atomically pushes
// the cell onto the pool's return list, with no allocation or native
// destruction. Only an empty-to-nonempty return list transition invokes the
// copied pool notification. NULL is ignored.
void iree_hal_amd_gpu_signal_release(iree_hal_amd_gpu_signal_t* signal);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMD_GPU_QUEUE_SIGNAL_H_
