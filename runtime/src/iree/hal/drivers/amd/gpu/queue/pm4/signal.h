// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMD_GPU_QUEUE_PM4_SIGNAL_H_
#define IREE_HAL_DRIVERS_AMD_GPU_QUEUE_PM4_SIGNAL_H_

#include "iree/base/api.h"
#include "iree/base/internal/atomics.h"
#include "iree/base/threading/mutex.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_hal_amd_gpu_pm4_signal_pool_t
    iree_hal_amd_gpu_pm4_signal_pool_t;
typedef struct iree_hal_amd_gpu_pm4_signal_block_t
    iree_hal_amd_gpu_pm4_signal_block_t;

// One native binary signal, independently retained by its publisher,
// producer, caches, and dependent executions. A producer completing does not
// retire the native readers. This metadata stays in ordinary host memory;
// only the separately backed value is accessed by the GPU.
typedef struct iree_hal_amd_gpu_pm4_signal_t {
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
    // Link accessed only under the pool mutex after the last use has joined.
    struct iree_hal_amd_gpu_pm4_signal_t* next;
  } state;
  // Borrowed block owning this slot, live until every reference is returned.
  iree_hal_amd_gpu_pm4_signal_block_t* block;
  // Immutable address of an eight-byte cell in qualified native backing.
  uint64_t device_address;
  // Borrowed host cell, cleared only during exclusive acquisition. Native
  // release writes one; readers wait for equality with one in either width.
  uint64_t* host_value;
} iree_hal_amd_gpu_pm4_signal_t;

// Coalescible capacity hint, invoked after returning a slot and releasing the
// pool mutex. The callback notifies the enclosing owner's existing progress
// mechanism; it does not wait or destroy the pool. Another acquisition may
// consume the available slot before the callback runs.
typedef struct iree_hal_amd_gpu_pm4_signal_pool_notification_t {
  // Optional callback for exhausted-to-available transitions.
  void (*fn)(void* user_data);
  // Borrowed callback state, live through all external pool calls.
  void* user_data;
} iree_hal_amd_gpu_pm4_signal_pool_notification_t;

// Available cells across independently allocated blocks. Acquisition and last
// release take one short metadata lock and perform constant-time list updates.
// A pending native allocation is independent of returned capacity. This object
// performs no allocation or native call and owns no worker.
typedef struct iree_hal_amd_gpu_pm4_signal_pool_t {
  // Protects block membership, free lists, and each block's live_count.
  iree_slim_mutex_t mutex;
  // Blocks with at least one free cell, linked without an availability scan.
  iree_hal_amd_gpu_pm4_signal_block_t* available;
  // Number of attached blocks, including exhausted and zero-capacity blocks.
  uint32_t block_count;
  // Immutable wake target for exhausted-to-available transitions.
  iree_hal_amd_gpu_pm4_signal_pool_notification_t notification;
} iree_hal_amd_gpu_pm4_signal_pool_t;

// Finite bookkeeping borrowing one native extent and its host metadata. The
// enclosing owner joins every native writer, reader, and cache reference before
// detaching a block. The pool and notification target outlive those releases.
typedef struct iree_hal_amd_gpu_pm4_signal_block_t {
  // Borrowed pool, or NULL after successful detachment.
  iree_hal_amd_gpu_pm4_signal_pool_t* pool;
  // Number of slots with at least one live use, protected by mutex.
  uint32_t live_count;
  // Complete immutable native-domain queue count, including inactive queues.
  uint32_t queue_count;
  // Available metadata slots, protected by mutex.
  iree_hal_amd_gpu_pm4_signal_t* free_head;
  // Previous available block, or NULL at the list head or while exhausted.
  iree_hal_amd_gpu_pm4_signal_block_t* available_previous;
  // Next available block, or NULL at the list tail or while exhausted.
  iree_hal_amd_gpu_pm4_signal_block_t* available_next;
  // Borrowed qualified address/visibility/atomic domain. Matching identities
  // permit native reads of these exact addresses; NULL permits only host waits.
  const void* domain;
  // Borrowed metadata base for constant-time signal-to-frontier indexing.
  iree_hal_amd_gpu_pm4_signal_t* signals;
  // Signal-major publication epochs, capacity * queue_count host entries.
  // Each retained signal's vector is immutable after native acceptance.
  uint64_t* frontiers;
} iree_hal_amd_gpu_pm4_signal_block_t;

// Initializes an empty owner. Native blocks are attached only after their
// allocation completes. All blocks must be detached before deinitialization.
void iree_hal_amd_gpu_pm4_signal_pool_initialize(
    iree_hal_amd_gpu_pm4_signal_pool_notification_t notification,
    iree_hal_amd_gpu_pm4_signal_pool_t* out_pool);
void iree_hal_amd_gpu_pm4_signal_pool_deinitialize(
    iree_hal_amd_gpu_pm4_signal_pool_t* pool);

// Initializes exclusively owned backing to zero and links the host metadata.
// Both extents contain capacity entries; device_address names values[0].
// The native storage is eight-byte aligned with qualified host/device
// visibility and atomicity for the producer's and readers' chosen 32- or 64-bit
// width. The DWORD form uses the low word and leaves its high word zero.
// Metadata, values, and frontiers do not overlap. Frontiers has
// capacity * queue_count entries; publication initializes a complete vector
// before exposing a signal. A zero capacity attaches an empty block.
// Attaches the initialized block under the pool mutex. The attaching publisher
// already owns its waiting work; attachment does not invoke the notification.
void iree_hal_amd_gpu_pm4_signal_block_initialize(
    iree_hal_amd_gpu_pm4_signal_pool_t* pool, const void* domain,
    uint64_t* values, uint64_t device_address,
    iree_hal_amd_gpu_pm4_signal_t* signals, uint32_t capacity,
    uint32_t queue_count, uint64_t* frontiers,
    iree_hal_amd_gpu_pm4_signal_block_t* out_block);

// Detaches an entirely unused block from acquisition in constant time. False
// leaves it attached with live readers. True joins the last release's block
// access and permits reclamation of its metadata and native extent. A copied
// notification may still run; its target belongs to the enclosing owner.
bool iree_hal_amd_gpu_pm4_signal_block_try_deinitialize(
    iree_hal_amd_gpu_pm4_signal_block_t* block);

// Detaches a block after the enclosing owner has joined all signal references.
void iree_hal_amd_gpu_pm4_signal_block_deinitialize(
    iree_hal_amd_gpu_pm4_signal_block_t* block);

// Constant-time capacity observation used to close a miss/enrollment race.
bool iree_hal_amd_gpu_pm4_signal_pool_has_available(
    iree_hal_amd_gpu_pm4_signal_pool_t* pool);

// Returns one owning reference, or NULL for ordinary capacity pressure. Clears
// the exclusively owned cell to zero outside the metadata lock. The caller's
// native publication barrier orders this store before any GPU access. Release
// writes one and dependent queues wait for equality with one. An abandoned
// unpublished signal can return without a write; there is no generation wrap
// or dependency on a previous producer having executed. The caller transfers
// or retains a reference before exposing the signal to another owner.
iree_hal_amd_gpu_pm4_signal_t* iree_hal_amd_gpu_pm4_signal_pool_try_acquire(
    iree_hal_amd_gpu_pm4_signal_pool_t* pool);

// Captures an accepted producer's complete native-domain frontier before the
// signal is exposed to readers or completion. The vector has block->queue_count
// entries and includes the producer's own accepted publication epoch. Exactly
// one publisher writes it; every subsequent retained reader sees that immutable
// snapshot, even after the producer queue advances or disappears.
void iree_hal_amd_gpu_pm4_signal_publish_frontier(
    iree_hal_amd_gpu_pm4_signal_t* signal, const uint64_t* frontier);

// Joins the retained producer's immutable snapshot into an invocation's
// candidate frontier. Domain compatibility and equal queue counts are proven
// at admission. This does not change the accepted FIFO frontier; rejection
// discards the candidate without claiming that its native waits executed.
void iree_hal_amd_gpu_pm4_signal_join_frontier(
    const iree_hal_amd_gpu_pm4_signal_t* signal, uint64_t* frontier);

// Adds a use from an existing live reference. A cache must pin the reference
// under the same lock that protects removal; raw pointer loads cannot retain a
// signal racing its last release and reuse. NULL is ignored.
void iree_hal_amd_gpu_pm4_signal_retain(iree_hal_amd_gpu_pm4_signal_t* signal);

// Publishes the terminal outcome after all producer payload references and
// public completion effects have joined. Native execution alone does not
// establish this fact. Only the producer's completion owner calls this, once;
// final publication-bookkeeping release cannot overwrite the outcome.
static inline void iree_hal_amd_gpu_pm4_signal_retire(
    iree_hal_amd_gpu_pm4_signal_t* signal, iree_status_code_t status_code) {
  if (signal) {
    iree_atomic_store(&signal->state.live.retirement, (int32_t)status_code + 1,
                      iree_memory_order_release);
  }
}

// Queries a retained native dependency without visiting its original queue,
// capture, semaphore, or global frontier tracker. A true result publishes the
// terminal status code; false leaves out_status_code unchanged.
static inline bool iree_hal_amd_gpu_pm4_signal_query_retirement(
    const iree_hal_amd_gpu_pm4_signal_t* signal,
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
// does not permit another read of the signal. Last release takes the pool's
// short metadata lock, with no allocation or native destruction. Only an
// exhausted-to-available transition invokes the pool's notification, outside
// the mutex. NULL is ignored.
void iree_hal_amd_gpu_pm4_signal_release(iree_hal_amd_gpu_pm4_signal_t* signal);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMD_GPU_QUEUE_PM4_SIGNAL_H_
