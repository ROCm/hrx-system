// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVER_AMD_GPU_PUBLICATION_H_
#define IREE_HAL_DRIVER_AMD_GPU_PUBLICATION_H_

#include "iree/base/internal/atomics.h"
#include "iree/base/threading/mutex.h"
#include "iree/base/threading/notification.h"
#include "iree/base/threading/thread.h"
#include "iree/hal/driver/amd/gpu/queue/publisher.h"

#ifdef __cplusplus
extern "C" {
#endif

// Device construction admits up to 256 execution queues, one transfer queue,
// and allocation and mapping result sources. Five words represent the complete
// ready set without a source walk.
#define IREE_HAL_AMD_GPU_PUBLICATION_MAX_SOURCES 259

// One device progress source: a native queue, completed acquisition, or
// mapping result publication. A turn processes a bounded ready prefix and
// returns true only when another prefix can be published immediately.
// Unresolved dependencies and native allocation requests leave the ready set.
// Capacity deferral enrolls a return notification before its final recheck.
// No native memory call, I/O, dependency wait, or application callback runs
// here. Packet protocol and USER/KERNEL publication belong to the exact queue;
// completed acquisitions commit their buffer metadata and HAL signals here.
typedef struct iree_hal_amd_gpu_publication_source_t {
  // Sole deferred consumer of this source's ready state, or NULL for a vacant
  // ordinal whose stale readiness hints have no remaining work.
  bool (*publish)(void* user_data);
  // Borrowed queue, live until this source is removed or the worker is joined.
  void* user_data;
} iree_hal_amd_gpu_publication_source_t;

// One device-owned deferred publisher, separate from native allocation and
// completion. Ready API callers can publish inline through their queue's
// publisher; this owner handles bounded suffixes and enrolled readiness wakes.
// Source replacement joins only removed callbacks without touching caller
// publication ownership. There is no arbitrary work-item API.
typedef struct iree_hal_amd_gpu_publication_t {
  // Coalesced source identities, set by producers and consumed by the worker.
  iree_atomic_int64_t
      ready[(IREE_HAL_AMD_GPU_PUBLICATION_MAX_SOURCES + 63) / 64];
  // Parks the worker when no queue has an enrolled ready turn.
  iree_notification_t notification;
  // Stop and failure-watch requests, handled between bounded turns.
  iree_atomic_int32_t state;
  // Number of ready words to observe, published by source replacement.
  iree_atomic_int32_t word_count;
  // Owned worker, or NULL after failed initialization or final join.
  iree_thread_t* thread;
  // Callback borrowing is separate from the producers' ready cache line.
  iree_alignas(iree_hardware_destructive_interference_size) struct {
    // Protects only table access and active-callback bookkeeping.
    iree_slim_mutex_t mutex;
    // Borrowed immutable table; callbacks run from a by-value stack copy.
    const iree_hal_amd_gpu_publication_source_t* values;
    // Admitted table extent, including vacant ordinals.
    uint32_t count;
    // Sole active callback's ordinal, or UINT32_MAX between callbacks.
    uint32_t active_ordinal;
    // A replacement is joining this callback because it was not retained.
    bool joining;
    // Joins only the removed callback's final access, outside the mutex.
    iree_notification_t joined;
  } sources;
  // First-fault monitor, installed once before its control-bit publication.
  // The device joins this worker before releasing the borrowed failure state.
  struct iree_hal_amd_gpu_failure_t* failure_watch;
} iree_hal_amd_gpu_publication_t;

// Starts the deferred publisher after every source is prepared, before public
// submission is exposed. source_count was admitted by device construction.
// Failure leaves out_publication safe to deinitialize. The embedding device
// keeps all source state live through replacement or deinitialization.
iree_status_t iree_hal_amd_gpu_publication_initialize(
    uint32_t source_count, const iree_hal_amd_gpu_publication_source_t* sources,
    iree_thread_affinity_t affinity, iree_allocator_t host_allocator,
    iree_hal_amd_gpu_publication_t* out_publication);

// Installs a prepared callback table under short metadata exclusion. The old
// table can be reclaimed on return even while a retained callback is running:
// the worker holds its callback pair by value. Removing or replacing that
// active callback joins its final access outside the metadata mutex.
// Ordinals retain their identities; vacant slots have NULL callbacks. Stale
// ready bits may invoke a replacement at the same ordinal, so a callback always
// checks its authoritative readiness before publishing. The device admits
// source_count up to MAX_SOURCES. The replacement remains live until the next
// replacement or deinitialization. The owner serializes replacements with
// deinitialization and separately joins removed sources' accepted work and
// external notification producers. Called outside callbacks, with no allocation
// or native operation inside the handoff.
void iree_hal_amd_gpu_publication_set_sources(
    iree_hal_amd_gpu_publication_t* publication, uint32_t source_count,
    const iree_hal_amd_gpu_publication_source_t* sources);

// Requests a turn for one provisioned source. Duplicate notifications coalesce;
// a notification racing its current turn is retained for the next turn.
// The caller keeps publication and its queue alive through this final access.
void iree_hal_amd_gpu_publication_notify(
    iree_hal_amd_gpu_publication_t* publication, uint32_t source_ordinal);

// Arms the first native execution fault's independent terminal deadline.
// The existing worker reports a stalled native stop through the device sink;
// healthy publication has no deadline. Called once after failure timing is
// initialized, before the completion worker can enter blocking native removal.
void iree_hal_amd_gpu_publication_watch_failure(
    iree_hal_amd_gpu_publication_t* publication,
    struct iree_hal_amd_gpu_failure_t* failure);

// Transfers a completed native result to its queue's existing typed FIFO.
// The caller keeps the publisher alive through enqueue, independently of the
// entry: another owner can consume the entry before enqueue returns. After the
// final handoff, the wake accesses only device-owned values passed by value.
// A producer that acquires the publisher reserves a service turn instead of
// executing queue work on the native allocation callback thread.
void iree_hal_amd_gpu_publication_enqueue(
    iree_hal_amd_gpu_publication_t* publication, uint32_t source_ordinal,
    iree_hal_amd_gpu_publisher_t* publisher,
    iree_hal_amd_gpu_publication_entry_t* entry);

// Joins the final source access after accepted work and external notification
// producers have joined. It neither waits for GPU work nor cancels it. Called
// outside publication callbacks, before destroying any queue source storage.
void iree_hal_amd_gpu_publication_deinitialize(
    iree_hal_amd_gpu_publication_t* publication);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_HAL_DRIVER_AMD_GPU_PUBLICATION_H_
