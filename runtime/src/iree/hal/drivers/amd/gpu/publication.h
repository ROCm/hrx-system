// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMD_GPU_PUBLICATION_H_
#define IREE_HAL_DRIVERS_AMD_GPU_PUBLICATION_H_

#include "iree/base/internal/atomics.h"
#include "iree/base/threading/notification.h"
#include "iree/base/threading/thread.h"

#ifdef __cplusplus
extern "C" {
#endif

// Device construction admits up to 256 native queues and allocation and mapping
// result sources. Five words represent the complete ready set without a source
// walk.
#define IREE_HAL_AMD_GPU_PUBLICATION_MAX_SOURCES 258

// One fixed device progress source: a native queue, completed acquisition, or
// mapping result publication. A turn processes a bounded ready prefix and
// returns true only when another prefix can be published immediately.
// Unresolved dependencies and native allocation requests leave the ready set.
// Capacity deferral enrolls a return notification before its final recheck.
// No native memory call, I/O, dependency wait, or application callback runs
// here. Packet protocol and USER/KERNEL publication belong to the exact queue;
// completed acquisitions commit their buffer metadata and HAL signals here.
typedef struct iree_hal_amd_gpu_publication_source_t {
  // Sole deferred consumer of this source's ready state.
  bool (*publish)(void* user_data);
  // Borrowed queue, live through publication deinitialization.
  void* user_data;
} iree_hal_amd_gpu_publication_source_t;

// One device-owned deferred publisher, separate from native allocation and
// completion. Ready API callers can publish inline through their queue's
// publisher; this owner handles bounded suffixes and enrolled readiness wakes.
// Sources are fixed at construction. There is no arbitrary work-item API.
typedef struct iree_hal_amd_gpu_publication_t {
  // Owned worker, or NULL after failed initialization or final join.
  iree_thread_t* thread;
  // Borrowed immutable array of native queues and typed result sources.
  const iree_hal_amd_gpu_publication_source_t* sources;
  // Number of populated words in ready, derived from the admitted source count.
  uint32_t word_count;
  // Coalesced source identities, set by producers and consumed by the worker.
  iree_atomic_int64_t
      ready[(IREE_HAL_AMD_GPU_PUBLICATION_MAX_SOURCES + 63) / 64];
  // Exit requested only after accepted work and external producers join.
  iree_atomic_int32_t stopping;
  // Parks the worker when no queue has an enrolled ready turn.
  iree_notification_t notification;
} iree_hal_amd_gpu_publication_t;

// Starts the deferred publisher after every source is prepared, before public
// submission is exposed. source_count was admitted by device construction.
// Failure leaves out_publication safe to deinitialize. The embedding device
// keeps all source state live through deinitialization.
iree_status_t iree_hal_amd_gpu_publication_initialize(
    uint32_t source_count, const iree_hal_amd_gpu_publication_source_t* sources,
    iree_thread_affinity_t affinity, iree_allocator_t host_allocator,
    iree_hal_amd_gpu_publication_t* out_publication);

// Requests a turn for one provisioned source. Duplicate notifications coalesce;
// a notification racing its current turn is retained for the next turn.
// The caller keeps publication and its queue alive through this final access.
void iree_hal_amd_gpu_publication_notify(
    iree_hal_amd_gpu_publication_t* publication, uint32_t source_ordinal);

// Joins the final source access after accepted work and external notification
// producers have joined. It neither waits for GPU work nor cancels it. Called
// outside publication callbacks, before destroying any queue source storage.
void iree_hal_amd_gpu_publication_deinitialize(
    iree_hal_amd_gpu_publication_t* publication);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_HAL_DRIVERS_AMD_GPU_PUBLICATION_H_
