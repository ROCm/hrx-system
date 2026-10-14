// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMD_GPU_QUEUE_CONTINUATION_H_
#define IREE_HAL_DRIVERS_AMD_GPU_QUEUE_CONTINUATION_H_

#include "iree/hal/memory/maintenance.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Runs one bounded service turn. True means more work can progress immediately;
// false means idle or waiting on a separately enrolled readiness/capacity wake.
// The callback consumes authoritative predicates, never notification counts.
// It runs outside the continuation's metadata lock on the device's existing
// service/maintenance executor. Native publication and cold driver calls do
// not occupy the completion worker.
typedef bool(IREE_API_PTR* iree_hal_amd_gpu_continuation_fn_t)(void* user_data);

// One persistent queue continuation, coalesced onto the existing device service
// owner. No thread, native event, operation capture or reference is created.
// An idle calling publisher does not use this object. It is notified only for
// deferred publication/readiness, then yields between bounded service turns so
// other queues and cold maintenance keep progressing on that same worker.
typedef struct iree_hal_amd_gpu_continuation_t {
  // Intrusive entry queued at most once, including while its callback runs.
  iree_hal_memory_maintenance_entry_t entry;
  // Borrowed thread-backed device service, live through deinitialization.
  iree_hal_memory_maintenance_t* maintenance;
  // Bounded ready-work callback, immutable after initialization.
  iree_hal_amd_gpu_continuation_fn_t fn;
  // Borrowed callback state, live through deinitialization.
  void* user_data;
  // Protects only scheduling state and the optional teardown waiter.
  iree_slim_mutex_t mutex;
  // Private scheduled/notified bits; callback execution does not hold mutex.
  uint32_t state;
  // Borrowed stack notification while the sole deinitializer joins a callback.
  struct iree_notification_t* join_notification;
} iree_hal_amd_gpu_continuation_t;

void iree_hal_amd_gpu_continuation_initialize(
    iree_hal_memory_maintenance_t* maintenance,
    iree_hal_amd_gpu_continuation_fn_t fn, void* user_data,
    iree_hal_amd_gpu_continuation_t* out_continuation);

// Joins the final callback before reclaiming this embedded object. All external
// notifiers must have joined first. This does not cancel work or wait for GPU
// execution; a queue establishes its own operation/resource joins separately.
// Must run outside the service executor. The callback must eventually return
// false after exhausting ready work. No timeout or periodic polling is used.
void iree_hal_amd_gpu_continuation_deinitialize(
    iree_hal_amd_gpu_continuation_t* continuation);

// Publishes a coalescible hint after updating the authoritative predicate. A
// notification racing the current turn is remembered for a subsequent turn,
// closing the predicate-check-to-park race. This allocates nothing and never
// waits for native execution. The caller keeps the object live through return.
void iree_hal_amd_gpu_continuation_notify(
    iree_hal_amd_gpu_continuation_t* continuation);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMD_GPU_QUEUE_CONTINUATION_H_
