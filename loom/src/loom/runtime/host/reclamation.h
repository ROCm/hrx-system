// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_RUNTIME_HOST_RECLAMATION_H_
#define LOOM_RUNTIME_HOST_RECLAMATION_H_

#include "loom/runtime/host/host.h"
#include "loom/runtime/host/readiness.h"
#include "loom/runtime/host/timer.h"

#ifdef __cplusplus
extern "C" {
#endif

// Owner-driven compiler reclamation over a shared host. Preparation activity
// schedules an idle alarm; reclamation runs on the existing CPU workers.
// Completed idle cycles park their alarms until preparation happens again.
// Independent owners may observe the same host; the host claims each
// reclamation once.
typedef struct loom_runtime_reclamation_t {
  // Borrowed shared host, alive through deinitialization; NULL disables work.
  loom_runtime_host_t* host;
  // Minimum quiet interval before reclaiming compiler scratch, in nanoseconds.
  iree_duration_t timeout;
  // Borrowed serialized-owner wake through final callback retirement.
  loom_runtime_wake_callback_t observer;
  // Registered host activity notification, detached when stopping begins.
  loom_runtime_host_preparation_observer_t preparation;
  // Shared-proactor alarm, configured lazily when the first delay is needed.
  loom_runtime_timer_t timer;
  // Shared-worker result publication and final task retirement.
  loom_runtime_readiness_t readiness;
  // Callback activity awaiting an owner turn; quiet turns need no mutex.
  iree_atomic_int32_t notified;
  // Irreversible stop observed by a queued worker before beginning its work.
  iree_atomic_int32_t stopping;
  // An accepted CPU task or its unconsumed result still borrows this owner.
  bool task_pending;
  // An accepted alarm or its outcome awaits the next owner turn. Callbacks
  // cannot create new work, so this stays authoritative between advances.
  bool timer_pending;
  // The host can still invoke the preparation observer.
  bool observing;
  // One allocation-free CPU borrower, reused after its result is consumed.
  loom_runtime_readiness_task_t task;
} loom_runtime_reclamation_t;

// Initializes without accepting work or creating an I/O service. timeout is
// nonnegative; zero reclaims immediately after preparation scopes retire.
// NULL host disables reclamation. Even on failure, deinitialize is required.
iree_status_t loom_runtime_reclamation_initialize(
    loom_runtime_host_t* host, iree_duration_t timeout,
    loom_runtime_reclamation_t* reclamation);

// Installs the owner before its first advance. Clear only after stopping has
// retired accepted work. Both host and asynchronous callbacks are joined before
// detachment completes; the callback only makes the owner runnable.
void loom_runtime_reclamation_set_observer(
    loom_runtime_reclamation_t* reclamation,
    loom_runtime_wake_callback_t observer);

// Consumes outcomes and starts eligible cold work without blocking its owner.
// A delayed worker rechecks host activity immediately before reclamation.
// Stopping detaches activity observation, cancels alarms, and joins accepted
// worker outcomes. A trim already running may finish; its status is returned.
// Platform and allocator failures stop this owner and propagate here.
iree_status_t loom_runtime_reclamation_advance(
    loom_runtime_reclamation_t* reclamation, bool stopping, bool* out_progress);

// Actual task/timer borrowers or outcomes, queried by the serialized owner.
bool loom_runtime_reclamation_is_pending(
    loom_runtime_reclamation_t* reclamation);

// Releases synchronization after stopping has retired every accepted callback.
// An initialized owner that never installed an observer can be destroyed
// directly.
void loom_runtime_reclamation_deinitialize(
    loom_runtime_reclamation_t* reclamation);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_RUNTIME_HOST_RECLAMATION_H_
