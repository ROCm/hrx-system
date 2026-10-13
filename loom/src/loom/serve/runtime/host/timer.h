// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_RUNTIME_HOST_TIMER_H_
#define LOOM_SERVE_RUNTIME_HOST_TIMER_H_

#include "iree/async/operations/scheduling.h"
#include "iree/async/proactor.h"
#include "iree/base/threading/mutex.h"
#include "loom/serve/runtime/host/wake.h"

#ifdef __cplusplus
extern "C" {
#endif

// Periodic service readiness with owned cancellation. One serialized owner
// advances the timer; proactor callbacks publish outcomes and wake that owner.
// The timer never reads model state or occupies a task worker while waiting.
typedef struct loom_serve_timer_t {
  // Joins callback publication and its final borrowed access.
  iree_slim_mutex_t mutex;
  // Borrowed host I/O owner, outliving every accepted operation.
  iree_async_proactor_t* proactor;
  // Borrowed nonblocking service wake target, protected by mutex.
  loom_serve_wake_callback_t observer;
  // Immutable reporting period in nanoseconds; zero disables ticks.
  iree_duration_t interval;
  // Owner has requested shutdown; no further timers are armed.
  bool stopping;
  // Callback publication consumed by serialized owner turns.
  struct {
    // Intrusive timer, stop handoff and cancellation receipt ownership bits.
    uint32_t pending;
    // A callback outcome awaits consumption after operation retirement.
    bool ready;
    // Owned callback failures, returned exactly once by advance.
    iree_status_t status;
  } events;
  // One deadline, reused only after its final callback access.
  iree_async_timer_operation_t operation;
  // Poll-owner handoff for requesting cancellation without blocking a worker.
  iree_async_nop_operation_t stop;
  // Receipt retiring any native cancellation key before timer storage dies.
  iree_async_cancel_request_t cancellation;
} loom_serve_timer_t;

// Initializes synchronization without accepting work. Partial construction
// may deinitialize immediately without configuring or advancing the timer.
void loom_serve_timer_initialize(loom_serve_timer_t* timer);

// Configures borrowed I/O and a nonnegative interval before the first advance.
void loom_serve_timer_configure(loom_serve_timer_t* timer,
                                iree_async_proactor_t* proactor,
                                iree_duration_t interval);

// Installs the owner's wake before advancing; clear only after pending work
// and callback outcomes have been consumed. Joins the last callback access.
void loom_serve_timer_set_observer(loom_serve_timer_t* timer,
                                   loom_serve_wake_callback_t observer);

// Whether a callback or unconsumed outcome still borrows this timer.
// Timer readiness alone cannot release model admission capacity.
bool loom_serve_timer_is_pending(loom_serve_timer_t* timer);

// Consumes readiness and arms the next deadline. The initial advance and each
// successful expiry produce a tick; cancellation readiness only reports
// progress. Stopping is sticky and cancels through the poll owner. Continue
// advancing pending work before detaching the observer or destroying storage.
iree_status_t loom_serve_timer_advance(loom_serve_timer_t* timer, bool stopping,
                                       bool* out_progress, bool* out_tick);

// Releases synchronization after all callbacks and outcomes retire. Never
// waits.
void loom_serve_timer_deinitialize(loom_serve_timer_t* timer);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVE_RUNTIME_HOST_TIMER_H_
