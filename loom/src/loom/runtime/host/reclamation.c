// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/runtime/host/reclamation.h"

static void loom_runtime_reclamation_notify(void* user_data) {
  loom_runtime_reclamation_t* reclamation = user_data;
  iree_atomic_store(&reclamation->notified, 1, iree_memory_order_release);
  reclamation->observer.wake(reclamation->observer.user_data);
}

iree_status_t loom_runtime_reclamation_initialize(
    loom_runtime_host_t* host, iree_duration_t timeout,
    loom_runtime_reclamation_t* reclamation) {
  *reclamation = (loom_runtime_reclamation_t){.host = host, .timeout = timeout};
  loom_runtime_timer_initialize(&reclamation->timer);
  loom_runtime_readiness_initialize(
      host ? loom_runtime_host_executor(host) : NULL, NULL,
      &reclamation->readiness);
  if (timeout < 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "compiler idle timeout must be nonnegative");
  }
  return iree_ok_status();
}

void loom_runtime_reclamation_set_observer(
    loom_runtime_reclamation_t* reclamation,
    loom_runtime_wake_callback_t observer) {
  if (reclamation->observing) {
    loom_runtime_host_unobserve_preparation(reclamation->host,
                                            &reclamation->preparation);
    reclamation->observing = false;
  }
  reclamation->observer = observer;
  const loom_runtime_wake_callback_t wake = {loom_runtime_reclamation_notify,
                                             reclamation};
  loom_runtime_timer_set_observer(&reclamation->timer, wake);
  loom_runtime_readiness_set_observer(&reclamation->readiness, wake);
  if (reclamation->host && observer.wake) {
    reclamation->preparation.callback = wake;
    loom_runtime_host_observe_preparation(reclamation->host,
                                          &reclamation->preparation);
    reclamation->observing = true;
    // Initial observation includes activity that preceded attachment.
    loom_runtime_reclamation_notify(reclamation);
  }
}

static void loom_runtime_reclamation_stop(
    loom_runtime_reclamation_t* reclamation) {
  iree_atomic_store(&reclamation->stopping, 1, iree_memory_order_release);
  if (reclamation->observing) {
    loom_runtime_host_unobserve_preparation(reclamation->host,
                                            &reclamation->preparation);
    reclamation->observing = false;
  }
}

static iree_status_t loom_runtime_reclamation_run(void* user_data) {
  loom_runtime_reclamation_t* reclamation = user_data;
  if (iree_atomic_load(&reclamation->stopping, iree_memory_order_acquire)) {
    return iree_ok_status();
  }
  return loom_runtime_host_trim_compiler_idle(
      reclamation->host, reclamation->timeout, iree_time_now());
}

iree_status_t loom_runtime_reclamation_advance(
    loom_runtime_reclamation_t* reclamation, bool stopping,
    bool* out_progress) {
  *out_progress = false;
  if (!reclamation->host) {
    return iree_ok_status();
  }
  if (!stopping &&
      !iree_atomic_load(&reclamation->notified, iree_memory_order_acquire)) {
    return iree_ok_status();
  }
  *out_progress = iree_atomic_exchange(&reclamation->notified, 0,
                                       iree_memory_order_acquire) != 0;
  if (stopping) {
    loom_runtime_reclamation_stop(reclamation);
  }
  iree_status_t status = iree_ok_status();
  if (reclamation->task_pending &&
      loom_runtime_readiness_take(&reclamation->readiness,
                                  &reclamation->task.result, &status)) {
    reclamation->task_pending = false;
    *out_progress = true;
  }
  if (!iree_status_is_ok(status)) {
    loom_runtime_reclamation_stop(reclamation);
  }
  stopping =
      iree_atomic_load(&reclamation->stopping, iree_memory_order_acquire);
  const iree_time_t deadline =
      stopping ? IREE_TIME_INFINITE_FUTURE
               : loom_runtime_host_compiler_idle_deadline(reclamation->host,
                                                          reclamation->timeout);
  bool delayed = false;
  if (!stopping && !reclamation->task_pending &&
      deadline != IREE_TIME_INFINITE_FUTURE) {
    if (deadline <= iree_time_now()) {
      reclamation->task_pending = true;
      loom_runtime_readiness_schedule(&reclamation->readiness,
                                      loom_runtime_reclamation_run, reclamation,
                                      &reclamation->task);
      *out_progress = true;
    } else {
      delayed = true;
      if (!reclamation->timer.interval) {
        iree_async_proactor_t* proactor = NULL;
        status = iree_async_proactor_pool_get(
            loom_runtime_host_proactor_pool(reclamation->host), 0, &proactor);
        if (iree_status_is_ok(status)) {
          loom_runtime_timer_configure(&reclamation->timer, proactor,
                                       reclamation->timeout);
        }
      }
    }
  }
  if (!iree_status_is_ok(status)) {
    loom_runtime_reclamation_stop(reclamation);
  }
  bool timer_progress = false;
  bool tick = false;
  iree_status_t timer_status = loom_runtime_timer_advance(
      &reclamation->timer, stopping || !iree_status_is_ok(status), delayed,
      &timer_progress, &tick);
  status = iree_status_join(status, timer_status);
  reclamation->timer_pending =
      loom_runtime_timer_is_pending(&reclamation->timer);
  *out_progress |= timer_progress;
  if (!iree_status_is_ok(status)) {
    loom_runtime_reclamation_stop(reclamation);
  }
  return status;
}

bool loom_runtime_reclamation_is_pending(
    loom_runtime_reclamation_t* reclamation) {
  return reclamation->task_pending || reclamation->timer_pending;
}

void loom_runtime_reclamation_deinitialize(
    loom_runtime_reclamation_t* reclamation) {
  IREE_ASSERT(!reclamation->observing);
  IREE_ASSERT(!loom_runtime_reclamation_is_pending(reclamation));
  loom_runtime_readiness_deinitialize(&reclamation->readiness);
  loom_runtime_timer_deinitialize(&reclamation->timer);
}
