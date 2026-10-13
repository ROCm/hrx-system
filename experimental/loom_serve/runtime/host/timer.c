// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/loom_serve/runtime/host/timer.h"

#include <string.h>

#include "experimental/loom_serve/runtime/host/wake.h"

enum {
  TIMER_DEADLINE = 1u << 0,
  TIMER_STOP = 1u << 1,
  TIMER_CANCEL = 1u << 2,
};

static void timer_cancel_retired(void* user_data) {
  loom_serve_timer_t* timer = user_data;
  iree_slim_mutex_lock(&timer->mutex);
  timer->events.pending &= ~TIMER_CANCEL;
  timer->events.ready = true;
  timer->observer.wake(timer->observer.user_data);
  iree_slim_mutex_unlock(&timer->mutex);
}

static void timer_deadline_completed(void* user_data,
                                     iree_async_operation_t* operation,
                                     iree_status_t status,
                                     iree_async_completion_flags_t flags) {
  (void)operation;
  (void)flags;
  loom_serve_timer_t* timer = user_data;
  iree_slim_mutex_lock(&timer->mutex);
  timer->events.status = iree_status_join(timer->events.status, status);
  timer->events.pending &= ~TIMER_DEADLINE;
  timer->events.ready = true;
  const bool cancel_pending =
      iree_any_bit_set(timer->events.pending, TIMER_CANCEL);
  if (!cancel_pending) {
    timer->observer.wake(timer->observer.user_data);
  }
  iree_slim_mutex_unlock(&timer->mutex);
  if (cancel_pending) {
    // The receipt may publish the last borrowed access. No access follows it.
    iree_async_proactor_cancel_request_target_retired(timer->proactor,
                                                      &timer->cancellation);
  }
}

static void timer_stop_received(void* user_data,
                                iree_async_operation_t* operation,
                                iree_status_t status,
                                iree_async_completion_flags_t flags) {
  (void)operation;
  (void)flags;
  loom_serve_timer_t* timer = user_data;
  iree_slim_mutex_lock(&timer->mutex);
  const bool cancel = iree_status_is_ok(status) &&
                      iree_any_bit_set(timer->events.pending, TIMER_DEADLINE);
  timer->events.status = iree_status_join(timer->events.status, status);
  if (cancel) {
    timer->events.pending |= TIMER_CANCEL;
  }
  iree_slim_mutex_unlock(&timer->mutex);
  if (cancel) {
    // This poll-owner callback owns a private live timer and an idle receipt.
    IREE_CHECK_OK(iree_async_proactor_request_cancel(
        timer->proactor, &timer->operation.base, &timer->cancellation));
  }
  iree_slim_mutex_lock(&timer->mutex);
  timer->events.pending &= ~TIMER_STOP;
  timer->events.ready = true;
  timer->observer.wake(timer->observer.user_data);
  iree_slim_mutex_unlock(&timer->mutex);
}

void loom_serve_timer_initialize(loom_serve_timer_t* timer) {
  *timer = (loom_serve_timer_t){0};
  iree_slim_mutex_initialize(&timer->mutex);
  iree_async_cancel_request_initialize(
      (iree_async_cancel_callback_t){timer_cancel_retired, timer},
      &timer->cancellation);
}

void loom_serve_timer_configure(loom_serve_timer_t* timer,
                                iree_async_proactor_t* proactor,
                                iree_duration_t interval) {
  timer->proactor = proactor;
  timer->interval = interval;
}

void loom_serve_timer_set_observer(loom_serve_timer_t* timer,
                                   loom_serve_wake_callback_t observer) {
  iree_slim_mutex_lock(&timer->mutex);
  timer->observer = observer;
  iree_slim_mutex_unlock(&timer->mutex);
}

bool loom_serve_timer_is_pending(loom_serve_timer_t* timer) {
  iree_slim_mutex_lock(&timer->mutex);
  const bool pending = timer->events.pending || timer->events.ready;
  iree_slim_mutex_unlock(&timer->mutex);
  return pending;
}

iree_status_t loom_serve_timer_advance(loom_serve_timer_t* timer, bool stopping,
                                       bool* out_progress, bool* out_tick) {
  *out_tick = false;
  iree_slim_mutex_lock(&timer->mutex);
  iree_status_t status = timer->events.status;
  timer->events.status = iree_ok_status();
  *out_progress = timer->events.ready;
  timer->events.ready = false;
  const uint32_t pending = timer->events.pending;
  iree_slim_mutex_unlock(&timer->mutex);
  stopping |= !iree_status_is_ok(status);
  if (stopping && !timer->stopping) {
    timer->stopping = true;
    *out_progress = true;
    if (iree_any_bit_set(pending, TIMER_DEADLINE)) {
      iree_async_nop_operation_t* stop = &timer->stop;
      memset(stop, 0, sizeof(*stop));
      iree_async_operation_initialize(
          &stop->base, IREE_ASYNC_OPERATION_TYPE_NOP,
          IREE_ASYNC_OPERATION_FLAG_NONE, timer_stop_received, timer);
      iree_slim_mutex_lock(&timer->mutex);
      timer->events.pending |= TIMER_STOP;
      iree_slim_mutex_unlock(&timer->mutex);
      iree_status_t stop_status =
          iree_async_proactor_submit_one(timer->proactor, &stop->base);
      if (!iree_status_is_ok(stop_status)) {
        iree_slim_mutex_lock(&timer->mutex);
        timer->events.pending &= ~TIMER_STOP;
        iree_slim_mutex_unlock(&timer->mutex);
      }
      status = iree_status_join(status, stop_status);
    }
  }
  if (!timer->stopping && !pending && timer->interval) {
    *out_tick = true;
    *out_progress = true;
    iree_async_timer_operation_t* operation = &timer->operation;
    memset(operation, 0, sizeof(*operation));
    iree_async_operation_initialize(
        &operation->base, IREE_ASYNC_OPERATION_TYPE_TIMER,
        IREE_ASYNC_OPERATION_FLAG_CANCELLATION_IS_SUCCESS,
        timer_deadline_completed, timer);
    operation->deadline_ns = iree_time_now() + timer->interval;
    iree_slim_mutex_lock(&timer->mutex);
    timer->events.pending |= TIMER_DEADLINE;
    iree_slim_mutex_unlock(&timer->mutex);
    status = iree_async_proactor_submit_one(timer->proactor, &operation->base);
    if (!iree_status_is_ok(status)) {
      iree_slim_mutex_lock(&timer->mutex);
      timer->events.pending &= ~TIMER_DEADLINE;
      iree_slim_mutex_unlock(&timer->mutex);
      timer->stopping = true;
    }
  }
  return status;
}

void loom_serve_timer_deinitialize(loom_serve_timer_t* timer) {
  IREE_ASSERT(!loom_serve_timer_is_pending(timer));
  iree_slim_mutex_deinitialize(&timer->mutex);
}
