// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/loom_serve/runtime/signals.h"

#include "iree/async/operations/scheduling.h"
#include "iree/base/threading/mutex.h"
#include "iree/base/threading/notification.h"

struct loom_serve_signals_t {
  // Allocator owning the subscription controller.
  iree_allocator_t allocator;
  // Retained progress ownership shared with network and device work.
  iree_async_proactor_pool_entry_t* entry;
  // Borrowed application callback, detached before destroy returns.
  iree_async_signal_callback_t callback;
  // Handles accessed only by the poll owner.
  iree_async_signal_subscription_t* subscriptions[2];
  // Registration/unregistration operation, reused only after callback
  // retirement.
  iree_async_nop_operation_t operation;
  // Joins the registration result and final poll callback access.
  iree_slim_mutex_t mutex;
  // Wakes the outer application owner, never a task-pool worker.
  iree_notification_t notification;
  // True after the current handoff has returned its storage.
  bool ready;
  // Registration or submission error, transferred to the creating caller.
  iree_status_t status;
};

static void serving_signals_publish(loom_serve_signals_t* signals,
                                    iree_status_t status) {
  iree_slim_mutex_lock(&signals->mutex);
  signals->status = status;
  signals->ready = true;
  iree_notification_post(&signals->notification, IREE_ALL_WAITERS);
  iree_slim_mutex_unlock(&signals->mutex);
}

static void serving_signals_unsubscribe(loom_serve_signals_t* signals) {
  iree_async_proactor_t* proactor =
      iree_async_proactor_pool_entry_proactor(signals->entry);
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(signals->subscriptions);
       ++i) {
    iree_async_proactor_unsubscribe_signal(proactor, signals->subscriptions[i]);
    signals->subscriptions[i] = NULL;
  }
}

static void serving_signals_register(void* user_data,
                                     iree_async_operation_t* operation,
                                     iree_status_t status,
                                     iree_async_completion_flags_t flags) {
  (void)operation;
  (void)flags;
  loom_serve_signals_t* signals = user_data;
  iree_async_proactor_t* proactor =
      iree_async_proactor_pool_entry_proactor(signals->entry);
  const iree_async_signal_t types[] = {IREE_ASYNC_SIGNAL_INTERRUPT,
                                       IREE_ASYNC_SIGNAL_TERMINATE};
  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(types) && iree_status_is_ok(status); ++i) {
    status = iree_async_proactor_subscribe_signal(
        proactor, types[i], signals->callback, &signals->subscriptions[i]);
  }
  if (!iree_status_is_ok(status)) {
    serving_signals_unsubscribe(signals);
  }
  serving_signals_publish(signals, status);
}

static void serving_signals_unregister(void* user_data,
                                       iree_async_operation_t* operation,
                                       iree_status_t status,
                                       iree_async_completion_flags_t flags) {
  (void)operation;
  (void)flags;
  IREE_CHECK_OK(status);
  loom_serve_signals_t* signals = user_data;
  serving_signals_unsubscribe(signals);
  serving_signals_publish(signals, iree_ok_status());
}

static bool serving_signals_ready(void* user_data) {
  loom_serve_signals_t* signals = user_data;
  iree_slim_mutex_lock(&signals->mutex);
  const bool ready = signals->ready;
  iree_slim_mutex_unlock(&signals->mutex);
  return ready;
}

static void serving_signals_free(loom_serve_signals_t* signals) {
  iree_async_proactor_pool_entry_release(signals->entry);
  iree_notification_deinitialize(&signals->notification);
  iree_slim_mutex_deinitialize(&signals->mutex);
  iree_allocator_free(signals->allocator, signals);
}

iree_status_t loom_serve_signals_create(iree_async_proactor_pool_t* pool,
                                        iree_async_signal_callback_t callback,
                                        loom_serve_signals_t** out_signals,
                                        iree_allocator_t allocator) {
  *out_signals = NULL;
  if (!callback.fn) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "signal callback is required");
  }
  loom_serve_signals_t* signals = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(allocator, sizeof(*signals), (void**)&signals));
  signals->allocator = allocator;
  signals->callback = callback;
  iree_slim_mutex_initialize(&signals->mutex);
  iree_notification_initialize(&signals->notification);
  iree_status_t status =
      iree_async_proactor_pool_acquire(pool, 0, &signals->entry);
  if (iree_status_is_ok(status)) {
    iree_async_operation_initialize(
        &signals->operation.base, IREE_ASYNC_OPERATION_TYPE_NOP,
        IREE_ASYNC_OPERATION_FLAG_NONE, serving_signals_register, signals);
    status = iree_async_proactor_submit_one(
        iree_async_proactor_pool_entry_proactor(signals->entry),
        &signals->operation.base);
  }
  if (iree_status_is_ok(status)) {
    iree_notification_await(&signals->notification, serving_signals_ready,
                            signals, iree_infinite_timeout());
    status = signals->status;
  }
  if (iree_status_is_ok(status)) {
    *out_signals = signals;
  } else {
    serving_signals_free(signals);
  }
  return status;
}

void loom_serve_signals_destroy(loom_serve_signals_t* signals) {
  if (!signals) {
    return;
  }
  signals->ready = false;
  iree_async_operation_initialize(
      &signals->operation.base, IREE_ASYNC_OPERATION_TYPE_NOP,
      IREE_ASYNC_OPERATION_FLAG_NONE, serving_signals_unregister, signals);
  IREE_CHECK_OK(iree_async_proactor_submit_one(
      iree_async_proactor_pool_entry_proactor(signals->entry),
      &signals->operation.base));
  iree_notification_await(&signals->notification, serving_signals_ready,
                          signals, iree_infinite_timeout());
  serving_signals_free(signals);
}
