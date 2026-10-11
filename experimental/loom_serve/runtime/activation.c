// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/loom_serve/runtime/activation.h"

#include "iree/async/operations/semaphore.h"
#include "iree/base/threading/mutex.h"

struct loom_serve_activation_t {
  // Allocator owning this cold operation, released after callback retirement.
  iree_allocator_t allocator;
  // Borrowed parameter group outliving this operation and any failed load.
  loom_serve_residency_t* residency;
  // Retained proactor progress ownership, shared with model and network I/O.
  iree_async_proactor_pool_entry_t* entry;
  // Owner-only phase; callback publication cannot alter admission or pins.
  loom_serve_activation_phase_t phase;
  // Accepted readiness operation, reused only by creating a new warmup.
  iree_async_semaphore_wait_operation_t wait;
  // Borrowed exact model work and feedback values after accepted activation.
  loom_serve_execution_frontier_t frontier;
  // Toll-free async handles for the frontier's HAL semaphores.
  iree_async_semaphore_t* semaphores[2];
  // Completion publication is the only cross-thread mutable state.
  struct {
    // Joins the callback's final access before the owner frees operation
    // storage.
    iree_slim_mutex_t mutex;
    // Schedules the owner without accessing model resources inline.
    void (*wake)(void* user_data);
    // Borrowed scheduling target, live until terminal advance.
    void* user_data;
    // A terminal readiness result is available to the owner.
    bool ready;
    // Owned callback status, moved to the owner by advance.
    iree_status_t status;
  } event;
};

static void activation_ready(void* user_data, iree_async_operation_t* operation,
                             iree_status_t status,
                             iree_async_completion_flags_t flags) {
  (void)operation;
  (void)flags;
  loom_serve_activation_t* activation = user_data;
  iree_slim_mutex_lock(&activation->event.mutex);
  activation->event.status = status;
  activation->event.ready = true;
  activation->event.wake(activation->event.user_data);
  iree_slim_mutex_unlock(&activation->event.mutex);
}

iree_status_t loom_serve_activation_create(
    loom_serve_residency_t* residency, iree_async_proactor_pool_t* pool,
    void (*wake)(void* user_data), void* user_data,
    loom_serve_activation_t** out_activation, iree_allocator_t allocator) {
  *out_activation = NULL;
  loom_serve_activation_t* activation = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(allocator, sizeof(*activation),
                                             (void**)&activation));
  activation->allocator = allocator;
  activation->residency = residency;
  activation->event.wake = wake;
  activation->event.user_data = user_data;
  iree_status_t status =
      iree_async_proactor_pool_acquire(pool, 0, &activation->entry);
  if (iree_status_is_ok(status)) {
    iree_slim_mutex_initialize(&activation->event.mutex);
    *out_activation = activation;
  } else {
    iree_allocator_free(allocator, activation);
  }
  return status;
}

iree_status_t loom_serve_activation_advance(
    loom_serve_activation_t* activation, loom_serve_activation_flags_t flags,
    loom_serve_activation_phase_t* out_phase) {
  iree_status_t status = iree_ok_status();
  if (activation->phase == LOOM_SERVE_ACTIVATION_LOADING) {
    iree_slim_mutex_lock(&activation->event.mutex);
    const bool ready = activation->event.ready;
    if (ready) {
      status = activation->event.status;
      activation->event.status = iree_ok_status();
    }
    iree_slim_mutex_unlock(&activation->event.mutex);
    if (ready) {
      if (iree_status_is_ok(status)) {
        loom_serve_residency_release(activation->residency);
      }
      activation->phase = LOOM_SERVE_ACTIVATION_COMPLETE;
    }
  } else if (iree_any_bit_set(flags, LOOM_SERVE_ACTIVATION_FLAG_CANCEL)) {
    loom_serve_residency_cancel_admission(activation->residency);
    activation->phase = LOOM_SERVE_ACTIVATION_COMPLETE;
  } else {
    loom_serve_residency_admission_t admission;
    status = loom_serve_residency_try_admit(activation->residency, &admission);
    if (iree_status_is_ok(status) &&
        admission == LOOM_SERVE_RESIDENCY_ADMISSION_ACQUIRED) {
      activation->frontier =
          loom_serve_residency_frontier(activation->residency);
      iree_async_semaphore_wait_operation_t* wait = &activation->wait;
      iree_async_operation_initialize(
          &wait->base, IREE_ASYNC_OPERATION_TYPE_SEMAPHORE_WAIT,
          IREE_ASYNC_OPERATION_FLAG_NONE, activation_ready, activation);
      for (iree_host_size_t i = 0; i < 2; ++i) {
        activation->semaphores[i] =
            (iree_async_semaphore_t*)activation->frontier.semaphores[i];
      }
      wait->semaphores = activation->semaphores;
      wait->values = activation->frontier.values;
      wait->count = 2;
      wait->mode = IREE_ASYNC_WAIT_MODE_ALL;
      status = iree_async_proactor_submit_one(
          iree_async_proactor_pool_entry_proactor(activation->entry),
          &wait->base);
      activation->phase = LOOM_SERVE_ACTIVATION_LOADING;
    }
    if (!iree_status_is_ok(status)) {
      // Rejected submission owes no callback. Accepted load protection remains
      // on the residency until the model's actual buffer borrowers retire.
      loom_serve_residency_cancel_admission(activation->residency);
      activation->phase = LOOM_SERVE_ACTIVATION_COMPLETE;
    }
  }
  *out_phase = activation->phase;
  return status;
}

void loom_serve_activation_destroy(loom_serve_activation_t* activation) {
  IREE_ASSERT(activation->phase == LOOM_SERVE_ACTIVATION_COMPLETE);
  iree_async_proactor_pool_entry_release(activation->entry);
  iree_slim_mutex_deinitialize(&activation->event.mutex);
  iree_allocator_free(activation->allocator, activation);
}
