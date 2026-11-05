// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/runtime/parameter/activation.h"

#include "iree/async/operations/semaphore.h"
#include "iree/base/threading/mutex.h"

struct loom_serve_activation_t {
  // Allocator owning this cold operation, released after callback retirement.
  iree_allocator_t host_allocator;
  // Complete immutable membership in the shared owner's admission order.
  iree_host_size_t residency_count;
  // Coallocated array of borrowed residencies outliving this operation.
  loom_serve_residency_t** residencies;
  // Retained proactor progress ownership, shared with model and network I/O.
  iree_async_proactor_pool_entry_t* entry;
  // Owner-only phase; callback publication cannot alter admission or pins.
  loom_serve_activation_phase_t phase;
  // Accepted readiness operation, reused only by creating a new warmup.
  iree_async_semaphore_wait_operation_t wait;
  // Coallocated array of borrowed work/feedback handles, two per residency.
  iree_async_semaphore_t** semaphores;
  // Exact accepted activation frontiers in the same order as semaphores.
  uint64_t* values;
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

static void loom_serve_activation_ready(void* user_data,
                                        iree_async_operation_t* operation,
                                        iree_status_t status,
                                        iree_async_completion_flags_t flags) {
  loom_serve_activation_t* activation = user_data;
  iree_slim_mutex_lock(&activation->event.mutex);
  activation->event.status = status;
  activation->event.ready = true;
  if (activation->event.wake) {
    activation->event.wake(activation->event.user_data);
  }
  iree_slim_mutex_unlock(&activation->event.mutex);
}

iree_status_t loom_serve_activation_allocate(
    iree_host_size_t residency_count,
    loom_serve_residency_t* const* residencies,
    iree_async_proactor_pool_t* pool, void (*wake)(void* user_data),
    void* user_data, iree_allocator_t host_allocator,
    loom_serve_activation_t** out_activation) {
  *out_activation = NULL;
  iree_host_size_t total = 0, residencies_offset = 0, semaphores_offset = 0;
  iree_host_size_t values_offset = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      sizeof(loom_serve_activation_t), &total,
      IREE_STRUCT_FIELD(residency_count, loom_serve_residency_t*,
                        &residencies_offset),
      IREE_STRUCT_ARRAY_FIELD(residency_count, 2, iree_async_semaphore_t*,
                              &semaphores_offset),
      IREE_STRUCT_ARRAY_FIELD_ALIGNED(residency_count, 2, uint64_t,
                                      iree_alignof(uint64_t), &values_offset)));
  loom_serve_activation_t* activation = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, total, (void**)&activation));
  activation->host_allocator = host_allocator;
  activation->residency_count = residency_count;
  activation->residencies =
      (loom_serve_residency_t**)((uint8_t*)activation + residencies_offset);
  activation->semaphores =
      (iree_async_semaphore_t**)((uint8_t*)activation + semaphores_offset);
  activation->values = (uint64_t*)((uint8_t*)activation + values_offset);
  for (iree_host_size_t i = 0; i < residency_count; ++i) {
    activation->residencies[i] = residencies[i];
  }
  activation->event.wake = wake;
  activation->event.user_data = user_data;
  iree_status_t status =
      iree_async_proactor_pool_acquire(pool, 0, &activation->entry);
  if (iree_status_is_ok(status)) {
    iree_slim_mutex_initialize(&activation->event.mutex);
    *out_activation = activation;
  } else {
    iree_allocator_free(host_allocator, activation);
  }
  return status;
}

static void loom_serve_activation_cancel_admission(
    loom_serve_activation_t* activation) {
  for (iree_host_size_t i = 0; i < activation->residency_count; ++i) {
    loom_serve_residency_cancel_admission(activation->residencies[i]);
  }
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
        for (iree_host_size_t i = 0; i < activation->residency_count; ++i) {
          loom_serve_residency_release(activation->residencies[i]);
        }
      }
      activation->phase = LOOM_SERVE_ACTIVATION_COMPLETE;
    }
  } else if (iree_any_bit_set(flags, LOOM_SERVE_ACTIVATION_FLAG_CANCEL)) {
    loom_serve_activation_cancel_admission(activation);
    activation->phase = LOOM_SERVE_ACTIVATION_COMPLETE;
  } else {
    loom_serve_residency_admission_t admission;
    status = loom_serve_residency_try_admit(
        activation->residency_count, activation->residencies, &admission);
    if (iree_status_is_ok(status) &&
        admission == LOOM_SERVE_RESIDENCY_ADMISSION_ACQUIRED) {
      loom_serve_activation_cancel_admission(activation);
      iree_async_semaphore_wait_operation_t* wait = &activation->wait;
      iree_async_operation_initialize(&wait->base,
                                      IREE_ASYNC_OPERATION_TYPE_SEMAPHORE_WAIT,
                                      IREE_ASYNC_OPERATION_FLAG_NONE,
                                      loom_serve_activation_ready, activation);
      for (iree_host_size_t i = 0; i < activation->residency_count; ++i) {
        const loom_serve_execution_frontier_t frontier =
            loom_serve_residency_frontier(activation->residencies[i]);
        for (iree_host_size_t j = 0; j < 2; ++j) {
          activation->semaphores[2 * i + j] =
              (iree_async_semaphore_t*)frontier.semaphores[j];
          activation->values[2 * i + j] = frontier.values[j];
        }
      }
      wait->semaphores = activation->semaphores;
      wait->values = activation->values;
      wait->count = 2 * activation->residency_count;
      wait->mode = IREE_ASYNC_WAIT_MODE_ALL;
      status = iree_async_proactor_submit_one(
          iree_async_proactor_pool_entry_proactor(activation->entry),
          &wait->base);
      activation->phase = LOOM_SERVE_ACTIVATION_LOADING;
    }
    if (!iree_status_is_ok(status)) {
      // Rejected submission owes no callback. Accepted load protection remains
      // on the residency until the model's actual buffer borrowers retire.
      loom_serve_activation_cancel_admission(activation);
      activation->phase = LOOM_SERVE_ACTIVATION_COMPLETE;
    }
  }
  *out_phase = activation->phase;
  return status;
}

void loom_serve_activation_set_observer(loom_serve_activation_t* activation,
                                        void (*wake)(void* user_data),
                                        void* user_data) {
  if (!activation) {
    return;
  }
  iree_slim_mutex_lock(&activation->event.mutex);
  activation->event.wake = wake;
  activation->event.user_data = user_data;
  if (wake && activation->event.ready) {
    wake(user_data);
  }
  iree_slim_mutex_unlock(&activation->event.mutex);
}

void loom_serve_activation_free(loom_serve_activation_t* activation) {
  IREE_ASSERT(activation->phase == LOOM_SERVE_ACTIVATION_COMPLETE);
  iree_async_proactor_pool_entry_release(activation->entry);
  iree_slim_mutex_deinitialize(&activation->event.mutex);
  iree_allocator_free(activation->host_allocator, activation);
}
