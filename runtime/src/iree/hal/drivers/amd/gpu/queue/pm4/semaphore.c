// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/gpu/queue/pm4/semaphore.h"

#include "iree/hal/utils/host_semaphore.h"

typedef struct iree_hal_amd_gpu_pm4_semaphore_t {
  // Completed timeline, frontier, failure, and asynchronous host waits.
  iree_async_semaphore_t async;
  // Highest published native producer, protected by async.mutex.
  struct {
    // Timeline value covered by the cached producer, or zero when empty.
    uint64_t value;
    // Owned reference, independent of producer and consumer executions.
    iree_hal_amd_gpu_pm4_signal_t* signal;
    // Exact producer coordinate, separate from completed frontier storage.
    iree_async_frontier_entry_t producer;
  } submitted;
  // Optional owned host semaphore, installed once under async.mutex.
  iree_hal_semaphore_t* submission_semaphore;
  // Allocator for this semaphore and its cold submission wait state.
  iree_allocator_t host_allocator;
} iree_hal_amd_gpu_pm4_semaphore_t;

static const iree_hal_semaphore_vtable_t iree_hal_amd_gpu_pm4_semaphore_vtable;

static iree_hal_amd_gpu_pm4_semaphore_t* iree_hal_amd_gpu_pm4_semaphore_cast(
    iree_hal_semaphore_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_amd_gpu_pm4_semaphore_vtable);
  return (iree_hal_amd_gpu_pm4_semaphore_t*)base_value;
}

iree_status_t iree_hal_amd_gpu_pm4_semaphore_create(
    iree_async_proactor_t* proactor, uint64_t initial_value,
    uint8_t frontier_capacity, iree_allocator_t host_allocator,
    iree_hal_semaphore_t** out_semaphore) {
  *out_semaphore = NULL;
  iree_host_size_t frontier_offset = 0;
  iree_host_size_t total_size = 0;
  IREE_RETURN_IF_ERROR(iree_async_semaphore_layout(
      sizeof(iree_hal_amd_gpu_pm4_semaphore_t), frontier_capacity,
      &frontier_offset, &total_size));
  iree_hal_amd_gpu_pm4_semaphore_t* semaphore = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, total_size, (void**)&semaphore));
  iree_async_semaphore_initialize(&iree_hal_amd_gpu_pm4_semaphore_vtable.async,
                                  proactor, initial_value, frontier_offset,
                                  frontier_capacity, &semaphore->async);
  semaphore->host_allocator = host_allocator;
  *out_semaphore = iree_hal_semaphore_cast(&semaphore->async);
  return iree_ok_status();
}

static void iree_hal_amd_gpu_pm4_semaphore_destroy(
    iree_async_semaphore_t* base_semaphore) {
  iree_hal_amd_gpu_pm4_semaphore_t* semaphore =
      (iree_hal_amd_gpu_pm4_semaphore_t*)base_semaphore;
  iree_allocator_t host_allocator = semaphore->host_allocator;
  iree_hal_amd_gpu_pm4_signal_release(semaphore->submitted.signal);
  iree_hal_semaphore_release(semaphore->submission_semaphore);
  iree_async_semaphore_deinitialize(&semaphore->async);
  iree_allocator_free(host_allocator, semaphore);
}

bool iree_hal_amd_gpu_pm4_semaphore_isa(iree_hal_semaphore_t* semaphore) {
  return iree_hal_resource_is((const iree_hal_resource_t*)semaphore,
                              &iree_hal_amd_gpu_pm4_semaphore_vtable);
}

static uint64_t iree_hal_amd_gpu_pm4_semaphore_query(
    iree_async_semaphore_t* semaphore) {
  iree_status_t failure = (iree_status_t)iree_atomic_load(
      &semaphore->failure_status, iree_memory_order_acquire);
  if (IREE_UNLIKELY(!iree_status_is_ok(failure))) {
    return iree_hal_status_as_semaphore_failure(failure);
  }
  return (uint64_t)iree_atomic_load(&semaphore->timeline_value,
                                    iree_memory_order_acquire);
}

// Returns covered cache ownership before delivering submission notifications.
// The enclosing call retains the parent, so its installed child can be borrowed
// after unlocking. Cell release and callback dispatch never hold this mutex.
static iree_status_t iree_hal_amd_gpu_pm4_semaphore_update_submission(
    iree_hal_amd_gpu_pm4_semaphore_t* semaphore) {
  iree_slim_mutex_lock(&semaphore->async.mutex);
  uint64_t value = (uint64_t)iree_atomic_load(&semaphore->async.timeline_value,
                                              iree_memory_order_acquire);
  iree_status_t failure = (iree_status_t)iree_atomic_load(
      &semaphore->async.failure_status, iree_memory_order_acquire);
  iree_hal_amd_gpu_pm4_signal_t* released_signal = NULL;
  if (value >= semaphore->submitted.value) {
    released_signal = semaphore->submitted.signal;
    semaphore->submitted.value = 0;
    semaphore->submitted.signal = NULL;
  } else {
    value = semaphore->submitted.value;
  }
  iree_hal_semaphore_t* submission_semaphore = semaphore->submission_semaphore;
  iree_slim_mutex_unlock(&semaphore->async.mutex);
  iree_hal_amd_gpu_pm4_signal_release(released_signal);
  if (IREE_UNLIKELY(!iree_status_is_ok(failure))) {
    return iree_status_clone(failure);
  }
  if (submission_semaphore) {
    return iree_async_semaphore_publish_untainted(
        (iree_async_semaphore_t*)submission_semaphore, value, NULL);
  }
  return iree_ok_status();
}

iree_status_t iree_hal_amd_gpu_pm4_semaphore_acquire_wait(
    iree_hal_semaphore_t* base_semaphore, uint64_t minimum_value,
    iree_hal_amd_gpu_pm4_semaphore_wait_t* out_wait) {
  *out_wait = (iree_hal_amd_gpu_pm4_semaphore_wait_t){0};
  iree_hal_amd_gpu_pm4_semaphore_t* semaphore =
      iree_hal_amd_gpu_pm4_semaphore_cast(base_semaphore);
  iree_slim_mutex_lock(&semaphore->async.mutex);
  iree_status_t failure = (iree_status_t)iree_atomic_load(
      &semaphore->async.failure_status, iree_memory_order_acquire);
  if (iree_status_is_ok(failure)) {
    const uint64_t current_value = (uint64_t)iree_atomic_load(
        &semaphore->async.timeline_value, iree_memory_order_acquire);
    if (current_value >= minimum_value) {
      out_wait->state = IREE_HAL_AMD_GPU_PM4_SEMAPHORE_WAIT_COMPLETED;
    } else if (semaphore->submitted.value >= minimum_value) {
      out_wait->state = IREE_HAL_AMD_GPU_PM4_SEMAPHORE_WAIT_NATIVE;
      out_wait->signal = semaphore->submitted.signal;
      out_wait->producer = semaphore->submitted.producer;
      iree_hal_amd_gpu_pm4_signal_retain(out_wait->signal);
    }
  }
  iree_slim_mutex_unlock(&semaphore->async.mutex);
  return iree_status_clone(failure);
}

iree_status_t iree_hal_amd_gpu_pm4_semaphore_publish_signal(
    iree_hal_semaphore_t* base_semaphore, uint64_t value,
    iree_async_frontier_entry_t producer,
    iree_hal_amd_gpu_pm4_signal_t* signal) {
  iree_hal_amd_gpu_pm4_semaphore_t* semaphore =
      iree_hal_amd_gpu_pm4_semaphore_cast(base_semaphore);
  iree_hal_amd_gpu_pm4_signal_t* released_signal = NULL;
  iree_slim_mutex_lock(&semaphore->async.mutex);
  const uint64_t current_value = (uint64_t)iree_atomic_load(
      &semaphore->async.timeline_value, iree_memory_order_acquire);
  iree_status_t failure = (iree_status_t)iree_atomic_load(
      &semaphore->async.failure_status, iree_memory_order_acquire);
  if (iree_status_is_ok(failure) && value > current_value &&
      value > semaphore->submitted.value) {
    iree_hal_amd_gpu_pm4_signal_retain(signal);
    released_signal = semaphore->submitted.signal;
    semaphore->submitted.value = value;
    semaphore->submitted.signal = signal;
    semaphore->submitted.producer = producer;
  }
  iree_hal_semaphore_t* submission_semaphore = semaphore->submission_semaphore;
  iree_slim_mutex_unlock(&semaphore->async.mutex);
  iree_hal_amd_gpu_pm4_signal_release(released_signal);
  if (IREE_UNLIKELY(!iree_status_is_ok(failure))) {
    return iree_status_clone(failure);
  }
  if (submission_semaphore) {
    return iree_async_semaphore_publish_untainted(
        (iree_async_semaphore_t*)submission_semaphore, value, NULL);
  }
  return iree_ok_status();
}

iree_status_t iree_hal_amd_gpu_pm4_semaphore_publish_completed(
    iree_hal_semaphore_t* base_semaphore, uint64_t value,
    const iree_async_frontier_t* frontier) {
  iree_hal_amd_gpu_pm4_semaphore_t* semaphore =
      iree_hal_amd_gpu_pm4_semaphore_cast(base_semaphore);
  IREE_RETURN_IF_ERROR(iree_async_semaphore_publish_untainted(&semaphore->async,
                                                              value, frontier));
  return iree_hal_amd_gpu_pm4_semaphore_update_submission(semaphore);
}

iree_status_t iree_hal_amd_gpu_pm4_semaphore_acquire_submission_semaphore(
    iree_hal_semaphore_t* base_semaphore,
    iree_hal_semaphore_t** out_semaphore) {
  *out_semaphore = NULL;
  iree_hal_amd_gpu_pm4_semaphore_t* semaphore =
      iree_hal_amd_gpu_pm4_semaphore_cast(base_semaphore);
  iree_slim_mutex_lock(&semaphore->async.mutex);
  iree_hal_semaphore_t* submission_semaphore = semaphore->submission_semaphore;
  iree_status_t failure = (iree_status_t)iree_atomic_load(
      &semaphore->async.failure_status, iree_memory_order_acquire);
  iree_slim_mutex_unlock(&semaphore->async.mutex);
  if (IREE_UNLIKELY(!iree_status_is_ok(failure))) {
    return iree_status_clone(failure);
  }
  if (!submission_semaphore) {
    iree_hal_semaphore_t* candidate = NULL;
    IREE_RETURN_IF_ERROR(iree_hal_host_semaphore_create(
        semaphore->async.proactor, 0, semaphore->host_allocator, &candidate));
    iree_slim_mutex_lock(&semaphore->async.mutex);
    if (!semaphore->submission_semaphore) {
      semaphore->submission_semaphore = candidate;
      candidate = NULL;
    }
    submission_semaphore = semaphore->submission_semaphore;
    iree_slim_mutex_unlock(&semaphore->async.mutex);
    iree_hal_semaphore_release(candidate);
  }
  IREE_RETURN_IF_ERROR(
      iree_hal_amd_gpu_pm4_semaphore_update_submission(semaphore));
  iree_hal_semaphore_retain(submission_semaphore);
  *out_semaphore = submission_semaphore;
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_gpu_pm4_semaphore_signal(
    iree_async_semaphore_t* base_semaphore, uint64_t value,
    const iree_async_frontier_t* frontier) {
  iree_status_t failure = (iree_status_t)iree_atomic_load(
      &base_semaphore->failure_status, iree_memory_order_acquire);
  if (IREE_UNLIKELY(!iree_status_is_ok(failure))) {
    return iree_status_clone(failure);
  }
  IREE_RETURN_IF_ERROR(
      iree_async_semaphore_advance_timeline(base_semaphore, value, frontier));
  iree_status_t status = iree_hal_amd_gpu_pm4_semaphore_update_submission(
      (iree_hal_amd_gpu_pm4_semaphore_t*)base_semaphore);
  iree_async_semaphore_dispatch_timepoints(base_semaphore, value);
  return status;
}

static void iree_hal_amd_gpu_pm4_semaphore_on_fail(
    iree_async_semaphore_t* base_semaphore, iree_status_code_t status_code) {
  (void)status_code;
  iree_hal_amd_gpu_pm4_semaphore_t* semaphore =
      (iree_hal_amd_gpu_pm4_semaphore_t*)base_semaphore;
  iree_slim_mutex_lock(&semaphore->async.mutex);
  iree_hal_amd_gpu_pm4_signal_t* released_signal = semaphore->submitted.signal;
  semaphore->submitted.value = 0;
  semaphore->submitted.signal = NULL;
  iree_hal_semaphore_t* submission_semaphore = semaphore->submission_semaphore;
  iree_slim_mutex_unlock(&semaphore->async.mutex);
  iree_hal_amd_gpu_pm4_signal_release(released_signal);
  if (submission_semaphore) {
    iree_status_t failure = (iree_status_t)iree_atomic_load(
        &semaphore->async.failure_status, iree_memory_order_acquire);
    iree_hal_semaphore_fail(submission_semaphore, iree_status_clone(failure));
  }
}

static iree_status_t iree_hal_amd_gpu_pm4_semaphore_wait(
    iree_hal_semaphore_t* semaphore, uint64_t value, iree_timeout_t timeout,
    iree_async_wait_flags_t flags) {
  return iree_async_semaphore_multi_wait(
      IREE_ASYNC_WAIT_MODE_ALL, (iree_async_semaphore_t**)&semaphore, &value, 1,
      timeout, flags, iree_allocator_system());
}

static const iree_hal_semaphore_vtable_t iree_hal_amd_gpu_pm4_semaphore_vtable =
    {
        .async =
            {
                .destroy = iree_hal_amd_gpu_pm4_semaphore_destroy,
                .query = iree_hal_amd_gpu_pm4_semaphore_query,
                .signal = iree_hal_amd_gpu_pm4_semaphore_signal,
                .on_fail = iree_hal_amd_gpu_pm4_semaphore_on_fail,
            },
        .wait = iree_hal_amd_gpu_pm4_semaphore_wait,
};
