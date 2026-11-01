// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/gpu/queue/semaphore.h"

#include "iree/base/threading/processor.h"

typedef struct iree_hal_amd_gpu_submission_semaphore_t {
  // Producer availability, independent of the completed timeline and frontier.
  iree_async_semaphore_t async;
  // Shared allocation, released after the public owner and all view users join.
  void* allocation;
} iree_hal_amd_gpu_submission_semaphore_t;

typedef struct iree_hal_amd_gpu_semaphore_t {
  // Completed timeline, frontier, failure, and asynchronous host waits.
  iree_async_semaphore_t async;
  // Highest published native producer, protected by async.mutex.
  struct {
    // Timeline value covered by the cached producer, or zero when empty.
    uint64_t value;
    // Owned reference, independent of producer and consumer executions.
    iree_hal_amd_gpu_signal_t* signal;
  } submitted;
  // Owned availability view constructed in this object's trailing allocation.
  iree_hal_semaphore_t* submission_semaphore;
  // Enables availability publication after first acquisition, under
  // async.mutex.
  bool submission_observed;
  // Optional immutable device-owned policy, borrowed only by active waits.
  const iree_hal_amd_gpu_semaphore_wait_context_t* wait_context;
} iree_hal_amd_gpu_semaphore_t;

static const iree_hal_semaphore_vtable_t iree_hal_amd_gpu_semaphore_vtable;
static const iree_hal_semaphore_vtable_t
    iree_hal_amd_gpu_submission_semaphore_vtable;

static iree_hal_amd_gpu_semaphore_t* iree_hal_amd_gpu_semaphore_cast(
    iree_hal_semaphore_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_amd_gpu_semaphore_vtable);
  return (iree_hal_amd_gpu_semaphore_t*)base_value;
}

iree_status_t iree_hal_amd_gpu_semaphore_create(
    iree_async_proactor_t* proactor,
    const iree_hal_amd_gpu_semaphore_wait_context_t* wait_context,
    uint64_t initial_value, uint8_t frontier_capacity,
    iree_allocator_t host_allocator, iree_hal_semaphore_t** out_semaphore) {
  *out_semaphore = NULL;
  iree_host_size_t frontier_offset = 0;
  iree_host_size_t total_size = 0;
  IREE_RETURN_IF_ERROR(iree_async_semaphore_layout(
      sizeof(iree_hal_amd_gpu_semaphore_t), frontier_capacity, &frontier_offset,
      &total_size));
  const iree_host_size_t submission_offset =
      iree_host_align(total_size, iree_max_align_t);
  iree_host_size_t submission_frontier_offset = 0;
  iree_host_size_t submission_size = 0;
  IREE_RETURN_IF_ERROR(iree_async_semaphore_layout(
      sizeof(iree_hal_amd_gpu_submission_semaphore_t), 0,
      &submission_frontier_offset, &submission_size));
  iree_hal_amd_gpu_semaphore_t* semaphore = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      host_allocator, submission_offset + submission_size, (void**)&semaphore));
  iree_async_semaphore_initialize(
      &iree_hal_amd_gpu_semaphore_vtable.async, proactor, initial_value,
      frontier_offset, frontier_capacity, host_allocator, &semaphore->async);
  iree_hal_amd_gpu_submission_semaphore_t* submission =
      (void*)((uint8_t*)semaphore + submission_offset);
  iree_async_semaphore_initialize(
      &iree_hal_amd_gpu_submission_semaphore_vtable.async, proactor,
      initial_value, submission_frontier_offset, 0, host_allocator,
      &submission->async);
  submission->allocation = semaphore;
  semaphore->submission_semaphore = iree_hal_semaphore_cast(&submission->async);
  semaphore->wait_context = wait_context;
  *out_semaphore = iree_hal_semaphore_cast(&semaphore->async);
  return iree_ok_status();
}

static void iree_hal_amd_gpu_semaphore_destroy(
    iree_async_semaphore_t* base_semaphore) {
  iree_hal_amd_gpu_semaphore_t* semaphore =
      (iree_hal_amd_gpu_semaphore_t*)base_semaphore;
  iree_hal_amd_gpu_signal_release(semaphore->submitted.signal);
  iree_async_semaphore_deinitialize(&semaphore->async);
  // The availability view may outlive the public owner. Its last reference
  // releases the common allocation; no parent access follows this release.
  iree_hal_semaphore_release(semaphore->submission_semaphore);
}

static void iree_hal_amd_gpu_submission_semaphore_destroy(
    iree_async_semaphore_t* base_semaphore) {
  iree_hal_amd_gpu_submission_semaphore_t* semaphore =
      (iree_hal_amd_gpu_submission_semaphore_t*)base_semaphore;
  const iree_allocator_t host_allocator = semaphore->async.allocator;
  void* allocation = semaphore->allocation;
  iree_async_semaphore_deinitialize(&semaphore->async);
  iree_allocator_free(host_allocator, allocation);
}

bool iree_hal_amd_gpu_semaphore_isa(iree_hal_semaphore_t* semaphore) {
  return iree_hal_resource_is((const iree_hal_resource_t*)semaphore,
                              &iree_hal_amd_gpu_semaphore_vtable);
}

static uint64_t iree_hal_amd_gpu_semaphore_query(
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
// The enclosing call retains the parent, so its availability view can be
// borrowed after unlocking. Cell release and callback dispatch never hold this
// mutex.
static iree_status_t iree_hal_amd_gpu_semaphore_update_submission(
    iree_hal_amd_gpu_semaphore_t* semaphore) {
  iree_slim_mutex_lock(&semaphore->async.mutex);
  uint64_t value = (uint64_t)iree_atomic_load(&semaphore->async.timeline_value,
                                              iree_memory_order_acquire);
  iree_status_t failure = (iree_status_t)iree_atomic_load(
      &semaphore->async.failure_status, iree_memory_order_acquire);
  iree_hal_amd_gpu_signal_t* released_signal = NULL;
  if (value >= semaphore->submitted.value) {
    released_signal = semaphore->submitted.signal;
    semaphore->submitted.value = 0;
    semaphore->submitted.signal = NULL;
  } else {
    value = semaphore->submitted.value;
  }
  iree_hal_semaphore_t* submission_semaphore =
      semaphore->submission_observed ? semaphore->submission_semaphore : NULL;
  iree_slim_mutex_unlock(&semaphore->async.mutex);
  iree_hal_amd_gpu_signal_release(released_signal);
  if (IREE_UNLIKELY(!iree_status_is_ok(failure))) {
    return iree_status_clone(failure);
  }
  if (submission_semaphore) {
    return iree_async_semaphore_publish_untainted(
        (iree_async_semaphore_t*)submission_semaphore, value, NULL);
  }
  return iree_ok_status();
}

iree_status_t iree_hal_amd_gpu_semaphore_acquire_wait(
    iree_hal_semaphore_t* base_semaphore, uint64_t minimum_value,
    const void* domain, iree_hal_amd_gpu_semaphore_wait_t* out_wait) {
  *out_wait = (iree_hal_amd_gpu_semaphore_wait_t){0};
  iree_hal_amd_gpu_semaphore_t* semaphore =
      iree_hal_amd_gpu_semaphore_cast(base_semaphore);
  iree_status_t failure = (iree_status_t)iree_atomic_load(
      &semaphore->async.failure_status, iree_memory_order_acquire);
  if (IREE_UNLIKELY(!iree_status_is_ok(failure))) {
    return iree_status_clone(failure);
  }
  if ((uint64_t)iree_atomic_load(&semaphore->async.timeline_value,
                                 iree_memory_order_acquire) >= minimum_value) {
    out_wait->state = IREE_HAL_AMD_GPU_SEMAPHORE_WAIT_COMPLETED;
    return iree_ok_status();
  }
  // Only an unresolved wait can borrow replaceable native producer storage.
  // Recheck completion and failure while acquiring that storage's reference.
  iree_slim_mutex_lock(&semaphore->async.mutex);
  failure = (iree_status_t)iree_atomic_load(&semaphore->async.failure_status,
                                            iree_memory_order_acquire);
  if (iree_status_is_ok(failure)) {
    const uint64_t current_value = (uint64_t)iree_atomic_load(
        &semaphore->async.timeline_value, iree_memory_order_acquire);
    if (current_value >= minimum_value) {
      out_wait->state = IREE_HAL_AMD_GPU_SEMAPHORE_WAIT_COMPLETED;
    } else if (semaphore->submitted.value >= minimum_value) {
      if (semaphore->submitted.signal->block->domain == domain) {
        out_wait->state = IREE_HAL_AMD_GPU_SEMAPHORE_WAIT_NATIVE;
        out_wait->signal = semaphore->submitted.signal;
        iree_hal_amd_gpu_signal_retain(out_wait->signal);
      } else {
        out_wait->state = IREE_HAL_AMD_GPU_SEMAPHORE_WAIT_HOST;
      }
    }
  }
  iree_slim_mutex_unlock(&semaphore->async.mutex);
  return iree_status_clone(failure);
}

iree_status_t iree_hal_amd_gpu_semaphore_publish_signal(
    iree_hal_semaphore_t* base_semaphore, uint64_t value,
    iree_hal_amd_gpu_signal_t* signal) {
  iree_hal_amd_gpu_semaphore_t* semaphore =
      iree_hal_amd_gpu_semaphore_cast(base_semaphore);
  iree_hal_amd_gpu_signal_t* released_signal = NULL;
  iree_slim_mutex_lock(&semaphore->async.mutex);
  const uint64_t current_value = (uint64_t)iree_atomic_load(
      &semaphore->async.timeline_value, iree_memory_order_acquire);
  iree_status_t failure = (iree_status_t)iree_atomic_load(
      &semaphore->async.failure_status, iree_memory_order_acquire);
  if (iree_status_is_ok(failure) && value > current_value &&
      value > semaphore->submitted.value) {
    iree_hal_amd_gpu_signal_retain(signal);
    released_signal = semaphore->submitted.signal;
    semaphore->submitted.value = value;
    semaphore->submitted.signal = signal;
  }
  iree_hal_semaphore_t* submission_semaphore =
      semaphore->submission_observed ? semaphore->submission_semaphore : NULL;
  iree_slim_mutex_unlock(&semaphore->async.mutex);
  iree_hal_amd_gpu_signal_release(released_signal);
  if (IREE_UNLIKELY(!iree_status_is_ok(failure))) {
    return iree_status_clone(failure);
  }
  if (submission_semaphore) {
    return iree_async_semaphore_publish_untainted(
        (iree_async_semaphore_t*)submission_semaphore, value, NULL);
  }
  return iree_ok_status();
}

iree_status_t iree_hal_amd_gpu_semaphore_publish_completed(
    iree_hal_semaphore_t* base_semaphore, uint64_t value,
    const iree_async_frontier_t* frontier) {
  iree_hal_amd_gpu_semaphore_t* semaphore =
      iree_hal_amd_gpu_semaphore_cast(base_semaphore);
  IREE_RETURN_IF_ERROR(iree_async_semaphore_publish_untainted(&semaphore->async,
                                                              value, frontier));
  return iree_hal_amd_gpu_semaphore_update_submission(semaphore);
}

iree_status_t iree_hal_amd_gpu_semaphore_acquire_submission_semaphore(
    iree_hal_semaphore_t* base_semaphore,
    iree_hal_semaphore_t** out_semaphore) {
  *out_semaphore = NULL;
  iree_hal_amd_gpu_semaphore_t* semaphore =
      iree_hal_amd_gpu_semaphore_cast(base_semaphore);
  iree_slim_mutex_lock(&semaphore->async.mutex);
  iree_hal_semaphore_t* submission_semaphore = semaphore->submission_semaphore;
  iree_status_t failure = (iree_status_t)iree_atomic_load(
      &semaphore->async.failure_status, iree_memory_order_acquire);
  semaphore->submission_observed = true;
  iree_slim_mutex_unlock(&semaphore->async.mutex);
  if (IREE_UNLIKELY(!iree_status_is_ok(failure))) {
    return iree_status_clone(failure);
  }
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_semaphore_update_submission(semaphore));
  iree_hal_semaphore_retain(submission_semaphore);
  *out_semaphore = submission_semaphore;
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_gpu_semaphore_signal(
    iree_async_semaphore_t* base_semaphore, uint64_t value,
    const iree_async_frontier_t* frontier) {
  IREE_RETURN_IF_ERROR(
      iree_async_semaphore_advance_timeline(base_semaphore, value, frontier));
  iree_status_t status = iree_hal_amd_gpu_semaphore_update_submission(
      (iree_hal_amd_gpu_semaphore_t*)base_semaphore);
  iree_async_semaphore_dispatch_timepoints(base_semaphore, value);
  return status;
}

static iree_status_t iree_hal_amd_gpu_submission_semaphore_signal(
    iree_async_semaphore_t* semaphore, uint64_t value,
    const iree_async_frontier_t* frontier) {
  IREE_RETURN_IF_ERROR(
      iree_async_semaphore_advance_timeline(semaphore, value, frontier));
  iree_async_semaphore_dispatch_timepoints(semaphore, value);
  return iree_ok_status();
}

static void iree_hal_amd_gpu_semaphore_on_fail(
    iree_async_semaphore_t* base_semaphore, iree_status_code_t status_code) {
  (void)status_code;
  iree_hal_amd_gpu_semaphore_t* semaphore =
      (iree_hal_amd_gpu_semaphore_t*)base_semaphore;
  iree_slim_mutex_lock(&semaphore->async.mutex);
  iree_hal_amd_gpu_signal_t* released_signal = semaphore->submitted.signal;
  semaphore->submitted.value = 0;
  semaphore->submitted.signal = NULL;
  iree_hal_semaphore_t* submission_semaphore = semaphore->submission_semaphore;
  iree_slim_mutex_unlock(&semaphore->async.mutex);
  iree_hal_amd_gpu_signal_release(released_signal);
  if (submission_semaphore) {
    iree_status_t failure = (iree_status_t)iree_atomic_load(
        &semaphore->async.failure_status, iree_memory_order_acquire);
    iree_hal_semaphore_fail(submission_semaphore, iree_status_clone(failure));
  }
}

// A later cached producer may require host work performed after this wait.
// Only the exact requested value can accelerate a synchronous host wait.
static iree_hal_amd_gpu_signal_t*
iree_hal_amd_gpu_semaphore_acquire_host_wait_signal(
    iree_hal_amd_gpu_semaphore_t* semaphore, uint64_t value) {
  iree_hal_amd_gpu_signal_t* signal = NULL;
  iree_slim_mutex_lock(&semaphore->async.mutex);
  if (semaphore->submitted.value == value && semaphore->submitted.signal &&
      semaphore->submitted.signal->block->domain ==
          semaphore->wait_context->domain) {
    signal = semaphore->submitted.signal;
    iree_hal_amd_gpu_signal_retain(signal);
  }
  iree_slim_mutex_unlock(&semaphore->async.mutex);
  return signal;
}

static iree_status_t iree_hal_amd_gpu_semaphore_wait(
    iree_hal_semaphore_t* base_semaphore, uint64_t value,
    iree_timeout_t timeout, iree_async_wait_flags_t flags) {
  iree_hal_amd_gpu_semaphore_t* semaphore =
      iree_hal_amd_gpu_semaphore_cast(base_semaphore);
  const iree_hal_amd_gpu_semaphore_wait_context_t* context =
      semaphore->wait_context;
  if (context && iree_any_bit_set(flags, IREE_ASYNC_WAIT_FLAG_ACTIVE) &&
      !iree_timeout_is_immediate(timeout) &&
      iree_hal_amd_gpu_semaphore_query(&semaphore->async) < value) {
    iree_convert_timeout_to_absolute(&timeout);
    iree_hal_amd_gpu_signal_t* signal =
        iree_hal_amd_gpu_semaphore_acquire_host_wait_signal(semaphore, value);
    if (signal) {
      while (iree_hal_amd_gpu_semaphore_query(&semaphore->async) < value) {
        if ((uint64_t)iree_atomic_load(
                (const iree_atomic_int64_t*)signal->host_value,
                iree_memory_order_acquire) != context->pending_value) {
          context->try_retire(context->user_data);
          break;
        }
        if (timeout.nanos != IREE_TIME_INFINITE_FUTURE &&
            iree_time_now() >= timeout.nanos) {
          break;
        }
        iree_processor_yield();
      }
      iree_hal_amd_gpu_signal_release(signal);
    }
  }
  // Only retirement publishes public success, after host ownership and shader
  // reports join. Preserve the same deadline if native observation used time.
  return iree_async_semaphore_multi_wait(
      IREE_ASYNC_WAIT_MODE_ALL, (iree_async_semaphore_t**)&base_semaphore,
      &value, 1, timeout, flags, iree_allocator_system());
}

static const iree_hal_semaphore_vtable_t iree_hal_amd_gpu_semaphore_vtable = {
    .async =
        {
            .destroy = iree_hal_amd_gpu_semaphore_destroy,
            .query = iree_hal_amd_gpu_semaphore_query,
            .signal = iree_hal_amd_gpu_semaphore_signal,
            .on_fail = iree_hal_amd_gpu_semaphore_on_fail,
        },
    .wait = iree_hal_amd_gpu_semaphore_wait,
};

static iree_status_t iree_hal_amd_gpu_submission_semaphore_wait(
    iree_hal_semaphore_t* semaphore, uint64_t value, iree_timeout_t timeout,
    iree_async_wait_flags_t flags) {
  return iree_async_semaphore_multi_wait(
      IREE_ASYNC_WAIT_MODE_ALL, (iree_async_semaphore_t**)&semaphore, &value, 1,
      timeout, flags, iree_allocator_system());
}

static const iree_hal_semaphore_vtable_t
    iree_hal_amd_gpu_submission_semaphore_vtable = {
        .async =
            {
                .destroy = iree_hal_amd_gpu_submission_semaphore_destroy,
                .query = iree_hal_amd_gpu_semaphore_query,
                .signal = iree_hal_amd_gpu_submission_semaphore_signal,
            },
        .wait = iree_hal_amd_gpu_submission_semaphore_wait,
};
