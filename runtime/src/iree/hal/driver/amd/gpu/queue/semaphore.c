// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/driver/amd/gpu/queue/semaphore.h"

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
  // Highest native publication value; availability never regresses.
  uint64_t submitted_value;
  // Borrowed live producer index, protected by async.mutex.
  iree_hal_amd_gpu_semaphore_producer_t* producers;
  // Borrowed most recently published producer for the ordinary chain path.
  iree_hal_amd_gpu_semaphore_producer_t* latest_producer;
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
  IREE_ASSERT(!semaphore->producers,
              "producer captures retain their semaphore");
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

// Live producer storage belongs to captures, so publication cannot fail for
// index allocation and completion does not leave historical native cells
// pinned.
static uint32_t iree_hal_amd_gpu_producer_height(
    const iree_hal_amd_gpu_semaphore_producer_t* producer) {
  return producer ? producer->height : 0;
}

static void iree_hal_amd_gpu_producer_update(
    iree_hal_amd_gpu_semaphore_producer_t* producer) {
  producer->height =
      1 + iree_max(iree_hal_amd_gpu_producer_height(producer->left),
                   iree_hal_amd_gpu_producer_height(producer->right));
}

static void iree_hal_amd_gpu_producer_replace(
    iree_hal_amd_gpu_semaphore_t* semaphore,
    iree_hal_amd_gpu_semaphore_producer_t* previous,
    iree_hal_amd_gpu_semaphore_producer_t* replacement) {
  if (!previous->parent) {
    semaphore->producers = replacement;
  } else if (previous->parent->left == previous) {
    previous->parent->left = replacement;
  } else {
    previous->parent->right = replacement;
  }
  if (replacement) {
    replacement->parent = previous->parent;
  }
}

static iree_hal_amd_gpu_semaphore_producer_t* iree_hal_amd_gpu_producer_rotate(
    iree_hal_amd_gpu_semaphore_t* semaphore,
    iree_hal_amd_gpu_semaphore_producer_t* producer, bool left) {
  iree_hal_amd_gpu_semaphore_producer_t* next =
      left ? producer->right : producer->left;
  iree_hal_amd_gpu_producer_replace(semaphore, producer, next);
  if (left) {
    producer->right = next->left;
    if (producer->right) {
      producer->right->parent = producer;
    }
    next->left = producer;
  } else {
    producer->left = next->right;
    if (producer->left) {
      producer->left->parent = producer;
    }
    next->right = producer;
  }
  producer->parent = next;
  iree_hal_amd_gpu_producer_update(producer);
  iree_hal_amd_gpu_producer_update(next);
  return next;
}

static void iree_hal_amd_gpu_producer_balance(
    iree_hal_amd_gpu_semaphore_t* semaphore,
    iree_hal_amd_gpu_semaphore_producer_t* producer) {
  while (producer) {
    iree_hal_amd_gpu_producer_update(producer);
    const int32_t balance =
        (int32_t)iree_hal_amd_gpu_producer_height(producer->left) -
        (int32_t)iree_hal_amd_gpu_producer_height(producer->right);
    if (balance > 1) {
      if (iree_hal_amd_gpu_producer_height(producer->left->left) <
          iree_hal_amd_gpu_producer_height(producer->left->right)) {
        iree_hal_amd_gpu_producer_rotate(semaphore, producer->left, true);
      }
      producer = iree_hal_amd_gpu_producer_rotate(semaphore, producer, false);
    } else if (balance < -1) {
      if (iree_hal_amd_gpu_producer_height(producer->right->right) <
          iree_hal_amd_gpu_producer_height(producer->right->left)) {
        iree_hal_amd_gpu_producer_rotate(semaphore, producer->right, false);
      }
      producer = iree_hal_amd_gpu_producer_rotate(semaphore, producer, true);
    }
    producer = producer->parent;
  }
}

static iree_hal_amd_gpu_semaphore_producer_t* iree_hal_amd_gpu_producer_find(
    iree_hal_amd_gpu_semaphore_t* semaphore, uint64_t value) {
  iree_hal_amd_gpu_semaphore_producer_t* producer = semaphore->latest_producer;
  if (producer && producer->value == value) {
    return producer;
  }
  producer = semaphore->producers;
  while (producer && producer->value != value) {
    producer = value < producer->value ? producer->left : producer->right;
  }
  return producer;
}

static void iree_hal_amd_gpu_producer_insert(
    iree_hal_amd_gpu_semaphore_t* semaphore, uint64_t value,
    iree_hal_amd_gpu_signal_t* signal,
    iree_hal_amd_gpu_semaphore_producer_t* producer) {
  iree_hal_amd_gpu_semaphore_producer_t* parent = NULL;
  iree_hal_amd_gpu_semaphore_producer_t** link = &semaphore->producers;
  while (*link) {
    parent = *link;
    // Repeated output values share the first published native predicate.
    if (parent->value == value) {
      return;
    }
    link = value < parent->value ? &parent->left : &parent->right;
  }
  *producer = (iree_hal_amd_gpu_semaphore_producer_t){
      .parent = parent, .value = value, .signal = signal, .height = 1};
  *link = producer;
  semaphore->latest_producer = producer;
  iree_hal_amd_gpu_producer_balance(semaphore, parent);
}

void iree_hal_amd_gpu_semaphore_unpublish_signal(
    iree_hal_semaphore_t* base_semaphore,
    iree_hal_amd_gpu_semaphore_producer_t* producer) {
  iree_hal_amd_gpu_semaphore_t* semaphore =
      iree_hal_amd_gpu_semaphore_cast(base_semaphore);
  iree_slim_mutex_lock(&semaphore->async.mutex);
  if (producer->signal) {
    IREE_ASSERT((uint64_t)iree_atomic_load(&semaphore->async.timeline_value,
                                           iree_memory_order_acquire) >=
                        producer->value ||
                    iree_atomic_load(&semaphore->async.failure_status,
                                     iree_memory_order_acquire),
                "producer withdrawal requires public completion or failure");
    iree_hal_amd_gpu_semaphore_producer_t* parent = producer->parent;
    if (producer->left && producer->right) {
      iree_hal_amd_gpu_semaphore_producer_t* next = producer->right;
      while (next->left) {
        next = next->left;
      }
      if (next->parent != producer) {
        parent = next->parent;
        iree_hal_amd_gpu_producer_replace(semaphore, next, next->right);
        next->right = producer->right;
        next->right->parent = next;
      } else {
        parent = next;
      }
      iree_hal_amd_gpu_producer_replace(semaphore, producer, next);
      next->left = producer->left;
      next->left->parent = next;
      next->height = producer->height;
    } else {
      iree_hal_amd_gpu_producer_replace(
          semaphore, producer,
          producer->left ? producer->left : producer->right);
    }
    if (semaphore->latest_producer == producer) {
      semaphore->latest_producer = NULL;
    }
    producer->signal = NULL;
    iree_hal_amd_gpu_producer_balance(semaphore, parent);
  }
  iree_slim_mutex_unlock(&semaphore->async.mutex);
}

// The enclosing call retains the parent, so its availability view can be
// borrowed after unlocking. Callback dispatch never holds this mutex.
static iree_status_t iree_hal_amd_gpu_semaphore_update_submission(
    iree_hal_amd_gpu_semaphore_t* semaphore) {
  iree_slim_mutex_lock(&semaphore->async.mutex);
  uint64_t value = (uint64_t)iree_atomic_load(&semaphore->async.timeline_value,
                                              iree_memory_order_acquire);
  iree_status_t failure = (iree_status_t)iree_atomic_load(
      &semaphore->async.failure_status, iree_memory_order_acquire);
  value = iree_max(value, semaphore->submitted_value);
  iree_hal_semaphore_t* submission_semaphore =
      semaphore->submission_observed ? semaphore->submission_semaphore : NULL;
  iree_slim_mutex_unlock(&semaphore->async.mutex);
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
    } else if (semaphore->submitted_value >= minimum_value) {
      iree_hal_amd_gpu_semaphore_producer_t* producer =
          iree_hal_amd_gpu_producer_find(semaphore, minimum_value);
      if (producer && producer->signal->block->domain == domain) {
        out_wait->state = IREE_HAL_AMD_GPU_SEMAPHORE_WAIT_NATIVE;
        out_wait->signal = producer->signal;
        iree_hal_amd_gpu_signal_retain(out_wait->signal);
      } else {
        // A later producer may depend on this consumer. Missing exact values
        // and foreign native domains keep the original timeline predicate.
        out_wait->state = IREE_HAL_AMD_GPU_SEMAPHORE_WAIT_HOST;
      }
    }
  }
  iree_slim_mutex_unlock(&semaphore->async.mutex);
  return iree_status_clone(failure);
}

iree_status_t iree_hal_amd_gpu_semaphore_publish_signal(
    iree_hal_semaphore_t* base_semaphore, uint64_t value,
    iree_hal_amd_gpu_signal_t* signal,
    iree_hal_amd_gpu_semaphore_producer_t* producer) {
  iree_hal_amd_gpu_semaphore_t* semaphore =
      iree_hal_amd_gpu_semaphore_cast(base_semaphore);
  iree_slim_mutex_lock(&semaphore->async.mutex);
  const uint64_t current_value = (uint64_t)iree_atomic_load(
      &semaphore->async.timeline_value, iree_memory_order_acquire);
  iree_status_t failure = (iree_status_t)iree_atomic_load(
      &semaphore->async.failure_status, iree_memory_order_acquire);
  if (iree_status_is_ok(failure) && value > current_value) {
    iree_hal_amd_gpu_producer_insert(semaphore, value, signal, producer);
    semaphore->submitted_value = iree_max(semaphore->submitted_value, value);
  }
  iree_hal_semaphore_t* submission_semaphore =
      semaphore->submission_observed ? semaphore->submission_semaphore : NULL;
  iree_slim_mutex_unlock(&semaphore->async.mutex);
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
  iree_hal_semaphore_t* submission_semaphore = semaphore->submission_semaphore;
  iree_slim_mutex_unlock(&semaphore->async.mutex);
  if (submission_semaphore) {
    iree_status_t failure = (iree_status_t)iree_atomic_load(
        &semaphore->async.failure_status, iree_memory_order_acquire);
    iree_hal_semaphore_fail(submission_semaphore, iree_status_clone(failure));
  }
}

// A later producer may require host work performed after this wait.
// Only the exact requested value can accelerate a synchronous host wait.
static iree_hal_amd_gpu_signal_t*
iree_hal_amd_gpu_semaphore_acquire_host_wait_signal(
    iree_hal_amd_gpu_semaphore_t* semaphore, uint64_t value) {
  iree_hal_amd_gpu_signal_t* signal = NULL;
  iree_slim_mutex_lock(&semaphore->async.mutex);
  iree_hal_amd_gpu_semaphore_producer_t* producer =
      iree_hal_amd_gpu_producer_find(semaphore, value);
  if (producer &&
      producer->signal->block->domain == semaphore->wait_context->domain) {
    signal = producer->signal;
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
  if (context &&
      iree_any_bit_set(
          flags, IREE_ASYNC_WAIT_FLAG_ACTIVE | IREE_ASYNC_WAIT_FLAG_YIELD) &&
      !iree_timeout_is_immediate(timeout) &&
      iree_hal_amd_gpu_semaphore_query(&semaphore->async) < value) {
    iree_convert_timeout_to_absolute(&timeout);
    iree_hal_amd_gpu_signal_t* signal =
        iree_hal_amd_gpu_semaphore_acquire_host_wait_signal(semaphore, value);
    if (signal) {
      iree_time_t observe_deadline = timeout.nanos;
      if (!iree_any_bit_set(flags, IREE_ASYNC_WAIT_FLAG_ACTIVE)) {
        // Observe short GPU work before paying for an event wakeup. Longer
        // work returns to the ordinary host notification after this budget.
        observe_deadline =
            iree_timeout_min(timeout, iree_make_timeout_ns(16000)).nanos;
        // This wait has spent its bounded observation budget here.
        flags &= ~IREE_ASYNC_WAIT_FLAG_YIELD;
      }
      while (iree_hal_amd_gpu_semaphore_query(&semaphore->async) < value) {
        if ((uint64_t)iree_atomic_load(
                (const iree_atomic_int64_t*)signal->host_value,
                iree_memory_order_acquire) != context->pending_value) {
          context->try_retire(context->user_data);
          break;
        }
        if (observe_deadline != IREE_TIME_INFINITE_FUTURE &&
            iree_time_now() >= observe_deadline) {
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
