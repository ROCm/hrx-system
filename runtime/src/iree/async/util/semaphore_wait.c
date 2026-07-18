// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/async/util/semaphore_wait.h"

#include <limits.h>
#include <string.h>

enum iree_async_semaphore_wait_completion_state_e {
  IREE_ASYNC_SEMAPHORE_WAIT_COMPLETION_STATE_PENDING = 0,
  IREE_ASYNC_SEMAPHORE_WAIT_COMPLETION_STATE_CLAIMED = 1,
  IREE_ASYNC_SEMAPHORE_WAIT_COMPLETION_STATE_PUBLISHED = 2,
};

enum iree_async_semaphore_wait_queue_state_e {
  // No queue entry or poll thread owns the tracker.
  IREE_ASYNC_SEMAPHORE_WAIT_QUEUE_STATE_IDLE = 0,
  // The tracker is queued or being processed by the poll thread.
  IREE_ASYNC_SEMAPHORE_WAIT_QUEUE_STATE_OWNED = 1,
  // A callback requested another poll pass while the tracker was owned.
  IREE_ASYNC_SEMAPHORE_WAIT_QUEUE_STATE_REQUEUE_REQUESTED = 2,
};

enum {
  IREE_ASYNC_SEMAPHORE_WAIT_QUEUE_STATE_MASK = 0x3,
  IREE_ASYNC_SEMAPHORE_WAIT_ACTIVE_COUNT_SHIFT = 2,
};

static int32_t iree_async_semaphore_wait_active_count(int64_t lifecycle_state) {
  return (int32_t)((uint64_t)lifecycle_state >>
                   IREE_ASYNC_SEMAPHORE_WAIT_ACTIVE_COUNT_SHIFT);
}

static int32_t iree_async_semaphore_wait_queue_state(int64_t lifecycle_state) {
  return (int32_t)(lifecycle_state &
                   IREE_ASYNC_SEMAPHORE_WAIT_QUEUE_STATE_MASK);
}

static int64_t iree_async_semaphore_wait_set_queue_state(
    int64_t lifecycle_state, int32_t queue_state) {
  return (lifecycle_state &
          ~(int64_t)IREE_ASYNC_SEMAPHORE_WAIT_QUEUE_STATE_MASK) |
         queue_state;
}

// Per-semaphore timepoint owned by a wait tracker.
typedef struct iree_async_semaphore_wait_timepoint_t {
  // Semaphore-owned timepoint while the wait remains pending.
  iree_async_semaphore_timepoint_t base;

  // Tracker shared by every timepoint in the wait operation.
  iree_async_semaphore_wait_tracker_t* tracker;

  // Index of this timepoint in the wait operation's semaphore arrays.
  iree_host_size_t index;
} iree_async_semaphore_wait_timepoint_t;

// Tracks a pending proactor semaphore wait until all timepoint callbacks have
// returned. Heap-allocated per wait operation and freed during finalization.
struct iree_async_semaphore_wait_tracker_t {
  // Intrusive link used by the backend completion queue.
  iree_atomic_slist_entry_t slist_entry;

  // Wait operation owned by the proactor until final completion.
  iree_async_semaphore_wait_operation_t* operation;

  // Context serializing association with |operation|.
  iree_async_semaphore_wait_context_t* context;

  // Allocator used for the tracker allocation.
  iree_allocator_t allocator;

  // Backend callback used to enqueue and wake the poll thread.
  iree_async_semaphore_wait_enqueue_callback_t enqueue_callback;

  // Total number of semaphore timepoints represented by this tracker.
  iree_host_size_t count;

  // Number of timepoints successfully passed to a semaphore.
  // Published by |registration_complete| before the tracker can be enqueued.
  iree_host_size_t registered_count;

  // Active timepoint count and queue ownership/requeue handshake state.
  // Combining them makes callback release and requeue request one transition.
  iree_atomic_int64_t lifecycle_state;

  // Set after registration and |registered_count| publication are complete.
  iree_atomic_int32_t registration_complete;

  // Claimed and published terminal completion state.
  iree_atomic_int32_t completion_state;

  // For ALL waits, the number of unsatisfied timepoints.
  // For ANY waits, the first satisfied index or -1 while unsatisfied.
  iree_atomic_int32_t remaining_or_satisfied;

  // Owned terminal status published before the completion becomes enqueueable.
  iree_status_t completion_status;

  // LINKED continuation chain detached from the wait operation at creation.
  iree_async_operation_t* continuation_head;

  // One timepoint for each semaphore in the wait operation.
  iree_async_semaphore_wait_timepoint_t timepoints[];
};

void iree_async_semaphore_wait_context_initialize(
    iree_async_semaphore_wait_context_t* out_context) {
  iree_slim_mutex_initialize(&out_context->association_mutex);
}

void iree_async_semaphore_wait_context_deinitialize(
    iree_async_semaphore_wait_context_t* context) {
  iree_slim_mutex_deinitialize(&context->association_mutex);
}

// Enqueues a published completion after registration is visible. If the queue
// or poll thread already owns the tracker, records a requeue request that the
// poll thread must observe before releasing ownership.
static void iree_async_semaphore_wait_tracker_try_enqueue(
    iree_async_semaphore_wait_tracker_t* tracker) {
  if (!iree_atomic_load(&tracker->registration_complete,
                        iree_memory_order_acquire) ||
      iree_atomic_load(&tracker->completion_state, iree_memory_order_acquire) !=
          IREE_ASYNC_SEMAPHORE_WAIT_COMPLETION_STATE_PUBLISHED) {
    return;
  }

  while (true) {
    int64_t lifecycle_state =
        iree_atomic_load(&tracker->lifecycle_state, iree_memory_order_acquire);
    int32_t queue_state =
        iree_async_semaphore_wait_queue_state(lifecycle_state);
    if (queue_state ==
        IREE_ASYNC_SEMAPHORE_WAIT_QUEUE_STATE_REQUEUE_REQUESTED) {
      return;
    }
    if (queue_state == IREE_ASYNC_SEMAPHORE_WAIT_QUEUE_STATE_OWNED) {
      int64_t desired_state = iree_async_semaphore_wait_set_queue_state(
          lifecycle_state,
          IREE_ASYNC_SEMAPHORE_WAIT_QUEUE_STATE_REQUEUE_REQUESTED);
      if (iree_atomic_compare_exchange_weak(
              &tracker->lifecycle_state, &lifecycle_state, desired_state,
              iree_memory_order_acq_rel, iree_memory_order_acquire)) {
        return;
      }
      continue;
    }
    int64_t desired_state = iree_async_semaphore_wait_set_queue_state(
        lifecycle_state, IREE_ASYNC_SEMAPHORE_WAIT_QUEUE_STATE_OWNED);
    if (iree_atomic_compare_exchange_weak(
            &tracker->lifecycle_state, &lifecycle_state, desired_state,
            iree_memory_order_acq_rel, iree_memory_order_acquire)) {
      tracker->enqueue_callback.fn(tracker->enqueue_callback.user_data,
                                   &tracker->slist_entry);
      return;
    }
  }
}

// Claims the first terminal event and publishes its owned status. A separate
// CLAIMED state prevents registration from enqueueing a failure before the
// winning callback has stored its status.
static void iree_async_semaphore_wait_tracker_publish_completion(
    iree_async_semaphore_wait_tracker_t* tracker, iree_status_t status) {
  int32_t expected = IREE_ASYNC_SEMAPHORE_WAIT_COMPLETION_STATE_PENDING;
  if (!iree_atomic_compare_exchange_strong(
          &tracker->completion_state, &expected,
          IREE_ASYNC_SEMAPHORE_WAIT_COMPLETION_STATE_CLAIMED,
          iree_memory_order_acq_rel, iree_memory_order_acquire)) {
    iree_status_free(status);
    return;
  }

  tracker->completion_status = status;
  iree_atomic_store(&tracker->completion_state,
                    IREE_ASYNC_SEMAPHORE_WAIT_COMPLETION_STATE_PUBLISHED,
                    iree_memory_order_release);
  iree_async_semaphore_wait_tracker_try_enqueue(tracker);
}

// Releases a callback's ownership unit while atomically publishing any requeue
// intent. A poll thread observing zero may immediately free the tracker, so no
// tracker access is permitted after the lifecycle transition unless this
// callback acquired queue ownership and still needs to push the entry.
static void iree_async_semaphore_wait_tracker_release_callback(
    iree_async_semaphore_wait_tracker_t* tracker) {
  const bool is_enqueueable =
      iree_atomic_load(&tracker->registration_complete,
                       iree_memory_order_acquire) &&
      iree_atomic_load(&tracker->completion_state, iree_memory_order_acquire) ==
          IREE_ASYNC_SEMAPHORE_WAIT_COMPLETION_STATE_PUBLISHED;
  while (true) {
    int64_t lifecycle_state =
        iree_atomic_load(&tracker->lifecycle_state, iree_memory_order_acquire);
    IREE_ASSERT(iree_async_semaphore_wait_active_count(lifecycle_state) > 0);
    int64_t desired_state =
        lifecycle_state -
        ((int64_t)1 << IREE_ASYNC_SEMAPHORE_WAIT_ACTIVE_COUNT_SHIFT);
    bool enqueue = false;
    if (is_enqueueable) {
      int32_t queue_state =
          iree_async_semaphore_wait_queue_state(lifecycle_state);
      if (queue_state == IREE_ASYNC_SEMAPHORE_WAIT_QUEUE_STATE_IDLE) {
        desired_state = iree_async_semaphore_wait_set_queue_state(
            desired_state, IREE_ASYNC_SEMAPHORE_WAIT_QUEUE_STATE_OWNED);
        enqueue = true;
      } else if (queue_state == IREE_ASYNC_SEMAPHORE_WAIT_QUEUE_STATE_OWNED) {
        desired_state = iree_async_semaphore_wait_set_queue_state(
            desired_state,
            IREE_ASYNC_SEMAPHORE_WAIT_QUEUE_STATE_REQUEUE_REQUESTED);
      }
    }
    if (!iree_atomic_compare_exchange_weak(
            &tracker->lifecycle_state, &lifecycle_state, desired_state,
            iree_memory_order_acq_rel, iree_memory_order_acquire)) {
      continue;
    }
    if (enqueue) {
      tracker->enqueue_callback.fn(tracker->enqueue_callback.user_data,
                                   &tracker->slist_entry);
    }
    return;
  }
}

static void iree_async_semaphore_wait_tracker_timepoint_callback(
    void* user_data, iree_async_semaphore_timepoint_t* base_timepoint,
    iree_status_t status) {
  (void)user_data;
  iree_async_semaphore_wait_timepoint_t* timepoint =
      (iree_async_semaphore_wait_timepoint_t*)base_timepoint;
  iree_async_semaphore_wait_tracker_t* tracker = timepoint->tracker;

  if (!iree_status_is_ok(status)) {
    iree_async_semaphore_wait_tracker_publish_completion(tracker, status);
  } else if (tracker->operation->mode == IREE_ASYNC_WAIT_MODE_ANY) {
    int32_t expected = -1;
    if (iree_atomic_compare_exchange_strong(
            &tracker->remaining_or_satisfied, &expected,
            (int32_t)timepoint->index, iree_memory_order_acq_rel,
            iree_memory_order_acquire)) {
      iree_async_semaphore_wait_tracker_publish_completion(tracker,
                                                           iree_ok_status());
    }
  } else {
    int32_t remaining = iree_atomic_fetch_sub(&tracker->remaining_or_satisfied,
                                              1, iree_memory_order_acq_rel) -
                        1;
    if (remaining == 0) {
      iree_async_semaphore_wait_tracker_publish_completion(tracker,
                                                           iree_ok_status());
    }
  }

  // No tracker access is permitted after this release.
  iree_async_semaphore_wait_tracker_release_callback(tracker);
}

iree_status_t iree_async_semaphore_wait_tracker_create(
    iree_async_semaphore_wait_context_t* context,
    iree_async_semaphore_wait_operation_t* operation,
    iree_async_semaphore_wait_enqueue_callback_t enqueue_callback,
    iree_allocator_t allocator,
    iree_async_semaphore_wait_tracker_t** out_tracker) {
  *out_tracker = NULL;
  if (IREE_UNLIKELY(!enqueue_callback.fn)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "semaphore wait enqueue callback is required");
  }
  if (IREE_UNLIKELY(operation->count > INT32_MAX)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "semaphore wait count %" PRIhsz
                            " exceeds the maximum of %d",
                            operation->count, INT32_MAX);
  }

  iree_host_size_t total_size = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      sizeof(iree_async_semaphore_wait_tracker_t), &total_size,
      IREE_STRUCT_FIELD_FAM(operation->count,
                            iree_async_semaphore_wait_timepoint_t)));

  iree_async_semaphore_wait_tracker_t* tracker = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(allocator, total_size, (void**)&tracker));
  memset(tracker, 0, total_size);
  tracker->operation = operation;
  tracker->context = context;
  tracker->allocator = allocator;
  tracker->enqueue_callback = enqueue_callback;
  tracker->count = operation->count;
  tracker->completion_status = iree_ok_status();
  tracker->continuation_head = operation->base.linked_next;
  operation->base.linked_next = NULL;
  iree_atomic_store(
      &tracker->lifecycle_state,
      (int64_t)operation->count << IREE_ASYNC_SEMAPHORE_WAIT_ACTIVE_COUNT_SHIFT,
      iree_memory_order_release);
  iree_atomic_store(&tracker->registration_complete, 0,
                    iree_memory_order_release);
  iree_atomic_store(&tracker->completion_state,
                    IREE_ASYNC_SEMAPHORE_WAIT_COMPLETION_STATE_PENDING,
                    iree_memory_order_release);
  iree_atomic_store(&tracker->remaining_or_satisfied,
                    operation->mode == IREE_ASYNC_WAIT_MODE_ALL
                        ? (int32_t)operation->count
                        : -1,
                    iree_memory_order_release);
  for (iree_host_size_t i = 0; i < operation->count; ++i) {
    iree_async_semaphore_retain(operation->semaphores[i]);
  }

  // Publish the fully initialized tracker to concurrent cancellation last.
  iree_slim_mutex_lock(&context->association_mutex);
  operation->base.next = (iree_async_operation_t*)tracker;
  iree_slim_mutex_unlock(&context->association_mutex);

  *out_tracker = tracker;
  return iree_ok_status();
}

void iree_async_semaphore_wait_tracker_register_timepoints(
    iree_async_semaphore_wait_tracker_t* tracker) {
  for (iree_host_size_t i = 0; i < tracker->count; ++i) {
    iree_async_semaphore_wait_timepoint_t* timepoint = &tracker->timepoints[i];
    timepoint->tracker = tracker;
    timepoint->index = i;
    timepoint->base.callback =
        iree_async_semaphore_wait_tracker_timepoint_callback;
    timepoint->base.user_data = NULL;

    iree_status_t status = iree_async_semaphore_acquire_timepoint(
        tracker->operation->semaphores[i], tracker->operation->values[i],
        &timepoint->base);
    if (!iree_status_is_ok(status)) {
      iree_async_semaphore_wait_tracker_publish_completion(tracker, status);
      break;
    }
    tracker->registered_count = i + 1;

    // ANY satisfaction or any failure makes additional registrations
    // unnecessary. ALL success can only be published by the final timepoint.
    if (iree_atomic_load(&tracker->completion_state,
                         iree_memory_order_acquire) !=
        IREE_ASYNC_SEMAPHORE_WAIT_COMPLETION_STATE_PENDING) {
      break;
    }
  }

  // Slots not registered with a semaphore have no callback to release their
  // ownership units. Registration is not yet published, so no poll thread can
  // observe a transient zero while this thread continues touching the tracker.
  iree_host_size_t unregistered_count =
      tracker->count - tracker->registered_count;
  if (unregistered_count > 0) {
    iree_atomic_fetch_sub(&tracker->lifecycle_state,
                          (int64_t)unregistered_count
                              << IREE_ASYNC_SEMAPHORE_WAIT_ACTIVE_COUNT_SHIFT,
                          iree_memory_order_acq_rel);
  }

  iree_atomic_store(&tracker->registration_complete, 1,
                    iree_memory_order_release);
  iree_async_semaphore_wait_tracker_try_enqueue(tracker);
}

void iree_async_semaphore_wait_context_request_cancellation(
    iree_async_semaphore_wait_context_t* context,
    iree_async_semaphore_wait_operation_t* operation) {
  iree_slim_mutex_lock(&context->association_mutex);
  iree_async_semaphore_wait_tracker_t* tracker =
      (iree_async_semaphore_wait_tracker_t*)operation->base.next;
  if (tracker) {
    iree_async_semaphore_wait_tracker_publish_completion(
        tracker,
        iree_make_status(IREE_STATUS_CANCELLED, "operation cancelled"));
  }
  iree_slim_mutex_unlock(&context->association_mutex);
}

bool iree_async_semaphore_wait_tracker_try_prepare_completion(
    iree_async_semaphore_wait_tracker_t* tracker) {
  iree_async_semaphore_wait_operation_t* operation = tracker->operation;
  for (iree_host_size_t i = 0; i < tracker->registered_count; ++i) {
    if (iree_async_semaphore_cancel_timepoint(operation->semaphores[i],
                                              &tracker->timepoints[i].base)) {
      iree_atomic_fetch_sub(
          &tracker->lifecycle_state,
          (int64_t)1 << IREE_ASYNC_SEMAPHORE_WAIT_ACTIVE_COUNT_SHIFT,
          iree_memory_order_acq_rel);
    }
  }

  // Release queue ownership or consume a callback's requeue request. Callback
  // release changes the active count and queue state in one atomic transition,
  // so this loop cannot consume a request before its callback becomes inactive.
  while (true) {
    int64_t lifecycle_state =
        iree_atomic_load(&tracker->lifecycle_state, iree_memory_order_acquire);
    if (iree_async_semaphore_wait_active_count(lifecycle_state) == 0) {
      return true;
    }
    int32_t queue_state =
        iree_async_semaphore_wait_queue_state(lifecycle_state);
    int32_t next_queue_state =
        queue_state == IREE_ASYNC_SEMAPHORE_WAIT_QUEUE_STATE_REQUEUE_REQUESTED
            ? IREE_ASYNC_SEMAPHORE_WAIT_QUEUE_STATE_OWNED
            : IREE_ASYNC_SEMAPHORE_WAIT_QUEUE_STATE_IDLE;
    int64_t desired_state = iree_async_semaphore_wait_set_queue_state(
        lifecycle_state, next_queue_state);
    if (!iree_atomic_compare_exchange_weak(
            &tracker->lifecycle_state, &lifecycle_state, desired_state,
            iree_memory_order_acq_rel, iree_memory_order_acquire)) {
      continue;
    }
    if (next_queue_state == IREE_ASYNC_SEMAPHORE_WAIT_QUEUE_STATE_OWNED) {
      tracker->enqueue_callback.fn(tracker->enqueue_callback.user_data,
                                   &tracker->slist_entry);
    }
    return false;
  }
}

void iree_async_semaphore_wait_tracker_finalize(
    iree_async_semaphore_wait_tracker_t* tracker,
    iree_async_semaphore_wait_completion_t* out_completion) {
  iree_async_semaphore_wait_operation_t* operation = tracker->operation;
  iree_status_t status = tracker->completion_status;
  if (operation->mode == IREE_ASYNC_WAIT_MODE_ANY &&
      iree_status_is_ok(status)) {
    int32_t satisfied = iree_atomic_load(&tracker->remaining_or_satisfied,
                                         iree_memory_order_acquire);
    operation->satisfied_index = (iree_host_size_t)satisfied;
  }

  iree_slim_mutex_lock(&tracker->context->association_mutex);
  operation->base.next = NULL;
  iree_slim_mutex_unlock(&tracker->context->association_mutex);
  out_completion->operation = operation;
  out_completion->continuation_head = tracker->continuation_head;
  out_completion->status = status;

  for (iree_host_size_t i = 0; i < tracker->count; ++i) {
    iree_async_semaphore_release(operation->semaphores[i]);
  }
  iree_allocator_t allocator = tracker->allocator;
  iree_allocator_free(allocator, tracker);
}
