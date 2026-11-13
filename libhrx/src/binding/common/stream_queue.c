// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/stream_queue.h"

#include "common/context.h"
#include "common/device.h"
#include "common/stream.h"
#include "iree/async/util/proactor_pool.h"
#include "libhrx/src/libhrx/runtime.h"

// Active streams require independent queues; only completed queues count
// against this bounded context cache.
enum { IREE_HAL_STREAMING_VALUE_WAIT_IDLE_LANE_LIMIT = 8 };

static bool iree_hal_streaming_value_wait_lane_matches(
    const iree_hal_streaming_value_wait_lane_t* lane,
    const iree_hal_queue_family_t* family, iree_hal_queue_priority_t priority,
    iree_hal_queue_feature_flags_t features,
    iree_hal_queue_execution_resource_list_t execution_resources) {
  return lane->family == family && lane->priority == priority &&
         iree_hal_queue_features(lane->queue) == features &&
         lane->execution_resources.count == execution_resources.count &&
         (execution_resources.count == 0 ||
          memcmp(lane->execution_resources.ordinals,
                 execution_resources.ordinals,
                 execution_resources.count *
                     sizeof(*execution_resources.ordinals)) == 0);
}

void iree_hal_streaming_value_wait_lanes_initialize(
    iree_hal_streaming_context_t* context) {
  context->idle_value_wait_lanes = NULL;
  context->idle_value_wait_lane_count = 0;
  context->pending_value_wait_lanes = NULL;
  context->active_value_wait_observers = NULL;
  context->shutdown_value_wait_submissions = NULL;
  iree_atomic_store(&context->active_value_wait_observer_count, 0,
                    iree_memory_order_relaxed);
  context->value_wait_lanes_shutting_down = false;
  context->live_value_wait_submission_count = 0;
  context->peak_value_wait_submission_count = 0;
  context->value_wait_completion_query_count = 0;
  context->value_wait_record_visit_count = 0;
  context->value_wait_observer_removal_count = 0;
  iree_notification_initialize(&context->value_wait_observer_notification);
  context->value_wait_observer_finish_hook = NULL;
  context->value_wait_observer_finish_hook_user_data = NULL;
  iree_slim_mutex_initialize(&context->value_wait_lane_mutex);
}

static void iree_hal_streaming_destroy_value_wait_submissions(
    iree_hal_streaming_context_t* context,
    iree_hal_streaming_value_wait_submission_t* submissions) {
  while (submissions) {
    iree_hal_streaming_value_wait_submission_t* next = submissions->next;
    iree_hal_semaphore_release(submissions->completion_semaphore);
    iree_async_proactor_release(submissions->observer_proactor);
    iree_hal_streaming_stream_release(submissions->stream);
    iree_allocator_free(context->host_allocator, submissions);
    submissions = next;
  }
}

static void iree_hal_streaming_destroy_value_wait_lanes(
    iree_hal_streaming_context_t* context,
    iree_hal_streaming_value_wait_lane_t* lanes) {
  while (lanes) {
    iree_hal_streaming_value_wait_lane_t* next = lanes->next;
    // Queue teardown is authoritative for still-pending backend work. Keep
    // every private completion semaphore and record alive until it returns.
    iree_hal_queue_release(lanes->queue);
    iree_hal_streaming_destroy_value_wait_submissions(context,
                                                      lanes->submission_head);
    iree_hal_streaming_destroy_value_wait_submissions(
        context, lanes->retired_failure_head);
    iree_slim_mutex_deinitialize(&lanes->submission_mutex);
    iree_allocator_free(context->host_allocator, lanes);
    lanes = next;
  }
}

static void iree_hal_streaming_remove_value_wait_observer_locked(
    iree_hal_streaming_context_t* context,
    iree_hal_streaming_value_wait_submission_t* submission) {
  IREE_ASSERT(submission->observer_active,
              "value-wait completion observer must be context-owned");
  if (submission->observer_prev) {
    submission->observer_prev->observer_next = submission->observer_next;
  } else {
    IREE_ASSERT(context->active_value_wait_observers == submission,
                "value-wait observer head must be context-owned");
    context->active_value_wait_observers = submission->observer_next;
  }
  if (submission->observer_next) {
    submission->observer_next->observer_prev = submission->observer_prev;
  }
  submission->observer_next = NULL;
  submission->observer_prev = NULL;
  submission->observer_active = false;
  ++context->value_wait_observer_removal_count;
}

static void iree_hal_streaming_insert_value_wait_lane_locked(
    iree_hal_streaming_context_t* context,
    iree_hal_streaming_value_wait_lane_t* lane,
    iree_hal_streaming_value_wait_lane_list_state_t list_state) {
  IREE_ASSERT(
      lane->list_state == IREE_HAL_STREAMING_VALUE_WAIT_LANE_LIST_STATE_NONE,
      "value-wait lane must be detached before insertion");
  IREE_ASSERT(
      list_state == IREE_HAL_STREAMING_VALUE_WAIT_LANE_LIST_STATE_IDLE ||
          list_state == IREE_HAL_STREAMING_VALUE_WAIT_LANE_LIST_STATE_PENDING,
      "value-wait lane must enter a context-owned list");
  iree_hal_streaming_value_wait_lane_t** list_head =
      list_state == IREE_HAL_STREAMING_VALUE_WAIT_LANE_LIST_STATE_IDLE
          ? &context->idle_value_wait_lanes
          : &context->pending_value_wait_lanes;
  lane->prev = NULL;
  lane->next = *list_head;
  if (*list_head) {
    (*list_head)->prev = lane;
  }
  *list_head = lane;
  lane->list_state = list_state;
  if (list_state == IREE_HAL_STREAMING_VALUE_WAIT_LANE_LIST_STATE_IDLE) {
    ++context->idle_value_wait_lane_count;
  }
}

static void iree_hal_streaming_remove_value_wait_lane_locked(
    iree_hal_streaming_context_t* context,
    iree_hal_streaming_value_wait_lane_t* lane) {
  IREE_ASSERT(
      lane->list_state != IREE_HAL_STREAMING_VALUE_WAIT_LANE_LIST_STATE_NONE,
      "value-wait lane must be context-owned before removal");
  iree_hal_streaming_value_wait_lane_t** list_head =
      lane->list_state == IREE_HAL_STREAMING_VALUE_WAIT_LANE_LIST_STATE_IDLE
          ? &context->idle_value_wait_lanes
          : &context->pending_value_wait_lanes;
  if (lane->prev) {
    lane->prev->next = lane->next;
  } else {
    IREE_ASSERT(*list_head == lane,
                "value-wait lane head must be context-owned");
    *list_head = lane->next;
  }
  if (lane->next) {
    lane->next->prev = lane->prev;
  }
  if (lane->list_state == IREE_HAL_STREAMING_VALUE_WAIT_LANE_LIST_STATE_IDLE) {
    IREE_ASSERT(context->idle_value_wait_lane_count > 0,
                "idle value-wait lane count underflow");
    --context->idle_value_wait_lane_count;
  }
  lane->next = NULL;
  lane->prev = NULL;
  lane->list_state = IREE_HAL_STREAMING_VALUE_WAIT_LANE_LIST_STATE_NONE;
}

static void iree_hal_streaming_finish_value_wait_observer(
    iree_hal_streaming_context_t* context) {
  // Teardown tests the zero predicate while holding this same mutex. Keep the
  // transition to zero and its notification in one critical section so the
  // predicate cannot observe zero until the final callback has finished all
  // accesses to notification/context storage.
  iree_slim_mutex_lock(&context->value_wait_lane_mutex);
  int32_t previous = iree_atomic_fetch_sub(
      &context->active_value_wait_observer_count, 1, iree_memory_order_acq_rel);
  IREE_ASSERT(previous > 0, "value-wait observer count underflow");
  if (previous == 1 && context->value_wait_observer_finish_hook) {
    context->value_wait_observer_finish_hook(
        context->value_wait_observer_finish_hook_user_data);
  }
  iree_notification_post(&context->value_wait_observer_notification,
                         IREE_ALL_WAITERS);
  iree_slim_mutex_unlock(&context->value_wait_lane_mutex);
}

// Removes exactly one terminal record in O(1). Completion callbacks use this
// path so in-order completion of a long burst cannot repeatedly scan the lane.
static void iree_hal_streaming_detach_terminal_value_wait_submission_locked(
    iree_hal_streaming_context_t* context,
    iree_hal_streaming_value_wait_submission_t* submission,
    iree_hal_streaming_value_wait_submission_t** out_reclaimed_submissions,
    iree_hal_streaming_value_wait_lane_t** out_completed_lanes,
    iree_hal_streaming_value_wait_lane_t** out_failed_lanes) {
  IREE_ASSERT(submission->state ==
                  IREE_HAL_STREAMING_VALUE_WAIT_SUBMISSION_STATE_PUBLISHED,
              "only a published value-wait record can be detached");
  IREE_ASSERT(submission->is_terminal,
              "only a terminal value-wait record can be detached");
  iree_hal_streaming_value_wait_lane_t* lane = submission->lane;
  IREE_ASSERT(lane->submission_count > 0,
              "value-wait lane record count underflow");
  ++context->value_wait_record_visit_count;
  if (submission->prev) {
    submission->prev->next = submission->next;
  } else {
    IREE_ASSERT(lane->submission_head == submission,
                "value-wait record head must be lane-owned");
    lane->submission_head = submission->next;
  }
  if (submission->next) {
    submission->next->prev = submission->prev;
  } else {
    IREE_ASSERT(lane->submission_tail == submission,
                "value-wait record tail must be lane-owned");
    lane->submission_tail = submission->prev;
  }
  submission->prev = NULL;
  --lane->submission_count;
  if (submission->has_failed) {
    lane->has_failed_submission = true;
    submission->next = lane->retired_failure_head;
    lane->retired_failure_head = submission;
    ++lane->retired_failure_count;
  } else {
    submission->next = *out_reclaimed_submissions;
    *out_reclaimed_submissions = submission;
    IREE_ASSERT(context->live_value_wait_submission_count > 0,
                "value-wait live record count underflow");
    --context->live_value_wait_submission_count;
  }
  if (lane->submission_count != 0) {
    return;
  }

  IREE_ASSERT(!lane->submission_head && !lane->submission_tail,
              "empty value-wait lane must not retain record links");
  // A lane temporarily acquired for a same-owner append is deliberately absent
  // from the context lists. Its publisher/rejector owns the empty-lane route.
  if (lane->list_state !=
      IREE_HAL_STREAMING_VALUE_WAIT_LANE_LIST_STATE_PENDING) {
    return;
  }
  iree_hal_streaming_remove_value_wait_lane_locked(context, lane);
  if (submission->stream && submission->stream->value_wait_lane == lane) {
    submission->stream->value_wait_lane = NULL;
  }
  if (lane->has_failed_submission) {
    // Failed exact proofs remain lane-owned until queue teardown returns. This
    // is required because queue release is the authoritative backend drain.
    IREE_ASSERT(context->live_value_wait_submission_count >=
                    lane->retired_failure_count,
                "value-wait retained failure count underflow");
    context->live_value_wait_submission_count -= lane->retired_failure_count;
    lane->next = *out_failed_lanes;
    if (lane->next) {
      lane->next->prev = lane;
    }
    lane->prev = NULL;
    *out_failed_lanes = lane;
  } else {
    lane->next = *out_completed_lanes;
    if (lane->next) {
      lane->next->prev = lane;
    }
    lane->prev = NULL;
    *out_completed_lanes = lane;
  }
}

static void iree_hal_streaming_retire_value_wait_state(
    iree_hal_streaming_context_t* context,
    iree_hal_streaming_value_wait_submission_t* reclaimed_submissions,
    iree_hal_streaming_value_wait_lane_t* completed_lanes,
    iree_hal_streaming_value_wait_lane_t* failed_lanes) {
  // Failed queues are drained before their final exact completion proof is
  // released; destroy_value_wait_lanes preserves that queue-first order.
  iree_hal_streaming_destroy_value_wait_lanes(context, failed_lanes);
  iree_hal_streaming_destroy_value_wait_submissions(context,
                                                    reclaimed_submissions);

  for (iree_hal_streaming_value_wait_lane_t* lane = completed_lanes; lane;
       lane = lane->next) {
    IREE_ASSERT(!lane->submission_head && !lane->submission_tail,
                "completed value-wait lane must have no live records");
    IREE_ASSERT(lane->submission_count == 0,
                "completed value-wait lane must have no live record count");
    IREE_ASSERT(!lane->retired_failure_head && lane->retired_failure_count == 0,
                "recyclable value-wait lane cannot retain failed proofs");
    lane->owner_stream_id = 0;
    lane->restore_pending = false;
    lane->has_failed_submission = false;
  }

  iree_hal_streaming_value_wait_lane_t* discarded_lanes = NULL;
  iree_slim_mutex_lock(&context->value_wait_lane_mutex);
  while (completed_lanes) {
    iree_hal_streaming_value_wait_lane_t* lane = completed_lanes;
    completed_lanes = lane->next;
    lane->next = NULL;
    lane->prev = NULL;
    if (!context->value_wait_lanes_shutting_down &&
        context->idle_value_wait_lane_count <
            IREE_HAL_STREAMING_VALUE_WAIT_IDLE_LANE_LIMIT) {
      iree_hal_streaming_insert_value_wait_lane_locked(
          context, lane, IREE_HAL_STREAMING_VALUE_WAIT_LANE_LIST_STATE_IDLE);
    } else {
      lane->next = discarded_lanes;
      if (discarded_lanes) {
        discarded_lanes->prev = lane;
      }
      discarded_lanes = lane;
    }
  }
  iree_slim_mutex_unlock(&context->value_wait_lane_mutex);
  iree_hal_streaming_destroy_value_wait_lanes(context, discarded_lanes);
}

static void iree_hal_streaming_value_wait_observer_callback(
    void* user_data, iree_async_operation_t* operation, iree_status_t status,
    iree_async_completion_flags_t flags);

static void iree_hal_streaming_initialize_value_wait_observer_operation(
    iree_hal_streaming_value_wait_submission_t* submission) {
  iree_async_operation_initialize(
      &submission->observer_operation.base,
      IREE_ASYNC_OPERATION_TYPE_SEMAPHORE_WAIT, IREE_ASYNC_OPERATION_FLAG_NONE,
      iree_hal_streaming_value_wait_observer_callback, submission);
  submission->observer_operation.semaphores = &submission->observer_semaphore;
  submission->observer_operation.values = &submission->observer_value;
  submission->observer_operation.count = 1;
  submission->observer_operation.mode = IREE_ASYNC_WAIT_MODE_ALL;
  submission->observer_operation.satisfied_index = 0;
}

static void iree_hal_streaming_value_wait_observer_callback(
    void* user_data, iree_async_operation_t* operation, iree_status_t status,
    iree_async_completion_flags_t flags) {
  (void)operation;
  (void)flags;
  iree_hal_streaming_value_wait_submission_t* submission =
      (iree_hal_streaming_value_wait_submission_t*)user_data;
  iree_hal_streaming_context_t* context = submission->context;

  iree_hal_streaming_stream_t* stream = submission->stream;
  if (stream) {
    iree_slim_mutex_lock(&stream->mutex);
  }

  // Shutdown cancellation is observer-only: it must not turn cancellation
  // into backend completion proof or release a lane before queue teardown.
  iree_slim_mutex_lock(&context->value_wait_lane_mutex);
  if (context->value_wait_lanes_shutting_down) {
    iree_hal_streaming_remove_value_wait_observer_locked(context, submission);
    if (submission->state ==
        IREE_HAL_STREAMING_VALUE_WAIT_SUBMISSION_STATE_REJECTED) {
      submission->next = context->shutdown_value_wait_submissions;
      context->shutdown_value_wait_submissions = submission;
    } else {
      IREE_ASSERT(submission->state ==
                      IREE_HAL_STREAMING_VALUE_WAIT_SUBMISSION_STATE_PUBLISHED,
                  "a prepared observer cannot outlive its retaining API call");
    }
    iree_slim_mutex_unlock(&context->value_wait_lane_mutex);
    iree_status_free(status);
    if (stream) {
      iree_slim_mutex_unlock(&stream->mutex);
    }
    iree_hal_streaming_finish_value_wait_observer(context);
    return;
  }
  const bool was_rejected =
      submission->state ==
      IREE_HAL_STREAMING_VALUE_WAIT_SUBMISSION_STATE_REJECTED;
  iree_hal_streaming_value_wait_lane_t* lane = submission->lane;
  iree_slim_mutex_unlock(&context->value_wait_lane_mutex);

  // The async wait status alone can also represent observer cancellation.
  // Query the private semaphore once to establish the exact submission's
  // terminal success/failure before mutating lane ownership.
  uint64_t completion_value = 0;
  iree_status_t query_status = iree_hal_semaphore_query(
      submission->completion_semaphore, &completion_value);
  const bool is_terminal =
      !iree_status_is_ok(query_status) || completion_value >= 1;
  const bool has_failed = !iree_status_is_ok(query_status);
  iree_status_free(query_status);
  iree_status_free(status);

  // A published observer takes the same per-lane gate as acceptance and
  // publication. If an older failure wins this gate, a new append observes the
  // sticky failure and rejects. If the append wins, its accepted record is
  // published before this callback can make the lane destroy-only. Rejected
  // records own no lane and must not dereference one that its caller may free.
  if (!was_rejected) {
    iree_slim_mutex_lock(&lane->submission_mutex);
  }

  iree_hal_streaming_value_wait_submission_t* reclaimed_submissions = NULL;
  iree_hal_streaming_value_wait_lane_t* completed_lanes = NULL;
  iree_hal_streaming_value_wait_lane_t* failed_lanes = NULL;
  iree_slim_mutex_lock(&context->value_wait_lane_mutex);
  ++context->value_wait_completion_query_count;
  if (context->value_wait_lanes_shutting_down) {
    iree_hal_streaming_remove_value_wait_observer_locked(context, submission);
    if (submission->state ==
        IREE_HAL_STREAMING_VALUE_WAIT_SUBMISSION_STATE_REJECTED) {
      submission->next = context->shutdown_value_wait_submissions;
      context->shutdown_value_wait_submissions = submission;
    } else {
      IREE_ASSERT(submission->state ==
                      IREE_HAL_STREAMING_VALUE_WAIT_SUBMISSION_STATE_PUBLISHED,
                  "a prepared observer cannot outlive its retaining API call");
    }
  } else if (!is_terminal) {
    // This module exposes no normal cancellation path. Teardown sets shutdown
    // before requesting cancellation, and every natural callback is dispatched
    // only after this private semaphore is terminal. Fail closed if a proactor
    // violates that contract: retain the record and make the lane destroy-only
    // until authoritative backend teardown, never recycle it on false proof.
    // There is deliberately no fallible rearm path that could strand a record
    // without an observer: nonterminal completion is unreachable on every
    // supported native proactor backend.
    IREE_ASSERT(false, "live value-wait observer completed before terminal");
    iree_hal_streaming_remove_value_wait_observer_locked(context, submission);
    submission->has_failed = true;
    if (submission->state ==
        IREE_HAL_STREAMING_VALUE_WAIT_SUBMISSION_STATE_PUBLISHED) {
      submission->lane->has_failed_submission = true;
    } else if (submission->state ==
               IREE_HAL_STREAMING_VALUE_WAIT_SUBMISSION_STATE_REJECTED) {
      submission->next = reclaimed_submissions;
      reclaimed_submissions = submission;
    }
  } else {
    submission->is_terminal = true;
    submission->has_failed = has_failed;
    iree_hal_streaming_remove_value_wait_observer_locked(context, submission);
    if (submission->state ==
        IREE_HAL_STREAMING_VALUE_WAIT_SUBMISSION_STATE_REJECTED) {
      submission->next = reclaimed_submissions;
      reclaimed_submissions = submission;
    } else if (submission->state ==
               IREE_HAL_STREAMING_VALUE_WAIT_SUBMISSION_STATE_PUBLISHED) {
      iree_hal_streaming_detach_terminal_value_wait_submission_locked(
          context, submission, &reclaimed_submissions, &completed_lanes,
          &failed_lanes);
    }
    // PREPARED remains owned by the publisher/rejector. It observes these
    // terminal fields under this same mutex before transferring ownership. In
    // the accepted case the publisher already holds the lane gate, so a normal
    // callback cannot remain PREPARED here.
  }
  iree_slim_mutex_unlock(&context->value_wait_lane_mutex);
  if (!was_rejected) {
    iree_slim_mutex_unlock(&lane->submission_mutex);
  }

  if (stream) {
    iree_slim_mutex_unlock(&stream->mutex);
  }
  iree_hal_streaming_retire_value_wait_state(context, reclaimed_submissions,
                                             completed_lanes, failed_lanes);
  // This is intentionally the final context access. Teardown waits for the
  // count to reach zero before deinitializing the mutex or freeing the context.
  iree_hal_streaming_finish_value_wait_observer(context);
}

// Acquires a queue that remains exclusive to |stream_id| until all accepted
// waits on it have completed. Terminal callbacks, rather than foreign scans,
// maintain the pending list so a permanently blocked lane has no polling cost.
static iree_status_t iree_hal_streaming_acquire_value_wait_lane(
    iree_hal_streaming_context_t* context,
    const iree_hal_queue_family_t* family, iree_hal_queue_priority_t priority,
    iree_hal_queue_feature_flags_t features,
    iree_hal_queue_execution_resource_list_t execution_resources,
    iree_hal_queue_t* excluded_queue, unsigned long long stream_id,
    iree_hal_streaming_value_wait_lane_t** out_lane) {
  IREE_ASSERT_ARGUMENT(out_lane);
  *out_lane = NULL;

  iree_slim_mutex_lock(&context->value_wait_lane_mutex);
  for (iree_hal_streaming_value_wait_lane_t* lane =
           context->idle_value_wait_lanes;
       lane && !*out_lane; lane = lane->next) {
    if (lane->queue != excluded_queue &&
        iree_hal_streaming_value_wait_lane_matches(
            lane, family, priority, features, execution_resources)) {
      iree_hal_streaming_remove_value_wait_lane_locked(context, lane);
      lane->owner_stream_id = stream_id;
      lane->restore_pending = false;
      *out_lane = lane;
    }
  }
  iree_slim_mutex_unlock(&context->value_wait_lane_mutex);
  if (*out_lane) {
    return iree_ok_status();
  }

  iree_hal_queue_params_t params;
  iree_hal_queue_params_initialize(&params);
  params.priority = priority;
  params.features = features;
  params.execution_resources = execution_resources;
  iree_hal_queue_t* queue = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_queue_acquire(family, &params, &queue));
  if (IREE_UNLIKELY(queue == excluded_queue)) {
    iree_hal_queue_release(queue);
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "dynamic queue acquisition returned the stream operation queue");
  }

  iree_hal_streaming_value_wait_lane_t* lane = NULL;
  iree_status_t status = iree_allocator_malloc(context->host_allocator,
                                               sizeof(*lane), (void**)&lane);
  if (!iree_status_is_ok(status)) {
    iree_hal_queue_release(queue);
    return status;
  }
  memset(lane, 0, sizeof(*lane));
  iree_slim_mutex_initialize(&lane->submission_mutex);
  lane->queue = queue;
  lane->family = iree_hal_queue_family(queue);
  lane->priority = iree_hal_queue_priority(queue);
  lane->execution_resources = iree_hal_queue_execution_resources(queue);
  lane->owner_stream_id = stream_id;
  *out_lane = lane;
  return iree_ok_status();
}

iree_status_t iree_hal_streaming_prepare_value_wait_submission(
    iree_hal_streaming_context_t* context,
    iree_hal_streaming_value_wait_lane_t* lane,
    iree_hal_streaming_stream_t* stream,
    iree_hal_streaming_value_wait_submission_t** out_submission) {
  IREE_ASSERT_ARGUMENT(out_submission);
  *out_submission = NULL;
  iree_hal_streaming_value_wait_submission_t* submission = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      context->host_allocator, sizeof(*submission), (void**)&submission));
  memset(submission, 0, sizeof(*submission));
  submission->context = context;
  submission->lane = lane;
  submission->state = IREE_HAL_STREAMING_VALUE_WAIT_SUBMISSION_STATE_PREPARED;

  const iree_hal_queue_family_affinity_t queue_family_affinity =
      iree_hal_make_queue_family_affinity(
          iree_hal_queue_family_ordinal(lane->family));
  iree_status_t status = iree_hal_semaphore_create(
      context->device, queue_family_affinity, /*initial_value=*/0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &submission->completion_semaphore);
  if (!iree_status_is_ok(status)) {
    iree_allocator_free(context->host_allocator, submission);
    return status;
  }

  submission->stream = stream;
  iree_hal_streaming_stream_retain(stream);

  iree_async_proactor_pool_t* proactor_pool =
      hrx_device_proactor_pool(context->device_entry->hrx_device);
  status = iree_async_proactor_pool_get(proactor_pool, /*index=*/0,
                                        &submission->observer_proactor);
  if (!iree_status_is_ok(status)) {
    iree_hal_streaming_destroy_value_wait_submissions(context, submission);
    return status;
  }
  iree_async_proactor_retain(submission->observer_proactor);
  submission->observer_semaphore =
      (iree_async_semaphore_t*)submission->completion_semaphore;
  submission->observer_value = 1;
  iree_hal_streaming_initialize_value_wait_observer_operation(submission);

  iree_slim_mutex_lock(&context->value_wait_lane_mutex);
  if (context->value_wait_lanes_shutting_down) {
    iree_slim_mutex_unlock(&context->value_wait_lane_mutex);
    iree_hal_streaming_destroy_value_wait_submissions(context, submission);
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "value-wait lane context is shutting down");
  }
  int32_t observer_count = iree_atomic_load(
      &context->active_value_wait_observer_count, iree_memory_order_acquire);
  while (true) {
    if (IREE_UNLIKELY(observer_count == INT32_MAX)) {
      iree_slim_mutex_unlock(&context->value_wait_lane_mutex);
      iree_hal_streaming_destroy_value_wait_submissions(context, submission);
      return iree_make_status(
          IREE_STATUS_RESOURCE_EXHAUSTED,
          "too many active value-wait completion observers");
    }
    if (iree_atomic_compare_exchange_weak(
            &context->active_value_wait_observer_count, &observer_count,
            observer_count + 1, iree_memory_order_acq_rel,
            iree_memory_order_acquire)) {
      break;
    }
  }
  submission->observer_prev = NULL;
  submission->observer_next = context->active_value_wait_observers;
  if (submission->observer_next) {
    submission->observer_next->observer_prev = submission;
  }
  context->active_value_wait_observers = submission;
  submission->observer_active = true;

  // Submit while the context mutex pins the record. Native semaphore-wait
  // submission never invokes the user callback inline; teardown therefore
  // cannot cancel an operation that has not yet been accepted by the proactor.
  status = iree_async_proactor_submit_one(submission->observer_proactor,
                                          &submission->observer_operation.base);
  if (!iree_status_is_ok(status)) {
    iree_hal_streaming_remove_value_wait_observer_locked(context, submission);
    iree_slim_mutex_unlock(&context->value_wait_lane_mutex);
    iree_hal_streaming_destroy_value_wait_submissions(context, submission);
    iree_hal_streaming_finish_value_wait_observer(context);
    return status;
  }
  iree_slim_mutex_unlock(&context->value_wait_lane_mutex);

  *out_submission = submission;
  return iree_ok_status();
}

bool iree_hal_streaming_value_wait_lane_accepts_submission(
    iree_hal_streaming_context_t* context,
    const iree_hal_streaming_value_wait_lane_t* lane) {
  iree_slim_mutex_lock(&context->value_wait_lane_mutex);
  const bool accepts =
      !context->value_wait_lanes_shutting_down && !lane->has_failed_submission;
  iree_slim_mutex_unlock(&context->value_wait_lane_mutex);
  return accepts;
}

// Returns an unsubmitted lane to its prior state. A lane taken from the
// pending list keeps any unresolved record so a rejected later submission
// cannot make the still-occupied queue available to another stream.
void iree_hal_streaming_release_value_wait_lane(
    iree_hal_streaming_context_t* context,
    iree_hal_streaming_value_wait_lane_t* lane) {
  if (!lane) {
    return;
  }
  bool destroy_lane = false;
  iree_slim_mutex_lock(&context->value_wait_lane_mutex);
  IREE_ASSERT(
      lane->list_state == IREE_HAL_STREAMING_VALUE_WAIT_LANE_LIST_STATE_NONE,
      "released value-wait lane must be caller-owned");
  if (lane->restore_pending && lane->submission_count != 0) {
    lane->restore_pending = false;
    iree_hal_streaming_insert_value_wait_lane_locked(
        context, lane, IREE_HAL_STREAMING_VALUE_WAIT_LANE_LIST_STATE_PENDING);
  } else if (lane->has_failed_submission) {
    lane->restore_pending = false;
    IREE_ASSERT(context->live_value_wait_submission_count >=
                    lane->retired_failure_count,
                "value-wait retained failure count underflow");
    context->live_value_wait_submission_count -= lane->retired_failure_count;
    destroy_lane = true;
  } else if (context->idle_value_wait_lane_count <
             IREE_HAL_STREAMING_VALUE_WAIT_IDLE_LANE_LIMIT) {
    IREE_ASSERT(!lane->retired_failure_head && lane->retired_failure_count == 0,
                "recyclable value-wait lane cannot retain failed proofs");
    lane->restore_pending = false;
    lane->owner_stream_id = 0;
    iree_hal_streaming_insert_value_wait_lane_locked(
        context, lane, IREE_HAL_STREAMING_VALUE_WAIT_LANE_LIST_STATE_IDLE);
  } else {
    destroy_lane = true;
  }
  iree_slim_mutex_unlock(&context->value_wait_lane_mutex);
  if (destroy_lane) {
    lane->next = NULL;
    lane->prev = NULL;
    iree_hal_streaming_destroy_value_wait_lanes(context, lane);
  }
}

void iree_hal_streaming_publish_pending_value_wait_lane(
    iree_hal_streaming_context_t* context,
    iree_hal_streaming_value_wait_lane_t* lane,
    iree_hal_streaming_value_wait_submission_t* submission) {
  IREE_ASSERT_ARGUMENT(submission);
  iree_slim_mutex_lock(&context->value_wait_lane_mutex);
  IREE_ASSERT(!context->value_wait_lanes_shutting_down,
              "publication cannot race context teardown");
  IREE_ASSERT(submission->state ==
                  IREE_HAL_STREAMING_VALUE_WAIT_SUBMISSION_STATE_PREPARED,
              "value-wait submission must publish exactly once");
  IREE_ASSERT(
      lane->list_state == IREE_HAL_STREAMING_VALUE_WAIT_LANE_LIST_STATE_NONE,
      "published value-wait lane must be caller-owned");
  IREE_ASSERT(!submission->is_terminal,
              "lane gate must serialize completion with publication");
  submission->state = IREE_HAL_STREAMING_VALUE_WAIT_SUBMISSION_STATE_PUBLISHED;
  submission->next = NULL;
  submission->prev = lane->submission_tail;
  if (lane->submission_tail) {
    lane->submission_tail->next = submission;
  } else {
    lane->submission_head = submission;
  }
  lane->submission_tail = submission;
  ++lane->submission_count;
  ++context->live_value_wait_submission_count;
  if (context->live_value_wait_submission_count >
      context->peak_value_wait_submission_count) {
    context->peak_value_wait_submission_count =
        context->live_value_wait_submission_count;
  }
  if (submission->has_failed) {
    lane->has_failed_submission = true;
  }
  lane->restore_pending = false;
  iree_hal_streaming_insert_value_wait_lane_locked(
      context, lane, IREE_HAL_STREAMING_VALUE_WAIT_LANE_LIST_STATE_PENDING);
  iree_slim_mutex_unlock(&context->value_wait_lane_mutex);
}

void iree_hal_streaming_reject_value_wait_submission(
    iree_hal_streaming_context_t* context,
    iree_hal_streaming_value_wait_submission_t* submission) {
  if (!submission) {
    return;
  }
  bool observer_active = false;
  iree_slim_mutex_lock(&context->value_wait_lane_mutex);
  IREE_ASSERT(submission->state ==
                  IREE_HAL_STREAMING_VALUE_WAIT_SUBMISSION_STATE_PREPARED,
              "value-wait submission must reject exactly once");
  submission->state = IREE_HAL_STREAMING_VALUE_WAIT_SUBMISSION_STATE_REJECTED;
  observer_active = submission->observer_active;
  iree_slim_mutex_unlock(&context->value_wait_lane_mutex);

  if (observer_active) {
    // Marking REJECTED happens-before this failure can enqueue the callback.
    // The callback becomes the final owner of the record.
    iree_hal_semaphore_fail(
        submission->completion_semaphore,
        iree_make_status(IREE_STATUS_CANCELLED,
                         "value-wait submission rejected before acceptance"));
  } else {
    // Completion-before-rejection or observer infrastructure failure left the
    // caller as the sole owner; there is no callback that could race release.
    iree_hal_streaming_destroy_value_wait_submissions(context, submission);
  }
}

static bool iree_hal_streaming_no_active_value_wait_observers(void* user_data) {
  iree_hal_streaming_context_t* context =
      (iree_hal_streaming_context_t*)user_data;
  iree_slim_mutex_lock(&context->value_wait_lane_mutex);
  const bool no_active_observers =
      iree_atomic_load(&context->active_value_wait_observer_count,
                       iree_memory_order_acquire) == 0;
  iree_slim_mutex_unlock(&context->value_wait_lane_mutex);
  return no_active_observers;
}

void iree_hal_streaming_value_wait_lanes_deinitialize(
    iree_hal_streaming_context_t* context) {
  // Phase 1 closes publication and cancels only the asynchronous observers.
  // Their callbacks retain no context reference; this explicit join keeps the
  // context and every embedded operation alive without creating a cycle.
  iree_slim_mutex_lock(&context->value_wait_lane_mutex);
  context->value_wait_lanes_shutting_down = true;
  for (iree_hal_streaming_value_wait_submission_t* submission =
           context->active_value_wait_observers;
       submission; submission = submission->observer_next) {
    IREE_ASSERT(!submission->cancellation_requested,
                "value-wait observer cancellation must be requested once");
    submission->cancellation_requested = true;
    // Native SEMAPHORE_WAIT cancellation only publishes to an MPSC queue; the
    // user callback cannot run inline. Holding the lane mutex pins the record
    // through this call and closes the remove/free gap.
    iree_status_t cancel_status = iree_async_proactor_cancel(
        submission->observer_proactor, &submission->observer_operation.base);
    if (!iree_status_is_ok(cancel_status)) {
      iree_status_free(cancel_status);
      // All native proactors support SEMAPHORE_WAIT cancellation. Failing the
      // private semaphore is a shutdown-only fallback that retires the observer
      // but is never used as lane terminal proof by the callback.
      iree_hal_semaphore_fail(
          submission->completion_semaphore,
          iree_make_status(IREE_STATUS_CANCELLED,
                           "value-wait observer cancelled during shutdown"));
    }
  }
  iree_slim_mutex_unlock(&context->value_wait_lane_mutex);
  iree_notification_await(&context->value_wait_observer_notification,
                          iree_hal_streaming_no_active_value_wait_observers,
                          context, iree_infinite_timeout());

  // Phase 2 hands pending queue shutdown to the backend with record/semaphore
  // storage intact. Only after queue release returns may those proofs be freed.
  iree_slim_mutex_lock(&context->value_wait_lane_mutex);
  IREE_ASSERT(!context->active_value_wait_observers,
              "all value-wait observer callbacks must be joined");
  iree_hal_streaming_value_wait_lane_t* idle_lanes =
      context->idle_value_wait_lanes;
  iree_hal_streaming_value_wait_lane_t* pending_lanes =
      context->pending_value_wait_lanes;
  iree_hal_streaming_value_wait_submission_t* shutdown_submissions =
      context->shutdown_value_wait_submissions;
  iree_host_size_t teardown_live_submission_count = 0;
  for (iree_hal_streaming_value_wait_lane_t* lane = pending_lanes; lane;
       lane = lane->next) {
    teardown_live_submission_count +=
        lane->submission_count + lane->retired_failure_count;
  }
  for (iree_hal_streaming_value_wait_lane_t* lane = idle_lanes; lane;
       lane = lane->next) {
    IREE_ASSERT(lane->submission_count == 0 &&
                    lane->retired_failure_count == 0 &&
                    !lane->submission_head && !lane->retired_failure_head,
                "idle value-wait lane cannot own submission records");
  }
  IREE_ASSERT(context->live_value_wait_submission_count ==
                  teardown_live_submission_count,
              "value-wait live record counter must match teardown ownership");
  context->idle_value_wait_lanes = NULL;
  context->idle_value_wait_lane_count = 0;
  context->pending_value_wait_lanes = NULL;
  context->shutdown_value_wait_submissions = NULL;
  context->live_value_wait_submission_count = 0;
  iree_slim_mutex_unlock(&context->value_wait_lane_mutex);

  for (iree_hal_streaming_value_wait_lane_t* lane = pending_lanes; lane;
       lane = lane->next) {
    iree_hal_streaming_value_wait_submission_t* record =
        lane->submission_head ? lane->submission_head
                              : lane->retired_failure_head;
    if (record && record->stream) {
      iree_slim_mutex_lock(&record->stream->mutex);
      if (record->stream->value_wait_lane == lane) {
        record->stream->value_wait_lane = NULL;
      }
      iree_slim_mutex_unlock(&record->stream->mutex);
    }
  }
  iree_hal_streaming_destroy_value_wait_lanes(context, pending_lanes);
  iree_hal_streaming_destroy_value_wait_lanes(context, idle_lanes);
  iree_hal_streaming_destroy_value_wait_submissions(context,
                                                    shutdown_submissions);
  iree_notification_deinitialize(&context->value_wait_observer_notification);
  iree_slim_mutex_deinitialize(&context->value_wait_lane_mutex);
}

iree_status_t iree_hal_streaming_submission_prepare_locked(
    iree_hal_streaming_stream_t* stream, iree_hal_queue_t* queue,
    uint64_t signal_value, iree_hal_streaming_submission_flags_t flags,
    iree_hal_streaming_submission_t* out_submission) {
  *out_submission = (iree_hal_streaming_submission_t){.queue = queue};
  out_submission->signal_semaphores[0] = stream->timeline_semaphore;
  out_submission->signal_values[0] = signal_value;
  out_submission->signals = (iree_hal_semaphore_list_t){
      .count = 1,
      .semaphores = out_submission->signal_semaphores,
      .payload_values = out_submission->signal_values,
  };
  if (!stream->value_wait_lane &&
      !iree_any_bit_set(flags,
                        IREE_HAL_STREAMING_SUBMISSION_FLAG_INDEPENDENT)) {
    return iree_ok_status();
  }

  iree_hal_streaming_context_t* context = stream->context;
  iree_hal_streaming_value_wait_lane_t* lane = stream->value_wait_lane;
  if (lane) {
    // The stream mutex pins this link against its completion callback. Remove
    // it from the pending list until this acceptance transaction ends.
    iree_slim_mutex_lock(&context->value_wait_lane_mutex);
    iree_hal_streaming_remove_value_wait_lane_locked(context, lane);
    lane->restore_pending = true;
    iree_slim_mutex_unlock(&context->value_wait_lane_mutex);
  } else {
    IREE_RETURN_IF_ERROR(iree_hal_streaming_acquire_value_wait_lane(
        context, iree_hal_queue_family(stream->queue),
        iree_hal_queue_priority(stream->queue),
        iree_hal_queue_features(stream->queue),
        iree_hal_queue_execution_resources(stream->queue), stream->queue,
        stream->stream_id, &lane));
    stream->value_wait_lane = lane;
  }

  iree_status_t status = iree_hal_streaming_prepare_value_wait_submission(
      context, lane, stream, &out_submission->completion);
  if (iree_status_is_ok(status)) {
    iree_slim_mutex_lock(&lane->submission_mutex);
    if (!iree_hal_streaming_value_wait_lane_accepts_submission(context, lane)) {
      status = iree_make_status(IREE_STATUS_ABORTED,
                                "stream execution queue has failed");
      iree_slim_mutex_unlock(&lane->submission_mutex);
    }
  }
  if (!iree_status_is_ok(status)) {
    iree_hal_streaming_reject_value_wait_submission(context,
                                                    out_submission->completion);
    out_submission->completion = NULL;
    if (lane->submission_count == 0) {
      stream->value_wait_lane = NULL;
    }
    iree_hal_streaming_release_value_wait_lane(context, lane);
    return status;
  }

  // A cooperative realization already belongs exclusively to this stream.
  // Keep it for cooperative dispatch while the same lease tracks completion.
  out_submission->queue = queue == stream->queue ? lane->queue : queue;
  out_submission->signal_semaphores[1] =
      out_submission->completion->completion_semaphore;
  out_submission->signal_values[1] = 1;
  out_submission->signals.count = 2;
  return iree_ok_status();
}

void iree_hal_streaming_submission_finish_locked(
    iree_hal_streaming_stream_t* stream,
    iree_hal_streaming_submission_t* submission, bool accepted) {
  if (accepted) {
    stream->pending_value = submission->signal_values[0];
  }
  if (!submission->completion) {
    return;
  }

  iree_hal_streaming_context_t* context = stream->context;
  iree_hal_streaming_value_wait_submission_t* completion =
      submission->completion;
  iree_hal_streaming_value_wait_lane_t* lane = completion->lane;
  if (accepted) {
    iree_hal_streaming_publish_pending_value_wait_lane(context, lane,
                                                       completion);
  } else {
    iree_hal_streaming_reject_value_wait_submission(context, completion);
  }
  iree_slim_mutex_unlock(&lane->submission_mutex);
  if (!accepted) {
    if (lane->submission_count == 0) {
      stream->value_wait_lane = NULL;
    }
    iree_hal_streaming_release_value_wait_lane(context, lane);
  }
  submission->completion = NULL;
}

iree_status_t iree_hal_streaming_submission_release_system_locked(
    iree_hal_streaming_stream_t* stream) {
  if (stream->native_release_state == IREE_HAL_STREAMING_NATIVE_RELEASE_EMPTY) {
    return iree_ok_status();
  }
  if (stream->native_release_state ==
      IREE_HAL_STREAMING_NATIVE_RELEASE_SYSTEM) {
    // The stream mutex pins an active independent lane. A retired lane has
    // already completed its native publication and needs no further flush.
    iree_hal_queue_t* queue = stream->value_wait_lane
                                  ? stream->value_wait_lane->queue
                                  : stream->queue;
    IREE_RETURN_IF_ERROR(iree_hal_queue_flush(queue));
    stream->native_release_state = IREE_HAL_STREAMING_NATIVE_RELEASE_EMPTY;
    return iree_ok_status();
  }

  uint64_t wait_value = 0;
  uint64_t signal_value = 0;
  IREE_RETURN_IF_ERROR(iree_hal_streaming_stream_reserve_next_value_locked(
      stream, &wait_value, &signal_value));
  const iree_hal_semaphore_list_t waits = {
      .count = 1,
      .semaphores = &stream->timeline_semaphore,
      .payload_values = &wait_value,
  };
  static const iree_hal_barrier_t release = {
      .source_stage_mask = IREE_HAL_EXECUTION_STAGE_DISPATCH,
      .flags = IREE_HAL_BARRIER_FLAG_RELEASE_SYSTEM_SCOPE,
  };
  static const iree_hal_barrier_list_t before = {
      .count = 1,
      .values = &release,
  };
  static const iree_hal_barrier_list_t after = {0};
  static const iree_hal_queue_barriers_t barriers = {
      .before = &before,
      .after = &after,
  };
  iree_hal_streaming_submission_t submission;
  iree_status_t status = iree_hal_streaming_submission_prepare_locked(
      stream, stream->queue, signal_value,
      IREE_HAL_STREAMING_SUBMISSION_FLAG_NONE, &submission);
  if (iree_status_is_ok(status)) {
    status =
        iree_hal_queue_barrier(submission.queue, waits, submission.signals,
                               &barriers, IREE_HAL_QUEUE_BARRIER_FLAG_NONE);
  }
  iree_hal_streaming_submission_finish_locked(stream, &submission,
                                              iree_status_is_ok(status));
  if (iree_status_is_ok(status)) {
    stream->native_release_state = IREE_HAL_STREAMING_NATIVE_RELEASE_EMPTY;
    status = iree_hal_queue_flush(submission.queue);
  }
  return status;
}
