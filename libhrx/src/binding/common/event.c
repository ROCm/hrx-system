// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/internal.h"
#include "iree/base/internal/math.h"

//===----------------------------------------------------------------------===//
// Event management
//===----------------------------------------------------------------------===//

iree_status_t iree_hal_streaming_event_create(
    iree_hal_streaming_context_t* context,
    iree_hal_streaming_event_flags_t flags, iree_allocator_t host_allocator,
    iree_hal_streaming_event_t** out_event) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(out_event);
  *out_event = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_streaming_event_t* event = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_allocator_malloc(host_allocator, sizeof(*event), (void**)&event));

  // Initialize event.
  iree_atomic_ref_count_init(&event->ref_count);
  event->flags = flags;
  iree_slim_mutex_initialize(&event->mutex);
  event->recorded_point = (iree_hal_streaming_recorded_point_t){0};
  event->recording_stream = NULL;
  event->context = context;
  iree_hal_streaming_context_retain(context);
  event->ipc_handle = NULL;
  event->capture_graph = NULL;
  event->capture_id = 0;
  event->capture_mode = IREE_HAL_STREAMING_CAPTURE_MODE_GLOBAL;
  event->capture_owner_thread_id = 0;
  event->capture_dependencies = NULL;
  event->capture_dependency_count = 0;
  event->capture_dependency_capacity = 0;
  event->host_allocator = host_allocator;

  *out_event = event;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_hal_streaming_event_destroy(
    iree_hal_streaming_event_t* event) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Release the recorded point.
  iree_hal_streaming_event_release_recorded_point(&event->recorded_point);

  // Release recording stream reference.
  iree_hal_streaming_stream_release(event->recording_stream);

  // Release context.
  iree_hal_streaming_context_release(event->context);

  iree_hal_streaming_graph_release(event->capture_graph);
  iree_allocator_free(event->host_allocator, event->capture_dependencies);

  // Free event memory.
  iree_slim_mutex_deinitialize(&event->mutex);
  iree_allocator_t host_allocator = event->host_allocator;
  iree_allocator_free(host_allocator, event);

  IREE_TRACE_ZONE_END(z0);
}

void iree_hal_streaming_event_retain(iree_hal_streaming_event_t* event) {
  if (event) {
    iree_atomic_ref_count_inc(&event->ref_count);
  }
}

void iree_hal_streaming_event_release(iree_hal_streaming_event_t* event) {
  if (event && iree_atomic_ref_count_dec(&event->ref_count) == 1) {
    iree_hal_streaming_event_destroy(event);
  }
}

void iree_hal_streaming_event_acquire_recorded_point(
    iree_hal_streaming_event_t* event,
    iree_hal_streaming_recorded_point_t* out_point) {
  iree_slim_mutex_lock(&event->mutex);
  *out_point = event->recorded_point;
  iree_hal_semaphore_retain(out_point->semaphore);
  iree_hal_streaming_event_timestamp_slot_retain(out_point->timestamp_slot);
  iree_slim_mutex_unlock(&event->mutex);
}

void iree_hal_streaming_event_release_recorded_point(
    iree_hal_streaming_recorded_point_t* point) {
  // Released before the semaphore: the slot's retirement condition is the
  // point's own semaphore payload, so the pool reads it while this reference
  // still holds the semaphore up.
  iree_hal_streaming_event_timestamp_slot_release(
      point->timestamp_slot, point->semaphore, point->value);
  iree_hal_semaphore_release(point->semaphore);
  *point = (iree_hal_streaming_recorded_point_t){0};
}

iree_hal_streaming_event_displaced_capture_t
iree_hal_streaming_event_commit_recorded_point(
    iree_hal_streaming_event_t* event,
    iree_hal_streaming_recorded_point_t point) {
  iree_slim_mutex_lock(&event->mutex);
  iree_hal_streaming_recorded_point_t previous = event->recorded_point;
  event->recorded_point = point;
  iree_hal_streaming_event_displaced_capture_t displaced_capture = {
      .graph = event->capture_graph,
      .recording_stream = event->recording_stream,
  };
  event->capture_graph = NULL;
  event->recording_stream = NULL;
  event->capture_id = 0;
  event->capture_mode = IREE_HAL_STREAMING_CAPTURE_MODE_GLOBAL;
  event->capture_owner_thread_id = 0;
  iree_slim_mutex_unlock(&event->mutex);
  // Dropped outside the lock: returning a tick slot to its pool takes the pool
  // mutex, and that mutex and this one are both leaves that no path holds at
  // the same time.
  iree_hal_streaming_event_release_recorded_point(&previous);
  return displaced_capture;
}

void iree_hal_streaming_event_release_displaced_capture(
    iree_hal_streaming_event_displaced_capture_t* displaced_capture) {
  iree_hal_streaming_stream_release(displaced_capture->recording_stream);
  iree_hal_streaming_graph_release(displaced_capture->graph);
  *displaced_capture = (iree_hal_streaming_event_displaced_capture_t){0};
}

bool iree_hal_streaming_event_has_capture_graph(
    iree_hal_streaming_event_t* event) {
  iree_slim_mutex_lock(&event->mutex);
  const bool has_capture_graph = event->capture_graph != NULL;
  iree_slim_mutex_unlock(&event->mutex);
  return has_capture_graph;
}

void iree_hal_streaming_event_acquire_capture_association(
    iree_hal_streaming_event_t* event,
    iree_hal_streaming_event_capture_association_t* out_association) {
  *out_association = (iree_hal_streaming_event_capture_association_t){0};
  iree_slim_mutex_lock(&event->mutex);
  if (event->capture_graph) {
    out_association->graph = event->capture_graph;
    out_association->recording_stream = event->recording_stream;
    out_association->capture_id = event->capture_id;
    out_association->capture_mode = event->capture_mode;
    out_association->capture_owner_thread_id = event->capture_owner_thread_id;
    iree_hal_streaming_graph_retain(out_association->graph);
    iree_hal_streaming_stream_retain(out_association->recording_stream);
  }
  iree_slim_mutex_unlock(&event->mutex);
}

bool iree_hal_streaming_event_capture_association_is_active(
    const iree_hal_streaming_event_capture_association_t* association) {
  if (!association->graph || !association->recording_stream ||
      association->capture_id == 0) {
    return false;
  }
  iree_hal_streaming_stream_t* stream = association->recording_stream;
  iree_slim_mutex_lock(&stream->mutex);
  const bool is_active =
      stream->capture_status == IREE_HAL_STREAMING_CAPTURE_STATUS_ACTIVE &&
      stream->capture_graph == association->graph &&
      stream->capture_id == association->capture_id;
  iree_slim_mutex_unlock(&stream->mutex);
  return is_active;
}

void iree_hal_streaming_event_release_capture_association(
    iree_hal_streaming_event_capture_association_t* association) {
  iree_hal_streaming_stream_release(association->recording_stream);
  iree_hal_streaming_graph_release(association->graph);
  *association = (iree_hal_streaming_event_capture_association_t){0};
}

static void iree_hal_streaming_event_set_capture_association(
    iree_hal_streaming_event_t* event,
    iree_hal_streaming_stream_t* recording_stream,
    iree_hal_streaming_graph_t** out_previous_graph,
    iree_hal_streaming_stream_t** out_previous_stream) {
  iree_hal_streaming_graph_t* graph = recording_stream->capture_graph;
  const unsigned long long capture_id = recording_stream->capture_id;
  IREE_ASSERT(graph);
  IREE_ASSERT(capture_id != 0);

  iree_hal_streaming_graph_retain(graph);
  iree_hal_streaming_stream_retain(recording_stream);
  iree_slim_mutex_lock(&event->mutex);
  *out_previous_graph = event->capture_graph;
  *out_previous_stream = event->recording_stream;
  event->capture_graph = graph;
  event->recording_stream = recording_stream;
  event->capture_id = capture_id;
  event->capture_mode = recording_stream->capture_mode;
  event->capture_owner_thread_id = recording_stream->capture_owner_thread_id;
  iree_slim_mutex_unlock(&event->mutex);
}

iree_status_t iree_hal_streaming_event_record_after_streams(
    iree_hal_streaming_event_t* event,
    iree_hal_streaming_stream_t* const* streams, iree_host_size_t stream_count,
    iree_hal_streaming_operation_timeline_t* additional_timeline) {
  IREE_ASSERT_ARGUMENT(event);
  if (IREE_UNLIKELY(stream_count > 0 && !streams)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "stream list storage is null");
  }

  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_streaming_context_t* context = event->context;
  iree_slim_mutex_lock(&context->event_record_mutex);

  iree_hal_semaphore_t** wait_semaphores = NULL;
  uint64_t* wait_values = NULL;
  iree_status_t status = iree_ok_status();
  uint64_t context_signal_value = 0;
  uint64_t additional_signal_value = 0;
  if (IREE_UNLIKELY(!context->event_record_timeline.semaphore)) {
    status = iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "context event record timeline is unavailable");
  } else if (IREE_UNLIKELY(context->event_record_timeline.pending_value ==
                           UINT64_MAX)) {
    status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "context event record timeline is exhausted");
  } else {
    context_signal_value = context->event_record_timeline.pending_value + 1;
  }
  if (iree_status_is_ok(status) && additional_timeline) {
    if (IREE_UNLIKELY(!additional_timeline->semaphore ||
                      additional_timeline->semaphore ==
                          context->event_record_timeline.semaphore)) {
      status =
          iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                           "additional event record timeline must be distinct");
    } else if (IREE_UNLIKELY(additional_timeline->pending_value ==
                             UINT64_MAX)) {
      status =
          iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                           "additional event record timeline is exhausted");
    } else {
      additional_signal_value = additional_timeline->pending_value + 1;
    }
  }

  iree_host_size_t wait_capacity = stream_count;
  if (iree_status_is_ok(status) &&
      context->event_record_timeline.pending_value > 0 &&
      IREE_UNLIKELY(
          !iree_host_size_checked_add(wait_capacity, 1, &wait_capacity))) {
    status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "event wait list size overflow");
  }
  if (iree_status_is_ok(status) && additional_timeline &&
      additional_timeline->pending_value > 0 &&
      IREE_UNLIKELY(
          !iree_host_size_checked_add(wait_capacity, 1, &wait_capacity))) {
    status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "event wait list size overflow");
  }
  if (iree_status_is_ok(status) && wait_capacity > 0) {
    iree_host_size_t semaphore_storage_size = 0;
    iree_host_size_t value_storage_size = 0;
    if (IREE_UNLIKELY(
            !iree_host_size_checked_mul(wait_capacity, sizeof(*wait_semaphores),
                                        &semaphore_storage_size) ||
            !iree_host_size_checked_mul(wait_capacity, sizeof(*wait_values),
                                        &value_storage_size))) {
      status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "event wait list size overflow");
    }
    if (iree_status_is_ok(status)) {
      status =
          iree_allocator_malloc(context->host_allocator, semaphore_storage_size,
                                (void**)&wait_semaphores);
    }
    if (iree_status_is_ok(status)) {
      status = iree_allocator_malloc(context->host_allocator,
                                     value_storage_size, (void**)&wait_values);
    }
  }

  // Reject invalid membership and capture state before flushing any source.
  for (iree_host_size_t i = 0; i < stream_count && iree_status_is_ok(status);
       ++i) {
    iree_hal_streaming_stream_t* stream = streams[i];
    if (IREE_UNLIKELY(!stream)) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "event record stream is null");
      break;
    }
    iree_slim_mutex_lock(&stream->mutex);
    const bool belongs_to_context = stream->context == context;
    const bool is_capturing =
        stream->capture_status != IREE_HAL_STREAMING_CAPTURE_STATUS_NONE;
    iree_slim_mutex_unlock(&stream->mutex);
    if (IREE_UNLIKELY(!belongs_to_context)) {
      status = iree_make_status(IREE_STATUS_INCOMPATIBLE,
                                "event record stream belongs to a different "
                                "context");
    } else if (IREE_UNLIKELY(is_capturing)) {
      status = iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "context event recording is unavailable during stream capture");
    }
  }

  for (iree_host_size_t i = 0; i < stream_count && iree_status_is_ok(status);
       ++i) {
    status = iree_hal_streaming_stream_flush(streams[i]);
  }

  iree_host_size_t wait_count = 0;
  if (iree_status_is_ok(status) &&
      context->event_record_timeline.pending_value > 0) {
    wait_semaphores[wait_count] = context->event_record_timeline.semaphore;
    wait_values[wait_count] = context->event_record_timeline.pending_value;
    ++wait_count;
  }
  if (iree_status_is_ok(status) && additional_timeline &&
      additional_timeline->pending_value > 0) {
    wait_semaphores[wait_count] = additional_timeline->semaphore;
    wait_values[wait_count] = additional_timeline->pending_value;
    ++wait_count;
  }
  for (iree_host_size_t i = 0; i < stream_count && iree_status_is_ok(status);
       ++i) {
    iree_hal_streaming_stream_t* stream = streams[i];
    iree_slim_mutex_lock(&stream->mutex);
    if (IREE_UNLIKELY(stream->context != context)) {
      status = iree_make_status(IREE_STATUS_INCOMPATIBLE,
                                "event record stream belongs to a different "
                                "context");
    } else if (IREE_UNLIKELY(stream->capture_status !=
                             IREE_HAL_STREAMING_CAPTURE_STATUS_NONE)) {
      status = iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "context event recording is unavailable during stream capture");
    } else if (stream->pending_value > 0) {
      wait_semaphores[wait_count] = stream->timeline_semaphore;
      wait_values[wait_count] = stream->pending_value;
      ++wait_count;
    }
    iree_slim_mutex_unlock(&stream->mutex);
  }

  iree_hal_streaming_event_displaced_capture_t displaced_capture = {0};
  if (iree_status_is_ok(status)) {
    const iree_hal_semaphore_list_t waits = {
        .count = wait_count,
        .semaphores = wait_semaphores,
        .payload_values = wait_values,
    };
    iree_hal_semaphore_t* signal_semaphores[2] = {
        context->event_record_timeline.semaphore,
        additional_timeline ? additional_timeline->semaphore : NULL,
    };
    uint64_t signal_values[2] = {
        context_signal_value,
        additional_signal_value,
    };
    const iree_hal_semaphore_list_t signals = {
        .count = additional_timeline ? 2 : 1,
        .semaphores = signal_semaphores,
        .payload_values = signal_values,
    };
    iree_hal_streaming_recorded_point_t recorded_point = {
        .semaphore = context->event_record_timeline.semaphore,
        .value = context_signal_value,
    };
    status = iree_hal_streaming_event_enqueue_record(
        event, context, context->queue, waits, signals, &recorded_point);
    if (iree_status_is_ok(status)) {
      context->event_record_timeline.pending_value = context_signal_value;
      if (additional_timeline) {
        additional_timeline->pending_value = additional_signal_value;
      }
      displaced_capture =
          iree_hal_streaming_event_commit_recorded_point(event, recorded_point);
    }
  }

  iree_slim_mutex_unlock(&context->event_record_mutex);
  iree_hal_streaming_event_release_displaced_capture(&displaced_capture);
  iree_allocator_free(context->host_allocator, wait_values);
  iree_allocator_free(context->host_allocator, wait_semaphores);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

// Returns whether |point| has been reached. A point naming no timeline has no
// submitted record and reads as reached. Borrows |point|: the reference taken
// when the point was acquired stays with the acquirer, which releases it.
static iree_status_t iree_hal_streaming_recorded_point_query(
    const iree_hal_streaming_recorded_point_t* point, bool* out_reached) {
  *out_reached = true;
  if (!point->semaphore) return iree_ok_status();
  uint64_t current_value = 0;
  IREE_RETURN_IF_ERROR(
      iree_hal_semaphore_query(point->semaphore, &current_value));
  *out_reached = current_value >= point->value;
  return iree_ok_status();
}

iree_status_t iree_hal_streaming_event_query(iree_hal_streaming_event_t* event,
                                             int* status) {
  IREE_ASSERT_ARGUMENT(event);
  IREE_ASSERT_ARGUMENT(status);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_streaming_recorded_point_t recorded_point;
  iree_hal_streaming_event_acquire_recorded_point(event, &recorded_point);
  bool reached = false;
  iree_status_t query_status =
      iree_hal_streaming_recorded_point_query(&recorded_point, &reached);
  iree_hal_streaming_event_release_recorded_point(&recorded_point);
  IREE_RETURN_AND_END_ZONE_IF_ERROR(z0, query_status);

  // 0=complete, 1=not complete.
  *status = reached ? 0 : 1;

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

iree_status_t iree_hal_streaming_event_enqueue_record(
    iree_hal_streaming_event_t* event, iree_hal_streaming_context_t* context,
    iree_hal_queue_t* queue, iree_hal_semaphore_list_t wait_semaphores,
    iree_hal_semaphore_list_t signal_semaphores,
    iree_hal_streaming_recorded_point_t* point) {
  // The slot a timed record captures into is suballocated from |context|'s
  // pool and outlives the record on the point the event holds. Nothing the
  // point names keeps that pool alive - a slot reference is a count on the
  // slot alone - so only the reference the event holds on its own context
  // does, and a slot drawn from any other context's pool can be left naming
  // storage that context freed. Ticks are counted on the recording device's
  // clock besides, and iree_hal_streaming_event_elapsed_time converts them
  // with the event's own context domain.
  if (event->context != context) {
    return iree_make_status(
        IREE_STATUS_INCOMPATIBLE,
        "an event can only be recorded with the context that created it");
  }

  const bool captures_tick =
      context->timestamp_domain.frequency_hz != 0 &&
      !(event->flags & IREE_HAL_STREAMING_EVENT_FLAG_DISABLE_TIMING);

  iree_hal_streaming_event_timestamp_slot_t* slot = NULL;
  if (captures_tick) {
    IREE_RETURN_IF_ERROR(iree_hal_streaming_event_timestamp_pool_acquire(
        &context->timestamp_pool, &slot));
  }

  const iree_status_t status =
      slot ? iree_hal_queue_timestamp(
                 queue, wait_semaphores, signal_semaphores,
                 iree_hal_streaming_event_timestamp_slot_buffer(slot),
                 iree_hal_streaming_event_timestamp_slot_offset(slot),
                 IREE_HAL_TIMESTAMP_FLAG_NONE)
           : iree_hal_queue_barrier(queue, wait_semaphores, signal_semaphores,
                                    IREE_HAL_QUEUE_BARRIER_FLAG_NONE);
  if (!iree_status_is_ok(status)) {
    // A rejected enqueue leaves no write outstanding against the slot, which
    // only this path can say; every other release names the point's own
    // retirement condition.
    iree_hal_streaming_event_timestamp_slot_release(slot, NULL, 0);
    return status;
  }

  // The point owns what it names from here.
  iree_hal_semaphore_retain(point->semaphore);
  point->timestamp_slot = slot;
  return iree_ok_status();
}

iree_status_t iree_hal_streaming_event_record(
    iree_hal_streaming_event_t* event, iree_hal_streaming_stream_t* stream) {
  IREE_ASSERT_ARGUMENT(event);
  IREE_ASSERT_ARGUMENT(stream);
  IREE_TRACE_ZONE_BEGIN(z0);

  // Check if we're capturing to a graph.
  if (stream->capture_status == IREE_HAL_STREAMING_CAPTURE_STATUS_ACTIVE) {
    if (event->capture_dependency_capacity < stream->capture_dependency_count) {
      IREE_RETURN_AND_END_ZONE_IF_ERROR(
          z0, iree_allocator_realloc(event->host_allocator,
                                     stream->capture_dependency_count *
                                         sizeof(*event->capture_dependencies),
                                     (void**)&event->capture_dependencies));
      event->capture_dependency_capacity = stream->capture_dependency_count;
    }
    if (stream->capture_dependency_count > 0) {
      memcpy(event->capture_dependencies, stream->capture_dependencies,
             stream->capture_dependency_count *
                 sizeof(*event->capture_dependencies));
    }
    event->capture_dependency_count = stream->capture_dependency_count;
    // A captured record produces no submission, so it leaves the recorded
    // point alone. The graph, source stream, and immutable session metadata are
    // published together for later wait and invalidation consumers.
    iree_hal_streaming_graph_t* previous_graph = NULL;
    iree_hal_streaming_stream_t* previous_stream = NULL;
    iree_hal_streaming_event_set_capture_association(
        event, stream, &previous_graph, &previous_stream);
    // No lock is held here, so displaced references can be released inline.
    iree_hal_streaming_stream_release(previous_stream);
    iree_hal_streaming_graph_release(previous_graph);
    IREE_TRACE_ZONE_END(z0);
    return iree_ok_status();
  }

  // Flush the stream to ensure all prior operations are submitted.
  IREE_RETURN_AND_END_ZONE_IF_ERROR(z0,
                                    iree_hal_streaming_stream_flush(stream));

  iree_slim_mutex_lock(&stream->mutex);

  // The record marks a point on the recording stream's timeline: the barrier
  // takes the stream's next value and signals it once everything already on the
  // stream has completed.
  uint64_t stream_wait_value = 0;
  uint64_t stream_signal_value = 0;
  iree_status_t status = iree_hal_streaming_stream_reserve_next_value_locked(
      stream, &stream_wait_value, &stream_signal_value);
  if (!iree_status_is_ok(status)) {
    iree_slim_mutex_unlock(&stream->mutex);
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  // A stream that has never submitted has nothing behind it to wait on.
  iree_hal_semaphore_list_t wait_semaphores = {
      .count = stream_wait_value > 0 ? 1 : 0,
      .semaphores = &stream->timeline_semaphore,
      .payload_values = &stream_wait_value,
  };
  iree_hal_semaphore_list_t signal_semaphores = {
      .count = 1,
      .semaphores = &stream->timeline_semaphore,
      .payload_values = &stream_signal_value,
  };

  // The record signals the stream's own timeline, so reaching the point is
  // exactly the stream reaching that value.
  iree_hal_streaming_recorded_point_t recorded_point = {
      .semaphore = stream->timeline_semaphore,
      .value = stream_signal_value,
      .ordered_after_stream_id = stream->stream_id,
      .ordered_after_stream_value = stream_signal_value,
  };
  iree_hal_streaming_event_displaced_capture_t displaced_capture = {0};
  status = iree_hal_streaming_event_enqueue_record(
      event, stream->context, stream->queue, wait_semaphores, signal_semaphores,
      &recorded_point);
  if (iree_status_is_ok(status)) {
    // The accepted enqueue owns the value it signals, so the timeline advances
    // here and stays advanced even when the flush below fails.
    stream->pending_value = stream_signal_value;
    // A rejected submission signals nothing, so the event keeps its old point
    // and stays associated with whatever capture it belonged to.
    displaced_capture =
        iree_hal_streaming_event_commit_recorded_point(event, recorded_point);
    status = iree_hal_queue_flush(stream->queue);
  }
  iree_slim_mutex_unlock(&stream->mutex);
  // Stream teardown re-enters the streaming layer, and the last reference to a
  // captured graph frees the allocations it owns, which synchronizes every
  // context and relocks this stream. Both displaced references are therefore
  // dropped outside this stream's mutex.
  iree_hal_streaming_event_release_displaced_capture(&displaced_capture);
  IREE_RETURN_AND_END_ZONE_IF_ERROR(z0, status);

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

iree_status_t iree_hal_streaming_event_synchronize(
    iree_hal_streaming_event_t* event) {
  IREE_ASSERT_ARGUMENT(event);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_streaming_recorded_point_t recorded_point;
  iree_hal_streaming_event_acquire_recorded_point(event, &recorded_point);
  // A point naming no timeline has no submitted record behind it and nothing to
  // wait for.
  iree_status_t status = iree_ok_status();
  if (recorded_point.semaphore) {
    status = iree_hal_semaphore_wait(
        recorded_point.semaphore, recorded_point.value, iree_infinite_timeout(),
        IREE_ASYNC_WAIT_FLAG_NONE);
  }
  iree_hal_streaming_event_release_recorded_point(&recorded_point);
  IREE_RETURN_AND_END_ZONE_IF_ERROR(z0, status);

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

float iree_hal_streaming_timestamp_domain_elapsed_ms(
    iree_hal_streaming_timestamp_domain_t domain, uint64_t start_tick,
    uint64_t stop_tick) {
  const uint64_t mask = domain.valid_bits == 64
                            ? UINT64_MAX
                            : ((1ull << domain.valid_bits) - 1ull);
  const uint64_t delta = (stop_tick - start_tick) & mask;
  // The counter's top bit, derived from the mask so it is defined at every
  // width the domain can carry, including none: there the mask takes the pair
  // to zero, which sits below this.
  const uint64_t sign_bit = (mask >> 1) + 1ull;
  // Formed as a double because the scaling below is floating point and the
  // result is a float. The offset itself fits int64_t at every width the
  // domain can carry: for delta >= sign_bit, mask - delta is at most
  // INT64_MAX, so -(mask - delta) - 1 bottoms out at exactly INT64_MIN.
  const double signed_delta =
      delta >= sign_bit ? -(double)(mask - delta) - 1.0 : (double)delta;
  return (float)(signed_delta * 1000.0 / (double)domain.frequency_hz);
}

// Reads the ticks the reached records |start_point| and |stop_point| captured
// and converts the interval between them with |domain|, writing |*out_ms| only
// when both reads succeed. Both points must name a tick slot, which the caller
// establishes before querying either record.
static iree_status_t iree_hal_streaming_recorded_point_pair_elapsed_ms(
    iree_hal_streaming_timestamp_domain_t domain,
    const iree_hal_streaming_recorded_point_t* start_point,
    const iree_hal_streaming_recorded_point_t* stop_point, float* out_ms) {
  uint64_t start_tick = 0;
  IREE_RETURN_IF_ERROR(
      iree_hal_buffer_map_read(iree_hal_streaming_event_timestamp_slot_buffer(
                                   start_point->timestamp_slot),
                               iree_hal_streaming_event_timestamp_slot_offset(
                                   start_point->timestamp_slot),
                               &start_tick, sizeof(start_tick)));
  uint64_t stop_tick = 0;
  IREE_RETURN_IF_ERROR(
      iree_hal_buffer_map_read(iree_hal_streaming_event_timestamp_slot_buffer(
                                   stop_point->timestamp_slot),
                               iree_hal_streaming_event_timestamp_slot_offset(
                                   stop_point->timestamp_slot),
                               &stop_tick, sizeof(stop_tick)));
  *out_ms = iree_hal_streaming_timestamp_domain_elapsed_ms(domain, start_tick,
                                                           stop_tick);
  return iree_ok_status();
}

iree_status_t iree_hal_streaming_event_elapsed_time(
    float* ms, iree_hal_streaming_event_t* start,
    iree_hal_streaming_event_t* stop,
    iree_hal_streaming_event_timing_t* out_timing) {
  IREE_ASSERT_ARGUMENT(ms);
  IREE_ASSERT_ARGUMENT(start);
  IREE_ASSERT_ARGUMENT(stop);
  IREE_ASSERT_ARGUMENT(out_timing);
  *out_timing = IREE_HAL_STREAMING_EVENT_TIMING_UNTIMED;

  // An event with timing disabled captures no tick, so there is no interval
  // between two such records to measure.
  if ((start->flags & IREE_HAL_STREAMING_EVENT_FLAG_DISABLE_TIMING) ||
      (stop->flags & IREE_HAL_STREAMING_EVENT_FLAG_DISABLE_TIMING)) {
    return iree_ok_status();
  }

  // A record made into a capture names a dependency frontier and no queue
  // point, so an event whose last record went into one names no time. The
  // associations are read ahead of the points because a submitted record
  // installs its point and clears its association together: a reader taking
  // the association first never finds it clear while the point it goes on to
  // read is still the one that capture left behind. Only the presence of an
  // association decides the outcome, so it is read without naming either
  // graph: this path holds no reference whose release could free a graph's
  // allocations and synchronize every context.
  if (iree_hal_streaming_event_has_capture_graph(start) ||
      iree_hal_streaming_event_has_capture_graph(stop)) {
    *out_timing = IREE_HAL_STREAMING_EVENT_TIMING_CAPTURED;
    return iree_ok_status();
  }

  // Each record is taken once and the interval is measured between exactly the
  // two records queried below, so a record landing on either event concurrently
  // cannot pair one record's tick with another record's point.
  iree_hal_streaming_recorded_point_t start_point;
  iree_hal_streaming_recorded_point_t stop_point;
  iree_hal_streaming_event_acquire_recorded_point(start, &start_point);
  iree_hal_streaming_event_acquire_recorded_point(stop, &stop_point);

  iree_status_t status = iree_ok_status();
  if (!start_point.semaphore || !stop_point.semaphore) {
    // An event with no submitted record names no point to measure from, which
    // is what the outcome already reads as.
  } else if (!start_point.timestamp_slot || !stop_point.timestamp_slot) {
    // Both records were submitted with timing enabled, and a record is made
    // only on a stream of its own event's context, so the one thing that can
    // leave a record without a tick is that context advertising no domain to
    // capture in. Answered before the timelines are queried: reaching a record
    // that captured nothing produces no measurement either.
    *out_timing = IREE_HAL_STREAMING_EVENT_TIMING_UNSUPPORTED;
  } else {
    // Both ticks are defined once both records have been reached. The stop
    // record is only queried once the start record is known to have been
    // reached, so an outstanding start reports the interval as incomplete
    // without touching the stop timeline.
    bool reached = false;
    status = iree_hal_streaming_recorded_point_query(&start_point, &reached);
    if (iree_status_is_ok(status) && reached) {
      status = iree_hal_streaming_recorded_point_query(&stop_point, &reached);
    }
    if (iree_status_is_ok(status)) {
      if (reached) {
        // Ticks are comparable only inside one device's domain, and this
        // converts with the start event's context domain. A record is made
        // only on a stream of its own event's context, and both bindings
        // refuse a pair whose events come from different contexts before it
        // reaches here, so both ticks were captured in that domain.
        status = iree_hal_streaming_recorded_point_pair_elapsed_ms(
            start->context->timestamp_domain, &start_point, &stop_point, ms);
        if (iree_status_is_ok(status)) {
          *out_timing = IREE_HAL_STREAMING_EVENT_TIMING_MEASURED;
        }
      } else {
        *out_timing = IREE_HAL_STREAMING_EVENT_TIMING_INCOMPLETE;
      }
    }
  }

  iree_hal_streaming_event_release_recorded_point(&start_point);
  iree_hal_streaming_event_release_recorded_point(&stop_point);
  return status;
}
