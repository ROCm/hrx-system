// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/internal.h"

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
  event->record_time_ns = 0;
  event->ipc_handle = NULL;
  event->capture_graph = NULL;
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
  iree_hal_semaphore_release(event->recorded_point.semaphore);

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
  iree_slim_mutex_unlock(&event->mutex);
}

void iree_hal_streaming_event_commit_recorded_point(
    iree_hal_streaming_event_t* event,
    iree_hal_streaming_recorded_point_t point, iree_time_t record_time_ns) {
  iree_hal_semaphore_retain(point.semaphore);
  iree_slim_mutex_lock(&event->mutex);
  iree_hal_semaphore_t* previous_semaphore = event->recorded_point.semaphore;
  event->recorded_point = point;
  event->record_time_ns = record_time_ns;
  iree_slim_mutex_unlock(&event->mutex);
  iree_hal_semaphore_release(previous_semaphore);
}

iree_hal_streaming_stream_t* iree_hal_streaming_event_exchange_recording_stream(
    iree_hal_streaming_event_t* event, iree_hal_streaming_stream_t* stream) {
  iree_slim_mutex_lock(&event->mutex);
  iree_hal_streaming_stream_t* previous_stream = event->recording_stream;
  if (previous_stream != stream) {
    iree_hal_streaming_stream_retain(stream);
    event->recording_stream = stream;
  } else {
    previous_stream = NULL;
  }
  iree_slim_mutex_unlock(&event->mutex);
  return previous_stream;
}

iree_time_t iree_hal_streaming_event_record_time_ns(
    iree_hal_streaming_event_t* event) {
  iree_slim_mutex_lock(&event->mutex);
  const iree_time_t record_time_ns = event->record_time_ns;
  iree_slim_mutex_unlock(&event->mutex);
  return record_time_ns;
}

iree_status_t iree_hal_streaming_event_query(iree_hal_streaming_event_t* event,
                                             int* status) {
  IREE_ASSERT_ARGUMENT(event);
  IREE_ASSERT_ARGUMENT(status);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_streaming_recorded_point_t recorded_point;
  iree_hal_streaming_event_acquire_recorded_point(event, &recorded_point);
  if (!recorded_point.semaphore) {
    // An event with no submitted record reads as complete.
    *status = 0;
    IREE_TRACE_ZONE_END(z0);
    return iree_ok_status();
  }

  uint64_t current_value = 0;
  iree_status_t query_status =
      iree_hal_semaphore_query(recorded_point.semaphore, &current_value);
  iree_hal_semaphore_release(recorded_point.semaphore);
  IREE_RETURN_AND_END_ZONE_IF_ERROR(z0, query_status);

  // 0=complete, 1=not complete.
  *status = (current_value >= recorded_point.value) ? 0 : 1;

  IREE_TRACE_ZONE_END(z0);
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
    if (event->capture_graph != stream->capture_graph) {
      iree_hal_streaming_graph_release(event->capture_graph);
      event->capture_graph = stream->capture_graph;
      iree_hal_streaming_graph_retain(event->capture_graph);
    }
    // A captured record produces no submission, so it leaves the recorded point
    // and the record time alone; only the stream is adopted, for the later wait
    // that picks the capture up from the stream that captured it.
    iree_hal_streaming_stream_release(
        iree_hal_streaming_event_exchange_recording_stream(event, stream));
    IREE_TRACE_ZONE_END(z0);
    return iree_ok_status();
  }

  // Sampled before the submission so the timestamp reflects when the caller
  // issued the record.
  const iree_time_t record_time_ns = iree_time_now();

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

  iree_hal_streaming_stream_t* previous_stream = NULL;
  status = iree_hal_device_queue_barrier(
      stream->context->device, stream->queue_affinity, wait_semaphores,
      signal_semaphores, IREE_HAL_EXECUTE_FLAG_NONE);
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_queue_flush(stream->context->device,
                                         stream->queue_affinity);
  }
  if (iree_status_is_ok(status)) {
    stream->pending_value = stream_signal_value;
    stream->submitted_value = stream_signal_value;
    // The barrier signals the stream's own timeline, so reaching the point is
    // exactly the stream reaching that value.
    const iree_hal_streaming_recorded_point_t recorded_point = {
        .semaphore = stream->timeline_semaphore,
        .value = stream_signal_value,
        .ordered_after_stream_id = stream->stream_id,
        .ordered_after_stream_value = stream_signal_value,
    };
    // A rejected submission signals nothing, so the event keeps its old point.
    iree_hal_streaming_event_commit_recorded_point(event, recorded_point,
                                                   record_time_ns);
    previous_stream =
        iree_hal_streaming_event_exchange_recording_stream(event, stream);
  }
  iree_slim_mutex_unlock(&stream->mutex);
  // Stream teardown re-enters the streaming layer, so the displaced reference
  // is dropped outside this stream's mutex.
  iree_hal_streaming_stream_release(previous_stream);
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
  if (!recorded_point.semaphore) {
    // An event with no submitted record has nothing to wait for.
    IREE_TRACE_ZONE_END(z0);
    return iree_ok_status();
  }

  iree_status_t status = iree_hal_semaphore_wait(
      recorded_point.semaphore, recorded_point.value, iree_infinite_timeout(),
      IREE_ASYNC_WAIT_FLAG_NONE);
  iree_hal_semaphore_release(recorded_point.semaphore);
  IREE_RETURN_AND_END_ZONE_IF_ERROR(z0, status);

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

iree_status_t iree_hal_streaming_event_elapsed_time(
    float* ms, iree_hal_streaming_event_t* start,
    iree_hal_streaming_event_t* stop) {
  IREE_ASSERT_ARGUMENT(ms);
  IREE_ASSERT_ARGUMENT(start);
  IREE_ASSERT_ARGUMENT(stop);

  // Check if either event has timing disabled.
  if ((start->flags & IREE_HAL_STREAMING_EVENT_FLAG_DISABLE_TIMING) ||
      (stop->flags & IREE_HAL_STREAMING_EVENT_FLAG_DISABLE_TIMING)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "cannot measure elapsed time with timing disabled");
  }

  // Ensure both events have been recorded.
  const iree_time_t start_time_ns =
      iree_hal_streaming_event_record_time_ns(start);
  const iree_time_t stop_time_ns =
      iree_hal_streaming_event_record_time_ns(stop);
  if (start_time_ns == 0 || stop_time_ns == 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "events must be recorded before measuring elapsed time");
  }

  // Ensure both events have completed.
  int start_status = 0;
  IREE_RETURN_IF_ERROR(iree_hal_streaming_event_query(start, &start_status));
  if (start_status != 0) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "start event has not completed");
  }

  int stop_status = 0;
  IREE_RETURN_IF_ERROR(iree_hal_streaming_event_query(stop, &stop_status));
  if (stop_status != 0) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "stop event has not completed");
  }

  // Calculate elapsed time in milliseconds.
  int64_t elapsed_ns = stop_time_ns - start_time_ns;
  *ms = (float)elapsed_ns / 1000000.0f;  // Convert nanoseconds to milliseconds.

  return iree_ok_status();
}
