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

  // Initialize event. An event holds no timeline of its own; it starts with no
  // recorded point and takes a reference to one when a record is submitted.
  iree_atomic_ref_count_init(&event->ref_count);
  event->flags = flags;
  iree_slim_mutex_initialize(&event->mutex);
  event->signal_semaphore = NULL;
  event->signal_value = 0;
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

  // Release the recorded point. The semaphore belongs to the stream or graph
  // executable that carried the record, both of which may already be gone.
  iree_hal_semaphore_release(event->signal_semaphore);

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
    iree_hal_streaming_event_t* event, iree_hal_semaphore_t** out_semaphore,
    uint64_t* out_value) {
  iree_slim_mutex_lock(&event->mutex);
  *out_semaphore = event->signal_semaphore;
  *out_value = event->signal_value;
  iree_hal_semaphore_retain(*out_semaphore);
  iree_slim_mutex_unlock(&event->mutex);
}

void iree_hal_streaming_event_commit_recorded_point(
    iree_hal_streaming_event_t* event, iree_hal_semaphore_t* semaphore,
    uint64_t value, iree_time_t record_time_ns) {
  iree_hal_semaphore_retain(semaphore);
  iree_slim_mutex_lock(&event->mutex);
  iree_hal_semaphore_t* previous_semaphore = event->signal_semaphore;
  event->signal_semaphore = semaphore;
  event->signal_value = value;
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

  iree_hal_semaphore_t* semaphore = NULL;
  uint64_t signal_value = 0;
  iree_hal_streaming_event_acquire_recorded_point(event, &semaphore,
                                                  &signal_value);
  if (!semaphore) {
    // Nothing has been submitted for this event, so there is nothing to wait
    // for and the event reads as complete.
    *status = 0;
    IREE_TRACE_ZONE_END(z0);
    return iree_ok_status();
  }

  uint64_t current_value = 0;
  iree_status_t query_status =
      iree_hal_semaphore_query(semaphore, &current_value);
  iree_hal_semaphore_release(semaphore);
  IREE_RETURN_AND_END_ZONE_IF_ERROR(z0, query_status);

  *status =
      (current_value >= signal_value) ? 0 : 1;  // 0=complete, 1=not complete

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
    // A captured record produces neither a submission nor a graph node, only
    // the frontier snapshot above, so there is no point for it to name and no
    // moment for a timestamp to describe: both the recorded point and the
    // record time are left naming the last submitted record. An event whose
    // only record was captured therefore reads as never recorded, and elapsed
    // time over it is rejected rather than reporting the host-side gap between
    // two capture calls. The stream is adopted because a later wait on this
    // event has to pick the capture up from the stream that captured it.
    iree_hal_streaming_stream_release(
        iree_hal_streaming_event_exchange_recording_stream(event, stream));
    IREE_TRACE_ZONE_END(z0);
    return iree_ok_status();
  }

  // Sampled before the submission so the timestamp reflects when the caller
  // issued the record, and adopted only once that submission is accepted.
  const iree_time_t record_time_ns = iree_time_now();

  // Flush the stream to ensure all prior operations are submitted.
  IREE_RETURN_AND_END_ZONE_IF_ERROR(z0,
                                    iree_hal_streaming_stream_flush(stream));

  iree_slim_mutex_lock(&stream->mutex);

  // The record marks a point on the recording stream's timeline: the barrier
  // reserves the stream's next value under the stream mutex and signals it once
  // everything already on the stream has completed. Every value on a stream
  // timeline is reserved exactly once and only ever increases, so a record can
  // never select a value that timeline has already passed no matter which
  // stream recorded the event before, and two streams recording the same event
  // at once draw from different timelines under different mutexes.
  uint64_t stream_wait_value = stream->pending_value;
  if (IREE_UNLIKELY(stream_wait_value == UINT64_MAX)) {
    iree_slim_mutex_unlock(&stream->mutex);
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "stream timeline value overflow");
  }
  uint64_t stream_signal_value = stream_wait_value + 1;

  // The wait is dropped when the stream has never submitted, matching the other
  // submission paths on the stream; nothing can be behind value zero.
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
  iree_status_t status = iree_hal_device_queue_barrier(
      stream->context->device, stream->queue_affinity, wait_semaphores,
      signal_semaphores, IREE_HAL_EXECUTE_FLAG_NONE);
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_queue_flush(stream->context->device,
                                         stream->queue_affinity);
  }
  if (iree_status_is_ok(status)) {
    stream->pending_value = stream_signal_value;
    stream->submitted_value = stream_signal_value;
    // A rejected submission signals nothing, so the event keeps its previous
    // point instead of naming one that will never be reached.
    iree_hal_streaming_event_commit_recorded_point(
        event, stream->timeline_semaphore, stream_signal_value, record_time_ns);
    previous_stream =
        iree_hal_streaming_event_exchange_recording_stream(event, stream);
  }
  iree_slim_mutex_unlock(&stream->mutex);
  // Dropping the last reference to the previously recording stream runs its
  // teardown, which re-enters the streaming layer to synchronize the stream and
  // unregister it from its context, so the reference is dropped only after this
  // stream's mutex is released.
  iree_hal_streaming_stream_release(previous_stream);
  IREE_RETURN_AND_END_ZONE_IF_ERROR(z0, status);

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

iree_status_t iree_hal_streaming_event_synchronize(
    iree_hal_streaming_event_t* event) {
  IREE_ASSERT_ARGUMENT(event);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_semaphore_t* semaphore = NULL;
  uint64_t signal_value = 0;
  iree_hal_streaming_event_acquire_recorded_point(event, &semaphore,
                                                  &signal_value);
  if (!semaphore) {
    // Nothing has been submitted for this event, so there is nothing to wait
    // for.
    IREE_TRACE_ZONE_END(z0);
    return iree_ok_status();
  }

  // Waiting outside the event mutex leaves the event recordable while a wait on
  // it is outstanding; the retained reference keeps the timeline alive across a
  // concurrent re-record onto another one.
  iree_status_t status =
      iree_hal_semaphore_wait(semaphore, signal_value, iree_infinite_timeout(),
                              IREE_ASYNC_WAIT_FLAG_NONE);
  iree_hal_semaphore_release(semaphore);
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

  // Ensure both events have been recorded. A record that failed to submit
  // leaves no timestamp, so this also rejects events whose only record never
  // reached the device.
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
