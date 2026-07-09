// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/internal.h"

//===----------------------------------------------------------------------===//
// Event management
//===----------------------------------------------------------------------===//

static void iree_hal_streaming_event_destroy(iree_hal_streaming_event_t* event);
static iree_status_t iree_hal_streaming_event_enable_device_timing(
    iree_hal_streaming_event_t* event, iree_hal_streaming_context_t* context);

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
  event->signal_value = 0;
  event->recording_stream = NULL;
  event->context = context;
  iree_hal_streaming_context_retain(context);
  event->record_time_ns = 0;
  event->timestamp_buffer = NULL;
  event->timestamp_frequency_hz = 0;
  event->device_tick_valid = false;
  event->ipc_handle = NULL;
  event->capture_graph = NULL;
  event->capture_dependencies = NULL;
  event->capture_dependency_count = 0;
  event->capture_dependency_capacity = 0;
  event->semaphore = NULL;
  event->host_allocator = host_allocator;

  // Create HAL semaphore for synchronization.
  iree_status_t status = iree_hal_semaphore_create(
      context->device, IREE_HAL_QUEUE_AFFINITY_ANY, 0ULL,
      IREE_HAL_SEMAPHORE_FLAG_NONE, &event->semaphore);

  // Enable device-side timing when the device advertises a device timestamp
  // domain. Devices without one stay host-timed; a capable device that cannot
  // allocate its tick buffer fails creation rather than silently using
  // (wrong) host timing.
  if (iree_status_is_ok(status) &&
      !iree_all_bits_set(flags, IREE_HAL_STREAMING_EVENT_FLAG_DISABLE_TIMING)) {
    status = iree_hal_streaming_event_enable_device_timing(event, context);
  }

  if (iree_status_is_ok(status)) {
    *out_event = event;
  } else {
    iree_hal_streaming_event_destroy(event);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_streaming_event_destroy(
    iree_hal_streaming_event_t* event) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Release semaphore.
  iree_hal_semaphore_release(event->semaphore);

  // Release device timestamp buffer.
  iree_hal_buffer_release(event->timestamp_buffer);

  // Release recording stream reference.
  iree_hal_streaming_stream_release(event->recording_stream);

  // Release context.
  iree_hal_streaming_context_release(event->context);

  iree_hal_streaming_graph_release(event->capture_graph);
  iree_allocator_free(event->host_allocator, event->capture_dependencies);

  // Free event memory.
  iree_allocator_t host_allocator = event->host_allocator;
  iree_allocator_free(host_allocator, event);

  IREE_TRACE_ZONE_END(z0);
}

// Enables device-side timestamping for |event| when the device advertises a
// device timestamp domain, allocating the per-event tick buffer. Returns OK and
// leaves the event host-timed (timestamp_buffer == NULL) when the device does
// not advertise the capability. Returns a failure status when the device DOES
// advertise it but the tick buffer cannot be allocated: dropping to host timing
// there would report wrong (host-enqueue) durations on a device that can
// measure real ones.
static iree_status_t iree_hal_streaming_event_enable_device_timing(
    iree_hal_streaming_event_t* event, iree_hal_streaming_context_t* context) {
  const iree_hal_device_spec_t* spec = iree_hal_device_spec(context->device);
  if (!spec) return iree_ok_status();
  // iree_hal_device_spec_timing returns &spec->timing, so it is non-NULL once
  // spec is.
  const iree_hal_device_timing_spec_t* timing =
      iree_hal_device_spec_timing(spec);
  if (!iree_all_bits_set(timing->flags,
                         IREE_HAL_DEVICE_TIMING_SPEC_FLAG_DEVICE_TIMESTAMPS) ||
      timing->timestamp_frequency_hz == 0) {
    // Device does not advertise device-side timestamps; the event stays
    // host-timed.
    return iree_ok_status();
  }
  iree_hal_buffer_params_t params = {0};
  // Host-visible host-pool buffer; adding DEVICE_LOCAL would require a
  // fine-grained device pool that only large-BAR / APU GPUs expose. The host
  // pool is device-visible, so the device still writes the tick here and the
  // host maps it to read it back.
  params.type = IREE_HAL_MEMORY_TYPE_HOST_VISIBLE;
  params.usage = IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_MAPPING;
  params.access = IREE_HAL_MEMORY_ACCESS_ALL;
  params.queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY;
  iree_hal_buffer_t* buffer = NULL;
  // The device advertised the capability, so a failed allocation is a real
  // error rather than a reason to fall back to (wrong) host timing.
  IREE_RETURN_IF_ERROR(iree_hal_allocator_allocate_buffer(
      context->device_allocator, params, sizeof(uint64_t), &buffer));
  event->timestamp_buffer = buffer;
  event->timestamp_frequency_hz = timing->timestamp_frequency_hz;
  return iree_ok_status();
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

iree_status_t iree_hal_streaming_event_query(iree_hal_streaming_event_t* event,
                                             int* status) {
  IREE_ASSERT_ARGUMENT(event);
  IREE_ASSERT_ARGUMENT(status);
  IREE_TRACE_ZONE_BEGIN(z0);

  uint64_t current_value = 0;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_semaphore_query(event->semaphore, &current_value));

  *status = (current_value >= event->signal_value)
                ? 0
                : 1;  // 0=complete, 1=not complete

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

iree_status_t iree_hal_streaming_event_record(
    iree_hal_streaming_event_t* event, iree_hal_streaming_stream_t* stream) {
  IREE_ASSERT_ARGUMENT(event);
  IREE_ASSERT_ARGUMENT(stream);
  IREE_TRACE_ZONE_BEGIN(z0);

  // Reset per-record: only a successful device-tick write below sets this true,
  // so capture- and graph-recorded events remain host-timed.
  event->device_tick_valid = false;

  // Check if we're capturing to a graph.
  if (stream->capture_status == IREE_HAL_STREAMING_CAPTURE_STATUS_ACTIVE) {
    event->record_time_ns = iree_time_now();
    if (event->recording_stream != stream) {
      iree_hal_streaming_stream_release(event->recording_stream);
      event->recording_stream = stream;
      iree_hal_streaming_stream_retain(stream);
    }
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
    IREE_TRACE_ZONE_END(z0);
    return iree_ok_status();
  }

  event->record_time_ns = iree_time_now();

  // Set recording stream so we can track when we cross streams in a signal/wait
  // sequence.
  if (event->recording_stream != stream) {
    iree_hal_streaming_stream_release(event->recording_stream);
    event->recording_stream = stream;
    iree_hal_streaming_stream_retain(stream);
  }

  // Flush the stream to ensure all prior operations are submitted.
  IREE_RETURN_AND_END_ZONE_IF_ERROR(z0,
                                    iree_hal_streaming_stream_flush(stream));

  iree_slim_mutex_lock(&stream->mutex);

  // Use stream's current pending value as wait value and increment for signal.
  uint64_t wait_value = stream->pending_value;
  event->signal_value = wait_value + 1;

  // Create a queue barrier to signal the event semaphore.
  // This waits for the stream's last submission to complete before signaling.
  iree_hal_semaphore_list_t wait_semaphores = {
      .count = 1,
      .semaphores = &stream->timeline_semaphore,
      .payload_values = &wait_value,
  };
  iree_hal_semaphore_t* semaphores[] = {stream->timeline_semaphore,
                                        event->semaphore};
  uint64_t signal_values[] = {event->signal_value, event->signal_value};
  iree_hal_semaphore_list_t signal_semaphores = {
      .count = 2,
      .semaphores = semaphores,
      .payload_values = signal_values,
  };

  // Device-timed events write a device GPU-clock tick at this queue point; the
  // timestamp op signals the same semaphores a barrier would. Host-timed events
  // use a plain barrier (record_time_ns above is the timestamp).
  iree_status_t status;
  if (event->timestamp_buffer) {
    status = iree_hal_device_queue_timestamp(
        stream->context->device, stream->queue_affinity, wait_semaphores,
        signal_semaphores, event->timestamp_buffer,
        /*target_offset=*/0, IREE_HAL_TIMESTAMP_FLAG_NONE);
  } else {
    status = iree_hal_device_queue_barrier(
        stream->context->device, stream->queue_affinity, wait_semaphores,
        signal_semaphores, IREE_HAL_EXECUTE_FLAG_NONE);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_queue_flush(stream->context->device,
                                         stream->queue_affinity);
  }
  if (iree_status_is_ok(status)) {
    stream->pending_value = event->signal_value;
    stream->submitted_value = event->signal_value;
    // A device tick was submitted for this record iff the timestamp path was
    // taken; mark it valid only after the whole record submission succeeded.
    event->device_tick_valid = event->timestamp_buffer != NULL;
  }
  iree_slim_mutex_unlock(&stream->mutex);
  IREE_RETURN_AND_END_ZONE_IF_ERROR(z0, status);

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

iree_status_t iree_hal_streaming_event_synchronize(
    iree_hal_streaming_event_t* event) {
  IREE_ASSERT_ARGUMENT(event);
  IREE_TRACE_ZONE_BEGIN(z0);

  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_semaphore_wait(event->semaphore, event->signal_value,
                                  iree_infinite_timeout(),
                                  IREE_ASYNC_WAIT_FLAG_NONE));

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

float iree_hal_streaming_event_ticks_to_ms(uint64_t start_tick,
                                           uint64_t stop_tick,
                                           uint64_t frequency_hz) {
  // Signed delta so a stop-before-start yields a negative duration rather than
  // an unsigned wrap.
  const int64_t delta_ticks = (int64_t)stop_tick - (int64_t)start_tick;
  return (float)((double)delta_ticks * 1000.0 / (double)frequency_hz);
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
  if (start->record_time_ns == 0 || stop->record_time_ns == 0) {
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

  // Device-side timestamps: read both GPU-clock ticks and convert using the
  // device tick frequency. The ticks must come from one device clock, so both
  // events must be on the same device (as hipEventElapsedTime requires); ticks
  // are then valid across streams on that device. Same device means stop shares
  // start's nonzero frequency, so the start->timestamp_frequency_hz guard and
  // conversion below cover both events. Events on different devices fall
  // through to the host-observed path below.
  if (start->device_tick_valid && stop->device_tick_valid &&
      start->timestamp_frequency_hz != 0 &&
      start->context->device == stop->context->device) {
    uint64_t start_tick = 0;
    uint64_t stop_tick = 0;
    IREE_RETURN_IF_ERROR(iree_hal_buffer_map_read(
        start->timestamp_buffer, 0, &start_tick, sizeof(start_tick)));
    IREE_RETURN_IF_ERROR(iree_hal_buffer_map_read(
        stop->timestamp_buffer, 0, &stop_tick, sizeof(stop_tick)));
    // Full 64-bit counter: the amdgpu device timestamp domain reports a
    // full-width clock, so the signed subtraction cannot wrap. A device that
    // advertised a narrower counter (timestamp_valid_bits < 64, from
    // iree_hal_device_spec_timing) is not currently supported here.
    *ms = iree_hal_streaming_event_ticks_to_ms(start_tick, stop_tick,
                                               start->timestamp_frequency_hz);
    return iree_ok_status();
  }

  // Host-observed fallback (devices without a device timestamp domain).
  int64_t elapsed_ns = stop->record_time_ns - start->record_time_ns;
  *ms = (float)elapsed_ns / 1000000.0f;  // Convert nanoseconds to milliseconds.

  return iree_ok_status();
}
