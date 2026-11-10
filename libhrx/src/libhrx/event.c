// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0
//
// Event implementation. An event marks a point in a stream's execution
// timeline. Cross-stream synchronization is achieved by recording an event
// on one stream and waiting for it on another. Each event owns a dedicated
// semaphore that is signaled when the recorded stream timepoint completes.

#include <stdlib.h>

#include "hrx_internal.h"

//===----------------------------------------------------------------------===//
// Lifecycle
//===----------------------------------------------------------------------===//

hrx_status_t hrx_event_create(hrx_device_t device, hrx_event_flags_t flags,
                              hrx_event_t* out_event) {
  if (!device || !out_event) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                           "device or out_event is NULL");
  }

  hrx_event_s* event = (hrx_event_s*)calloc(1, sizeof(hrx_event_s));
  if (!event) {
    return hrx_make_status(HRX_STATUS_OUT_OF_MEMORY,
                           "failed to allocate event");
  }

  iree_atomic_ref_count_init(&event->ref_count);
  event->flags = flags;
  event->signal_value = 0;
  event->device = device;
  event->record_time_ns = 0;

  hrx_status_t status =
      hrx_semaphore_create(device, /*initial_value=*/0, &event->semaphore);
  if (!hrx_status_is_ok(status)) {
    free(event);
    return status;
  }

  *out_event = event;
  return hrx_ok_status();
}

void hrx_event_retain(hrx_event_t event) {
  if (!event) {
    return;
  }
  iree_atomic_ref_count_inc(&event->ref_count);
}

void hrx_event_release(hrx_event_t event) {
  if (!event) {
    return;
  }
  if (iree_atomic_ref_count_dec(&event->ref_count) == 1) {
    hrx_semaphore_release(event->semaphore);
    free(event);
  }
}

//===----------------------------------------------------------------------===//
// Record
//===----------------------------------------------------------------------===//

hrx_status_t hrx_event_record(hrx_event_t event, hrx_stream_t stream) {
  if (!event || !stream) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                           "event or stream is NULL");
  }

  // Flush pending work so it's submitted before the barrier.
  hrx_status_t status = hrx_stream_flush(stream);
  if (!hrx_status_is_ok(status)) {
    return status;
  }

  if (IREE_UNLIKELY(stream->timepoint >= IREE_HAL_SEMAPHORE_MAX_VALUE ||
                    event->signal_value >= IREE_HAL_SEMAPHORE_MAX_VALUE)) {
    return hrx_make_status(HRX_STATUS_OUT_OF_RANGE,
                           "stream or event timeline value exhausted");
  }

  // Each timeline has its own next value. Successive event producers must
  // remain ordered even when they come from independent stream queues.
  iree_hal_semaphore_t* wait_semaphores[2];
  uint64_t wait_values[2];
  iree_host_size_t wait_count = 0;
  if (stream->timepoint) {
    wait_semaphores[wait_count] = stream->semaphore->hal_semaphore;
    wait_values[wait_count++] = stream->timepoint;
  }
  if (event->signal_value) {
    wait_semaphores[wait_count] = event->semaphore->hal_semaphore;
    wait_values[wait_count++] = event->signal_value;
  }
  const iree_hal_semaphore_list_t wait_list = {
      .count = wait_count,
      .semaphores = wait_semaphores,
      .payload_values = wait_values,
  };

  iree_hal_semaphore_t* signal_semaphores[] = {stream->semaphore->hal_semaphore,
                                               event->semaphore->hal_semaphore};
  uint64_t signal_values[] = {stream->timepoint + 1, event->signal_value + 1};
  const iree_hal_semaphore_list_t signal_list = {
      .count = 2,
      .semaphores = signal_semaphores,
      .payload_values = signal_values,
  };

  // Event recording publishes only semaphore timepoints. Explicitly empty
  // boundaries keep this on the queue's empty-barrier path; NULL would request
  // conservative payload visibility and may serialize otherwise independent
  // streams sharing one native queue.
  const iree_hal_barrier_list_t empty_barrier_list = {0};
  const iree_hal_queue_barriers_t empty_barriers = {
      &empty_barrier_list,
      &empty_barrier_list,
  };
  iree_status_t iree_status =
      iree_hal_queue_barrier(stream->hal_queue, wait_list, signal_list,
                             &empty_barriers, IREE_HAL_QUEUE_BARRIER_FLAG_NONE);
  if (!iree_status_is_ok(iree_status)) {
    return hrx_status_from_iree(iree_status);
  }

  stream->timepoint = signal_values[0];
  event->signal_value = signal_values[1];
  event->record_time_ns = (int64_t)iree_time_now();
  return hrx_ok_status();
}

//===----------------------------------------------------------------------===//
// Query & synchronize
//===----------------------------------------------------------------------===//

hrx_status_t hrx_event_query(hrx_event_t event, bool* complete) {
  if (!event || !complete) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                           "event or complete is NULL");
  }

  uint64_t current = 0;
  hrx_status_t status = hrx_semaphore_query(event->semaphore, &current);
  if (!hrx_status_is_ok(status)) {
    return status;
  }

  *complete = (current >= event->signal_value);
  return hrx_ok_status();
}

hrx_status_t hrx_event_synchronize(hrx_event_t event) {
  if (!event) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT, "event is NULL");
  }
  iree_status_t status = hrx_status_to_iree(
      hrx_semaphore_wait(event->semaphore, event->signal_value, UINT64_MAX));
  return hrx_status_from_iree(
      iree_status_join(status, hrx_runtime_profile_flush(event->device)));
}

//===----------------------------------------------------------------------===//
// Elapsed time
//===----------------------------------------------------------------------===//

hrx_status_t hrx_event_elapsed_time(hrx_event_t start, hrx_event_t stop,
                                    float* ms) {
  if (!start || !stop || !ms) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                           "start, stop, or ms is NULL");
  }

  if ((start->flags & HRX_EVENT_FLAG_DISABLE_TIMING) ||
      (stop->flags & HRX_EVENT_FLAG_DISABLE_TIMING)) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                           "cannot measure elapsed time with timing disabled");
  }

  if (start->record_time_ns == 0 || stop->record_time_ns == 0) {
    return hrx_make_status(
        HRX_STATUS_INVALID_ARGUMENT,
        "events must be recorded before measuring elapsed time");
  }

  // Ensure both events have completed.
  bool start_done = false, stop_done = false;
  hrx_status_t status = hrx_event_query(start, &start_done);
  if (!hrx_status_is_ok(status)) {
    return status;
  }
  if (!start_done) {
    return hrx_make_status(HRX_STATUS_UNAVAILABLE,
                           "start event has not completed");
  }

  status = hrx_event_query(stop, &stop_done);
  if (!hrx_status_is_ok(status)) {
    return status;
  }
  if (!stop_done) {
    return hrx_make_status(HRX_STATUS_UNAVAILABLE,
                           "stop event has not completed");
  }

  int64_t elapsed_ns = stop->record_time_ns - start->record_time_ns;
  *ms = (float)elapsed_ns / 1000000.0f;
  return hrx_ok_status();
}

//===----------------------------------------------------------------------===//
// Stream wait event
//===----------------------------------------------------------------------===//

hrx_status_t hrx_stream_wait_event(hrx_stream_t stream, hrx_event_t event) {
  if (!stream || !event) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                           "stream or event is NULL");
  }

  const hrx_timeline_point_t position = {
      .semaphore = event->semaphore,
      .value = event->signal_value,
  };
  return hrx_stream_wait_on(stream, position);
}
