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
  event->recording_stream = NULL;
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
    hrx_stream_release(event->recording_stream);
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

  event->record_time_ns = (int64_t)iree_time_now();

  // Track recording stream (retain new, release old).
  if (event->recording_stream != stream) {
    hrx_stream_release(event->recording_stream);
    event->recording_stream = stream;
    hrx_stream_retain(stream);
  }

  // Flush pending work so it's submitted before the barrier.
  hrx_status_t status = hrx_stream_flush(stream);
  if (!hrx_status_is_ok(status)) return status;

  // Wait on stream's current timepoint, signal the next value on both the
  // stream's timeline and the event's dedicated semaphore.
  uint64_t wait_value = stream->timepoint;
  uint64_t signal_value = wait_value + 1;
  event->signal_value = signal_value;
  stream->timepoint = signal_value;

  iree_hal_semaphore_t* wait_sem = stream->semaphore->hal_semaphore;
  iree_hal_semaphore_list_t wait_list = {
      .count = (wait_value > 0) ? 1 : 0,
      .semaphores = &wait_sem,
      .payload_values = &wait_value,
  };

  iree_hal_semaphore_t* sig_sems[] = {stream->semaphore->hal_semaphore,
                                      event->semaphore->hal_semaphore};
  uint64_t sig_vals[] = {signal_value, signal_value};
  iree_hal_semaphore_list_t signal_list = {
      .count = 2,
      .semaphores = sig_sems,
      .payload_values = sig_vals,
  };

  iree_status_t iree_status = iree_hal_device_queue_barrier(
      stream->device->hal_device, IREE_HAL_QUEUE_AFFINITY_ANY, wait_list,
      signal_list, IREE_HAL_EXECUTE_FLAG_NONE);
  if (!iree_status_is_ok(iree_status)) {
    return hrx_status_from_iree(iree_status);
  }

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
  if (!hrx_status_is_ok(status)) return status;

  *complete = (current >= event->signal_value);
  return hrx_ok_status();
}

hrx_status_t hrx_event_synchronize(hrx_event_t event) {
  if (!event) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT, "event is NULL");
  }
  return hrx_semaphore_wait(event->semaphore, event->signal_value, UINT64_MAX);
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
  if (!hrx_status_is_ok(status)) return status;
  if (!start_done) {
    return hrx_make_status(HRX_STATUS_UNAVAILABLE,
                           "start event has not completed");
  }

  status = hrx_event_query(stop, &stop_done);
  if (!hrx_status_is_ok(status)) return status;
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

  // Flush pending work before inserting the wait barrier.
  hrx_status_t status = hrx_stream_flush(stream);
  if (!hrx_status_is_ok(status)) return status;

  // Insert a queue barrier that waits on the event's semaphore and signals
  // the stream's next timepoint.
  uint64_t signal_value = stream->timepoint + 1;

  iree_hal_semaphore_t* wait_sem = event->semaphore->hal_semaphore;
  uint64_t wait_val = event->signal_value;
  iree_hal_semaphore_list_t wait_list = {
      .count = 1,
      .semaphores = &wait_sem,
      .payload_values = &wait_val,
  };

  iree_hal_semaphore_t* sig_sem = stream->semaphore->hal_semaphore;
  iree_hal_semaphore_list_t signal_list = {
      .count = 1,
      .semaphores = &sig_sem,
      .payload_values = &signal_value,
  };

  iree_status_t iree_status = iree_hal_device_queue_barrier(
      stream->device->hal_device, IREE_HAL_QUEUE_AFFINITY_ANY, wait_list,
      signal_list, IREE_HAL_EXECUTE_FLAG_NONE);
  if (!iree_status_is_ok(iree_status)) {
    return hrx_status_from_iree(iree_status);
  }

  stream->timepoint = signal_value;
  return hrx_ok_status();
}
