// Copyright 2026 The Pyre Authors
// SPDX-License-Identifier: Apache-2.0
//
// Event implementation. An event marks a point in a stream's execution
// timeline. Cross-stream synchronization is achieved by recording an event
// on one stream and waiting for it on another. Each event owns a dedicated
// semaphore that is signaled when the recorded stream timepoint completes.

#include "pyre_internal.h"

#include <stdlib.h>

//===----------------------------------------------------------------------===//
// Lifecycle
//===----------------------------------------------------------------------===//

pyre_status_t pyre_event_create(pyre_device_t device,
                                pyre_event_flags_t flags,
                                pyre_event_t* out_event) {
  if (!device || !out_event) {
    return pyre_make_status(PYRE_STATUS_INVALID_ARGUMENT,
                            "device or out_event is NULL");
  }

  pyre_event_s* event = (pyre_event_s*)calloc(1, sizeof(pyre_event_s));
  if (!event) {
    return pyre_make_status(PYRE_STATUS_OUT_OF_MEMORY,
                            "failed to allocate event");
  }

  iree_atomic_ref_count_init(&event->ref_count);
  event->flags = flags;
  event->signal_value = 0;
  event->recording_stream = NULL;
  event->device = device;
  event->record_time_ns = 0;

  pyre_status_t status =
      pyre_semaphore_create(device, /*initial_value=*/0, &event->semaphore);
  if (!pyre_status_is_ok(status)) {
    free(event);
    return status;
  }

  *out_event = event;
  return pyre_ok_status();
}

pyre_status_t pyre_event_retain(pyre_event_t event) {
  if (!event) {
    return pyre_make_status(PYRE_STATUS_INVALID_ARGUMENT, "event is NULL");
  }
  iree_atomic_ref_count_inc(&event->ref_count);
  return pyre_ok_status();
}

pyre_status_t pyre_event_release(pyre_event_t event) {
  if (!event) {
    return pyre_make_status(PYRE_STATUS_INVALID_ARGUMENT, "event is NULL");
  }
  if (iree_atomic_ref_count_dec(&event->ref_count) == 1) {
    if (event->semaphore) {
      pyre_status_ignore(pyre_semaphore_release(event->semaphore));
    }
    if (event->recording_stream) {
      pyre_status_ignore(pyre_stream_release(event->recording_stream));
    }
    free(event);
  }
  return pyre_ok_status();
}

//===----------------------------------------------------------------------===//
// Record
//===----------------------------------------------------------------------===//

pyre_status_t pyre_event_record(pyre_event_t event, pyre_stream_t stream) {
  if (!event || !stream) {
    return pyre_make_status(PYRE_STATUS_INVALID_ARGUMENT,
                            "event or stream is NULL");
  }

  event->record_time_ns = (int64_t)iree_time_now();

  // Track recording stream (retain new, release old).
  if (event->recording_stream != stream) {
    if (event->recording_stream) {
      pyre_status_ignore(pyre_stream_release(event->recording_stream));
    }
    event->recording_stream = stream;
    pyre_status_ignore(pyre_stream_retain(stream));
  }

  // Flush pending work so it's submitted before the barrier.
  pyre_status_t status = pyre_stream_flush(stream);
  if (!pyre_status_is_ok(status)) return status;

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
    return pyre_status_from_iree(iree_status);
  }

  return pyre_ok_status();
}

//===----------------------------------------------------------------------===//
// Query & synchronize
//===----------------------------------------------------------------------===//

pyre_status_t pyre_event_query(pyre_event_t event, bool* complete) {
  if (!event || !complete) {
    return pyre_make_status(PYRE_STATUS_INVALID_ARGUMENT,
                            "event or complete is NULL");
  }

  uint64_t current = 0;
  pyre_status_t status = pyre_semaphore_query(event->semaphore, &current);
  if (!pyre_status_is_ok(status)) return status;

  *complete = (current >= event->signal_value);
  return pyre_ok_status();
}

pyre_status_t pyre_event_synchronize(pyre_event_t event) {
  if (!event) {
    return pyre_make_status(PYRE_STATUS_INVALID_ARGUMENT, "event is NULL");
  }
  return pyre_semaphore_wait(event->semaphore, event->signal_value, UINT64_MAX);
}

//===----------------------------------------------------------------------===//
// Elapsed time
//===----------------------------------------------------------------------===//

pyre_status_t pyre_event_elapsed_time(pyre_event_t start, pyre_event_t stop,
                                      float* ms) {
  if (!start || !stop || !ms) {
    return pyre_make_status(PYRE_STATUS_INVALID_ARGUMENT,
                            "start, stop, or ms is NULL");
  }

  if ((start->flags & PYRE_EVENT_FLAG_DISABLE_TIMING) ||
      (stop->flags & PYRE_EVENT_FLAG_DISABLE_TIMING)) {
    return pyre_make_status(PYRE_STATUS_INVALID_ARGUMENT,
                            "cannot measure elapsed time with timing disabled");
  }

  if (start->record_time_ns == 0 || stop->record_time_ns == 0) {
    return pyre_make_status(
        PYRE_STATUS_INVALID_ARGUMENT,
        "events must be recorded before measuring elapsed time");
  }

  // Ensure both events have completed.
  bool start_done = false, stop_done = false;
  pyre_status_t status = pyre_event_query(start, &start_done);
  if (!pyre_status_is_ok(status)) return status;
  if (!start_done) {
    return pyre_make_status(PYRE_STATUS_UNAVAILABLE,
                            "start event has not completed");
  }

  status = pyre_event_query(stop, &stop_done);
  if (!pyre_status_is_ok(status)) return status;
  if (!stop_done) {
    return pyre_make_status(PYRE_STATUS_UNAVAILABLE,
                            "stop event has not completed");
  }

  int64_t elapsed_ns = stop->record_time_ns - start->record_time_ns;
  *ms = (float)elapsed_ns / 1000000.0f;
  return pyre_ok_status();
}

//===----------------------------------------------------------------------===//
// Stream wait event
//===----------------------------------------------------------------------===//

pyre_status_t pyre_stream_wait_event(pyre_stream_t stream,
                                     pyre_event_t event) {
  if (!stream || !event) {
    return pyre_make_status(PYRE_STATUS_INVALID_ARGUMENT,
                            "stream or event is NULL");
  }

  // Flush pending work before inserting the wait barrier.
  pyre_status_t status = pyre_stream_flush(stream);
  if (!pyre_status_is_ok(status)) return status;

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
    return pyre_status_from_iree(iree_status);
  }

  stream->timepoint = signal_value;
  return pyre_ok_status();
}
