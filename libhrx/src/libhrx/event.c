// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0
//
// Event implementation. An event marks a point in a stream's execution
// timeline. Cross-stream synchronization is achieved by recording an event
// on one stream and waiting for it on another. Each record retains a snapshot
// of the recording stream's accepted timeline point.

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
  hrx_device_retain(device);

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
    if (event->timestamp_slot) {
      hrx_event_timestamp_slot_release(event->timestamp_slot,
                                       event->semaphore->hal_semaphore,
                                       event->signal_value);
    }
    hrx_semaphore_release(event->semaphore);
    hrx_device_release(event->device);
    free(event);
  }
}

//===----------------------------------------------------------------------===//
// Record
//===----------------------------------------------------------------------===//

// Captures a tick after this stream's accepted work. Only an accepted write
// advances the timeline or transfers a slot to the event.
static iree_status_t hrx_event_enqueue_timestamp(
    hrx_event_t event, hrx_stream_t stream,
    hrx_event_timestamp_slot_t** out_slot) {
  *out_slot = NULL;
  const iree_hal_queue_family_t* family =
      iree_hal_queue_family(stream->hal_queue);
  if (!hrx_event_timestamp_domain_query(iree_hal_queue_family_spec(family))
           .frequency_hz) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "recording queue family has no timestamp clock");
  }
  if (IREE_UNLIKELY(stream->timepoint >= IREE_HAL_SEMAPHORE_MAX_VALUE)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "stream timeline value exhausted");
  }

  hrx_device_t device = stream->queue->device;
  if (!event->timestamp_slot || event->semaphore->device != device) {
    // The prior timed record proves this device's source is already installed.
    // First use acquires it outside the timestamp pool lock, without adding
    // source construction to device creation or untimed recording.
    iree_hal_pool_t* source = NULL;
    IREE_RETURN_IF_ERROR(hrx_mem_pool_backing_retain_allocation_source(
        device, HRX_MEMORY_TYPE_HOST_LOCAL, HRX_BUFFER_USAGE_DEFAULT, &source));
    hrx_event_timestamp_pool_set_source(&device->event_timestamps, source);
    iree_hal_pool_release(source);
  }
  hrx_event_timestamp_slot_t* slot = NULL;
  IREE_RETURN_IF_ERROR(
      hrx_event_timestamp_pool_acquire(&device->event_timestamps, &slot));
  iree_hal_semaphore_t* semaphore = stream->semaphore->hal_semaphore;
  uint64_t wait_value = stream->timepoint;
  uint64_t signal_value = wait_value + 1;
  const iree_hal_semaphore_list_t waits = {
      .count = wait_value > 0 ? 1 : 0,
      .semaphores = &semaphore,
      .payload_values = &wait_value,
  };
  const iree_hal_semaphore_list_t signals = {
      .count = 1,
      .semaphores = &semaphore,
      .payload_values = &signal_value,
  };
  iree_status_t status = iree_hal_queue_timestamp(
      stream->hal_queue, waits, signals, hrx_event_timestamp_slot_buffer(slot),
      hrx_event_timestamp_slot_offset(slot), /*barriers=*/NULL,
      IREE_HAL_TIMESTAMP_FLAG_NONE);
  if (iree_status_is_ok(status)) {
    stream->timepoint = signal_value;
    *out_slot = slot;
  } else {
    hrx_event_timestamp_slot_release(slot, NULL, 0);
  }
  return status;
}

hrx_status_t hrx_event_record(hrx_event_t event, hrx_stream_t stream) {
  if (!event || !stream) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                           "event or stream is NULL");
  }

  // Submit the pending recording before naming its accepted completion point.
  hrx_status_t status = hrx_stream_flush(stream);
  if (!hrx_status_is_ok(status)) {
    return status;
  }

  const bool timing_enabled = !(event->flags & HRX_EVENT_FLAG_DISABLE_TIMING);
  if (timing_enabled) {
    hrx_event_timestamp_slot_t* slot = NULL;
    HRX_RETURN_IF_IREE_ERROR(hrx_event_enqueue_timestamp(event, stream, &slot));
    if (event->timestamp_slot) {
      hrx_event_timestamp_slot_release(event->timestamp_slot,
                                       event->semaphore->hal_semaphore,
                                       event->signal_value);
    }
    event->timestamp_slot = slot;
    event->timestamp_family = iree_hal_queue_family(stream->hal_queue);
  }

  // Re-recording observes the new producer without ordering it after the old
  // one. Already-enqueued waits retain their own semaphore/value snapshots.
  if (event->semaphore != stream->semaphore) {
    hrx_semaphore_t previous_semaphore = event->semaphore;
    event->semaphore = stream->semaphore;
    hrx_semaphore_retain(event->semaphore);
    hrx_semaphore_release(previous_semaphore);
  }
  event->signal_value = stream->timepoint;
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

  if (event->signal_value == 0) {
    *complete = true;
    return hrx_ok_status();
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
  iree_status_t status = iree_ok_status();
  if (event->signal_value > 0) {
    status = hrx_status_to_iree(
        hrx_semaphore_wait(event->semaphore, event->signal_value, UINT64_MAX));
  }
  hrx_device_t device =
      event->semaphore ? event->semaphore->device : event->device;
  return hrx_status_from_iree(
      iree_status_join(status, hrx_runtime_profile_flush(device)));
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

  if (!start->timestamp_slot || !stop->timestamp_slot) {
    return hrx_make_status(
        HRX_STATUS_INVALID_ARGUMENT,
        "events must be recorded before measuring elapsed time");
  }
  if (start->timestamp_family != stop->timestamp_family) {
    return hrx_make_status(
        HRX_STATUS_INVALID_ARGUMENT,
        "event timestamps belong to different queue families");
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

  return hrx_status_from_iree(hrx_event_timestamp_elapsed_ms(
      hrx_event_timestamp_domain_query(
          iree_hal_queue_family_spec(start->timestamp_family)),
      start->timestamp_slot, stop->timestamp_slot, ms));
}

//===----------------------------------------------------------------------===//
// Stream wait event
//===----------------------------------------------------------------------===//

hrx_status_t hrx_stream_wait_event(hrx_stream_t stream, hrx_event_t event) {
  if (!stream || !event) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                           "stream or event is NULL");
  }

  if (event->signal_value == 0) {
    return hrx_ok_status();
  }

  const hrx_timeline_point_t position = {
      .semaphore = event->semaphore,
      .value = event->signal_value,
  };
  return hrx_stream_wait_on(stream, position);
}
