// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "binding/hip/event_operations.h"

#include "common/internal.h"

// Maps common event-operation status without changing the generic HIP status
// conversion used by unrelated entry points. Query and synchronize expose an
// aborted event operation as a destroyed context; elapsed-time failures keep
// the generic unknown result for that otherwise unreachable status.
static hipError_t iree_hip_event_status_to_result(iree_status_t status,
                                                  bool map_aborted) {
  if (iree_status_is_ok(status)) return hipSuccess;

  const iree_status_code_t code = iree_status_code(status);
  iree_status_free(status);
  switch (code) {
    case IREE_STATUS_ABORTED:
      return map_aborted ? hipErrorContextIsDestroyed : hipErrorUnknown;
    case IREE_STATUS_INVALID_ARGUMENT:
    case IREE_STATUS_OUT_OF_RANGE:
      return hipErrorInvalidValue;
    case IREE_STATUS_RESOURCE_EXHAUSTED:
      return hipErrorOutOfMemory;
    case IREE_STATUS_NOT_FOUND:
      return hipErrorNotFound;
    case IREE_STATUS_PERMISSION_DENIED:
      return hipErrorInvalidContext;
    case IREE_STATUS_UNIMPLEMENTED:
      return hipErrorNotSupported;
    case IREE_STATUS_UNAVAILABLE:
      return hipErrorNotReady;
    case IREE_STATUS_FAILED_PRECONDITION:
      return hipErrorNotInitialized;
    default:
      return hipErrorUnknown;
  }
}

// Invalidates the active sessions named by up to two captured events. Every
// association is acquired before the common layer prepares one process-wide
// snapshot, so allocation failure changes neither session. Stale associations
// still report as captured to the caller but invalidate no successor session.
static hipError_t iree_hip_invalidate_captured_events(
    iree_hal_streaming_event_t* const* events, iree_host_size_t event_count,
    bool* out_any_captured) {
  IREE_ASSERT(event_count <= 2);
  *out_any_captured = false;
  iree_hal_streaming_event_capture_association_t associations[2] = {0};
  for (iree_host_size_t i = 0; i < event_count; ++i) {
    iree_hal_streaming_event_acquire_capture_association(events[i],
                                                         &associations[i]);
    *out_any_captured |= associations[i].graph != NULL;
  }

  bool invalidated = false;
  iree_status_t status = iree_hal_streaming_invalidate_event_captures(
      associations, event_count, &invalidated);
  for (iree_host_size_t i = event_count; i > 0; --i) {
    iree_hal_streaming_event_release_capture_association(&associations[i - 1]);
  }
  return iree_hip_event_status_to_result(status, /*map_aborted=*/true);
}

hipError_t iree_hip_event_synchronize_retained(
    iree_hal_streaming_event_t* event, bool* out_captured_path) {
  IREE_ASSERT_ARGUMENT(event);
  IREE_ASSERT_ARGUMENT(out_captured_path);
  *out_captured_path = false;

  iree_hal_streaming_event_t* captured_events[1] = {event};
  bool event_is_captured = false;
  hipError_t result = iree_hip_invalidate_captured_events(
      captured_events, IREE_ARRAYSIZE(captured_events), &event_is_captured);
  if (result != hipSuccess || event_is_captured) {
    *out_captured_path = true;
    return result != hipSuccess ? result : hipErrorCapturedEvent;
  }

  return iree_hip_event_status_to_result(
      iree_hal_streaming_event_synchronize(event), /*map_aborted=*/true);
}

hipError_t iree_hip_event_query_retained(iree_hal_streaming_event_t* event,
                                         bool* out_captured_path) {
  IREE_ASSERT_ARGUMENT(event);
  IREE_ASSERT_ARGUMENT(out_captured_path);
  *out_captured_path = false;

  iree_hal_streaming_event_t* captured_events[1] = {event};
  bool event_is_captured = false;
  hipError_t result = iree_hip_invalidate_captured_events(
      captured_events, IREE_ARRAYSIZE(captured_events), &event_is_captured);
  if (result != hipSuccess || event_is_captured) {
    *out_captured_path = true;
    return result != hipSuccess ? result : hipErrorCapturedEvent;
  }

  int is_complete = 0;
  iree_status_t status = iree_hal_streaming_event_query(event, &is_complete);
  return iree_status_is_ok(status)
             ? (is_complete == 0 ? hipSuccess : hipErrorNotReady)
             : iree_hip_event_status_to_result(status, /*map_aborted=*/true);
}

hipError_t iree_hip_event_elapsed_time_retained(
    float* milliseconds, iree_hal_streaming_event_t* start_event,
    iree_hal_streaming_event_t* stop_event) {
  IREE_ASSERT_ARGUMENT(milliseconds);
  IREE_ASSERT_ARGUMENT(start_event);
  IREE_ASSERT_ARGUMENT(stop_event);
  if (start_event->context != stop_event->context) {
    return hipErrorInvalidHandle;
  }

  iree_hal_streaming_event_timing_t timing =
      IREE_HAL_STREAMING_EVENT_TIMING_UNTIMED;
  iree_status_t status = iree_hal_streaming_event_elapsed_time(
      milliseconds, start_event, stop_event, &timing);
  if (!iree_status_is_ok(status)) {
    // The timeline a record names has failed; the event handles are fine.
    return iree_hip_event_status_to_result(status, /*map_aborted=*/false);
  }

  switch (timing) {
    case IREE_HAL_STREAMING_EVENT_TIMING_MEASURED:
      return hipSuccess;
    case IREE_HAL_STREAMING_EVENT_TIMING_INCOMPLETE:
      return hipErrorNotReady;
    case IREE_HAL_STREAMING_EVENT_TIMING_UNTIMED:
      // An event with timing disabled or without a submitted record is a bad
      // handle here, not a bad value.
      return hipErrorInvalidHandle;
    case IREE_HAL_STREAMING_EVENT_TIMING_CAPTURED: {
      // Acquire both associations before preparing one participant snapshot so
      // allocation failure cannot invalidate only one captured session.
      iree_hal_streaming_event_t* captured_events[2] = {start_event,
                                                        stop_event};
      bool any_captured = false;
      hipError_t result = iree_hip_invalidate_captured_events(
          captured_events, IREE_ARRAYSIZE(captured_events), &any_captured);
      return result == hipSuccess ? hipErrorCapturedEvent : result;
    }
    case IREE_HAL_STREAMING_EVENT_TIMING_UNSUPPORTED:
      return hipErrorNotSupported;
  }
  return hipErrorUnknown;
}
