// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/ipc_event.h"

#include "common/internal.h"

iree_status_t iree_hal_streaming_event_attach_ipc_event(
    iree_hal_streaming_event_t* event,
    iree_hal_streaming_ipc_event_t* ipc_event) {
  IREE_ASSERT_ARGUMENT(event);
  IREE_ASSERT_ARGUMENT(ipc_event);
  if (IREE_UNLIKELY(event->ipc_event)) {
    return iree_make_status(IREE_STATUS_ALREADY_EXISTS,
                            "event already has an IPC implementation");
  }
  if (IREE_UNLIKELY(
          !ipc_event->ops || !ipc_event->ops->destroy ||
          !ipc_event->ops->begin_record || !ipc_event->ops->commit_record ||
          !ipc_event->ops->abort_record || !ipc_event->ops->query ||
          !ipc_event->ops->synchronize || !ipc_event->ops->reserve_wait ||
          !ipc_event->ops->arm_wait || !ipc_event->ops->commit_wait ||
          !ipc_event->ops->abort_wait || !ipc_event->ops->export_token)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "IPC event implementation is incomplete");
  }
  event->ipc_event = ipc_event;
  return iree_ok_status();
}

iree_status_t iree_hal_streaming_event_export_ipc_token(
    iree_hal_streaming_event_t* event,
    uint8_t out_token[IREE_HAL_STREAMING_IPC_EVENT_TOKEN_SIZE]) {
  IREE_ASSERT_ARGUMENT(event);
  IREE_ASSERT_ARGUMENT(out_token);
  if (IREE_UNLIKELY(!event->ipc_event)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "event is not interprocess capable");
  }
  return event->ipc_event->ops->export_token(event->ipc_event, out_token);
}

iree_status_t iree_hal_streaming_event_reserve_wait_point(
    iree_hal_streaming_event_t* event, iree_hal_device_t* destination_device,
    iree_hal_streaming_recorded_point_t* out_point,
    iree_hal_streaming_ipc_event_wait_state_t* out_wait_state) {
  IREE_ASSERT_ARGUMENT(event);
  IREE_ASSERT_ARGUMENT(destination_device);
  IREE_ASSERT_ARGUMENT(out_point);
  IREE_ASSERT_ARGUMENT(out_wait_state);
  *out_point = (iree_hal_streaming_recorded_point_t){0};
  *out_wait_state = NULL;
  if (IREE_LIKELY(!event->ipc_event)) {
    iree_hal_streaming_event_acquire_recorded_point(event, out_point);
    return iree_ok_status();
  }

  iree_status_t status = event->ipc_event->ops->reserve_wait(
      event->ipc_event, destination_device, &out_point->semaphore,
      &out_point->value, out_wait_state);
  if (!iree_status_is_ok(status)) {
    iree_hal_semaphore_release(out_point->semaphore);
    *out_point = (iree_hal_streaming_recorded_point_t){0};
    *out_wait_state = NULL;
    return status;
  }
  return iree_ok_status();
}

bool iree_hal_streaming_event_arm_wait(
    iree_hal_streaming_event_t* event,
    iree_hal_streaming_ipc_event_wait_state_t wait_state) {
  IREE_ASSERT_ARGUMENT(event);
  if (!wait_state) return false;
  return event->ipc_event->ops->arm_wait(event->ipc_event, wait_state);
}

iree_status_t iree_hal_streaming_event_commit_wait(
    iree_hal_streaming_event_t* event,
    iree_hal_streaming_ipc_event_wait_state_t wait_state) {
  IREE_ASSERT_ARGUMENT(event);
  if (!wait_state) return iree_ok_status();
  return event->ipc_event->ops->commit_wait(event->ipc_event, wait_state);
}

void iree_hal_streaming_event_abort_wait(
    iree_hal_streaming_event_t* event,
    iree_hal_streaming_ipc_event_wait_state_t wait_state) {
  IREE_ASSERT_ARGUMENT(event);
  if (!wait_state) return;
  event->ipc_event->ops->abort_wait(event->ipc_event, wait_state);
}
