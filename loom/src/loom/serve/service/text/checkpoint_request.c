// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/service/text/checkpoint_request.h"

#include <string.h>

static iree_status_t loom_serve_text_checkpoint_reject(
    loom_serve_text_checkpoint_operation_t* control,
    loom_serve_text_telemetry_t* telemetry, iree_status_t failure,
    iree_string_builder_t* scratch) {
  const loom_serve_text_checkpoint_operation_t operation = *control;
  memset(control, 0, sizeof(*control));
  iree_status_t status =
      operation.consumer.end(operation.consumer.self, &operation.request, NULL,
                             failure, telemetry, &operation.receipt, scratch);
  iree_status_free(failure);
  return status;
}

static void loom_serve_text_checkpoint_fail(
    loom_serve_text_checkpoint_operation_t* control,
    loom_serve_text_telemetry_t* telemetry, const iree_status_t status) {
  const loom_serve_text_checkpoint_operation_t operation = *control;
  memset(control, 0, sizeof(*control));
  operation.consumer.abort(operation.consumer.self, LOOM_SERVE_TELEMETRY_FAILED,
                           status, telemetry, &operation.receipt);
}

static iree_status_t loom_serve_text_checkpoint_reply(
    loom_serve_text_checkpoint_operation_t* control,
    loom_serve_text_telemetry_t* telemetry, iree_string_builder_t* scratch,
    iree_host_size_t position, iree_host_size_t host_bytes) {
  static const loom_serve_telemetry_event_t* const events[] = {
      &LOOM_SERVE_TEXT_CHECKPOINT_PIN,
      &LOOM_SERVE_TEXT_CHECKPOINT_SUSPEND,
      &LOOM_SERVE_TEXT_CHECKPOINT_RELEASE,
  };
  loom_serve_text_checkpoint_record_t value = {.position = position,
                                               .host_bytes = host_bytes};
  memcpy(value.name, control->request.name.data, control->request.name.size);
  loom_serve_text_record(telemetry->stream, events[control->request.action],
                         control->receipt.serial, &value, sizeof(value));
  if (!control->consumer.is_open(control->consumer.self)) {
    loom_serve_text_checkpoint_request_abort(control, telemetry);
    return iree_ok_status();
  }
  const loom_serve_text_checkpoint_operation_t operation = *control;
  memset(control, 0, sizeof(*control));
  const loom_serve_text_checkpoint_result_t result = {.position = position,
                                                      .host_bytes = host_bytes};
  return operation.consumer.end(operation.consumer.self, &operation.request,
                                &result, iree_ok_status(), telemetry,
                                &operation.receipt, scratch);
}

iree_status_t loom_serve_text_checkpoint_request_submit(
    loom_serve_text_checkpoint_operation_t* control,
    loom_serve_text_checkpoint_registry_t* registry,
    loom_serve_text_checkpoint_source_t source, bool suspension_supported,
    loom_serve_text_telemetry_t* telemetry, iree_string_builder_t* scratch,
    bool* trim_pending, loom_serve_execution_frontier_t* out_frontier) {
  const bool create =
      control->request.action == LOOM_SERVE_TEXT_CHECKPOINT_ACTION_PIN;
  const bool suspend =
      control->request.action == LOOM_SERVE_TEXT_CHECKPOINT_ACTION_SUSPEND;
  loom_serve_text_checkpoint_entry_t* checkpoint =
      loom_serve_text_checkpoint_registry_find(registry, control->request.name);
  if (create && checkpoint) {
    return loom_serve_text_checkpoint_reject(
        control, telemetry,
        iree_make_status(IREE_STATUS_ALREADY_EXISTS,
                         "checkpoint name is already pinned"),
        scratch);
  }
  if (!create && !checkpoint) {
    return loom_serve_text_checkpoint_reject(
        control, telemetry,
        iree_make_status(IREE_STATUS_NOT_FOUND, "checkpoint is not pinned"),
        scratch);
  }
  if (control->request.action == LOOM_SERVE_TEXT_CHECKPOINT_ACTION_RELEASE &&
      checkpoint->pending) {
    return loom_serve_text_checkpoint_reject(
        control, telemetry,
        iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                         "checkpoint is borrowed by a queued request"),
        scratch);
  }
  if (create) {
    if (!source.session && !source.cached) {
      return loom_serve_text_checkpoint_reject(
          control, telemetry,
          iree_make_status(IREE_STATUS_NOT_FOUND, "session is not retained"),
          scratch);
    }
    if (source.busy ||
        (source.session && !source.session->completion.reusable)) {
      return loom_serve_text_checkpoint_reject(
          control, telemetry,
          iree_make_status(
              IREE_STATUS_FAILED_PRECONDITION,
              "pin requires an idle session with reusable history"),
          scratch);
    }
    checkpoint = loom_serve_text_checkpoint_registry_available(registry);
    if (!checkpoint) {
      return loom_serve_text_checkpoint_reject(
          control, telemetry,
          iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                           "checkpoint capacity exhausted; release a pin or "
                           "configure checkpoint_capacity"),
          scratch);
    }
    if (source.cached) {
      loom_serve_text_checkpoint_entry_retain(source.cached, checkpoint);
      loom_serve_text_checkpoint_entry_publish(checkpoint,
                                               control->request.name);
      return loom_serve_text_checkpoint_reply(
          control, telemetry, scratch, checkpoint->position,
          loom_serve_text_checkpoint_suspended_bytes(checkpoint->state));
    }
    iree_status_t status = loom_serve_text_checkpoint_entry_submit_pin(
        checkpoint, source.session->row, &source.session->completion,
        out_frontier);
    if (!iree_status_is_ok(status)) {
      loom_serve_text_checkpoint_fail(control, telemetry, status);
      return status;
    }
    if (!checkpoint->state) {
      return loom_serve_text_checkpoint_reject(
          control, telemetry,
          iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                           "native pin capacity exhausted"),
          scratch);
    }
    control->checkpoint = checkpoint;
    return iree_ok_status();
  }
  const iree_host_size_t position = checkpoint->position;
  const iree_host_size_t host_bytes =
      loom_serve_text_checkpoint_suspended_bytes(checkpoint->state);
  if (suspend) {
    if (!suspension_supported) {
      return loom_serve_text_checkpoint_reject(
          control, telemetry,
          iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                           "checkpoint suspension requires elastic backing"),
          scratch);
    }
    if (host_bytes) {
      return loom_serve_text_checkpoint_reply(control, telemetry, scratch,
                                              position, host_bytes);
    }
    iree_status_t status = loom_serve_text_checkpoint_submit_suspend(
        checkpoint->state, out_frontier);
    if (!iree_status_is_ok(status)) {
      loom_serve_text_checkpoint_fail(control, telemetry, status);
      return status;
    }
    control->checkpoint = checkpoint;
    return iree_ok_status();
  }
  loom_serve_text_checkpoint_entry_release(checkpoint);
  *trim_pending = true;
  return loom_serve_text_checkpoint_reply(control, telemetry, scratch, position,
                                          0);
}

iree_status_t loom_serve_text_checkpoint_request_complete(
    loom_serve_text_checkpoint_operation_t* control, bool stopping,
    loom_serve_text_telemetry_t* telemetry, iree_string_builder_t* scratch,
    bool* trim_pending) {
  loom_serve_text_checkpoint_entry_t* checkpoint = control->checkpoint;
  if (control->request.action == LOOM_SERVE_TEXT_CHECKPOINT_ACTION_PIN) {
    loom_serve_text_checkpoint_entry_publish(checkpoint, control->request.name);
  } else {
    loom_serve_text_checkpoint_complete_suspend(checkpoint->state);
    *trim_pending = true;
  }
  if (stopping) {
    loom_serve_text_checkpoint_request_abort(control, telemetry);
    return iree_ok_status();
  }
  return loom_serve_text_checkpoint_reply(
      control, telemetry, scratch, checkpoint->position,
      loom_serve_text_checkpoint_suspended_bytes(checkpoint->state));
}

void loom_serve_text_checkpoint_request_abort(
    loom_serve_text_checkpoint_operation_t* control,
    loom_serve_text_telemetry_t* telemetry) {
  if (control->consumer.self) {
    const loom_serve_text_checkpoint_operation_t operation = *control;
    memset(control, 0, sizeof(*control));
    const bool failed = telemetry->failure_code != IREE_STATUS_OK;
    operation.consumer.abort(
        operation.consumer.self,
        failed ? LOOM_SERVE_TELEMETRY_FAILED : LOOM_SERVE_TELEMETRY_CANCELLED,
        iree_status_from_code(failed ? telemetry->failure_code
                                     : IREE_STATUS_CANCELLED),
        telemetry, &operation.receipt);
  }
}
