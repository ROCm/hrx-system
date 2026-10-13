// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/service/text/checkpoint_request.h"

#include <string.h>

#include "loom/serve/service/text/response.h"

iree_status_t loom_serve_text_checkpoint_request_parse(
    loom_serve_text_checkpoint_request_t* control,
    iree_string_view_t* out_session_name) {
  *out_session_name = iree_string_view_empty();
  const loom_serve_http_request_t* request = control->request;
  const bool suspend =
      iree_string_view_consume_suffix(&control->name, IREE_SV("/suspend"));
  const bool post = iree_string_view_equal(request->method, IREE_SV("POST"));
  const bool create = post && !suspend;
  const bool release =
      !suspend && iree_string_view_equal(request->method, IREE_SV("DELETE"));
  if ((!create && !release && !(suspend && post)) || !control->name.size ||
      !loom_serve_text_session_name_valid(control->name) ||
      request->body.size) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "use POST/DELETE /v1/checkpoints/NAME or POST "
                            "/v1/checkpoints/NAME/suspend with an empty body; "
                            "NAME is 1-64 ASCII identifier characters");
  }
  control->action = create    ? LOOM_SERVE_TEXT_CHECKPOINT_ACTION_PIN
                    : suspend ? LOOM_SERVE_TEXT_CHECKPOINT_ACTION_SUSPEND
                              : LOOM_SERVE_TEXT_CHECKPOINT_ACTION_RELEASE;
  if (create) {
    loom_serve_http_request_lookup_header(request, IREE_SV("X-Loom-Session"),
                                          out_session_name);
    if (!out_session_name->size ||
        !loom_serve_text_session_name_valid(*out_session_name)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "pin requires a nonempty X-Loom-Session identifier");
    }
  }
  return iree_ok_status();
}

static iree_status_t checkpoint_reject(
    loom_serve_text_checkpoint_request_t* control,
    loom_serve_text_telemetry_t* telemetry, int code, const char* reason,
    iree_status_t failure, iree_allocator_t host_allocator) {
  const loom_serve_text_checkpoint_request_t request = *control;
  memset(control, 0, sizeof(*control));
  return loom_serve_text_response_reject(telemetry, request.connection,
                                         request.receipt, code, reason, failure,
                                         host_allocator);
}

static iree_status_t checkpoint_reply(
    loom_serve_text_checkpoint_request_t* control,
    loom_serve_text_telemetry_t* telemetry, iree_string_builder_t* scratch,
    iree_host_size_t position, iree_host_size_t host_bytes) {
  const loom_serve_text_checkpoint_request_t request = *control;
  memset(control, 0, sizeof(*control));
  const bool create = request.action == LOOM_SERVE_TEXT_CHECKPOINT_ACTION_PIN;
  const bool suspend =
      request.action == LOOM_SERVE_TEXT_CHECKPOINT_ACTION_SUSPEND;
  const bool release =
      request.action == LOOM_SERVE_TEXT_CHECKPOINT_ACTION_RELEASE;
  loom_serve_text_checkpoint_record_t value = {.position = position,
                                               .host_bytes = host_bytes};
  memcpy(value.name, request.name.data, request.name.size);
  loom_serve_text_record(telemetry->stream,
                         create    ? &LOOM_SERVE_TEXT_CHECKPOINT_PIN
                         : suspend ? &LOOM_SERVE_TEXT_CHECKPOINT_SUSPEND
                                   : &LOOM_SERVE_TEXT_CHECKPOINT_RELEASE,
                         request.receipt.serial, &value, sizeof(value));
  iree_string_builder_reset(scratch);
  iree_status_t status = iree_string_builder_append_format(
      scratch,
      "HTTP/1.1 %s\r\nContent-Type: application/json\r\nConnection: "
      "close\r\n\r\n"
      "{\"checkpoint\":\"%.*s\",\"position\":%zu,\"pinned\":%s,"
      "\"host_snapshot_bytes\":%zu}",
      create ? "201 Created" : "200 OK", (int)request.name.size,
      request.name.data, position, release ? "false" : "true", host_bytes);
  if (iree_status_is_ok(status)) {
    iree_status_t sent = loom_serve_http_connection_send(
        request.connection, iree_string_builder_view(scratch));
    if (iree_status_is_ok(sent)) {
      loom_serve_text_telemetry_end(telemetry, request.receipt,
                                    LOOM_SERVE_TELEMETRY_COMPLETED, 0, sent);
      loom_serve_http_connection_finish(request.connection);
    } else {
      loom_serve_text_telemetry_end(telemetry, request.receipt,
                                    LOOM_SERVE_TELEMETRY_CANCELLED, 0, sent);
      iree_status_free(sent);
      loom_serve_http_connection_abort(request.connection);
    }
  } else {
    loom_serve_text_telemetry_end(telemetry, request.receipt,
                                  LOOM_SERVE_TELEMETRY_FAILED, 0, status);
    loom_serve_http_connection_abort(request.connection);
  }
  return status;
}

iree_status_t loom_serve_text_checkpoint_request_submit(
    loom_serve_text_checkpoint_request_t* control,
    loom_serve_text_checkpoint_registry_t* registry,
    loom_serve_text_checkpoint_source_t source, bool suspension_supported,
    loom_serve_text_telemetry_t* telemetry, iree_string_builder_t* scratch,
    loom_serve_execution_frontier_t* out_frontier, bool* trim_pending,
    iree_allocator_t host_allocator) {
  const bool create = control->action == LOOM_SERVE_TEXT_CHECKPOINT_ACTION_PIN;
  const bool suspend =
      control->action == LOOM_SERVE_TEXT_CHECKPOINT_ACTION_SUSPEND;
  loom_serve_text_checkpoint_entry_t* checkpoint =
      loom_serve_text_checkpoint_registry_find(registry, control->name);
  if (create && checkpoint) {
    return checkpoint_reject(
        control, telemetry, 409, "Conflict",
        iree_make_status(IREE_STATUS_ALREADY_EXISTS,
                         "checkpoint name is already pinned"),
        host_allocator);
  }
  if (!create && !checkpoint) {
    return checkpoint_reject(
        control, telemetry, 404, "Not Found",
        iree_make_status(IREE_STATUS_NOT_FOUND, "checkpoint is not pinned"),
        host_allocator);
  }
  if (control->action == LOOM_SERVE_TEXT_CHECKPOINT_ACTION_RELEASE &&
      checkpoint->pending) {
    return checkpoint_reject(
        control, telemetry, 409, "Conflict",
        iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                         "checkpoint is borrowed by a queued request"),
        host_allocator);
  }
  if (create) {
    if (!source.session && !source.cached) {
      return checkpoint_reject(
          control, telemetry, 404, "Not Found",
          iree_make_status(IREE_STATUS_NOT_FOUND, "session is not retained"),
          host_allocator);
    }
    if (source.busy ||
        (source.session && !source.session->completion.storage)) {
      return checkpoint_reject(
          control, telemetry, 409, "Conflict",
          iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                           "pin requires an idle completed session"),
          host_allocator);
    }
    checkpoint = loom_serve_text_checkpoint_registry_available(registry);
    if (!checkpoint) {
      return checkpoint_reject(
          control, telemetry, 503, "Service Unavailable",
          iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                           "checkpoint capacity exhausted; release a pin or "
                           "configure checkpoint_capacity"),
          host_allocator);
    }
    if (source.cached) {
      loom_serve_text_checkpoint_entry_retain(source.cached, checkpoint);
      loom_serve_text_checkpoint_entry_publish(checkpoint, control->name);
      return checkpoint_reply(
          control, telemetry, scratch, checkpoint->position,
          loom_serve_text_checkpoint_suspended_bytes(checkpoint->state));
    }
    iree_status_t status = loom_serve_text_checkpoint_entry_submit_pin(
        checkpoint, source.session->row, &source.session->completion,
        out_frontier);
    if (!iree_status_is_ok(status)) {
      loom_serve_text_telemetry_end(telemetry, control->receipt,
                                    LOOM_SERVE_TELEMETRY_FAILED, 0, status);
      loom_serve_http_connection_abort(control->connection);
      memset(control, 0, sizeof(*control));
      return status;
    }
    if (!checkpoint->state) {
      return checkpoint_reject(
          control, telemetry, 503, "Service Unavailable",
          iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                           "native pin capacity exhausted"),
          host_allocator);
    }
    control->checkpoint = checkpoint;
    return iree_ok_status();
  }
  const iree_host_size_t position = checkpoint->position;
  const iree_host_size_t host_bytes =
      loom_serve_text_checkpoint_suspended_bytes(checkpoint->state);
  if (suspend) {
    if (!suspension_supported) {
      return checkpoint_reject(
          control, telemetry, 400, "Bad Request",
          iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                           "checkpoint suspension requires elastic backing"),
          host_allocator);
    }
    if (host_bytes) {
      return checkpoint_reply(control, telemetry, scratch, position,
                              host_bytes);
    }
    iree_status_t status = loom_serve_text_checkpoint_submit_suspend(
        checkpoint->state, out_frontier);
    if (!iree_status_is_ok(status)) {
      loom_serve_text_telemetry_end(telemetry, control->receipt,
                                    LOOM_SERVE_TELEMETRY_FAILED, 0, status);
      loom_serve_http_connection_abort(control->connection);
      memset(control, 0, sizeof(*control));
      return status;
    }
    control->checkpoint = checkpoint;
    return iree_ok_status();
  }
  loom_serve_text_checkpoint_entry_release(checkpoint);
  *trim_pending = true;
  return checkpoint_reply(control, telemetry, scratch, position, 0);
}

iree_status_t loom_serve_text_checkpoint_request_complete(
    loom_serve_text_checkpoint_request_t* control, bool stopping,
    loom_serve_text_telemetry_t* telemetry, iree_string_builder_t* scratch,
    bool* trim_pending) {
  loom_serve_text_checkpoint_entry_t* checkpoint = control->checkpoint;
  if (control->action == LOOM_SERVE_TEXT_CHECKPOINT_ACTION_PIN) {
    loom_serve_text_checkpoint_entry_publish(checkpoint, control->name);
  } else {
    loom_serve_text_checkpoint_complete_suspend(checkpoint->state);
    *trim_pending = true;
  }
  if (stopping) {
    loom_serve_text_checkpoint_request_abort(control, telemetry);
    return iree_ok_status();
  }
  return checkpoint_reply(
      control, telemetry, scratch, checkpoint->position,
      loom_serve_text_checkpoint_suspended_bytes(checkpoint->state));
}

void loom_serve_text_checkpoint_request_abort(
    loom_serve_text_checkpoint_request_t* control,
    loom_serve_text_telemetry_t* telemetry) {
  if (control->connection) {
    loom_serve_text_telemetry_end(telemetry, control->receipt,
                                  telemetry->failure_code
                                      ? LOOM_SERVE_TELEMETRY_FAILED
                                      : LOOM_SERVE_TELEMETRY_CANCELLED,
                                  0, iree_ok_status());
    loom_serve_http_connection_abort(control->connection);
    memset(control, 0, sizeof(*control));
  }
}
