// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/service/text/checkpoint_http.h"

#include "loom/serve/service/text/http.h"

iree_status_t loom_serve_text_checkpoint_http_parse(
    const loom_serve_http_request_t* request,
    loom_serve_text_checkpoint_request_t* out_request) {
  *out_request = (loom_serve_text_checkpoint_request_t){0};
  iree_string_view_t name = request->target;
  const bool checkpoint =
      iree_string_view_consume_prefix(&name, IREE_SV("/v1/checkpoints/"));
  const bool suspend =
      iree_string_view_consume_suffix(&name, IREE_SV("/suspend"));
  const bool post = iree_string_view_equal(request->method, IREE_SV("POST"));
  const bool create = post && !suspend;
  const bool release =
      !suspend && iree_string_view_equal(request->method, IREE_SV("DELETE"));
  if (!checkpoint || (!create && !release && !(suspend && post)) ||
      request->body.size) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "use POST/DELETE /v1/checkpoints/NAME or POST "
                            "/v1/checkpoints/NAME/suspend with an empty body");
  }
  *out_request = (loom_serve_text_checkpoint_request_t){
      .action = create    ? LOOM_SERVE_TEXT_CHECKPOINT_ACTION_PIN
                : suspend ? LOOM_SERVE_TEXT_CHECKPOINT_ACTION_SUSPEND
                          : LOOM_SERVE_TEXT_CHECKPOINT_ACTION_RELEASE,
      .name = name};
  if (create) {
    loom_serve_http_request_lookup_header(request, IREE_SV("X-Loom-Session"),
                                          &out_request->session);
  }
  return loom_serve_text_checkpoint_request_validate(out_request);
}

static bool loom_serve_text_checkpoint_http_open(void* self) {
  return !loom_serve_http_connection_failed(self);
}

static void loom_serve_text_checkpoint_http_abort(
    void* self, loom_serve_telemetry_outcome_t outcome,
    const iree_status_t status, loom_serve_text_telemetry_t* telemetry,
    const loom_serve_text_receipt_t* receipt) {
  loom_serve_text_telemetry_end(telemetry, *receipt, outcome, 0, status);
  loom_serve_http_connection_abort(self);
}

static iree_status_t loom_serve_text_checkpoint_http_end(
    void* self, const loom_serve_text_checkpoint_request_t* request,
    const loom_serve_text_checkpoint_result_t* result,
    const iree_status_t failure, loom_serve_text_telemetry_t* telemetry,
    const loom_serve_text_receipt_t* receipt, iree_string_builder_t* scratch) {
  const bool create = request->action == LOOM_SERVE_TEXT_CHECKPOINT_ACTION_PIN;
  const bool release =
      request->action == LOOM_SERVE_TEXT_CHECKPOINT_ACTION_RELEASE;
  if (!result) {
    int code = 503;
    const char* reason = "Service Unavailable";
    if (iree_status_is_not_found(failure)) {
      code = 404;
      reason = "Not Found";
    } else if (iree_status_is_already_exists(failure) ||
               (iree_status_is_failed_precondition(failure) &&
                request->action != LOOM_SERVE_TEXT_CHECKPOINT_ACTION_SUSPEND)) {
      code = 409;
      reason = "Conflict";
    } else if (iree_status_is_failed_precondition(failure)) {
      code = 400;
      reason = "Bad Request";
    }
    return loom_serve_text_http_reject(telemetry, self, *receipt, code, reason,
                                       iree_status_clone(failure),
                                       scratch->allocator);
  }
  iree_string_builder_reset(scratch);
  iree_status_t status = iree_string_builder_append_format(
      scratch,
      "HTTP/1.1 %s\r\nContent-Type: application/json\r\nConnection: "
      "close\r\n\r\n"
      "{\"checkpoint\":\"%.*s\",\"position\":%zu,\"pinned\":%s,"
      "\"host_snapshot_bytes\":%zu}",
      create ? "201 Created" : "200 OK", (int)request->name.size,
      request->name.data, result->position, release ? "false" : "true",
      result->host_bytes);
  if (!iree_status_is_ok(status)) {
    loom_serve_text_checkpoint_http_abort(self, LOOM_SERVE_TELEMETRY_FAILED,
                                          status, telemetry, receipt);
    return status;
  }
  iree_status_t sent =
      loom_serve_http_connection_send(self, iree_string_builder_view(scratch));
  if (iree_status_is_ok(sent)) {
    loom_serve_text_telemetry_end(telemetry, *receipt,
                                  LOOM_SERVE_TELEMETRY_COMPLETED, 0, sent);
    loom_serve_http_connection_finish(self);
  } else {
    loom_serve_text_checkpoint_http_abort(self, LOOM_SERVE_TELEMETRY_CANCELLED,
                                          sent, telemetry, receipt);
    iree_status_free(sent);
  }
  return iree_ok_status();
}

loom_serve_text_checkpoint_consumer_t loom_serve_text_checkpoint_http_consumer(
    loom_serve_http_connection_t* connection) {
  return (loom_serve_text_checkpoint_consumer_t){
      .self = connection,
      .is_open = loom_serve_text_checkpoint_http_open,
      .end = loom_serve_text_checkpoint_http_end,
      .abort = loom_serve_text_checkpoint_http_abort};
}
