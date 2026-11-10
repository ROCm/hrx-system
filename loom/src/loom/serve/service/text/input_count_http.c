// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/service/text/input_count_http.h"

#include "loom/serve/service/text/http.h"

static bool loom_serve_text_input_count_http_open(void* self) {
  return !loom_serve_http_connection_failed(self);
}

static void loom_serve_text_input_count_http_abort(
    void* self, loom_service_telemetry_outcome_t outcome,
    const iree_status_t status, loom_serve_text_telemetry_t* telemetry,
    loom_serve_text_receipt_t receipt) {
  loom_serve_text_telemetry_end(telemetry, receipt, outcome, 0, status);
  loom_serve_http_connection_abort(self);
}

static iree_status_t loom_serve_text_input_count_http_end(
    void* self, iree_host_size_t input_tokens, const iree_status_t failure,
    loom_serve_text_telemetry_t* telemetry, loom_serve_text_receipt_t receipt,
    iree_string_builder_t* scratch) {
  if (!iree_status_is_ok(failure)) {
    const bool unsupported = iree_status_is_unimplemented(failure);
    const bool exhausted = iree_status_is_resource_exhausted(failure);
    return loom_serve_text_http_reject(telemetry, self, receipt,
                                       unsupported ? 501
                                       : exhausted ? 503
                                                   : 400,
                                       unsupported ? "Not Implemented"
                                       : exhausted ? "Service Unavailable"
                                                   : "Bad Request",
                                       iree_status_clone(failure),
                                       scratch->allocator);
  }
  iree_string_builder_reset(scratch);
  iree_status_t status = iree_string_builder_append_format(
      scratch,
      "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
      "Connection: close\r\n\r\n{\"input_tokens\":%zu}",
      input_tokens);
  if (!iree_status_is_ok(status)) {
    loom_serve_text_input_count_http_abort(self, LOOM_SERVICE_TELEMETRY_FAILED,
                                           status, telemetry, receipt);
    return status;
  }
  iree_status_t sent =
      loom_serve_http_connection_send(self, iree_string_builder_view(scratch));
  if (iree_status_is_ok(sent)) {
    loom_serve_text_telemetry_end(telemetry, receipt,
                                  LOOM_SERVICE_TELEMETRY_COMPLETED, 0, sent);
    loom_serve_http_connection_finish(self);
  } else {
    loom_serve_text_input_count_http_abort(
        self, LOOM_SERVICE_TELEMETRY_CANCELLED, sent, telemetry, receipt);
    iree_status_free(sent);
  }
  return iree_ok_status();
}

iree_status_t loom_serve_text_input_count_http_enqueue(
    loom_serve_text_input_count_t* queue,
    loom_serve_http_connection_t* connection,
    const loom_serve_http_request_t* request, loom_serve_text_receipt_t receipt,
    loom_serve_text_telemetry_t* telemetry) {
  iree_string_view_t header;
  if (loom_serve_http_request_lookup_header(request, IREE_SV("X-Loom-Session"),
                                            &header) ||
      loom_serve_http_request_lookup_header(
          request, IREE_SV("X-Loom-Checkpoint"), &header)) {
    return loom_serve_text_http_reject(
        telemetry, connection, receipt, 400, "Bad Request",
        iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                         "input counting is stateless; omit X-Loom-Session "
                         "and X-Loom-Checkpoint"),
        queue->host_allocator);
  }
  if (queue->count == queue->capacity) {
    return loom_serve_text_http_reject(
        telemetry, connection, receipt, 503, "Service Unavailable",
        iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                         "input counting queue is full"),
        queue->host_allocator);
  }
  static const loom_serve_text_input_count_consumer_vtable_t vtable = {
      .is_open = loom_serve_text_input_count_http_open,
      .end = loom_serve_text_input_count_http_end,
      .abort = loom_serve_text_input_count_http_abort};
  loom_serve_text_input_count_enqueue(
      queue, request->body,
      (loom_serve_text_input_count_consumer_t){.self = connection,
                                               .vtable = &vtable},
      receipt);
  return iree_ok_status();
}
