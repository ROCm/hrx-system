// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/service/text/input_count.h"

#include <string.h>

#include "loom/serve/service/text/http.h"

iree_status_t loom_serve_text_input_count_initialize(
    iree_host_size_t capacity, iree_allocator_t host_allocator,
    loom_serve_text_input_count_t* out_queue) {
  *out_queue = (loom_serve_text_input_count_t){.host_allocator = host_allocator,
                                               .capacity = capacity};
  return iree_allocator_malloc_array(host_allocator, capacity,
                                     sizeof(*out_queue->requests),
                                     (void**)&out_queue->requests);
}

void loom_serve_text_input_count_deinitialize(
    loom_serve_text_input_count_t* queue) {
  iree_allocator_free(queue->host_allocator, queue->requests);
  memset(queue, 0, sizeof(*queue));
}

static void loom_serve_text_input_count_remove(
    loom_serve_text_input_count_t* queue, iree_host_size_t index) {
  --queue->count;
  memmove(queue->requests + index, queue->requests + index + 1,
          (queue->count - index) * sizeof(*queue->requests));
}

iree_status_t loom_serve_text_input_count_enqueue(
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
  queue->requests[queue->count++] = (loom_serve_text_input_count_request_t){
      .connection = connection, .body = request->body, .receipt = receipt};
  return iree_ok_status();
}

iree_status_t loom_serve_text_input_count_run(
    loom_serve_text_input_count_t* queue,
    const loom_serve_text_chat_policy_t* policy, iree_string_view_t name,
    iree_host_size_t default_max_tokens, iree_host_size_t context_capacity) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_TEXT(z0, name.data, name.size);
  const loom_serve_text_input_count_request_t request = queue->requests[0];
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, request.receipt.serial);
  queue->input_tokens = 0;
  loom_serve_text_chat_t chat = {0};
  iree_status_t status = loom_serve_text_chat_initialize(
      policy, name, request.body, request.receipt.serial, default_max_tokens,
      queue->host_allocator, &chat);
  loom_serve_text_media_span_t* media_spans = NULL;
  iree_host_size_t media_span_count = 0;
  if (iree_status_is_ok(status)) {
    status = loom_serve_text_chat_prepare_input(
        &chat, 0, LOOM_SERVE_TEXT_CHAT_INPUT_RENDERED,
        LOOM_SERVE_TEXT_CHAT_BOUNDARY_FRESH, context_capacity, 0,
        queue->host_allocator, NULL, &queue->input_tokens, &media_spans,
        &media_span_count);
  }
  iree_allocator_free(queue->host_allocator, media_spans);
  loom_serve_text_chat_deinitialize(&chat);
  IREE_RETURN_AND_END_ZONE(z0, status);
}

iree_status_t loom_serve_text_input_count_complete(
    loom_serve_text_input_count_t* queue, bool stopping, iree_status_t status,
    loom_serve_text_telemetry_t* telemetry, iree_string_builder_t* scratch) {
  const loom_serve_text_input_count_request_t request = queue->requests[0];
  loom_serve_text_input_count_remove(queue, 0);
  const bool request_error = iree_status_is_invalid_argument(status) ||
                             iree_status_is_out_of_range(status) ||
                             iree_status_is_unimplemented(status) ||
                             iree_status_is_resource_exhausted(status);
  if (!iree_status_is_ok(status) && !request_error) {
    // A disconnected client cannot hide a source failure from the model owner.
    loom_serve_text_telemetry_end(telemetry, request.receipt,
                                  LOOM_SERVE_TELEMETRY_FAILED, 0, status);
    loom_serve_http_connection_abort(request.connection);
    return status;
  }
  if (stopping || loom_serve_http_connection_failed(request.connection)) {
    loom_serve_text_telemetry_end(telemetry, request.receipt,
                                  telemetry->failure_code
                                      ? LOOM_SERVE_TELEMETRY_FAILED
                                      : LOOM_SERVE_TELEMETRY_CANCELLED,
                                  0, status);
    iree_status_free(status);
    loom_serve_http_connection_abort(request.connection);
    return iree_ok_status();
  }
  if (request_error) {
    const bool unsupported = iree_status_is_unimplemented(status);
    const bool exhausted = iree_status_is_resource_exhausted(status);
    return loom_serve_text_http_reject(telemetry, request.connection,
                                       request.receipt,
                                       unsupported ? 501
                                       : exhausted ? 503
                                                   : 400,
                                       unsupported ? "Not Implemented"
                                       : exhausted ? "Service Unavailable"
                                                   : "Bad Request",
                                       status, queue->host_allocator);
  }
  iree_string_builder_reset(scratch);
  status = iree_string_builder_append_format(
      scratch,
      "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
      "Connection: close\r\n\r\n{\"input_tokens\":%zu}",
      queue->input_tokens);
  if (iree_status_is_ok(status)) {
    iree_status_t sent = loom_serve_http_connection_send(
        request.connection, iree_string_builder_view(scratch));
    loom_serve_text_telemetry_end(telemetry, request.receipt,
                                  iree_status_is_ok(sent)
                                      ? LOOM_SERVE_TELEMETRY_COMPLETED
                                      : LOOM_SERVE_TELEMETRY_CANCELLED,
                                  0, sent);
    if (iree_status_is_ok(sent)) {
      loom_serve_http_connection_finish(request.connection);
    } else {
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

bool loom_serve_text_input_count_cancel(
    loom_serve_text_input_count_t* queue, bool borrowed, bool stopping,
    loom_serve_text_telemetry_t* telemetry) {
  bool progress = false;
  for (iree_host_size_t i = borrowed ? 1 : 0; i < queue->count;) {
    const loom_serve_text_input_count_request_t request = queue->requests[i];
    if (stopping || loom_serve_http_connection_failed(request.connection)) {
      loom_serve_text_telemetry_end(telemetry, request.receipt,
                                    telemetry->failure_code
                                        ? LOOM_SERVE_TELEMETRY_FAILED
                                        : LOOM_SERVE_TELEMETRY_CANCELLED,
                                    0, iree_ok_status());
      loom_serve_http_connection_abort(request.connection);
      loom_serve_text_input_count_remove(queue, i);
      progress = true;
    } else {
      ++i;
    }
  }
  return progress;
}
