// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/service/text/response.h"

#include "loom/util/json.h"

static iree_status_t text_quote(iree_string_builder_t* output,
                                iree_string_view_t value) {
  loom_output_stream_t stream;
  loom_output_stream_for_builder(output, &stream);
  return loom_json_write_escaped_string(&stream, value);
}

// Error translation is an intentional status ownership boundary. Original
// failures have already been copied for owner publication; failure to construct
// the response propagates.
iree_status_t loom_serve_text_response_error(iree_status_t failure,
                                             iree_string_builder_t* output) {
  iree_status_t status =
      iree_string_builder_append_cstring(output, "{\"error\":{\"message\":");
  if (iree_status_is_ok(status)) {
    iree_string_view_t message = iree_status_message(failure);
    if (!message.size) {
      message = iree_make_cstring_view(
          iree_status_code_string(iree_status_code(failure)));
    }
    status = text_quote(output, message);
  }
  if (iree_status_is_ok(status)) {
    status = iree_string_builder_append_cstring(
        output, ",\"type\":\"loom_request_error\"}}");
  }
  iree_status_free(failure);
  return status;
}

iree_status_t loom_serve_text_response_reject(
    loom_serve_text_telemetry_t* telemetry,
    loom_serve_http_connection_t* connection, loom_serve_text_receipt_t receipt,
    int code, const char* reason, iree_status_t failure,
    iree_allocator_t host_allocator) {
  // Input rejection may overlap a CPU output job using model scratch.
  loom_serve_text_telemetry_end(telemetry, receipt,
                                LOOM_SERVE_TELEMETRY_REJECTED, 0, failure);
  iree_string_builder_t response;
  iree_string_builder_initialize(host_allocator, &response);
  iree_status_t status = iree_string_builder_append_format(
      &response,
      "HTTP/1.1 %d %s\r\nContent-Type: application/json\r\nConnection: "
      "close\r\n\r\n",
      code, reason);
  if (iree_status_is_ok(status)) {
    status = loom_serve_text_response_error(failure, &response);
  } else {
    status = iree_status_join(status, failure);
  }
  if (iree_status_is_ok(status)) {
    iree_status_t send_status = loom_serve_http_connection_send(
        connection, iree_string_builder_view(&response));
    if (iree_status_is_ok(send_status)) {
      loom_serve_http_connection_finish(connection);
    } else {
      loom_serve_text_telemetry_diagnose(telemetry, receipt.serial,
                                         send_status);
      loom_serve_http_connection_abort(connection);
    }
  } else {
    loom_serve_http_connection_abort(connection);
  }
  iree_string_builder_deinitialize(&response);
  return status;
}
