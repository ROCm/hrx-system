// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/service/text/response.h"

#include "loom/serve/service/text/chat.h"
#include "loom/util/json.h"

static iree_status_t loom_serve_text_response_quote(
    iree_string_builder_t* output, iree_string_view_t value) {
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
    status = loom_serve_text_response_quote(output, message);
  }
  if (iree_status_is_ok(status)) {
    status = iree_string_builder_append_cstring(
        output, ",\"type\":\"loom_request_error\"}}");
  }
  iree_status_free(failure);
  return status;
}

iree_status_t loom_serve_text_response_tokens(
    loom_serve_text_session_t* session, iree_string_view_t name,
    iree_host_size_t count, const int32_t* tokens,
    iree_string_builder_t* scratch, loom_serve_text_tool_calls_t* tool_calls,
    loom_serve_text_chat_scratch_t* completion_scratch,
    iree_allocator_t scratch_allocator) {
  iree_status_t status = loom_serve_text_session_append_tokens(
      session, name, count, tokens, scratch, tool_calls, completion_scratch,
      scratch_allocator);
  if (!iree_status_is_ok(status)) {
    // Device work completed, but translation cannot establish a client history.
    // Streamed headers are already published; JSON headers are not. Keep the
    // wire format honest without changing request or row retirement.
    loom_serve_text_chat_completion_deinitialize(&session->completion);
    loom_serve_telemetry_diagnostic_capture(status,
                                            &session->request.diagnostic);
    iree_string_builder_reset(&session->packet);
    session->request.publication.pending = false;
    const bool streaming = iree_any_bit_set(session->request.chat.flags,
                                            LOOM_SERVE_TEXT_CHAT_FLAG_STREAM);
    iree_status_t packet_status = iree_string_builder_append_cstring(
        &session->packet,
        streaming
            ? "data: "
            : "HTTP/1.1 500 Internal Server Error\r\n"
              "Content-Type: application/json\r\nConnection: close\r\n\r\n");
    if (iree_status_is_ok(packet_status)) {
      packet_status = loom_serve_text_response_error(status, &session->packet);
    } else {
      packet_status = iree_status_join(packet_status, status);
    }
    if (iree_status_is_ok(packet_status) && streaming) {
      packet_status = iree_string_builder_append_cstring(
          &session->packet, "\n\ndata: [DONE]\n\n");
    }
    session->request.finish_reason = "error";
    session->request.phase = LOOM_SERVE_TEXT_REQUEST_FINISHING;
    status = packet_status;
  }
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
