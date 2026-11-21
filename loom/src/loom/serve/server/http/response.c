// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/server/http/response.h"

#include "loom/util/json.h"

void loom_serve_http_response_send(loom_serve_http_server_t* server,
                                   loom_serve_http_connection_t* connection,
                                   iree_string_view_t response) {
  if (loom_serve_http_connection_failed(connection) ||
      !loom_serve_http_connection_can_send(connection)) {
    loom_serve_http_connection_abort(connection);
    return;
  }
  iree_status_t status = loom_serve_http_connection_send(connection, response);
  if (iree_status_is_ok(status)) {
    loom_serve_http_connection_finish(connection);
  } else {
    loom_serve_http_server_record_error(server, status);
    iree_status_free(status);
    loom_serve_http_connection_abort(connection);
  }
}

static iree_status_t loom_serve_http_write_rejection(
    loom_serve_http_server_t* server, loom_serve_http_connection_t* connection,
    int code, const char* reason, iree_string_view_t message,
    const char* error_code, iree_string_builder_t* scratch) {
  iree_string_builder_reset(scratch);
  iree_status_t status = iree_string_builder_append_format(
      scratch,
      "HTTP/1.1 %d %s\r\nContent-Type: application/json\r\nConnection: "
      "close\r\n\r\n{\"error\":{\"message\":",
      code, reason);
  if (iree_status_is_ok(status)) {
    loom_output_stream_t stream;
    loom_output_stream_for_builder(scratch, &stream);
    status = loom_json_write_escaped_string(&stream, message);
  }
  if (iree_status_is_ok(status) && error_code) {
    status = iree_string_builder_append_format(scratch, ",\"code\":\"%s\"",
                                               error_code);
  }
  if (iree_status_is_ok(status)) {
    status = iree_string_builder_append_cstring(
        scratch, ",\"type\":\"loom_route_error\"}}");
  }
  if (iree_status_is_ok(status)) {
    loom_serve_http_response_send(server, connection,
                                  iree_string_builder_view(scratch));
  } else {
    loom_serve_http_connection_abort(connection);
  }
  return status;
}

iree_status_t loom_serve_http_response_reject(
    loom_serve_http_server_t* server, loom_serve_http_connection_t* connection,
    int code, const char* reason, iree_string_view_t message,
    iree_string_builder_t* scratch) {
  return loom_serve_http_write_rejection(server, connection, code, reason,
                                         message, NULL, scratch);
}

static iree_status_t loom_serve_http_send_status(
    loom_serve_http_server_t* server, loom_serve_http_connection_t* connection,
    iree_status_t status, const char* error_code,
    iree_string_builder_t* scratch) {
  int code = 500;
  const char* reason = "Internal Server Error";
  switch (iree_status_code(status)) {
    case IREE_STATUS_INVALID_ARGUMENT:
    case IREE_STATUS_OUT_OF_RANGE:
      code = 400;
      reason = "Bad Request";
      break;
    case IREE_STATUS_NOT_FOUND:
      code = 404;
      reason = "Not Found";
      break;
    case IREE_STATUS_CANCELLED:
    case IREE_STATUS_ALREADY_EXISTS:
    case IREE_STATUS_FAILED_PRECONDITION:
      code = 409;
      reason = "Conflict";
      break;
    case IREE_STATUS_RESOURCE_EXHAUSTED:
      code = 429;
      reason = "Too Many Requests";
      break;
    case IREE_STATUS_UNIMPLEMENTED:
      code = 501;
      reason = "Not Implemented";
      break;
    default:
      break;
  }
  loom_serve_http_server_record_error(server, status);
  iree_string_view_t message = iree_status_message(status);
  if (!message.size) {
    message = iree_make_cstring_view(reason);
  }
  iree_status_t response = loom_serve_http_write_rejection(
      server, connection, code, reason, message, error_code, scratch);
  iree_status_free(status);
  return response;
}

iree_status_t loom_serve_http_response_error(
    loom_serve_http_server_t* server, loom_serve_http_connection_t* connection,
    iree_status_t status, iree_string_builder_t* scratch) {
  return loom_serve_http_send_status(server, connection, status, NULL, scratch);
}

iree_status_t loom_serve_http_response_model_not_found(
    loom_serve_http_server_t* server, loom_serve_http_connection_t* connection,
    iree_string_view_t name, iree_string_builder_t* scratch) {
  // The decoded selector may borrow scratch. Capture it before rendering.
  iree_status_t status =
      iree_make_status(IREE_STATUS_NOT_FOUND,
                       "model '%.*s' is not registered. List available "
                       "model IDs with GET /v1/models",
                       (int)name.size, name.data);
  return loom_serve_http_send_status(server, connection, status,
                                     "model_not_found", scratch);
}
