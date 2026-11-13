// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/service/finite/http.h"

#include "loom/serve/service/endpoint.h"
#include "loom/serve/transport/http/request.h"
#include "loom/serve/transport/http/server.h"
#include "loom/service/kind.h"
#include "loom/util/json.h"

static iree_status_t loom_serve_finite_http_begin(iree_string_builder_t* output,
                                                  const char* status) {
  iree_string_builder_reset(output);
  return iree_string_builder_append_format(
      output,
      "HTTP/1.1 %s\r\nContent-Type: application/json\r\n"
      "Connection: close\r\n\r\n",
      status);
}

static bool loom_serve_finite_http_open(void* self) {
  return !loom_serve_http_connection_failed(self);
}

static void loom_serve_finite_http_cancel(void* self) {
  loom_serve_http_connection_cancel(self);
}

static iree_status_t loom_serve_finite_http_send(
    loom_serve_http_connection_t* connection, iree_string_view_t response,
    bool* out_delivered) {
  *out_delivered = false;
  if (loom_serve_http_connection_failed(connection) ||
      !loom_serve_http_connection_can_send(connection)) {
    loom_serve_http_connection_abort(connection);
    return iree_ok_status();
  }
  iree_status_t status = loom_serve_http_connection_send(connection, response);
  if (iree_status_is_ok(status)) {
    loom_serve_http_connection_finish(connection);
    *out_delivered = true;
  } else {
    loom_serve_http_connection_abort(connection);
  }
  return status;
}

static iree_status_t loom_serve_finite_http_reject(
    loom_serve_http_connection_t* connection, const char* code,
    const char* type, iree_string_view_t message,
    iree_string_builder_t* scratch, bool* out_delivered) {
  *out_delivered = false;
  iree_status_t status = loom_serve_finite_http_begin(scratch, code);
  if (iree_status_is_ok(status)) {
    status = iree_string_builder_append_format(
        scratch, "{\"error\":{\"type\":\"%s\",\"message\":", type);
  }
  if (iree_status_is_ok(status)) {
    loom_output_stream_t stream;
    loom_output_stream_for_builder(scratch, &stream);
    status = loom_json_write_escaped_string(&stream, message);
  }
  if (iree_status_is_ok(status)) {
    status = iree_string_builder_append_cstring(scratch, "}}");
  }
  if (iree_status_is_ok(status)) {
    status = loom_serve_finite_http_send(
        connection, iree_string_builder_view(scratch), out_delivered);
  } else {
    loom_serve_http_connection_abort(connection);
  }
  return status;
}

static iree_status_t loom_serve_finite_http_encode(
    void* self, const loom_service_finite_protocol_t* protocol,
    const loom_service_finite_request_t* request, iree_string_view_t name,
    const iree_const_byte_span_t* rows, uint64_t input_tokens,
    iree_string_builder_t* output) {
  IREE_RETURN_IF_ERROR(loom_serve_finite_http_begin(output, "200 OK"));
  return protocol->response(request, name, rows, input_tokens, output);
}

static iree_status_t loom_serve_finite_http_end(
    void* self, const loom_service_finite_protocol_t* protocol,
    const iree_status_t status, bool rejected, iree_string_builder_t* output,
    bool* out_delivered) {
  *out_delivered = false;
  if (iree_status_is_cancelled(status) ||
      loom_serve_http_connection_failed(self)) {
    loom_serve_http_connection_abort(self);
    return iree_ok_status();
  }
  if (iree_status_is_ok(status)) {
    return loom_serve_finite_http_send(self, iree_string_builder_view(output),
                                       out_delivered);
  }
  // System One uses 422 for invalid decision inputs; embeddings uses 400.
  const char* invalid_status = protocol->kind == LOOM_SERVICE_KIND_DECISION
                                   ? "422 Unprocessable Entity"
                                   : "400 Bad Request";
  const char* code = iree_status_is_resource_exhausted(status)
                         ? "503 Service Unavailable"
                     : rejected ? invalid_status
                                : "500 Internal Server Error";
  iree_string_view_t message = iree_status_message(status);
  if (!message.size) {
    message = iree_make_cstring_view(
        iree_status_code_string(iree_status_code(status)));
  }
  return loom_serve_finite_http_reject(
      self, code, rejected ? "finite_request_error" : "loom_execution_error",
      message, output, out_delivered);
}

static loom_service_finite_consumer_t loom_serve_finite_http_consumer(
    loom_serve_http_connection_t* connection) {
  static const loom_service_finite_consumer_vtable_t vtable = {
      .is_open = loom_serve_finite_http_open,
      .cancel = loom_serve_finite_http_cancel,
      .encode = loom_serve_finite_http_encode,
      .end = loom_serve_finite_http_end,
  };
  return (loom_service_finite_consumer_t){.self = connection,
                                          .vtable = &vtable};
}

static void loom_serve_finite_http_decline(
    loom_service_finite_service_t* service, iree_time_t arrival_time,
    loom_serve_http_connection_t* connection, iree_string_view_t response) {
  bool delivered = false;
  iree_status_t status =
      loom_serve_finite_http_send(connection, response, &delivered);
  loom_service_finite_service_record_decline(service, arrival_time, delivered,
                                             status);
  iree_status_free(status);
}

static iree_status_t loom_serve_finite_http_accept(
    void* self, loom_serve_http_connection_t* connection,
    const loom_serve_http_request_t* request,
    loom_runtime_service_flags_t flags) {
  loom_service_finite_service_t* service = self;
  const iree_time_t arrival_time = iree_time_now();
  if (!iree_string_view_equal(request->method, IREE_SV("POST")) ||
      !iree_string_view_equal(
          request->target, iree_make_cstring_view(loom_serve_service_endpoint(
                               loom_service_finite_service_kind(service))))) {
    loom_serve_finite_http_decline(
        service, arrival_time, connection,
        IREE_SV("HTTP/1.1 404 Not Found\r\nContent-Type: application/json\r\n"
                "Connection: close\r\n\r\n{\"error\":{\"type\":"
                "\"finite_request_error\",\"message\":"
                "\"unknown finite-result endpoint\"}}"));
    return iree_ok_status();
  }
  iree_string_view_t content_type;
  if (!loom_serve_http_request_lookup_header(request, IREE_SV("content-type"),
                                             &content_type) ||
      !(iree_string_view_equal_case(content_type,
                                    IREE_SV("application/json")) ||
        iree_string_view_equal_case(
            content_type, IREE_SV("application/json; charset=utf-8")))) {
    loom_serve_finite_http_decline(
        service, arrival_time, connection,
        IREE_SV("HTTP/1.1 415 Unsupported Media Type\r\n"
                "Content-Type: application/json\r\nConnection: close\r\n\r\n"
                "{\"error\":{\"type\":\"finite_request_error\",\"message\":"
                "\"Content-Type must be application/json\"}}"));
    return iree_ok_status();
  }
  bool accepted = false;
  iree_status_t status = loom_service_finite_service_try_accept(
      service, request->body, arrival_time,
      loom_serve_finite_http_consumer(connection), &accepted);
  if (!iree_status_is_ok(status)) {
    loom_serve_http_connection_abort(connection);
    loom_service_finite_service_record_decline(service, arrival_time, false,
                                               status);
    return status;
  }
  if (!accepted) {
    loom_serve_finite_http_decline(
        service, arrival_time, connection,
        IREE_SV("HTTP/1.1 503 Service Unavailable\r\n"
                "Content-Type: application/json\r\nConnection: close\r\n\r\n"
                "{\"error\":{\"type\":\"finite_request_error\",\"message\":"
                "\"finite admission queue is full\"}}"));
  }
  return iree_ok_status();
}

loom_serve_service_t loom_serve_finite_http_interface(
    loom_service_finite_service_t* service) {
  const loom_service_finite_service_interface_t finite =
      loom_service_finite_service_interface(service);
  static const iree_string_view_t input_modalities[] = {IREE_SVL("text")};
  return (loom_serve_service_t){
      .name = finite.name,
      .runtime = finite.runtime,
      .accept = loom_serve_finite_http_accept,
      .tokenizer = finite.tokenizer,
      .kind = finite.kind,
      .endpoint = loom_serve_service_endpoint(finite.kind),
      .capabilities = {.input_modalities = {IREE_ARRAYSIZE(input_modalities),
                                            input_modalities},
                       .output_modalities = {1, finite.output_modality}},
      .snapshot = finite.snapshot,
      .inspection = finite.inspection,
      .reconfiguration = finite.reconfiguration,
      .telemetry = {.stream = finite.telemetry.stream,
                    .counters = finite.telemetry.counters,
                    .latency = finite.telemetry.latency}};
}
