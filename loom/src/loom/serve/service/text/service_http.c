// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/service/text/service_http.h"

#include "loom/serve/service/endpoint.h"
#include "loom/serve/service/text/checkpoint_http.h"
#include "loom/serve/service/text/http.h"
#include "loom/serve/service/text/input_count_http.h"

// Stack-only reply context; native accounting surrounds synchronous delivery.
typedef struct loom_serve_text_http_rejection_t {
  // Claimed connection, consumed by the reply on every return.
  loom_serve_http_connection_t* connection;
  // Selected HTTP status code.
  int code;
  // Process-lifetime HTTP status reason.
  const char* reason;
} loom_serve_text_http_rejection_t;

static iree_status_t loom_serve_text_http_reply(
    void* user_data, iree_status_t failure,
    loom_serve_text_telemetry_t* telemetry, loom_serve_text_receipt_t receipt,
    iree_allocator_t host_allocator) {
  const loom_serve_text_http_rejection_t* rejection = user_data;
  return loom_serve_text_http_reply_failure(
      telemetry, rejection->connection, receipt, rejection->code,
      rejection->reason, failure, host_allocator);
}

static iree_status_t loom_serve_text_http_decline(
    loom_serve_text_service_t* service,
    loom_serve_http_connection_t* connection, iree_time_t arrival_time,
    int code, const char* reason, iree_status_t failure) {
  loom_serve_text_http_rejection_t rejection = {
      .connection = connection, .code = code, .reason = reason};
  return loom_serve_text_service_reject(service, arrival_time, failure,
                                        loom_serve_text_http_reply, &rejection);
}

// Receipt construction runs only after native selectors and credit succeed.
typedef struct loom_serve_text_http_admission_t {
  // Claimed connection, transferred only by successful construction.
  loom_serve_http_connection_t* connection;
  // Selects known-input delivery instead of chat output encoding.
  bool sequence;
  // Distinguishes receipt-allocation failure from input rejection.
  bool allocation_attempted;
} loom_serve_text_http_admission_t;

static iree_status_t loom_serve_text_http_create_consumer(
    void* user_data, iree_allocator_t host_allocator,
    loom_serve_text_consumer_t* out_consumer) {
  loom_serve_text_http_admission_t* admission = user_data;
  admission->allocation_attempted = true;
  return loom_serve_text_http_consumer_allocate(
      admission->connection, admission->sequence, host_allocator, out_consumer);
}

static iree_status_t loom_serve_text_http_finish_admission(
    loom_serve_text_service_t* service,
    loom_serve_http_connection_t* connection, iree_time_t arrival_time,
    bool accepted, iree_status_t status, const char* pressure_message,
    bool allocation_attempted) {
  if (iree_status_is_ok(status) && accepted) {
    return status;
  }
  int code = 400;
  const char* reason = "Bad Request";
  if (iree_status_is_ok(status)) {
    status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED, "%s",
                              pressure_message);
    code = 503;
    reason = "Service Unavailable";
  } else if (allocation_attempted || iree_status_is_unavailable(status)) {
    code = 503;
    reason = "Service Unavailable";
  } else if (iree_status_is_not_found(status)) {
    code = 404;
    reason = "Not Found";
  } else if (iree_status_is_already_exists(status)) {
    code = 409;
    reason = "Conflict";
  }
  return loom_serve_text_http_decline(service, connection, arrival_time, code,
                                      reason, status);
}

static iree_status_t loom_serve_text_http_accept(
    void* self, loom_serve_http_connection_t* connection,
    const loom_serve_http_request_t* request,
    loom_runtime_service_flags_t flags) {
  loom_serve_text_service_t* service = self;
  const iree_time_t arrival_time = iree_time_now();
  if (iree_any_bit_set(flags, LOOM_RUNTIME_SERVICE_FLAG_STOPPING)) {
    loom_serve_text_service_stop(service);
  }
  if (!loom_serve_text_service_is_accepting(service)) {
    return loom_serve_text_http_decline(
        service, connection, arrival_time, 503, "Service Unavailable",
        iree_make_status(IREE_STATUS_UNAVAILABLE, "text service is stopping"));
  }
  iree_string_view_t checkpoint_name = request->target;
  if (iree_string_view_consume_prefix(&checkpoint_name,
                                      IREE_SV("/v1/checkpoints/"))) {
    loom_serve_text_checkpoint_request_t input;
    iree_status_t status =
        loom_serve_text_checkpoint_http_parse(request, &input);
    bool accepted = false;
    if (iree_status_is_ok(status)) {
      status = loom_serve_text_service_try_accept_checkpoint(
          service, &input, arrival_time,
          loom_serve_text_checkpoint_http_consumer(connection), &accepted);
    }
    return loom_serve_text_http_finish_admission(
        service, connection, arrival_time, accepted, status,
        "model control queue is full", false);
  }
  const bool input_count = iree_string_view_equal(
      request->target, IREE_SV("/v1/chat/completions/input_tokens"));
  const bool encodes =
      loom_serve_text_service_kind(service) == LOOM_SERVICE_KIND_SEQUENCE;
  const bool sequence_endpoint =
      iree_string_view_equal(request->target, IREE_SV("/v1/sequences"));
  if (!iree_string_view_equal(request->method, IREE_SV("POST")) ||
      (encodes ? !sequence_endpoint
               : (!input_count &&
                  !iree_string_view_equal(request->target,
                                          IREE_SV("/v1/chat/completions"))))) {
    return loom_serve_text_http_decline(
        service, connection, arrival_time, 404, "Not Found",
        iree_make_status(
            IREE_STATUS_NOT_FOUND,
            "use the model's declared endpoint or /v1/checkpoints/NAME; "
            "generative models also support "
            "/v1/chat/completions/input_tokens"));
  }
  iree_string_view_t content_type;
  if (!loom_serve_http_request_lookup_header(request, IREE_SV("Content-Type"),
                                             &content_type) ||
      !(iree_string_view_equal_case(content_type,
                                    IREE_SV("application/json")) ||
        iree_string_view_equal_case(
            content_type, IREE_SV("application/json; charset=utf-8")))) {
    return loom_serve_text_http_decline(
        service, connection, arrival_time, 415, "Unsupported Media Type",
        iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                         "Content-Type must be application/json"));
  }
  iree_string_view_t session = iree_string_view_empty();
  const bool named = loom_serve_http_request_lookup_header(
      request, IREE_SV("X-Loom-Session"), &session);
  checkpoint_name = iree_string_view_empty();
  const bool checkpoint = loom_serve_http_request_lookup_header(
      request, IREE_SV("X-Loom-Checkpoint"), &checkpoint_name);
  if (input_count) {
    if (named || checkpoint) {
      return loom_serve_text_http_decline(
          service, connection, arrival_time, 400, "Bad Request",
          iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                           "input counting is stateless; omit X-Loom-Session "
                           "and X-Loom-Checkpoint"));
    }
    bool accepted = false;
    iree_status_t status = loom_serve_text_service_try_accept_input_count(
        service, request->body, arrival_time,
        loom_serve_text_input_count_http_consumer(connection), &accepted);
    return loom_serve_text_http_finish_admission(
        service, connection, arrival_time, accepted, status,
        "input counting queue is full", false);
  }
  if (encodes && named) {
    return loom_serve_text_http_decline(
        service, connection, arrival_time, 400, "Bad Request",
        iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                         "sequences use a body session handle, not "
                         "X-Loom-Session"));
  }
  if (checkpoint && !checkpoint_name.size) {
    return loom_serve_text_http_decline(
        service, connection, arrival_time, 400, "Bad Request",
        iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                         "X-Loom-Checkpoint requires a nonempty identifier"));
  }
  loom_serve_text_http_admission_t admission = {.connection = connection,
                                                .sequence = encodes};
  bool accepted = false;
  iree_status_t status = loom_serve_text_service_try_accept(
      service, request->body, session, checkpoint_name, arrival_time,
      loom_serve_text_http_create_consumer, &admission, &accepted);
  return loom_serve_text_http_finish_admission(
      service, connection, arrival_time, accepted, status,
      "pending request queue is full", admission.allocation_attempted);
}

static iree_status_t loom_serve_text_http_open(
    void* self, loom_serve_text_request_t* request,
    loom_serve_text_stream_t** out_stream) {
  return loom_serve_text_service_try_open(self, request, out_stream);
}

static iree_status_t loom_serve_text_http_open_checkpoint(
    void* self, const loom_serve_text_checkpoint_request_t* request,
    loom_serve_text_checkpoint_control_t** out_control) {
  return loom_serve_text_service_try_open_checkpoint(self, request,
                                                     out_control);
}

loom_serve_service_t loom_serve_text_http_interface(
    loom_serve_text_service_t* service) {
  const loom_serve_text_service_interface_t text =
      loom_serve_text_service_interface(service);
  static const iree_string_view_t modalities[] = {IREE_SVL("text")};
  static const iree_string_view_t image_modalities[] = {IREE_SVL("text"),
                                                        IREE_SVL("image")};
  static const iree_string_view_t token_modalities[] = {IREE_SVL("tokens")};
  static const iree_string_view_t activation_modalities[] = {
      IREE_SVL("activations")};
  const bool encodes = text.kind == LOOM_SERVICE_KIND_SEQUENCE;
  return (loom_serve_service_t){
      .name = text.name,
      .runtime = text.runtime,
      .accept = loom_serve_text_http_accept,
      .try_open_text = encodes ? NULL : loom_serve_text_http_open,
      .try_open_text_checkpoint = loom_serve_text_http_open_checkpoint,
      .tokenizer = text.tokenizer,
      .kind = text.kind,
      .endpoint = loom_serve_service_endpoint(text.kind),
      .capabilities =
          {.input_modalities = {text.image_input
                                    ? IREE_ARRAYSIZE(image_modalities)
                                    : IREE_ARRAYSIZE(modalities),
                                text.image_input ? image_modalities
                                : encodes        ? token_modalities
                                                 : modalities},
           .output_modalities = {IREE_ARRAYSIZE(modalities),
                                 encodes ? activation_modalities : modalities},
           .context_length = text.context_length,
           .max_output_tokens = encodes ? 0 : text.context_length,
           .input_token_count = !encodes,
           .reasoning = text.reasoning},
      .snapshot = text.snapshot,
      .inspection = text.inspection,
      .reconfiguration = text.reconfiguration,
      .telemetry = {.stream = text.telemetry.stream,
                    .counters = text.telemetry.counters,
                    .latency = text.telemetry.latency,
                    .sample = text.telemetry.sample}};
}
