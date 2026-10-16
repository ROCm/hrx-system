// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/service/text/http.h"

#include "loom/util/json.h"

static iree_status_t loom_serve_text_http_quote(iree_string_builder_t* output,
                                                iree_string_view_t value) {
  loom_output_stream_t stream;
  loom_output_stream_for_builder(output, &stream);
  return loom_json_write_escaped_string(&stream, value);
}

// Error translation is an intentional status ownership boundary. Original
// failures have already been copied for owner publication; failure to construct
// the response propagates.
static iree_status_t loom_serve_text_http_error(iree_status_t failure,
                                                iree_string_builder_t* output) {
  iree_status_t status =
      iree_string_builder_append_cstring(output, "{\"error\":{\"message\":");
  if (iree_status_is_ok(status)) {
    iree_string_view_t message = iree_status_message(failure);
    if (!message.size) {
      message = iree_make_cstring_view(
          iree_status_code_string(iree_status_code(failure)));
    }
    status = loom_serve_text_http_quote(output, message);
  }
  if (iree_status_is_ok(status)) {
    status = iree_string_builder_append_cstring(
        output, ",\"type\":\"loom_request_error\"}}");
  }
  iree_status_free(failure);
  return status;
}

static iree_status_t loom_serve_text_http_reply_failure(
    loom_serve_text_telemetry_t* telemetry,
    loom_serve_http_connection_t* connection, loom_serve_text_receipt_t receipt,
    int code, const char* reason, iree_status_t failure,
    iree_allocator_t host_allocator) {
  // Input rejection may overlap a CPU output job using model scratch.
  iree_string_builder_t response;
  iree_string_builder_initialize(host_allocator, &response);
  iree_status_t status = iree_string_builder_append_format(
      &response,
      "HTTP/1.1 %d %s\r\nContent-Type: application/json\r\nConnection: "
      "close\r\n\r\n",
      code, reason);
  if (iree_status_is_ok(status)) {
    status = loom_serve_text_http_error(failure, &response);
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

iree_status_t loom_serve_text_http_reject(
    loom_serve_text_telemetry_t* telemetry,
    loom_serve_http_connection_t* connection, loom_serve_text_receipt_t receipt,
    int code, const char* reason, iree_status_t failure,
    iree_allocator_t host_allocator) {
  loom_serve_text_telemetry_end(telemetry, receipt,
                                LOOM_SERVE_TELEMETRY_REJECTED, 0, failure);
  return loom_serve_text_http_reply_failure(
      telemetry, connection, receipt, code, reason, failure, host_allocator);
}

typedef struct loom_serve_text_http_consumer_t {
  // Allocator owning this receipt and its staged encoded packet.
  iree_allocator_t host_allocator;
  // Claimed immutable input and owner-only delivery state.
  loom_serve_http_connection_t* connection;
  // Borrowed owner accounting, used only for rejected-response send errors.
  loom_serve_text_telemetry_t* telemetry;
  // Stable protocol identity for rejected-response diagnostics.
  loom_serve_text_receipt_t receipt;
  // CPU-written stage, published only after its finite borrower retires.
  iree_string_builder_t packet;
} loom_serve_text_http_consumer_t;

static bool loom_serve_text_http_open(void* self) {
  const loom_serve_text_http_consumer_t* consumer = self;
  return !loom_serve_http_connection_failed(consumer->connection);
}

static bool loom_serve_text_http_blocked(void* self) {
  const loom_serve_text_http_consumer_t* consumer = self;
  return iree_string_builder_size(&consumer->packet) &&
         !loom_serve_http_connection_can_send(consumer->connection);
}

static iree_status_t loom_serve_text_http_begin(
    void* self, const loom_serve_text_chat_response_t* response) {
  loom_serve_text_http_consumer_t* consumer = self;
  IREE_RETURN_IF_ERROR(iree_string_builder_reserve(&consumer->packet, 16384));
  if (iree_any_bit_set(response->flags, LOOM_SERVE_TEXT_CHAT_FLAG_STREAM)) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(
        &consumer->packet,
        "HTTP/1.1 200 OK\r\nContent-Type: "
        "text/event-stream\r\nCache-Control: "
        "no-cache\r\nConnection: close\r\n\r\n"));
    return loom_serve_text_chat_event(
        response, IREE_SV("{\"role\":\"assistant\"}"), iree_string_view_empty(),
        &consumer->packet);
  }
  return iree_ok_status();
}

static iree_status_t loom_serve_text_http_append(
    void* self, const loom_serve_text_chat_response_t* response,
    const loom_serve_text_chat_output_t* output,
    iree_string_builder_t* scratch) {
  loom_serve_text_http_consumer_t* consumer = self;
  iree_string_builder_reset(scratch);
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(
      scratch, output->kind == LOOM_SERVE_TEXT_CHAT_OUTPUT_REASONING
                   ? "{\"reasoning_content\":"
                   : "{\"content\":"));
  IREE_RETURN_IF_ERROR(loom_serve_text_http_quote(scratch, output->text));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(scratch, "}"));
  return loom_serve_text_chat_event(response, iree_string_builder_view(scratch),
                                    iree_string_view_empty(),
                                    &consumer->packet);
}

static iree_status_t loom_serve_text_http_complete(
    void* self, const loom_serve_text_chat_response_t* response,
    iree_string_builder_t* content, iree_string_builder_t* reasoning,
    loom_serve_text_tool_calls_t* calls, iree_string_view_t finish_reason,
    iree_string_builder_t* scratch) {
  loom_serve_text_http_consumer_t* consumer = self;
  if (iree_any_bit_set(response->flags, LOOM_SERVE_TEXT_CHAT_FLAG_STREAM)) {
    if (calls->count) {
      iree_string_builder_reset(scratch);
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(scratch, "{\"tool_calls\":"));
      IREE_RETURN_IF_ERROR(
          loom_serve_text_tool_calls_write_json(calls, true, scratch));
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(scratch, "}"));
      IREE_RETURN_IF_ERROR(loom_serve_text_chat_event(
          response, iree_string_builder_view(scratch), iree_string_view_empty(),
          &consumer->packet));
    }
    IREE_RETURN_IF_ERROR(loom_serve_text_chat_event(
        response, IREE_SV("{}"), finish_reason, &consumer->packet));
    if (iree_any_bit_set(response->flags,
                         LOOM_SERVE_TEXT_CHAT_FLAG_INCLUDE_USAGE)) {
      IREE_RETURN_IF_ERROR(
          loom_serve_text_chat_usage(response, &consumer->packet));
    }
    return iree_string_builder_append_cstring(&consumer->packet,
                                              "data: [DONE]\n\n");
  }
  // Nonstreaming headers remain unpublished until translation succeeds.
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(
      &consumer->packet,
      "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
      "Connection: close\r\n\r\n"));
  return loom_serve_text_chat_response(response,
                                       iree_string_builder_view(content),
                                       iree_string_builder_view(reasoning),
                                       calls, finish_reason, &consumer->packet);
}

static iree_status_t loom_serve_text_http_fail(
    void* self, loom_serve_text_chat_flags_t flags, iree_status_t failure) {
  loom_serve_text_http_consumer_t* consumer = self;
  iree_string_builder_reset(&consumer->packet);
  const bool streaming =
      iree_any_bit_set(flags, LOOM_SERVE_TEXT_CHAT_FLAG_STREAM);
  iree_status_t status = iree_string_builder_append_cstring(
      &consumer->packet,
      streaming
          ? "data: "
          : "HTTP/1.1 500 Internal Server Error\r\n"
            "Content-Type: application/json\r\nConnection: close\r\n\r\n");
  if (iree_status_is_ok(status)) {
    status = loom_serve_text_http_error(failure, &consumer->packet);
  } else {
    status = iree_status_join(status, failure);
  }
  if (iree_status_is_ok(status) && streaming) {
    status = iree_string_builder_append_cstring(&consumer->packet,
                                                "\n\ndata: [DONE]\n\n");
  }
  return status;
}

static iree_status_t loom_serve_text_http_publish(void* self, bool* out_ready,
                                                  bool* out_progress) {
  loom_serve_text_http_consumer_t* consumer = self;
  *out_ready = false;
  *out_progress = false;
  const iree_string_view_t packet = iree_string_builder_view(&consumer->packet);
  if (packet.size) {
    if (!loom_serve_http_connection_can_send(consumer->connection)) {
      return iree_ok_status();
    }
    IREE_RETURN_IF_ERROR(
        loom_serve_http_connection_send(consumer->connection, packet));
    iree_string_builder_reset(&consumer->packet);
    *out_progress = true;
  }
  *out_ready = true;
  return iree_ok_status();
}

static void loom_serve_text_http_release(
    loom_serve_text_http_consumer_t* consumer) {
  iree_string_builder_deinitialize(&consumer->packet);
  iree_allocator_free(consumer->host_allocator, consumer);
}

static iree_status_t loom_serve_text_http_reject_consumer(
    void* self, iree_status_t failure) {
  loom_serve_text_http_consumer_t* consumer = self;
  int code = 503;
  const char* reason = "Service Unavailable";
  if (iree_status_is_invalid_argument(failure) ||
      iree_status_is_out_of_range(failure)) {
    code = 400;
    reason = "Bad Request";
  } else if (iree_status_is_unimplemented(failure)) {
    code = 501;
    reason = "Not Implemented";
  }
  const iree_status_t status = loom_serve_text_http_reply_failure(
      consumer->telemetry, consumer->connection, consumer->receipt, code,
      reason, failure, consumer->host_allocator);
  loom_serve_text_http_release(consumer);
  return status;
}

static void loom_serve_text_http_end(void* self, bool complete,
                                     iree_status_code_t code) {
  loom_serve_text_http_consumer_t* consumer = self;
  if (complete) {
    loom_serve_http_connection_finish(consumer->connection);
  } else {
    loom_serve_http_connection_abort(consumer->connection);
  }
  loom_serve_text_http_release(consumer);
}

static const loom_serve_text_consumer_vtable_t loom_serve_text_http_vtable = {
    .is_open = loom_serve_text_http_open,
    .is_blocked = loom_serve_text_http_blocked,
    .begin = loom_serve_text_http_begin,
    .append = loom_serve_text_http_append,
    .complete = loom_serve_text_http_complete,
    .fail = loom_serve_text_http_fail,
    .publish = loom_serve_text_http_publish,
    .reject = loom_serve_text_http_reject_consumer,
    .end = loom_serve_text_http_end,
};

iree_status_t loom_serve_text_http_consumer_allocate(
    loom_serve_http_connection_t* connection,
    loom_serve_text_telemetry_t* telemetry, loom_serve_text_receipt_t receipt,
    iree_allocator_t host_allocator, loom_serve_text_consumer_t* out_consumer) {
  *out_consumer = (loom_serve_text_consumer_t){0};
  loom_serve_text_http_consumer_t* consumer = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator, sizeof(*consumer),
                                             (void**)&consumer));
  consumer->host_allocator = host_allocator;
  consumer->connection = connection;
  consumer->telemetry = telemetry;
  consumer->receipt = receipt;
  iree_string_builder_initialize(host_allocator, &consumer->packet);
  *out_consumer = (loom_serve_text_consumer_t){
      .self = consumer,
      .vtable = &loom_serve_text_http_vtable,
  };
  return iree_ok_status();
}
