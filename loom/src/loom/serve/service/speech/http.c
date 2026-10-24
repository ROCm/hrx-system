// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/service/speech/http.h"

#include "loom/serve/service/speech/output.h"
#include "loom/util/json.h"

static const char loom_serve_speech_http_headers[] =
    "HTTP/1.1 200 OK\r\nContent-Type: audio/pcm;rate=24000;channels=1\r\n"
    "Cache-Control: no-store\r\nConnection: close\r\n\r\n";

typedef struct loom_serve_speech_http_consumer_t {
  // Allocator owning this receipt and its bounded encoded output.
  iree_allocator_t host_allocator;
  // Owner-only claim; cancel relinquishes it without destroying the receipt.
  loom_serve_http_connection_t* connection;
  // One-time registration produces a JSON receipt only after source release.
  bool voice_upload;
  // First response publication has committed the HTTP headers. CPU encoding
  // only reads this between publication turns; cancellation never changes it.
  bool header_sent;
  // CPU-written response, borrowed only on owner publication after task
  // release.
  iree_string_builder_t response;
} loom_serve_speech_http_consumer_t;

iree_status_t loom_serve_speech_http_reject(
    loom_serve_http_connection_t* connection, const char* code,
    iree_string_view_t message, iree_allocator_t host_allocator) {
  iree_string_builder_t response;
  iree_string_builder_initialize(host_allocator, &response);
  iree_status_t status = iree_string_builder_append_format(
      &response,
      "HTTP/1.1 %s\r\nContent-Type: application/json\r\nConnection: "
      "close\r\n\r\n"
      "{\"error\":{\"type\":\"speech_request_error\",\"message\":",
      code);
  if (iree_status_is_ok(status)) {
    loom_output_stream_t output;
    loom_output_stream_for_builder(&response, &output);
    status = loom_json_write_escaped_string(&output, message);
  }
  if (iree_status_is_ok(status)) {
    status = iree_string_builder_append_cstring(&response, "}}");
  }
  if (iree_status_is_ok(status) &&
      !loom_serve_http_connection_failed(connection) &&
      loom_serve_http_connection_can_send(connection)) {
    status = loom_serve_http_connection_send(
        connection, iree_string_builder_view(&response));
    if (iree_status_is_ok(status)) {
      loom_serve_http_connection_finish(connection);
      connection = NULL;
    } else if (iree_status_is_cancelled(status) ||
               iree_status_is_unavailable(status)) {
      iree_status_free(status);
      status = iree_ok_status();
    }
  }
  if (connection) {
    loom_serve_http_connection_abort(connection);
  }
  iree_string_builder_deinitialize(&response);
  return status;
}

static bool loom_serve_speech_http_open(void* self) {
  const loom_serve_speech_http_consumer_t* consumer = self;
  return consumer->connection &&
         !loom_serve_http_connection_failed(consumer->connection);
}

static bool loom_serve_speech_http_credit(void* self) {
  const loom_serve_speech_http_consumer_t* consumer = self;
  return consumer->connection &&
         loom_serve_http_connection_can_send(consumer->connection);
}

static iree_status_t loom_serve_speech_http_encode(
    void* self, const loom_serve_speech_block_t* block) {
  loom_serve_speech_http_consumer_t* consumer = self;
  if (consumer->voice_upload) {
    return iree_ok_status();
  }
  iree_string_builder_reset(&consumer->response);
  if (!consumer->header_sent && (block->frame_count || block->complete)) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(
        &consumer->response, loom_serve_speech_http_headers));
  }
  return loom_serve_speech_encode_pcm16(block->samples, &consumer->response);
}

static iree_status_t loom_serve_speech_http_publish(void* self,
                                                    bool* out_published) {
  loom_serve_speech_http_consumer_t* consumer = self;
  *out_published = false;
  if (consumer->voice_upload) {
    *out_published = true;
    return iree_ok_status();
  }
  const iree_string_view_t bytes =
      iree_string_builder_view(&consumer->response);
  if (bytes.size) {
    if (!loom_serve_http_connection_can_send(consumer->connection)) {
      return iree_ok_status();
    }
    IREE_RETURN_IF_ERROR(
        loom_serve_http_connection_send(consumer->connection, bytes));
    consumer->header_sent = true;
  }
  iree_string_builder_reset(&consumer->response);
  *out_published = true;
  return iree_ok_status();
}

static void loom_serve_speech_http_cancel(void* self) {
  loom_serve_speech_http_consumer_t* consumer = self;
  if (consumer->connection) {
    loom_serve_http_connection_abort(consumer->connection);
    consumer->connection = NULL;
  }
}

static iree_status_t loom_serve_speech_http_end(
    void* self, loom_serve_telemetry_outcome_t outcome,
    const iree_status_t request_status) {
  loom_serve_speech_http_consumer_t* consumer = self;
  iree_status_t status = iree_ok_status();
  if (consumer->connection) {
    if (outcome == LOOM_SERVE_TELEMETRY_COMPLETED) {
      if (consumer->voice_upload) {
        status = loom_serve_http_connection_send(
            consumer->connection,
            iree_string_builder_view(&consumer->response));
      }
      if (iree_status_is_ok(status)) {
        loom_serve_http_connection_finish(consumer->connection);
      } else {
        loom_serve_http_connection_abort(consumer->connection);
        if (iree_status_is_cancelled(status) ||
            iree_status_is_unavailable(status)) {
          iree_status_free(status);
          status = iree_ok_status();
        }
      }
    } else if ((outcome == LOOM_SERVE_TELEMETRY_REJECTED ||
                outcome == LOOM_SERVE_TELEMETRY_FAILED) &&
               !consumer->header_sent) {
      iree_string_view_t message = iree_status_message(request_status);
      if (!message.size) {
        message = iree_make_cstring_view(
            iree_status_code_string(iree_status_code(request_status)));
      }
      status = loom_serve_speech_http_reject(
          consumer->connection,
          iree_status_is_resource_exhausted(request_status)
              ? "503 Service Unavailable"
          : outcome == LOOM_SERVE_TELEMETRY_REJECTED
              ? "400 Bad Request"
              : "500 Internal Server Error",
          message, consumer->host_allocator);
    } else {
      loom_serve_http_connection_abort(consumer->connection);
    }
  }
  iree_string_builder_deinitialize(&consumer->response);
  iree_allocator_free(consumer->host_allocator, consumer);
  return status;
}

iree_status_t loom_serve_speech_http_consumer_allocate(
    loom_serve_http_connection_t* connection, loom_serve_speech_format_t format,
    iree_host_size_t block_capacity, iree_allocator_t host_allocator,
    loom_serve_speech_consumer_t* out_consumer) {
  *out_consumer = (loom_serve_speech_consumer_t){0};
  if (format.sample_rate != 24000 || format.channels != 1) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "PCM speech responses require 24000 Hz mono output");
  }
  if (block_capacity > (SIZE_MAX - sizeof(loom_serve_speech_http_consumer_t) -
                        sizeof(loom_serve_speech_http_headers)) /
                           sizeof(int16_t)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "HTTP speech output allocation is too large");
  }
  const iree_host_size_t capacity =
      block_capacity * sizeof(int16_t) + sizeof(loom_serve_speech_http_headers);
  loom_serve_speech_http_consumer_t* consumer = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      host_allocator, sizeof(*consumer) + capacity, (void**)&consumer));
  consumer->host_allocator = host_allocator;
  iree_string_builder_initialize_with_storage((char*)(consumer + 1), capacity,
                                              &consumer->response);
  consumer->connection = connection;
  *out_consumer = (loom_serve_speech_consumer_t){
      .self = consumer,
      .is_open = loom_serve_speech_http_open,
      .can_write = loom_serve_speech_http_credit,
      .encode = loom_serve_speech_http_encode,
      .publish = loom_serve_speech_http_publish,
      .cancel = loom_serve_speech_http_cancel,
      .end = loom_serve_speech_http_end,
  };
  return iree_ok_status();
}

iree_status_t loom_serve_speech_http_voice_consumer_allocate(
    loom_serve_http_connection_t* connection, iree_string_view_t id,
    iree_allocator_t host_allocator,
    loom_serve_speech_consumer_t* out_consumer) {
  *out_consumer = (loom_serve_speech_consumer_t){0};
  loom_serve_speech_http_consumer_t* consumer = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator, sizeof(*consumer),
                                             (void**)&consumer));
  consumer->host_allocator = host_allocator;
  consumer->voice_upload = true;
  iree_string_builder_initialize(host_allocator, &consumer->response);
  iree_status_t status = iree_string_builder_append_cstring(
      &consumer->response,
      "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
      "Cache-Control: no-store\r\nConnection: close\r\n\r\n{\"id\":");
  if (iree_status_is_ok(status)) {
    loom_output_stream_t output;
    loom_output_stream_for_builder(&consumer->response, &output);
    status = loom_json_write_escaped_string(&output, id);
  }
  if (iree_status_is_ok(status)) {
    status = iree_string_builder_append_cstring(&consumer->response, "}");
  }
  if (iree_status_is_ok(status)) {
    consumer->connection = connection;
    *out_consumer = (loom_serve_speech_consumer_t){
        .self = consumer,
        .is_open = loom_serve_speech_http_open,
        .can_write = loom_serve_speech_http_credit,
        .encode = loom_serve_speech_http_encode,
        .publish = loom_serve_speech_http_publish,
        .cancel = loom_serve_speech_http_cancel,
        .end = loom_serve_speech_http_end,
    };
  } else {
    iree_string_builder_deinitialize(&consumer->response);
    iree_allocator_free(host_allocator, consumer);
  }
  return status;
}
