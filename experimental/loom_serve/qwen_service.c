// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/loom_serve/qwen_service.h"

#include <stdio.h>
#include <string.h>

#include "experimental/loom_serve/qwen_chat.h"
#include "loom/util/json.h"

typedef enum qwen_request_phase_e {
  QWEN_REQUEST_PREFILL,
  QWEN_REQUEST_DECODE,
  QWEN_REQUEST_FINISHING,
} qwen_request_phase_t;

typedef struct qwen_session_t {
  // Borrowed private device row in the shared model residency.
  loom_serve_qwen_row_t* row;
  // Local client cache key; empty means an untagged request.
  char name[65];
  // Last admitted request serial, also the idle-row LRU ordering key.
  uint64_t serial;
  // Canonical completed client history validating the retained raw state.
  iree_string_builder_t checkpoint;
  // Preallocated streaming decoder storage reused between requests.
  iree_byte_span_t decoder_storage;
  // Active decoder within decoder_storage, or null between requests.
  iree_tokenizer_decode_state_t* decoder;
  // Fixed host token storage sized to the compiled context.
  int32_t* tokens;
  // Accumulated decoded response, preserving the original model text.
  iree_string_builder_t response;
  // Copied SSE bytes waiting for one carrier credit; no unbounded send queue.
  iree_string_builder_t packet;
  // Active request and its borrowed HTTP storage. A null connection means idle.
  struct {
    // Application claim, relinquished exactly once by finish or abort.
    loom_serve_http_connection_t* connection;
    // Validated chat borrowing the claimed request body.
    loom_serve_qwen_chat_t chat;
    // Next completed-stage action or terminal output drain.
    qwen_request_phase_t phase;
    // Host input tokens prepared for this turn.
    iree_host_size_t input_count;
    // Input tokens already consumed by prefill.
    iree_host_size_t input_offset;
    // Retained device tokens preceding this request's appended input.
    iree_host_size_t retained_count;
    // Selected output tokens, including EOS.
    iree_host_size_t output_count;
    // Ordinary response bytes already incorporated in copied SSE output.
    iree_host_size_t emitted_length;
    // Terminal wire reason, or "error" for failed output translation.
    const char* finish_reason;
    // Application admission time; network framing is not included.
    iree_time_t start_time;
    // First selected token time, including a possible immediate EOS.
    iree_time_t first_token_time;
    // Cumulative model counters at admission after any reset.
    loom_serve_qwen_metrics_t initial_metrics;
  } request;
} qwen_session_t;

typedef struct qwen_service_t {
  // Allocator for the service's host-only storage.
  iree_allocator_t allocator;
  // Borrowed model residency and its single host execution owner.
  loom_serve_qwen_model_t* model;
  // Borrowed transport with an independent network progress thread.
  loom_serve_http_server_t* server;
  // Number of allocated device/session rows.
  iree_host_size_t row_count;
  // Active input tokens issued in one scheduling turn.
  iree_host_size_t chunk_size;
  // Compiled context capacity of each retained row.
  iree_host_size_t context_capacity;
  // Output bound when the client omits one.
  iree_host_size_t default_max_tokens;
  // Monotonic request identity, independent of TCP connection identity.
  uint64_t next_serial;
  // Shared cold input rendering scratch, never borrowed by device work.
  iree_string_builder_t input_text;
  // Shared JSON/SSE serialization scratch, copied into a row packet or carrier.
  iree_string_builder_t scratch;
  // Shared typed tool-call serialization scratch used at generation end.
  iree_string_builder_t tool_calls;
  // Fixed retained rows sharing one model, command set and execution timeline.
  qwen_session_t sessions[8];
} qwen_service_t;

static void qwen_diagnose(const char* operation, iree_status_t status) {
  fprintf(stderr, "%s: ", operation);
  iree_status_fprint(stderr, status);
  iree_status_free(status);
}

static iree_status_t qwen_quote(iree_string_builder_t* output,
                                iree_string_view_t value) {
  loom_output_stream_t stream;
  loom_output_stream_for_builder(output, &stream);
  return loom_json_write_escaped_string(&stream, value);
}

// Error translation is an intentional status ownership boundary. Original
// failures are diagnosed once; failure to construct the response propagates.
static iree_status_t qwen_error_json(iree_status_t failure,
                                     iree_string_builder_t* output) {
  iree_status_t status =
      iree_string_builder_append_cstring(output, "{\"error\":{\"message\":");
  if (iree_status_is_ok(status)) {
    iree_string_view_t message = iree_status_message(failure);
    if (!message.size) {
      message = iree_make_cstring_view(
          iree_status_code_string(iree_status_code(failure)));
    }
    status = qwen_quote(output, message);
  }
  if (iree_status_is_ok(status)) {
    status = iree_string_builder_append_cstring(
        output, ",\"type\":\"loom_request_error\"}}");
  }
  qwen_diagnose("Chat request failed", failure);
  return status;
}

static iree_status_t qwen_reject(qwen_service_t* service,
                                 loom_serve_http_connection_t* connection,
                                 int code, const char* reason,
                                 iree_status_t failure) {
  iree_string_builder_reset(&service->scratch);
  iree_status_t status = iree_string_builder_append_format(
      &service->scratch,
      "HTTP/1.1 %d %s\r\nContent-Type: application/json\r\nConnection: "
      "close\r\n\r\n",
      code, reason);
  if (iree_status_is_ok(status)) {
    status = qwen_error_json(failure, &service->scratch);
  } else {
    status = iree_status_join(status, failure);
  }
  if (iree_status_is_ok(status)) {
    iree_status_t send_status = loom_serve_http_connection_send(
        connection, iree_string_builder_view(&service->scratch));
    if (iree_status_is_ok(send_status)) {
      loom_serve_http_connection_finish(connection);
    } else {
      qwen_diagnose("Chat rejection peer failed", send_status);
      loom_serve_http_connection_abort(connection);
    }
  } else {
    loom_serve_http_connection_abort(connection);
  }
  return status;
}

static void qwen_request_release(qwen_session_t* session) {
  loom_serve_qwen_chat_deinitialize(&session->request.chat);
  iree_tokenizer_decode_state_deinitialize(session->decoder);
  session->decoder = NULL;
  memset(&session->request, 0, sizeof(session->request));
  iree_string_builder_reset(&session->packet);
}

static void qwen_request_cancel(qwen_session_t* session) {
  fprintf(stderr,
          "{\"event\":\"cancel\",\"request\":%" PRIu64
          ",\"session\":\"%s\",\"position\":%zu}\n",
          session->serial, session->name,
          loom_serve_qwen_row_position(session->row));
  loom_serve_http_connection_abort(session->request.connection);
  iree_string_builder_reset(&session->checkpoint);
  qwen_request_release(session);
}

static void qwen_request_finish(qwen_session_t* session) {
  const loom_serve_qwen_metrics_t metrics =
      loom_serve_qwen_row_metrics(session->row);
  const loom_serve_qwen_metrics_t initial = session->request.initial_metrics;
  fprintf(
      stderr,
      "{\"event\":\"complete\",\"request\":%" PRIu64
      ",\"session\":\"%s\","
      "\"finish_reason\":\"%s\",\"retained_tokens\":%zu,\"appended_tokens\":%"
      "zu,"
      "\"output_tokens_including_eos\":%zu,\"position\":%zu,\"prefill_steps\":"
      "%" PRIu64
      ","
      "\"prefill_ms\":%.3f,\"decode_steps\":%" PRIu64
      ",\"decode_ms\":%.3f,"
      "\"model_ttft_ms\":%.3f,\"request_ms\":%.3f}\n",
      session->serial, session->name, session->request.finish_reason,
      session->request.retained_count, session->request.input_count,
      session->request.output_count, loom_serve_qwen_row_position(session->row),
      metrics.prefill_steps - initial.prefill_steps,
      (metrics.prefill_duration - initial.prefill_duration) / 1e6,
      metrics.decode_steps - initial.decode_steps,
      (metrics.decode_duration - initial.decode_duration) / 1e6,
      (session->request.first_token_time - session->request.start_time) / 1e6,
      (iree_time_now() - session->request.start_time) / 1e6);
  loom_serve_http_connection_finish(session->request.connection);
  qwen_request_release(session);
}

static bool qwen_session_name_valid(iree_string_view_t name) {
  if (name.size > 64) {
    return false;
  }
  for (iree_host_size_t i = 0; i < name.size; ++i) {
    const char c = name.data[i];
    if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
          (c >= '0' && c <= '9') || c == '_' || c == '-' || c == '.')) {
      return false;
    }
  }
  return true;
}

static qwen_session_t* qwen_session_select(qwen_service_t* service,
                                           iree_string_view_t name) {
  qwen_session_t* oldest = NULL;
  for (iree_host_size_t i = 0; i < service->row_count; ++i) {
    qwen_session_t* session = &service->sessions[i];
    if (name.size &&
        iree_string_view_equal(name, iree_make_cstring_view(session->name))) {
      return session;
    }
    if (!session->request.connection &&
        (!oldest || session->serial < oldest->serial)) {
      oldest = session;
    }
  }
  return oldest;
}

// Prepares host input without changing retained device state. Rejection leaves
// the previous checkpoint usable. The selected token belongs to the previous
// response but has not yet entered KV/GDN, so it leads a retained append.
static iree_status_t qwen_prepare_input(qwen_service_t* service,
                                        qwen_session_t* session,
                                        const loom_serve_qwen_chat_t* chat,
                                        iree_string_view_t name,
                                        iree_host_size_t* out_count,
                                        iree_host_size_t* out_retained) {
  const iree_string_view_t prompt = iree_string_builder_view(&chat->prompt);
  const iree_string_view_t checkpoint =
      iree_string_builder_view(&session->checkpoint);
  const bool retained =
      name.size && checkpoint.size &&
      iree_string_view_equal(name, iree_make_cstring_view(session->name)) &&
      iree_string_view_starts_with(prompt, checkpoint);
  *out_retained = retained ? loom_serve_qwen_row_position(session->row) : 0;
  iree_string_builder_reset(&service->input_text);
  iree_string_view_t input = prompt;
  if (retained) {
    session->tokens[0] = loom_serve_qwen_row_token(session->row);
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(
        &service->input_text,
        loom_serve_qwen_row_is_eos(session->row) ? "\n" : "<|im_end|>\n"));
    IREE_RETURN_IF_ERROR(iree_string_builder_append_string(
        &service->input_text,
        iree_string_view_substr(prompt, checkpoint.size, IREE_HOST_SIZE_MAX)));
    input = iree_string_builder_view(&service->input_text);
  }
  const iree_host_size_t prefix_count = retained ? 1 : 0;
  IREE_RETURN_IF_ERROR(
      iree_tokenizer_encode(loom_serve_qwen_model_tokenizer(service->model),
                            input, IREE_TOKENIZER_ENCODE_FLAG_NONE,
                            iree_tokenizer_make_token_output(
                                session->tokens + prefix_count, NULL, NULL,
                                service->context_capacity - prefix_count),
                            service->allocator, out_count));
  *out_count += prefix_count;
  if (*out_count + chat->max_tokens - 1 >
      service->context_capacity - *out_retained) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "request input and output exceed remaining %zu-token context; compact "
        "history or lower max_tokens",
        service->context_capacity);
  }
  return iree_ok_status();
}

static iree_status_t qwen_admit(qwen_service_t* service,
                                loom_serve_http_connection_t* connection,
                                const loom_serve_http_request_t* request) {
  if (iree_string_view_equal(request->method, IREE_SV("GET")) &&
      iree_string_view_equal(request->target, IREE_SV("/healthz"))) {
    iree_status_t status = loom_serve_http_connection_send(
        connection,
        IREE_SV(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nConnection: "
            "close\r\n\r\n{\"status\":\"ready\"}"));
    if (iree_status_is_ok(status)) {
      loom_serve_http_connection_finish(connection);
    } else {
      qwen_diagnose("Health peer failed", status);
      loom_serve_http_connection_abort(connection);
    }
    return iree_ok_status();
  }
  if (!iree_string_view_equal(request->method, IREE_SV("POST")) ||
      !iree_string_view_equal(request->target,
                              IREE_SV("/v1/chat/completions"))) {
    return qwen_reject(
        service, connection, 404, "Not Found",
        iree_make_status(IREE_STATUS_NOT_FOUND,
                         "use POST /v1/chat/completions or GET /healthz"));
  }
  iree_string_view_t content_type;
  if (!loom_serve_http_request_lookup_header(request, IREE_SV("Content-Type"),
                                             &content_type) ||
      !(iree_string_view_equal_case(content_type,
                                    IREE_SV("application/json")) ||
        iree_string_view_equal_case(
            content_type, IREE_SV("application/json; charset=utf-8")))) {
    return qwen_reject(
        service, connection, 415, "Unsupported Media Type",
        iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                         "Content-Type must be application/json"));
  }
  iree_string_view_t name;
  loom_serve_http_request_lookup_header(request, IREE_SV("X-Loom-Session"),
                                        &name);
  if (!qwen_session_name_valid(name)) {
    return qwen_reject(
        service, connection, 400, "Bad Request",
        iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "X-Loom-Session must be at most 64 ASCII identifier characters"));
  }
  qwen_session_t* session = qwen_session_select(service, name);
  if (!session || session->request.connection) {
    return qwen_reject(
        service, connection, session ? 409 : 503,
        session ? "Conflict" : "Service Unavailable",
        iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                         session ? "session already has an active request"
                                 : "all model rows are active"));
  }
  loom_serve_qwen_chat_t chat;
  iree_status_t status = loom_serve_qwen_chat_initialize(
      request->body, service->default_max_tokens, service->allocator, &chat);
  if (!iree_status_is_ok(status)) {
    return qwen_reject(service, connection, 400, "Bad Request", status);
  }
  iree_host_size_t input_count = 0;
  iree_host_size_t retained_count = 0;
  status = qwen_prepare_input(service, session, &chat, name, &input_count,
                              &retained_count);
  if (!iree_status_is_ok(status)) {
    loom_serve_qwen_chat_deinitialize(&chat);
    return qwen_reject(service, connection, 400, "Bad Request", status);
  }
  if (session->name[0] &&
      !iree_string_view_equal(name, iree_make_cstring_view(session->name))) {
    fprintf(stderr,
            "{\"event\":\"evict\",\"session\":\"%s\",\"position\":%zu}\n",
            session->name, loom_serve_qwen_row_position(session->row));
  }
  if (name.size) {
    memcpy(session->name, name.data, name.size);
  }
  session->name[name.size] = 0;
  session->serial = ++service->next_serial;
  session->request.connection = connection;
  session->request.chat = chat;
  session->request.phase = QWEN_REQUEST_PREFILL;
  session->request.input_count = input_count;
  session->request.retained_count = retained_count;
  session->request.start_time = iree_time_now();
  iree_string_builder_reset(&session->response);
  iree_string_builder_reset(&session->checkpoint);
  if (!retained_count) {
    status = loom_serve_qwen_row_reset(session->row);
  }
  session->request.initial_metrics = loom_serve_qwen_row_metrics(session->row);
  if (iree_status_is_ok(status)) {
    status = iree_tokenizer_decode_state_initialize(
        loom_serve_qwen_model_tokenizer(service->model),
        IREE_TOKENIZER_DECODE_FLAG_SKIP_SPECIAL_TOKENS,
        session->decoder_storage, &session->decoder);
  }
  if (iree_status_is_ok(status)) {
    status = iree_string_builder_append_cstring(
        &session->packet,
        "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: "
        "no-cache\r\nConnection: close\r\n\r\n");
  }
  if (iree_status_is_ok(status)) {
    status = loom_serve_qwen_chat_event(
        session->serial, IREE_SV("{\"role\":\"assistant\"}"),
        iree_string_view_empty(), &session->packet);
  }
  if (iree_status_is_ok(status)) {
    fprintf(stderr,
            "{\"event\":\"admit\",\"request\":%" PRIu64
            ",\"session\":\"%s\","
            "\"row\":%zu,\"cache\":\"%s\",\"retained_tokens\":%zu,\"appended_"
            "tokens\":%zu,\"max_tokens\":%zu}\n",
            session->serial, session->name,
            (iree_host_size_t)(session - service->sessions),
            retained_count ? "hit" : "replay", retained_count, input_count,
            chat.max_tokens);
  }
  return status;
}

static iree_status_t qwen_emit_text(qwen_service_t* service,
                                    qwen_session_t* session,
                                    iree_host_size_t end) {
  const iree_host_size_t start = session->request.emitted_length;
  if (start == end) {
    return iree_ok_status();
  }
  iree_string_builder_reset(&service->scratch);
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(&service->scratch, "{\"content\":"));
  IREE_RETURN_IF_ERROR(qwen_quote(
      &service->scratch,
      iree_string_view_substr(iree_string_builder_view(&session->response),
                              start, end - start)));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(&service->scratch, "}"));
  IREE_RETURN_IF_ERROR(loom_serve_qwen_chat_event(
      session->serial, iree_string_builder_view(&service->scratch),
      iree_string_view_empty(), &session->packet));
  session->request.emitted_length = end;
  return iree_ok_status();
}

static iree_status_t qwen_append_response(qwen_session_t* session,
                                          iree_string_view_t text) {
  if (text.size > 1024 * 1024 - iree_string_builder_size(&session->response)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "generated response exceeds 1 MiB text limit");
  }
  return iree_string_builder_append_string(&session->response, text);
}

static iree_status_t qwen_complete(qwen_service_t* service,
                                   qwen_session_t* session) {
  char text[8192];
  iree_host_size_t length = 0;
  IREE_RETURN_IF_ERROR(iree_tokenizer_decode_state_finalize(
      session->decoder, iree_make_mutable_string_view(text, sizeof(text)),
      &length));
  IREE_RETURN_IF_ERROR(
      qwen_append_response(session, iree_make_string_view(text, length)));
  iree_string_builder_reset(&service->tool_calls);
  iree_host_size_t tool_count = 0;
  IREE_RETURN_IF_ERROR(loom_serve_qwen_chat_complete(
      &session->request.chat, iree_string_builder_view(&session->response),
      session->serial, &session->checkpoint, &service->tool_calls,
      &tool_count));
  const iree_string_view_t response =
      iree_string_builder_view(&session->response);
  const iree_host_size_t text_end =
      tool_count ? iree_string_view_find(response, IREE_SV("<tool_call>"),
                                         session->request.emitted_length)
                 : response.size;
  IREE_RETURN_IF_ERROR(qwen_emit_text(service, session, text_end));
  if (tool_count) {
    iree_string_builder_reset(&service->scratch);
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        &service->scratch, "{\"tool_calls\":%.*s}",
        (int)iree_string_builder_size(&service->tool_calls),
        iree_string_builder_buffer(&service->tool_calls)));
    IREE_RETURN_IF_ERROR(loom_serve_qwen_chat_event(
        session->serial, iree_string_builder_view(&service->scratch),
        iree_string_view_empty(), &session->packet));
  }
  session->request.finish_reason = tool_count ? "tool_calls"
                                   : loom_serve_qwen_row_is_eos(session->row)
                                       ? "stop"
                                       : "length";
  IREE_RETURN_IF_ERROR(loom_serve_qwen_chat_event(
      session->serial, IREE_SV("{}"),
      iree_make_cstring_view(session->request.finish_reason),
      &session->packet));
  if (session->request.chat.include_usage) {
    IREE_RETURN_IF_ERROR(loom_serve_qwen_chat_usage(
        session->serial,
        session->request.retained_count + session->request.input_count,
        session->request.retained_count, session->request.output_count,
        &session->packet));
  }
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(&session->packet, "data: [DONE]\n\n"));
  session->request.phase = QWEN_REQUEST_FINISHING;
  return iree_ok_status();
}

static iree_status_t qwen_selected_token(qwen_service_t* service,
                                         qwen_session_t* session) {
  if (!session->request.output_count) {
    session->request.first_token_time = iree_time_now();
  }
  ++session->request.output_count;
  const int32_t token = loom_serve_qwen_row_token(session->row);
  iree_host_size_t consumed = 0;
  iree_status_t status = iree_ok_status();
  while (!consumed && iree_status_is_ok(status)) {
    char text[8192];
    iree_host_size_t length = 0;
    status = iree_tokenizer_decode_state_feed(
        session->decoder, iree_tokenizer_make_token_id_list(&token, 1),
        iree_make_mutable_string_view(text, sizeof(text)), &consumed, &length);
    if (iree_status_is_ok(status) && !consumed && !length) {
      status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "token exceeds streaming text capacity");
    }
    if (iree_status_is_ok(status)) {
      status =
          qwen_append_response(session, iree_make_string_view(text, length));
    }
  }
  if (iree_status_is_ok(status)) {
    const bool complete =
        loom_serve_qwen_row_is_eos(session->row) ||
        session->request.output_count == session->request.chat.max_tokens;
    status = complete ? qwen_complete(service, session)
                      : qwen_emit_text(
                            service, session,
                            loom_serve_qwen_chat_text_end(
                                iree_string_builder_view(&session->response),
                                session->request.emitted_length));
  }
  if (!iree_status_is_ok(status)) {
    // Device work completed, but translation cannot establish a client history.
    // Preserve prior streamed text and terminate with a real SSE error.
    iree_string_builder_reset(&session->checkpoint);
    iree_string_builder_reset(&session->packet);
    iree_status_t packet_status =
        iree_string_builder_append_cstring(&session->packet, "data: ");
    if (iree_status_is_ok(packet_status)) {
      packet_status = qwen_error_json(status, &session->packet);
    } else {
      packet_status = iree_status_join(packet_status, status);
    }
    if (iree_status_is_ok(packet_status)) {
      packet_status = iree_string_builder_append_cstring(
          &session->packet, "\n\ndata: [DONE]\n\n");
    }
    session->request.finish_reason = "error";
    session->request.phase = QWEN_REQUEST_FINISHING;
    status = packet_status;
  }
  return status;
}

static iree_status_t qwen_advance(qwen_service_t* service,
                                  qwen_session_t* session, bool* out_progress) {
  if (!session->request.connection) {
    return iree_ok_status();
  }
  if (loom_serve_http_connection_failed(session->request.connection)) {
    qwen_request_cancel(session);
    *out_progress = true;
    return iree_ok_status();
  }
  if (!loom_serve_http_connection_can_send(session->request.connection)) {
    return iree_ok_status();
  }
  *out_progress = true;
  if (iree_string_builder_size(&session->packet)) {
    iree_status_t status = loom_serve_http_connection_send(
        session->request.connection,
        iree_string_builder_view(&session->packet));
    if (!iree_status_is_ok(status)) {
      qwen_diagnose("Chat stream peer failed", status);
      qwen_request_cancel(session);
      return iree_ok_status();
    }
    iree_string_builder_reset(&session->packet);
    if (session->request.phase == QWEN_REQUEST_FINISHING) {
      qwen_request_finish(session);
    }
    return iree_ok_status();
  }
  if (session->request.phase == QWEN_REQUEST_PREFILL) {
    const iree_host_size_t count =
        iree_min(service->chunk_size,
                 session->request.input_count - session->request.input_offset);
    IREE_RETURN_IF_ERROR(loom_serve_qwen_row_prefill(
        session->row, count, session->tokens + session->request.input_offset));
    session->request.input_offset += count;
    if (session->request.input_offset != session->request.input_count) {
      return iree_ok_status();
    }
    session->request.phase = QWEN_REQUEST_DECODE;
  } else {
    IREE_RETURN_IF_ERROR(loom_serve_qwen_row_decode(session->row));
  }
  return qwen_selected_token(service, session);
}

static iree_status_t qwen_service_initialize(qwen_service_t* service) {
  iree_host_size_t decoder_size = 0;
  IREE_RETURN_IF_ERROR(iree_tokenizer_decode_state_calculate_size(
      loom_serve_qwen_model_tokenizer(service->model), &decoder_size));
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < service->row_count && iree_status_is_ok(status); ++i) {
    qwen_session_t* session = &service->sessions[i];
    session->row = loom_serve_qwen_model_row(service->model, i);
    status = iree_allocator_malloc_array(
        service->allocator, service->context_capacity, sizeof(*session->tokens),
        (void**)&session->tokens);
    if (iree_status_is_ok(status)) {
      status = iree_allocator_malloc(service->allocator, decoder_size,
                                     (void**)&session->decoder_storage.data);
      session->decoder_storage.data_length = decoder_size;
    }
    if (iree_status_is_ok(status)) {
      status = iree_string_builder_reserve(&session->response, 65536);
    }
    if (iree_status_is_ok(status)) {
      status = iree_string_builder_reserve(&session->packet, 16384);
    }
  }
  return status;
}

iree_status_t loom_serve_qwen_service_run(loom_serve_qwen_model_t* model,
                                          loom_serve_http_server_t* server,
                                          iree_host_size_t row_count,
                                          iree_host_size_t chunk_size,
                                          iree_host_size_t default_max_tokens,
                                          iree_allocator_t host_allocator) {
  qwen_service_t service = {
      .allocator = host_allocator,
      .model = model,
      .server = server,
      .row_count = row_count,
      .chunk_size = chunk_size,
      .context_capacity = loom_serve_qwen_model_context_capacity(model),
      .default_max_tokens = default_max_tokens};
  iree_string_builder_initialize(host_allocator, &service.input_text);
  iree_string_builder_initialize(host_allocator, &service.scratch);
  iree_string_builder_initialize(host_allocator, &service.tool_calls);
  for (iree_host_size_t i = 0; i < row_count; ++i) {
    iree_string_builder_initialize(host_allocator,
                                   &service.sessions[i].checkpoint);
    iree_string_builder_initialize(host_allocator,
                                   &service.sessions[i].response);
    iree_string_builder_initialize(host_allocator, &service.sessions[i].packet);
  }
  iree_status_t status = qwen_service_initialize(&service);
  iree_notification_t* notification =
      loom_serve_http_server_notification(server);
  while (iree_status_is_ok(status) &&
         !loom_serve_http_server_is_stopping(server)) {
    const iree_wait_token_t token =
        iree_notification_prepare_wait(notification);
    bool progress = false;
    // Admit at most one request per cycle so new arrivals cannot starve rows.
    const loom_serve_http_request_t* request = NULL;
    loom_serve_http_connection_t* connection =
        loom_serve_http_server_take_request(server, &request);
    if (connection) {
      status = qwen_admit(&service, connection, request);
      progress = true;
    }
    for (iree_host_size_t i = 0; i < row_count && iree_status_is_ok(status) &&
                                 !loom_serve_http_server_is_stopping(server);
         ++i) {
      status = qwen_advance(&service, &service.sessions[i], &progress);
    }
    if (progress || !iree_status_is_ok(status) ||
        loom_serve_http_server_is_stopping(server)) {
      iree_notification_cancel_wait(notification);
    } else {
      iree_notification_commit_wait(notification, token, IREE_DURATION_ZERO,
                                    IREE_TIME_INFINITE_FUTURE);
    }
  }
  for (iree_host_size_t i = 0; i < row_count; ++i) {
    qwen_session_t* session = &service.sessions[i];
    if (session->request.connection) {
      qwen_request_cancel(session);
    }
    iree_allocator_free(host_allocator, session->decoder_storage.data);
    iree_allocator_free(host_allocator, session->tokens);
    iree_string_builder_deinitialize(&session->packet);
    iree_string_builder_deinitialize(&session->response);
    iree_string_builder_deinitialize(&session->checkpoint);
  }
  iree_string_builder_deinitialize(&service.tool_calls);
  iree_string_builder_deinitialize(&service.scratch);
  iree_string_builder_deinitialize(&service.input_text);
  return status;
}
