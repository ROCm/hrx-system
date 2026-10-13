// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/loom_serve/text/session.h"

#include <string.h>

#include "loom/util/json.h"

static iree_status_t text_quote(iree_string_builder_t* output,
                                iree_string_view_t value) {
  loom_output_stream_t stream;
  loom_output_stream_for_builder(output, &stream);
  return loom_json_write_escaped_string(&stream, value);
}

iree_status_t loom_serve_text_session_initialize(
    loom_serve_text_row_t* row, const iree_tokenizer_t* tokenizer,
    iree_host_size_t context_capacity, loom_serve_text_session_t* out_session,
    iree_allocator_t host_allocator) {
  *out_session = (loom_serve_text_session_t){
      .allocator = host_allocator,
      .row = row,
  };
  iree_string_builder_initialize(host_allocator, &out_session->response);
  iree_string_builder_initialize(host_allocator, &out_session->packet);
  iree_host_size_t decoder_size = 0;
  IREE_RETURN_IF_ERROR(
      iree_tokenizer_decode_state_calculate_size(tokenizer, &decoder_size));
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
      host_allocator, context_capacity, sizeof(*out_session->tokens),
      (void**)&out_session->tokens));
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, decoder_size,
                            (void**)&out_session->decoder_storage.data));
  out_session->decoder_storage.data_length = decoder_size;
  IREE_RETURN_IF_ERROR(
      iree_string_builder_reserve(&out_session->response, 65536));
  return iree_string_builder_reserve(&out_session->packet, 16384);
}

void loom_serve_text_session_release_request(
    loom_serve_text_session_t* session) {
  loom_serve_text_chat_deinitialize(&session->request.chat);
  iree_tokenizer_decode_state_deinitialize(session->decoder);
  session->decoder = NULL;
  memset(&session->request, 0, sizeof(session->request));
  iree_string_builder_reset(&session->packet);
}

void loom_serve_text_session_deinitialize(loom_serve_text_session_t* session) {
  if (!session->row) {
    return;
  }
  loom_serve_text_session_release_request(session);
  iree_allocator_free(session->allocator, session->decoder_storage.data);
  iree_allocator_free(session->allocator, session->tokens);
  iree_string_builder_deinitialize(&session->packet);
  iree_string_builder_deinitialize(&session->response);
  loom_serve_text_chat_completion_deinitialize(&session->completion);
  memset(session, 0, sizeof(*session));
}

loom_serve_http_request_snapshot_t loom_serve_text_session_request_snapshot(
    const loom_serve_text_session_t* session) {
  return (loom_serve_http_request_snapshot_t){
      .id = session->request.connection ? session->lifetime.serial : 0,
      .session = session->lifetime.id,
      .phase = "active",
      .arrival_time =
          session->request.start_time - session->request.queue_duration,
      .counters = session->request.counters};
}

static loom_serve_text_chat_response_t text_response_metadata(
    const loom_serve_text_session_t* session, iree_string_view_t model_name) {
  return (loom_serve_text_chat_response_t){
      .model_name = model_name,
      .request_id = session->lifetime.serial,
      .creation_time = session->request.start_time,
      .flags = session->request.chat.flags,
      .usage = {.input_tokens = session->request.retained_count +
                                session->request.input_count,
                .cached_tokens = session->request.retained_count,
                .output_tokens = session->request.output_count}};
}

iree_status_t loom_serve_text_session_begin(loom_serve_text_session_t* session,
                                            const iree_tokenizer_t* tokenizer,
                                            iree_string_view_t model_name) {
  iree_string_builder_reset(&session->response);
  loom_serve_text_chat_completion_deinitialize(&session->completion);
  IREE_RETURN_IF_ERROR(iree_tokenizer_decode_state_initialize(
      tokenizer, IREE_TOKENIZER_DECODE_FLAG_SKIP_SPECIAL_TOKENS,
      session->decoder_storage, &session->decoder));
  if (iree_any_bit_set(session->request.chat.flags,
                       LOOM_SERVE_TEXT_CHAT_FLAG_STREAM)) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(
        &session->packet,
        "HTTP/1.1 200 OK\r\nContent-Type: "
        "text/event-stream\r\nCache-Control: "
        "no-cache\r\nConnection: close\r\n\r\n"));
    const loom_serve_text_chat_response_t response =
        text_response_metadata(session, model_name);
    IREE_RETURN_IF_ERROR(loom_serve_text_chat_event(
        &response, IREE_SV("{\"role\":\"assistant\"}"),
        iree_string_view_empty(), &session->packet));
  }
  return iree_ok_status();
}

static iree_status_t text_emit_text(
    loom_serve_text_session_t* session,
    const loom_serve_text_chat_response_t* response, iree_host_size_t end,
    iree_string_builder_t* scratch) {
  const iree_host_size_t start = session->request.emitted_length;
  if (start == end) {
    return iree_ok_status();
  }
  iree_string_builder_reset(scratch);
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(scratch, "{\"content\":"));
  IREE_RETURN_IF_ERROR(text_quote(
      scratch,
      iree_string_view_substr(iree_string_builder_view(&session->response),
                              start, end - start)));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(scratch, "}"));
  IREE_RETURN_IF_ERROR(
      loom_serve_text_chat_event(response, iree_string_builder_view(scratch),
                                 iree_string_view_empty(), &session->packet));
  session->request.emitted_length = end;
  return iree_ok_status();
}

static iree_status_t text_append_response(loom_serve_text_session_t* session,
                                          iree_string_view_t text) {
  if (text.size > 1024 * 1024 - iree_string_builder_size(&session->response)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "generated response exceeds 1 MiB text limit");
  }
  return iree_string_builder_append_string(&session->response, text);
}

static iree_status_t text_complete(
    loom_serve_text_session_t* session,
    const loom_serve_text_chat_response_t* response,
    iree_string_builder_t* scratch, iree_string_builder_t* tool_calls) {
  char text[8192];
  iree_host_size_t length = 0;
  IREE_RETURN_IF_ERROR(iree_tokenizer_decode_state_finalize(
      session->decoder, iree_make_mutable_string_view(text, sizeof(text)),
      &length));
  IREE_RETURN_IF_ERROR(
      text_append_response(session, iree_make_string_view(text, length)));
  iree_string_builder_reset(tool_calls);
  IREE_RETURN_IF_ERROR(loom_serve_text_chat_complete(
      &session->request.chat, iree_string_builder_view(&session->response),
      session->request.emitted_length, session->lifetime.serial, tool_calls,
      &session->completion));
  const iree_host_size_t tool_count = session->completion.tool_count;
  session->request.finish_reason = tool_count ? "tool_calls"
                                   : loom_serve_text_row_is_eos(session->row)
                                       ? "stop"
                                       : "length";
  const iree_string_view_t finish_reason =
      iree_make_cstring_view(session->request.finish_reason);
  if (iree_any_bit_set(response->flags, LOOM_SERVE_TEXT_CHAT_FLAG_STREAM)) {
    IREE_RETURN_IF_ERROR(text_emit_text(session, response,
                                        session->completion.text_end, scratch));
    if (tool_count) {
      iree_string_builder_reset(scratch);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          scratch, "{\"tool_calls\":%.*s}",
          (int)iree_string_builder_size(tool_calls),
          iree_string_builder_buffer(tool_calls)));
      IREE_RETURN_IF_ERROR(loom_serve_text_chat_event(
          response, iree_string_builder_view(scratch), iree_string_view_empty(),
          &session->packet));
    }
    IREE_RETURN_IF_ERROR(loom_serve_text_chat_event(
        response, IREE_SV("{}"), finish_reason, &session->packet));
    if (iree_any_bit_set(response->flags,
                         LOOM_SERVE_TEXT_CHAT_FLAG_INCLUDE_USAGE)) {
      IREE_RETURN_IF_ERROR(
          loom_serve_text_chat_usage(response, &session->packet));
    }
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(
        &session->packet, "data: [DONE]\n\n"));
  } else {
    // No headers have been published. Translation failures can still become
    // an HTTP error instead of being hidden inside a successful response.
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(
        &session->packet,
        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
        "Connection: close\r\n\r\n"));
    IREE_RETURN_IF_ERROR(loom_serve_text_chat_response(
        response,
        iree_string_view_substr(iree_string_builder_view(&session->response), 0,
                                session->completion.text_end),
        tool_count ? iree_string_builder_view(tool_calls)
                   : iree_string_view_empty(),
        finish_reason, &session->packet));
  }
  session->request.phase = LOOM_SERVE_TEXT_REQUEST_FINISHING;
  return iree_ok_status();
}

iree_status_t loom_serve_text_session_append_tokens(
    loom_serve_text_session_t* session, iree_string_view_t model_name,
    iree_host_size_t count, const int32_t* tokens,
    iree_string_builder_t* scratch, iree_string_builder_t* tool_calls) {
  if (!session->request.output_count) {
    session->request.first_token_time = iree_time_now();
  }
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < count && iree_status_is_ok(status); ++i) {
    ++session->request.output_count;
    iree_host_size_t consumed = 0;
    while (!consumed && iree_status_is_ok(status)) {
      char text[8192];
      iree_host_size_t length = 0;
      status = iree_tokenizer_decode_state_feed(
          session->decoder, iree_tokenizer_make_token_id_list(tokens + i, 1),
          iree_make_mutable_string_view(text, sizeof(text)), &consumed,
          &length);
      if (iree_status_is_ok(status) && !consumed && !length) {
        status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                  "token exceeds streaming text capacity");
      }
      if (iree_status_is_ok(status)) {
        status =
            text_append_response(session, iree_make_string_view(text, length));
      }
    }
  }
  if (iree_status_is_ok(status)) {
    const loom_serve_text_chat_response_t response =
        text_response_metadata(session, model_name);
    const bool complete =
        loom_serve_text_row_is_eos(session->row) ||
        session->request.output_count == session->request.chat.max_tokens;
    if (complete) {
      status = text_complete(session, &response, scratch, tool_calls);
    } else if (iree_any_bit_set(response.flags,
                                LOOM_SERVE_TEXT_CHAT_FLAG_STREAM)) {
      iree_host_size_t text_end = 0;
      status = loom_serve_text_chat_text_end(
          session->request.chat.policy,
          iree_string_builder_view(&session->response),
          session->request.emitted_length,
          LOOM_SERVE_TEXT_CHAT_OUTPUT_STREAMING, &text_end, session->allocator);
      if (iree_status_is_ok(status)) {
        status = text_emit_text(session, &response, text_end, scratch);
      }
    }
  }
  return status;
}
