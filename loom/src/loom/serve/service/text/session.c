// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/service/text/session.h"

#include <string.h>

#include "loom/serve/runtime/host/clock.h"

bool loom_serve_text_session_name_valid(iree_string_view_t name) {
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

loom_serve_text_session_t* loom_serve_text_session_select(
    iree_host_size_t count, loom_serve_text_session_t* sessions,
    iree_string_view_t name, const loom_serve_text_chat_t* chat) {
  loom_serve_text_session_t* oldest = NULL;
  loom_serve_text_session_t* continuation = NULL;
  for (iree_host_size_t i = 0; i < count; ++i) {
    loom_serve_text_session_t* session = &sessions[i];
    if (name.size &&
        iree_string_view_equal(
            name, iree_make_cstring_view(session->lifetime.name))) {
      return session;
    }
    if (session->request.consumer.self) {
      continue;
    }
    if (!oldest || session->lifetime.serial < oldest->lifetime.serial) {
      oldest = session;
    }
    if (!name.size && chat && !session->lifetime.name[0] &&
        session->completion.storage && session->completion.reusable &&
        (!continuation || session->completion.transcript.size >
                              continuation->completion.transcript.size) &&
        loom_serve_text_chat_completion_is_prefix(&session->completion, chat)) {
      continuation = session;
    }
  }
  return continuation ? continuation : oldest;
}

iree_status_t loom_serve_text_session_initialize(
    loom_serve_text_row_t* row, const iree_tokenizer_t* tokenizer,
    iree_host_size_t context_capacity, iree_allocator_t host_allocator,
    loom_serve_text_session_t* out_session) {
  *out_session = (loom_serve_text_session_t){
      .host_allocator = host_allocator,
      .row = row,
  };
  iree_string_builder_initialize(host_allocator, &out_session->response);
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
  return iree_string_builder_reserve(&out_session->response, 65536);
}

void loom_serve_text_session_release_request(
    loom_serve_text_session_t* session) {
  loom_serve_text_media_request_free(session->request.media);
  loom_serve_text_chat_deinitialize(&session->request.chat);
  iree_tokenizer_decode_state_deinitialize(session->decoder);
  session->decoder = NULL;
  memset(&session->request, 0, sizeof(session->request));
}

void loom_serve_text_session_deinitialize(loom_serve_text_session_t* session) {
  if (!session->row) {
    return;
  }
  loom_serve_text_session_release_request(session);
  iree_allocator_free(session->host_allocator, session->decoder_storage.data);
  iree_allocator_free(session->host_allocator, session->tokens);
  iree_string_builder_deinitialize(&session->response);
  loom_serve_text_chat_completion_deinitialize(&session->completion);
  memset(session, 0, sizeof(*session));
}

loom_serve_http_request_snapshot_t loom_serve_text_session_request_snapshot(
    const loom_serve_text_session_t* session) {
  return (loom_serve_http_request_snapshot_t){
      .id = session->request.consumer.self ? session->lifetime.serial : 0,
      .session = session->lifetime.id,
      .phase = "active",
      .arrival_time =
          session->request.start_time - session->request.queue_duration,
      .counters = session->request.counters};
}

static loom_serve_text_chat_response_t loom_serve_text_response_metadata(
    const loom_serve_text_session_t* session, iree_string_view_t model_name) {
  return (loom_serve_text_chat_response_t){
      .model_name = model_name,
      .request_id = session->lifetime.serial,
      .created_unix_seconds = session->request.created_unix_seconds,
      .flags = session->request.chat.flags,
      .usage = {.input_tokens = session->request.retained_count +
                                session->request.input_count,
                .cached_tokens = session->request.retained_count,
                .output_tokens = session->request.output_count}};
}

iree_status_t loom_serve_text_session_begin(loom_serve_text_session_t* session,
                                            const iree_tokenizer_t* tokenizer,
                                            iree_string_view_t model_name) {
  session->request.created_unix_seconds = loom_serve_clock_now_unix_seconds();
  iree_string_builder_reset(&session->response);
  loom_serve_text_chat_completion_deinitialize(&session->completion);
  IREE_RETURN_IF_ERROR(iree_tokenizer_decode_state_initialize(
      tokenizer, IREE_TOKENIZER_DECODE_FLAG_SKIP_SPECIAL_TOKENS,
      session->decoder_storage, &session->decoder));
  const loom_serve_text_chat_response_t response =
      loom_serve_text_response_metadata(session, model_name);
  return session->request.consumer.vtable->begin(session->request.consumer.self,
                                                 &response);
}

static iree_status_t loom_serve_text_emit_output(
    loom_serve_text_session_t* session,
    const loom_serve_text_chat_response_t* response,
    loom_serve_text_chat_output_phase_t phase, iree_string_builder_t* scratch,
    iree_allocator_t scratch_allocator) {
  iree_status_t status = iree_ok_status();
  loom_serve_text_chat_output_t output;
  do {
    status = loom_serve_text_chat_output_next(
        &session->request.chat, iree_string_builder_view(&session->response),
        phase, scratch_allocator, &output);
    if (iree_status_is_ok(status) && output.text.size) {
      status = session->request.consumer.vtable->append(
          session->request.consumer.self, response, &output, scratch);
      if (iree_status_is_ok(status)) {
        session->request.publication.pending = true;
      }
    }
  } while (iree_status_is_ok(status) && output.advanced);
  return status;
}

static iree_status_t loom_serve_text_append_response(
    loom_serve_text_session_t* session, iree_string_view_t text) {
  if (text.size > 1024 * 1024 - iree_string_builder_size(&session->response)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "generated response exceeds 1 MiB text limit");
  }
  return iree_string_builder_append_string(&session->response, text);
}

static iree_status_t loom_serve_text_complete(
    loom_serve_text_session_t* session,
    const loom_serve_text_chat_response_t* response,
    iree_string_builder_t* scratch, loom_serve_text_tool_calls_t* tool_calls,
    loom_serve_text_chat_scratch_t* completion_scratch,
    iree_allocator_t scratch_allocator) {
  char text[8192];
  iree_host_size_t length = 0;
  IREE_RETURN_IF_ERROR(iree_tokenizer_decode_state_finalize(
      session->decoder, iree_make_mutable_string_view(text, sizeof(text)),
      &length));
  IREE_RETURN_IF_ERROR(loom_serve_text_append_response(
      session, iree_make_string_view(text, length)));
  if (iree_any_bit_set(response->flags, LOOM_SERVE_TEXT_CHAT_FLAG_STREAM)) {
    IREE_RETURN_IF_ERROR(loom_serve_text_emit_output(
        session, response, LOOM_SERVE_TEXT_CHAT_OUTPUT_COMPLETE, scratch,
        scratch_allocator));
  }
  IREE_RETURN_IF_ERROR(loom_serve_text_chat_complete(
      &session->request.chat, iree_string_builder_view(&session->response),
      session->lifetime.serial, tool_calls, completion_scratch,
      scratch_allocator, &session->completion));
  const iree_host_size_t tool_count = tool_calls->count;
  session->request.finish_reason = tool_count ? "tool_calls"
                                   : loom_serve_text_row_is_eos(session->row)
                                       ? "stop"
                                       : "length";
  const iree_string_view_t finish_reason =
      iree_make_cstring_view(session->request.finish_reason);
  if (iree_any_bit_set(response->flags, LOOM_SERVE_TEXT_CHAT_FLAG_STREAM)) {
    session->request.publication.pending |= tool_count != 0;
  } else {
    session->request.publication.pending =
        iree_string_builder_size(&session->request.chat.output.content) ||
        iree_string_builder_size(&session->request.chat.output.reasoning) ||
        tool_count;
  }
  IREE_RETURN_IF_ERROR(session->request.consumer.vtable->complete(
      session->request.consumer.self, response,
      &session->request.chat.output.content,
      &session->request.chat.output.reasoning, tool_calls, finish_reason,
      scratch));
  session->request.phase = LOOM_SERVE_TEXT_REQUEST_FINISHING;
  return iree_ok_status();
}

static iree_status_t loom_serve_text_session_decode_tokens(
    loom_serve_text_session_t* session, iree_string_view_t model_name,
    iree_host_size_t count, const int32_t* tokens,
    iree_string_builder_t* scratch, loom_serve_text_tool_calls_t* tool_calls,
    loom_serve_text_chat_scratch_t* completion_scratch,
    iree_allocator_t scratch_allocator) {
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
        status = loom_serve_text_append_response(
            session, iree_make_string_view(text, length));
      }
    }
  }
  if (iree_status_is_ok(status)) {
    const loom_serve_text_chat_response_t response =
        loom_serve_text_response_metadata(session, model_name);
    const bool complete =
        loom_serve_text_row_is_eos(session->row) ||
        session->request.output_count == session->request.chat.max_tokens;
    if (complete) {
      status = loom_serve_text_complete(session, &response, scratch, tool_calls,
                                        completion_scratch, scratch_allocator);
    } else if (iree_any_bit_set(response.flags,
                                LOOM_SERVE_TEXT_CHAT_FLAG_STREAM)) {
      status = loom_serve_text_emit_output(
          session, &response, LOOM_SERVE_TEXT_CHAT_OUTPUT_STREAMING, scratch,
          scratch_allocator);
    }
  }
  return status;
}

iree_status_t loom_serve_text_session_append_tokens(
    loom_serve_text_session_t* session, iree_string_view_t model_name,
    iree_host_size_t count, const int32_t* tokens,
    iree_string_builder_t* scratch, loom_serve_text_tool_calls_t* tool_calls,
    loom_serve_text_chat_scratch_t* completion_scratch,
    iree_allocator_t scratch_allocator) {
  iree_status_t status = loom_serve_text_session_decode_tokens(
      session, model_name, count, tokens, scratch, tool_calls,
      completion_scratch, scratch_allocator);
  if (!iree_status_is_ok(status)) {
    // Device work completed, but translation cannot establish client history.
    // The consumer discards the partial stage and preserves the terminal error.
    loom_serve_text_chat_completion_deinitialize(&session->completion);
    loom_serve_telemetry_diagnostic_capture(status,
                                            &session->request.diagnostic);
    session->request.publication.pending = false;
    status = session->request.consumer.vtable->fail(
        session->request.consumer.self, session->request.chat.flags, status);
    session->request.finish_reason = "error";
    session->request.phase = LOOM_SERVE_TEXT_REQUEST_FINISHING;
  }
  return status;
}
