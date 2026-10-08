// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXPERIMENTAL_LOOM_SERVE_TEXT_SESSION_H_
#define EXPERIMENTAL_LOOM_SERVE_TEXT_SESSION_H_

#include "experimental/loom_serve/http/server.h"
#include "experimental/loom_serve/text/chat.h"
#include "experimental/loom_serve/text/model.h"
#include "iree/tokenizer/tokenizer.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum loom_serve_text_request_phase_e {
  LOOM_SERVE_TEXT_REQUEST_PREFILL,
  LOOM_SERVE_TEXT_REQUEST_DECODE,
  LOOM_SERVE_TEXT_REQUEST_FINISHING,
} loom_serve_text_request_phase_t;

// Host continuation and request output owned by one serialized model service.
// Source VM calls and output decoding may run on a finite CPU job only while
// that job exclusively borrows this session and the model invocation.
typedef struct loom_serve_text_session_t {
  // Allocator for this session's reusable host storage.
  iree_allocator_t allocator;
  // Borrowed private device row in the shared model residency.
  loom_serve_text_row_t* row;
  // Local client cache key; empty means an untagged request.
  char name[65];
  // Last admitted request serial, also the idle-row LRU ordering key.
  uint64_t serial;
  // Canonical completed client history validating the retained raw state.
  loom_serve_text_chat_completion_t completion;
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
    loom_serve_text_chat_t chat;
    // Next completed-stage action or terminal output drain.
    loom_serve_text_request_phase_t phase;
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
    // Time spent awaiting row and completion capacity before admission.
    iree_duration_t queue_duration;
    // Page-rounded maximum extent; admission charges only remaining growth.
    iree_host_size_t reserved_tokens;
    // First selected token time, including a possible immediate EOS.
    iree_time_t first_token_time;
    // Completed epochs that consumed prompt input for this request.
    uint64_t prefill_steps;
    // Completed epochs that consumed a pending generated token.
    uint64_t decode_steps;
  } request;
} loom_serve_text_session_t;

// Allocates reusable host tokens, decoder storage and response builders. The
// borrowed row and tokenizer outlive the session. No device storage is
// acquired. The caller deinitializes the result on success or failure.
iree_status_t loom_serve_text_session_initialize(
    loom_serve_text_row_t* row, const iree_tokenizer_t* tokenizer,
    iree_host_size_t context_capacity, loom_serve_text_session_t* out_session,
    iree_allocator_t host_allocator);

// Releases host storage only. The caller has retired CPU borrowers and released
// every HTTP claim; native row/reservation ownership remains with the service.
// Zero-initialized or partially initialized state is accepted.
void loom_serve_text_session_deinitialize(loom_serve_text_session_t* session);

// Starts output for the request already transferred into session->request.
// Discards the previous canonical completion and initializes the
// shared-tokenizer decoder and initial SSE packet. Failure leaves request
// cleanup to the owner.
iree_status_t loom_serve_text_session_begin(loom_serve_text_session_t* session,
                                            const iree_tokenizer_t* tokenizer,
                                            iree_string_view_t model_name);

// Releases request-owned host state after finish or abort. The canonical
// completion and reusable host buffers survive for retained continuation.
void loom_serve_text_session_release_request(
    loom_serve_text_session_t* session);

// Decodes completed selected tokens and appends bounded SSE output. Source VM
// calls use the request's policy; scratch/tool_calls are exclusive shared model
// builders, not per-row caches. No transport send or native release occurs.
// On failure, the caller translates the error and invalidates the continuation;
// the partially modified packet is not a valid complete response.
iree_status_t loom_serve_text_session_append_tokens(
    loom_serve_text_session_t* session, iree_string_view_t model_name,
    iree_host_size_t count, const int32_t* tokens,
    iree_string_builder_t* scratch, iree_string_builder_t* tool_calls);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // EXPERIMENTAL_LOOM_SERVE_TEXT_SESSION_H_
