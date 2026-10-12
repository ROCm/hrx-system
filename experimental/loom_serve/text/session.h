// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXPERIMENTAL_LOOM_SERVE_TEXT_SESSION_H_
#define EXPERIMENTAL_LOOM_SERVE_TEXT_SESSION_H_

#include "experimental/loom_serve/http/inspection.h"
#include "experimental/loom_serve/http/server.h"
#include "experimental/loom_serve/telemetry/record.h"
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

// Logical identity and accounting owned by the serialized model service. This
// value contains no row, decoder, request or device borrow. Moving it transfers
// one lifetime; row reuse alone does not imply a new session lifetime.
typedef struct loom_serve_text_session_lifetime_t {
  // Local client cache key; empty means an untagged request.
  char name[65];
  // Last admitted request serial, also the idle-session LRU ordering key.
  uint64_t serial;
  // ID assigned by the first admitted request; zero before first admission.
  uint64_t id;
  // Cumulative work across retained continuations, independent of log
  // retention.
  loom_serve_telemetry_counters_t counters;
  // Last completed endpoint, retained even when replacement resets the row.
  iree_host_size_t position;
  // Owned positions at that endpoint, including any shared prefix.
  iree_host_size_t resident_tokens;
} loom_serve_text_session_lifetime_t;

// Host continuation and request output owned by one serialized model service.
// Source VM calls and output decoding may run on a finite CPU job only while
// that job exclusively borrows this session and the model invocation.
typedef struct loom_serve_text_session_t {
  // Allocator for this session's reusable host storage.
  iree_allocator_t allocator;
  // Borrowed private device row in the shared model residency.
  loom_serve_text_row_t* row;
  // Owned logical session currently assigned to this reusable execution row.
  loom_serve_text_session_lifetime_t lifetime;
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
    // Token-array origin after exact-prefix reuse. Full replay IDs preceding
    // this offset remain available for later consumed-frontier identity.
    iree_host_size_t input_origin;
    // Input tokens already consumed by prefill.
    iree_host_size_t input_offset;
    // Retained device tokens preceding this request's appended input.
    iree_host_size_t retained_count;
    // Selected output tokens, including EOS.
    iree_host_size_t output_count;
    // Ordinary response bytes already incorporated in copied SSE output.
    iree_host_size_t emitted_length;
    // Terminal wire literal: stop, length, tool_calls, or error. NULL until
    // translation establishes a terminal response.
    const char* finish_reason;
    // Application admission time; network framing is not included.
    iree_time_t start_time;
    // Time spent awaiting row and completion capacity before admission.
    iree_duration_t queue_duration;
    // Page-rounded maximum extent; admission charges only remaining growth.
    iree_host_size_t reserved_tokens;
    // First selected token time, including a possible immediate EOS.
    iree_time_t first_token_time;
    // Owner-only completed work, independent of client output translation.
    loom_serve_telemetry_counters_t counters;
    // CPU output failure copied before consuming its status; owner publishes
    // only after the CPU job releases the session.
    loom_serve_telemetry_diagnostic_t diagnostic;
  } request;
} loom_serve_text_session_t;

// Copies only owner-published work and immutable admission metadata. Safe on
// the serialized owner while a CPU task borrows the row's response storage;
// request phase/output buffers are deliberately excluded. ID zero means idle.
loom_serve_http_request_snapshot_t loom_serve_text_session_request_snapshot(
    const loom_serve_text_session_t* session);

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
