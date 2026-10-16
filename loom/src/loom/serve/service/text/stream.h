// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_SERVICE_TEXT_STREAM_H_
#define LOOM_SERVE_SERVICE_TEXT_STREAM_H_

#include "loom/serve/service/text/consumer.h"

#ifdef __cplusplus
extern "C" {
#endif

// Native admission preserves the complete source-defined chat request. The
// caller builds JSON before entering the serialized serving owner; acceptance
// moves its storage without copying conversation or image payloads. Pressure
// and failure leave this value untouched.
typedef struct loom_serve_text_request_t {
  // Initialized owning builder containing the supported chat JSON document.
  // Fixed borrowed storage and sizing-only builders are not accepted.
  iree_string_builder_t body;
  // Optional retained-session identifier, borrowed until admission returns.
  iree_string_view_t session;
  // Optional pinned-checkpoint identifier, borrowed until admission returns.
  iree_string_view_t checkpoint;
} loom_serve_text_request_t;

typedef struct loom_serve_text_stream_t loom_serve_text_stream_t;
typedef struct loom_serve_text_stream_block_t loom_serve_text_stream_block_t;

// One source-classified span, in generation order across output channels.
typedef struct loom_serve_text_stream_span_t {
  // Content or reasoning; hidden source framing is never published.
  loom_serve_text_chat_output_kind_t kind;
  // Owned block bytes, borrowed until consume or close.
  iree_string_view_t text;
} loom_serve_text_stream_span_t;

// Final validated output, independent of source VM and retained model history.
typedef struct loom_serve_text_stream_result_t {
  // Stable identity and committed token usage; model_name is handle-owned.
  loom_serve_text_chat_response_t response;
  // Complete answer bytes, immutable until the caller closes its handle.
  iree_string_view_t content;
  // Complete reasoning bytes, immutable until close.
  iree_string_view_t reasoning;
  // Validated calls published only after the entire source transaction
  // succeeds.
  const loom_serve_text_tool_calls_t* calls;
  // Source completion classification with process lifetime.
  iree_string_view_t finish_reason;
} loom_serve_text_stream_result_t;

// Admission implementation: copies the deployment name and validated bounded
// selectors, moves request on success, and creates independent caller/service
// ownership. Source validation and numerical work remain with the service.
iree_status_t loom_serve_text_stream_allocate(
    iree_string_view_t model_name, loom_serve_text_request_t* request,
    iree_allocator_t host_allocator, loom_serve_text_stream_t** out_stream,
    loom_serve_text_consumer_t* out_consumer);

// Service-only immutable input, alive until its consumer's reject/end call.
// The admitted queue and source rendering borrow this storage through all CPU
// and media work; the final caller result does not retain the input document.
const loom_serve_text_request_t* loom_serve_text_stream_request(
    const loom_serve_text_stream_t* stream);

// All caller operations share the service's serialized owner. CPU collection
// uses a separate staging block and never reads caller interest or output
// credit. A stalled caller makes only its own request ineligible.
//
// Borrows the pending ordered block, including an empty final block, or NULL
// when none is published. The borrow remains valid until consume/close, even
// if service shutdown discards its pending publication in the meantime.
const loom_serve_text_stream_block_t* loom_serve_text_stream_peek(
    const loom_serve_text_stream_t* stream);
iree_host_size_t loom_serve_text_stream_block_count(
    const loom_serve_text_stream_block_t* block);
// index is less than block_count, established by the consuming iteration.
loom_serve_text_stream_span_t loom_serve_text_stream_block_at(
    const loom_serve_text_stream_block_t* block, iree_host_size_t index);
// Consumes the non-NULL block returned by peek on this owner turn.
void loom_serve_text_stream_consume(loom_serve_text_stream_t* stream);

// Completion follows execution retirement and consumption of the final block.
// Cancellation/failure can discard pending publication without freeing a live
// caller's previously borrowed bytes. Failed device readiness leaves accepted
// device resources with model teardown, independently of this host result.
bool loom_serve_text_stream_is_complete(const loom_serve_text_stream_t* stream);
// Owned copy of the owner-published failure; querying never clears it.
iree_status_t loom_serve_text_stream_status(
    const loom_serve_text_stream_t* stream);
// Final output after successful execution retirement, or NULL beforehand/on
// failure. A final block may still await consume. All views survive service,
// model, and registry destruction until the caller closes its handle.
const loom_serve_text_stream_result_t* loom_serve_text_stream_result(
    const loom_serve_text_stream_t* stream);

// Consumes caller ownership immediately. Accepted source/device borrowers keep
// the receipt alive until retirement; closing never waits for that work.
void loom_serve_text_stream_close(loom_serve_text_stream_t* stream);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVE_SERVICE_TEXT_STREAM_H_
