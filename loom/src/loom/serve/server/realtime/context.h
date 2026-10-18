// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_SERVER_REALTIME_CONTEXT_H_
#define LOOM_SERVE_SERVER_REALTIME_CONTEXT_H_

#include "loom/serve/server/realtime/conversation.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_serve_realtime_context_insert_t {
  // Offset in the immutable request at which a transcript string is inserted.
  iree_host_size_t offset;
  // Retained pending recognition result, independent of canonical item
  // lifetime.
  loom_serve_realtime_input_item_t* input;
} loom_serve_realtime_context_insert_t;

// Accepted native input; no mutable conversation or command views survive.
typedef struct loom_serve_realtime_context_t {
  // Owned native JSON, moved directly to admission when no insertions exist.
  iree_string_builder_t body;
  // Ordered insertions into body; storage exists only for pending transcripts.
  loom_serve_realtime_context_insert_t* inserts;
  // Number of retained input occurrences, including repeated references.
  iree_host_size_t count;
  // Allocated insertion capacity, bounded by the selected-item limit.
  iree_host_size_t capacity;
  // Decoded input bytes already serialized, excluding instructions and tools.
  iree_host_size_t bytes;
} loom_serve_realtime_context_t;

// Captures the current default history when input is absent, or the exact
// explicit raw/reference item array when present (including []). Pending audio
// retains its result receipt; all ready bytes and grouping are fixed now.
// Failure leaves context deinitializable and changes no canonical history.
iree_status_t loom_serve_realtime_context_initialize(
    loom_serve_realtime_conversation_t* conversation, iree_string_view_t input,
    iree_string_view_t model, iree_string_view_t instructions,
    iree_string_view_t tools, uint32_t max_tokens, uint64_t seed,
    loom_serve_realtime_context_t* context);
void loom_serve_realtime_context_deinitialize(
    loom_serve_realtime_context_t* context);
// Readiness is independent of wire credit; a real recognition failure
// propagates.
iree_status_t loom_serve_realtime_context_ready(
    const loom_serve_realtime_context_t* context, bool* out_ready);
// Consumes a ready context into an owning native request builder. Pending final
// text is bounded before insertion; allocation or size failures remain
// fallible.
iree_status_t loom_serve_realtime_context_take(
    loom_serve_realtime_context_t* context, iree_string_builder_t* out_body);

#ifdef __cplusplus
}  // extern "C"
#endif
#endif  // LOOM_SERVE_SERVER_REALTIME_CONTEXT_H_
