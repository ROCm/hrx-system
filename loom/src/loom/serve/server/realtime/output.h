// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_SERVER_REALTIME_OUTPUT_H_
#define LOOM_SERVE_SERVER_REALTIME_OUTPUT_H_

#include "loom/serve/server/realtime/conversation.h"
#include "loom/serve/server/realtime/wire.h"
#include "loom/util/stream.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum loom_serve_realtime_output_state_e {
  LOOM_SERVE_REALTIME_OUTPUT_GENERATING,
  LOOM_SERVE_REALTIME_OUTPUT_COMPLETED,
  LOOM_SERVE_REALTIME_OUTPUT_TOKEN_LIMIT,
  LOOM_SERVE_REALTIME_OUTPUT_CLIENT_CANCELLED,
  LOOM_SERVE_REALTIME_OUTPUT_TURN_DETECTED,
  LOOM_SERVE_REALTIME_OUTPUT_FAILED,
} loom_serve_realtime_output_state_t;

// Read-only wire view, independent of model, worker, and response ownership.
typedef struct loom_serve_realtime_output_t {
  // Borrowed stable response identity.
  iree_string_view_t id;
  // First output member, or NULL for a response with no announced output.
  const loom_serve_realtime_item_t* first;
  // Last announced member; later reserved members are not visible on the wire.
  const loom_serve_realtime_item_t* last;
  // Owned elsewhere, immutable JSON object; empty means null metadata.
  iree_string_view_t metadata;
  // Effective response modality, including its accompanying transcript.
  bool audio;
  // Selected terminal cause, or GENERATING for response.created.
  loom_serve_realtime_output_state_t state;
  // Borrowed diagnostic, present only for FAILED output.
  iree_status_t failure;
  // Native prompt tokens, including reused tokens.
  iree_host_size_t input_tokens;
  // Reused native prompt prefix tokens.
  iree_host_size_t cached_tokens;
  // Native generated token count, including source terminal tokens.
  iree_host_size_t output_tokens;
} loom_serve_realtime_output_t;

// Item and event serializers share real and null sinks. Event-body writers
// follow wire_begin's open prefix and close the outer object.
iree_status_t loom_serve_realtime_item_write(
    const loom_serve_realtime_item_t* item,
    loom_serve_realtime_item_status_t status, bool initial,
    loom_output_stream_t* output);
iree_status_t loom_serve_realtime_output_write(
    const loom_serve_realtime_output_t* view, loom_output_stream_t* output);
iree_status_t loom_serve_realtime_output_item_write(
    iree_string_view_t response_id, const loom_serve_realtime_item_t* item,
    iree_host_size_t index, loom_serve_realtime_item_status_t status,
    bool initial, loom_output_stream_t* output);
iree_status_t loom_serve_realtime_output_arguments_write(
    iree_string_view_t response_id, const loom_serve_realtime_item_t* item,
    iree_host_size_t index, bool done, loom_output_stream_t* output);
iree_status_t loom_serve_realtime_output_conversation_write(
    const loom_serve_realtime_item_t* item,
    const loom_serve_realtime_item_t* previous, bool initial,
    loom_output_stream_t* output);

// Publishes a canonical item acknowledgement using the same formatter as
// response admission. The queue owns the resulting bytes, never the item.
iree_status_t loom_serve_realtime_item_event(
    loom_serve_realtime_wire_t* wire, const char* type,
    const loom_serve_realtime_item_t* item,
    const loom_serve_realtime_item_t* previous);

// Preflights a complete generated call transaction before executable output.
// The view includes every reserved member and successful usage. Calls are still
// in_progress (the longest status spelling); cancelled message status and both
// cancellation reasons are measured too. Every form uses the actual serializer
// and the maximum future event-ID width, without allocating a terminal copy.
iree_status_t loom_serve_realtime_output_admit(
    uint64_t session_id, const loom_serve_realtime_output_t* view);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVE_SERVER_REALTIME_OUTPUT_H_
