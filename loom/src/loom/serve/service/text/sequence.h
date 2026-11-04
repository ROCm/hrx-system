// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_SERVICE_TEXT_SEQUENCE_H_
#define LOOM_SERVE_SERVICE_TEXT_SEQUENCE_H_

#include "iree/base/api.h"
#include "iree/base/string_builder.h"
#include "iree/vm/buffer.h"
#include "loom/serve/service/text/input.h"
#include "loom/serve/service/text/model.h"

#ifdef __cplusplus
extern "C" {
#endif

// Known-token append protocol, independent of chat framing and device rows.
typedef struct loom_serve_text_sequence_t {
  // Opaque logical identity; a new attachment receives a random 128-bit name.
  char handle[33];
  // Caller supplied an existing identity rather than requesting a new one.
  bool append;
  // Expected consumed frontier, checked against retained state before mutation.
  iree_host_size_t position;
  // Borrowed raw JSON token array in the claimed immutable request body.
  iree_string_view_t token_ids;
  // Owned source-validated policy; null until CPU preparation succeeds.
  iree_vm_buffer_t* options;
} loom_serve_text_sequence_t;

// Parses the request envelope without acquiring a row or decoding token IDs.
// Unknown, duplicate, and wrongly typed fields fail at this public boundary.
iree_status_t loom_serve_text_sequence_parse(
    iree_string_view_t body, loom_serve_text_sequence_t* out_sequence);
void loom_serve_text_sequence_deinitialize(
    loom_serve_text_sequence_t* sequence);

// An exclusive model-VM borrower validates the current retained identity and
// frontier, actual IDs, and source options before reservation. Tokens are
// copied once into compact input storage; capacity retries reuse them but
// recheck the retained frontier. A checkpoint creates a new logical branch
// identity.
iree_status_t loom_serve_text_sequence_prepare(
    loom_serve_text_sequence_t* sequence, loom_serve_text_model_t* model,
    iree_string_view_t body, uint64_t request_id,
    loom_serve_text_input_prefix_t prefix, bool checkpoint,
    iree_allocator_t host_allocator, loom_serve_text_input_t* out_input);

// Appends one completed epoch's opaque bytes and exact consumed IDs to the
// credited packet. Position is the pre-epoch frontier. The final chunk includes
// a terminal event; no peer acknowledgment changes the local commit frontier.
iree_status_t loom_serve_text_sequence_chunk(
    iree_string_view_t handle, iree_host_size_t position,
    iree_host_size_t count, const int32_t* tokens, iree_const_byte_span_t bytes,
    bool final, iree_string_builder_t* output);

#ifdef __cplusplus
}  // extern "C"
#endif
#endif  // LOOM_SERVE_SERVICE_TEXT_SEQUENCE_H_
