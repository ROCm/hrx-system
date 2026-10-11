// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXPERIMENTAL_LOOM_SERVE_TEXT_INPUT_H_
#define EXPERIMENTAL_LOOM_SERVE_TEXT_INPUT_H_

#include "experimental/loom_serve/text/chat.h"

#ifdef __cplusplus
extern "C" {
#endif

// Borrowed completed frontier, stable during one exclusive preparation call.
// Its immutable transcript storage identifies the exact completed history,
// independently of the row or cache entry currently holding native state.
typedef struct loom_serve_text_input_prefix_t {
  // Canonical transcript; null means no retained continuation is available.
  const loom_serve_text_chat_completion_t* completion;
  // Consumed native positions preceding the selected but unconsumed token.
  iree_host_size_t position;
  // Exact selected token, never reconstructed from rendered text.
  int32_t token;
  // Source framing for the text appended after this completed history.
  loom_serve_text_chat_boundary_t boundary;
} loom_serve_text_input_prefix_t;

// Compact host input owned independently of device rows or reservations.
// A parked input retains host history identity, not native cache ownership.
typedef struct loom_serve_text_input_t {
  // Owned token IDs, null until successfully prepared, including empty input.
  int32_t* tokens;
  // Number of token IDs in tokens.
  iree_host_size_t count;
  // Owned immutable history identity; null for full replay.
  iree_vm_buffer_t* history;
  // Consumed native positions preceding tokens.
  iree_host_size_t retained_count;
} loom_serve_text_input_t;

// Releases compact tokens and the host history reference; accepts zero state.
void loom_serve_text_input_deinitialize(loom_serve_text_input_t* input,
                                        iree_allocator_t allocator);

// Tests whether preparation still names the available immutable frontier.
// The chat and optional required-prefix check were validated by prepare.
bool loom_serve_text_input_is_current(const loom_serve_text_input_t* input,
                                      const loom_serve_text_chat_t* chat,
                                      loom_serve_text_input_prefix_t prefix);

// Prepares once per actual history identity. Capacity retries with the same
// history reuse tokens; eviction prepares full replay instead of using stale
// suffix tokens. Required prefixes reject mismatching canonical histories.
// Scratch holds context_capacity IDs and is borrowed only during this call.
// Input remains unchanged on failure; new storage uses host_allocator.
iree_status_t loom_serve_text_input_prepare(
    const loom_serve_text_chat_t* chat, loom_serve_text_input_prefix_t prefix,
    bool require_prefix, iree_host_size_t context_capacity, int32_t* scratch,
    loom_serve_text_input_t* input, iree_allocator_t host_allocator);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // EXPERIMENTAL_LOOM_SERVE_TEXT_INPUT_H_
