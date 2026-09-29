// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_EXPERIMENTAL_LOOM_SERVE_QWEN_CHAT_H_
#define IREE_EXPERIMENTAL_LOOM_SERVE_QWEN_CHAT_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif

#define LOOM_SERVE_QWEN_CHAT_MODEL "qwen3.8-27b"
#define LOOM_SERVE_QWEN_CHAT_TOOL_CAPACITY 16

// Indexed tool schema borrowing the request body. XML names are literal ASCII
// identifiers. Property schemas supply the JSON types lost by XML generation;
// the agent remains responsible for tool execution and full schema validation.
typedef struct loom_serve_qwen_chat_tool_t {
  // Literal function name, without JSON quotes.
  iree_string_view_t name;
  // JSON object mapping parameter names to schemas.
  iree_string_view_t properties;
  // JSON array of required parameter names, or empty when absent.
  iree_string_view_t required;
} loom_serve_qwen_chat_tool_t;

// One validated text-only, greedy, non-thinking streaming chat request. The
// prompt owns its canonical rendered text; tool views borrow the HTTP body.
typedef struct loom_serve_qwen_chat_t {
  // Canonical Qwen template ending with the assistant generation prefix.
  iree_string_builder_t prompt;
  // Maximum selected output tokens, including a terminal EOS prediction.
  iree_host_size_t max_tokens;
  // Whether the client requested a final usage event.
  bool include_usage;
  // Number of populated tool descriptors.
  iree_host_size_t tool_count;
  // Validated tool descriptors in request order.
  loom_serve_qwen_chat_tool_t tools[LOOM_SERVE_QWEN_CHAT_TOOL_CAPACITY];
} loom_serve_qwen_chat_t;

// Validates and renders the supported chat request. Unknown execution options
// fail instead of pretending to implement sampling, thinking or other modes.
// Body storage outlives the initialized chat. Deinitialize after success only;
// failure releases partial storage. The HTTP layer bounds the input body.
iree_status_t loom_serve_qwen_chat_initialize(
    iree_string_view_t body, iree_host_size_t default_max_tokens,
    iree_allocator_t host_allocator, loom_serve_qwen_chat_t* out_chat);
void loom_serve_qwen_chat_deinitialize(loom_serve_qwen_chat_t* chat);

// Finds the next safe text extent in an accumulated decoded response. An XML
// tool marker or its incomplete suffix is withheld from ordinary text output.
// Scan starts at the previous safe extent; no generated prefix is rescanned.
iree_host_size_t loom_serve_qwen_chat_text_end(iree_string_view_t response,
                                               iree_host_size_t previous_end);

// Parses complete generated XML calls against the indexed request schemas.
// Appends the canonical completed transcript to checkpoint and an OpenAI
// tool_calls delta array (including indexes and stable IDs) to tool_calls.
// Plain text responses produce an empty array. Malformed/truncated XML fails;
// it is never submitted to the agent as a partial executable call.
// The checkpoint validates subsequent client history; it does not replace the
// original generated tokens retained in device KV/recurrent state.
iree_status_t loom_serve_qwen_chat_complete(const loom_serve_qwen_chat_t* chat,
                                            iree_string_view_t response,
                                            uint64_t request_id,
                                            iree_string_builder_t* checkpoint,
                                            iree_string_builder_t* tool_calls,
                                            iree_host_size_t* out_tool_count);

// Appends one SSE chat event. Delta is an already serialized JSON object;
// finish_reason is empty for nonterminal events. Usage belongs to the complete
// client transcript, with retained_tokens reported as cached input tokens.
iree_status_t loom_serve_qwen_chat_event(uint64_t request_id,
                                         iree_string_view_t delta,
                                         iree_string_view_t finish_reason,
                                         iree_string_builder_t* output);
iree_status_t loom_serve_qwen_chat_usage(uint64_t request_id,
                                         iree_host_size_t input_tokens,
                                         iree_host_size_t retained_tokens,
                                         iree_host_size_t output_tokens,
                                         iree_string_builder_t* output);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_EXPERIMENTAL_LOOM_SERVE_QWEN_CHAT_H_
