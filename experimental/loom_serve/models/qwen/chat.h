// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_EXPERIMENTAL_LOOM_SERVE_MODELS_QWEN_CHAT_H_
#define IREE_EXPERIMENTAL_LOOM_SERVE_MODELS_QWEN_CHAT_H_

#include "experimental/loom_serve/runtime/program.h"
#include "iree/base/api.h"
#include "iree/vm/buffer.h"

#ifdef __cplusplus
extern "C" {
#endif

#define LOOM_SERVE_QWEN_CHAT_MODEL "qwen3.8-27b"
#define LOOM_SERVE_QWEN_CHAT_TOOL_CAPACITY 16

// Cold-resolved chat functions in the model's shared source VM. The model
// serializes calls and outlives every request borrowing this policy.
typedef struct loom_serve_qwen_chat_policy_t {
  // Canonical buffer identity borrowed from the model environment.
  iree_vm_ref_types_t types;
  // Invocation borrowed from the model-wide program.
  iree_vm_invocation_t* invocation;
  // Canonical tool formatter shared by input history and generated output.
  iree_vm_function_t render_tool;
  // Turn framing and complete raw-token encoding shared by all text callers.
  iree_vm_function_t prepare_input;
  // Safe ordinary-text extent in an accumulated generated response.
  iree_vm_function_t text_end;
  // Canonical completed transcript, returned as retained VM storage.
  iree_vm_function_t complete_text;
} loom_serve_qwen_chat_policy_t;

iree_status_t loom_serve_qwen_chat_policy_initialize(
    iree_vm_environment_t* environment, loom_serve_program_t* program,
    loom_serve_qwen_chat_policy_t* out_policy);

// Source input format. Rendered text passes through unchanged; user content
// receives one user envelope and the assistant generation prefix.
typedef enum loom_serve_qwen_chat_input_format_e {
  LOOM_SERVE_QWEN_CHAT_INPUT_RENDERED = 0,
  LOOM_SERVE_QWEN_CHAT_INPUT_USER = 1,
} loom_serve_qwen_chat_input_format_t;

// Framing before the new text. Retained callers separately preserve the exact
// selected-but-unconsumed token ID; this operation never reconstructs that ID
// from the canonical transcript. TERMINATED means that ID already ends the
// prior turn; OPEN requires the source program to close it.
typedef enum loom_serve_qwen_chat_boundary_e {
  LOOM_SERVE_QWEN_CHAT_BOUNDARY_FRESH = 0,
  LOOM_SERVE_QWEN_CHAT_BOUNDARY_OPEN = 1,
  LOOM_SERVE_QWEN_CHAT_BOUNDARY_TERMINATED = 2,
} loom_serve_qwen_chat_boundary_t;

// Frames and encodes the complete input in the shared source VM. Text is
// borrowed only during this synchronous call. On success, copies at most
// capacity IDs into tokens; on failure, tokens is untouched and count is zero.
// Incomplete prefixes are rejected. No VM reference escapes into session state.
iree_status_t loom_serve_qwen_chat_prepare_input(
    const loom_serve_qwen_chat_policy_t* policy, iree_string_view_t text,
    loom_serve_qwen_chat_input_format_t format,
    loom_serve_qwen_chat_boundary_t boundary, iree_host_size_t capacity,
    int32_t* tokens, iree_host_size_t* out_count,
    iree_allocator_t host_allocator);

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
  // Source policy borrowed from the model for the request lifetime.
  const loom_serve_qwen_chat_policy_t* policy;
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
    const loom_serve_qwen_chat_policy_t* policy, iree_string_view_t body,
    iree_host_size_t default_max_tokens, iree_allocator_t host_allocator,
    loom_serve_qwen_chat_t* out_chat);
void loom_serve_qwen_chat_deinitialize(loom_serve_qwen_chat_t* chat);

// Source output phase. Final text releases incomplete marker prefixes; complete
// tool syntax remains withheld for structured parsing in either phase.
typedef enum loom_serve_qwen_chat_output_phase_e {
  LOOM_SERVE_QWEN_CHAT_OUTPUT_STREAMING = 0,
  LOOM_SERVE_QWEN_CHAT_OUTPUT_COMPLETE = 1,
} loom_serve_qwen_chat_output_phase_t;

// Queries the next safe text extent in the shared source VM. Response is
// borrowed synchronously. Scan starts at the previous safe extent and never
// retracts published bytes; source failure publishes no new extent.
iree_status_t loom_serve_qwen_chat_text_end(
    const loom_serve_qwen_chat_policy_t* policy, iree_string_view_t response,
    iree_host_size_t previous_end, loom_serve_qwen_chat_output_phase_t phase,
    iree_host_size_t* out_end, iree_allocator_t host_allocator);

// Completed response metadata and its canonical retained-history checkpoint.
// The source environment outlives this value; request/body/response storage
// need not. Original generated token IDs remain separately owned by the row.
typedef struct loom_serve_qwen_chat_completion_t {
  // Owned VM storage, including any backing aliased by the source result.
  iree_vm_buffer_t* storage;
  // Canonical transcript borrowing storage, not request or invocation memory.
  iree_string_view_t transcript;
  // Safe ordinary-text extent in the original generated response.
  iree_host_size_t text_end;
  // Number of validated structured tool calls.
  iree_host_size_t tool_count;
} loom_serve_qwen_chat_completion_t;

// Releases a completed checkpoint; zero-initialized values are accepted.
void loom_serve_qwen_chat_completion_deinitialize(
    loom_serve_qwen_chat_completion_t* completion);

// Parses complete generated XML calls against the indexed request schemas.
// Returns the source-owned canonical completed transcript and appends an OpenAI
// tool_calls delta array (including indexes and stable IDs) to tool_calls.
// previous_end is the already published ordinary-text extent in response.
// Plain text responses produce an empty array. Malformed/truncated XML fails;
// it is never submitted to the agent as a partial executable call.
// Replaces completion only on success, releasing its prior storage. Failure
// leaves completion unchanged and may leave partial tool_calls bytes, which
// the caller must not publish. The checkpoint validates subsequent client
// history; it does not replace original generated tokens in device state.
iree_status_t loom_serve_qwen_chat_complete(
    const loom_serve_qwen_chat_t* chat, iree_string_view_t response,
    iree_host_size_t previous_end, uint64_t request_id,
    iree_string_builder_t* tool_calls,
    loom_serve_qwen_chat_completion_t* completion);

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

#endif  // IREE_EXPERIMENTAL_LOOM_SERVE_MODELS_QWEN_CHAT_H_
