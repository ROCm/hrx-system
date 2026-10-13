// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_EXPERIMENTAL_LOOM_SERVE_TEXT_CHAT_H_
#define IREE_EXPERIMENTAL_LOOM_SERVE_TEXT_CHAT_H_

#include "experimental/loom_serve/runtime/program.h"
#include "iree/base/api.h"
#include "iree/vm/buffer.h"

#ifdef __cplusplus
extern "C" {
#endif

#define LOOM_SERVE_TEXT_CHAT_TOOL_CAPACITY 16

// Protocol output budget bound, independent of a deployment's context limit.
#define LOOM_SERVE_TEXT_CHAT_MAX_OUTPUT_TOKENS 16384

// Cold-resolved chat functions in the model's shared source VM. The model
// serializes calls and outlives every request borrowing this policy.
typedef struct loom_serve_text_chat_policy_t {
  // Canonical buffer identity borrowed from the model environment.
  iree_vm_ref_types_t types;
  // Invocation borrowed from the model-wide program.
  iree_vm_invocation_t* invocation;
  // Owned source-declared public model identity.
  iree_vm_buffer_t* name_storage;
  // UTF-8 identity borrowing name_storage.
  iree_string_view_t name;
  // Initial prompt fragment and opaque rendering state.
  iree_vm_function_t chat_begin;
  // Message fragment and next opaque rendering state.
  iree_vm_function_t chat_message;
  // Final prompt fragment and conversation validation.
  iree_vm_function_t chat_end;
  // Generated suffix to standard function-call records.
  iree_vm_function_t parse_tools;
  // Canonical tool formatter shared by input history and generated output.
  iree_vm_function_t render_tool;
  // Turn framing and complete raw-token encoding shared by all text callers.
  iree_vm_function_t prepare_input;
  // Safe ordinary-text extent in an accumulated generated response.
  iree_vm_function_t text_end;
  // Canonical completed transcript, returned as retained VM storage.
  iree_vm_function_t complete_text;
} loom_serve_text_chat_policy_t;

iree_status_t loom_serve_text_chat_policy_initialize(
    iree_vm_environment_t* environment, loom_serve_program_t* program,
    loom_serve_text_chat_policy_t* out_policy);

// Releases the owned identity; other fields borrow the model program.
void loom_serve_text_chat_policy_deinitialize(
    loom_serve_text_chat_policy_t* policy);

// Creates the optional tools.parse_xml(schemas, text) -> buffer capability.
// Both inputs are UTF-8 buffers. The result is a JSON array of standard
// function tool calls, with JSON-encoded argument strings and no assigned IDs.
// The codec validates declared names and scalar/container parameter types.
// Source explicitly selects this grammar by importing it; native chat handling
// consumes only the returned function records. The environment outlives the
// module and every returned buffer.
iree_status_t loom_serve_text_chat_tools_module_create(
    iree_vm_environment_t* environment, iree_vm_module_t** out_module,
    iree_allocator_t host_allocator);

// Source input format. Rendered text passes through unchanged; user content
// receives one user envelope and the assistant generation prefix.
typedef enum loom_serve_text_chat_input_format_e {
  LOOM_SERVE_TEXT_CHAT_INPUT_RENDERED = 0,
  LOOM_SERVE_TEXT_CHAT_INPUT_USER = 1,
} loom_serve_text_chat_input_format_t;

// Framing before the new text. Retained callers separately preserve the exact
// selected-but-unconsumed token ID; this operation never reconstructs that ID
// from the canonical transcript. TERMINATED means that ID already ends the
// prior turn; OPEN requires the source program to close it.
typedef enum loom_serve_text_chat_boundary_e {
  LOOM_SERVE_TEXT_CHAT_BOUNDARY_FRESH = 0,
  LOOM_SERVE_TEXT_CHAT_BOUNDARY_OPEN = 1,
  LOOM_SERVE_TEXT_CHAT_BOUNDARY_TERMINATED = 2,
} loom_serve_text_chat_boundary_t;

// Frames and encodes the complete input in the shared source VM. Text is
// borrowed only during this synchronous call. On success, copies at most
// capacity IDs into tokens; on failure, tokens is untouched and count is zero.
// Incomplete prefixes are rejected. No VM reference escapes into session state.
iree_status_t loom_serve_text_chat_prepare_input(
    const loom_serve_text_chat_policy_t* policy, iree_string_view_t text,
    loom_serve_text_chat_input_format_t format,
    loom_serve_text_chat_boundary_t boundary, iree_host_size_t capacity,
    int32_t* tokens, iree_host_size_t* out_count,
    iree_allocator_t host_allocator);

// Indexed function-tool schema borrowing the request body. Names are literal
// ASCII identifiers. The optional XML codec uses property types to recover
// generated JSON values; the agent owns execution and full schema validation.
typedef struct loom_serve_text_chat_tool_t {
  // Literal function name, without JSON quotes.
  iree_string_view_t name;
  // JSON object mapping parameter names to schemas.
  iree_string_view_t properties;
  // JSON array of required parameter names, or empty when absent.
  iree_string_view_t required;
} loom_serve_text_chat_tool_t;

enum loom_serve_text_chat_flag_bits_e {
  // Publish incremental SSE output instead of one JSON completion.
  LOOM_SERVE_TEXT_CHAT_FLAG_STREAM = 1u << 0,
  // Append a final usage chunk and include usage:null on preceding chunks.
  LOOM_SERVE_TEXT_CHAT_FLAG_INCLUDE_USAGE = 1u << 1,
};
typedef uint32_t loom_serve_text_chat_flags_t;

// One validated text-only, greedy chat request. The
// prompt owns its canonical rendered text; tool views borrow the HTTP body.
typedef struct loom_serve_text_chat_t {
  // Source policy borrowed from the model for the request lifetime.
  const loom_serve_text_chat_policy_t* policy;
  // Source-rendered canonical prompt ending at its generation boundary.
  iree_string_builder_t prompt;
  // Maximum selected output tokens, including a terminal EOS prediction.
  iree_host_size_t max_tokens;
  // Validated transport and usage options; inference is independent of these.
  loom_serve_text_chat_flags_t flags;
  // Original tool declarations borrowing the HTTP body, or the literal [].
  iree_string_view_t tool_schemas;
  // Number of populated tool descriptors.
  iree_host_size_t tool_count;
  // Validated tool descriptors in request order.
  loom_serve_text_chat_tool_t tools[LOOM_SERVE_TEXT_CHAT_TOOL_CAPACITY];
} loom_serve_text_chat_t;

// Validates and renders the supported chat request. Unknown execution options
// fail instead of pretending to implement unsupported execution modes. Source
// policy interprets model_options and enable_thinking from the original body.
// model_name is the expected deployment route, independent of policy->name.
// Body storage outlives the initialized chat. Deinitialize after success only;
// failure releases partial storage. The HTTP layer bounds the input body.
iree_status_t loom_serve_text_chat_initialize(
    const loom_serve_text_chat_policy_t* policy, iree_string_view_t model_name,
    iree_string_view_t body, iree_host_size_t default_max_tokens,
    iree_allocator_t host_allocator, loom_serve_text_chat_t* out_chat);
void loom_serve_text_chat_deinitialize(loom_serve_text_chat_t* chat);

// Source output phase. Final text releases incomplete marker prefixes; complete
// tool syntax remains withheld for structured parsing in either phase.
typedef enum loom_serve_text_chat_output_phase_e {
  LOOM_SERVE_TEXT_CHAT_OUTPUT_STREAMING = 0,
  LOOM_SERVE_TEXT_CHAT_OUTPUT_COMPLETE = 1,
} loom_serve_text_chat_output_phase_t;

// Queries the next safe text extent in the shared source VM. Response is
// borrowed synchronously. Scan starts at the previous safe extent and never
// retracts published bytes; source failure publishes no new extent.
iree_status_t loom_serve_text_chat_text_end(
    const loom_serve_text_chat_policy_t* policy, iree_string_view_t response,
    iree_host_size_t previous_end, loom_serve_text_chat_output_phase_t phase,
    iree_host_size_t* out_end, iree_allocator_t host_allocator);

// Completed response metadata and its canonical retained-history checkpoint.
// The source environment outlives this value; request/body/response storage
// need not. Original generated token IDs remain separately owned by the row.
typedef struct loom_serve_text_chat_completion_t {
  // Owned VM storage, including any backing aliased by the source result.
  iree_vm_buffer_t* storage;
  // Canonical transcript borrowing storage, not request or invocation memory.
  iree_string_view_t transcript;
  // Safe ordinary-text extent in the original generated response.
  iree_host_size_t text_end;
  // Number of validated structured tool calls.
  iree_host_size_t tool_count;
} loom_serve_text_chat_completion_t;

// Releases a completed checkpoint; zero-initialized values are accepted.
void loom_serve_text_chat_completion_deinitialize(
    loom_serve_text_chat_completion_t* completion);

// Invokes source parsing of the generated suffix into typed function calls.
// Returns the source-owned canonical completed transcript and appends an OpenAI
// tool_calls array to tool_calls. IDs are stable across both output forms;
// streaming calls also carry their delta indexes.
// previous_end is the already published ordinary-text extent in response.
// Plain text responses produce an empty array. Source parser failures never
// submit a partial executable call to the agent.
// Replaces completion only on success, releasing its prior storage. Failure
// leaves completion unchanged and may leave partial tool_calls bytes, which
// the caller must not publish. The checkpoint validates subsequent client
// history; it does not replace original generated tokens in device state.
iree_status_t loom_serve_text_chat_complete(
    const loom_serve_text_chat_t* chat, iree_string_view_t response,
    iree_host_size_t previous_end, uint64_t request_id,
    iree_string_builder_t* tool_calls,
    loom_serve_text_chat_completion_t* completion);

// Borrowed wire metadata copied from a session at each publication. Identity
// and creation time remain fixed across a response; counters include only
// committed tokens, never rejected speculative predictions.
typedef struct loom_serve_text_chat_response_t {
  // Deployment identity borrowed for the encoder call.
  iree_string_view_t model_name;
  // Server-assigned request identity shared by every chunk and tool call.
  uint64_t request_id;
  // Wall-clock admission time in nanoseconds; the wire uses whole seconds.
  iree_time_t creation_time;
  // Validated request output options.
  loom_serve_text_chat_flags_t flags;
  // Complete client transcript accounting, independent of output format.
  struct {
    // Total input tokens, including any retained prefix.
    iree_host_size_t input_tokens;
    // Input tokens reused from retained state.
    iree_host_size_t cached_tokens;
    // Selected output tokens, including a terminal EOS prediction.
    iree_host_size_t output_tokens;
  } usage;
} loom_serve_text_chat_response_t;

// Appends one SSE chat event. Delta is already serialized JSON; finish_reason
// is empty for nonterminal events. Requested usage is null until the final
// usage-only chunk, emitted by chat_usage.
iree_status_t loom_serve_text_chat_event(
    const loom_serve_text_chat_response_t* response, iree_string_view_t delta,
    iree_string_view_t finish_reason, iree_string_builder_t* output);
iree_status_t loom_serve_text_chat_usage(
    const loom_serve_text_chat_response_t* response,
    iree_string_builder_t* output);

// Appends a complete JSON chat response, without HTTP headers. Content is
// decoded text; tool_calls is a serialized array from chat_complete or empty
// when no tools were produced. Nonstreamed completions always include usage.
iree_status_t loom_serve_text_chat_response(
    const loom_serve_text_chat_response_t* response, iree_string_view_t content,
    iree_string_view_t tool_calls, iree_string_view_t finish_reason,
    iree_string_builder_t* output);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_EXPERIMENTAL_LOOM_SERVE_TEXT_CHAT_H_
