// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_SERVICE_TEXT_CHAT_H_
#define LOOM_SERVE_SERVICE_TEXT_CHAT_H_

#include "iree/base/api.h"
#include "iree/vm/buffer.h"
#include "loom/serve/runtime/program/program.h"
#include "loom/serve/service/text/media.h"

#ifdef __cplusplus
extern "C" {
#endif

// Protocol output budget bound, independent of a deployment's context limit.
#define LOOM_SERVE_TEXT_CHAT_MAX_OUTPUT_TOKENS 16384

// Protocol resource bound for declared or generated function tools. Storage is
// allocated only for the declarations present in a request.
#define LOOM_SERVE_TEXT_CHAT_MAX_TOOL_COUNT 4096

// Protocol resource bound for ordered content parts in one message. The HTTP
// body has an independent byte bound; this limits the compact source-policy
// record set created from pathologically small JSON objects.
#define LOOM_SERVE_TEXT_CHAT_MAX_CONTENT_PART_COUNT 4096

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
  // Owned source-validated deployment defaults, opaque to native code.
  iree_vm_buffer_t* defaults;
  // Resolves request JSON and nonce against defaults into owned opaque bytes.
  iree_vm_function_t request_options;
  // Source-declared image input policy. A zero count disables image input;
  // enabled policies provide complete codec allocation bounds.
  struct {
    // Maximum image occurrences in one request.
    iree_host_size_t maximum_count;
    // Per-image compressed-byte and encoded-geometry bounds.
    loom_serve_media_image_limits_t limits;
  } images;
  // Initial prompt fragment and opaque rendering state.
  iree_vm_function_t chat_begin;
  // Message fragment and next opaque rendering state.
  iree_vm_function_t chat_message;
  // Final prompt fragment, conversation validation and initial output state.
  iree_vm_function_t chat_end;
  // Generated suffix to standard function-call records.
  iree_vm_function_t parse_tools;
  // Canonical tool formatter shared by input history and generated output.
  iree_vm_function_t render_tool;
  // Turn framing and complete raw-token encoding shared by all text callers.
  iree_vm_function_t prepare_input;
  // Next typed output span and opaque parser state.
  iree_vm_function_t output_next;
  // complete_text(prompt, content, reasoning, tools, initial_state,
  // final_state) returns (owned transcript buffer, reusable i32 boolean). False
  // requires replay instead of matching the canonical transcript to retained
  // device state.
  iree_vm_function_t complete_text;
} loom_serve_text_chat_policy_t;

// Calls request_defaults(options) once and retains its opaque result. Options
// has owned backing that source may retain. The program/environment outlive
// the policy and every chat initialized from it.
iree_status_t loom_serve_text_chat_policy_initialize(
    iree_vm_environment_t* environment, loom_serve_program_t* program,
    iree_vm_buffer_t* options, loom_serve_text_chat_policy_t* out_policy);

// Releases the owned identity/defaults; functions borrow the model program.
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
    iree_vm_environment_t* environment, iree_allocator_t host_allocator,
    iree_vm_module_t** out_module);

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

// Frames and encodes the rendered prompt suffix beginning at prompt_offset in
// the shared source VM. Source receives the exact checked marker plan and
// returns token IDs plus one logical input span per included image. On success,
// copies at most capacity IDs into tokens and allocates any media spans; on
// failure all outputs are empty and tokens is untouched. input_shift accounts
// for a retained prediction prepended by native code. No VM reference escapes.
iree_status_t loom_serve_text_chat_prepare_input(
    const loom_serve_text_chat_t* chat, iree_host_size_t prompt_offset,
    loom_serve_text_chat_input_format_t format,
    loom_serve_text_chat_boundary_t boundary, iree_host_size_t capacity,
    iree_host_size_t input_shift, iree_allocator_t host_allocator,
    int32_t* out_tokens, iree_host_size_t* out_count,
    loom_serve_text_media_span_t** out_media_spans,
    iree_host_size_t* out_media_span_count);

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

// One validated chat request. The prompt owns its canonical rendered
// text; tool and encoded-image views borrow the HTTP body.
typedef struct loom_serve_text_chat_t {
  // Source policy borrowed from the model for the request lifetime.
  const loom_serve_text_chat_policy_t* policy;
  // Owned source-validated request parameters, copied to model state at bind.
  iree_vm_buffer_t* options;
  // Source-rendered canonical prompt ending at its generation boundary.
  iree_string_builder_t prompt;
  // Incremental output owned by this request, independent of transport mode.
  struct {
    // Source state at the generation boundary, used for canonical history.
    int64_t initial_state;
    // Source-private current parser state.
    int64_t state;
    // Consumed byte frontier in the accumulated decoded response.
    iree_host_size_t cursor;
    // Answer bytes without source framing.
    iree_string_builder_t content;
    // Reasoning bytes without source framing.
    iree_string_builder_t reasoning;
  } output;
  // Maximum selected output tokens, including a terminal EOS prediction.
  iree_host_size_t max_tokens;
  // Validated transport and usage options; inference is independent of these.
  loom_serve_text_chat_flags_t flags;
  // Original tool declarations borrowing the HTTP body, or the literal [].
  iree_string_view_t tool_schemas;
  // Host allocator owning the tool descriptor array.
  iree_allocator_t host_allocator;
  // Number of populated tool descriptors.
  iree_host_size_t tool_count;
  // Allocated descriptor capacity.
  iree_host_size_t tool_capacity;
  // Dynamically sized validated tool descriptors sorted by literal name.
  loom_serve_text_chat_tool_t* tools;
  // Ordered inspected image occurrences.
  struct {
    // Number of populated image descriptors.
    iree_host_size_t count;
    // Allocated descriptor capacity.
    iree_host_size_t capacity;
    // Dynamically sized descriptors in message/part order.
    loom_serve_text_chat_image_t* values;
    // Owned concatenated source-private preprocessing metadata.
    iree_string_builder_t metadata;
  } images;
} loom_serve_text_chat_t;

// Validates and renders the supported chat request. Unknown execution options
// fail instead of pretending to implement unsupported execution modes. Source
// policy interprets sampling, model_options and enable_thinking from the
// original body. request_id is a unique nonce for source-owned default seeds;
// an explicit seed can override it independently of row or session identity.
// model_name is the expected deployment route, independent of policy->name.
// Body storage outlives the initialized chat. Deinitialize after success only;
// failure releases partial storage. The HTTP layer bounds the input body.
iree_status_t loom_serve_text_chat_initialize(
    const loom_serve_text_chat_policy_t* policy, iree_string_view_t model_name,
    iree_string_view_t body, uint64_t request_id,
    iree_host_size_t default_max_tokens, iree_allocator_t host_allocator,
    loom_serve_text_chat_t* out_chat);
void loom_serve_text_chat_deinitialize(loom_serve_text_chat_t* chat);

// Source output phase. Final text releases incomplete marker prefixes; complete
// tool syntax remains withheld for structured parsing in either phase.
typedef enum loom_serve_text_chat_output_phase_e {
  LOOM_SERVE_TEXT_CHAT_OUTPUT_STREAMING = 0,
  LOOM_SERVE_TEXT_CHAT_OUTPUT_COMPLETE = 1,
} loom_serve_text_chat_output_phase_t;

// Model-neutral output channels. Hidden spans consume framing without exposing
// it on the wire. Source controls all model-specific delimiters and
// transitions.
typedef enum loom_serve_text_chat_output_kind_e {
  LOOM_SERVE_TEXT_CHAT_OUTPUT_HIDDEN = 0,
  LOOM_SERVE_TEXT_CHAT_OUTPUT_CONTENT = 1,
  LOOM_SERVE_TEXT_CHAT_OUTPUT_REASONING = 2,
} loom_serve_text_chat_output_kind_t;

typedef struct loom_serve_text_chat_output_t {
  // True if this call consumed bytes; false means wait or parse the tool
  // suffix.
  bool advanced;
  // Destination of text; hidden spans have no visible bytes.
  loom_serve_text_chat_output_kind_t kind;
  // Borrowed response bytes, valid until the caller changes response storage.
  iree_string_view_t text;
} loom_serve_text_chat_output_t;

// Consumes one source-defined span and appends it to the request's channel.
// output_next(response, cursor, state, phase) returns (kind, begin, end,
// state). Bounds are cursor <= begin <= end <= response.size; end advances the
// cursor, while [cursor, begin) is framing. No progress leaves parser state
// unchanged. Response is borrowed synchronously; published bytes are never
// retracted.
iree_status_t loom_serve_text_chat_output_next(
    loom_serve_text_chat_t* chat, iree_string_view_t response,
    loom_serve_text_chat_output_phase_t phase,
    loom_serve_text_chat_output_t* out_output);

// Completed response metadata and its canonical retained-history checkpoint.
// The source environment outlives this value; request/body/response storage
// need not. Original generated token IDs remain separately owned by the row.
typedef struct loom_serve_text_chat_completion_t {
  // Owned VM storage, including any backing aliased by the source result.
  iree_vm_buffer_t* storage;
  // Canonical transcript borrowing storage, not request or invocation memory.
  iree_string_view_t transcript;
  // Source authorization to reuse device state for this canonical transcript.
  // A truncated source construct can require replay despite valid wire output.
  bool reusable;
  // Exact ordered media identity contributing to this completed history.
  loom_serve_text_chat_media_identity_t media;
  // Number of validated structured tool calls.
  iree_host_size_t tool_count;
} loom_serve_text_chat_completion_t;

// Releases a completed checkpoint; zero-initialized values are accepted.
void loom_serve_text_chat_completion_deinitialize(
    loom_serve_text_chat_completion_t* completion);

// Copies a completed checkpoint reference. Source and target may be released
// independently; transcript and media remain immutable.
void loom_serve_text_chat_completion_retain(
    const loom_serve_text_chat_completion_t* source,
    loom_serve_text_chat_completion_t* out_completion);

// Whether the request extends both the canonical transcript and exact ordered
// media identity of a completed checkpoint.
bool loom_serve_text_chat_completion_is_prefix(
    const loom_serve_text_chat_completion_t* completion,
    const loom_serve_text_chat_t* chat);

// Invokes source parsing of the generated suffix into typed function calls.
// Returns the source-owned canonical completed transcript and appends an OpenAI
// tool_calls array to tool_calls. IDs are stable across both output forms;
// streaming calls also carry their delta indexes.
// Finishes output parsing before translating the remaining structured suffix.
// Plain text responses produce an empty array. Source parser failures never
// submit a partial executable call to the agent.
// Replaces completion only on success, releasing its prior storage. Failure
// leaves completion unchanged and may leave partial tool_calls bytes, which
// the caller must not publish. The checkpoint validates subsequent client
// history; it does not replace original generated tokens in device state.
iree_status_t loom_serve_text_chat_complete(
    loom_serve_text_chat_t* chat, iree_string_view_t response,
    uint64_t request_id, iree_string_builder_t* tool_calls,
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
// decoded answer text; reasoning is independently decoded thought text and is
// omitted from the wire when empty. tool_calls is a serialized array or empty
// when no tools were produced. Nonstreamed completions always include usage.
iree_status_t loom_serve_text_chat_response(
    const loom_serve_text_chat_response_t* response, iree_string_view_t content,
    iree_string_view_t reasoning, iree_string_view_t tool_calls,
    iree_string_view_t finish_reason, iree_string_builder_t* output);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVE_SERVICE_TEXT_CHAT_H_
