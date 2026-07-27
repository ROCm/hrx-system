// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Diagnostic infrastructure for loom subsystems (parser, verifier, bytecode).
//
// Diagnostics carry severity, source location (with byte range for caret
// underlining), and structured error identity. They are emitted through a
// sink callback that the caller provides — the sink decides rendering.
// Subsystems that want the caller to materialize diagnostics use the
// lighter iree_diagnostic_emitter_t from emitter.h instead.
//
// The default stderr sink produces Clang-style caret diagnostics:
//
//   test.loom:42:15: error [PARSE/001]: undefined SSA value '%y'
//    42 |   %r = test.addi %x, %y : i32
//       |                      ^^
//
// Alternative sinks can collect diagnostics into a list (for testing),
// emit JSON (for agent/IDE consumption), or route to a logging framework.

#ifndef LOOM_ERROR_DIAGNOSTIC_H_
#define LOOM_ERROR_DIAGNOSTIC_H_

#include "iree/base/api.h"
#include "loom/error/emitter.h"
#include "loom/error/error_defs.h"
#include "loom/error/renderer.h"
#include "loom/util/stream.h"

#ifdef __cplusplus
extern "C" {
#endif

// Identifies which bytes back a diagnostic source range.
typedef enum loom_source_provenance_t {
  // The range's source bytes come from the original user-authored input text.
  LOOM_SOURCE_PROVENANCE_EXACT_SOURCE = 0,
  // The range's source bytes come from a canonical printed IR fallback.
  LOOM_SOURCE_PROVENANCE_PRINTED_IR_FALLBACK = 1,
  // The range points at a logical location whose original source bytes are not
  // available in this process.
  LOOM_SOURCE_PROVENANCE_UNAVAILABLE_SOURCE = 2,
} loom_source_provenance_t;

static inline const char* loom_source_provenance_name(
    loom_source_provenance_t provenance) {
  switch (provenance) {
    case LOOM_SOURCE_PROVENANCE_EXACT_SOURCE:
      return "exact_source";
    case LOOM_SOURCE_PROVENANCE_PRINTED_IR_FALLBACK:
      return "printed_ir_fallback";
    case LOOM_SOURCE_PROVENANCE_UNAVAILABLE_SOURCE:
      return "unavailable_source";
    default:
      return "exact_source";
  }
}

// A source range identifying a span of text in a source buffer.
// Both offsets are byte positions into |source|. The range is
// [start, end) — end is one past the last byte.
typedef struct loom_source_range_t {
  loom_source_provenance_t provenance;
  iree_string_view_t filename;
  iree_string_view_t source;
  iree_host_size_t start;
  iree_host_size_t end;
  uint32_t start_line;
  uint32_t start_column;
  uint32_t end_line;
  uint32_t end_column;
} loom_source_range_t;

// Constructs a source range from a token's position and source spelling.
static inline loom_source_range_t loom_source_range_from_token(
    iree_string_view_t filename, iree_string_view_t source,
    iree_string_view_t token_source_text, uint32_t line, uint32_t column,
    uint32_t end_column) {
  iree_host_size_t start =
      (iree_host_size_t)(token_source_text.data - source.data);
  return (loom_source_range_t){
      /*.provenance=*/LOOM_SOURCE_PROVENANCE_EXACT_SOURCE,
      /*.filename=*/filename,
      /*.source=*/source,
      /*.start=*/start,
      /*.end=*/start + token_source_text.size,
      /*.start_line=*/line,
      /*.start_column=*/column,
      /*.end_line=*/line,
      /*.end_column=*/end_column,
  };
}

// A byte range within a source buffer for per-token highlighting.
// All offsets are into the same source buffer as the diagnostic's
// primary range. Used for multi-caret diagnostics where multiple
// tokens on the same source line need underlines.
typedef struct loom_highlight_range_t {
  iree_host_size_t start;  // Byte offset into source.
  iree_host_size_t end;    // One past last byte.
  // Optional structured field ref for the highlighted token.
  loom_diagnostic_field_ref_t field_ref;
  // Index into diagnostic->params for the parameter this highlight came from.
  // Only meaningful when field_ref is set and less than param_count.
  iree_host_size_t param_index;
} loom_highlight_range_t;

// A single diagnostic: severity + location + structured error.
//
// Maximum number of labeled related locations an emitter should attach to one
// diagnostic. This keeps stack-owned note buffers bounded at emission sites.
#define LOOM_DIAGNOSTIC_MAX_RELATED_LOCATIONS 4

// A labeled secondary source location that gives context for the primary
// diagnostic site, such as the op that consumed a tied operand or the
// declaration that introduced a symbol. As with `loom_diagnostic_t`, the
// payload is only valid for the duration of the sink callback and sinks must
// copy any data they retain.
typedef struct loom_diagnostic_related_location_t {
  // Human-readable note label identifying why this location is relevant.
  iree_string_view_t label;
  // Source range for the related note.
  loom_source_range_t source_location;
  // Optional per-token highlights within source_location.
  const loom_highlight_range_t* highlights;
  // Number of entries in highlights.
  iree_host_size_t highlight_count;
} loom_diagnostic_related_location_t;

// Every diagnostic carries a structured error definition and typed
// parameters. Sinks render the message from the error_def's template
// and the runtime params — the diagnostic itself carries no pre-rendered
// message. This ensures JSON sinks get structured data and text sinks
// get consistent formatting without duplicate work.
typedef struct loom_diagnostic_t {
  loom_diagnostic_severity_t severity;

  // Structured error identity and parameters. Always non-NULL.
  const loom_error_def_t* error;
  const loom_diagnostic_param_t* params;
  iree_host_size_t param_count;

  // Which subsystem emitted this diagnostic.
  loom_emitter_t emitter;

  // Where the error was detected (byte range in source text or bytecode).
  // The text sink uses this for caret rendering.
  loom_source_range_t origin;

  // Where the user should look to fix it. Same as origin when there
  // is no location override (typical for the text parser). Differs
  // from origin when an op carries a loc() that points elsewhere
  // (bytecode reader, verifier with source resolver).
  loom_source_range_t source_location;

  // Per-token highlight ranges for multi-caret diagnostics. When
  // non-NULL, the caret formatter draws underlines at each range
  // instead of underlining the full primary range. All ranges must
  // be within the same source line as origin.
  const loom_highlight_range_t* highlights;
  // Number of entries in highlights.
  iree_host_size_t highlight_count;
  // Number of requested source-resolved highlights omitted due to emitter
  // storage limits.
  iree_host_size_t highlight_omitted_count;

  // Optional labeled related locations that provide secondary context.
  const loom_diagnostic_related_location_t* related_locations;
  // Number of entries in related_locations. Text formatting prints each entry
  // as a follow-on note, and JSON serializes them as a machine-readable array.
  iree_host_size_t related_location_count;
  // Number of source-resolved related locations omitted due to emitter storage
  // limits.
  iree_host_size_t related_location_omitted_count;
} loom_diagnostic_t;

// Callback invoked for each diagnostic. The diagnostic is valid only
// for the duration of the call — the sink must copy any data it wants
// to keep. Returns iree_ok_status() to continue parsing, or an error
// to abort immediately.
typedef iree_status_t (*loom_diagnostic_fn_t)(
    void* user_data, const loom_diagnostic_t* diagnostic);

// A diagnostic sink: callback + context. Passed to the parser at
// creation time. The parser emits all diagnostics through this sink.
typedef struct loom_diagnostic_sink_t {
  loom_diagnostic_fn_t fn;
  void* user_data;
} loom_diagnostic_sink_t;

// Emits a diagnostic through the sink. If the sink is NULL or its fn
// is NULL, the diagnostic is silently dropped.
static inline iree_status_t loom_diagnostic_emit(
    const loom_diagnostic_sink_t* sink, const loom_diagnostic_t* diagnostic) {
  if (sink && sink->fn) {
    return sink->fn(sink->user_data, diagnostic);
  }
  return iree_ok_status();
}

// Options controlling human-readable diagnostic formatting.
typedef struct loom_diagnostic_format_options_t {
  // Type formatter used for LOOM_PARAM_TYPE values in messages and fix hints.
  loom_type_formatter_t type_formatter;
} loom_diagnostic_format_options_t;

// Initializes diagnostic format options with the default minimal type
// formatter.
static inline void loom_diagnostic_format_options_initialize(
    loom_diagnostic_format_options_t* out_options) {
  *out_options = (loom_diagnostic_format_options_t){
      /*.type_formatter=*/{loom_type_format_minimal, NULL},
  };
}

// Formats a diagnostic in Rust/Clang-style caret format. The output
// flows through a loom_output_stream_t:
//
//   error: undefined value '%y'
//     --> model.loom:42:15
//      |
//   42 |   %r = test.addi %x, %y : i32
//      |                      ^^
//
iree_status_t loom_diagnostic_format(const loom_diagnostic_t* diagnostic,
                                     loom_output_stream_t* stream);

// Formats a diagnostic using explicit options. A NULL options pointer uses the
// same defaults as loom_diagnostic_format().
iree_status_t loom_diagnostic_format_with_options(
    const loom_diagnostic_t* diagnostic,
    const loom_diagnostic_format_options_t* options,
    loom_output_stream_t* stream);

// A diagnostic sink that prints to stderr using loom_diagnostic_format.
// Pass NULL as user_data.
iree_status_t loom_diagnostic_stderr_sink(void* user_data,
                                          const loom_diagnostic_t* diagnostic);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_ERROR_DIAGNOSTIC_H_
