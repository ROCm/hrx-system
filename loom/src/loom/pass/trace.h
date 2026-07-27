// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Pass pipeline IR tracing.
//
// The pass interpreter calls this cold diagnostic boundary immediately before
// and after selected pass invocations. Callers own destination policy by
// providing an output stream; the trace layer owns stable event formatting,
// pass/stage matching, and IR snapshot printing.

#ifndef LOOM_PASS_TRACE_H_
#define LOOM_PASS_TRACE_H_

#include "iree/base/api.h"
#include "loom/format/text/printer.h"
#include "loom/ir/ir.h"
#include "loom/pass/program.h"
#include "loom/pass/types.h"
#include "loom/util/stream.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum loom_pass_trace_format_e {
  // Human-readable text with deterministic dividers and inline IR.
  LOOM_PASS_TRACE_FORMAT_TEXT = 0,
  // One JSON object per emitted boundary, newline-delimited.
  LOOM_PASS_TRACE_FORMAT_JSONL = 1,
} loom_pass_trace_format_t;

typedef enum loom_pass_trace_point_e {
  // Snapshot taken immediately before create/run for the selected pass.
  LOOM_PASS_TRACE_POINT_BEFORE = 0,
  // Snapshot taken immediately after create/run for the selected pass.
  LOOM_PASS_TRACE_POINT_AFTER = 1,
} loom_pass_trace_point_t;

typedef struct loom_pass_trace_event_t {
  // Module snapshot printed for this event.
  const loom_module_t* module;
  // Compiled invoke instruction associated with this event.
  const loom_pass_program_instruction_t* instruction;
  // Program instruction index for deterministic correlation with reports.
  iree_host_size_t instruction_index;
  // Runtime invocation ordinal shared by before/after events.
  iree_host_size_t invocation_ordinal;
  // Source pass.pipeline symbol associated with the invoke instruction.
  iree_string_view_t pipeline_symbol;
  // Current anchor symbol name, or "<none>" at module anchor.
  iree_string_view_t symbol_name;
  // Active anchor kind for the invocation.
  loom_pass_kind_t anchor_kind;
  // Boundary being emitted.
  loom_pass_trace_point_t point;
  // True when the just-finished invocation changed IR or semantic module state.
  bool changed;
  // Terminal status code for the invocation at this boundary.
  iree_status_code_t status_code;
  // Error diagnostics emitted by the invocation.
  uint32_t error_count;
  // Warning diagnostics emitted by the invocation.
  uint32_t warning_count;
  // Remark diagnostics emitted by the invocation.
  uint32_t remark_count;
} loom_pass_trace_event_t;

typedef struct loom_pass_trace_artifact_t {
  // Writable stream receiving the per-event IR artifact.
  loom_output_stream_t* stream;
  // Bundle-relative path to the artifact for index metadata.
  iree_string_view_t path;
} loom_pass_trace_artifact_t;

typedef struct loom_pass_trace_artifact_sink_t {
  // Opens a writable artifact stream for one emitted event.
  iree_status_t (*open)(void* user_data, const loom_pass_trace_event_t* event,
                        iree_host_size_t event_ordinal,
                        loom_pass_trace_artifact_t* out_artifact);
  // Closes the stream returned from |open|.
  iree_status_t (*close)(void* user_data, loom_pass_trace_artifact_t* artifact);
  // User data passed to open and close.
  void* user_data;
} loom_pass_trace_artifact_sink_t;

typedef struct loom_pass_trace_options_t {
  // Destination receiving formatted trace events. Required when tracing.
  loom_output_stream_t* stream;
  // Optional sink for per-event IR artifacts referenced from |stream|.
  loom_pass_trace_artifact_sink_t artifact_sink;
  // Trace event format.
  loom_pass_trace_format_t format;
  // Tool name included in human and JSONL metadata.
  iree_string_view_t tool_name;
  // Input identity included in human and JSONL metadata.
  iree_string_view_t input_path;
  // Stable caller-defined compile stage label matched by dump filters.
  iree_string_view_t stage;
  // Pass keys, pipeline symbols, or stage names to dump before.
  iree_string_view_list_t dump_before;
  // Pass keys, pipeline symbols, or stage names to dump after.
  iree_string_view_list_t dump_after;
  // Dumps every before boundary when true.
  bool dump_before_all;
  // Dumps every after boundary when true.
  bool dump_after_all;
  // Text printer options used for inline IR snapshots.
  loom_text_print_options_t print_options;
} loom_pass_trace_options_t;

typedef struct loom_pass_trace_t {
  // Immutable caller-owned trace configuration.
  const loom_pass_trace_options_t* options;
  // Number assigned to the next emitted event.
  iree_host_size_t next_event_ordinal;
} loom_pass_trace_t;

// Initializes trace options with human-readable text output and canonical IR
// printing. The caller must still provide a stream and dump requests.
void loom_pass_trace_options_initialize(loom_pass_trace_options_t* out_options);

// Parses a trace format name: empty/"text" or "jsonl".
iree_status_t loom_pass_trace_parse_format(
    iree_string_view_t value, loom_pass_trace_format_t* out_format);

// Returns true when |options| requests any trace boundary and has a stream.
bool loom_pass_trace_options_is_enabled(
    const loom_pass_trace_options_t* options);

// Initializes state for one pass pipeline execution.
void loom_pass_trace_initialize(const loom_pass_trace_options_t* options,
                                loom_pass_trace_t* out_trace);

// Emits one trace event when it matches the configured before/after filters.
iree_status_t loom_pass_trace_emit(loom_pass_trace_t* trace,
                                   const loom_pass_trace_event_t* event);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_PASS_TRACE_H_
