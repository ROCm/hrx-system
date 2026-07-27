// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Shared command-line IR tracing flags for Loom tools.

#ifndef LOOM_TOOLING_PASS_TRACE_CLI_H_
#define LOOM_TOOLING_PASS_TRACE_CLI_H_

#include "iree/base/api.h"
#include "loom/pass/trace.h"
#include "loom/tooling/io/file.h"

#ifdef __cplusplus
extern "C" {
#endif

#define LOOM_TOOLING_PASS_TRACE_USAGE                                      \
  "Use --dump-ir-before/--dump-ir-after=<pass-or-stage> or "               \
  "--dump-ir-before-all/--dump-ir-after-all to capture pass-boundary IR. " \
  "Trace events snapshot the whole module, even for function passes. Use " \
  "--dump-ir-format=jsonl for one event per line; "                        \
  "--dump-ir-output=dir/ writes trace.jsonl plus ir/*.loom.\n"

typedef struct loom_tooling_pass_trace_stdout_conflict_t {
  // True when this output path is used by the active tool invocation.
  bool active;
  // User-visible flag name that owns the potentially stdout-backed path.
  iree_string_view_t flag_name;
  // Output path selected for |flag_name|.
  iree_string_view_t path;
} loom_tooling_pass_trace_stdout_conflict_t;

typedef struct loom_tooling_pass_trace_open_options_t {
  // Tool name included in trace metadata.
  iree_string_view_t tool_name;
  // Input identity included in trace metadata.
  iree_string_view_t input_path;
  // Descriptor-set key selected for low asm region printing.
  iree_string_view_t low_asm_descriptor_set_key;
  // Active outputs that cannot share stdout with dump output.
  const loom_tooling_pass_trace_stdout_conflict_t* stdout_conflicts;
  // Number of entries in stdout_conflicts.
  iree_host_size_t stdout_conflict_count;
} loom_tooling_pass_trace_open_options_t;

typedef struct loom_tooling_pass_trace_t {
  // Open output destination backing pass_options.stream when enabled.
  loom_tooling_output_stream_t output;
  // Allocator used for owned bundle paths.
  iree_allocator_t host_allocator;
  // Owned bundle root directory, or NULL for stream/file output.
  char* bundle_directory;
  // Owned bundle IR artifact directory, or NULL for stream/file output.
  char* bundle_ir_directory;
  // Owned bundle index path backing |output.path|.
  char* bundle_index_path;
  // Owned currently-open artifact path backing artifact_output.path.
  char* bundle_artifact_path;
  // Owned bundle-relative currently-open artifact path.
  char* bundle_artifact_relative_path;
  // Open artifact output while a bundle event is being emitted.
  loom_tooling_output_stream_t bundle_artifact_output;
  // Pass trace options configured from the shared CLI flags.
  loom_pass_trace_options_t pass_options;
  // True when at least one dump flag was requested and output is open.
  bool enabled;
} loom_tooling_pass_trace_t;

// Returns true when any shared dump flag requests IR tracing.
bool loom_tooling_pass_trace_flags_requested(void);

// Opens pass tracing from the shared dump flags. No output is opened when no
// dump flag is requested.
iree_status_t loom_tooling_pass_trace_open_from_flags(
    const loom_tooling_pass_trace_open_options_t* options,
    iree_allocator_t allocator, loom_tooling_pass_trace_t* out_trace);

// Flushes or closes the trace output when tracing was enabled.
iree_status_t loom_tooling_pass_trace_close(loom_tooling_pass_trace_t* trace);

// Returns pass trace options when tracing is enabled, otherwise NULL.
const loom_pass_trace_options_t* loom_tooling_pass_trace_options(
    const loom_tooling_pass_trace_t* trace);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLING_PASS_TRACE_CLI_H_
