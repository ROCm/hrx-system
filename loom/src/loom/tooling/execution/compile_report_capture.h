// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Compile report capture helpers for one-shot execution tools.

#ifndef LOOM_TOOLING_EXECUTION_COMPILE_REPORT_CAPTURE_H_
#define LOOM_TOOLING_EXECUTION_COMPILE_REPORT_CAPTURE_H_

#include "iree/base/api.h"
#include "loom/target/compile_report_format.h"
#include "loom/tooling/execution/compile_options.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum loom_run_compile_report_sink_format_e {
  // Report capture and output are disabled.
  LOOM_RUN_COMPILE_REPORT_SINK_FORMAT_NONE = 0,
  // Emits structured JSON from typed report fields.
  LOOM_RUN_COMPILE_REPORT_SINK_FORMAT_JSON = 1,
  // Emits human-readable text for interactive debugging.
  LOOM_RUN_COMPILE_REPORT_SINK_FORMAT_TEXT = 2,
} loom_run_compile_report_sink_format_t;

typedef struct loom_run_compile_report_capture_options_t {
  // Output sink format requested by the caller.
  loom_run_compile_report_sink_format_t sink_format;
  // Report detail level captured and emitted by the sink.
  loom_target_compile_report_format_mode_t detail_mode;
} loom_run_compile_report_capture_options_t;

typedef struct loom_run_compile_report_capture_t {
  // Capture options used to configure and format |report|.
  loom_run_compile_report_capture_options_t options;
  // Host allocator used for optional detail rows.
  iree_allocator_t host_allocator;
  // Compile report populated by candidate compilation.
  loom_target_compile_report_t report;
} loom_run_compile_report_capture_t;

// Initializes capture options with report output disabled.
void loom_run_compile_report_capture_options_initialize(
    loom_run_compile_report_capture_options_t* out_options);

// Parses and stores a compile report request.
//
// Empty and "none" disable capture. "summary", "details", "json",
// "json-summary", and "json-details" request the structured JSON sink.
// "text", "text-summary", and "text-details" request the human-readable text
// adapter explicitly.
iree_status_t loom_run_compile_report_capture_options_parse_request(
    iree_string_view_t value,
    loom_run_compile_report_capture_options_t* options);

// Returns true when report capture and sink emission are enabled.
bool loom_run_compile_report_capture_options_is_enabled(
    const loom_run_compile_report_capture_options_t* options);

// Returns true when report capture and sink emission are enabled.
bool loom_run_compile_report_capture_is_enabled(
    const loom_run_compile_report_capture_t* capture);

// Initializes |out_capture| and its allocator-owned report.
iree_status_t loom_run_compile_report_capture_initialize(
    const loom_run_compile_report_capture_options_t* options,
    iree_allocator_t host_allocator,
    loom_run_compile_report_capture_t* out_capture);

// Configures |compile_options| to populate |capture| when capture is enabled.
void loom_run_compile_report_capture_configure_compile_options(
    loom_run_compile_report_capture_t* capture,
    loom_run_candidate_compile_options_t* compile_options);

// Appends the captured report to |builder| when capture is enabled.
iree_status_t loom_run_compile_report_capture_append_text(
    const loom_run_compile_report_capture_t* capture,
    iree_string_builder_t* builder);

// Appends the captured report as one JSON object when capture is enabled.
iree_status_t loom_run_compile_report_capture_append_json(
    const loom_run_compile_report_capture_t* capture,
    loom_output_stream_t* stream);

// Appends the captured report to |builder| using the configured sink format.
//
// This is for one-shot tools that already own a textual result buffer. JSON
// sinks append one structured object; text sinks append the human-readable
// adapter output. A separating newline is inserted when the builder already
// contains output.
iree_status_t loom_run_compile_report_capture_append_output(
    const loom_run_compile_report_capture_t* capture,
    iree_string_builder_t* builder);

// Writes the captured report to |stream| using the configured sink format.
iree_status_t loom_run_compile_report_capture_write_output(
    const loom_run_compile_report_capture_t* capture,
    loom_output_stream_t* stream, iree_allocator_t host_allocator);

// Releases storage owned by |capture|.
void loom_run_compile_report_capture_deinitialize(
    loom_run_compile_report_capture_t* capture);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLING_EXECUTION_COMPILE_REPORT_CAPTURE_H_
