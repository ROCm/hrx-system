// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Cold compile report formatting adapters.

#ifndef LOOM_TARGET_REPORTING_FORMAT_H_
#define LOOM_TARGET_REPORTING_FORMAT_H_

#include "iree/base/api.h"
#include "loom/target/reporting/report.h"
#include "loom/util/stream.h"

#ifdef __cplusplus
extern "C" {
#endif

// Compile report schema version emitted by the JSON formatter.
//
// Version zero is an unstable same-compiler-time-horizon contract. Consumers
// must require exact equality and are not expected to migrate older reports.
#define LOOM_TARGET_COMPILE_REPORT_SCHEMA_VERSION 0u

typedef enum loom_target_compile_report_format_mode_e {
  // Does not format a compile report.
  LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_NONE = 0,
  // Formats one bounded summary block without per-row details.
  LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_SUMMARY = 1,
  // Formats the summary block plus copied pressure and spill rows.
  LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_DETAILS = 2,
} loom_target_compile_report_format_mode_t;

typedef struct loom_target_compile_report_format_options_t {
  // Selected report detail mode.
  loom_target_compile_report_format_mode_t mode;
  // Diagnostic detail available to report formatters.
  struct {
    // Canonical diagnostic JSON object bodies retained for detail reports.
    iree_string_view_t json_objects;
    // Total diagnostics observed, including those without retained JSON.
    iree_host_size_t count;
  } diagnostics;
} loom_target_compile_report_format_options_t;

// Initializes text formatting options with report output disabled.
void loom_target_compile_report_format_options_initialize(
    loom_target_compile_report_format_options_t* out_options);

// Parses "", "none", "summary", or "details" into a text formatting mode.
iree_status_t loom_target_compile_report_format_mode_parse(
    iree_string_view_t value,
    loom_target_compile_report_format_mode_t* out_mode);

// Returns the stable JSON spelling for |mode|.
iree_string_view_t loom_target_compile_report_format_mode_name(
    loom_target_compile_report_format_mode_t mode);

// Formats |report| as bounded line-oriented text into |builder|.
iree_status_t loom_target_compile_report_format_text(
    const loom_target_compile_report_t* report,
    const loom_target_compile_report_format_options_t* options,
    iree_string_builder_t* builder);

// Formats |report| as one structured JSON object into |stream|.
//
// SUMMARY mode emits stable summary fields, row counts, and the entry index.
// DETAILS mode additionally emits copied row arrays such as pressure, spill,
// wait-plan, source-low, and target-legalization rows. NONE mode writes
// nothing.
iree_status_t loom_target_compile_report_format_json(
    const loom_target_compile_report_t* report,
    const loom_target_compile_report_format_options_t* options,
    loom_output_stream_t* stream);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_REPORTING_FORMAT_H_
