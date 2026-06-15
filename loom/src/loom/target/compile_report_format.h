// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Cold compile report formatting adapters.

#ifndef LOOM_TARGET_COMPILE_REPORT_FORMAT_H_
#define LOOM_TARGET_COMPILE_REPORT_FORMAT_H_

#include "iree/base/api.h"
#include "loom/target/compile_report.h"
#include "loom/util/stream.h"

#ifdef __cplusplus
extern "C" {
#endif

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
} loom_target_compile_report_format_options_t;

// Initializes text formatting options with report output disabled.
void loom_target_compile_report_format_options_initialize(
    loom_target_compile_report_format_options_t* out_options);

// Parses "", "none", "summary", or "details" into a text formatting mode.
iree_status_t loom_target_compile_report_format_mode_parse(
    iree_string_view_t value,
    loom_target_compile_report_format_mode_t* out_mode);

// Formats |report| as bounded line-oriented text into |builder|.
iree_status_t loom_target_compile_report_format_text(
    const loom_target_compile_report_t* report,
    const loom_target_compile_report_format_options_t* options,
    iree_string_builder_t* builder);

// Formats |report| as one structured JSON object into |stream|.
//
// SUMMARY mode emits stable summary fields and row counts. DETAILS mode
// additionally emits pressure, spill, source-low, and target-legalization row
// arrays. NONE mode writes nothing.
iree_status_t loom_target_compile_report_format_json(
    const loom_target_compile_report_t* report,
    const loom_target_compile_report_format_options_t* options,
    loom_output_stream_t* stream);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_COMPILE_REPORT_FORMAT_H_
