// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// loom-analyze report formats.
//
// Text output follows the GCC/Clang diagnostic convention
// (`file:line:col: warning: message [rule-id]` followed by `note:` lines), so
// editors and CI problem matchers annotate findings without configuration.
// SARIF output follows the OASIS SARIF 2.1.0 schema for code-scanning
// services and baseline tooling. JSON output is a compact tool-specific schema
// carrying the same data for scripts.
//
// Reporting granularity: a duplicate-sequence finding is one result anchored
// at its first occurrence with the remaining copies as related locations. A
// reimplemented-motif finding yields one result per copy, each relating back
// to the motif definition, because every copy is a separate edit site.

#ifndef LOOM_TOOLS_LOOM_ANALYZE_REPORT_H_
#define LOOM_TOOLS_LOOM_ANALYZE_REPORT_H_

#include "iree/base/api.h"
#include "loom/tools/loom-analyze/clone_analysis.h"
#include "loom/util/stream.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum loom_analyze_report_format_e {
  LOOM_ANALYZE_REPORT_FORMAT_TEXT = 0,
  LOOM_ANALYZE_REPORT_FORMAT_JSON = 1,
  LOOM_ANALYZE_REPORT_FORMAT_SARIF = 2,
} loom_analyze_report_format_t;

// Parses "text", "json" or "sarif".
iree_status_t loom_analyze_report_parse_format(
    iree_string_view_t value, loom_analyze_report_format_t* out_format);

// Returns the number of reported results: one per duplicate-sequence finding
// and one per reimplemented copy. Nonzero results select exit code 1.
iree_host_size_t loom_analyze_report_result_count(
    const loom_analyze_result_t* result);

// Writes |result| in |format| to |stream|. |tool_version| is recorded in JSON
// and SARIF metadata.
iree_status_t loom_analyze_report_write(const loom_analyze_result_t* result,
                                        const loom_analyze_options_t* options,
                                        loom_analyze_report_format_t format,
                                        iree_string_view_t tool_version,
                                        loom_output_stream_t* stream);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLS_LOOM_ANALYZE_REPORT_H_
