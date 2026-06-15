// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// JSON output for loom-check --json mode.
//
// Writes a structured JSON object for each processed file, containing
// per-case outcomes, the file's default mode, and a summary of counts.
// All string values are properly escaped via loom/util/json.h. Output
// is written directly to a loom_output_stream_t — no intermediate
// buffer, no file I/O (the caller decides where the stream goes).
//
// Output schema:
//
//   {
//     "file": "<filename>",
//     "default_mode": "roundtrip",
//     "default_pipeline": null,
//     "default_format_target": null,
//     "default_emit_target": null,
//     "default_requirements": [],
//     "cases": [
//       {
//         "index": 1,
//         "mode": "roundtrip",
//         "has_run_directive": true,
//         "has_requires_directive": false,
//         "requires_directive_range": null,
//         "requirements": [],
//         "pipeline": null,
//         "format_target": null,
//         "emit_target": null,
//         "source_range": {"start_byte": 0, "end_byte": 42},
//         "separator_range": null,
//         "run_directive_range": {"start_byte": 0, "end_byte": 17},
//         "xfail": false,
//         "xfail_directive_range": null,
//         "xfail_reason": null,
//         "raw_outcome": "pass",
//         "final_outcome": "pass",
//         "detail": "",
//         "diff": null,
//         "update_edit": null,
//         "annotation_edits": [
//           {
//             "kind": "insert_diagnostic_annotations",
//             "range": {"start_byte": 18, "end_byte": 18},
//             "target_line": 2,
//             "text": "// ERROR@+1: PARSE/006\n"
//           }
//         ],
//         "input_range": {"start_byte": 18, "end_byte": 42},
//         "expected_separator_range": null,
//         "expected_range": null,
//         "has_expected_section": false,
//         "annotations": [],
//         "diagnostics": []
//       }
//     ],
//     "summary": {
//       "total": 1,
//       "passed": 1,
//       "failed": 0,
//       "skipped": 0
//     }
//   }

#ifndef LOOM_TOOLS_LOOM_CHECK_JSON_OUTPUT_H_
#define LOOM_TOOLS_LOOM_CHECK_JSON_OUTPUT_H_

#include "iree/base/api.h"
#include "loom/tools/loom-check/check.h"
#include "loom/tools/loom-check/execute.h"
#include "loom/tools/loom-check/report.h"
#include "loom/util/stream.h"

#ifdef __cplusplus
extern "C" {
#endif

// Controls how much per-case detail is included in the JSON output. The summary
// is always emitted.
typedef enum loom_check_json_output_mode_t {
  // Include only failing cases. This is the default CLI mode for --json.
  LOOM_CHECK_JSON_OUTPUT_FAILURES = 0,
  // Omit all cases and emit the summary only.
  LOOM_CHECK_JSON_OUTPUT_SUMMARY = 1,
  // Include every case, including passing cases and their diagnostics.
  LOOM_CHECK_JSON_OUTPUT_ALL = 2,
} loom_check_json_output_mode_t;

// Writes the JSON output for a processed file. The caller provides all
// results and the stream to write to. Does not do file I/O — the
// caller can direct the stream to stdout, a string builder, or wherever.
//
// |filename| appears as the "file" field. |file| provides parsed case and
// annotation metadata. |report| provides per-run annotation match state.
// |results| is a parallel array of per-case outcomes; it may be NULL only when
// |file| has no cases. The pass/fail/skip counts are provided directly by the
// caller (rather than recomputed) to stay consistent with the text output path.
iree_status_t loom_check_json_write_file_result(
    iree_string_view_t filename, const loom_check_file_t* file,
    const loom_check_file_report_t* report, const loom_check_result_t* results,
    iree_host_size_t pass_count, iree_host_size_t fail_count,
    iree_host_size_t skip_count, loom_check_json_output_mode_t output_mode,
    loom_output_stream_t* stream);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_TOOLS_LOOM_CHECK_JSON_OUTPUT_H_
