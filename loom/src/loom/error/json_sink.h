// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// JSON diagnostic sink for loom diagnostics.
//
// Emits one JSON object per diagnostic (JSONL format — one object per line)
// to a caller-provided loom_output_stream_t. Every diagnostic is
// structured: all error identity fields, source ranges, labeled related
// locations, typed parameters, rendered message, and fix hints are included.
// The message is rendered from the error def's template.
//
// Output example:
//
//   {"severity":"error","error_id":"ERR_TYPE_001","domain":"TYPE","code":1,
//    "summary":"SameType constraint violated.","emitter":"verifier",
//    "origin":{"provenance":"exact_source","filename":"model.loom",
//              "start_line":42,"start_column":15,
//              "end_line":42,"end_column":17,"start_byte":128,
//              "end_byte":130,
//              "excerpt":{"start_byte":120,"end_byte":140,
//                         "truncated_prefix":false,
//                         "truncated_suffix":false,
//                         "text":"  %x = test.addi ..."}},
//    "source_location":{"provenance":"exact_source","filename":"model.loom",
//                       "start_line":42,"start_column":15,"end_line":42,
//                       "end_column":17,"start_byte":128,"end_byte":130,
//                       "excerpt":{"start_byte":120,"end_byte":140,
//                                  "truncated_prefix":false,
//                                  "truncated_suffix":false,
//                                  "text":"  %x = test.addi ..."}},
//    "highlights":[{"start_byte":128,"end_byte":130,
//                   "field":{"kind":"operand","index":1,"occurrence":0},
//                   "param":"field_b"}],
//    "highlight_omitted_count":1,
//    "related_locations":[
//      {"label":"consumed here",
//       "source_location":{"provenance":"exact_source",
//                          "filename":"model.loom",
//                          "start_line":41,"start_column":3,
//                          "end_line":41,"end_column":31,
//                          "start_byte":96,"end_byte":124,
//                          "excerpt":{"start_byte":96,"end_byte":124,
//                                     "truncated_prefix":false,
//                                     "truncated_suffix":false,
//                                     "text":"  %y = test.invoke @f(%rhs)"}},
//       "highlights":[{"start_byte":115,"end_byte":119,
//                      "field":{"kind":"operand","index":0,
//                               "occurrence":1}}]}],
//    "related_location_omitted_count":2,
//    "message":"'rhs' type f32 does not match 'lhs' type i32",
//    "fix_hint":"Ensure 'rhs' and 'lhs' have the same type",
//    "params":{"field_a":"lhs","type_a":"i32","field_b":"rhs","type_b":"f32"},
//    "param_fields":{"field_a":{"kind":"operand","index":0,"occurrence":0},
//                    "field_b":{"kind":"operand","index":1,
//                               "occurrence":0}}}
//
// Usage:
//
//   iree_string_builder_t json_output;
//   iree_string_builder_initialize(allocator, &json_output);
//   loom_output_stream_t stream;
//   loom_output_stream_for_builder(&json_output, &stream);
//   loom_json_sink_options_t json_options = {
//       .stream = &stream,
//       .type_formatter = {loom_type_format_minimal, NULL},
//   };
//   loom_diagnostic_sink_t sink = {
//       .fn = loom_diagnostic_json_sink,
//       .user_data = &json_options,
//   };
//   // ... pass sink to parser/verifier ...
//   // json_output now contains one line per diagnostic.

#ifndef LOOM_ERROR_JSON_SINK_H_
#define LOOM_ERROR_JSON_SINK_H_

#include "iree/base/api.h"
#include "loom/error/diagnostic.h"
#include "loom/error/renderer.h"

#ifdef __cplusplus
extern "C" {
#endif

// Options for the JSON sink. Passed as user_data to
// loom_diagnostic_json_sink.
typedef struct loom_json_sink_options_t {
  // Output stream. One JSON object is written per diagnostic,
  // terminated by a newline.
  loom_output_stream_t* stream;

  // Type formatter for rendering TYPE params. When .fn is NULL,
  // TYPE params emit "<type>".
  loom_type_formatter_t type_formatter;
} loom_json_sink_options_t;

// Writes one structured diagnostic JSON object with no trailing newline.
//
// This is the shared object writer underneath loom_diagnostic_json_sink().
// Tool-level JSON outputs that need to embed diagnostics in a larger result
// object should use this API instead of re-spelling diagnostic fields.
iree_status_t loom_diagnostic_json_write_object(
    loom_output_stream_t* stream, const loom_diagnostic_t* diagnostic,
    loom_type_formatter_t type_formatter);

// Diagnostic sink callback that emits JSONL output. Pass a
// loom_json_sink_options_t* as user_data.
iree_status_t loom_diagnostic_json_sink(void* user_data,
                                        const loom_diagnostic_t* diagnostic);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_ERROR_JSON_SINK_H_
