// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Parser for loom-check .loom-test files.
//
// A .loom-test file contains one or more test cases, each optionally
// separated by a case separator. Each case has directives controlling
// what operation to perform, an input section, and optionally an
// expected output section.
//
// File format:
//
//   // RUN: roundtrip
//   // TEMPLATE: loom/src/loom/test/corpus/vector/arithmetic.loom-test
//   <input IR>
//   // ----
//   <expected output>
//   // ====
//   // RUN: verify
//   <input IR with annotations>
//
// Directives:
//   // RUN: roundtrip       Parse -> print -> compare (default).
//   // RUN: with-locations roundtrip
//                           Parse -> print with loc() annotations -> compare.
//   // RUN: verify          Parse -> verify -> match annotations.
//   // RUN: pass <pipeline> Parse -> run pipeline -> print -> compare.
//   // RUN: with-locations pass <pipeline>
//                           Run pipeline and print loc() annotations.
//   // RUN: pass-report <pipeline>
//                           Parse -> run pipeline -> compile report -> compare.
//   // RUN: format <target> Parse -> convert format -> print -> compare.
//   // RUN: emit <target>   Parse -> emit analysis or target-structured
//                           output -> compare. Core targets include
//                           target-low-registry-manifest and source-low.
//                           Analysis targets include liveness-json @function,
//                           low-schedule-json @function [strategy=...]
//                           [diagnostics=...] [cliff=<reg-class>:...],
//                           low-allocation @function
//                           [diagnostics=...] [class=units...]
//                           [fixed=%value:<kind>:<base>:<count>],
//                           low-allocation-json @function for the full
//                           allocation table, diagnostics include
//                           predicted-spills, copy-decisions, and
//                           placement-decisions, and low-packet-json @function.
//                           Source-to-low tests use source-low
//                           [output=module|low]
//                           [sanitizer=none|access|value|operation|race|all]
//                           [sanitizer-reporting=default|trap|report-only]
//                           [diagnostics=none|memory|all]. Linked providers
//                           may add target-specific emit forms.
//                           Low schedule diagnostics are one of none,
//                           pressure, resources, hazards, candidates, model,
//                           or all.
//   // REQUIRES: <name>[, <name>...]
//                           Skip the case when external tools or target
//                           backends are unavailable.
//   // XFAIL: <reason>      Mark case as expected failure.
//   // TEMPLATE: <path>      Declare that this file is synchronized from a
//                           root-relative corpus template. File-level only;
//                           ignored by ordinary execution.
//
// Separators:
//   // ====                 Case separator. The first separator must appear
//                           after the first case body.
//   // ----                 Input/expected separator within a case.
//
// RUN inheritance:
//   A // RUN: directive in a template file preamble or in the first case sets
//   the file-level default mode. Cases without their own // RUN: inherit from
//   the file default. Cases with their own // RUN: override it. When no default
//   // RUN: exists, the default is roundtrip.
//
// REQUIRES inheritance:
//   A // REQUIRES: directive in a template file preamble or in the first case
//   sets file-level default environment requirements. Every case receives the
//   file defaults, plus any case-local requirements it declares.
//
// TEMPLATE:
//   A // TEMPLATE: directive in the leading file preamble declares a source
//   corpus template for generated target-specific expectation files. It is
//   metadata for update tooling, not a linking mechanism and not a case
//   namespace. The preamble is the leading directive/comment block containing
//   TEMPLATE; do not add a // ==== separator before the first real case.
//   Individual case names are the function symbols inside the case IR;
//   // CASE directives are intentionally unsupported.
//
// Annotations (for verify mode — uppercase to distinguish from comments):
//   // ERROR: DOMAIN/CODE "substring"
//   // WARNING: DOMAIN/CODE
//   // REMARK: "substring"
//   // ERROR@+1: DOMAIN/CODE "substring"
//   // ERROR@-2: "substring"
//   // ERROR@+1: DOMAIN/CODE "first" "second"
//
// Multiple quoted substrings may follow the DOMAIN/CODE (or appear
// alone). All listed substrings must appear in the diagnostic message
// for the annotation to match. Order does not matter; substrings may
// overlap. Up to LOOM_CHECK_MAX_ANNOTATION_SUBSTRINGS substrings per
// annotation are supported.
//
// The offset (@+N / @-N) targets a line relative to the annotation.
// Without an offset, the annotation targets its own line.
//
// When no // ---- separator is present, the expected output equals
// the input (round-trip identity). When present, everything after
// // ---- is the expected output.
//
// All string views returned by parsing point into the original source
// buffer — the caller must keep the source alive while using the
// parsed file structure.
//
// All internal allocations (cases array, annotation arrays) are bump-
// allocated from the caller's arena. Cleanup is iree_arena_deinitialize
// — no per-field freeing required, no cleanup-on-error paths needed.

#ifndef LOOM_TOOLS_LOOM_CHECK_CHECK_H_
#define LOOM_TOOLS_LOOM_CHECK_CHECK_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/error/error_defs.h"

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// Test mode
//===----------------------------------------------------------------------===//

// What operation to perform on the test case.
typedef enum loom_check_mode_e {
  LOOM_CHECK_MODE_ROUNDTRIP = 0,  // Parse -> print -> compare (default).
  LOOM_CHECK_MODE_VERIFY = 1,     // Parse -> verify -> match annotations.
  LOOM_CHECK_MODE_PASS = 2,       // Parse -> run pipeline -> print -> compare.
  LOOM_CHECK_MODE_FORMAT = 3,  // Parse -> convert format -> print -> compare.
  LOOM_CHECK_MODE_EMIT = 4,    // Parse -> emit target/check output -> compare.
  LOOM_CHECK_MODE_PASS_REPORT = 5,  // Parse -> run pipeline -> report.
} loom_check_mode_t;

// Flags controlling optional textual output surfaces for a test case.
enum loom_check_output_flag_bits_e {
  // Emit trailing loc() annotations in canonical IR output.
  LOOM_CHECK_OUTPUT_LOCATIONS = 1u << 0,
};
typedef uint32_t loom_check_output_flags_t;

// Returns a human-readable name for the mode.
static inline const char* loom_check_mode_name(loom_check_mode_t mode) {
  switch (mode) {
    case LOOM_CHECK_MODE_ROUNDTRIP:
      return "roundtrip";
    case LOOM_CHECK_MODE_VERIFY:
      return "verify";
    case LOOM_CHECK_MODE_PASS:
      return "pass";
    case LOOM_CHECK_MODE_FORMAT:
      return "format";
    case LOOM_CHECK_MODE_EMIT:
      return "emit";
    case LOOM_CHECK_MODE_PASS_REPORT:
      return "pass-report";
    default:
      return "unknown";
  }
}

//===----------------------------------------------------------------------===//
// Source ranges
//===----------------------------------------------------------------------===//

// Byte range in the original .loom-test source buffer. Ranges are half-open:
// [start_byte, end_byte). Newlines that separate structural lines are excluded
// from line-level directive and annotation ranges.
typedef struct loom_check_source_range_t {
  iree_host_size_t start_byte;
  iree_host_size_t end_byte;
} loom_check_source_range_t;

// Returns the sentinel empty source range used for absent optional ranges.
static inline loom_check_source_range_t loom_check_source_range_empty(void) {
  return (loom_check_source_range_t){0};
}

// Returns true if |source_range| is the absent optional range sentinel.
static inline bool loom_check_source_range_is_empty(
    loom_check_source_range_t source_range) {
  return source_range.start_byte == 0 && source_range.end_byte == 0;
}

//===----------------------------------------------------------------------===//
// Annotations
//===----------------------------------------------------------------------===//

// Maximum number of message substrings a single annotation may declare.
// Each substring is an independent constraint: the diagnostic message
// must contain all of them. Real annotations rarely need more than 2-3,
// but a small fixed cap keeps the struct compact and avoids a heap
// allocation per annotation.
#define LOOM_CHECK_MAX_ANNOTATION_SUBSTRINGS 4

// Maximum number of structured diagnostic parameter matchers declared by one
// annotation. Param matchers compare against generated error parameter names
// instead of the rendered diagnostic prose.
#define LOOM_CHECK_MAX_ANNOTATION_PARAM_MATCHES 8

// One structured diagnostic parameter expectation from an annotation.
typedef struct loom_check_annotation_param_match_t {
  // Generated diagnostic parameter name, such as "op_name".
  iree_string_view_t name;
  // Expected rendered parameter value.
  iree_string_view_t value;
} loom_check_annotation_param_match_t;

// One expected diagnostic annotation extracted from a comment.
typedef struct loom_check_annotation_t {
  // Number of populated entries in message_substrings. 0 means "match
  // any message"; non-zero means every entry in [0, count) must be a
  // substring of the diagnostic message. Placed first so the matcher's
  // hot path can check it without touching the 64-byte substring array.
  uint8_t message_substring_count;
  // Number of populated entries in param_matches. 0 means no structured
  // parameter constraints.
  uint8_t param_match_count;
  // Expected severity (error, warning, remark).
  loom_diagnostic_severity_t severity;
  // Error domain (TYPE, PARSE, etc.). LOOM_ERROR_DOMAIN_COUNT_ is the
  // sentinel meaning "match any domain" — used when the annotation
  // omits the domain (e.g., // ERROR: "substring").
  loom_error_domain_t domain;
  // Error code within the domain. 0 matches any code.
  uint16_t code;
  // Source range of the annotation comment line.
  loom_check_source_range_t source_range;
  // 1-based line in the input that this annotation targets.
  iree_host_size_t target_line;
  // Substrings that must all appear in the diagnostic message.
  iree_string_view_t message_substrings[LOOM_CHECK_MAX_ANNOTATION_SUBSTRINGS];
  // Structured diagnostic parameter constraints that must all match.
  loom_check_annotation_param_match_t
      param_matches[LOOM_CHECK_MAX_ANNOTATION_PARAM_MATCHES];
} loom_check_annotation_t;

//===----------------------------------------------------------------------===//
// Test case
//===----------------------------------------------------------------------===//

// One parsed test case (one // ==== section).
typedef struct loom_check_case_t {
  // What operation to perform.
  loom_check_mode_t mode;
  // Optional output surfaces enabled for this case.
  loom_check_output_flags_t output_flags;
  // Source range of the complete case text, excluding the preceding case
  // separator line when present.
  loom_check_source_range_t source_range;
  // Source range of the preceding // ==== separator line. Empty when this case
  // was not preceded by a separator.
  loom_check_source_range_t separator_range;
  // Whether this case contained its own // RUN: directive. When false,
  // the mode/pipeline/format_target were inherited from the file default.
  bool has_run_directive;
  // Source range of the // RUN: directive line. Empty when inherited.
  loom_check_source_range_t run_directive_range;
  // For PASS mode: comma-separated pass pipeline (e.g. "dce,cse").
  iree_string_view_t pipeline;
  // For FORMAT mode: target format name (e.g. "bytecode").
  iree_string_view_t format_target;
  // For EMIT mode: target emission request (e.g. "target-form @symbol").
  iree_string_view_t emit_target;
  // Whether this case is expected to fail.
  bool xfail;
  // Source range of the // XFAIL: directive line. Empty when absent.
  loom_check_source_range_t xfail_directive_range;
  // Reason for expected failure.
  iree_string_view_t xfail_reason;
  // Whether this case contained its own // REQUIRES: directive. File-level
  // defaults are present in requirements either way after parsing completes.
  bool has_requires_directive;
  // Source range of the // REQUIRES: directive line. Empty when absent.
  loom_check_source_range_t requires_directive_range;
  // Arena-allocated requirement name array, including inherited file-level
  // requirements after parsing completes.
  iree_string_view_t* requirements;
  // Number of requirement names in requirements.
  iree_host_size_t requirement_count;
  // IR text with directives stripped. Points into source.
  iree_string_view_t input;
  // Source range of input.
  loom_check_source_range_t input_range;
  // Expected output. Equals input when no // ---- separator is present.
  iree_string_view_t expected;
  // Source range of expected output. Empty when no // ---- separator is
  // present.
  loom_check_source_range_t expected_range;
  // Whether a // ---- separator was present.
  bool has_expected_section;
  // Source range of the // ---- separator line. Empty when absent.
  loom_check_source_range_t expected_separator_range;
  // Arena-allocated array of annotations extracted from the input.
  loom_check_annotation_t* annotations;
  // Number of annotations.
  iree_host_size_t annotation_count;
} loom_check_case_t;

//===----------------------------------------------------------------------===//
// Test file
//===----------------------------------------------------------------------===//

// A parsed test file containing one or more cases. All internal
// allocations live in the arena passed to loom_check_parse. The
// caller owns the arena and deinitializes it to free everything.
typedef struct loom_check_file_t {
  // Arena-allocated array of test cases.
  loom_check_case_t* cases;
  // Number of test cases.
  iree_host_size_t case_count;
  // Whether the file declares a corpus template source.
  bool has_template_directive;
  // Source range of the // TEMPLATE: directive line. Empty when absent.
  loom_check_source_range_t template_directive_range;
  // Root-relative corpus template path from // TEMPLATE:.
  iree_string_view_t template_path;

  // File-level default mode inherited by cases without their own // RUN:
  // directive. Defaults to ROUNDTRIP when no default // RUN: directive exists.
  loom_check_mode_t default_mode;
  loom_check_output_flags_t default_output_flags;
  iree_string_view_t default_pipeline;
  iree_string_view_t default_format_target;
  iree_string_view_t default_emit_target;
  // Arena-allocated file-level default requirement name array.
  iree_string_view_t* default_requirements;
  // Number of requirement names in default_requirements.
  iree_host_size_t default_requirement_count;
} loom_check_file_t;

//===----------------------------------------------------------------------===//
// Parsing API
//===----------------------------------------------------------------------===//

// Parses a .loom-test file into cases. All string views point into
// |source| — the caller must keep source alive while using the file.
//
// All internal allocations are bump-allocated from |arena|. The
// caller owns the arena and deinitializes it when the file is no
// longer needed.
iree_status_t loom_check_parse(iree_string_view_t source,
                               iree_arena_allocator_t* arena,
                               loom_check_file_t* out_file);

// Strips standalone comment lines from IR text. Lines that consist
// entirely of a comment (trimmed line starts with "//") become blank
// lines to preserve the structural line count. Non-comment lines are
// preserved as-is, including any trailing comments on IR lines.
iree_status_t loom_check_strip_comments(iree_string_view_t input,
                                        iree_string_builder_t* output);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_TOOLS_LOOM_CHECK_CHECK_H_
