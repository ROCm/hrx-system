// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Line-oriented unified diff of two text strings.
//
// Computes the shortest edit script between |expected| and |actual|,
// then formats the result as a unified diff with context lines.
// Output follows the standard unified diff format:
//
//   --- expected
//   +++ actual
//   @@ -1,4 +1,3 @@
//    context line
//   -removed line
//   +added line
//    context line
//
// When the inputs are identical, produces no output (builder is
// untouched). When either input is empty, every line in the other
// appears as an addition or removal.
//
// The algorithm is O(N*M) where N and M are the line counts of the
// two inputs. This is fine for IR text (typically tens to hundreds of
// lines). For very large inputs, consider a streaming approach.
//
// Callers can either format the diff directly as text with loom_diff(), or
// compute a structured result and format/serialize it themselves.

#ifndef LOOM_UTIL_DIFF_H_
#define LOOM_UTIL_DIFF_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif

// Number of unchanged context lines to show around each change hunk.
#define LOOM_DIFF_DEFAULT_CONTEXT 3

// Line kind inside a diff hunk.
typedef enum loom_diff_hunk_line_kind_e {
  LOOM_DIFF_HUNK_LINE_CONTEXT = 0,
  LOOM_DIFF_HUNK_LINE_DELETE = 1,
  LOOM_DIFF_HUNK_LINE_INSERT = 2,
} loom_diff_hunk_line_kind_t;

// One displayed line in a diff hunk. |text| is a view into the original
// expected or actual input passed to loom_diff_compute().
typedef struct loom_diff_hunk_line_t {
  loom_diff_hunk_line_kind_t kind;
  iree_string_view_t text;
} loom_diff_hunk_line_t;

// A line-oriented diff hunk. Line numbers are 1-based.
typedef struct loom_diff_hunk_t {
  iree_host_size_t expected_start_line;
  iree_host_size_t expected_line_count;
  iree_host_size_t actual_start_line;
  iree_host_size_t actual_line_count;
  iree_host_size_t line_offset;
  iree_host_size_t line_count;
} loom_diff_hunk_t;

// Structured diff result. Hunk lines are stored in |lines|; each hunk owns a
// contiguous subrange described by |line_offset| and |line_count|.
typedef struct loom_diff_result_t {
  loom_diff_hunk_t* hunks;
  iree_host_size_t hunk_count;
  loom_diff_hunk_line_t* lines;
  iree_host_size_t line_count;
} loom_diff_result_t;

// Returns a stable JSON/text spelling for a hunk line kind.
const char* loom_diff_hunk_line_kind_name(loom_diff_hunk_line_kind_t kind);

// Releases memory owned by |result|. The expected/actual input strings passed
// to loom_diff_compute() remain caller-owned and must outlive |result| while
// hunk line text views are being read.
void loom_diff_result_deinitialize(iree_allocator_t allocator,
                                   loom_diff_result_t* result);

// Computes a structured diff between |expected| and |actual|. When the inputs
// have no line-level differences, |out_result| is left empty.
//
// The caller must keep |expected| and |actual| alive while reading
// |out_result|, because hunk lines contain views into those input strings.
iree_status_t loom_diff_compute(iree_string_view_t expected,
                                iree_string_view_t actual,
                                iree_host_size_t context_lines,
                                iree_allocator_t allocator,
                                loom_diff_result_t* out_result);

// Formats a structured diff result as unified diff text. Empty results append
// nothing.
iree_status_t loom_diff_format_result(const loom_diff_result_t* result,
                                      iree_string_builder_t* builder);

// Computes a unified diff between |expected| and |actual|.
//
// Both inputs are treated as line-oriented text (split on '\n').
// Trailing content after the last '\n' is treated as a final line.
//
// If the inputs are identical, nothing is appended to |builder| and
// the function returns iree_ok_status().
//
// The diff is formatted with |context_lines| lines of unchanged
// context around each change hunk. Adjacent hunks within context
// distance are merged into a single hunk.
//
// |allocator| is used for scratch allocations (line arrays, DP table).
// All scratch memory is freed before the function returns. An arena
// allocator is suitable when the caller already has one available.
iree_status_t loom_diff(iree_string_view_t expected, iree_string_view_t actual,
                        iree_host_size_t context_lines,
                        iree_allocator_t allocator,
                        iree_string_builder_t* builder);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_UTIL_DIFF_H_
