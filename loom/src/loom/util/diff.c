// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/util/diff.h"

#include <string.h>

//===----------------------------------------------------------------------===//
// Line splitting
//===----------------------------------------------------------------------===//

// A single line: a view into the source string, not including the
// trailing newline (if any).
typedef struct loom_diff_source_line_t {
  iree_string_view_t text;
} loom_diff_source_line_t;

// Splits |source| into lines on '\n' boundaries. Lines do not include
// the newline character. A trailing newline produces no empty final
// line (i.e., "a\nb\n" splits into ["a", "b"], not ["a", "b", ""]).
// Content after the last newline IS a line (i.e., "a\nb" splits into
// ["a", "b"]).
//
// |out_lines| is allocated from |allocator| and must be freed by the
// caller. |out_count| receives the number of lines.
static iree_status_t loom_diff_split_lines(iree_string_view_t source,
                                           iree_allocator_t allocator,
                                           loom_diff_source_line_t** out_lines,
                                           iree_host_size_t* out_count) {
  *out_lines = NULL;
  *out_count = 0;

  if (source.size == 0) return iree_ok_status();

  // Count lines first.
  iree_host_size_t count = 0;
  for (iree_host_size_t i = 0; i < source.size; ++i) {
    if (source.data[i] == '\n') ++count;
  }
  // If the source does not end with '\n', there is a final partial line.
  if (source.data[source.size - 1] != '\n') ++count;

  loom_diff_source_line_t* lines = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
      allocator, count, sizeof(loom_diff_source_line_t), (void**)&lines));

  iree_host_size_t line_index = 0;
  iree_host_size_t line_start = 0;
  for (iree_host_size_t i = 0; i < source.size; ++i) {
    if (source.data[i] == '\n') {
      lines[line_index].text =
          iree_make_string_view(source.data + line_start, i - line_start);
      ++line_index;
      line_start = i + 1;
    }
  }
  // Final partial line (no trailing newline).
  if (line_start < source.size) {
    lines[line_index].text = iree_make_string_view(source.data + line_start,
                                                   source.size - line_start);
    ++line_index;
  }

  *out_lines = lines;
  *out_count = line_index;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// LCS computation
//===----------------------------------------------------------------------===//

// Edit operation kind for the edit script.
typedef enum loom_diff_edit_kind_e {
  LOOM_DIFF_EQUAL = 0,   // Line present in both.
  LOOM_DIFF_DELETE = 1,  // Line only in expected (removed).
  LOOM_DIFF_INSERT = 2,  // Line only in actual (added).
} loom_diff_edit_kind_t;

typedef struct loom_diff_edit_t {
  loom_diff_edit_kind_t kind;
  iree_host_size_t expected_index;  // Index into expected lines.
  iree_host_size_t actual_index;    // Index into actual lines.
} loom_diff_edit_t;

// Computes the edit script between |expected_lines| and |actual_lines|
// using the standard DP LCS algorithm. The edit script is written to
// |out_edits| (allocated from |allocator|) in forward order.
//
// The DP table is (expected_count+1) x (actual_count+1) entries of
// iree_host_size_t. For typical IR diffs (< 1000 lines), this is
// < 8 MB — fine for a test tool.
static iree_status_t loom_diff_compute_edits(
    const loom_diff_source_line_t* expected_lines,
    iree_host_size_t expected_count,
    const loom_diff_source_line_t* actual_lines, iree_host_size_t actual_count,
    iree_allocator_t allocator, loom_diff_edit_t** out_edits,
    iree_host_size_t* out_edit_count) {
  *out_edits = NULL;
  *out_edit_count = 0;

  iree_host_size_t rows = expected_count + 1;
  iree_host_size_t cols = actual_count + 1;

  // Allocate DP table. table[i][j] = LCS length of
  // expected[0..i-1] and actual[0..j-1].
  iree_host_size_t table_count = 0;
  if (!iree_host_size_checked_mul(rows, cols, &table_count)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "diff DP table too large");
  }
  iree_host_size_t* table = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
      allocator, table_count, sizeof(iree_host_size_t), (void**)&table));

  // Fill the table bottom-up.
  for (iree_host_size_t i = 1; i < rows; ++i) {
    for (iree_host_size_t j = 1; j < cols; ++j) {
      if (iree_string_view_equal(expected_lines[i - 1].text,
                                 actual_lines[j - 1].text)) {
        table[i * cols + j] = table[(i - 1) * cols + (j - 1)] + 1;
      } else {
        iree_host_size_t from_above = table[(i - 1) * cols + j];
        iree_host_size_t from_left = table[i * cols + (j - 1)];
        table[i * cols + j] =
            (from_above >= from_left) ? from_above : from_left;
      }
    }
  }

  // Backtrace to produce the edit script (in reverse).
  iree_host_size_t max_edits = expected_count + actual_count;
  if (max_edits == 0) {
    iree_allocator_free(allocator, table);
    *out_edits = NULL;
    *out_edit_count = 0;
    return iree_ok_status();
  }
  loom_diff_edit_t* edits = NULL;
  iree_status_t status = iree_allocator_malloc_array(
      allocator, max_edits, sizeof(loom_diff_edit_t), (void**)&edits);
  if (!iree_status_is_ok(status)) {
    iree_allocator_free(allocator, table);
    return status;
  }

  iree_host_size_t edit_count = 0;
  iree_host_size_t i = expected_count;
  iree_host_size_t j = actual_count;

  while (i > 0 || j > 0) {
    if (i > 0 && j > 0 &&
        iree_string_view_equal(expected_lines[i - 1].text,
                               actual_lines[j - 1].text)) {
      edits[edit_count].kind = LOOM_DIFF_EQUAL;
      edits[edit_count].expected_index = i - 1;
      edits[edit_count].actual_index = j - 1;
      ++edit_count;
      --i;
      --j;
    } else if (j > 0 && (i == 0 || table[(i)*cols + (j - 1)] >=
                                       table[(i - 1) * cols + (j)])) {
      edits[edit_count].kind = LOOM_DIFF_INSERT;
      edits[edit_count].expected_index = i;
      edits[edit_count].actual_index = j - 1;
      ++edit_count;
      --j;
    } else {
      edits[edit_count].kind = LOOM_DIFF_DELETE;
      edits[edit_count].expected_index = i - 1;
      edits[edit_count].actual_index = j;
      ++edit_count;
      --i;
    }
  }

  // Reverse the edit script to get forward order.
  if (edit_count > 1) {
    for (iree_host_size_t left = 0, right = edit_count - 1; left < right;
         ++left, --right) {
      loom_diff_edit_t temporary = edits[left];
      edits[left] = edits[right];
      edits[right] = temporary;
    }
  }

  iree_allocator_free(allocator, table);

  *out_edits = edits;
  *out_edit_count = edit_count;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Hunk building and formatting
//===----------------------------------------------------------------------===//

static loom_diff_hunk_line_kind_t loom_diff_edit_line_kind(
    loom_diff_edit_kind_t kind) {
  switch (kind) {
    case LOOM_DIFF_EQUAL:
      return LOOM_DIFF_HUNK_LINE_CONTEXT;
    case LOOM_DIFF_DELETE:
      return LOOM_DIFF_HUNK_LINE_DELETE;
    case LOOM_DIFF_INSERT:
      return LOOM_DIFF_HUNK_LINE_INSERT;
  }
  return LOOM_DIFF_HUNK_LINE_CONTEXT;
}

static iree_string_view_t loom_diff_edit_line_text(
    const loom_diff_edit_t* edit, const loom_diff_source_line_t* expected_lines,
    const loom_diff_source_line_t* actual_lines) {
  switch (edit->kind) {
    case LOOM_DIFF_EQUAL:
    case LOOM_DIFF_DELETE:
      return expected_lines[edit->expected_index].text;
    case LOOM_DIFF_INSERT:
      return actual_lines[edit->actual_index].text;
  }
  return iree_string_view_empty();
}

// Appends a structured hunk covering edits[start..end) into |result|.
static void loom_diff_append_hunk(const loom_diff_edit_t* edits,
                                  iree_host_size_t start, iree_host_size_t end,
                                  const loom_diff_source_line_t* expected_lines,
                                  const loom_diff_source_line_t* actual_lines,
                                  loom_diff_result_t* result) {
  iree_host_size_t expected_start = edits[start].expected_index;
  iree_host_size_t actual_start = edits[start].actual_index;
  iree_host_size_t expected_length = 0;
  iree_host_size_t actual_length = 0;
  for (iree_host_size_t k = start; k < end; ++k) {
    if (edits[k].kind == LOOM_DIFF_EQUAL || edits[k].kind == LOOM_DIFF_DELETE) {
      ++expected_length;
    }
    if (edits[k].kind == LOOM_DIFF_EQUAL || edits[k].kind == LOOM_DIFF_INSERT) {
      ++actual_length;
    }
  }

  loom_diff_hunk_t* hunk = &result->hunks[result->hunk_count++];
  hunk->expected_start_line = expected_start + 1;
  hunk->expected_line_count = expected_length;
  hunk->actual_start_line = actual_start + 1;
  hunk->actual_line_count = actual_length;
  hunk->line_offset = result->line_count;
  hunk->line_count = end - start;
  for (iree_host_size_t k = start; k < end; ++k) {
    loom_diff_hunk_line_t* line = &result->lines[result->line_count++];
    line->kind = loom_diff_edit_line_kind(edits[k].kind);
    line->text =
        loom_diff_edit_line_text(&edits[k], expected_lines, actual_lines);
  }
}

// Builds the structured hunk list from the edit script.
// Adjacent changes within |context_lines| of each other are merged
// into a single hunk.
static void loom_diff_build_hunks(const loom_diff_edit_t* edits,
                                  iree_host_size_t edit_count,
                                  const loom_diff_source_line_t* expected_lines,
                                  const loom_diff_source_line_t* actual_lines,
                                  iree_host_size_t context_lines,
                                  loom_diff_result_t* result) {
  // Walk the edit list, tracking the current hunk boundaries. When we
  // see a gap of more than 2*context_lines EQUAL edits between
  // changes, flush the current hunk and start a new one.
  iree_host_size_t hunk_start = 0;
  bool in_hunk = false;
  iree_host_size_t last_change = 0;

  for (iree_host_size_t i = 0; i < edit_count; ++i) {
    if (edits[i].kind != LOOM_DIFF_EQUAL) {
      if (!in_hunk) {
        hunk_start = (i > context_lines) ? (i - context_lines) : 0;
        in_hunk = true;
      }
      last_change = i;
    } else if (in_hunk && (i - last_change) > 2 * context_lines) {
      iree_host_size_t hunk_end = last_change + context_lines + 1;
      if (hunk_end > edit_count) hunk_end = edit_count;
      loom_diff_append_hunk(edits, hunk_start, hunk_end, expected_lines,
                            actual_lines, result);
      in_hunk = false;
    }
  }

  if (in_hunk) {
    iree_host_size_t hunk_end = last_change + context_lines + 1;
    if (hunk_end > edit_count) hunk_end = edit_count;
    loom_diff_append_hunk(edits, hunk_start, hunk_end, expected_lines,
                          actual_lines, result);
  }
}

//===----------------------------------------------------------------------===//
// Public API
//===----------------------------------------------------------------------===//

const char* loom_diff_hunk_line_kind_name(loom_diff_hunk_line_kind_t kind) {
  switch (kind) {
    case LOOM_DIFF_HUNK_LINE_CONTEXT:
      return "context";
    case LOOM_DIFF_HUNK_LINE_DELETE:
      return "delete";
    case LOOM_DIFF_HUNK_LINE_INSERT:
      return "insert";
  }
  return "unknown";
}

void loom_diff_result_deinitialize(iree_allocator_t allocator,
                                   loom_diff_result_t* result) {
  iree_allocator_free(allocator, result->lines);
  iree_allocator_free(allocator, result->hunks);
  memset(result, 0, sizeof(*result));
}

iree_status_t loom_diff_compute(iree_string_view_t expected,
                                iree_string_view_t actual,
                                iree_host_size_t context_lines,
                                iree_allocator_t allocator,
                                loom_diff_result_t* out_result) {
  memset(out_result, 0, sizeof(*out_result));

  // Fast path: identical inputs produce no diff.
  if (iree_string_view_equal(expected, actual)) return iree_ok_status();

  // Split into lines.
  loom_diff_source_line_t* expected_lines = NULL;
  iree_host_size_t expected_count = 0;
  loom_diff_source_line_t* actual_lines = NULL;
  iree_host_size_t actual_count = 0;

  iree_status_t status = loom_diff_split_lines(
      expected, allocator, &expected_lines, &expected_count);
  if (iree_status_is_ok(status)) {
    status =
        loom_diff_split_lines(actual, allocator, &actual_lines, &actual_count);
  }

  // Compute edit script.
  loom_diff_edit_t* edits = NULL;
  iree_host_size_t edit_count = 0;
  if (iree_status_is_ok(status)) {
    status =
        loom_diff_compute_edits(expected_lines, expected_count, actual_lines,
                                actual_count, allocator, &edits, &edit_count);
  }

  // Check whether there are any actual changes. The raw strings may differ
  // (e.g., trailing newline) while line content is identical.
  bool has_changes = false;
  if (iree_status_is_ok(status)) {
    for (iree_host_size_t i = 0; i < edit_count; ++i) {
      if (edits[i].kind != LOOM_DIFF_EQUAL) {
        has_changes = true;
        break;
      }
    }
  }

  // Build hunks only when there are line-level differences.
  if (iree_status_is_ok(status) && has_changes) {
    status = iree_allocator_malloc_array(allocator, edit_count,
                                         sizeof(loom_diff_hunk_t),
                                         (void**)&out_result->hunks);
    if (iree_status_is_ok(status)) {
      status = iree_allocator_malloc_array(allocator, edit_count,
                                           sizeof(loom_diff_hunk_line_t),
                                           (void**)&out_result->lines);
    }
    if (iree_status_is_ok(status)) {
      loom_diff_build_hunks(edits, edit_count, expected_lines, actual_lines,
                            context_lines, out_result);
    }
  }

  iree_allocator_free(allocator, edits);
  iree_allocator_free(allocator, actual_lines);
  iree_allocator_free(allocator, expected_lines);

  if (!iree_status_is_ok(status)) {
    loom_diff_result_deinitialize(allocator, out_result);
  }
  return status;
}

iree_status_t loom_diff_format_result(const loom_diff_result_t* result,
                                      iree_string_builder_t* builder) {
  if (result->hunk_count == 0) return iree_ok_status();

  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder,
                                                          "--- expected\n"
                                                          "+++ actual\n"));
  for (iree_host_size_t i = 0; i < result->hunk_count; ++i) {
    const loom_diff_hunk_t* hunk = &result->hunks[i];
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder, "@@ -%zu,%zu +%zu,%zu @@\n", (size_t)hunk->expected_start_line,
        (size_t)hunk->expected_line_count, (size_t)hunk->actual_start_line,
        (size_t)hunk->actual_line_count));
    for (iree_host_size_t j = 0; j < hunk->line_count; ++j) {
      const loom_diff_hunk_line_t* line = &result->lines[hunk->line_offset + j];
      const char* prefix = " ";
      switch (line->kind) {
        case LOOM_DIFF_HUNK_LINE_CONTEXT:
          prefix = " ";
          break;
        case LOOM_DIFF_HUNK_LINE_DELETE:
          prefix = "-";
          break;
        case LOOM_DIFF_HUNK_LINE_INSERT:
          prefix = "+";
          break;
      }
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder, "%s%.*s\n", prefix, (int)line->text.size, line->text.data));
    }
  }
  return iree_ok_status();
}

iree_status_t loom_diff(iree_string_view_t expected, iree_string_view_t actual,
                        iree_host_size_t context_lines,
                        iree_allocator_t allocator,
                        iree_string_builder_t* builder) {
  loom_diff_result_t result = {0};
  iree_status_t status =
      loom_diff_compute(expected, actual, context_lines, allocator, &result);
  if (iree_status_is_ok(status)) {
    status = loom_diff_format_result(&result, builder);
  }
  loom_diff_result_deinitialize(allocator, &result);
  return status;
}
