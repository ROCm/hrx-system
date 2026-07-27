// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Reporting state for loom-check test runs.

#ifndef LOOM_TOOLS_LOOM_CHECK_REPORT_H_
#define LOOM_TOOLS_LOOM_CHECK_REPORT_H_

#include "iree/base/api.h"
#include "iree/base/bitmap.h"
#include "iree/base/internal/arena.h"
#include "loom/tools/loom-check/check.h"

#ifdef __cplusplus
extern "C" {
#endif

// Per-run reporting state for a parsed loom-check file.
//
// loom_check_file_t is the immutable parse product: cases, directives, input
// sections, expected sections, and parsed annotations. This report table owns
// execution state that is only meaningful for one run of that parsed file, such
// as whether each expected diagnostic annotation matched an actual diagnostic.
typedef struct loom_check_file_report_t {
  // Number of cases in the parsed file this report corresponds to.
  iree_host_size_t case_count;

  // Total annotation count across all cases.
  iree_host_size_t annotation_count;

  // Per-case offset into the global annotation bitmap.
  iree_host_size_t* case_annotation_offsets;

  // One bit per parsed annotation. A set bit means the annotation matched an
  // actual diagnostic in this run.
  iree_bitmap_t matched_annotations;
} loom_check_file_report_t;

// Initializes a file report for |file|. Storage is arena-backed and freed with
// the arena; no report deinitialize is needed.
iree_status_t loom_check_file_report_initialize(
    const loom_check_file_t* file, iree_arena_allocator_t* arena,
    loom_check_file_report_t* out_report);

// Clears all match bits so the report can be reused for another run of the same
// parsed file.
void loom_check_file_report_reset_matches(loom_check_file_report_t* report);

// Marks annotation |annotation_index| within |case_index| as matched. Returns
// INVALID_ARGUMENT if either index is out of bounds.
iree_status_t loom_check_file_report_mark_annotation_matched(
    loom_check_file_report_t* report, iree_host_size_t case_index,
    iree_host_size_t annotation_index);

// Returns whether annotation |annotation_index| within |case_index| matched.
// Returns INVALID_ARGUMENT if either index is out of bounds.
iree_status_t loom_check_file_report_annotation_matched(
    const loom_check_file_report_t* report, iree_host_size_t case_index,
    iree_host_size_t annotation_index, bool* out_matched);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_TOOLS_LOOM_CHECK_REPORT_H_
