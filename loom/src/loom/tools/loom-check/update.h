// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Update edit logic for loom-check.
//
// Expected-output updates can reconstruct a test file source for --update mode.
// The reconstruction is pure computation (no file I/O) — the caller decides
// where to write the result. This separation keeps the logic testable without
// touching the filesystem.
//
// The typical call sequence:
//
//   1. Parse the file with loom_check_parse().
//   2. Execute each case with loom_check_execute_case().
//   3. Build loom_check_case_update_t entries from the results.
//   4. Call loom_check_apply_updates() to reconstruct the source.
//   5. Write the result wherever appropriate (file, stdout, etc.).

#ifndef LOOM_TOOLS_LOOM_CHECK_UPDATE_H_
#define LOOM_TOOLS_LOOM_CHECK_UPDATE_H_

#include "iree/base/api.h"
#include "loom/tools/loom-check/check.h"

#ifdef __cplusplus
extern "C" {
#endif

// Per-case update tracking. The caller populates one of these per case after
// execution, indicating whether the expected section should be rewritten or
// deleted and providing the actual output text.
typedef struct loom_check_case_update_t {
  // Whether this case's expected section should be rewritten or deleted.
  bool needs_update;
  // The actual output to write. Must remain valid until apply_updates
  // returns (typically a view into a result's actual_output builder).
  iree_string_view_t actual_output;
  // Pointer to the end of the input section in the original source. Used to
  // locate where to insert a // ---- separator when none exists.
  const char* input_end;
  // Pointer to the start of the existing expected payload in the original
  // source. Only valid when the case has_expected_section.
  const char* expected_start;
  // Pointer to the end of the existing expected payload in the original source.
  // Only valid when the case has_expected_section.
  const char* expected_end;
} loom_check_case_update_t;

// Kind of machine-readable text edit needed to apply an update.
typedef enum loom_check_update_edit_kind_e {
  LOOM_CHECK_UPDATE_EDIT_REPLACE_EXPECTED_OUTPUT = 0,
  LOOM_CHECK_UPDATE_EDIT_INSERT_EXPECTED_OUTPUT = 1,
  LOOM_CHECK_UPDATE_EDIT_INSERT_DIAGNOSTIC_ANNOTATIONS = 2,
  LOOM_CHECK_UPDATE_EDIT_DELETE_DIAGNOSTIC_ANNOTATION = 3,
} loom_check_update_edit_kind_t;

// Machine-readable update edit metadata. |range| is a byte range in the
// original source. The replacement text is caller-owned separately so the same
// metadata can be used with different string storage strategies.
typedef struct loom_check_update_edit_t {
  // Kind of text edit to apply.
  loom_check_update_edit_kind_t kind;
  // Half-open byte range in the original source.
  loom_check_source_range_t range;
} loom_check_update_edit_t;

// Returns a stable JSON/text spelling for an update edit kind.
const char* loom_check_update_edit_kind_name(
    loom_check_update_edit_kind_t kind);

// Reconstructs the source with updated expected sections. Writes the
// result into |new_source|. Does NOT write to disk — the caller decides
// whether and where to write. Returns the number of cases that were
// updated via |out_update_count|.
//
// The reconstruction walks the original source, copying unchanged
// regions verbatim and replacing, inserting, or deleting expected sections for
// cases where needs_update is true. Trailing newlines are ensured after each
// inserted/replaced section.
iree_status_t loom_check_apply_updates(iree_string_view_t original_source,
                                       const loom_check_file_t* file,
                                       const loom_check_case_update_t* updates,
                                       iree_string_builder_t* new_source,
                                       iree_host_size_t* out_update_count);

// Builds the exact replacement text and range that loom_check_apply_updates()
// would use for one case. The caller owns |new_text| and |out_edit|. The source
// and parsed case must come from the same buffer.
iree_status_t loom_check_build_update_edit(iree_string_view_t original_source,
                                           const loom_check_case_t* test_case,
                                           iree_string_view_t actual_output,
                                           iree_string_builder_t* new_text,
                                           loom_check_update_edit_t* out_edit);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_TOOLS_LOOM_CHECK_UPDATE_H_
