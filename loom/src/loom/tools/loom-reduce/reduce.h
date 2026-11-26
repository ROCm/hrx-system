// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Rewrites Loom sources to remove the duplication loom-analyze reports.
//
// A plan turns the selected, exactly matching findings of a clone analysis
// into jobs. A job either creates one `func.def inline` motif from the first
// copy of a duplicated sequence and replaces every callable copy with a call,
// or replaces copies of an existing `func.def` body with calls to it.
//
// Scopes decide where new motifs live:
//   - file: at the end of the file holding the copies; no declarations or link
//     inputs change.
//   - corpus: copies are grouped by the first directory of their path (one
//     link group, such as one kernel family); each group's motifs go to one
//     motif file with the declarations their bodies reference, and calling
//     files gain a `func.decl`. Existing motifs are only called from their own
//     file, since linking a definition file elsewhere can bring conflicting
//     definitions with it.
//
// A job is skipped when its rewrite cannot be expressed: a block terminator or
// buffer.alloca in the sequence, a parameter type naming a value that is not
// an earlier parameter, an undeclared referenced symbol, or a call cycle
// through the new motif (inline motifs must not recurse). Applying the plan
// renders every affected file; when a verifier rejects a file, the jobs
// touching it are withdrawn and re-admitted one at a time, largest saving
// first, and those that still fail are reported as reverted.

#ifndef LOOM_TOOLS_LOOM_REDUCE_REDUCE_H_
#define LOOM_TOOLS_LOOM_REDUCE_REDUCE_H_

#include "iree/base/api.h"
#include "loom/tools/loom-analyze/clone_analysis.h"
#include "loom/util/stream.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum loom_reduce_scope_e {
  LOOM_REDUCE_SCOPE_FILE = 0,
  LOOM_REDUCE_SCOPE_CORPUS = 1,
} loom_reduce_scope_t;

typedef struct loom_reduce_options_t {
  loom_reduce_scope_t scope;
  // Motif file name created in each top-level directory (corpus scope).
  iree_string_view_t motif_file_name;
} loom_reduce_options_t;

// One analyzed source file. All views are borrowed for the plan's lifetime.
typedef struct loom_reduce_input_t {
  // Path the module was loaded under; matches occurrence file names.
  iree_string_view_t load_path;
  // Root-relative path with '/' separators, used for output and grouping.
  iree_string_view_t relative_path;
  // Original source text.
  iree_string_view_t text;
  // Verified native module parsed from |text|. Symbol, call and declaration
  // facts come from it rather than from the source text.
  const loom_module_t* module;
} loom_reduce_input_t;

// Checks rewritten text. Sets |*out_accepted| false when it does not parse or
// verify; returns a failure status only for infrastructure errors.
typedef struct loom_reduce_verifier_t {
  iree_status_t (*fn)(void* user_data, iree_string_view_t relative_path,
                      iree_string_view_t text, bool* out_accepted);
  void* user_data;
} loom_reduce_verifier_t;

typedef struct loom_reduce_summary_t {
  // Applied jobs.
  uint32_t motif_count;
  // Copies replaced by calls in applied jobs.
  uint32_t replaced_copy_count;
  // Operations removed by applied jobs.
  uint64_t removed_op_count;
  // Files the applied jobs rewrite or create.
  uint32_t changed_file_count;
  // Findings and jobs not applied because their rewrite is not expressible.
  uint32_t skipped_count;
  // Jobs withdrawn because a rewritten file failed verification.
  uint32_t reverted_count;
  // Rewritten files that now need additional link inputs.
  uint32_t linked_file_count;
} loom_reduce_summary_t;

typedef struct loom_reduce_plan_t loom_reduce_plan_t;

// Builds jobs from |result|. |inputs| and |result| are borrowed for the plan's
// lifetime.
iree_status_t loom_reduce_plan_create(const loom_reduce_options_t* options,
                                      const loom_reduce_input_t* inputs,
                                      iree_host_size_t input_count,
                                      const loom_analyze_result_t* result,
                                      iree_allocator_t allocator,
                                      loom_reduce_plan_t** out_plan);

void loom_reduce_plan_free(loom_reduce_plan_t* plan);

// Validates jobs, rejects call cycles, renders, and verifies with |verifier|
// when its function is non-NULL. May be called once.
iree_status_t loom_reduce_plan_apply(loom_reduce_plan_t* plan,
                                     loom_reduce_verifier_t verifier);

// Returns the number of rewritten or created files after apply.
iree_host_size_t loom_reduce_plan_output_count(const loom_reduce_plan_t* plan);

// Returns one rewritten file: its root-relative path and full text.
void loom_reduce_plan_output(const loom_reduce_plan_t* plan,
                             iree_host_size_t index,
                             iree_string_view_t* out_relative_path,
                             iree_string_view_t* out_text);

void loom_reduce_plan_summary(const loom_reduce_plan_t* plan,
                              loom_reduce_summary_t* out_summary);

// Writes the JSON report: summary, every job with status and reason, skipped
// findings, and changed files. |op_count| is the analyzed operation total.
iree_status_t loom_reduce_plan_write_report(const loom_reduce_plan_t* plan,
                                            uint64_t op_count,
                                            loom_output_stream_t* stream);

// Writes `{"requires": {file: [link inputs...]}}` for rewritten files that
// call motifs defined in other files.
iree_status_t loom_reduce_plan_write_links(const loom_reduce_plan_t* plan,
                                           loom_output_stream_t* stream);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLS_LOOM_REDUCE_REDUCE_H_
