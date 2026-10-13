// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Source-to-Low lowering report construction.
//
// Reports are an optional cold path layered over the normal lowering plan.
// Selection rows are constructed from finalized decisions before emission;
// instruction counts are execution outcomes populated afterward. Memory
// rows retain target-provided packet geometry and source interval evidence.
// Row storage is owned by loom_low_lower_result_t and remains valid until the
// result is deinitialized. Per-function expression indexing is analysis-arena
// scratch retired before emission; only expression IDs survive in report rows.

#ifndef LOOM_CODEGEN_LOW_LOWER_REPORT_H_
#define LOOM_CODEGEN_LOW_LOWER_REPORT_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_low_lower_context_t loom_low_lower_context_t;
typedef struct loom_low_lower_memory_expression_entry_t
    loom_low_lower_memory_expression_entry_t;
typedef struct loom_low_lower_memory_report_row_t
    loom_low_lower_memory_report_row_t;
typedef struct loom_low_lower_result_t loom_low_lower_result_t;
typedef struct loom_low_lower_report_row_vec_t loom_low_lower_report_row_vec_t;
typedef struct loom_low_lower_selected_plan_t loom_low_lower_selected_plan_t;
typedef struct loom_low_source_memory_access_plan_t
    loom_low_source_memory_access_plan_t;
typedef struct loom_op_t loom_op_t;

// Function-local state used only when report rows are requested.
typedef struct loom_low_lower_report_state_t {
  // Number of operations finalized by the function's emission builder.
  uint64_t emitted_op_count;
  // Planned selection row block receiving the next emission outcome.
  loom_low_lower_report_row_vec_t* selection_cursor;
  // Next row within |selection_cursor|, excluding claimed source operations.
  iree_host_size_t selection_index;
  // Function-local symbolic byte expressions used by memory interval rows.
  struct {
    // Root of the analysis-arena-owned balanced expression tree.
    loom_low_lower_memory_expression_entry_t* root;
    // Number of unique expressions; also the next first-seen expression ID.
    uint32_t count;
  } memory_expressions;
} loom_low_lower_report_state_t;

// Installs emission accounting on the initialized function builder when
// selection reports are enabled. Builder copies retain the same report owner.
void loom_low_lower_report_initialize(loom_low_lower_context_t* context);

// Releases all report row storage owned by |result|.
void loom_low_lower_result_deinitialize(loom_low_lower_result_t* result);

// Projects finalized source plans and their execution evidence into report
// rows. No row storage or report-only analysis is created when reports are
// disabled.
iree_status_t loom_low_lower_report_prepare(loom_low_lower_context_t* context);

// Records an actual instruction count in the next planned selection row.
// Claimed operations have no row; elided operations consume a row with count
// zero.
void loom_low_lower_report_record_emission(loom_low_lower_context_t* context,
                                           uint32_t emitted_low_op_count);

// Returns exact source execution evidence for an operation when loop and CFG
// facts can prove it without target execution.
iree_status_t loom_low_lower_source_op_execution_count_plus_one(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    uint64_t* out_execution_count_plus_one);

// Populates row source interval evidence from |source_plan| and interns exact
// symbolic interval endpoints when report-only accounting can prove them.
iree_status_t loom_low_lower_memory_report_row_populate_source_interval(
    loom_low_lower_context_t* context,
    const loom_low_source_memory_access_plan_t* source_plan,
    loom_low_lower_memory_report_row_t* row);

// Records a complete planned source-memory row while reporting is enabled.
// Execution frequency is supplied by the owning selection row.
iree_status_t loom_low_lower_record_memory_report_row(
    loom_low_lower_context_t* context,
    const loom_low_lower_memory_report_row_t* row);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_LOWER_REPORT_H_
