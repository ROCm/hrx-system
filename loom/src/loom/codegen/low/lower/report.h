// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Source-to-Low lowering report construction.
//
// Reports are an optional cold path layered over the normal lowering walk.
// Selection rows describe the plan chosen for each source op, while memory
// rows retain target-provided packet geometry and source interval evidence.
// Row storage is owned by loom_low_lower_result_t and remains valid until the
// result is deinitialized. Per-function analysis state is arena-backed and
// lives only for the active lowering call.

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
typedef struct loom_low_lower_selected_plan_t loom_low_lower_selected_plan_t;
typedef struct loom_low_source_memory_access_plan_t
    loom_low_source_memory_access_plan_t;
typedef struct loom_op_t loom_op_t;

// Function-local state used only when report rows are requested.
typedef struct loom_low_lower_report_state_t {
  // Source CFG block execution counts indexed by source block ordinal.
  uint64_t* source_block_execution_counts;
  // True after source block execution count analysis has run.
  bool source_block_execution_counts_initialized;
  // True when every reachable source CFG backedge was counted exactly.
  bool source_block_execution_counts_exact;
  // Interned symbolic byte expressions used by memory interval rows.
  loom_low_lower_memory_expression_entry_t* memory_expression_entries;
  // Number of populated entries in |memory_expression_entries|.
  iree_host_size_t memory_expression_entry_count;
  // Number of allocated entries in |memory_expression_entries|.
  iree_host_size_t memory_expression_entry_capacity;
} loom_low_lower_report_state_t;

// Releases all report row storage owned by |result|.
void loom_low_lower_result_deinitialize(loom_low_lower_result_t* result);

// Records the report row for one source lowering plan after emission.
// The caller must have enabled report rows in the lowering options.
iree_status_t loom_low_lower_report_record_selected_plan(
    loom_low_lower_context_t* context,
    const loom_low_lower_selected_plan_t* selected_plan,
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

// Records an emitted source-memory packet report row.
iree_status_t loom_low_lower_record_memory_report_row(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_low_lower_memory_report_row_t* row);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_LOWER_REPORT_H_
