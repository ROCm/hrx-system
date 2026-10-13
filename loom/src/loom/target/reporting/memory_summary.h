// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TARGET_REPORTING_MEMORY_SUMMARY_H_
#define LOOM_TARGET_REPORTING_MEMORY_SUMMARY_H_

#include "iree/base/internal/arena.h"
#include "loom/target/reporting/report.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Scoped aggregation of one lowered function's memory packets. Function names
// belong to one batch in a report; separately compiled entries are combined by
// record_entry_report, which keeps their private expression identities
// separate. Rows and counters belong to the report. All indexing state belongs
// to |arena| and can be released immediately after finish, including on
// allocation failure. Producer-owned packet intervals are nonempty; expression
// IDs with corresponding precision flags never use UINT32_MAX.
typedef struct loom_target_compile_report_memory_batch_t {
  // Destination retaining detail rows and completed summaries.
  loom_target_compile_report_t* report;
  // Caller-owned scratch arena, valid through finish.
  iree_arena_allocator_t* arena;
  // Stable scratch references to the rows and their owning summary groups.
  loom_target_compile_report_row_list_t entries;
} loom_target_compile_report_memory_batch_t;

// Begins a function's batch without allocating storage.
void loom_target_compile_report_memory_batch_initialize(
    loom_target_compile_report_t* report, iree_arena_allocator_t* arena,
    loom_target_compile_report_memory_batch_t* out_batch);

// Retains one packet and accumulates its structural and dynamic counters.
iree_status_t loom_target_compile_report_memory_batch_append(
    loom_target_compile_report_memory_batch_t* batch,
    const loom_target_compile_report_source_low_memory_row_t* row);

// Completes unique-byte proofs in encounter order with O(n log n) interval work
// and O(n) scratch. Called once after the function's last packet; the batch is
// not used afterward. No index storage is retained by the destination report.
iree_status_t loom_target_compile_report_memory_batch_finish(
    loom_target_compile_report_memory_batch_t* batch);

// Accumulates structural bank-service counters from |source| into |target|.
void loom_target_compile_report_accumulate_bank_service_summaries(
    loom_target_compile_report_bank_service_summary_t* target,
    const loom_target_compile_report_bank_service_summary_t* source);

// Accumulates subgroup-access counters from |source| into |target|.
void loom_target_compile_report_accumulate_subgroup_access_summaries(
    loom_target_compile_report_subgroup_access_summary_t* target,
    const loom_target_compile_report_subgroup_access_summary_t* source);

// Accumulates source-memory summary counters from |source| into |target|.
void loom_target_compile_report_accumulate_source_low_memory_summaries(
    loom_target_compile_report_source_low_memory_summary_t* target,
    const loom_target_compile_report_source_low_memory_summary_t* source);

// Merges source-memory detail rows from |source| into |target|.
iree_status_t loom_target_compile_report_merge_source_low_memory_details(
    loom_target_compile_report_t* target,
    const loom_target_compile_report_t* source);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // LOOM_TARGET_REPORTING_MEMORY_SUMMARY_H_
