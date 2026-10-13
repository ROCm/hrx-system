// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TARGET_REPORTING_MEMORY_SUMMARY_TEST_UTIL_H_
#define LOOM_TARGET_REPORTING_MEMORY_SUMMARY_TEST_UTIL_H_

#include "iree/base/internal/arena.h"
#include "loom/target/reporting/memory_summary.h"

namespace loom::testing {

// Models one function capture, with construction state released independently
// of the report that owns its finished rows and counters.
class MemoryReportBatch {
 public:
  explicit MemoryReportBatch(loom_target_compile_report_t* report) {
    iree_arena_block_pool_initialize(64 * 1024, iree_allocator_system(),
                                     &pool_);
    iree_arena_initialize(&pool_, &arena_);
    loom_target_compile_report_memory_batch_initialize(report, &arena_,
                                                       &batch_);
  }
  ~MemoryReportBatch() {
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&pool_);
  }
  MemoryReportBatch(const MemoryReportBatch&) = delete;
  MemoryReportBatch& operator=(const MemoryReportBatch&) = delete;

  iree_status_t Append(
      const loom_target_compile_report_source_low_memory_row_t* row) {
    return loom_target_compile_report_memory_batch_append(&batch_, row);
  }

  iree_status_t Finish() {
    iree_status_t status =
        loom_target_compile_report_memory_batch_finish(&batch_);
    iree_arena_reset(&arena_);
    return status;
  }

 private:
  // Pool shared by the batch and its per-root scopes.
  iree_arena_block_pool_t pool_;
  // Owns the batch's construction-only packet references.
  iree_arena_allocator_t arena_;
  // Aggregation state borrowing the destination report.
  loom_target_compile_report_memory_batch_t batch_;
};

}  // namespace loom::testing

#endif  // LOOM_TARGET_REPORTING_MEMORY_SUMMARY_TEST_UTIL_H_
