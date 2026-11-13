// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/reporting/pipeline_plan.h"

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/target/reporting/report.h"

namespace loom {
namespace {

TEST(CompileReportPipelinePlanTest, GrowsInArenaWithoutRelocatingEarlierPlans) {
  iree_arena_block_pool_t pool;
  iree_arena_block_pool_initialize(16 * 1024, iree_allocator_system(), &pool);
  iree_arena_allocator_t arena;
  iree_arena_initialize(&pool, &arena);
  for (bool details : {false, true}) {
    loom_target_compile_report_t report;
    loom_target_compile_report_initialize(iree_arena_allocator(&arena),
                                          &report);
    report.requested_detail_flags =
        details ? LOOM_TARGET_COMPILE_REPORT_DETAIL_PIPELINE_PLAN_ROWS : 0;
    const loom_target_compile_report_pipeline_worker_row_t worker = {
        .code_byte_count = 32,
    };
    loom_target_compile_report_pipeline_plan_t plan = {
        .worker_rows = &worker,
        .worker_row_count = 1,
    };
    const void* first_plan = nullptr;
    for (uint32_t i = 0; i < 64; ++i) {
      plan.summary.worker_count = i;
      IREE_ASSERT_OK(
          loom_target_compile_report_record_pipeline_plan(&report, &plan));
      const void* current_first =
          loom_target_compile_report_vec_const_rows(report.pipeline_plans.head);
      if (i == 0) {
        first_plan = current_first;
      } else {
        EXPECT_EQ(current_first, first_plan);
      }
    }
    ASSERT_EQ(report.pipeline_plans.count, 64u);
    ASSERT_NE(report.pipeline_plans.head, report.pipeline_plans.tail);
    uint32_t index = 0;
    for (const auto* block = report.pipeline_plans.head; block;
         block = block->next) {
      const auto* plans =
          static_cast<const loom_target_compile_report_pipeline_plan_t*>(
              loom_target_compile_report_vec_const_rows(block));
      for (iree_host_size_t i = 0; i < block->count; ++i, ++index) {
        EXPECT_EQ(plans[i].summary.worker_count, index);
        EXPECT_EQ(plans[i].worker_row_count, details ? 1u : 0u);
        if (details) {
          EXPECT_NE(plans[i].worker_rows, &worker);
          EXPECT_EQ(plans[i].worker_rows[0].code_byte_count, 32u);
        }
      }
    }
    EXPECT_EQ(index, 64u);
    EXPECT_EQ(arena.allocation_head, nullptr);
    loom_target_compile_report_deinitialize(&report);
    iree_arena_reset(&arena);
  }
  iree_arena_deinitialize(&arena);
  iree_arena_block_pool_deinitialize(&pool);
}

}  // namespace
}  // namespace loom
