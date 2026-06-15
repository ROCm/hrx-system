// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/allocation/unit_liveness.h"

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace loom {
namespace {

loom_liveness_interval_t RegisterInterval(loom_value_id_t value_id,
                                          uint32_t start_point,
                                          uint32_t end_point,
                                          uint32_t unit_count) {
  loom_liveness_interval_t interval = {};
  interval.value_id = value_id;
  interval.value_class.type_kind = LOOM_TYPE_REGISTER;
  interval.start_point = start_point;
  interval.end_point = end_point;
  interval.unit_count = unit_count;
  return interval;
}

loom_liveness_analysis_t Liveness(const loom_value_id_t* value_ids,
                                  iree_host_size_t value_count,
                                  const uint32_t* value_interval_indices,
                                  const loom_liveness_interval_t* intervals,
                                  iree_host_size_t interval_count,
                                  const loom_liveness_block_info_t* blocks,
                                  iree_host_size_t block_count) {
  loom_liveness_analysis_t liveness = {};
  liveness.value_ids = value_ids;
  liveness.value_count = value_count;
  liveness.value_interval_indices = value_interval_indices;
  liveness.intervals = intervals;
  liveness.interval_count = interval_count;
  liveness.blocks = blocks;
  liveness.block_count = block_count;
  return liveness;
}

TEST(LowAllocationUnitLivenessTest, InitializesUnitStartsAndBoundaryUses) {
  iree_arena_block_pool_t block_pool;
  iree_arena_block_pool_initialize(/*block_size=*/4096, iree_allocator_system(),
                                   &block_pool);
  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool, &arena);

  loom_module_t module = {};
  loom_region_t body = {};
  loom_low_resolved_target_t target = {};
  const loom_value_id_t value_ids[] = {10, 20};
  const uint32_t value_interval_indices[] = {0, 1};
  const loom_liveness_interval_t intervals[] = {
      RegisterInterval(/*value_id=*/10, /*start_point=*/2,
                       /*end_point=*/2, /*unit_count=*/2),
      RegisterInterval(/*value_id=*/20, /*start_point=*/4,
                       /*end_point=*/8, /*unit_count=*/1),
  };
  const loom_value_id_t live_in_values[] = {10};
  const loom_liveness_block_info_t blocks[] = {
      {
          /*.block=*/nullptr,
          /*.start_point=*/5,
          /*.end_point=*/9,
          /*.live_in_values=*/live_in_values,
          /*.live_in_count=*/IREE_ARRAYSIZE(live_in_values),
          /*.live_out_values=*/nullptr,
          /*.live_out_count=*/0,
      },
  };
  const loom_liveness_analysis_t liveness = Liveness(
      value_ids, IREE_ARRAYSIZE(value_ids), value_interval_indices, intervals,
      IREE_ARRAYSIZE(intervals), blocks, IREE_ARRAYSIZE(blocks));

  loom_low_allocation_unit_liveness_t unit_liveness = {};
  IREE_ASSERT_OK(loom_low_allocation_unit_liveness_initialize(
      &module, &body, &target, loom_liveness_order_empty(), &liveness, &arena,
      &unit_liveness));

  EXPECT_EQ(loom_low_allocation_unit_liveness_end_point_start_for_value_ordinal(
                &unit_liveness, &liveness, /*value_ordinal=*/0),
            0u);
  EXPECT_EQ(loom_low_allocation_unit_liveness_end_point_start_for_value_ordinal(
                &unit_liveness, &liveness, /*value_ordinal=*/1),
            2u);
  ASSERT_EQ(unit_liveness.end_point_count, 3u);
  EXPECT_EQ(unit_liveness.end_points[0], 6u);
  EXPECT_EQ(unit_liveness.end_points[1], 6u);
  EXPECT_EQ(unit_liveness.end_points[2], 4u);

  iree_arena_deinitialize(&arena);
  iree_arena_block_pool_deinitialize(&block_pool);
}

TEST(LowAllocationUnitLivenessTest, ExtendsTiedResultSourceUnits) {
  iree_arena_block_pool_t block_pool;
  iree_arena_block_pool_initialize(/*block_size=*/4096, iree_allocator_system(),
                                   &block_pool);
  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool, &arena);

  loom_module_t module = {};
  loom_region_t body = {};
  loom_low_resolved_target_t target = {};
  const loom_value_id_t value_ids[] = {10, 20};
  const uint32_t value_interval_indices[] = {0, 1};
  const loom_liveness_interval_t intervals[] = {
      RegisterInterval(/*value_id=*/10, /*start_point=*/0,
                       /*end_point=*/5, /*unit_count=*/2),
      RegisterInterval(/*value_id=*/20, /*start_point=*/4,
                       /*end_point=*/7, /*unit_count=*/2),
  };
  const loom_value_id_t live_out_values[] = {20};
  const loom_liveness_block_info_t blocks[] = {
      {
          /*.block=*/nullptr,
          /*.start_point=*/0,
          /*.end_point=*/7,
          /*.live_in_values=*/nullptr,
          /*.live_in_count=*/0,
          /*.live_out_values=*/live_out_values,
          /*.live_out_count=*/IREE_ARRAYSIZE(live_out_values),
      },
  };
  const loom_liveness_analysis_t liveness = Liveness(
      value_ids, IREE_ARRAYSIZE(value_ids), value_interval_indices, intervals,
      IREE_ARRAYSIZE(intervals), blocks, IREE_ARRAYSIZE(blocks));

  loom_low_allocation_unit_liveness_t unit_liveness = {};
  IREE_ASSERT_OK(loom_low_allocation_unit_liveness_initialize(
      &module, &body, &target, loom_liveness_order_empty(), &liveness, &arena,
      &unit_liveness));
  ASSERT_EQ(unit_liveness.end_point_count, 4u);
  EXPECT_EQ(unit_liveness.end_points[0], 0u);
  EXPECT_EQ(unit_liveness.end_points[1], 0u);
  EXPECT_EQ(unit_liveness.end_points[2], 8u);
  EXPECT_EQ(unit_liveness.end_points[3], 8u);

  const loom_low_placement_relation_t relations[] = {
      {
          /*.op=*/nullptr,
          /*.result_ordinal=*/1,
          /*.source_ordinal=*/0,
          /*.result_unit_offset=*/0,
          /*.source_unit_offset=*/0,
          /*.unit_count=*/2,
          /*.kind=*/LOOM_LOW_PLACEMENT_RELATION_SAME_STORAGE,
          /*.cause=*/LOOM_LOW_PLACEMENT_CAUSE_TIED_RESULT,
          /*.flags=*/LOOM_LOW_PLACEMENT_RELATION_FLAG_HARD,
      },
  };
  loom_low_placement_table_t placement = {};
  placement.relations = relations;
  placement.relation_count = IREE_ARRAYSIZE(relations);

  IREE_ASSERT_OK(loom_low_allocation_unit_liveness_extend_for_tied_results(
      &unit_liveness, &liveness, &placement));
  EXPECT_EQ(unit_liveness.end_points[0], 8u);
  EXPECT_EQ(unit_liveness.end_points[1], 8u);

  iree_arena_deinitialize(&arena);
  iree_arena_block_pool_deinitialize(&block_pool);
}

}  // namespace
}  // namespace loom
