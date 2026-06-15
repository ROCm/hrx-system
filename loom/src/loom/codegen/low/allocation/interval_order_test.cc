// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/allocation/interval_order.h"

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace loom {
namespace {

class LowAllocationIntervalOrderTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    iree_arena_initialize(&block_pool_, &arena_);
  }

  void TearDown() override {
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  static loom_liveness_interval_t RegisterInterval(loom_value_id_t value_id,
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

  static loom_liveness_interval_t ScalarInterval(loom_value_id_t value_id) {
    loom_liveness_interval_t interval = {};
    interval.value_id = value_id;
    interval.value_class.type_kind = LOOM_TYPE_SCALAR;
    interval.unit_count = 1;
    return interval;
  }

  iree_arena_block_pool_t block_pool_;
  iree_arena_allocator_t arena_;
};

TEST_F(LowAllocationIntervalOrderTest, FiltersNonAllocatableIntervals) {
  loom_liveness_interval_t intervals[] = {
      ScalarInterval(/*value_id=*/1),
      RegisterInterval(/*value_id=*/2, /*start_point=*/0, /*end_point=*/1,
                       /*unit_count=*/0),
  };
  loom_liveness_analysis_t liveness = {};
  liveness.intervals = intervals;
  liveness.interval_count = IREE_ARRAYSIZE(intervals);

  loom_low_allocation_interval_order_t order = {};
  IREE_ASSERT_OK(
      loom_low_allocation_interval_order_build(&liveness, &arena_, &order));
  EXPECT_EQ(order.intervals, nullptr);
  EXPECT_EQ(order.interval_count, 0u);
  EXPECT_EQ(order.unit_count, 0u);
}

TEST_F(LowAllocationIntervalOrderTest, SortsByStartEndAndValueId) {
  loom_liveness_interval_t intervals[] = {
      RegisterInterval(/*value_id=*/4, /*start_point=*/7, /*end_point=*/9,
                       /*unit_count=*/1),
      RegisterInterval(/*value_id=*/3, /*start_point=*/3, /*end_point=*/8,
                       /*unit_count=*/2),
      RegisterInterval(/*value_id=*/2, /*start_point=*/3, /*end_point=*/6,
                       /*unit_count=*/4),
      RegisterInterval(/*value_id=*/1, /*start_point=*/3, /*end_point=*/6,
                       /*unit_count=*/8),
      ScalarInterval(/*value_id=*/0),
  };
  loom_liveness_analysis_t liveness = {};
  liveness.intervals = intervals;
  liveness.interval_count = IREE_ARRAYSIZE(intervals);

  loom_low_allocation_interval_order_t order = {};
  IREE_ASSERT_OK(
      loom_low_allocation_interval_order_build(&liveness, &arena_, &order));
  ASSERT_NE(order.intervals, nullptr);
  ASSERT_EQ(order.interval_count, 4u);
  EXPECT_EQ(order.unit_count, 15u);
  EXPECT_EQ(order.intervals[0]->value_id, 1u);
  EXPECT_EQ(order.intervals[1]->value_id, 2u);
  EXPECT_EQ(order.intervals[2]->value_id, 3u);
  EXPECT_EQ(order.intervals[3]->value_id, 4u);
}

}  // namespace
}  // namespace loom
