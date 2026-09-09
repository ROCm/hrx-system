// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/allocation/scalar_packing.h"

#include <algorithm>
#include <vector>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace loom {
namespace {

loom_liveness_interval_t Interval(uint32_t start, uint32_t end, uint32_t units,
                                  uint16_t class_id) {
  loom_liveness_interval_t interval = {};
  interval.start_point = start;
  interval.end_point = end;
  interval.unit_count = units;
  interval.value_class.type_kind = LOOM_TYPE_REGISTER;
  interval.value_class.register_class_id = class_id;
  return interval;
}

class LowAllocationScalarPackingTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(), &pool_);
    iree_arena_initialize(&pool_, &arena_);
    classes_[2].flags = LOOM_LOW_REG_CLASS_FLAG_PHYSICAL |
                        LOOM_LOW_REG_CLASS_FLAG_EXPLICIT_PHYSICAL_REGISTERS;
    descriptor_set_.reg_classes = classes_;
    descriptor_set_.reg_class_count = IREE_ARRAYSIZE(classes_);
    for (uint16_t i = 0; i < IREE_ARRAYSIZE(summaries_); ++i) {
      summaries_[i].value_class.type_kind = LOOM_TYPE_REGISTER;
      summaries_[i].value_class.register_class_id = i;
      summaries_[i].peak_live_units = 8 + i;
    }
  }
  void TearDown() override {
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&pool_);
  }

  iree_status_t Build(const std::vector<loom_liveness_interval_t>& intervals,
                      loom_low_allocation_scalar_packing_t* out_packing) {
    loom_liveness_analysis_t liveness = {};
    liveness.intervals = intervals.data();
    liveness.interval_count = intervals.size();
    liveness.pressure_summaries = summaries_;
    liveness.pressure_summary_count = IREE_ARRAYSIZE(summaries_);
    loom_low_allocation_interval_order_t order = {};
    IREE_RETURN_IF_ERROR(
        loom_low_allocation_interval_order_build(&liveness, &arena_, &order));
    return loom_low_allocation_scalar_packing_build(
        &descriptor_set_, &liveness, &order, &arena_, out_packing);
  }

  // Pool shared by the fixture's planning arena.
  iree_arena_block_pool_t pool_;
  // Planning storage released after each independent fixture.
  iree_arena_allocator_t arena_;
  // Two numbered classes and one class with explicit physical-ID geometry.
  loom_low_reg_class_t classes_[3] = {};
  // Target contract for the three classes.
  loom_low_descriptor_set_t descriptor_set_ = {};
  // Distinct per-class frontiers make cross-class leakage observable.
  loom_liveness_pressure_summary_t summaries_[3] = {};
};

TEST_F(LowAllocationScalarPackingTest, MatchesStorageOverlapOracle) {
  std::vector<loom_liveness_interval_t> intervals;
  for (uint16_t class_id = 0; class_id < 3; ++class_id) {
    for (uint32_t start = 0; start < 12; ++start) {
      intervals.push_back(Interval(start, start, 1, class_id));
      intervals.push_back(Interval(start, start + 1, 1, class_id));
      intervals.push_back(Interval(start, start + 4, 1, class_id));
    }
    intervals.push_back(Interval(2, 4, 4, class_id));
    intervals.push_back(Interval(6, 10, 2, class_id));
    intervals.push_back(Interval(11, 11, 4, class_id));
  }
  std::reverse(intervals.begin(), intervals.end());
  loom_low_allocation_scalar_packing_t packing = {};
  IREE_ASSERT_OK(Build(intervals, &packing));
  ASSERT_EQ(packing.overlapping_intervals.bit_count, intervals.size());
  for (size_t i = 0; i < intervals.size(); ++i) {
    const auto& scalar = intervals[i];
    bool expected = false;
    if (scalar.unit_count == 1 && scalar.value_class.register_class_id != 2) {
      for (const auto& aggregate : intervals) {
        if (aggregate.unit_count == 1 ||
            aggregate.value_class.register_class_id !=
                scalar.value_class.register_class_id) {
          continue;
        }
        expected |= scalar.start_point < std::max(aggregate.start_point + 1,
                                                  aggregate.end_point) &&
                    aggregate.start_point <
                        std::max(scalar.start_point + 1, scalar.end_point);
      }
    }
    EXPECT_EQ(iree_bitmap_test(packing.overlapping_intervals, i), expected)
        << "interval " << i;
  }
  EXPECT_EQ(packing.frontiers_by_reg_class[0], 8u);
  EXPECT_EQ(packing.frontiers_by_reg_class[1], 9u);
  EXPECT_EQ(packing.frontiers_by_reg_class[2], 0u);
}

TEST_F(LowAllocationScalarPackingTest,
       ScalarOnlyAndPhysicalIdClassesNeedNoPlan) {
  loom_low_allocation_scalar_packing_t packing = {};
  IREE_ASSERT_OK(
      Build({Interval(0, 4, 1, 0), Interval(0, 4, 1, 1), Interval(0, 4, 4, 2)},
            &packing));
  EXPECT_EQ(packing.overlapping_intervals.bit_count, 0u);
  EXPECT_EQ(packing.overlapping_intervals.words, nullptr);
  EXPECT_EQ(packing.frontiers_by_reg_class, nullptr);
}

}  // namespace
}  // namespace loom
