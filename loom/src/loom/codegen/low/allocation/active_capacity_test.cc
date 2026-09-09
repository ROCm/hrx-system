// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/allocation/active_capacity.h"

#include <algorithm>
#include <numeric>
#include <utility>
#include <vector>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace loom {
namespace {

loom_liveness_interval_t Interval(uint32_t start, uint32_t end, uint32_t units,
                                  uint16_t class_id = 0) {
  loom_liveness_interval_t interval = {};
  interval.value_class.type_kind = LOOM_TYPE_REGISTER;
  interval.value_class.register_class_id = class_id;
  interval.start_point = start;
  interval.end_point = end;
  interval.unit_count = units;
  return interval;
}

class LowAllocationActiveCapacityTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(), &pool_);
    classes_[1].flags = LOOM_LOW_REG_CLASS_FLAG_PHYSICAL |
                        LOOM_LOW_REG_CLASS_FLAG_EXPLICIT_PHYSICAL_REGISTERS;
    classes_[1].physical_atomic_unit_count = 2;
    classes_[2].flags = classes_[1].flags;
    classes_[2].physical_atomic_unit_count = 8;
    // Unused classes do not contribute to the bound.
    classes_[3].flags = classes_[1].flags;
    classes_[3].physical_atomic_unit_count = 64;
  }
  void TearDown() override { iree_arena_block_pool_deinitialize(&pool_); }

  void ExpectCapacity(std::vector<loom_liveness_interval_t> intervals,
                      size_t expected_units, size_t expected_points,
                      std::vector<std::pair<uint32_t, uint32_t>> edges = {},
                      std::vector<uint32_t> refined_ends = {}) {
    std::vector<uint32_t> interval_indices(intervals.size());
    std::iota(interval_indices.begin(), interval_indices.end(), 0);
    std::vector<uint32_t> unit_offsets;
    std::vector<uint32_t> unit_starts;
    std::vector<uint32_t> unit_ends;
    for (const auto& interval : intervals) {
      unit_offsets.push_back(static_cast<uint32_t>(unit_ends.size()));
      unit_starts.insert(unit_starts.end(), interval.unit_count,
                         interval.start_point);
      unit_ends.insert(unit_ends.end(), interval.unit_count,
                       std::max(interval.start_point + 1, interval.end_point));
    }
    if (!refined_ends.empty()) {
      ASSERT_EQ(refined_ends.size(), unit_ends.size());
      unit_ends = std::move(refined_ends);
    }
    loom_liveness_analysis_t liveness = {};
    liveness.intervals = intervals.data();
    liveness.interval_count = intervals.size();
    liveness.value_count = intervals.size();
    liveness.value_interval_indices = interval_indices.data();
    loom_low_allocation_unit_liveness_t unit_liveness = {};
    unit_liveness.point_starts_by_value_ordinal = unit_offsets.data();
    unit_liveness.start_points = unit_starts.data();
    unit_liveness.end_points = unit_ends.data();
    unit_liveness.point_count = unit_ends.size();

    // Build both adjacency directions used by production placement tables.
    std::sort(edges.begin(), edges.end());
    std::vector<loom_low_placement_relation_t> relations(edges.size());
    std::vector<loom_low_placement_relation_range_t> result_ranges(
        intervals.size());
    std::vector<loom_low_placement_relation_range_t> source_ranges(
        intervals.size());
    std::vector<uint32_t> source_indices(edges.size());
    for (size_t i = 0; i < edges.size(); ++i) {
      auto& relation = relations[i];
      relation.result_ordinal = edges[i].first;
      relation.source_ordinal = edges[i].second;
      auto& range = result_ranges[relation.result_ordinal];
      if (range.count++ == 0) range.start = static_cast<uint32_t>(i);
      source_indices[i] = static_cast<uint32_t>(i);
    }
    std::sort(source_indices.begin(), source_indices.end(),
              [&](uint32_t a, uint32_t b) {
                return relations[a].source_ordinal <
                       relations[b].source_ordinal;
              });
    for (size_t i = 0; i < source_indices.size(); ++i) {
      auto& range = source_ranges[relations[source_indices[i]].source_ordinal];
      if (range.count++ == 0) range.start = static_cast<uint32_t>(i);
    }
    loom_low_placement_table_t placement = {};
    placement.value_count = static_cast<loom_value_ordinal_t>(intervals.size());
    placement.relations = relations.data();
    placement.relation_count = relations.size();
    placement.ranges_by_result_ordinal = result_ranges.data();
    placement.ranges_by_source_ordinal = source_ranges.data();
    placement.relation_indices_by_source_ordinal = source_indices.data();
    loom_low_descriptor_set_t descriptor_set = {};
    descriptor_set.reg_classes = classes_;
    descriptor_set.reg_class_count = IREE_ARRAYSIZE(classes_);
    loom_low_allocation_active_capacity_t capacity = {};
    IREE_ASSERT_OK(loom_low_allocation_active_capacity_calculate(
        &descriptor_set, &liveness, &unit_liveness, &placement, &pool_,
        &capacity));
    EXPECT_EQ(capacity.unit_count, expected_units);
    EXPECT_EQ(capacity.program_point_count, expected_points);
  }

  // Pool shared by the bounded calculation's private scratch arena.
  iree_arena_block_pool_t pool_;
  // Scalar, narrow explicit, wide explicit, and unused explicit classes.
  loom_low_reg_class_t classes_[4] = {};
};

TEST_F(LowAllocationActiveCapacityTest, UsesOverlappingClassWidths) {
  ExpectCapacity({Interval(0, 5, 100), Interval(5, 9, 3, 1),
                  Interval(2, 7, 2, 2), Interval(10, 10, 1)},
                 /*expected_units=*/116, /*expected_points=*/12);
}

TEST_F(LowAllocationActiveCapacityTest, ReservesConnectedResultsEarly) {
  const std::vector<loom_liveness_interval_t> intervals = {
      Interval(0, 2, 1), Interval(6, 10, 2), Interval(3, 5, 16)};
  ExpectCapacity(intervals, /*expected_units=*/16, /*expected_points=*/11);
  ExpectCapacity(intervals, /*expected_units=*/18, /*expected_points=*/11,
                 /*edges=*/{{1, 0}});
}

TEST_F(LowAllocationActiveCapacityTest, TraversesBothDirectionsAndCycles) {
  // The root has the latest start; a cycle and a reverse edge reach the early
  // definition. Unconnected values retain their own start.
  ExpectCapacity({Interval(8, 10, 4), Interval(4, 6, 2), Interval(0, 2, 1),
                  Interval(3, 5, 8)},
                 /*expected_units=*/14, /*expected_points=*/11,
                 /*edges=*/{{0, 1}, {1, 0}, {2, 1}});
}

TEST_F(LowAllocationActiveCapacityTest, IgnoresNonallocatableComponents) {
  loom_liveness_interval_t nonallocatable = {};
  ExpectCapacity({nonallocatable, nonallocatable, Interval(2, 4, 3)},
                 /*expected_units=*/3, /*expected_points=*/5,
                 /*edges=*/{{0, 1}, {1, 0}});
}

TEST_F(LowAllocationActiveCapacityTest, IncludesRefinedStorageEnds) {
  ExpectCapacity({Interval(0, 2, 2), Interval(3, 5, 4)},
                 /*expected_units=*/6, /*expected_points=*/9,
                 /*edges=*/{}, /*refined_ends=*/{2, 8, 5, 5, 5, 5});
}

TEST_F(LowAllocationActiveCapacityTest, ReleasesBeforeSamePointActivation) {
  ExpectCapacity({Interval(0, 2, 4), Interval(2, 4, 8), Interval(4, 6, 2)},
                 /*expected_units=*/8, /*expected_points=*/7);
  ExpectCapacity({}, /*expected_units=*/0, /*expected_points=*/1);
}

}  // namespace
}  // namespace loom
