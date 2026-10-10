// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/allocation/interval_order.h"

#include <vector>

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
    classes_[0].flags = LOOM_LOW_REG_CLASS_FLAG_PHYSICAL;
    classes_[1].flags = LOOM_LOW_REG_CLASS_FLAG_PHYSICAL |
                        LOOM_LOW_REG_CLASS_FLAG_EXPLICIT_PHYSICAL_REGISTERS;
    descriptor_set_.reg_classes = classes_;
    descriptor_set_.reg_class_count = IREE_ARRAYSIZE(classes_);
  }

  void TearDown() override {
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  static loom_liveness_interval_t RegisterInterval(loom_value_id_t value_id,
                                                   uint32_t start_point,
                                                   uint32_t end_point,
                                                   uint32_t unit_count) {
    loom_liveness_interval_t interval = {
        .value_id = value_id,
    };
    interval.value_class.type_kind = LOOM_TYPE_REGISTER;
    interval.start_point = start_point;
    interval.end_point = end_point;
    interval.unit_count = unit_count;
    return interval;
  }

  static loom_liveness_interval_t ScalarInterval(loom_value_id_t value_id) {
    loom_liveness_interval_t interval = {
        .value_id = value_id,
    };
    interval.value_class.type_kind = LOOM_TYPE_SCALAR;
    interval.unit_count = 1;
    return interval;
  }

  static void ExpectOrderedValueIds(
      const loom_low_allocation_interval_order_t& order,
      const loom_value_id_t* expected_value_ids,
      iree_host_size_t expected_count) {
    ASSERT_EQ(order.interval_count, expected_count);
    for (iree_host_size_t i = 0; i < expected_count; ++i) {
      EXPECT_EQ(order.intervals[i].interval->value_id, expected_value_ids[i])
          << i;
    }
  }

  iree_status_t BuildOrder(const loom_liveness_interval_t* intervals,
                           iree_host_size_t count,
                           const uint32_t* acquisition_starts,
                           const loom_low_placement_table_t& placement,
                           loom_low_allocation_interval_order_t* out_order) {
    std::vector<uint32_t> interval_indices(count);
    std::vector<loom_value_id_t> value_ids(count);
    std::vector<loom_low_allocation_unit_liveness_value_t> values(count);
    uint32_t unit_count = 0;
    for (iree_host_size_t i = 0; i < count; ++i) {
      interval_indices[i] = static_cast<uint32_t>(i);
      value_ids[i] = intervals[i].value_id;
      const bool allocatable =
          intervals[i].value_class.type_kind == LOOM_TYPE_REGISTER &&
          intervals[i].unit_count != 0;
      values[i] = {allocatable ? unit_count : UINT32_MAX,
                   acquisition_starts[i]};
      if (allocatable) {
        unit_count += intervals[i].unit_count;
      }
    }
    loom_liveness_analysis_t liveness = {
        .intervals = intervals,
        .interval_count = count,
        .value_ids = value_ids.data(),
        .value_count = count,
        .value_interval_indices = interval_indices.data(),
    };
    loom_low_allocation_unit_liveness_t unit_liveness = {
        .values = values.data(),
    };
    unit_liveness.point_count = unit_count;
    return loom_low_allocation_interval_order_build(&descriptor_set_, &liveness,
                                                    &unit_liveness, &placement,
                                                    &arena_, out_order);
  }

  // Pool shared by the fixture's ordering arena.
  iree_arena_block_pool_t block_pool_;
  // Scratch storage for interval orders.
  iree_arena_allocator_t arena_;
  // Contiguous and explicit physical-register classes.
  loom_low_reg_class_t classes_[2] = {};
  // Target contract used to classify aggregate intervals.
  loom_low_descriptor_set_t descriptor_set_ = {};
};

TEST_F(LowAllocationIntervalOrderTest, FiltersNonAllocatableIntervals) {
  loom_liveness_interval_t intervals[] = {
      ScalarInterval(/*value_id=*/1),
      RegisterInterval(/*value_id=*/2, /*start_point=*/0, /*end_point=*/1,
                       /*unit_count=*/0),
  };
  const uint32_t acquisition_starts[] = {UINT32_MAX, UINT32_MAX};
  const loom_low_placement_table_t placement = {};
  loom_low_allocation_interval_order_t order = {};
  IREE_ASSERT_OK(BuildOrder(intervals, IREE_ARRAYSIZE(intervals),
                            acquisition_starts, placement, &order));
  EXPECT_EQ(order.intervals, nullptr);
  EXPECT_EQ(order.interval_count, 0u);
  EXPECT_FALSE(order.has_packable_aggregates);
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
  const uint32_t acquisition_starts[] = {7, 3, 3, 3, UINT32_MAX};
  const loom_low_placement_table_t placement = {};
  loom_low_allocation_interval_order_t order = {};
  IREE_ASSERT_OK(BuildOrder(intervals, IREE_ARRAYSIZE(intervals),
                            acquisition_starts, placement, &order));
  ASSERT_NE(order.intervals, nullptr);
  ASSERT_EQ(order.interval_count, 4u);
  EXPECT_TRUE(order.has_packable_aggregates);
  const loom_value_id_t expected_value_ids[] = {1, 2, 3, 4};
  ExpectOrderedValueIds(order, expected_value_ids,
                        IREE_ARRAYSIZE(expected_value_ids));
  for (iree_host_size_t i = 0; i < order.interval_count; ++i) {
    EXPECT_EQ(order.intervals[i].topology_rank, 0u);
    EXPECT_EQ(order.intervals[i].acquisition_start_point,
              order.intervals[i].interval->start_point);
  }
}

TEST_F(LowAllocationIntervalOrderTest, SortsLargeReverseStartOrder) {
  enum : iree_host_size_t {
    kIntervalCount = 96,
  };
  loom_liveness_interval_t intervals[kIntervalCount];
  uint32_t acquisition_starts[kIntervalCount];
  for (iree_host_size_t i = 0; i < kIntervalCount; ++i) {
    const uint32_t start_point = (uint32_t)(kIntervalCount - i);
    intervals[i] = RegisterInterval(/*value_id=*/(loom_value_id_t)(i + 1),
                                    start_point, start_point + 1u,
                                    /*unit_count=*/1);
    acquisition_starts[i] = start_point;
  }
  const loom_low_placement_table_t placement = {};
  loom_low_allocation_interval_order_t order = {};
  IREE_ASSERT_OK(BuildOrder(intervals, IREE_ARRAYSIZE(intervals),
                            acquisition_starts, placement, &order));
  ASSERT_NE(order.intervals, nullptr);
  ASSERT_EQ(order.interval_count, kIntervalCount);
  EXPECT_FALSE(order.has_packable_aggregates);
  for (iree_host_size_t i = 0; i < kIntervalCount; ++i) {
    EXPECT_EQ(order.intervals[i].acquisition_start_point,
              static_cast<uint32_t>(i + 1u))
        << i;
  }
}

TEST_F(LowAllocationIntervalOrderTest,
       OptionalStorageTopologyPreservesEndAndValueIdPreference) {
  const loom_liveness_interval_t intervals[] = {
      RegisterInterval(4, 3, 10, 1),
      RegisterInterval(3, 3, 8, 1),
      RegisterInterval(2, 3, 6, 1),
      RegisterInterval(1, 3, 6, 1),
  };
  const uint32_t acquisition_starts[] = {3, 3, 3, 3};
  const loom_value_ordinal_t storage_order[] = {3, 2, 1, 0};
  loom_low_placement_table_t placement = {
      .value_count = IREE_ARRAYSIZE(intervals),
      .storage_value_order = storage_order,
      .storage_value_order_count = IREE_ARRAYSIZE(storage_order),
  };
  // Optional-only topology has no tied-origin array; every value owns itself.
  loom_low_allocation_interval_order_t order = {};
  IREE_ASSERT_OK(BuildOrder(intervals, IREE_ARRAYSIZE(intervals),
                            acquisition_starts, placement, &order));
  const loom_value_id_t expected_value_ids[] = {1, 2, 3, 4};
  ExpectOrderedValueIds(order, expected_value_ids,
                        IREE_ARRAYSIZE(expected_value_ids));
  for (iree_host_size_t i = 0; i < order.interval_count; ++i) {
    EXPECT_EQ(order.intervals[i].topology_rank, 0u);
  }
}

TEST_F(LowAllocationIntervalOrderTest,
       EqualAcquisitionStartsFollowSourcesBeforeResults) {
  const loom_liveness_interval_t intervals[] = {
      RegisterInterval(40, 10, 30, 1),
      RegisterInterval(7, 2, 14, 1),
      RegisterInterval(20, 11, 20, 1),
      RegisterInterval(9, 2, 3, 1),
  };
  const uint32_t acquisition_starts[] = {2, 2, 2, 2};
  const loom_value_ordinal_t storage_order[] = {1, 3, 2, 0};
  const loom_value_ordinal_t tied_origins[] = {0, 0, 0, 3};
  loom_low_placement_table_t placement = {
      .value_count = IREE_ARRAYSIZE(intervals),
      .storage_value_order = storage_order,
      .storage_value_order_count = IREE_ARRAYSIZE(storage_order),
      .tied_storage_origins_by_value_ordinal = tied_origins,
  };
  loom_low_allocation_interval_order_t order = {};
  IREE_ASSERT_OK(BuildOrder(intervals, IREE_ARRAYSIZE(intervals),
                            acquisition_starts, placement, &order));
  // The unrelated owner retains shortest-end priority ahead of the tied root.
  const loom_value_id_t expected_value_ids[] = {9, 40, 20, 7};
  ExpectOrderedValueIds(order, expected_value_ids,
                        IREE_ARRAYSIZE(expected_value_ids));
  EXPECT_EQ(order.intervals[0].topology_rank, 0u);
  EXPECT_EQ(order.intervals[1].topology_rank, 0u);
  for (iree_host_size_t i = 0; i < order.interval_count; ++i) {
    EXPECT_EQ(order.intervals[i].acquisition_start_point, 2u);
    if (i > 1) {
      EXPECT_GT(order.intervals[i].topology_rank,
                order.intervals[i - 1].topology_rank);
    }
  }
  EXPECT_EQ(intervals[0].start_point, 10u);
  EXPECT_EQ(intervals[2].start_point, 11u);
}

TEST_F(LowAllocationIntervalOrderTest,
       PreservesEqualStartTopologyThroughLargeReverseOrder) {
  constexpr uint32_t kPairCount = 48;
  constexpr uint32_t kIntervalCount = 2 * kPairCount;
  loom_liveness_interval_t intervals[kIntervalCount];
  uint32_t acquisition_starts[kIntervalCount];
  loom_value_ordinal_t storage_order[kIntervalCount];
  loom_value_ordinal_t tied_origins[kIntervalCount];
  for (uint32_t i = 0; i < kPairCount; ++i) {
    const uint32_t acquisition_start = 2 * (kPairCount - i);
    intervals[2 * i] = RegisterInterval(2 * i, 200 + i, 400 + i, 1);
    intervals[2 * i + 1] =
        RegisterInterval(2 * i + 1, acquisition_start, 300 + i, 1);
    acquisition_starts[2 * i] = acquisition_start;
    acquisition_starts[2 * i + 1] = acquisition_start;
    tied_origins[2 * i] = 2 * i;
    tied_origins[2 * i + 1] = 2 * i;
  }
  for (uint32_t i = 0; i < kIntervalCount; ++i) {
    storage_order[i] = kIntervalCount - i - 1;
  }
  loom_low_placement_table_t placement = {
      .value_count = kIntervalCount,
      .storage_value_order = storage_order,
      .storage_value_order_count = kIntervalCount,
      .tied_storage_origins_by_value_ordinal = tied_origins,
  };
  loom_low_allocation_interval_order_t order = {};
  IREE_ASSERT_OK(BuildOrder(intervals, kIntervalCount, acquisition_starts,
                            placement, &order));
  ASSERT_EQ(order.interval_count, kIntervalCount);
  for (uint32_t i = 0; i < kPairCount; ++i) {
    const uint32_t source = 2 * (kPairCount - i - 1);
    const auto& source_entry = order.intervals[2 * i];
    const auto& result_entry = order.intervals[2 * i + 1];
    EXPECT_EQ(source_entry.interval->value_id, source);
    EXPECT_EQ(result_entry.interval->value_id, source + 1);
    EXPECT_EQ(source_entry.acquisition_start_point, 2 * (i + 1));
    EXPECT_EQ(result_entry.acquisition_start_point,
              source_entry.acquisition_start_point);
    EXPECT_EQ(source_entry.topology_rank, 0u);
    EXPECT_LT(source_entry.topology_rank, result_entry.topology_rank);
  }
}

TEST_F(LowAllocationIntervalOrderTest, ExcludesExplicitPhysicalAggregates) {
  loom_liveness_interval_t intervals[] = {
      RegisterInterval(/*value_id=*/1, /*start_point=*/0, /*end_point=*/4,
                       /*unit_count=*/1),
      RegisterInterval(/*value_id=*/2, /*start_point=*/0, /*end_point=*/4,
                       /*unit_count=*/8),
      ScalarInterval(/*value_id=*/3),
  };
  intervals[1].value_class.register_class_id = 1;
  // Only register intervals participate in scalar/aggregate packing.
  intervals[2].unit_count = 8;
  const uint32_t acquisition_starts[] = {0, 0, UINT32_MAX};
  const loom_low_placement_table_t placement = {};
  loom_low_allocation_interval_order_t order = {};
  IREE_ASSERT_OK(BuildOrder(intervals, IREE_ARRAYSIZE(intervals),
                            acquisition_starts, placement, &order));
  ASSERT_EQ(order.interval_count, 2u);
  EXPECT_FALSE(order.has_packable_aggregates);
}

}  // namespace
}  // namespace loom
