// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/allocation/active_unit.h"

#include <array>
#include <vector>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace loom {
namespace {

loom_low_allocation_assignment_t Assignment(
    loom_value_id_t value_id, uint16_t descriptor_reg_class_id,
    uint32_t start_point, uint32_t end_point, uint32_t location_base,
    uint32_t location_count, uint32_t unit_point_start) {
  loom_low_allocation_assignment_t assignment = {
      .value_id = value_id,
      .descriptor_reg_class_id = descriptor_reg_class_id,
      .start_point = start_point,
      .end_point = end_point,
      .unit_count = location_count,
      .location_kind = LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER,
      .location_base = location_base,
      .location_count = location_count,
      .unit_point_start = unit_point_start,
  };
  return assignment;
}

loom_low_descriptor_set_t DescriptorSet(const loom_low_reg_class_t* reg_classes,
                                        iree_host_size_t reg_class_count) {
  loom_low_descriptor_set_t descriptor_set = {
      .reg_classes = reg_classes,
  };
  descriptor_set.reg_class_count = reg_class_count;
  return descriptor_set;
}

loom_low_allocation_live_range_sweep_t LiveRangeSweep(
    const loom_low_allocation_assignment_t* assignments,
    iree_host_size_t assignment_count, uint32_t point,
    std::vector<uint32_t>* segment_starts) {
  segment_starts->resize(assignment_count);
  for (iree_host_size_t i = 0; i < assignment_count; ++i) {
    (*segment_starts)[i] = assignments[i].liveness_segments.start;
  }
  return {
      .point = point,
      .segment_starts_by_assignment_index = segment_starts->data(),
  };
}

TEST(LowAllocationActiveUnitTest, FindsAndRemovesIndexedConflicts) {
  iree_arena_block_pool_t block_pool;
  iree_arena_block_pool_initialize(/*block_size=*/4096, iree_allocator_system(),
                                   &block_pool);
  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool, &arena);

  const loom_low_reg_class_t reg_classes[1] = {};
  const loom_low_descriptor_set_t descriptor_set =
      DescriptorSet(reg_classes, IREE_ARRAYSIZE(reg_classes));
  uint32_t unit_end_points[] = {10, 10, 10, 10};
  loom_low_allocation_unit_liveness_t unit_liveness = {
      .end_points = unit_end_points,
      .point_count = IREE_ARRAYSIZE(unit_end_points),
  };
  const loom_low_allocation_assignment_t assignments[] = {
      Assignment(/*value_id=*/1, /*descriptor_reg_class_id=*/0,
                 /*start_point=*/0, /*end_point=*/10, /*location_base=*/4,
                 /*location_count=*/2, /*unit_point_start=*/0),
      Assignment(/*value_id=*/2, /*descriptor_reg_class_id=*/0,
                 /*start_point=*/5, /*end_point=*/10, /*location_base=*/5,
                 /*location_count=*/2, /*unit_point_start=*/2),
  };
  std::vector<uint32_t> segment_starts;
  loom_low_allocation_live_range_sweep_t live_range_sweep = LiveRangeSweep(
      assignments, IREE_ARRAYSIZE(assignments), /*point=*/5, &segment_starts);

  loom_low_allocation_active_unit_index_t index = {};
  IREE_ASSERT_OK(loom_low_allocation_active_unit_index_initialize(
      &descriptor_set, IREE_ARRAYSIZE(assignments), /*unit_capacity=*/32,
      &arena, &index));
  EXPECT_TRUE(loom_low_allocation_active_unit_index_is_enabled(&index));

  loom_low_allocation_active_unit_index_insert_assignment(
      &index, &descriptor_set, assignments, IREE_ARRAYSIZE(assignments),
      /*assignment_index=*/0);
  EXPECT_NE(index.entry_starts_by_assignment_index[0], UINT32_MAX);
  EXPECT_TRUE(loom_low_allocation_active_unit_index_conflicts(
      &index, &live_range_sweep, &descriptor_set, &unit_liveness, assignments,
      IREE_ARRAYSIZE(assignments), &assignments[1],
      /*ignored_value_ids=*/nullptr,
      /*ignored_value_count=*/0));
  uint32_t conflict_indices[2] = {};
  uint16_t conflict_count = 0;
  IREE_ASSERT_OK(loom_low_allocation_active_unit_index_collect_conflicts(
      &index, &live_range_sweep, &descriptor_set, &unit_liveness, assignments,
      IREE_ARRAYSIZE(assignments), &assignments[1],
      /*ignored_value_ids=*/nullptr,
      /*ignored_value_count=*/0, conflict_indices,
      IREE_ARRAYSIZE(conflict_indices), &conflict_count));
  ASSERT_EQ(conflict_count, 1u);
  EXPECT_EQ(conflict_indices[0], 0u);

  const loom_value_id_t ignored_value_ids[] = {1};
  EXPECT_FALSE(loom_low_allocation_active_unit_index_conflicts(
      &index, &live_range_sweep, &descriptor_set, &unit_liveness, assignments,
      IREE_ARRAYSIZE(assignments), &assignments[1], ignored_value_ids,
      IREE_ARRAYSIZE(ignored_value_ids)));
  conflict_count = 0;
  IREE_ASSERT_OK(loom_low_allocation_active_unit_index_collect_conflicts(
      &index, &live_range_sweep, &descriptor_set, &unit_liveness, assignments,
      IREE_ARRAYSIZE(assignments), &assignments[1], ignored_value_ids,
      IREE_ARRAYSIZE(ignored_value_ids), conflict_indices,
      IREE_ARRAYSIZE(conflict_indices), &conflict_count));
  EXPECT_EQ(conflict_count, 0u);

  loom_low_allocation_active_unit_index_remove_assignment(
      &index, assignments, IREE_ARRAYSIZE(assignments), /*assignment_index=*/0);
  EXPECT_EQ(index.entry_starts_by_assignment_index[0], UINT32_MAX);
  EXPECT_FALSE(loom_low_allocation_active_unit_index_conflicts(
      &index, &live_range_sweep, &descriptor_set, &unit_liveness, assignments,
      IREE_ARRAYSIZE(assignments), &assignments[1],
      /*ignored_value_ids=*/nullptr,
      /*ignored_value_count=*/0));

  iree_arena_deinitialize(&arena);
  iree_arena_block_pool_deinitialize(&block_pool);
}

TEST(LowAllocationActiveUnitTest, RecyclesEntriesAcrossAssignmentLifetimes) {
  iree_arena_block_pool_t block_pool;
  iree_arena_block_pool_initialize(/*block_size=*/4096, iree_allocator_system(),
                                   &block_pool);
  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool, &arena);

  constexpr uint32_t kUnitCount = 32;
  constexpr uint32_t kAssignmentCount = 128;
  const loom_low_reg_class_t reg_classes[1] = {};
  const loom_low_descriptor_set_t descriptor_set =
      DescriptorSet(reg_classes, IREE_ARRAYSIZE(reg_classes));
  loom_low_allocation_assignment_t assignments[kAssignmentCount];
  for (uint32_t i = 0; i < kAssignmentCount; ++i) {
    assignments[i] = Assignment(
        /*value_id=*/i + 1, /*descriptor_reg_class_id=*/0,
        /*start_point=*/0, /*end_point=*/10, /*location_base=*/i % kUnitCount,
        /*location_count=*/1, /*unit_point_start=*/0);
  }
  std::vector<uint32_t> segment_starts;
  loom_low_allocation_live_range_sweep_t live_range_sweep = LiveRangeSweep(
      assignments, kAssignmentCount, /*point=*/0, &segment_starts);
  uint32_t unit_end_points[] = {10};
  loom_low_allocation_unit_liveness_t unit_liveness = {
      .end_points = unit_end_points,
      .point_count = IREE_ARRAYSIZE(unit_end_points),
  };
  loom_low_allocation_active_unit_index_t index = {};
  IREE_ASSERT_OK(loom_low_allocation_active_unit_index_initialize(
      &descriptor_set, kAssignmentCount, kUnitCount, &arena, &index));
  const iree_host_size_t initialized_bytes = arena.used_allocation_size;
  for (uint32_t i = 0; i < kAssignmentCount; ++i) {
    if (i >= kUnitCount) {
      loom_low_allocation_active_unit_index_remove_assignment(
          &index, assignments, kAssignmentCount, i - kUnitCount);
      EXPECT_EQ(index.entry_starts_by_assignment_index[i - kUnitCount],
                UINT32_MAX);
    }
    loom_low_allocation_active_unit_index_insert_assignment(
        &index, &descriptor_set, assignments, kAssignmentCount, i);
    EXPECT_NE(index.entry_starts_by_assignment_index[i], UINT32_MAX);
    EXPECT_EQ(index.entry_count, iree_min(i + 1, kUnitCount));
    EXPECT_EQ(index.active_entry_count, index.entry_count);
    uint32_t conflict_indices[kUnitCount];
    uint16_t conflict_count = 0;
    IREE_ASSERT_OK(loom_low_allocation_active_unit_index_collect_conflicts(
        &index, &live_range_sweep, &descriptor_set, &unit_liveness, assignments,
        kAssignmentCount, &assignments[i], /*ignored_value_ids=*/nullptr,
        /*ignored_value_count=*/0, conflict_indices, kUnitCount,
        &conflict_count));
    ASSERT_EQ(conflict_count, 1u);
    EXPECT_EQ(conflict_indices[0], i);
  }
  for (uint32_t i = kAssignmentCount - kUnitCount; i < kAssignmentCount; ++i) {
    loom_low_allocation_active_unit_index_remove_assignment(
        &index, assignments, kAssignmentCount, i);
  }
  EXPECT_EQ(index.active_entry_count, 0u);
  EXPECT_EQ(index.entry_count, kUnitCount);
  EXPECT_EQ(arena.used_allocation_size, initialized_bytes);

  iree_arena_deinitialize(&arena);
  iree_arena_block_pool_deinitialize(&block_pool);
}

TEST(LowAllocationActiveUnitTest, OrdersOnlyDefiniteContinuousScalarConflicts) {
  iree_arena_block_pool_t block_pool;
  iree_arena_block_pool_initialize(/*block_size=*/4096, iree_allocator_system(),
                                   &block_pool);
  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool, &arena);

  loom_low_reg_class_t reg_classes[2] = {};
  reg_classes[0].alias_set_id = 1;
  reg_classes[1].alias_set_id = 1;
  const loom_low_descriptor_set_t descriptor_set =
      DescriptorSet(reg_classes, IREE_ARRAYSIZE(reg_classes));
  loom_low_allocation_assignment_t assignments[] = {
      Assignment(1, 0, 0, 10, 0, 1, 0), Assignment(2, 0, 0, 10, 1, 1, 0),
      Assignment(3, 0, 0, 10, 2, 1, 0), Assignment(4, 0, 0, 10, 4, 1, 0),
      Assignment(5, 0, 0, 10, 6, 1, 0), Assignment(6, 0, 0, 10, 5, 1, 0),
      Assignment(7, 1, 0, 10, 5, 1, 0), Assignment(8, 0, 0, 10, 7, 1, 0),
      Assignment(9, 0, 0, 10, 8, 1, 0),
  };
  assignments[7].liveness_segments = {.start = 0, .count = 1};
  assignments[8].location_kind = LOOM_LOW_ALLOCATION_LOCATION_TARGET_ID;

  loom_low_allocation_active_unit_index_t index = {};
  IREE_ASSERT_OK(loom_low_allocation_active_unit_index_initialize(
      &descriptor_set, IREE_ARRAYSIZE(assignments), /*unit_capacity=*/32,
      &arena, &index));
  for (uint32_t i : {0u, 1u, 2u, 3u, 4u}) {
    loom_low_allocation_active_unit_index_insert_assignment(
        &index, &descriptor_set, assignments, IREE_ARRAYSIZE(assignments), i);
  }

  loom_low_allocation_assignment_t candidate =
      Assignment(100, 0, 5, 12, 0, 1, 0);
  ASSERT_TRUE(loom_low_allocation_active_unit_index_can_order_candidate(
      &index, &candidate));
  uint32_t base = UINT32_MAX;
  EXPECT_TRUE(loom_low_allocation_active_unit_index_find_unoccupied_location(
      &index, &candidate, 0, 10, LOOM_LOW_ALLOCATION_LOCATION_SEARCH_ASCENDING,
      &base));
  EXPECT_EQ(base, 3u);
  EXPECT_TRUE(loom_low_allocation_active_unit_index_find_unoccupied_location(
      &index, &candidate, 0, 6, LOOM_LOW_ALLOCATION_LOCATION_SEARCH_DESCENDING,
      &base));
  EXPECT_EQ(base, 5u);

  // Aliasing classes and duplicate owners share one definite location. It is
  // released only after the last owner leaves the active set.
  loom_low_allocation_active_unit_index_insert_assignment(
      &index, &descriptor_set, assignments, IREE_ARRAYSIZE(assignments), 5);
  loom_low_allocation_active_unit_index_insert_assignment(
      &index, &descriptor_set, assignments, IREE_ARRAYSIZE(assignments), 6);
  candidate.descriptor_reg_class_id = 1;
  const bool found_dense_gap =
      loom_low_allocation_active_unit_index_find_unoccupied_location(
          &index, &candidate, 4, 6,
          LOOM_LOW_ALLOCATION_LOCATION_SEARCH_DESCENDING, &base);
  EXPECT_FALSE(found_dense_gap) << base;
  loom_low_allocation_active_unit_index_remove_assignment(
      &index, assignments, IREE_ARRAYSIZE(assignments), 5);
  EXPECT_FALSE(loom_low_allocation_active_unit_index_find_unoccupied_location(
      &index, &candidate, 5, 5, LOOM_LOW_ALLOCATION_LOCATION_SEARCH_ASCENDING,
      &base));
  loom_low_allocation_active_unit_index_remove_assignment(
      &index, assignments, IREE_ARRAYSIZE(assignments), 6);
  EXPECT_TRUE(loom_low_allocation_active_unit_index_find_unoccupied_location(
      &index, &candidate, 5, 5, LOOM_LOW_ALLOCATION_LOCATION_SEARCH_ASCENDING,
      &base));
  EXPECT_EQ(base, 5u);

  // Sparse reservations are still checked by the full conflict predicate;
  // the ordered index never treats them as a definite active conflict.
  loom_low_allocation_active_unit_index_insert_assignment(
      &index, &descriptor_set, assignments, IREE_ARRAYSIZE(assignments), 7);
  candidate.descriptor_reg_class_id = 0;
  EXPECT_TRUE(loom_low_allocation_active_unit_index_find_unoccupied_location(
      &index, &candidate, 7, 7, LOOM_LOW_ALLOCATION_LOCATION_SEARCH_ASCENDING,
      &base));
  EXPECT_EQ(base, 7u);

  // Location kinds have independent ordered domains.
  loom_low_allocation_active_unit_index_insert_assignment(
      &index, &descriptor_set, assignments, IREE_ARRAYSIZE(assignments), 8);
  candidate.location_kind = LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER;
  EXPECT_TRUE(loom_low_allocation_active_unit_index_find_unoccupied_location(
      &index, &candidate, 8, 8, LOOM_LOW_ALLOCATION_LOCATION_SEARCH_ASCENDING,
      &base));
  candidate.location_kind = LOOM_LOW_ALLOCATION_LOCATION_TARGET_ID;
  EXPECT_FALSE(loom_low_allocation_active_unit_index_find_unoccupied_location(
      &index, &candidate, 8, 8, LOOM_LOW_ALLOCATION_LOCATION_SEARCH_ASCENDING,
      &base));

  candidate.location_count = candidate.unit_count = 2;
  EXPECT_FALSE(loom_low_allocation_active_unit_index_can_order_candidate(
      &index, &candidate));

  iree_arena_deinitialize(&arena);
  iree_arena_block_pool_deinitialize(&block_pool);
}

TEST(LowAllocationActiveUnitTest,
     OrderedLocationsMatchExhaustiveGapQueriesAcrossUpdates) {
  iree_arena_block_pool_t block_pool;
  iree_arena_block_pool_initialize(/*block_size=*/4096, iree_allocator_system(),
                                   &block_pool);
  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool, &arena);

  constexpr uint32_t kLocationCount = 64;
  const loom_low_reg_class_t reg_classes[1] = {};
  const loom_low_descriptor_set_t descriptor_set =
      DescriptorSet(reg_classes, IREE_ARRAYSIZE(reg_classes));
  loom_low_allocation_assignment_t assignments[kLocationCount];
  for (uint32_t i = 0; i < kLocationCount; ++i) {
    assignments[i] = Assignment(
        /*value_id=*/i + 1, /*descriptor_reg_class_id=*/0,
        /*start_point=*/0, /*end_point=*/10,
        /*location_base=*/(i * 37u) % kLocationCount,
        /*location_count=*/1, /*unit_point_start=*/0);
  }

  loom_low_allocation_active_unit_index_t index = {};
  IREE_ASSERT_OK(loom_low_allocation_active_unit_index_initialize(
      &descriptor_set, IREE_ARRAYSIZE(assignments),
      /*unit_capacity=*/kLocationCount, &arena, &index));
  loom_low_allocation_assignment_t candidate =
      Assignment(100, 0, 5, 12, 0, 1, 0);
  bool occupied[kLocationCount] = {};

  auto verify_all_ranges = [&]() {
    for (uint32_t minimum = 0; minimum < kLocationCount; ++minimum) {
      for (uint32_t maximum = minimum; maximum < kLocationCount; ++maximum) {
        uint32_t expected_ascending = minimum;
        while (expected_ascending <= maximum && occupied[expected_ascending]) {
          ++expected_ascending;
        }
        uint32_t actual = UINT32_MAX;
        const bool found_ascending =
            loom_low_allocation_active_unit_index_find_unoccupied_location(
                &index, &candidate, minimum, maximum,
                LOOM_LOW_ALLOCATION_LOCATION_SEARCH_ASCENDING, &actual);
        EXPECT_EQ(found_ascending, expected_ascending <= maximum)
            << "range [" << minimum << ", " << maximum << "]";
        if (found_ascending) {
          EXPECT_EQ(actual, expected_ascending)
              << "range [" << minimum << ", " << maximum << "]";
        }

        uint32_t expected_descending = maximum;
        while (expected_descending >= minimum &&
               occupied[expected_descending]) {
          if (expected_descending == 0) {
            break;
          }
          --expected_descending;
        }
        const bool expected_descending_found =
            !occupied[expected_descending] && expected_descending >= minimum;
        actual = UINT32_MAX;
        const bool found_descending =
            loom_low_allocation_active_unit_index_find_unoccupied_location(
                &index, &candidate, minimum, maximum,
                LOOM_LOW_ALLOCATION_LOCATION_SEARCH_DESCENDING, &actual);
        EXPECT_EQ(found_descending, expected_descending_found)
            << "range [" << minimum << ", " << maximum << "]";
        if (found_descending) {
          EXPECT_EQ(actual, expected_descending)
              << "range [" << minimum << ", " << maximum << "]";
        }
      }
    }
  };

  for (uint32_t i = 0; i < kLocationCount; ++i) {
    loom_low_allocation_active_unit_index_insert_assignment(
        &index, &descriptor_set, assignments, IREE_ARRAYSIZE(assignments), i);
    occupied[assignments[i].location_base] = true;
    if (i == 0 || i == 15 || i == 31 || i == kLocationCount - 1) {
      verify_all_ranges();
    }
  }
  for (uint32_t i = 0; i < kLocationCount; ++i) {
    const uint32_t assignment_index = (i * 29u) % kLocationCount;
    loom_low_allocation_active_unit_index_remove_assignment(
        &index, assignments, IREE_ARRAYSIZE(assignments), assignment_index);
    occupied[assignments[assignment_index].location_base] = false;
    if (i == 15 || i == 31 || i == kLocationCount - 1) {
      verify_all_ranges();
    }
  }

  iree_arena_deinitialize(&arena);
  iree_arena_block_pool_deinitialize(&block_pool);
}

TEST(LowAllocationActiveUnitTest, OrderedLocationsHandleUint32Boundaries) {
  iree_arena_block_pool_t block_pool;
  iree_arena_block_pool_initialize(/*block_size=*/4096, iree_allocator_system(),
                                   &block_pool);
  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool, &arena);

  const loom_low_reg_class_t reg_classes[1] = {};
  const loom_low_descriptor_set_t descriptor_set =
      DescriptorSet(reg_classes, IREE_ARRAYSIZE(reg_classes));
  const loom_low_allocation_assignment_t assignments[] = {
      Assignment(1, 0, 0, 10, 0, 1, 0),
      Assignment(2, 0, 0, 10, 1, 1, 0),
      Assignment(3, 0, 0, 10, UINT32_MAX - 1u, 1, 0),
      Assignment(4, 0, 0, 10, UINT32_MAX, 1, 0),
  };
  loom_low_allocation_active_unit_index_t index = {};
  IREE_ASSERT_OK(loom_low_allocation_active_unit_index_initialize(
      &descriptor_set, IREE_ARRAYSIZE(assignments), /*unit_capacity=*/32,
      &arena, &index));
  for (uint32_t i = 0; i < IREE_ARRAYSIZE(assignments); ++i) {
    loom_low_allocation_active_unit_index_insert_assignment(
        &index, &descriptor_set, assignments, IREE_ARRAYSIZE(assignments), i);
  }
  loom_low_allocation_assignment_t candidate =
      Assignment(100, 0, 5, 12, 0, 1, 0);
  uint32_t base = UINT32_MAX;
  EXPECT_TRUE(loom_low_allocation_active_unit_index_find_unoccupied_location(
      &index, &candidate, 0, UINT32_MAX,
      LOOM_LOW_ALLOCATION_LOCATION_SEARCH_ASCENDING, &base));
  EXPECT_EQ(base, 2u);
  EXPECT_TRUE(loom_low_allocation_active_unit_index_find_unoccupied_location(
      &index, &candidate, 0, UINT32_MAX,
      LOOM_LOW_ALLOCATION_LOCATION_SEARCH_DESCENDING, &base));
  EXPECT_EQ(base, UINT32_MAX - 2u);
  EXPECT_FALSE(loom_low_allocation_active_unit_index_find_unoccupied_location(
      &index, &candidate, UINT32_MAX - 1u, UINT32_MAX,
      LOOM_LOW_ALLOCATION_LOCATION_SEARCH_ASCENDING, &base));
  EXPECT_FALSE(loom_low_allocation_active_unit_index_find_unoccupied_location(
      &index, &candidate, UINT32_MAX - 1u, UINT32_MAX,
      LOOM_LOW_ALLOCATION_LOCATION_SEARCH_DESCENDING, &base));

  loom_low_allocation_active_unit_index_remove_assignment(
      &index, assignments, IREE_ARRAYSIZE(assignments),
      /*assignment_index=*/2);
  EXPECT_TRUE(loom_low_allocation_active_unit_index_find_unoccupied_location(
      &index, &candidate, UINT32_MAX - 1u, UINT32_MAX,
      LOOM_LOW_ALLOCATION_LOCATION_SEARCH_ASCENDING, &base));
  EXPECT_EQ(base, UINT32_MAX - 1u);

  iree_arena_deinitialize(&arena);
  iree_arena_block_pool_deinitialize(&block_pool);
}

TEST(LowAllocationActiveUnitTest, RefinesIndexedConflictByUnitStart) {
  iree_arena_block_pool_t block_pool;
  iree_arena_block_pool_initialize(/*block_size=*/4096, iree_allocator_system(),
                                   &block_pool);
  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool, &arena);

  const loom_low_reg_class_t reg_classes[1] = {};
  const loom_low_descriptor_set_t descriptor_set =
      DescriptorSet(reg_classes, IREE_ARRAYSIZE(reg_classes));
  uint32_t unit_start_points[] = {0, 5, 2, 2};
  uint32_t unit_end_points[] = {10, 10, 5, 6};
  loom_low_allocation_unit_liveness_t unit_liveness = {
      .start_points = unit_start_points,
      .end_points = unit_end_points,
      .point_count = IREE_ARRAYSIZE(unit_end_points),
  };
  loom_low_allocation_assignment_t assignments[] = {
      Assignment(/*value_id=*/1, /*descriptor_reg_class_id=*/0,
                 /*start_point=*/0, /*end_point=*/10, /*location_base=*/4,
                 /*location_count=*/2, /*unit_point_start=*/0),
      Assignment(/*value_id=*/2, /*descriptor_reg_class_id=*/0,
                 /*start_point=*/2, /*end_point=*/5, /*location_base=*/5,
                 /*location_count=*/1, /*unit_point_start=*/2),
      Assignment(/*value_id=*/3, /*descriptor_reg_class_id=*/0,
                 /*start_point=*/2, /*end_point=*/6, /*location_base=*/5,
                 /*location_count=*/1, /*unit_point_start=*/3),
  };
  assignments[0].flags =
      LOOM_LOW_ALLOCATION_ASSIGNMENT_FLAG_REFINED_UNIT_STARTS;
  std::vector<uint32_t> segment_starts;
  loom_low_allocation_live_range_sweep_t live_range_sweep = LiveRangeSweep(
      assignments, IREE_ARRAYSIZE(assignments), /*point=*/2, &segment_starts);

  loom_low_allocation_active_unit_index_t index = {};
  IREE_ASSERT_OK(loom_low_allocation_active_unit_index_initialize(
      &descriptor_set, IREE_ARRAYSIZE(assignments), /*unit_capacity=*/32,
      &arena, &index));
  loom_low_allocation_active_unit_index_insert_assignment(
      &index, &descriptor_set, assignments, IREE_ARRAYSIZE(assignments),
      /*assignment_index=*/0);

  EXPECT_FALSE(loom_low_allocation_active_unit_index_conflicts(
      &index, &live_range_sweep, &descriptor_set, &unit_liveness, assignments,
      IREE_ARRAYSIZE(assignments), &assignments[1],
      /*ignored_value_ids=*/nullptr,
      /*ignored_value_count=*/0));
  EXPECT_TRUE(loom_low_allocation_active_unit_index_conflicts(
      &index, &live_range_sweep, &descriptor_set, &unit_liveness, assignments,
      IREE_ARRAYSIZE(assignments), &assignments[2],
      /*ignored_value_ids=*/nullptr,
      /*ignored_value_count=*/0));

  iree_arena_deinitialize(&arena);
  iree_arena_block_pool_deinitialize(&block_pool);
}

TEST(LowAllocationActiveUnitTest, StoresEntirePlannedSpan) {
  iree_arena_block_pool_t block_pool;
  iree_arena_block_pool_initialize(/*block_size=*/4096, iree_allocator_system(),
                                   &block_pool);
  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool, &arena);

  const loom_low_reg_class_t reg_classes[1] = {};
  const loom_low_descriptor_set_t descriptor_set =
      DescriptorSet(reg_classes, IREE_ARRAYSIZE(reg_classes));
  const loom_low_allocation_assignment_t assignment =
      Assignment(/*value_id=*/1, /*descriptor_reg_class_id=*/0,
                 /*start_point=*/0, /*end_point=*/10, /*location_base=*/4,
                 /*location_count=*/33, /*unit_point_start=*/0);

  loom_low_allocation_active_unit_index_t index = {};
  IREE_ASSERT_OK(loom_low_allocation_active_unit_index_initialize(
      &descriptor_set, /*assignment_capacity=*/1, /*unit_capacity=*/33, &arena,
      &index));
  loom_low_allocation_active_unit_index_insert_assignment(
      &index, &descriptor_set, &assignment, /*assignment_count=*/1,
      /*assignment_index=*/0);
  EXPECT_EQ(index.active_entry_count, 33u);
  loom_low_allocation_active_unit_index_remove_assignment(
      &index, &assignment, /*assignment_count=*/1, /*assignment_index=*/0);
  EXPECT_EQ(index.active_entry_count, 0u);

  iree_arena_deinitialize(&arena);
  iree_arena_block_pool_deinitialize(&block_pool);
}

TEST(LowAllocationActiveUnitTest, RejectsUnrepresentableCapacity) {
  iree_arena_block_pool_t block_pool;
  iree_arena_block_pool_initialize(4096, iree_allocator_system(), &block_pool);
  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool, &arena);
  const loom_low_descriptor_set_t descriptor_set = {};
  loom_low_allocation_active_unit_index_t index = {};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        loom_low_allocation_active_unit_index_initialize(
                            &descriptor_set, /*assignment_capacity=*/1,
                            /*unit_capacity=*/UINT32_MAX, &arena, &index));
  EXPECT_EQ(arena.used_allocation_size, 0u);
  EXPECT_FALSE(loom_low_allocation_active_unit_index_is_enabled(&index));
  IREE_ASSERT_OK(loom_low_allocation_active_unit_index_initialize(
      &descriptor_set, /*assignment_capacity=*/1, /*unit_capacity=*/1, &arena,
      &index));
  EXPECT_EQ(arena.used_allocation_size, 0u);
  EXPECT_FALSE(loom_low_allocation_active_unit_index_is_enabled(&index));
  iree_arena_deinitialize(&arena);
  iree_arena_block_pool_deinitialize(&block_pool);
}

TEST(LowAllocationActiveUnitTest, IndexesExplicitRegisterAtomicUnits) {
  iree_arena_block_pool_t block_pool;
  iree_arena_block_pool_initialize(/*block_size=*/4096, iree_allocator_system(),
                                   &block_pool);
  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool, &arena);

  loom_low_reg_class_t reg_classes[2] = {};
  for (loom_low_reg_class_t& reg_class : reg_classes) {
    reg_class.flags = LOOM_LOW_REG_CLASS_FLAG_PHYSICAL |
                      LOOM_LOW_REG_CLASS_FLAG_EXPLICIT_PHYSICAL_REGISTERS;
  }
  reg_classes[0].allocatable_count = 2;
  reg_classes[1].allocatable_count = 1;
  reg_classes[1].physical_register_candidate_start = 2;
  const loom_low_physical_register_t physical_registers[] = {
      {
          .name_string_ref = 0,
          .atomic_unit_start = 0,
          .atomic_unit_count = 2,
          .reserved = 0,
      },
      {
          .name_string_ref = 0,
          .atomic_unit_start = 2,
          .atomic_unit_count = 1,
          .reserved = 0,
      },
      {
          .name_string_ref = 0,
          .atomic_unit_start = 3,
          .atomic_unit_count = 2,
          .reserved = 0,
      },
  };
  reg_classes[0].candidate_lookup.register_count = 3;
  reg_classes[1].candidate_lookup.ordinal_start = 3;
  reg_classes[1].candidate_lookup.register_base = 1;
  reg_classes[1].candidate_lookup.register_count = 1;
  const uint16_t candidate_ordinals[] = {0, UINT16_MAX, 1, 0};
  const uint16_t candidates[] = {0, 2, 1};
  const uint16_t allocation_ordinals[] = {0, 1, 0};
  const uint16_t atomic_units[] = {0, 1, 0, 2, 3};
  loom_low_descriptor_set_t descriptor_set =
      DescriptorSet(reg_classes, IREE_ARRAYSIZE(reg_classes));
  descriptor_set.physical_registers = physical_registers;
  descriptor_set.physical_register_count = IREE_ARRAYSIZE(physical_registers);
  descriptor_set.physical_register_candidate_ordinals = candidate_ordinals;
  descriptor_set.physical_register_candidate_ordinal_count =
      IREE_ARRAYSIZE(candidate_ordinals);
  descriptor_set.physical_register_candidate_ids = candidates;
  descriptor_set.physical_register_allocation_ordinals = allocation_ordinals;
  descriptor_set.physical_register_candidate_count = IREE_ARRAYSIZE(candidates);
  descriptor_set.physical_register_atomic_units = atomic_units;
  descriptor_set.physical_register_atomic_unit_count =
      IREE_ARRAYSIZE(atomic_units);

  uint32_t unit_end_points[] = {10, 10};
  loom_low_allocation_unit_liveness_t unit_liveness = {
      .end_points = unit_end_points,
      .point_count = IREE_ARRAYSIZE(unit_end_points),
  };
  loom_low_allocation_assignment_t assignments[] = {
      Assignment(/*value_id=*/1, /*descriptor_reg_class_id=*/0,
                 /*start_point=*/0, /*end_point=*/10, /*location_base=*/0,
                 /*location_count=*/1, /*unit_point_start=*/0),
      Assignment(/*value_id=*/2, /*descriptor_reg_class_id=*/1,
                 /*start_point=*/0, /*end_point=*/10, /*location_base=*/1,
                 /*location_count=*/1, /*unit_point_start=*/1),
      Assignment(/*value_id=*/3, /*descriptor_reg_class_id=*/0,
                 /*start_point=*/0, /*end_point=*/10, /*location_base=*/2,
                 /*location_count=*/1, /*unit_point_start=*/1),
  };
  std::vector<uint32_t> segment_starts;
  loom_low_allocation_live_range_sweep_t live_range_sweep = LiveRangeSweep(
      assignments, IREE_ARRAYSIZE(assignments), /*point=*/0, &segment_starts);

  loom_low_allocation_active_unit_index_t index = {};
  IREE_ASSERT_OK(loom_low_allocation_active_unit_index_initialize(
      &descriptor_set, IREE_ARRAYSIZE(assignments), /*unit_capacity=*/32,
      &arena, &index));
  loom_low_allocation_active_unit_index_insert_assignment(
      &index, &descriptor_set, assignments, IREE_ARRAYSIZE(assignments),
      /*assignment_index=*/0);

  EXPECT_TRUE(loom_low_allocation_active_unit_index_conflicts(
      &index, &live_range_sweep, &descriptor_set, &unit_liveness, assignments,
      IREE_ARRAYSIZE(assignments), &assignments[1],
      /*ignored_value_ids=*/nullptr,
      /*ignored_value_count=*/0));
  EXPECT_FALSE(loom_low_allocation_active_unit_index_conflicts(
      &index, &live_range_sweep, &descriptor_set, &unit_liveness, assignments,
      IREE_ARRAYSIZE(assignments), &assignments[2],
      /*ignored_value_ids=*/nullptr,
      /*ignored_value_count=*/0));
  loom_low_allocation_active_unit_index_remove_assignment(
      &index, assignments, IREE_ARRAYSIZE(assignments),
      /*assignment_index=*/0);
  EXPECT_FALSE(loom_low_allocation_active_unit_index_conflicts(
      &index, &live_range_sweep, &descriptor_set, &unit_liveness, assignments,
      IREE_ARRAYSIZE(assignments), &assignments[1],
      /*ignored_value_ids=*/nullptr,
      /*ignored_value_count=*/0));

  iree_arena_deinitialize(&arena);
  iree_arena_block_pool_deinitialize(&block_pool);
}

class LowAllocationActiveOccupancyTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(128 * 1024, iree_allocator_system(),
                                     &block_pool_);
    iree_arena_initialize(&block_pool_, &arena_);
  }
  void TearDown() override {
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }
  iree_status_t Initialize(const loom_low_descriptor_set_t* descriptors,
                           const loom_low_allocation_unit_liveness_t* liveness,
                           const loom_low_allocation_assignment_t* assignments,
                           iree_host_size_t assignment_count,
                           uint32_t unit_capacity,
                           uint32_t program_point_count) {
    descriptors_ = descriptors;
    liveness_ = liveness;
    assignments_ = assignments;
    assignment_count_ = assignment_count;
    IREE_RETURN_IF_ERROR(loom_low_allocation_active_unit_index_initialize(
        descriptors, assignment_count, unit_capacity, &arena_, &index_));
    return loom_low_allocation_active_unit_index_initialize_occupancy(
        &index_, liveness, program_point_count, &arena_);
  }
  void Insert(uint32_t ordinal) {
    loom_low_allocation_active_unit_index_insert_assignment(
        &index_, descriptors_, assignments_, assignment_count_, ordinal);
  }
  void Remove(uint32_t ordinal) {
    loom_low_allocation_active_unit_index_remove_assignment(
        &index_, assignments_, assignment_count_, ordinal);
  }
  void Advance(uint32_t point) {
    loom_low_allocation_active_unit_index_advance(&index_, assignments_, point);
  }
  void ExpectProjection(const loom_low_allocation_assignment_t& candidate,
                        const std::vector<uint32_t>& locations) {
    std::array<uint64_t, LOOM_LOW_ALLOCATION_ACTIVE_CONFLICT_WORD_COUNT>
        actual{};
    auto expected = actual;
    for (uint32_t location : locations) {
      expected[location / 64] |= UINT64_C(1) << (location % 64);
    }
    loom_low_allocation_active_unit_index_project_conflicts(&index_, &candidate,
                                                            actual.data());
    EXPECT_EQ(actual, expected);
    std::vector<uint32_t> segment_starts;
    loom_low_allocation_live_range_sweep_t sweep =
        LiveRangeSweep(assignments_, assignment_count_, candidate.start_point,
                       &segment_starts);
    for (uint32_t base = 0;
         base < LOOM_LOW_ALLOCATION_ACTIVE_CONFLICT_LOCATION_COUNT; ++base) {
      if ((actual[base / 64] & (UINT64_C(1) << (base % 64))) == 0) {
        continue;
      }
      auto placed = candidate;
      placed.location_base = base;
      EXPECT_TRUE(loom_low_allocation_active_unit_index_conflicts(
          &index_, &sweep, descriptors_, liveness_, assignments_,
          assignment_count_, &placed, nullptr, 0))
          << "projected base=" << base;
    }
  }

  // Pooled invocation storage, including the index's event calendar.
  iree_arena_block_pool_t block_pool_;
  // Arena released after each API fixture.
  iree_arena_allocator_t arena_;
  // Active-unit owner under test.
  loom_low_allocation_active_unit_index_t index_ = {};
  // Borrowed target rows from the current test.
  const loom_low_descriptor_set_t* descriptors_ = nullptr;
  // Borrowed producer-owned unit lifetime facts.
  const loom_low_allocation_unit_liveness_t* liveness_ = nullptr;
  // Borrowed assignment records, stable through each index operation.
  const loom_low_allocation_assignment_t* assignments_ = nullptr;
  // Number of addressable assignment records.
  iree_host_size_t assignment_count_ = 0;
};

TEST_F(LowAllocationActiveOccupancyTest, ActivatesExpiresAndReusesUnitEntries) {
  const loom_low_reg_class_t classes[] = {{.allocatable_count = 256}};
  const auto descriptors = DescriptorSet(classes, IREE_ARRAYSIZE(classes));
  uint32_t starts[] = {2, 5, 2, 2, 2, 10};
  uint32_t ends[] = {7, 12, 9, 10, 20, 15};
  const loom_low_allocation_unit_liveness_t liveness = {
      .start_points = starts,
      .end_points = ends,
      .point_count = IREE_ARRAYSIZE(ends),
  };
  loom_low_allocation_assignment_t assignments[] = {
      Assignment(1, 0, 2, 12, 63, 3, 0),
      Assignment(2, 0, 2, 10, 64, 1, 3),
      Assignment(3, 0, 10, 15, 128, 1, 5),
  };
  assignments[0].flags =
      LOOM_LOW_ALLOCATION_ASSIGNMENT_FLAG_REFINED_UNIT_STARTS;
  IREE_ASSERT_OK(Initialize(&descriptors, &liveness, assignments,
                            IREE_ARRAYSIZE(assignments), 32, 21));
  const auto initialized_bytes = arena_.used_allocation_size;
  auto candidate = Assignment(4, 0, 2, 20, 0, 1, 4);
  Advance(2);
  Insert(0);
  Insert(1);
  ExpectProjection(candidate, {63, 64, 65});
  candidate.start_point = 5;
  Advance(candidate.start_point);
  Remove(1);  // The aggregate's newly activated unit still owns register 64.
  ExpectProjection(candidate, {63, 64, 65});
  candidate.start_point = 7;
  Advance(candidate.start_point);
  ExpectProjection(candidate, {64, 65});
  Remove(0);
  Insert(2);  // Reuses a hash entry with two old retirement events still ahead.
  candidate.start_point = 9;
  Advance(candidate.start_point);
  ExpectProjection(candidate, {});
  candidate.start_point = 10;
  Advance(candidate.start_point);
  ExpectProjection(candidate, {128});
  candidate.start_point = 12;
  Advance(candidate.start_point);
  ExpectProjection(candidate, {128});
  candidate.start_point = 15;
  Advance(candidate.start_point);
  ExpectProjection(candidate, {});
  EXPECT_EQ(arena_.used_allocation_size, initialized_bytes);
}

TEST_F(LowAllocationActiveOccupancyTest,
       PreservesSparseGapsAndExpandedDefinitions) {
  const loom_low_reg_class_t classes[] = {{.allocatable_count = 256}};
  const auto descriptors = DescriptorSet(classes, IREE_ARRAYSIZE(classes));
  uint32_t ends[] = {12, 12, 12, 14, 20};
  const loom_liveness_segment_t segments[] = {{1, 3}, {5, 7}, {9, 12}};
  loom_low_allocation_unit_liveness_t liveness = {
      .end_points = ends,
      .point_count = IREE_ARRAYSIZE(ends),
  };
  liveness.storage_segments.entries = segments;
  loom_low_allocation_assignment_t assignments[] = {
      Assignment(1, 0, 1, 12, 64, 1, 0),
      Assignment(2, 0, 1, 12, 65, 2, 1),
      Assignment(3, 0, 1, 14, 70, 1, 3),
  };
  assignments[0].liveness_segments = {0, 3};
  assignments[1].liveness_segments = {0, 3};
  IREE_ASSERT_OK(Initialize(&descriptors, &liveness, assignments,
                            IREE_ARRAYSIZE(assignments), 32, 21));
  Advance(1);
  Insert(0);
  Insert(1);
  Insert(2);
  auto candidate = Assignment(4, 0, 1, 20, 0, 1, 4);
  for (uint32_t point = 1; point <= 12; ++point) {
    SCOPED_TRACE(point);
    candidate.start_point = point;
    Advance(point);
    if (point == 6 || point == 8) {
      // Reinsertion seeks past retired spans, both within a live span and in a
      // gap preceding the next activation.
      Remove(0);
      Insert(0);
    }
    if (point < 3 || (point >= 5 && point < 7) || (point >= 9 && point < 12)) {
      ExpectProjection(candidate, {64, 65, 66, 70});
    } else {
      ExpectProjection(candidate, {70});
    }
    candidate.liveness_segments = {0, 3};
    if (point == 3 || point == 4 || point == 7 || point == 8 || point == 12) {
      ExpectProjection(candidate, {});
      candidate.flags =
          LOOM_LOW_ALLOCATION_ASSIGNMENT_FLAG_EXPANDED_EXECUTION_DEFINITION;
      ExpectProjection(candidate, {70});
      candidate.flags = 0;
    }
    candidate.liveness_segments = {};
  }
  Remove(0);
  // A late reinsertion seeks past historical sparse spans without replaying
  // them; the lifetime is already exhausted and cannot publish a stale bit.
  Insert(0);
  ExpectProjection(candidate, {70});
}

TEST_F(LowAllocationActiveOccupancyTest,
       RetainsSparseAggregateUnitLifetimesWithinTheWindow) {
  const loom_low_reg_class_t classes[] = {{.allocatable_count = 512}};
  const auto descriptors = DescriptorSet(classes, IREE_ARRAYSIZE(classes));
  const loom_liveness_segment_t segments[] = {{1, 3}, {5, 7}, {9, 12}};
  // Distinct starts, ends, and an empty unit span must survive every segment
  // transition. Widths cross bitmap words and the bounded occupancy window;
  // advancement skips unobserved spans and lands inside later spans.
  constexpr uint32_t kStarts[] = {1, 6, 10, 6};
  constexpr uint32_t kEnds[] = {6, 12, 11, 6};
  for (uint32_t width : {1u, 2u, 4u, 63u, 64u, 65u, 193u, 257u}) {
    SCOPED_TRACE(width);
    std::vector<uint32_t> starts(width + 1);
    std::vector<uint32_t> ends(width + 1, 20);
    for (uint32_t unit = 0; unit < width; ++unit) {
      starts[unit] = kStarts[unit % 4];
      ends[unit] = kEnds[unit % 4];
    }
    loom_low_allocation_unit_liveness_t liveness = {
        .start_points = starts.data(),
        .end_points = ends.data(),
        .point_count = ends.size(),
    };
    liveness.storage_segments.entries = segments;
    auto assignment = Assignment(1, 0, 1, 12, 63, width, 0);
    assignment.flags = LOOM_LOW_ALLOCATION_ASSIGNMENT_FLAG_REFINED_UNIT_STARTS;
    assignment.liveness_segments = {0, IREE_ARRAYSIZE(segments)};
    IREE_ASSERT_OK(Initialize(&descriptors, &liveness, &assignment, 1,
                              iree_max(32u, width), 21));
    const auto initialized_bytes = arena_.used_allocation_size;
    Insert(0);
    auto candidate = Assignment(2, 0, 1, 20, 0, 1, width);
    for (uint32_t point : {1u, 2u, 6u, 8u, 10u, 12u}) {
      SCOPED_TRACE(point);
      candidate.start_point = point;
      Advance(point);
      if (point == 6 || point == 8) {
        Remove(0);
        Insert(0);
      }
      std::vector<uint32_t> expected;
      const bool in_segment =
          point < 3 || (point >= 5 && point < 7) || (point >= 9 && point < 12);
      for (uint32_t unit = 0; unit < width && 63 + unit < 256; ++unit) {
        if (in_segment && starts[unit] <= point && point < ends[unit]) {
          expected.push_back(63 + unit);
        }
      }
      ExpectProjection(candidate, expected);
    }
    Remove(0);
    Insert(0);
    ExpectProjection(candidate, {});
    EXPECT_EQ(arena_.used_allocation_size, initialized_bytes);
  }
}

TEST_F(LowAllocationActiveOccupancyTest,
       ProjectsLiveUnitsAcrossWordBoundaries) {
  const loom_low_reg_class_t classes[] = {
      {.allocatable_count = 512, .alias_set_id = 1},
      {.allocatable_count = 512, .alias_set_id = 1},
  };
  const auto descriptors = DescriptorSet(classes, IREE_ARRAYSIZE(classes));
  uint32_t starts[66] = {};
  uint32_t ends[66] = {};
  ends[0] = 10;
  for (uint32_t i = 1; i < IREE_ARRAYSIZE(ends); ++i) {
    starts[i] = 2;
    ends[i] = 2;
  }
  starts[4] = 4;
  ends[1] = ends[3] = ends[4] = 8;
  const loom_low_allocation_unit_liveness_t liveness = {
      .start_points = starts,
      .end_points = ends,
      .point_count = IREE_ARRAYSIZE(ends),
  };
  const loom_low_allocation_assignment_t assignments[] = {
      Assignment(1, 0, 0, 10, 63, 1, 0),  Assignment(2, 0, 0, 10, 64, 1, 0),
      Assignment(3, 0, 0, 10, 127, 1, 0), Assignment(4, 0, 0, 10, 255, 1, 0),
      Assignment(5, 0, 0, 10, 256, 1, 0), Assignment(6, 1, 0, 10, 17, 1, 0),
  };
  IREE_ASSERT_OK(Initialize(&descriptors, &liveness, assignments,
                            IREE_ARRAYSIZE(assignments), 32, 11));
  for (uint32_t i = 0; i < IREE_ARRAYSIZE(assignments); ++i) {
    Insert(i);
  }
  Advance(2);
  auto candidate = Assignment(7, 0, 2, 8, 0, 4, 1);
  candidate.flags = LOOM_LOW_ALLOCATION_ASSIGNMENT_FLAG_REFINED_UNIT_STARTS;
  ExpectProjection(candidate, {61, 62, 63, 64, 125, 127, 253, 255});
  candidate.descriptor_reg_class_id = 1;
  ExpectProjection(candidate, {15, 17});
  candidate.location_kind = LOOM_LOW_ALLOCATION_LOCATION_TARGET_ID;
  ExpectProjection(candidate, {});
  // Only unit 64 is live, exercising a whole-word shift without shifting by 64.
  ends[1] = ends[3] = ends[4] = 2;
  ends[65] = 8;
  candidate = Assignment(7, 0, 2, 8, 0, 65, 1);
  ExpectProjection(candidate, {0, 63, 191});
}

TEST_F(LowAllocationActiveOccupancyTest,
       SegmentsCalendarAndReusesPooledStorage) {
  constexpr uint32_t kCount = 1025;
  const loom_low_reg_class_t classes[] = {{.allocatable_count = 256}};
  const auto descriptors = DescriptorSet(classes, IREE_ARRAYSIZE(classes));
  std::vector<uint32_t> ends(kCount + 1, 4098);
  std::vector<loom_low_allocation_assignment_t> assignments;
  for (uint32_t i = 0; i < kCount; ++i) {
    ends[i] = 4096 + i % 2;
    assignments.push_back(Assignment(i + 1, 0, 0, ends[i], i % 256, 1, i));
  }
  const loom_low_allocation_unit_liveness_t liveness = {
      .end_points = ends.data(),
      .point_count = ends.size(),
  };
  IREE_ASSERT_OK(Initialize(&descriptors, &liveness, assignments.data(),
                            assignments.size(), kCount, 4099));
  const auto initialized_bytes = arena_.used_allocation_size;
  for (uint32_t i = 0; i < kCount; ++i) {
    Insert(i);
  }
  auto candidate = Assignment(kCount + 1, 0, 4096, 4098, 0, 1, kCount);
  Advance(4096);
  std::array<uint64_t, LOOM_LOW_ALLOCATION_ACTIVE_CONFLICT_WORD_COUNT>
      conflicts{};
  loom_low_allocation_active_unit_index_project_conflicts(&index_, &candidate,
                                                          conflicts.data());
  for (uint64_t word : conflicts) {
    EXPECT_EQ(word, UINT64_C(0xAAAAAAAAAAAAAAAA));
  }
  candidate.start_point = 4097;
  Advance(candidate.start_point);
  ExpectProjection(candidate, {});
  // A caller may drain the index past the final program point.
  Advance(UINT32_MAX);
  EXPECT_EQ(arena_.used_allocation_size, initialized_bytes);
  EXPECT_EQ(arena_.allocation_head, nullptr);
}

}  // namespace
}  // namespace loom
