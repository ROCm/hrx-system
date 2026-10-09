// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/allocation/active_unit.h"

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace loom {
namespace {

loom_low_allocation_assignment_t Assignment(
    loom_value_id_t value_id, uint16_t descriptor_reg_class_id,
    uint32_t start_point, uint32_t end_point, uint32_t location_base,
    uint32_t location_count, uint32_t unit_point_start) {
  loom_low_allocation_assignment_t assignment = {};
  assignment.value_id = value_id;
  assignment.descriptor_reg_class_id = descriptor_reg_class_id;
  assignment.start_point = start_point;
  assignment.end_point = end_point;
  assignment.unit_count = location_count;
  assignment.location_kind = LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER;
  assignment.location_base = location_base;
  assignment.location_count = location_count;
  assignment.unit_point_start = unit_point_start;
  return assignment;
}

loom_low_descriptor_set_t DescriptorSet(const loom_low_reg_class_t* reg_classes,
                                        iree_host_size_t reg_class_count) {
  loom_low_descriptor_set_t descriptor_set = {};
  descriptor_set.reg_classes = reg_classes;
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
      /*.point=*/point,
      /*.segment_starts_by_assignment_index=*/segment_starts->data(),
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
  loom_low_allocation_unit_liveness_t unit_liveness = {};
  unit_liveness.end_points = unit_end_points;
  unit_liveness.point_count = IREE_ARRAYSIZE(unit_end_points);
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
  loom_low_allocation_unit_liveness_t unit_liveness = {};
  unit_liveness.end_points = unit_end_points;
  unit_liveness.point_count = IREE_ARRAYSIZE(unit_end_points);
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
  assignments[7].liveness_segments = {/*.start=*/0, /*.count=*/1};
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
  loom_low_allocation_unit_liveness_t unit_liveness = {};
  unit_liveness.start_points = unit_start_points;
  unit_liveness.end_points = unit_end_points;
  unit_liveness.point_count = IREE_ARRAYSIZE(unit_end_points);
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
          /*.name_string_ref=*/0,
          /*.atomic_unit_start=*/0,
          /*.atomic_unit_count=*/2,
          /*.reserved=*/0,
      },
      {
          /*.name_string_ref=*/0,
          /*.atomic_unit_start=*/2,
          /*.atomic_unit_count=*/1,
          /*.reserved=*/0,
      },
      {
          /*.name_string_ref=*/0,
          /*.atomic_unit_start=*/3,
          /*.atomic_unit_count=*/2,
          /*.reserved=*/0,
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
  loom_low_allocation_unit_liveness_t unit_liveness = {};
  unit_liveness.end_points = unit_end_points;
  unit_liveness.point_count = IREE_ARRAYSIZE(unit_end_points);
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

}  // namespace
}  // namespace loom
