// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/allocation/active_set.h"

#include <vector>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace loom {
namespace {

constexpr loom_liveness_analysis_t kEmptyLiveness = {};

loom_low_allocation_assignment_t Assignment(loom_value_id_t value_id,
                                            uint32_t start_point,
                                            uint32_t end_point,
                                            uint32_t location_base,
                                            uint32_t unit_point_start) {
  loom_low_allocation_assignment_t assignment = {};
  assignment.value_id = value_id;
  assignment.descriptor_reg_class_id = 0;
  assignment.start_point = start_point;
  assignment.end_point = end_point;
  assignment.unit_count = 1;
  assignment.location_kind = LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER;
  assignment.location_base = location_base;
  assignment.location_count = 1;
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

TEST(LowAllocationActiveSetTest, ExpiresAndRemovesIndexedUnits) {
  iree_arena_block_pool_t block_pool;
  iree_arena_block_pool_initialize(/*block_size=*/4096, iree_allocator_system(),
                                   &block_pool);
  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool, &arena);

  const loom_low_reg_class_t reg_classes[1] = {};
  const loom_low_descriptor_set_t descriptor_set =
      DescriptorSet(reg_classes, IREE_ARRAYSIZE(reg_classes));
  uint32_t unit_end_points[] = {10, 5, 9};
  loom_low_allocation_unit_liveness_t unit_liveness = {};
  unit_liveness.end_points = unit_end_points;
  unit_liveness.point_count = IREE_ARRAYSIZE(unit_end_points);
  const loom_low_allocation_assignment_t assignments[] = {
      Assignment(/*value_id=*/1, /*start_point=*/0, /*end_point=*/10,
                 /*location_base=*/4, /*unit_point_start=*/0),
      Assignment(/*value_id=*/2, /*start_point=*/0, /*end_point=*/5,
                 /*location_base=*/8, /*unit_point_start=*/1),
  };

  loom_low_allocation_active_set_t active_set = {};
  IREE_ASSERT_OK(loom_low_allocation_active_set_initialize(
      &kEmptyLiveness, IREE_ARRAYSIZE(assignments),
      /*program_point_count=*/11, /*unit_capacity=*/32, &arena, &active_set));
  loom_low_allocation_active_set_insert(
      &active_set, &descriptor_set, assignments, IREE_ARRAYSIZE(assignments),
      /*assignment_index=*/0);
  loom_low_allocation_active_set_insert(
      &active_set, &descriptor_set, assignments, IREE_ARRAYSIZE(assignments),
      /*assignment_index=*/1);
  ASSERT_EQ(active_set.count, 2u);

  const loom_low_allocation_assignment_t expired_conflict =
      Assignment(/*value_id=*/3, /*start_point=*/4, /*end_point=*/6,
                 /*location_base=*/8, /*unit_point_start=*/2);
  EXPECT_TRUE(loom_low_allocation_active_set_conflicts(
      &active_set, &descriptor_set, &unit_liveness, assignments,
      IREE_ARRAYSIZE(assignments), &expired_conflict,
      /*ignored_value_ids=*/nullptr,
      /*ignored_value_count=*/0));
  loom_low_allocation_active_set_expire(
      &active_set, assignments, IREE_ARRAYSIZE(assignments), /*start_point=*/6);
  ASSERT_EQ(active_set.count, 1u);
  EXPECT_EQ(active_set.assignment_indices[0], 0u);
  EXPECT_FALSE(loom_low_allocation_active_set_conflicts(
      &active_set, &descriptor_set, &unit_liveness, assignments,
      IREE_ARRAYSIZE(assignments), &expired_conflict,
      /*ignored_value_ids=*/nullptr,
      /*ignored_value_count=*/0));

  const loom_low_allocation_assignment_t live_conflict =
      Assignment(/*value_id=*/4, /*start_point=*/6, /*end_point=*/9,
                 /*location_base=*/4, /*unit_point_start=*/2);
  EXPECT_TRUE(loom_low_allocation_active_set_conflicts(
      &active_set, &descriptor_set, &unit_liveness, assignments,
      IREE_ARRAYSIZE(assignments), &live_conflict,
      /*ignored_value_ids=*/nullptr,
      /*ignored_value_count=*/0));
  loom_low_allocation_active_set_remove(&active_set, assignments,
                                        IREE_ARRAYSIZE(assignments),
                                        /*assignment_index=*/0);
  EXPECT_FALSE(loom_low_allocation_active_set_conflicts(
      &active_set, &descriptor_set, &unit_liveness, assignments,
      IREE_ARRAYSIZE(assignments), &live_conflict,
      /*ignored_value_ids=*/nullptr,
      /*ignored_value_count=*/0));
  EXPECT_EQ(active_set.count, 0u);
  loom_low_allocation_active_set_expire(&active_set, assignments,
                                        IREE_ARRAYSIZE(assignments),
                                        /*start_point=*/10);
  EXPECT_EQ(active_set.count, 0u);

  iree_arena_deinitialize(&arena);
  iree_arena_block_pool_deinitialize(&block_pool);
}

TEST(LowAllocationActiveSetTest,
     NestedLifetimesAndEarlyRemovalUseFixedStorage) {
  iree_arena_block_pool_t block_pool;
  iree_arena_block_pool_initialize(/*block_size=*/4096, iree_allocator_system(),
                                   &block_pool);
  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool, &arena);

  // Descending ends are the worst insertion order for a sorted active array.
  // Include equal-end groups and early eviction from the middle of membership.
  constexpr uint32_t kAssignmentCount = 1024;
  constexpr uint32_t kFirstEndPoint = kAssignmentCount + 1;
  constexpr uint32_t kLastEndPoint = kFirstEndPoint + kAssignmentCount / 2;
  std::vector<loom_low_allocation_assignment_t> assignments;
  for (uint32_t i = 0; i < kAssignmentCount; ++i) {
    assignments.push_back(Assignment(
        /*value_id=*/i + 1, /*start_point=*/i,
        /*end_point=*/kFirstEndPoint + (kAssignmentCount - i) / 2,
        /*location_base=*/i, /*unit_point_start=*/UINT32_MAX));
  }
  const loom_low_reg_class_t reg_classes[1] = {};
  const loom_low_descriptor_set_t descriptor_set =
      DescriptorSet(reg_classes, IREE_ARRAYSIZE(reg_classes));
  loom_low_allocation_active_set_t active_set = {};
  IREE_ASSERT_OK(loom_low_allocation_active_set_initialize(
      &kEmptyLiveness, kAssignmentCount, kLastEndPoint + 1,
      /*unit_capacity=*/kAssignmentCount, &arena, &active_set));
  const iree_host_size_t initialized_bytes = arena.used_allocation_size;
  for (uint32_t i = 0; i < kAssignmentCount; ++i) {
    loom_low_allocation_active_set_expire(&active_set, assignments.data(),
                                          assignments.size(), i);
    loom_low_allocation_active_set_insert(&active_set, &descriptor_set,
                                          assignments.data(),
                                          assignments.size(), i);
    if (i % 3 == 1) {
      loom_low_allocation_active_set_remove(&active_set, assignments.data(),
                                            assignments.size(), i - 1);
    }
  }
  for (uint32_t point = kFirstEndPoint; point <= kLastEndPoint; ++point) {
    loom_low_allocation_active_set_expire(&active_set, assignments.data(),
                                          assignments.size(), point);
    uint32_t expected_count = 0;
    for (uint32_t i = 0; i < kAssignmentCount; ++i) {
      const bool removed = i % 3 == 0 && i + 1 < kAssignmentCount;
      const bool active = !removed && assignments[i].end_point > point;
      if (active) {
        ++expected_count;
        ASSERT_LT(active_set.entries[i].position, active_set.count);
        EXPECT_EQ(active_set.assignment_indices[active_set.entries[i].position],
                  i);
      } else {
        EXPECT_EQ(active_set.entries[i].position, UINT32_MAX);
      }
    }
    EXPECT_EQ(active_set.count, expected_count);
    EXPECT_EQ(active_set.next_expiration_point, point + 1);
  }
  EXPECT_EQ(arena.used_allocation_size, initialized_bytes);
  EXPECT_EQ(active_set.count, 0u);

  iree_arena_deinitialize(&arena);
  iree_arena_block_pool_deinitialize(&block_pool);
}

TEST(LowAllocationActiveSetTest, ReusesStorageBeforeRemovedLifetimeExpires) {
  for (iree_host_size_t unit_capacity : {0u, 32u}) {
    iree_arena_block_pool_t block_pool;
    iree_arena_block_pool_initialize(
        /*block_size=*/4096, iree_allocator_system(), &block_pool);
    iree_arena_allocator_t arena;
    iree_arena_initialize(&block_pool, &arena);
    const loom_low_reg_class_t reg_classes[1] = {};
    const loom_low_descriptor_set_t descriptor_set =
        DescriptorSet(reg_classes, IREE_ARRAYSIZE(reg_classes));
    const loom_low_allocation_assignment_t assignments[] = {
        Assignment(/*value_id=*/1, /*start_point=*/0, /*end_point=*/4,
                   /*location_base=*/0, /*unit_point_start=*/0),
        Assignment(/*value_id=*/2, /*start_point=*/2, /*end_point=*/6,
                   /*location_base=*/0, /*unit_point_start=*/1),
        Assignment(/*value_id=*/3, /*start_point=*/6, /*end_point=*/8,
                   /*location_base=*/0, /*unit_point_start=*/2),
    };
    uint32_t unit_end_points[] = {4, 6, 8, 5};
    loom_low_allocation_unit_liveness_t unit_liveness = {};
    unit_liveness.end_points = unit_end_points;
    unit_liveness.point_count = IREE_ARRAYSIZE(unit_end_points);
    loom_low_allocation_active_set_t active_set = {};
    IREE_ASSERT_OK(loom_low_allocation_active_set_initialize(
        &kEmptyLiveness, IREE_ARRAYSIZE(assignments),
        /*program_point_count=*/9, unit_capacity, &arena, &active_set));
    loom_low_allocation_active_set_insert(
        &active_set, &descriptor_set, assignments, IREE_ARRAYSIZE(assignments),
        /*assignment_index=*/0);
    loom_low_allocation_active_set_remove(&active_set, assignments,
                                          IREE_ARRAYSIZE(assignments),
                                          /*assignment_index=*/0);
    loom_low_allocation_active_set_expire(&active_set, assignments,
                                          IREE_ARRAYSIZE(assignments),
                                          /*start_point=*/2);
    loom_low_allocation_active_set_insert(
        &active_set, &descriptor_set, assignments, IREE_ARRAYSIZE(assignments),
        /*assignment_index=*/1);
    loom_low_allocation_active_set_expire(&active_set, assignments,
                                          IREE_ARRAYSIZE(assignments),
                                          /*start_point=*/4);
    ASSERT_EQ(active_set.count, 1u);
    EXPECT_EQ(active_set.assignment_indices[0], 1u);
    const loom_low_allocation_assignment_t candidate =
        Assignment(/*value_id=*/4, /*start_point=*/4, /*end_point=*/5,
                   /*location_base=*/0, /*unit_point_start=*/3);
    EXPECT_TRUE(loom_low_allocation_active_set_conflicts(
        &active_set, &descriptor_set, &unit_liveness, assignments,
        IREE_ARRAYSIZE(assignments), &candidate,
        /*ignored_value_ids=*/nullptr, /*ignored_value_count=*/0));
    loom_low_allocation_active_set_expire(&active_set, assignments,
                                          IREE_ARRAYSIZE(assignments),
                                          /*start_point=*/6);
    EXPECT_EQ(active_set.count, 0u);
    EXPECT_FALSE(loom_low_allocation_active_set_conflicts(
        &active_set, &descriptor_set, &unit_liveness, assignments,
        IREE_ARRAYSIZE(assignments), &assignments[2],
        /*ignored_value_ids=*/nullptr, /*ignored_value_count=*/0));
    loom_low_allocation_active_set_insert(
        &active_set, &descriptor_set, assignments, IREE_ARRAYSIZE(assignments),
        /*assignment_index=*/2);
    loom_low_allocation_active_set_expire(&active_set, assignments,
                                          IREE_ARRAYSIZE(assignments),
                                          /*start_point=*/UINT32_MAX);
    EXPECT_EQ(active_set.count, 0u);
    EXPECT_EQ(active_set.next_expiration_point, 9u);
    iree_arena_deinitialize(&arena);
    iree_arena_block_pool_deinitialize(&block_pool);
  }
}

}  // namespace
}  // namespace loom
