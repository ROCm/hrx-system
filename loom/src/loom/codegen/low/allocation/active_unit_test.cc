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

constexpr loom_liveness_analysis_t kEmptyLiveness = {};

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

  loom_low_allocation_active_unit_index_t index = {};
  IREE_ASSERT_OK(loom_low_allocation_active_unit_index_initialize(
      IREE_ARRAYSIZE(assignments), /*unit_capacity=*/32, &arena, &index));
  EXPECT_TRUE(loom_low_allocation_active_unit_index_is_enabled(&index));

  loom_low_allocation_active_unit_index_insert_assignment(
      &index, &descriptor_set, assignments, IREE_ARRAYSIZE(assignments),
      /*assignment_index=*/0);
  EXPECT_TRUE(loom_low_allocation_active_unit_index_contains_assignment(
      &index, /*assignment_index=*/0));
  EXPECT_TRUE(loom_low_allocation_active_unit_index_conflicts(
      &index, &descriptor_set, &kEmptyLiveness, &unit_liveness, assignments,
      IREE_ARRAYSIZE(assignments), &assignments[1],
      /*ignored_value_ids=*/nullptr,
      /*ignored_value_count=*/0));
  uint32_t conflict_indices[2] = {};
  uint16_t conflict_count = 0;
  IREE_ASSERT_OK(loom_low_allocation_active_unit_index_collect_conflicts(
      &index, &descriptor_set, &kEmptyLiveness, &unit_liveness, assignments,
      IREE_ARRAYSIZE(assignments), &assignments[1],
      /*ignored_value_ids=*/nullptr,
      /*ignored_value_count=*/0, conflict_indices,
      IREE_ARRAYSIZE(conflict_indices), &conflict_count));
  ASSERT_EQ(conflict_count, 1u);
  EXPECT_EQ(conflict_indices[0], 0u);

  const loom_value_id_t ignored_value_ids[] = {1};
  EXPECT_FALSE(loom_low_allocation_active_unit_index_conflicts(
      &index, &descriptor_set, &kEmptyLiveness, &unit_liveness, assignments,
      IREE_ARRAYSIZE(assignments), &assignments[1], ignored_value_ids,
      IREE_ARRAYSIZE(ignored_value_ids)));
  conflict_count = 0;
  IREE_ASSERT_OK(loom_low_allocation_active_unit_index_collect_conflicts(
      &index, &descriptor_set, &kEmptyLiveness, &unit_liveness, assignments,
      IREE_ARRAYSIZE(assignments), &assignments[1], ignored_value_ids,
      IREE_ARRAYSIZE(ignored_value_ids), conflict_indices,
      IREE_ARRAYSIZE(conflict_indices), &conflict_count));
  EXPECT_EQ(conflict_count, 0u);

  loom_low_allocation_active_unit_index_remove_assignment(
      &index, assignments, IREE_ARRAYSIZE(assignments), /*assignment_index=*/0);
  EXPECT_FALSE(loom_low_allocation_active_unit_index_contains_assignment(
      &index, /*assignment_index=*/0));
  EXPECT_FALSE(loom_low_allocation_active_unit_index_conflicts(
      &index, &descriptor_set, &kEmptyLiveness, &unit_liveness, assignments,
      IREE_ARRAYSIZE(assignments), &assignments[1],
      /*ignored_value_ids=*/nullptr,
      /*ignored_value_count=*/0));

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

  loom_low_allocation_active_unit_index_t index = {};
  IREE_ASSERT_OK(loom_low_allocation_active_unit_index_initialize(
      IREE_ARRAYSIZE(assignments), /*unit_capacity=*/32, &arena, &index));
  loom_low_allocation_active_unit_index_insert_assignment(
      &index, &descriptor_set, assignments, IREE_ARRAYSIZE(assignments),
      /*assignment_index=*/0);

  EXPECT_FALSE(loom_low_allocation_active_unit_index_conflicts(
      &index, &descriptor_set, &kEmptyLiveness, &unit_liveness, assignments,
      IREE_ARRAYSIZE(assignments), &assignments[1],
      /*ignored_value_ids=*/nullptr,
      /*ignored_value_count=*/0));
  EXPECT_TRUE(loom_low_allocation_active_unit_index_conflicts(
      &index, &descriptor_set, &kEmptyLiveness, &unit_liveness, assignments,
      IREE_ARRAYSIZE(assignments), &assignments[2],
      /*ignored_value_ids=*/nullptr,
      /*ignored_value_count=*/0));

  iree_arena_deinitialize(&arena);
  iree_arena_block_pool_deinitialize(&block_pool);
}

TEST(LowAllocationActiveUnitTest, TracksUnindexedAssignments) {
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
      /*assignment_capacity=*/1, /*unit_capacity=*/32, &arena, &index));
  loom_low_allocation_active_unit_index_insert_assignment(
      &index, &descriptor_set, &assignment, /*assignment_count=*/1,
      /*assignment_index=*/0);
  EXPECT_EQ(loom_low_allocation_active_unit_index_unindexed_count(&index), 1u);
  loom_low_allocation_active_unit_index_remove_assignment(
      &index, &assignment, /*assignment_count=*/1, /*assignment_index=*/0);
  EXPECT_EQ(loom_low_allocation_active_unit_index_unindexed_count(&index), 0u);

  iree_arena_deinitialize(&arena);
  iree_arena_block_pool_deinitialize(&block_pool);
}

}  // namespace
}  // namespace loom
