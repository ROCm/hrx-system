// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/allocation/live_range.h"

#include "iree/testing/gtest.h"

namespace loom {
namespace {

constexpr loom_liveness_analysis_t kEmptyLiveness = {};

loom_low_allocation_assignment_t Assignment(
    loom_value_id_t value_id, uint16_t descriptor_reg_class_id,
    uint32_t start_point, uint32_t end_point, uint32_t location_base,
    uint32_t location_count, uint32_t unit_count, uint32_t unit_point_start) {
  loom_low_allocation_assignment_t assignment = {};
  assignment.value_id = value_id;
  assignment.descriptor_reg_class_id = descriptor_reg_class_id;
  assignment.start_point = start_point;
  assignment.end_point = end_point;
  assignment.location_kind = LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER;
  assignment.location_base = location_base;
  assignment.location_count = location_count;
  assignment.unit_count = unit_count;
  assignment.unit_point_start = unit_point_start;
  return assignment;
}

loom_low_reg_class_t RegClass(uint16_t alias_set_id) {
  loom_low_reg_class_t reg_class = {};
  reg_class.alias_set_id = alias_set_id;
  return reg_class;
}

loom_liveness_analysis_t Liveness(const loom_liveness_block_info_t* blocks,
                                  iree_host_size_t block_count) {
  loom_liveness_analysis_t liveness = {};
  liveness.blocks = blocks;
  liveness.block_count = block_count;
  return liveness;
}

TEST(LowAllocationLiveRangeTest, ReadsPerUnitEndPoints) {
  const uint32_t unit_end_points[] = {7, 11};
  const loom_low_allocation_assignment_t assignment = Assignment(
      /*value_id=*/1, /*descriptor_reg_class_id=*/0, /*start_point=*/2,
      /*end_point=*/5, /*location_base=*/0, /*location_count=*/2,
      /*unit_count=*/2, /*unit_point_start=*/0);

  EXPECT_EQ(loom_low_allocation_live_range_assignment_unit_end_point(
                unit_end_points, IREE_ARRAYSIZE(unit_end_points), &assignment,
                /*unit_offset=*/0),
            7u);
  EXPECT_EQ(loom_low_allocation_live_range_assignment_unit_end_point(
                unit_end_points, IREE_ARRAYSIZE(unit_end_points), &assignment,
                /*unit_offset=*/1),
            11u);
  EXPECT_EQ(loom_low_allocation_live_range_assignment_unit_end_point(
                unit_end_points, IREE_ARRAYSIZE(unit_end_points), &assignment,
                /*unit_offset=*/2),
            5u);
  EXPECT_EQ(loom_low_allocation_live_range_assignment_max_unit_end_point(
                unit_end_points, IREE_ARRAYSIZE(unit_end_points), &assignment),
            11u);
}

TEST(LowAllocationLiveRangeTest, ZeroUnitAssignmentsUseWholeAssignmentEnd) {
  const loom_low_allocation_assignment_t assignment = Assignment(
      /*value_id=*/1, /*descriptor_reg_class_id=*/0, /*start_point=*/2,
      /*end_point=*/5, /*location_base=*/0, /*location_count=*/1,
      /*unit_count=*/0, /*unit_point_start=*/0);

  EXPECT_EQ(loom_low_allocation_live_range_assignment_unit_end_point(
                nullptr, /*unit_point_count=*/0, &assignment,
                /*unit_offset=*/0),
            5u);
  EXPECT_EQ(loom_low_allocation_live_range_assignment_max_unit_end_point(
                nullptr, /*unit_point_count=*/0, &assignment),
            5u);
}

TEST(LowAllocationLiveRangeTest, ClassifiesAllocatableIntervals) {
  loom_liveness_interval_t register_interval = {};
  register_interval.value_class.type_kind = LOOM_TYPE_REGISTER;
  register_interval.unit_count = 2;

  loom_liveness_interval_t zero_unit_interval = register_interval;
  zero_unit_interval.unit_count = 0;

  loom_liveness_interval_t non_register_interval = register_interval;
  non_register_interval.value_class.type_kind = LOOM_TYPE_SCALAR;

  EXPECT_TRUE(loom_low_allocation_live_range_interval_is_allocatable(
      &register_interval));
  EXPECT_FALSE(loom_low_allocation_live_range_interval_is_allocatable(
      &zero_unit_interval));
  EXPECT_FALSE(loom_low_allocation_live_range_interval_is_allocatable(
      &non_register_interval));
}

TEST(LowAllocationLiveRangeTest, ComputesIntervalStorageEndPoints) {
  loom_liveness_interval_t live_interval = {};
  live_interval.start_point = 3;
  live_interval.end_point = 7;

  loom_liveness_interval_t dead_result_interval = {};
  dead_result_interval.start_point = 3;
  dead_result_interval.end_point = 3;

  loom_liveness_interval_t saturated_interval = {};
  saturated_interval.start_point = UINT32_MAX;
  saturated_interval.end_point = UINT32_MAX;

  EXPECT_EQ(
      loom_low_allocation_live_range_interval_storage_end_point(&live_interval),
      7u);
  EXPECT_EQ(loom_low_allocation_live_range_interval_initial_unit_end_point(
                &live_interval),
            3u);
  EXPECT_EQ(loom_low_allocation_live_range_interval_storage_end_point(
                &dead_result_interval),
            4u);
  EXPECT_EQ(loom_low_allocation_live_range_interval_initial_unit_end_point(
                &dead_result_interval),
            4u);
  EXPECT_EQ(loom_low_allocation_live_range_interval_storage_end_point(
                &saturated_interval),
            UINT32_MAX);
}

TEST(LowAllocationLiveRangeTest, ComputesIntervalAlignment) {
  loom_liveness_interval_t scalar_interval = {};
  scalar_interval.unit_count = 1;

  loom_liveness_interval_t vector_interval = {};
  vector_interval.unit_count = 4;

  loom_liveness_interval_t odd_interval = {};
  odd_interval.unit_count = 3;

  EXPECT_EQ(loom_low_allocation_live_range_interval_alignment(
                &scalar_interval, /*reg_class_flags=*/0),
            1u);
  EXPECT_EQ(loom_low_allocation_live_range_interval_alignment(
                &vector_interval, /*reg_class_flags=*/0),
            4u);
  EXPECT_EQ(loom_low_allocation_live_range_interval_alignment(
                &odd_interval, /*reg_class_flags=*/0),
            1u);
  EXPECT_EQ(loom_low_allocation_live_range_interval_alignment(
                &vector_interval, LOOM_LOW_REG_CLASS_FLAG_UNALIGNED_RANGES),
            1u);
}

TEST(LowAllocationLiveRangeTest, ChecksBlockObservableOverlap) {
  const loom_value_id_t live_in_values[] = {1};
  const loom_liveness_block_info_t blocks[] = {
      {
          /*.block=*/nullptr,
          /*.start_point=*/0,
          /*.end_point=*/10,
          /*.live_in_values=*/live_in_values,
          /*.live_in_count=*/IREE_ARRAYSIZE(live_in_values),
          /*.live_out_values=*/nullptr,
          /*.live_out_count=*/0,
      },
  };
  loom_liveness_analysis_t liveness = Liveness(blocks, IREE_ARRAYSIZE(blocks));

  EXPECT_TRUE(loom_low_allocation_live_range_values_overlap(
      &liveness, /*lhs_value_id=*/1, /*lhs_start_point=*/0,
      /*lhs_end_point=*/8, /*rhs_value_id=*/2, /*rhs_start_point=*/4,
      /*rhs_end_point=*/9));
  EXPECT_FALSE(loom_low_allocation_live_range_values_overlap(
      &liveness, /*lhs_value_id=*/1, /*lhs_start_point=*/0,
      /*lhs_end_point=*/4, /*rhs_value_id=*/2, /*rhs_start_point=*/4,
      /*rhs_end_point=*/9));
}

TEST(LowAllocationLiveRangeTest, ChecksAssignmentConflicts) {
  const loom_low_reg_class_t reg_classes[] = {
      RegClass(/*alias_set_id=*/1),
      RegClass(/*alias_set_id=*/1),
  };
  loom_low_descriptor_set_t descriptor_set = {};
  descriptor_set.reg_classes = reg_classes;
  descriptor_set.reg_class_count = IREE_ARRAYSIZE(reg_classes);

  uint32_t unit_end_points[] = {10, 10, 10, 10};
  const loom_low_allocation_assignment_t lhs = Assignment(
      /*value_id=*/1, /*descriptor_reg_class_id=*/0, /*start_point=*/0,
      /*end_point=*/10, /*location_base=*/4, /*location_count=*/2,
      /*unit_count=*/2, /*unit_point_start=*/0);
  const loom_low_allocation_assignment_t rhs = Assignment(
      /*value_id=*/2, /*descriptor_reg_class_id=*/1, /*start_point=*/5,
      /*end_point=*/10, /*location_base=*/5, /*location_count=*/2,
      /*unit_count=*/2, /*unit_point_start=*/2);
  const loom_low_allocation_assignment_t disjoint_location = Assignment(
      /*value_id=*/2, /*descriptor_reg_class_id=*/1, /*start_point=*/5,
      /*end_point=*/10, /*location_base=*/6, /*location_count=*/2,
      /*unit_count=*/2, /*unit_point_start=*/2);

  EXPECT_TRUE(loom_low_allocation_live_range_assignments_conflict(
      &descriptor_set, &kEmptyLiveness, /*unit_start_points=*/nullptr,
      unit_end_points, IREE_ARRAYSIZE(unit_end_points), &lhs, &rhs));
  EXPECT_FALSE(loom_low_allocation_live_range_assignments_conflict(
      &descriptor_set, &kEmptyLiveness, /*unit_start_points=*/nullptr,
      unit_end_points, IREE_ARRAYSIZE(unit_end_points), &lhs,
      &disjoint_location));
}

TEST(LowAllocationLiveRangeTest, RefinesAssignmentConflictsByUnitStart) {
  const loom_low_reg_class_t reg_classes[] = {
      RegClass(/*alias_set_id=*/1),
  };
  loom_low_descriptor_set_t descriptor_set = {};
  descriptor_set.reg_classes = reg_classes;
  descriptor_set.reg_class_count = IREE_ARRAYSIZE(reg_classes);

  uint32_t unit_end_points[] = {10, 10, 5, 6};
  uint32_t unit_start_points[] = {0, 5, 2, 2};
  loom_low_allocation_assignment_t reservation = Assignment(
      /*value_id=*/1, /*descriptor_reg_class_id=*/0, /*start_point=*/0,
      /*end_point=*/10, /*location_base=*/4, /*location_count=*/2,
      /*unit_count=*/2, /*unit_point_start=*/0);
  reservation.flags = LOOM_LOW_ALLOCATION_ASSIGNMENT_FLAG_REFINED_UNIT_STARTS;
  const loom_low_allocation_assignment_t ends_at_second_unit_start = Assignment(
      /*value_id=*/2, /*descriptor_reg_class_id=*/0, /*start_point=*/2,
      /*end_point=*/5, /*location_base=*/5, /*location_count=*/1,
      /*unit_count=*/1, /*unit_point_start=*/2);
  const loom_low_allocation_assignment_t overlaps_second_unit_start =
      Assignment(
          /*value_id=*/3, /*descriptor_reg_class_id=*/0, /*start_point=*/2,
          /*end_point=*/6, /*location_base=*/5, /*location_count=*/1,
          /*unit_count=*/1, /*unit_point_start=*/3);

  EXPECT_FALSE(loom_low_allocation_live_range_assignments_conflict(
      &descriptor_set, &kEmptyLiveness, unit_start_points, unit_end_points,
      IREE_ARRAYSIZE(unit_end_points), &reservation,
      &ends_at_second_unit_start));
  EXPECT_TRUE(loom_low_allocation_live_range_assignments_conflict(
      &descriptor_set, &kEmptyLiveness, unit_start_points, unit_end_points,
      IREE_ARRAYSIZE(unit_end_points), &reservation,
      &overlaps_second_unit_start));
}

TEST(LowAllocationLiveRangeTest, AssignmentConflictsRejectDisjointLifetime) {
  const loom_low_reg_class_t reg_classes[] = {
      RegClass(/*alias_set_id=*/1),
      RegClass(/*alias_set_id=*/1),
  };
  loom_low_descriptor_set_t descriptor_set = {};
  descriptor_set.reg_classes = reg_classes;
  descriptor_set.reg_class_count = IREE_ARRAYSIZE(reg_classes);

  const loom_low_allocation_assignment_t lhs = Assignment(
      /*value_id=*/1, /*descriptor_reg_class_id=*/0, /*start_point=*/0,
      /*end_point=*/5, /*location_base=*/4, /*location_count=*/2,
      /*unit_count=*/2, /*unit_point_start=*/0);
  const loom_low_allocation_assignment_t rhs = Assignment(
      /*value_id=*/2, /*descriptor_reg_class_id=*/1, /*start_point=*/5,
      /*end_point=*/10, /*location_base=*/4, /*location_count=*/2,
      /*unit_count=*/2, /*unit_point_start=*/2);
  EXPECT_FALSE(loom_low_allocation_live_range_assignments_conflict(
      &descriptor_set, &kEmptyLiveness, /*unit_start_points=*/nullptr,
      /*unit_end_points=*/nullptr, /*unit_point_count=*/0, &lhs, &rhs));
}

TEST(LowAllocationLiveRangeTest, AssignmentConflictsUsePhysicalStorageOverlap) {
  const loom_low_reg_class_t reg_classes[] = {
      RegClass(/*alias_set_id=*/1),
      RegClass(/*alias_set_id=*/1),
  };
  loom_low_descriptor_set_t descriptor_set = {};
  descriptor_set.reg_classes = reg_classes;
  descriptor_set.reg_class_count = IREE_ARRAYSIZE(reg_classes);

  const loom_liveness_block_info_t blocks[] = {
      {
          /*.block=*/nullptr,
          /*.start_point=*/40,
          /*.end_point=*/80,
          /*.live_in_values=*/nullptr,
          /*.live_in_count=*/0,
          /*.live_out_values=*/nullptr,
          /*.live_out_count=*/0,
      },
  };
  const loom_liveness_analysis_t liveness =
      Liveness(blocks, IREE_ARRAYSIZE(blocks));
  EXPECT_FALSE(loom_low_allocation_live_range_values_overlap(
      &liveness, /*lhs_value_id=*/1, /*lhs_start_point=*/0,
      /*lhs_end_point=*/100, /*rhs_value_id=*/2, /*rhs_start_point=*/40,
      /*rhs_end_point=*/80));

  uint32_t unit_end_points[] = {100, 100, 80, 80};
  const loom_low_allocation_assignment_t lhs = Assignment(
      /*value_id=*/1, /*descriptor_reg_class_id=*/0, /*start_point=*/0,
      /*end_point=*/100, /*location_base=*/4, /*location_count=*/2,
      /*unit_count=*/2, /*unit_point_start=*/0);
  const loom_low_allocation_assignment_t rhs = Assignment(
      /*value_id=*/2, /*descriptor_reg_class_id=*/1, /*start_point=*/40,
      /*end_point=*/80, /*location_base=*/4, /*location_count=*/2,
      /*unit_count=*/2, /*unit_point_start=*/2);

  EXPECT_TRUE(loom_low_allocation_live_range_assignments_conflict(
      &descriptor_set, &kEmptyLiveness, /*unit_start_points=*/nullptr,
      unit_end_points, IREE_ARRAYSIZE(unit_end_points), &lhs, &rhs));
}

}  // namespace
}  // namespace loom
