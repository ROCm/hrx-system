// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/allocation/live_range.h"

#include "iree/testing/gtest.h"

namespace loom {
namespace {

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
  loom_low_reg_class_t reg_class = {.alias_set_id = alias_set_id};
  return reg_class;
}

loom_liveness_analysis_t Liveness(const loom_liveness_block_info_t* blocks,
                                  iree_host_size_t block_count) {
  loom_liveness_analysis_t liveness = {.blocks = blocks,
                                       .block_count = block_count};
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
  loom_liveness_interval_t live_interval = {.start_point = 3, .end_point = 7};

  loom_liveness_interval_t dead_result_interval = {.start_point = 3,
                                                   .end_point = 3};

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

TEST(LowAllocationLiveRangeTest, SeparatesRequiredAndPreferredAlignment) {
  loom_low_reg_class_t reg_classes[2] = {};
  reg_classes[1].flags = LOOM_LOW_REG_CLASS_FLAG_EVEN_ALIGNED_TUPLES;
  loom_low_descriptor_set_t descriptor_set = {};
  descriptor_set.reg_classes = reg_classes;
  descriptor_set.reg_class_count = IREE_ARRAYSIZE(reg_classes);
  const uint32_t unit_counts[] = {1, 2, 3, 4, 5, 6, 7, 8, 12};
  const uint32_t unaligned[] = {1, 2, 1, 4, 1, 1, 1, 8, 1};
  const uint32_t aligned[] = {1, 2, 2, 4, 2, 2, 2, 8, 2};
  loom_low_placement_table_t placement = {};
  for (size_t i = 0; i < IREE_ARRAYSIZE(unit_counts); ++i) {
    loom_liveness_interval_t interval = {.unit_count = unit_counts[i]};
    loom_liveness_analysis_t liveness = {.intervals = &interval,
                                         .interval_count = 1};
    EXPECT_EQ(loom_low_allocation_live_range_interval_alignment(
                  &descriptor_set, &liveness,
                  placement.operand_constraints_by_interval, &interval),
              1u);
    EXPECT_EQ(loom_low_reg_class_preferred_unit_alignment(&reg_classes[0],
                                                          interval.unit_count),
              unaligned[i]);
    interval.value_class.register_class_id = 1;
    EXPECT_EQ(loom_low_allocation_live_range_interval_alignment(
                  &descriptor_set, &liveness,
                  placement.operand_constraints_by_interval, &interval),
              interval.unit_count > 1 ? 2u : 1u);
    EXPECT_EQ(loom_low_reg_class_preferred_unit_alignment(&reg_classes[1],
                                                          interval.unit_count),
              aligned[i]);
  }
}

TEST(LowAllocationLiveRangeTest, CombinesRetainedOperandAndClassAlignment) {
  loom_low_reg_class_t reg_class = {};
  reg_class.flags = LOOM_LOW_REG_CLASS_FLAG_EVEN_ALIGNED_TUPLES;
  loom_low_descriptor_set_t descriptor_set = {.reg_classes = &reg_class,
                                              .reg_class_count = 1};
  loom_liveness_interval_t intervals[3] = {};
  for (auto& interval : intervals) {
    interval.unit_count = 4;
  }
  loom_liveness_analysis_t liveness = {};
  liveness.intervals = intervals;
  liveness.interval_count = IREE_ARRAYSIZE(intervals);
  const loom_low_placement_operand_constraints_t operands[] = {
      {0, 0, false}, {0, 1, false}, {0, 3, false}};
  loom_low_placement_table_t placement = {.operand_constraints_by_interval =
                                              operands};
  const uint32_t expected[] = {2, 2, 8};
  for (size_t i = 0; i < IREE_ARRAYSIZE(intervals); ++i) {
    EXPECT_EQ(loom_low_allocation_live_range_interval_alignment(
                  &descriptor_set, &liveness,
                  placement.operand_constraints_by_interval, &intervals[i]),
              expected[i]);
  }
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
      &descriptor_set, /*storage_segments=*/nullptr,
      /*unit_start_points=*/nullptr, unit_end_points,
      IREE_ARRAYSIZE(unit_end_points), &lhs, &rhs));
  EXPECT_FALSE(loom_low_allocation_live_range_assignments_conflict(
      &descriptor_set, /*storage_segments=*/nullptr,
      /*unit_start_points=*/nullptr, unit_end_points,
      IREE_ARRAYSIZE(unit_end_points), &lhs, &disjoint_location));
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
      &descriptor_set, /*storage_segments=*/nullptr, unit_start_points,
      unit_end_points, IREE_ARRAYSIZE(unit_end_points), &reservation,
      &ends_at_second_unit_start));
  EXPECT_TRUE(loom_low_allocation_live_range_assignments_conflict(
      &descriptor_set, /*storage_segments=*/nullptr, unit_start_points,
      unit_end_points, IREE_ARRAYSIZE(unit_end_points), &reservation,
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
      &descriptor_set, /*storage_segments=*/nullptr,
      /*unit_start_points=*/nullptr,
      /*unit_end_points=*/nullptr, /*unit_point_count=*/0, &lhs, &rhs));
}

TEST(LowAllocationLiveRangeTest, PreservesSparseGapsAgainstContiguousStorage) {
  const uint16_t atomic_units[] = {0};
  const uint16_t candidate_ids[] = {0};
  const uint16_t candidate_ordinals[] = {0};
  const uint16_t allocation_ordinals[] = {0};
  const loom_low_physical_register_t physical_registers[] = {
      {.name_string_ref = 0,
       .atomic_unit_start = 0,
       .atomic_unit_count = 1,
       .reserved = 0},
  };
  const loom_liveness_segment_t segments[] = {{0, 4}, {20, 30}};
  for (const uint32_t flags :
       {0u, uint32_t(LOOM_LOW_REG_CLASS_FLAG_EXPLICIT_PHYSICAL_REGISTERS)}) {
    SCOPED_TRACE(flags);
    loom_low_reg_class_t reg_class = RegClass(/*alias_set_id=*/1);
    reg_class.flags = flags;
    reg_class.allocatable_count = 1;
    reg_class.candidate_lookup.register_count = 1;
    loom_low_descriptor_set_t descriptor_set = {};
    descriptor_set.reg_classes = &reg_class;
    descriptor_set.reg_class_count = 1;
    descriptor_set.physical_registers = physical_registers;
    descriptor_set.physical_register_count = IREE_ARRAYSIZE(physical_registers);
    descriptor_set.physical_register_atomic_units = atomic_units;
    descriptor_set.physical_register_atomic_unit_count =
        IREE_ARRAYSIZE(atomic_units);
    descriptor_set.physical_register_candidate_ids = candidate_ids;
    descriptor_set.physical_register_candidate_count =
        IREE_ARRAYSIZE(candidate_ids);
    descriptor_set.physical_register_candidate_ordinals = candidate_ordinals;
    descriptor_set.physical_register_candidate_ordinal_count =
        IREE_ARRAYSIZE(candidate_ordinals);
    descriptor_set.physical_register_allocation_ordinals = allocation_ordinals;

    loom_low_allocation_assignment_t sparse = Assignment(
        /*value_id=*/1, /*descriptor_reg_class_id=*/0, /*start_point=*/0,
        /*end_point=*/30, /*location_base=*/0, /*location_count=*/1,
        /*unit_count=*/1, /*unit_point_start=*/0);
    sparse.liveness_segments = {0, IREE_ARRAYSIZE(segments)};
    auto check_contiguous = [&](uint32_t start_point, uint32_t end_point,
                                bool expected_conflict) {
      SCOPED_TRACE(start_point);
      SCOPED_TRACE(end_point);
      const uint32_t unit_end_points[] = {30, end_point};
      const loom_low_allocation_assignment_t contiguous = Assignment(
          /*value_id=*/2, /*descriptor_reg_class_id=*/0, start_point, end_point,
          /*location_base=*/0, /*location_count=*/1,
          /*unit_count=*/1, /*unit_point_start=*/1);
      EXPECT_EQ(loom_low_allocation_live_range_assignments_conflict(
                    &descriptor_set, segments, /*unit_start_points=*/nullptr,
                    unit_end_points, IREE_ARRAYSIZE(unit_end_points), &sparse,
                    &contiguous),
                expected_conflict);
      EXPECT_EQ(loom_low_allocation_live_range_assignments_conflict(
                    &descriptor_set, segments, /*unit_start_points=*/nullptr,
                    unit_end_points, IREE_ARRAYSIZE(unit_end_points),
                    &contiguous, &sparse),
                expected_conflict);
    };
    check_contiguous(4, 20, false);
    check_contiguous(5, 19, false);
    check_contiguous(3, 20, true);
    check_contiguous(4, 21, true);
    check_contiguous(0, 30, true);
    check_contiguous(30, 31, false);
  }
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
          .block = nullptr,
          .start_point = 40,
          .end_point = 80,
          .live_in_values = nullptr,
          .live_in_count = 0,
          .live_out_values = nullptr,
          .live_out_count = 0,
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
      &descriptor_set, /*storage_segments=*/nullptr,
      /*unit_start_points=*/nullptr, unit_end_points,
      IREE_ARRAYSIZE(unit_end_points), &lhs, &rhs));
}

}  // namespace
}  // namespace loom
