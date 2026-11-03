// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/allocation/live_range.h"

#include "iree/testing/gtest.h"
#include "loom/codegen/low/allocation/content_identity.h"

namespace loom {
namespace {

loom_low_allocation_assignment_t Assignment(
    loom_value_id_t value_id, uint16_t descriptor_reg_class_id,
    uint32_t start_point, uint32_t end_point, uint32_t location_base,
    uint32_t location_count, uint32_t unit_count, uint32_t unit_point_start) {
  loom_low_allocation_assignment_t assignment = {
      .value_id = value_id,
      .descriptor_reg_class_id = descriptor_reg_class_id,
      .start_point = start_point,
      .end_point = end_point,
      .unit_count = unit_count,
      .location_kind = LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER,
      .location_base = location_base,
      .location_count = location_count,
      .unit_point_start = unit_point_start,
  };
  return assignment;
}

loom_low_reg_class_t RegClass(uint16_t alias_set_id) {
  loom_low_reg_class_t reg_class = {.alias_set_id = alias_set_id};
  return reg_class;
}

loom_liveness_analysis_t Liveness(const loom_liveness_block_info_t* blocks,
                                  iree_host_size_t block_count) {
  loom_liveness_analysis_t liveness = {
      .blocks = blocks,
      .block_count = block_count,
  };
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

  loom_liveness_interval_t dead_result_interval = {
      .start_point = 3,
      .end_point = 3,
  };

  loom_liveness_interval_t saturated_interval = {
      .start_point = UINT32_MAX,
      .end_point = UINT32_MAX,
  };

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
  loom_low_descriptor_set_t descriptor_set = {
      .reg_classes = reg_classes,
      .reg_class_count = IREE_ARRAYSIZE(reg_classes),
  };
  const uint32_t unit_counts[] = {1, 2, 3, 4, 5, 6, 7, 8, 12};
  const uint32_t unaligned[] = {1, 2, 1, 4, 1, 1, 1, 8, 1};
  const uint32_t aligned[] = {1, 2, 2, 4, 2, 2, 2, 8, 2};
  loom_low_placement_table_t placement = {};
  for (size_t i = 0; i < IREE_ARRAYSIZE(unit_counts); ++i) {
    loom_liveness_interval_t interval = {.unit_count = unit_counts[i]};
    loom_liveness_analysis_t liveness = {
        .intervals = &interval,
        .interval_count = 1,
    };
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
  loom_low_reg_class_t reg_class = {
      .flags = LOOM_LOW_REG_CLASS_FLAG_EVEN_ALIGNED_TUPLES,
  };
  loom_low_descriptor_set_t descriptor_set = {
      .reg_classes = &reg_class,
      .reg_class_count = 1,
  };
  loom_liveness_interval_t intervals[3] = {};
  for (auto& interval : intervals) {
    interval.unit_count = 4;
  }
  loom_liveness_analysis_t liveness = {
      .intervals = intervals,
      .interval_count = IREE_ARRAYSIZE(intervals),
  };
  const loom_low_placement_operand_constraints_t operands[] = {
      {0, 0, false}, {0, 1, false}, {0, 3, false}};
  loom_low_placement_table_t placement = {
      .operand_constraints_by_interval = operands,
  };
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
          .block = nullptr,
          .start_point = 0,
          .end_point = 10,
          .live_in_values = live_in_values,
          .live_in_count = IREE_ARRAYSIZE(live_in_values),
          .live_out_values = nullptr,
          .live_out_count = 0,
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
  loom_low_descriptor_set_t descriptor_set = {
      .reg_classes = reg_classes,
      .reg_class_count = IREE_ARRAYSIZE(reg_classes),
  };

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
      IREE_ARRAYSIZE(unit_end_points), &lhs, &rhs, /*shared_content=*/nullptr));
  EXPECT_FALSE(loom_low_allocation_live_range_assignments_conflict(
      &descriptor_set, /*storage_segments=*/nullptr,
      /*unit_start_points=*/nullptr, unit_end_points,
      IREE_ARRAYSIZE(unit_end_points), &lhs, &disjoint_location,
      /*shared_content=*/nullptr));
}

TEST(LowAllocationLiveRangeTest, RefinesAssignmentConflictsByUnitStart) {
  const loom_low_reg_class_t reg_classes[] = {
      RegClass(/*alias_set_id=*/1),
  };
  loom_low_descriptor_set_t descriptor_set = {
      .reg_classes = reg_classes,
      .reg_class_count = IREE_ARRAYSIZE(reg_classes),
  };

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
      &ends_at_second_unit_start, /*shared_content=*/nullptr));
  EXPECT_TRUE(loom_low_allocation_live_range_assignments_conflict(
      &descriptor_set, /*storage_segments=*/nullptr, unit_start_points,
      unit_end_points, IREE_ARRAYSIZE(unit_end_points), &reservation,
      &overlaps_second_unit_start, /*shared_content=*/nullptr));
}

TEST(LowAllocationLiveRangeTest, AssignmentConflictsRejectDisjointLifetime) {
  const loom_low_reg_class_t reg_classes[] = {
      RegClass(/*alias_set_id=*/1),
      RegClass(/*alias_set_id=*/1),
  };
  loom_low_descriptor_set_t descriptor_set = {
      .reg_classes = reg_classes,
      .reg_class_count = IREE_ARRAYSIZE(reg_classes),
  };

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
      /*unit_end_points=*/nullptr, /*unit_point_count=*/0, &lhs, &rhs,
      /*shared_content=*/nullptr));
}

TEST(LowAllocationLiveRangeTest, PreservesSparseGapsAgainstContiguousStorage) {
  const uint16_t atomic_units[] = {0};
  const uint16_t candidate_ids[] = {0};
  const uint16_t candidate_ordinals[] = {0};
  const uint16_t allocation_ordinals[] = {0};
  const loom_low_physical_register_t physical_registers[] = {
      {
          .name_string_ref = 0,
          .atomic_unit_start = 0,
          .atomic_unit_count = 1,
          .reserved = 0,
      },
  };
  const loom_liveness_segment_t segments[] = {{0, 4}, {20, 30}};
  for (const uint32_t flags :
       {0u, uint32_t(LOOM_LOW_REG_CLASS_FLAG_EXPLICIT_PHYSICAL_REGISTERS)}) {
    SCOPED_TRACE(flags);
    loom_low_reg_class_t reg_class = RegClass(/*alias_set_id=*/1);
    reg_class.flags = flags;
    reg_class.allocatable_count = 1;
    reg_class.candidate_lookup.register_count = 1;
    loom_low_descriptor_set_t descriptor_set = {
        .reg_classes = &reg_class,
        .reg_class_count = 1,
        .physical_registers = physical_registers,
        .physical_register_count = IREE_ARRAYSIZE(physical_registers),
        .physical_register_candidate_ids = candidate_ids,
        .physical_register_allocation_ordinals = allocation_ordinals,
        .physical_register_candidate_count = IREE_ARRAYSIZE(candidate_ids),
        .physical_register_candidate_ordinals = candidate_ordinals,
        .physical_register_candidate_ordinal_count =
            IREE_ARRAYSIZE(candidate_ordinals),
        .physical_register_atomic_units = atomic_units,
        .physical_register_atomic_unit_count = IREE_ARRAYSIZE(atomic_units),
    };

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
                    &contiguous, /*shared_content=*/nullptr),
                expected_conflict);
      EXPECT_EQ(loom_low_allocation_live_range_assignments_conflict(
                    &descriptor_set, segments, /*unit_start_points=*/nullptr,
                    unit_end_points, IREE_ARRAYSIZE(unit_end_points),
                    &contiguous, &sparse, /*shared_content=*/nullptr),
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

TEST(LowAllocationLiveRangeTest,
     ExpandedExecutionDefinitionsProtectRetainedSparseStorage) {
  const uint16_t atomic_units[] = {0};
  const uint16_t candidate_ids[] = {0};
  const uint16_t candidate_ordinals[] = {0};
  const uint16_t allocation_ordinals[] = {0};
  const loom_low_physical_register_t physical_registers[] = {
      {
          .name_string_ref = 0,
          .atomic_unit_start = 0,
          .atomic_unit_count = 1,
          .reserved = 0,
      },
  };
  const loom_liveness_segment_t segments[] = {{0, 4}, {20, 30}};
  for (const uint32_t class_flags :
       {0u, uint32_t(LOOM_LOW_REG_CLASS_FLAG_EXPLICIT_PHYSICAL_REGISTERS)}) {
    SCOPED_TRACE(class_flags);
    loom_low_reg_class_t reg_class = RegClass(/*alias_set_id=*/1);
    reg_class.flags = class_flags;
    reg_class.allocatable_count = 1;
    reg_class.candidate_lookup.register_count = 1;
    const loom_low_descriptor_set_t descriptor_set = {
        .reg_classes = &reg_class,
        .reg_class_count = 1,
        .physical_registers = physical_registers,
        .physical_register_count = IREE_ARRAYSIZE(physical_registers),
        .physical_register_candidate_ids = candidate_ids,
        .physical_register_allocation_ordinals = allocation_ordinals,
        .physical_register_candidate_count = IREE_ARRAYSIZE(candidate_ids),
        .physical_register_candidate_ordinals = candidate_ordinals,
        .physical_register_candidate_ordinal_count =
            IREE_ARRAYSIZE(candidate_ordinals),
        .physical_register_atomic_units = atomic_units,
        .physical_register_atomic_unit_count = IREE_ARRAYSIZE(atomic_units),
    };
    loom_low_allocation_assignment_t retained = Assignment(
        /*value_id=*/1, /*descriptor_reg_class_id=*/0, /*start_point=*/0,
        /*end_point=*/30, /*location_base=*/0, /*location_count=*/1,
        /*unit_count=*/1, /*unit_point_start=*/0);
    retained.liveness_segments = {0, IREE_ARRAYSIZE(segments)};
    loom_low_allocation_assignment_t expanded_definition = Assignment(
        /*value_id=*/2, /*descriptor_reg_class_id=*/0, /*start_point=*/10,
        /*end_point=*/11, /*location_base=*/0, /*location_count=*/1,
        /*unit_count=*/1, /*unit_point_start=*/1);
    const uint32_t unit_start_points[] = {0, 10};
    const uint32_t unit_end_points[] = {30, 11};

    EXPECT_FALSE(loom_low_allocation_live_range_assignments_conflict(
        &descriptor_set, segments, unit_start_points, unit_end_points,
        IREE_ARRAYSIZE(unit_end_points), &retained, &expanded_definition,
        /*shared_content=*/nullptr));

    expanded_definition.flags =
        LOOM_LOW_ALLOCATION_ASSIGNMENT_FLAG_EXPANDED_EXECUTION_DEFINITION;
    EXPECT_TRUE(loom_low_allocation_live_range_assignments_conflict(
        &descriptor_set, segments, unit_start_points, unit_end_points,
        IREE_ARRAYSIZE(unit_end_points), &retained, &expanded_definition,
        /*shared_content=*/nullptr));
    EXPECT_TRUE(loom_low_allocation_live_range_assignments_conflict(
        &descriptor_set, segments, unit_start_points, unit_end_points,
        IREE_ARRAYSIZE(unit_end_points), &expanded_definition, &retained,
        /*shared_content=*/nullptr));

    uint32_t sweep_segment = retained.liveness_segments.start;
    EXPECT_TRUE(loom_low_allocation_live_range_sweep_assignments_conflict(
        /*sweep_point=*/10, &sweep_segment, &descriptor_set, segments,
        unit_start_points, unit_end_points, IREE_ARRAYSIZE(unit_end_points),
        &retained, &expanded_definition, /*shared_content=*/nullptr));
  }
}

TEST(LowAllocationLiveRangeTest,
     ExpandedExecutionDefinitionsRetainPointLocalReuse) {
  const loom_low_reg_class_t reg_class = RegClass(/*alias_set_id=*/1);
  const loom_low_descriptor_set_t descriptor_set = {
      .reg_classes = &reg_class,
      .reg_class_count = 1,
  };
  loom_low_allocation_assignment_t expanded_definition = Assignment(
      /*value_id=*/1, /*descriptor_reg_class_id=*/0, /*start_point=*/10,
      /*end_point=*/11, /*location_base=*/1, /*location_count=*/1,
      /*unit_count=*/1, /*unit_point_start=*/0);
  expanded_definition.flags =
      LOOM_LOW_ALLOCATION_ASSIGNMENT_FLAG_EXPANDED_EXECUTION_DEFINITION;
  const loom_low_allocation_assignment_t ends_at_definition = Assignment(
      /*value_id=*/2, /*descriptor_reg_class_id=*/0, /*start_point=*/0,
      /*end_point=*/10, /*location_base=*/1, /*location_count=*/1,
      /*unit_count=*/1, /*unit_point_start=*/1);
  const loom_low_allocation_assignment_t starts_after_definition = Assignment(
      /*value_id=*/3, /*descriptor_reg_class_id=*/0, /*start_point=*/11,
      /*end_point=*/20, /*location_base=*/1, /*location_count=*/1,
      /*unit_count=*/1, /*unit_point_start=*/2);
  const uint32_t unit_start_points[] = {10, 0, 11};
  const uint32_t unit_end_points[] = {11, 10, 20};

  EXPECT_FALSE(loom_low_allocation_live_range_assignments_conflict(
      &descriptor_set, /*storage_segments=*/nullptr, unit_start_points,
      unit_end_points, IREE_ARRAYSIZE(unit_end_points), &expanded_definition,
      &ends_at_definition, /*shared_content=*/nullptr));
  EXPECT_FALSE(loom_low_allocation_live_range_assignments_conflict(
      &descriptor_set, /*storage_segments=*/nullptr, unit_start_points,
      unit_end_points, IREE_ARRAYSIZE(unit_end_points), &expanded_definition,
      &starts_after_definition, /*shared_content=*/nullptr));
}

TEST(LowAllocationLiveRangeTest,
     ExpandedExecutionDefinitionsRespectPerUnitStorage) {
  const loom_low_reg_class_t reg_class = RegClass(/*alias_set_id=*/1);
  const loom_low_descriptor_set_t descriptor_set = {
      .reg_classes = &reg_class,
      .reg_class_count = 1,
  };
  loom_low_allocation_assignment_t expanded_definition = Assignment(
      /*value_id=*/1, /*descriptor_reg_class_id=*/0, /*start_point=*/10,
      /*end_point=*/11, /*location_base=*/2, /*location_count=*/1,
      /*unit_count=*/1, /*unit_point_start=*/0);
  expanded_definition.flags =
      LOOM_LOW_ALLOCATION_ASSIGNMENT_FLAG_EXPANDED_EXECUTION_DEFINITION;
  loom_low_allocation_assignment_t retained = Assignment(
      /*value_id=*/2, /*descriptor_reg_class_id=*/0, /*start_point=*/0,
      /*end_point=*/20, /*location_base=*/1, /*location_count=*/2,
      /*unit_count=*/2, /*unit_point_start=*/1);
  retained.flags = LOOM_LOW_ALLOCATION_ASSIGNMENT_FLAG_REFINED_UNIT_STARTS;
  const uint32_t unit_start_points[] = {10, 0, 0};
  uint32_t unit_end_points[] = {11, 20, 10};

  EXPECT_FALSE(loom_low_allocation_live_range_assignments_conflict(
      &descriptor_set, /*storage_segments=*/nullptr, unit_start_points,
      unit_end_points, IREE_ARRAYSIZE(unit_end_points), &expanded_definition,
      &retained, /*shared_content=*/nullptr));
  unit_end_points[2] = 20;
  EXPECT_TRUE(loom_low_allocation_live_range_assignments_conflict(
      &descriptor_set, /*storage_segments=*/nullptr, unit_start_points,
      unit_end_points, IREE_ARRAYSIZE(unit_end_points), &expanded_definition,
      &retained, /*shared_content=*/nullptr));
}

TEST(LowAllocationLiveRangeTest, AssignmentConflictsUsePhysicalStorageOverlap) {
  const loom_low_reg_class_t reg_classes[] = {
      RegClass(/*alias_set_id=*/1),
      RegClass(/*alias_set_id=*/1),
  };
  loom_low_descriptor_set_t descriptor_set = {
      .reg_classes = reg_classes,
      .reg_class_count = IREE_ARRAYSIZE(reg_classes),
  };

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
      IREE_ARRAYSIZE(unit_end_points), &lhs, &rhs, /*shared_content=*/nullptr));
}

TEST(LowAllocationLiveRangeTest, SharesOnlyReadOnlyOverlappingUnits) {
  const loom_low_reg_class_t reg_class = RegClass(1);
  const loom_low_descriptor_set_t descriptors = {
      .reg_classes = &reg_class,
      .reg_class_count = 1,
  };
  auto source = Assignment(1, 0, 0, 10, 0, 2, 2, 0);
  auto aggregate = Assignment(2, 0, 2, 12, 0, 2, 2, 2);
  aggregate.flags = LOOM_LOW_ALLOCATION_ASSIGNMENT_FLAG_REFINED_UNIT_STARTS;
  const uint32_t starts[] = {0, 0, 2, 6};
  uint32_t ends[] = {10, 10, 12, 12};
  const uint32_t origins[] = {0, 1, 0, 3};
  uint64_t written = 0;
  const loom_low_allocation_content_identity_t contents = {
      .origins = origins,
      .written_origins = {.bit_count = 4, .words = &written},
  };
  auto conflicts = [&]() {
    return loom_low_allocation_live_range_assignments_conflict(
        &descriptors, nullptr, starts, ends, IREE_ARRAYSIZE(ends), &source,
        &aggregate, &contents);
  };
  // The equal first units may share, but the second aggregate unit still
  // overwrites a different live source unit.
  EXPECT_TRUE(conflicts());
  ends[1] = 6;
  EXPECT_FALSE(conflicts());
  // A destructive user of the shared contents needs an independent copy.
  written = 1;
  EXPECT_TRUE(conflicts());
  written = 0;
  // Expanded definitions retain their full physical-write obligation.
  aggregate.flags |=
      LOOM_LOW_ALLOCATION_ASSIGNMENT_FLAG_EXPANDED_EXECUTION_DEFINITION;
  EXPECT_TRUE(conflicts());
}

TEST(LowAllocationLiveRangeTest, ContentSharingRequiresExactPhysicalViews) {
  loom_low_reg_class_t classes[] = {RegClass(0), RegClass(0)};
  for (auto& reg_class : classes) {
    reg_class.flags = LOOM_LOW_REG_CLASS_FLAG_PHYSICAL |
                      LOOM_LOW_REG_CLASS_FLAG_EXPLICIT_PHYSICAL_REGISTERS;
    reg_class.allocatable_count = 1;
    reg_class.candidate_lookup.register_count = 1;
  }
  classes[1].physical_register_candidate_start = 1;
  classes[1].candidate_lookup.ordinal_start = 1;
  classes[1].candidate_lookup.register_base = 1;
  const uint16_t units[] = {0, 1, 0};
  const loom_low_physical_register_t registers[] = {
      {.atomic_unit_start = 0, .atomic_unit_count = 2},
      {.atomic_unit_start = 2, .atomic_unit_count = 1},
  };
  const uint16_t candidates[] = {0, 1};
  const uint16_t ordinals[] = {0, 0};
  const loom_low_descriptor_set_t descriptors = {
      .reg_classes = classes,
      .reg_class_count = IREE_ARRAYSIZE(classes),
      .physical_registers = registers,
      .physical_register_count = IREE_ARRAYSIZE(registers),
      .physical_register_candidate_ids = candidates,
      .physical_register_allocation_ordinals = ordinals,
      .physical_register_candidate_count = IREE_ARRAYSIZE(candidates),
      .physical_register_candidate_ordinals = ordinals,
      .physical_register_candidate_ordinal_count = IREE_ARRAYSIZE(ordinals),
      .physical_register_atomic_units = units,
      .physical_register_atomic_unit_count = IREE_ARRAYSIZE(units),
  };
  const auto source = Assignment(1, 0, 0, 10, 0, 1, 1, 0);
  auto candidate = Assignment(2, 0, 2, 10, 0, 1, 1, 1);
  const uint32_t ends[] = {10, 10};
  const uint32_t origins[] = {0, 0};
  const loom_low_allocation_content_identity_t contents = {.origins = origins};
  auto conflicts = [&]() {
    return loom_low_allocation_live_range_assignments_conflict(
        &descriptors, nullptr, nullptr, ends, IREE_ARRAYSIZE(ends), &source,
        &candidate, &contents);
  };
  EXPECT_FALSE(conflicts());
  candidate.descriptor_reg_class_id = 1;
  candidate.location_base = 1;
  EXPECT_TRUE(conflicts());
}

}  // namespace
}  // namespace loom
