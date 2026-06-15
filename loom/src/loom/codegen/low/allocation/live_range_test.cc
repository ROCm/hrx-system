// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/allocation/live_range.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace loom {
namespace {

loom_low_allocation_assignment_t Assignment(
    loom_value_id_t value_id, uint16_t descriptor_reg_class_id,
    uint32_t start_point, uint32_t end_point, uint32_t location_base,
    uint32_t location_count, uint32_t unit_count,
    uint32_t unit_end_point_start) {
  loom_low_allocation_assignment_t assignment = {};
  assignment.value_id = value_id;
  assignment.descriptor_reg_class_id = descriptor_reg_class_id;
  assignment.start_point = start_point;
  assignment.end_point = end_point;
  assignment.location_kind = LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER;
  assignment.location_base = location_base;
  assignment.location_count = location_count;
  assignment.unit_count = unit_count;
  assignment.unit_end_point_start = unit_end_point_start;
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
      /*unit_count=*/2, /*unit_end_point_start=*/0);

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
      /*unit_count=*/0, /*unit_end_point_start=*/0);

  EXPECT_EQ(loom_low_allocation_live_range_assignment_unit_end_point(
                nullptr, /*unit_end_point_count=*/0, &assignment,
                /*unit_offset=*/0),
            5u);
  EXPECT_EQ(loom_low_allocation_live_range_assignment_max_unit_end_point(
                nullptr, /*unit_end_point_count=*/0, &assignment),
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

  EXPECT_EQ(loom_low_allocation_live_range_interval_alignment(&scalar_interval),
            1u);
  EXPECT_EQ(loom_low_allocation_live_range_interval_alignment(&vector_interval),
            4u);
  EXPECT_EQ(loom_low_allocation_live_range_interval_alignment(&odd_interval),
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
  const loom_liveness_analysis_t liveness =
      Liveness(blocks, IREE_ARRAYSIZE(blocks));

  EXPECT_TRUE(loom_low_allocation_live_range_values_overlap(
      &liveness, /*lhs_value_id=*/1, /*lhs_start_point=*/0,
      /*lhs_end_point=*/8, /*rhs_value_id=*/2, /*rhs_start_point=*/4,
      /*rhs_end_point=*/9));
  EXPECT_FALSE(loom_low_allocation_live_range_values_overlap(
      &liveness, /*lhs_value_id=*/1, /*lhs_start_point=*/0,
      /*lhs_end_point=*/4, /*rhs_value_id=*/2, /*rhs_start_point=*/4,
      /*rhs_end_point=*/9));
}

TEST(LowAllocationLiveRangeTest, MapsOperationProgramPoints) {
  loom_block_t block = {};
  loom_op_t first_op = {};
  loom_op_t second_op = {};
  block.first_op = &first_op;
  block.last_op = &second_op;
  block.op_count = 2;
  first_op.parent_block = &block;
  first_op.next_op = &second_op;
  second_op.parent_block = &block;
  second_op.prev_op = &first_op;

  const loom_liveness_block_info_t blocks[] = {
      {
          /*.block=*/&block,
          /*.start_point=*/10,
          /*.end_point=*/12,
          /*.live_in_values=*/nullptr,
          /*.live_in_count=*/0,
          /*.live_out_values=*/nullptr,
          /*.live_out_count=*/0,
      },
  };
  const loom_liveness_analysis_t liveness =
      Liveness(blocks, IREE_ARRAYSIZE(blocks));

  uint32_t program_point = UINT32_MAX;
  IREE_ASSERT_OK(loom_low_allocation_live_range_op_program_point(
      &liveness, &first_op, &program_point));
  EXPECT_EQ(program_point, 10u);
  IREE_ASSERT_OK(loom_low_allocation_live_range_op_program_point(
      &liveness, &second_op, &program_point));
  EXPECT_EQ(program_point, 11u);
}

TEST(LowAllocationLiveRangeTest, MapsExplicitOperationProgramPoints) {
  loom_region_t region = {};
  loom_block_t block = {};
  loom_op_t first_op = {};
  loom_op_t second_op = {};
  region.block_count = 1;
  region.blocks = region.inline_blocks;
  region.inline_blocks[0] = &block;
  block.parent_region = &region;
  block.region_index = 0;
  block.first_op = &first_op;
  block.last_op = &second_op;
  block.op_count = 2;
  first_op.parent_block = &block;
  first_op.next_op = &second_op;
  second_op.parent_block = &block;
  second_op.prev_op = &first_op;

  const loom_liveness_block_info_t blocks[] = {
      {
          /*.block=*/&block,
          /*.start_point=*/20,
          /*.end_point=*/22,
          /*.live_in_values=*/nullptr,
          /*.live_in_count=*/0,
          /*.live_out_values=*/nullptr,
          /*.live_out_count=*/0,
      },
  };
  const loom_liveness_analysis_t liveness =
      Liveness(blocks, IREE_ARRAYSIZE(blocks));
  const loom_op_t* ordered_ops[] = {&second_op, &first_op};
  const loom_liveness_block_order_t block_orders[] = {
      {
          /*.block=*/&block,
          /*.ops=*/ordered_ops,
          /*.op_count=*/IREE_ARRAYSIZE(ordered_ops),
      },
  };
  const loom_liveness_order_t order = {
      /*.blocks=*/block_orders,
      /*.block_count=*/IREE_ARRAYSIZE(block_orders),
  };

  uint32_t program_point = UINT32_MAX;
  IREE_ASSERT_OK(loom_low_allocation_live_range_ordered_op_program_point(
      &liveness, &region, order, &second_op, &program_point));
  EXPECT_EQ(program_point, 20u);
  IREE_ASSERT_OK(loom_low_allocation_live_range_ordered_op_program_point(
      &liveness, &region, order, &first_op, &program_point));
  EXPECT_EQ(program_point, 21u);
}

TEST(LowAllocationLiveRangeTest, ChecksAssignmentConflicts) {
  const loom_low_reg_class_t reg_classes[] = {
      RegClass(/*alias_set_id=*/1),
      RegClass(/*alias_set_id=*/1),
  };
  loom_low_descriptor_set_t descriptor_set = {};
  descriptor_set.reg_classes = reg_classes;
  descriptor_set.reg_class_count = IREE_ARRAYSIZE(reg_classes);

  const loom_value_id_t live_in_values[] = {1, 2};
  const loom_liveness_block_info_t blocks[] = {
      {
          /*.block=*/nullptr,
          /*.start_point=*/0,
          /*.end_point=*/20,
          /*.live_in_values=*/live_in_values,
          /*.live_in_count=*/IREE_ARRAYSIZE(live_in_values),
          /*.live_out_values=*/nullptr,
          /*.live_out_count=*/0,
      },
  };
  const loom_liveness_analysis_t liveness =
      Liveness(blocks, IREE_ARRAYSIZE(blocks));

  const uint32_t unit_end_points[] = {10, 10, 10, 10};
  const loom_low_allocation_assignment_t lhs = Assignment(
      /*value_id=*/1, /*descriptor_reg_class_id=*/0, /*start_point=*/0,
      /*end_point=*/10, /*location_base=*/4, /*location_count=*/2,
      /*unit_count=*/2, /*unit_end_point_start=*/0);
  const loom_low_allocation_assignment_t rhs = Assignment(
      /*value_id=*/2, /*descriptor_reg_class_id=*/1, /*start_point=*/5,
      /*end_point=*/10, /*location_base=*/5, /*location_count=*/2,
      /*unit_count=*/2, /*unit_end_point_start=*/2);
  const loom_low_allocation_assignment_t disjoint_location = Assignment(
      /*value_id=*/2, /*descriptor_reg_class_id=*/1, /*start_point=*/5,
      /*end_point=*/10, /*location_base=*/6, /*location_count=*/2,
      /*unit_count=*/2, /*unit_end_point_start=*/2);

  EXPECT_TRUE(loom_low_allocation_live_range_assignments_conflict(
      &descriptor_set, &liveness, unit_end_points,
      IREE_ARRAYSIZE(unit_end_points), &lhs, &rhs));
  EXPECT_FALSE(loom_low_allocation_live_range_assignments_conflict(
      &descriptor_set, &liveness, unit_end_points,
      IREE_ARRAYSIZE(unit_end_points), &lhs, &disjoint_location));
}

}  // namespace
}  // namespace loom
