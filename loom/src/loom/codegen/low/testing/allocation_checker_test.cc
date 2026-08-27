// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/testing/allocation_checker.h"

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/analysis/liveness.h"
#include "loom/codegen/low/allocation/assignment.h"
#include "loom/codegen/low/frame.h"
#include "loom/codegen/low/placement.h"

namespace loom {
namespace {

loom_liveness_interval_t MakeInterval(loom_value_id_t value_id,
                                      uint32_t start_point, uint32_t end_point,
                                      loom_liveness_value_class_t value_class) {
  loom_liveness_interval_t interval = {};
  interval.value_id = value_id;
  interval.start_point = start_point;
  interval.end_point = end_point;
  interval.value_class = value_class;
  interval.unit_count = 1;
  return interval;
}

loom_low_allocation_assignment_t MakeAssignment(
    loom_value_id_t value_id, uint32_t start_point, uint32_t end_point,
    uint32_t location_base, uint32_t unit_point_start,
    loom_liveness_value_class_t value_class) {
  loom_low_allocation_assignment_t assignment = {};
  assignment.value_id = value_id;
  assignment.value_class = value_class;
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

loom_low_placement_relation_t MakeAliasRelation(
    loom_value_ordinal_t result_ordinal, loom_value_ordinal_t source_ordinal,
    loom_low_placement_relation_flags_t flags) {
  loom_low_placement_relation_t relation = {};
  relation.result_ordinal = result_ordinal;
  relation.source_ordinal = source_ordinal;
  relation.unit_count = 1;
  relation.kind = LOOM_LOW_PLACEMENT_RELATION_SAME_STORAGE;
  relation.flags = flags | LOOM_LOW_PLACEMENT_RELATION_FLAG_CAN_ALIAS_STORAGE;
  return relation;
}

class AllocationCheckerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    iree_arena_initialize(&block_pool_, &arena_);

    reg_class_.alloc_unit_bits = 32;
    reg_class_.allocatable_count = 8;
    descriptor_set_.stable_id = 1;
    descriptor_set_.reg_classes = &reg_class_;
    descriptor_set_.reg_class_count = 1;

    value_class_.type_kind = LOOM_TYPE_REGISTER;
    value_class_.register_descriptor_set_stable_id = descriptor_set_.stable_id;
    value_class_.register_class_id = 0;
    value_ids_[0] = 1;
    value_ids_[1] = 2;
    interval_indices_[0] = 0;
    interval_indices_[1] = 1;
    assignment_indices_[0] = 0;
    assignment_indices_[1] = 1;
    for (uint32_t i = 0; i < 2; ++i) {
      intervals_[i] = MakeInterval(value_ids_[i], 0, 4, value_class_);
      assignments_[i] = MakeAssignment(value_ids_[i], 0, 4, i, i, value_class_);
      unit_start_points_[i] = 0;
      unit_end_points_[i] = 4;
    }

    frame_.target.descriptor_set = &descriptor_set_;
    frame_.schedule.target.descriptor_set = &descriptor_set_;
    frame_.schedule.value_ids = value_ids_;
    frame_.schedule.value_count = 2;
    frame_.allocation.target.descriptor_set = &descriptor_set_;
    frame_.allocation.liveness.intervals = intervals_;
    frame_.allocation.liveness.interval_count = 2;
    frame_.allocation.liveness.value_ids = value_ids_;
    frame_.allocation.liveness.value_count = 2;
    frame_.allocation.liveness.value_interval_indices = interval_indices_;
    frame_.allocation.placement.value_ids = value_ids_;
    frame_.allocation.placement.value_count = 2;
    frame_.allocation.assignments = assignments_;
    frame_.allocation.assignment_count = 2;
    frame_.allocation.assignment_indices_by_value_ordinal = assignment_indices_;
    frame_.allocation.unit_start_points = unit_start_points_;
    frame_.allocation.unit_end_points = unit_end_points_;
    frame_.allocation.unit_point_count = 2;
  }

  void TearDown() override {
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  loom_low_allocation_check_result_t Check() {
    loom_low_allocation_check_result_t result = {};
    IREE_EXPECT_OK(loom_low_allocation_check_frame(&frame_, &arena_, &result));
    return result;
  }

  void ConfigureRefinedReservation(uint32_t temporary_end_point) {
    intervals_[0] = MakeInterval(value_ids_[0], /*start_point=*/2,
                                 /*end_point=*/4, value_class_);
    intervals_[0].unit_count = 2;
    assignments_[0] = MakeAssignment(
        value_ids_[0], /*start_point=*/0, /*end_point=*/4,
        /*location_base=*/0, /*unit_point_start=*/0, value_class_);
    assignments_[0].unit_count = 2;
    assignments_[0].location_count = 2;
    assignments_[0].flags =
        LOOM_LOW_ALLOCATION_ASSIGNMENT_FLAG_REFINED_UNIT_STARTS;
    unit_start_points_[0] = 0;
    unit_start_points_[1] = 2;
    unit_end_points_[0] = 4;
    unit_end_points_[1] = 4;

    intervals_[1] = MakeInterval(value_ids_[1], /*start_point=*/0,
                                 temporary_end_point, value_class_);
    assignments_[1] = MakeAssignment(
        value_ids_[1], /*start_point=*/0, temporary_end_point,
        /*location_base=*/1, /*unit_point_start=*/2, value_class_);
    unit_start_points_[2] = 0;
    unit_end_points_[2] = temporary_end_point;
    frame_.allocation.unit_point_count = 3;
  }

  iree_arena_block_pool_t block_pool_;
  iree_arena_allocator_t arena_;
  loom_low_reg_class_t reg_class_ = {};
  loom_low_descriptor_set_t descriptor_set_ = {};
  loom_liveness_value_class_t value_class_ = {};
  loom_value_id_t value_ids_[2] = {};
  uint32_t interval_indices_[2] = {};
  uint32_t assignment_indices_[2] = {};
  loom_liveness_interval_t intervals_[2] = {};
  loom_low_allocation_assignment_t assignments_[2] = {};
  uint32_t unit_start_points_[3] = {};
  uint32_t unit_end_points_[3] = {};
  loom_low_emission_frame_t frame_ = {};
};

TEST_F(AllocationCheckerTest, AcceptsDisjointAssignments) {
  const loom_low_allocation_check_result_t result = Check();
  EXPECT_EQ(result.violation_count, 0u);
}

TEST_F(AllocationCheckerTest, RejectsOverlappingLiveAssignments) {
  assignments_[1].location_base = assignments_[0].location_base;
  const loom_low_allocation_check_result_t result = Check();
  EXPECT_GT(result.violation_count, 0u);
  EXPECT_EQ(result.first_violation.kind,
            LOOM_LOW_ALLOCATION_CHECK_VIOLATION_STORAGE_CONFLICT);
  EXPECT_EQ(result.first_violation.value_id, value_ids_[0]);
  EXPECT_EQ(result.first_violation.related_value_id, value_ids_[1]);
}

TEST_F(AllocationCheckerTest, AcceptsReuseEndingAtFutureUnitStart) {
  ConfigureRefinedReservation(/*temporary_end_point=*/2);
  const loom_low_allocation_check_result_t result = Check();
  EXPECT_EQ(result.violation_count, 0u);
}

TEST_F(AllocationCheckerTest, RejectsReuseOverlappingFutureUnitStart) {
  ConfigureRefinedReservation(/*temporary_end_point=*/3);
  const loom_low_allocation_check_result_t result = Check();
  EXPECT_GT(result.violation_count, 0u);
  EXPECT_EQ(result.first_violation.kind,
            LOOM_LOW_ALLOCATION_CHECK_VIOLATION_STORAGE_CONFLICT);
}

TEST_F(AllocationCheckerTest, AcceptsExplicitStorageAlias) {
  assignments_[1].location_base = assignments_[0].location_base;
  loom_low_placement_relation_t relation =
      MakeAliasRelation(1, 0, LOOM_LOW_PLACEMENT_RELATION_FLAG_HARD);
  frame_.allocation.placement.relations = &relation;
  frame_.allocation.placement.relation_count = 1;
  const loom_low_allocation_check_result_t result = Check();
  EXPECT_EQ(result.violation_count, 0u);
}

TEST_F(AllocationCheckerTest, RejectsFixedLocationMismatch) {
  loom_low_allocation_resolved_fixed_value_t fixed = {};
  fixed.value_id = value_ids_[0];
  fixed.value_ordinal = 0;
  fixed.descriptor_reg_class_id = 0;
  fixed.interval = &intervals_[0];
  fixed.location_kind = LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER;
  fixed.location_base = 7;
  fixed.location_count = 1;
  frame_.allocation.fixed_values = &fixed;
  frame_.allocation.fixed_value_count = 1;
  const loom_low_allocation_check_result_t result = Check();
  EXPECT_GT(result.violation_count, 0u);
  EXPECT_EQ(result.first_violation.kind,
            LOOM_LOW_ALLOCATION_CHECK_VIOLATION_FIXED_LOCATION);
}

TEST_F(AllocationCheckerTest, RejectsReservedLocationOverlap) {
  loom_low_allocation_resolved_reserved_range_t reserved = {};
  reserved.descriptor_reg_class_id = 0;
  reserved.location_kind = LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER;
  reserved.location_base = assignments_[1].location_base;
  reserved.location_count = 1;
  frame_.allocation.reserved_ranges = &reserved;
  frame_.allocation.reserved_range_count = 1;
  const loom_low_allocation_check_result_t result = Check();
  EXPECT_GT(result.violation_count, 0u);
  EXPECT_EQ(result.first_violation.kind,
            LOOM_LOW_ALLOCATION_CHECK_VIOLATION_RESERVED_LOCATION);
}

}  // namespace
}  // namespace loom
