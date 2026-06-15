// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/allocation/assignment.h"

#include "iree/testing/gtest.h"

namespace loom {
namespace {

loom_low_allocation_assignment_t Assignment(
    uint16_t descriptor_reg_class_id,
    loom_low_allocation_location_kind_t location_kind, uint32_t location_base,
    uint32_t location_count) {
  loom_low_allocation_assignment_t assignment = {};
  assignment.descriptor_reg_class_id = descriptor_reg_class_id;
  assignment.location_kind = location_kind;
  assignment.location_base = location_base;
  assignment.location_count = location_count;
  return assignment;
}

TEST(LowAllocationAssignmentRangeTest, MatchesPhysicalRegisterClass) {
  EXPECT_TRUE(loom_low_allocation_location_kind_is_known(
      LOOM_LOW_ALLOCATION_LOCATION_UNASSIGNED));
  EXPECT_TRUE(loom_low_allocation_location_kind_is_known(
      LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER));
  EXPECT_TRUE(loom_low_allocation_location_kind_is_known(
      LOOM_LOW_ALLOCATION_LOCATION_TARGET_ID));
  EXPECT_TRUE(loom_low_allocation_location_kind_is_known(
      LOOM_LOW_ALLOCATION_LOCATION_SPILL_SLOT));
  EXPECT_FALSE(loom_low_allocation_location_kind_is_known(
      (loom_low_allocation_location_kind_t)99));

  const loom_low_allocation_assignment_t assignment = Assignment(
      /*descriptor_reg_class_id=*/3,
      LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER, /*location_base=*/4,
      /*location_count=*/2);
  EXPECT_TRUE(loom_low_allocation_location_kind_is_register_like(
      LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER));
  EXPECT_TRUE(loom_low_allocation_assignment_is_register_like(&assignment));
  EXPECT_FALSE(loom_low_allocation_assignment_is_spill_slot(&assignment));
  EXPECT_TRUE(loom_low_allocation_assignment_is_physical_register_class(
      &assignment, 3));
  EXPECT_FALSE(loom_low_allocation_assignment_is_physical_register_class(
      &assignment, 4));

  const loom_low_allocation_assignment_t target_id_assignment = Assignment(
      /*descriptor_reg_class_id=*/3, LOOM_LOW_ALLOCATION_LOCATION_TARGET_ID,
      /*location_base=*/4, /*location_count=*/2);
  EXPECT_TRUE(loom_low_allocation_location_kind_is_register_like(
      LOOM_LOW_ALLOCATION_LOCATION_TARGET_ID));
  EXPECT_TRUE(
      loom_low_allocation_assignment_is_register_like(&target_id_assignment));
  EXPECT_FALSE(
      loom_low_allocation_assignment_is_spill_slot(&target_id_assignment));
  EXPECT_FALSE(loom_low_allocation_assignment_is_physical_register_class(
      &target_id_assignment, 3));

  const loom_low_allocation_assignment_t spill_assignment = Assignment(
      /*descriptor_reg_class_id=*/3, LOOM_LOW_ALLOCATION_LOCATION_SPILL_SLOT,
      /*location_base=*/4, /*location_count=*/2);
  EXPECT_FALSE(loom_low_allocation_location_kind_is_register_like(
      LOOM_LOW_ALLOCATION_LOCATION_SPILL_SLOT));
  EXPECT_FALSE(
      loom_low_allocation_assignment_is_register_like(&spill_assignment));
  EXPECT_TRUE(loom_low_allocation_assignment_is_spill_slot(&spill_assignment));

  const loom_low_allocation_assignment_t empty_assignment = Assignment(
      /*descriptor_reg_class_id=*/3,
      LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER, /*location_base=*/0,
      /*location_count=*/0);
  EXPECT_FALSE(loom_low_allocation_assignment_is_physical_register_class(
      &empty_assignment, 3));
}

TEST(LowAllocationAssignmentRangeTest, ComputesLocationEnd) {
  const loom_low_allocation_assignment_t assignment = Assignment(
      /*descriptor_reg_class_id=*/0, LOOM_LOW_ALLOCATION_LOCATION_UNASSIGNED,
      /*location_base=*/UINT32_MAX, /*location_count=*/2);
  uint64_t end = 0;
  EXPECT_TRUE(
      loom_low_allocation_assignment_location_exclusive_end(&assignment, &end));
  EXPECT_EQ(end, (uint64_t)UINT32_MAX + 2u);

  const loom_low_allocation_assignment_t empty_assignment = Assignment(
      /*descriptor_reg_class_id=*/0, LOOM_LOW_ALLOCATION_LOCATION_UNASSIGNED,
      /*location_base=*/UINT32_MAX, /*location_count=*/0);
  end = 1;
  EXPECT_FALSE(loom_low_allocation_assignment_location_exclusive_end(
      &empty_assignment, &end));
  EXPECT_EQ(end, 0u);
}

TEST(LowAllocationAssignmentRangeTest, MatchesAndOverlapsConcreteRanges) {
  const loom_low_allocation_assignment_t lhs = Assignment(
      /*descriptor_reg_class_id=*/1,
      LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER, /*location_base=*/4,
      /*location_count=*/2);
  const loom_low_allocation_assignment_t same = Assignment(
      /*descriptor_reg_class_id=*/1,
      LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER, /*location_base=*/4,
      /*location_count=*/2);
  const loom_low_allocation_assignment_t overlapping = Assignment(
      /*descriptor_reg_class_id=*/1,
      LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER, /*location_base=*/5,
      /*location_count=*/2);
  const loom_low_allocation_assignment_t adjacent = Assignment(
      /*descriptor_reg_class_id=*/1,
      LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER, /*location_base=*/6,
      /*location_count=*/2);
  const loom_low_allocation_assignment_t different_class = Assignment(
      /*descriptor_reg_class_id=*/2,
      LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER, /*location_base=*/4,
      /*location_count=*/2);

  EXPECT_TRUE(loom_low_allocation_assignment_location_range_equal(&lhs, &same));
  EXPECT_TRUE(
      loom_low_allocation_assignment_location_ranges_overlap(&lhs, &same));
  EXPECT_FALSE(
      loom_low_allocation_assignment_location_range_equal(&lhs, &overlapping));
  EXPECT_TRUE(loom_low_allocation_assignment_location_ranges_overlap(
      &lhs, &overlapping));
  EXPECT_FALSE(
      loom_low_allocation_assignment_location_ranges_overlap(&lhs, &adjacent));
  EXPECT_FALSE(loom_low_allocation_assignment_location_ranges_overlap(
      &lhs, &different_class));

  const loom_low_allocation_assignment_t empty = Assignment(
      /*descriptor_reg_class_id=*/1,
      LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER, /*location_base=*/4,
      /*location_count=*/0);
  EXPECT_FALSE(
      loom_low_allocation_assignment_location_range_equal(&lhs, &empty));
  EXPECT_FALSE(
      loom_low_allocation_assignment_location_ranges_overlap(&lhs, &empty));
}

TEST(LowAllocationAssignmentRangeTest, MatchesAndOverlapsSubranges) {
  const loom_low_allocation_assignment_t lhs = Assignment(
      /*descriptor_reg_class_id=*/1,
      LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER, /*location_base=*/10,
      /*location_count=*/8);
  const loom_low_allocation_assignment_t rhs = Assignment(
      /*descriptor_reg_class_id=*/1,
      LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER, /*location_base=*/12,
      /*location_count=*/8);
  const loom_low_allocation_assignment_t different_kind = Assignment(
      /*descriptor_reg_class_id=*/1, LOOM_LOW_ALLOCATION_LOCATION_TARGET_ID,
      /*location_base=*/12, /*location_count=*/8);

  EXPECT_TRUE(loom_low_allocation_assignment_subranges_equal(&lhs, 2, &rhs, 0,
                                                             /*unit_count=*/2));
  EXPECT_TRUE(loom_low_allocation_assignment_subranges_overlap(
      &lhs, 1, &rhs, 0, /*unit_count=*/2));
  EXPECT_FALSE(loom_low_allocation_assignment_subranges_overlap(
      &lhs, 0, &rhs, 4, /*unit_count=*/2));
  EXPECT_FALSE(loom_low_allocation_assignment_subranges_equal(
      &lhs, 2, &different_kind, 0, /*unit_count=*/2));
}

}  // namespace
}  // namespace loom
