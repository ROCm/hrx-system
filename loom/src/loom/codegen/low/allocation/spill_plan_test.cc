// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/allocation/spill_plan.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace loom {
namespace {

using ::iree::StatusCode;

loom_low_allocation_assignment_t Assignment(uint32_t unit_count) {
  loom_low_allocation_assignment_t assignment = {};
  assignment.unit_count = unit_count;
  return assignment;
}

TEST(LowAllocationSpillPlanTest, ComputesByteLayout) {
  const loom_low_allocation_assignment_t assignment =
      Assignment(/*unit_count=*/3);

  uint32_t byte_size = 0;
  uint32_t byte_alignment = 0;
  IREE_ASSERT_OK(loom_low_allocation_spill_plan_layout(
      &assignment, /*alloc_unit_bits=*/16, &byte_size, &byte_alignment));
  EXPECT_EQ(byte_size, 6u);
  EXPECT_EQ(byte_alignment, 2u);

  IREE_ASSERT_OK(loom_low_allocation_spill_plan_layout(
      &assignment, /*alloc_unit_bits=*/24, &byte_size, &byte_alignment));
  EXPECT_EQ(byte_size, 9u);
  EXPECT_EQ(byte_alignment, 4u);
}

TEST(LowAllocationSpillPlanTest, RejectsZeroBitAllocationUnits) {
  const loom_low_allocation_assignment_t assignment =
      Assignment(/*unit_count=*/1);

  uint32_t byte_size = 0;
  uint32_t byte_alignment = 0;
  IREE_EXPECT_STATUS_IS(
      StatusCode::kFailedPrecondition,
      loom_low_allocation_spill_plan_layout(&assignment, /*alloc_unit_bits=*/0,
                                            &byte_size, &byte_alignment));
}

TEST(LowAllocationSpillPlanTest, RecordsSpillRemarks) {
  loom_low_allocation_remark_t remarks[1] = {};
  iree_host_size_t remark_count = 0;
  loom_low_allocation_spill_remark_record(
      remarks, &remark_count, /*assignment_index=*/7, /*budget_units=*/32,
      /*required_units=*/4);

  ASSERT_EQ(remark_count, 1u);
  EXPECT_EQ(remarks[0].kind, LOOM_LOW_ALLOCATION_REMARK_SPILL);
  EXPECT_EQ(remarks[0].assignment_index, 7u);
  EXPECT_EQ(remarks[0].budget_units, 32u);
  EXPECT_EQ(remarks[0].required_units, 4u);
}

}  // namespace
}  // namespace loom
