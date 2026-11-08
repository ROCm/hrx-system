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

TEST(LowAllocationSpillPlanTest, ComputesByteLayout) {
  struct LayoutCase {
    // Number of allocation units occupied by the spilled value.
    uint32_t unit_count;
    // Bit width of one allocation unit.
    uint16_t alloc_unit_bits;
    // Complete storage size, rounding partial bytes upward.
    uint32_t byte_size;
    // Natural alignment, capped at the spill layout's supported maximum.
    uint32_t byte_alignment;
  };
  const LayoutCase cases[] = {
      {1, 8, 1, 1},    {2, 8, 2, 2},    {3, 16, 6, 4}, {3, 24, 9, 8},
      {4, 32, 16, 16}, {4, 64, 32, 16}, {8, 1, 1, 4},  {9, 1, 2, 4},
      {8, 8, 8, 4},    {2, 32, 8, 8},
  };
  for (const auto& test_case : cases) {
    SCOPED_TRACE(::testing::Message() << "units=" << test_case.unit_count
                                      << " bits=" << test_case.alloc_unit_bits);
    const loom_low_allocation_assignment_t assignment = {
        .value_id = LOOM_VALUE_ID_INVALID,
        .unit_count = test_case.unit_count,
    };
    const loom_low_reg_class_t reg_class = {
        .spill_unit_bits = test_case.alloc_unit_bits,
        .spill_alignment_log2 = 0x40,
    };
    uint32_t byte_size = 0;
    uint32_t byte_alignment = 0;
    IREE_ASSERT_OK(loom_low_allocation_spill_plan_layout(
        &assignment, &reg_class, &byte_size, &byte_alignment));
    EXPECT_EQ(byte_size, test_case.byte_size);
    EXPECT_EQ(byte_alignment, test_case.byte_alignment);
  }
}

TEST(LowAllocationSpillPlanTest, UsesNativeMemoryWidthAndAlignment) {
  struct LayoutCase {
    // Register payload width in bits.
    uint16_t register_bits;
    // Memory width of one spilled allocation unit in bits.
    uint16_t spill_bits;
    // Native slot alignment in bytes, expressed as log2.
    uint8_t alignment_log2;
    // Number of allocation units in the value.
    uint32_t unit_count;
    // Exact payload extent, excluding inter-slot padding.
    uint32_t expected_bytes;
  };
  const LayoutCase cases[] = {
      {20, 32, 2, 1, 4},     {80, 128, 2, 1, 16},   {160, 256, 2, 1, 32},
      {256, 256, 5, 4, 128}, {512, 512, 6, 2, 128}, {576, 576, 6, 1, 72},
      {576, 576, 6, 2, 144},
  };
  for (const auto& test_case : cases) {
    const loom_low_reg_class_t reg_class = {
        .alloc_unit_bits = test_case.register_bits,
        .spill_unit_bits = test_case.spill_bits,
        .spill_alignment_log2 = static_cast<uint8_t>(
            test_case.alignment_log2 | (test_case.alignment_log2 << 4)),
    };
    const loom_low_allocation_assignment_t assignment = {
        .unit_count = test_case.unit_count,
    };
    uint32_t byte_size = 0;
    uint32_t byte_alignment = 0;
    IREE_ASSERT_OK(loom_low_allocation_spill_plan_layout(
        &assignment, &reg_class, &byte_size, &byte_alignment));
    EXPECT_EQ(byte_size, test_case.expected_bytes);
    EXPECT_EQ(byte_alignment, 1u << test_case.alignment_log2);
  }
}

TEST(LowAllocationSpillPlanTest, GroupsSlicesByTrafficAndDensity) {
  struct GroupCase {
    // Number of byte-addressable slice uses in one block.
    uint32_t slice_count;
    // Bytes transferred by separate narrow reloads.
    uint64_t narrow_reload_bytes;
    // Bytes transferred by one reload of the complete source.
    uint32_t spill_byte_size;
    // Whether one full reload replaces the separate slice reloads.
    bool use_full_reload;
  };
  const GroupCase cases[] = {
      {1, 12, 16, false}, {2, 8, 16, false}, {2, 12, 16, false},
      {2, 16, 16, false}, {2, 20, 16, true}, {2, 10, 8, true},
      {7, 28, 32, false}, {8, 32, 32, true}, {9, 36, 36, true},
  };
  for (const auto& test_case : cases) {
    SCOPED_TRACE(::testing::Message()
                 << "slices=" << test_case.slice_count
                 << " narrow bytes=" << test_case.narrow_reload_bytes
                 << " full bytes=" << test_case.spill_byte_size);
    EXPECT_EQ(loom_low_allocation_spill_plan_use_full_slice_reload(
                  test_case.slice_count, test_case.narrow_reload_bytes,
                  test_case.spill_byte_size),
              test_case.use_full_reload);
  }
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
