// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/allocation/spill_plan.h"

#include <vector>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/target/registers.h"

namespace loom {
namespace {

class LowAllocationSpillPlanTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    iree_arena_initialize(&block_pool_, &arena_);
    loom_context_initialize(iree_allocator_system(), &context_);
    IREE_ASSERT_OK(loom_context_finalize(&context_));
  }

  void TearDown() override {
    loom_context_deinitialize(&context_);
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  loom_module_t* AllocateModule() {
    loom_module_t* module = nullptr;
    IREE_CHECK_OK(loom_module_allocate(&context_, IREE_SV("test"), &block_pool_,
                                       nullptr, iree_allocator_system(),
                                       &module));
    return module;
  }

  iree_arena_block_pool_t block_pool_;
  iree_arena_allocator_t arena_;
  loom_context_t context_;
};

loom_low_allocation_assignment_t Assignment(loom_value_id_t value_id,
                                            uint32_t unit_count) {
  loom_low_allocation_assignment_t assignment = {};
  assignment.value_id = value_id;
  assignment.unit_count = unit_count;
  return assignment;
}

TEST_F(LowAllocationSpillPlanTest, ComputesByteLayout) {
  const loom_low_allocation_assignment_t assignment =
      Assignment(LOOM_VALUE_ID_INVALID, /*unit_count=*/3);

  uint32_t byte_size = 0;
  uint32_t byte_alignment = 0;
  IREE_ASSERT_OK(loom_low_allocation_spill_plan_layout(
      &assignment, /*alloc_unit_bits=*/16, &byte_size, &byte_alignment));
  EXPECT_EQ(byte_size, 6u);
  EXPECT_EQ(byte_alignment, 4u);

  IREE_ASSERT_OK(loom_low_allocation_spill_plan_layout(
      &assignment, /*alloc_unit_bits=*/24, &byte_size, &byte_alignment));
  EXPECT_EQ(byte_size, 9u);
  EXPECT_EQ(byte_alignment, 8u);

  const loom_low_allocation_assignment_t wide_assignment =
      Assignment(LOOM_VALUE_ID_INVALID, /*unit_count=*/4);
  IREE_ASSERT_OK(loom_low_allocation_spill_plan_layout(
      &wide_assignment, /*alloc_unit_bits=*/32, &byte_size, &byte_alignment));
  EXPECT_EQ(byte_size, 16u);
  EXPECT_EQ(byte_alignment, 16u);
}

struct SliceRange {
  // First source register unit projected by the slice.
  int64_t offset;
  // Number of register units in the result.
  uint32_t unit_count;
};

struct SliceReloadCase {
  // Stable case name reported by the parameterized test.
  const char* name;
  // Number of register units in the spilled value.
  uint32_t source_unit_count;
  // Bit width of one allocation unit.
  uint16_t alloc_unit_bits;
  // Slice uses in one block, including overlapping ranges.
  std::vector<SliceRange> slices;
  // Number of reload operations after block-local grouping.
  uint32_t reload_count;
  // Total bytes reloaded after block-local grouping.
  uint64_t reload_bytes;
};

class LowAllocationSliceReloadTest
    : public LowAllocationSpillPlanTest,
      public ::testing::WithParamInterface<SliceReloadCase> {};

TEST_P(LowAllocationSliceReloadTest, PredictsTraffic) {
  const auto& test_case = GetParam();
  loom_module_t* module = AllocateModule();
  loom_type_t source_type = loom_low_register_type(
      /*descriptor_set_stable_id=*/23, /*register_class_id=*/0,
      test_case.source_unit_count);
  IREE_ASSERT_OK(loom_module_intern_type(module, source_type, &source_type));

  loom_value_id_t source_value = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_module_define_value(module, source_type, &source_value));
  loom_builder_t builder;
  loom_builder_initialize(module, &module->arena, loom_module_block(module),
                          &builder);
  for (const auto& range : test_case.slices) {
    loom_type_t result_type = loom_low_register_carrier_type_with_unit_count(
        source_type, range.unit_count);
    IREE_ASSERT_OK(loom_module_intern_type(module, result_type, &result_type));
    loom_op_t* slice = nullptr;
    IREE_ASSERT_OK(loom_low_slice_build(&builder, source_value, range.offset,
                                        result_type, LOOM_LOCATION_UNKNOWN,
                                        &slice));
  }

  const loom_low_allocation_assignment_t assignment =
      Assignment(source_value, test_case.source_unit_count);
  loom_cfg_graph_t cfg_graph = {};
  IREE_ASSERT_OK(
      loom_cfg_graph_build(module, module->body, &arena_, &cfg_graph));
  loom_low_allocation_spill_plan_traffic_t traffic = {};
  IREE_ASSERT_OK(loom_low_allocation_spill_plan_traffic(
      module, &cfg_graph, &assignment, test_case.alloc_unit_bits, &traffic));
  EXPECT_EQ(traffic.store_count, 1u);
  EXPECT_EQ(
      traffic.store_bytes,
      (test_case.source_unit_count * test_case.alloc_unit_bits + 7u) / 8u);
  EXPECT_EQ(traffic.reload_count, test_case.reload_count);
  EXPECT_EQ(traffic.reload_bytes, test_case.reload_bytes);

  loom_module_free(module);
}

INSTANTIATE_TEST_SUITE_P(
    SliceWidths, LowAllocationSliceReloadTest,
    ::testing::Values(
        SliceReloadCase{"SparseScalars", 4, 32, {{0, 1}, {3, 1}}, 2, 8},
        SliceReloadCase{"SingleWideSlice", 4, 32, {{1, 3}}, 1, 12},
        SliceReloadCase{"MixedWidths", 4, 32, {{0, 1}, {2, 2}}, 2, 12},
        SliceReloadCase{"EqualBytes", 4, 32, {{0, 1}, {1, 3}}, 2, 16},
        SliceReloadCase{"OverlappingWidths", 4, 32, {{0, 3}, {2, 2}}, 1, 16},
        SliceReloadCase{"HalfwordUnits", 4, 16, {{0, 3}, {2, 2}}, 1, 8},
        SliceReloadCase{
            "DenseScalarUses",
            8,
            32,
            {{0, 1}, {1, 1}, {2, 1}, {3, 1}, {4, 1}, {5, 1}, {6, 1}, {7, 1}},
            1,
            32},
        SliceReloadCase{"PackedBitUnits", 8, 1, {{0, 1}, {4, 4}}, 2, 2}),
    [](const ::testing::TestParamInfo<SliceReloadCase>& info) {
      return info.param.name;
    });

TEST_F(LowAllocationSpillPlanTest, RecordsSpillRemarks) {
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
