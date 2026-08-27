// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/allocation/spill_plan.h"

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

TEST_F(LowAllocationSpillPlanTest, PredictsSliceReloadBytes) {
  loom_module_t* module = AllocateModule();
  loom_type_t wide_type =
      loom_low_register_type(/*descriptor_set_stable_id=*/23,
                             /*register_class_id=*/0, /*unit_count=*/4);
  IREE_ASSERT_OK(loom_module_intern_type(module, wide_type, &wide_type));
  loom_type_t lane_type =
      loom_low_register_carrier_type_with_unit_count(wide_type,
                                                     /*unit_count=*/1);
  IREE_ASSERT_OK(loom_module_intern_type(module, lane_type, &lane_type));

  loom_value_id_t source_value = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_module_define_value(module, wide_type, &source_value));
  loom_builder_t builder;
  loom_builder_initialize(module, &module->arena, loom_module_block(module),
                          &builder);
  loom_op_t* slice0 = nullptr;
  IREE_ASSERT_OK(loom_low_slice_build(&builder, source_value, /*offset=*/0,
                                      lane_type, LOOM_LOCATION_UNKNOWN,
                                      &slice0));
  loom_op_t* slice3 = nullptr;
  IREE_ASSERT_OK(loom_low_slice_build(&builder, source_value, /*offset=*/3,
                                      lane_type, LOOM_LOCATION_UNKNOWN,
                                      &slice3));

  const loom_low_allocation_assignment_t assignment =
      Assignment(source_value, /*unit_count=*/4);
  loom_cfg_graph_t cfg_graph = {};
  IREE_ASSERT_OK(
      loom_cfg_graph_build(module, module->body, &arena_, &cfg_graph));
  loom_low_allocation_spill_plan_traffic_t traffic = {};
  IREE_ASSERT_OK(loom_low_allocation_spill_plan_traffic(
      module, &cfg_graph, &assignment, /*alloc_unit_bits=*/32, &traffic));
  EXPECT_EQ(traffic.store_count, 1u);
  EXPECT_EQ(traffic.store_bytes, 16u);
  EXPECT_EQ(traffic.reload_count, 2u);
  EXPECT_EQ(traffic.reload_bytes, 8u);

  loom_module_free(module);
}

TEST_F(LowAllocationSpillPlanTest, PredictsDenseSliceReloadTraffic) {
  loom_module_t* module = AllocateModule();
  loom_type_t wide_type =
      loom_low_register_type(/*descriptor_set_stable_id=*/23,
                             /*register_class_id=*/0, /*unit_count=*/8);
  IREE_ASSERT_OK(loom_module_intern_type(module, wide_type, &wide_type));
  loom_type_t lane_type =
      loom_low_register_carrier_type_with_unit_count(wide_type,
                                                     /*unit_count=*/1);
  IREE_ASSERT_OK(loom_module_intern_type(module, lane_type, &lane_type));

  loom_value_id_t source_value = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_module_define_value(module, wide_type, &source_value));
  loom_builder_t builder;
  loom_builder_initialize(module, &module->arena, loom_module_block(module),
                          &builder);
  for (int64_t i = 0; i < 8; ++i) {
    loom_op_t* slice = nullptr;
    IREE_ASSERT_OK(loom_low_slice_build(&builder, source_value, i, lane_type,
                                        LOOM_LOCATION_UNKNOWN, &slice));
  }

  const loom_low_allocation_assignment_t assignment =
      Assignment(source_value, /*unit_count=*/8);
  loom_cfg_graph_t cfg_graph = {};
  IREE_ASSERT_OK(
      loom_cfg_graph_build(module, module->body, &arena_, &cfg_graph));
  loom_low_allocation_spill_plan_traffic_t traffic = {};
  IREE_ASSERT_OK(loom_low_allocation_spill_plan_traffic(
      module, &cfg_graph, &assignment, /*alloc_unit_bits=*/32, &traffic));
  EXPECT_EQ(traffic.store_count, 1u);
  EXPECT_EQ(traffic.store_bytes, 32u);
  EXPECT_EQ(traffic.reload_count, 1u);
  EXPECT_EQ(traffic.reload_bytes, 32u);

  loom_module_free(module);
}

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
