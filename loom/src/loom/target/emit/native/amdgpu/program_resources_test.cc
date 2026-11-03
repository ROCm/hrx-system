// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/native/amdgpu/program_resources.h"

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace loom {
namespace {

class AmdgpuProgramResourcesTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    iree_arena_initialize(&block_pool_, &arena_);
  }

  void TearDown() override {
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  iree_arena_block_pool_t block_pool_;
  iree_arena_allocator_t arena_;
};

TEST_F(AmdgpuProgramResourcesTest, AggregatesFunctionAndControlResources) {
  loom_amdgpu_program_resource_builder_t builder = {};
  IREE_ASSERT_OK(
      loom_amdgpu_program_resource_builder_initialize(4, &arena_, &builder));

  const uint32_t first_extents[] = {5, 7, 2, 0};
  loom_low_emission_frame_t first_frame = {};
  first_frame.allocation.physical_extents.ends_by_reg_class = first_extents;
  first_frame.allocation.physical_extents.count = IREE_ARRAYSIZE(first_extents);
  loom_amdgpu_function_plan_t first_plan = {};
  first_plan.preflight.next_free_sgpr = 5;
  first_plan.preflight.next_free_vgpr = 7;
  first_plan.preflight.next_free_agpr = 2;
  first_plan.packet_plan.wait_states.minimum_vgpr_count = 10;
  loom_amdgpu_storage_layout_t first_layout = {};
  IREE_ASSERT_OK(loom_amdgpu_program_resource_builder_append_function(
      &first_frame, &first_plan, &arena_, &builder, &first_layout));

  const uint32_t second_extents[] = {9, 4, 6, 1};
  loom_low_emission_frame_t second_frame = {};
  second_frame.allocation.physical_extents.ends_by_reg_class = second_extents;
  second_frame.allocation.physical_extents.count =
      IREE_ARRAYSIZE(second_extents);
  loom_amdgpu_function_plan_t second_plan = {};
  second_plan.preflight.next_free_sgpr = 9;
  second_plan.preflight.next_free_vgpr = 4;
  second_plan.preflight.next_free_agpr = 6;
  loom_amdgpu_storage_layout_t second_layout = {};
  IREE_ASSERT_OK(loom_amdgpu_program_resource_builder_append_function(
      &second_frame, &second_plan, &arena_, &builder, &second_layout));
  loom_amdgpu_program_resource_builder_require_registers(
      /*sgpr_count=*/36, /*vgpr_count=*/32, /*agpr_count=*/8, &builder);

  loom_amdgpu_program_resources_t resources = {};
  loom_amdgpu_program_resource_builder_finish(&builder, &resources);
  ASSERT_EQ(resources.direct_resource_unit_count, 4u);
  EXPECT_EQ(resources.direct_resource_units[0], 9u);
  EXPECT_EQ(resources.direct_resource_units[1], 7u);
  EXPECT_EQ(resources.direct_resource_units[2], 6u);
  EXPECT_EQ(resources.direct_resource_units[3], 1u);
  EXPECT_EQ(resources.sgpr_count, 36u);
  EXPECT_EQ(resources.vgpr_count, 32u);
  EXPECT_EQ(resources.agpr_count, 8u);
}

}  // namespace
}  // namespace loom
