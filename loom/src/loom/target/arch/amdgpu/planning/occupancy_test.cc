// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/planning/occupancy.h"

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace loom {
namespace {

class AmdgpuOccupancyTargetResourcesTest : public ::testing::Test {
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

  loom_amdgpu_occupancy_target_resources_t Build(
      iree_string_view_t processor_name, uint32_t wave_size,
      uint32_t scalar_register_count, uint32_t vector_register_count,
      uint32_t flat_workgroup_size, uint32_t local_memory_bytes) {
    const loom_amdgpu_processor_info_t* processor =
        loom_amdgpu_target_info_find_processor(processor_name);
    EXPECT_NE(processor, nullptr);
    loom_amdgpu_occupancy_target_resources_t resources = {};
    IREE_EXPECT_OK(loom_amdgpu_occupancy_build_target_resources(
        processor, wave_size, scalar_register_count, vector_register_count,
        flat_workgroup_size, local_memory_bytes, &arena_, &resources));
    return resources;
  }

  iree_arena_block_pool_t block_pool_;
  iree_arena_allocator_t arena_;
};

TEST_F(AmdgpuOccupancyTargetResourcesTest,
       ReportsSelectedGenericTargetTransition) {
  const loom_amdgpu_occupancy_target_resources_t resources =
      Build(IREE_SV("gfx11-generic"), /*wave_size=*/64,
            /*scalar_register_count=*/32, /*vector_register_count=*/160,
            /*flat_workgroup_size=*/128, /*local_memory_bytes=*/0);
  const loom_target_residency_summary_t& summary = resources.residency_summary;

  EXPECT_TRUE(loom_target_residency_summary_is_valid(&summary));
  EXPECT_TRUE(iree_all_bits_set(
      summary.flags,
      LOOM_TARGET_RESIDENCY_SUMMARY_FLAG_HAS_NEXT_BETTER_TIER |
          LOOM_TARGET_RESIDENCY_SUMMARY_FLAG_HAS_UNIQUE_LIMITING_RESOURCE |
          LOOM_TARGET_RESIDENCY_SUMMARY_FLAG_HAS_LIMITING_RESOURCE_NEXT_WORSE_TIER));
  EXPECT_EQ(summary.best_tier, 16u);
  EXPECT_EQ(summary.tier, 3u);
  EXPECT_EQ(summary.next_better_tier, 4u);
  EXPECT_EQ(summary.limiting_resource_count, 1u);
  EXPECT_TRUE(iree_string_view_equal(summary.limiting_resource,
                                     IREE_SV("amdgpu.vgpr")));
  EXPECT_EQ(summary.limiting_resource_units, 160u);
  EXPECT_EQ(summary.limiting_resource_reduction_units_to_next_better_tier, 32u);
  EXPECT_GT(summary.limiting_resource_additional_units_to_next_worse_tier, 0u);
}

TEST_F(AmdgpuOccupancyTargetResourcesTest,
       DistinguishesConservativeGenericFromLargePoolExactProcessor) {
  const loom_target_residency_summary_t generic_summary =
      Build(IREE_SV("gfx11-generic"), /*wave_size=*/64,
            /*scalar_register_count=*/32, /*vector_register_count=*/160,
            /*flat_workgroup_size=*/128, /*local_memory_bytes=*/0)
          .residency_summary;
  const loom_target_residency_summary_t exact_summary =
      Build(IREE_SV("gfx1151"), /*wave_size=*/64,
            /*scalar_register_count=*/32, /*vector_register_count=*/160,
            /*flat_workgroup_size=*/128, /*local_memory_bytes=*/0)
          .residency_summary;

  EXPECT_TRUE(loom_target_residency_summary_is_valid(&generic_summary));
  EXPECT_TRUE(loom_target_residency_summary_is_valid(&exact_summary));
  EXPECT_EQ(generic_summary.tier, 3u);
  EXPECT_EQ(exact_summary.tier, 4u);
  EXPECT_EQ(generic_summary.next_better_tier, 4u);
  EXPECT_EQ(exact_summary.next_better_tier, 5u);
  EXPECT_GT(
      generic_summary.limiting_resource_reduction_units_to_next_better_tier,
      exact_summary.limiting_resource_reduction_units_to_next_better_tier);
  EXPECT_EQ(exact_summary.limiting_resource_reduction_units_to_next_better_tier,
            16u);
}

TEST_F(AmdgpuOccupancyTargetResourcesTest,
       OmitsTransitionWhenAgprCountIsUnavailable) {
  const loom_amdgpu_occupancy_target_resources_t resources =
      Build(IREE_SV("gfx942"), /*wave_size=*/64,
            /*scalar_register_count=*/32, /*vector_register_count=*/160,
            /*flat_workgroup_size=*/64, /*local_memory_bytes=*/0);

  EXPECT_FALSE(
      loom_target_residency_summary_is_valid(&resources.residency_summary));
}

TEST_F(AmdgpuOccupancyTargetResourcesTest,
       OmitsTransitionWithoutExactLaunchInformation) {
  const loom_amdgpu_occupancy_target_resources_t resources =
      Build(IREE_SV("gfx11-generic"), /*wave_size=*/64,
            /*scalar_register_count=*/32, /*vector_register_count=*/160,
            /*flat_workgroup_size=*/0, /*local_memory_bytes=*/0);

  EXPECT_FALSE(
      loom_target_residency_summary_is_valid(&resources.residency_summary));
}

TEST_F(AmdgpuOccupancyTargetResourcesTest,
       OmitsTransitionBlockedByLaunchResources) {
  const loom_amdgpu_occupancy_target_resources_t resources =
      Build(IREE_SV("gfx11-generic"), /*wave_size=*/64,
            /*scalar_register_count=*/32, /*vector_register_count=*/160,
            /*flat_workgroup_size=*/512, /*local_memory_bytes=*/65537);

  EXPECT_FALSE(
      loom_target_residency_summary_is_valid(&resources.residency_summary));
  EXPECT_TRUE(iree_string_view_equal(resources.limiting_resource,
                                     IREE_SV("amdgpu.lds")));
}

}  // namespace
}  // namespace loom
