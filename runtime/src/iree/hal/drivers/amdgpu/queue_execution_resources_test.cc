// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/queue_execution_resources.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::amdgpu {
namespace {

TEST(QueueExecutionResourcesTest, MapsGfx942InterleavedPartitions) {
  iree_hal_amdgpu_queue_execution_resource_topology_t topology;
  IREE_ASSERT_OK(iree_hal_amdgpu_queue_execution_resource_topology_initialize(
      {.major = 9, .minor = 4, .stepping = 2},
      /*execution_unit_count=*/40, /*partition_count=*/8, &topology));

  ASSERT_EQ(iree_hal_amdgpu_queue_execution_resource_group_count(&topology),
            8u);
  ASSERT_EQ(iree_hal_amdgpu_queue_execution_resource_count(&topology), 40u);
  iree_hal_queue_execution_resource_group_spec_t groups[8];
  iree_hal_amdgpu_queue_execution_resource_populate_groups(&topology, groups);
  for (const auto& group : groups) {
    EXPECT_EQ(group.minimum_selected_resource_count, 1u);
  }

  iree_hal_queue_execution_resource_spec_t resources[40];
  iree_hal_amdgpu_queue_execution_resource_populate_resources(&topology,
                                                              resources);
  for (uint32_t i = 0; i < IREE_ARRAYSIZE(resources); ++i) {
    EXPECT_EQ(resources[i].group_ordinal, i % 8u);
    EXPECT_EQ(resources[i].first_execution_unit_ordinal, i);
    EXPECT_EQ(resources[i].execution_unit_count, 1u);
  }

  const iree_hal_queue_execution_resource_ordinal_t selected_resources[] = {
      0, 1, 2, 3, 4, 5, 6, 7};
  uint32_t mask[2] = {0};
  IREE_ASSERT_OK(iree_hal_amdgpu_queue_execution_resource_write_mask(
      &topology,
      {.count = IREE_ARRAYSIZE(selected_resources),
       .ordinals = selected_resources},
      /*out_mask_bit_count=*/64, mask));
  EXPECT_EQ(mask[0], UINT32_C(0x000000FF));
  EXPECT_EQ(mask[1], 0u);
}

TEST(QueueExecutionResourcesTest, MapsGfx11WgpResources) {
  iree_hal_amdgpu_queue_execution_resource_topology_t topology;
  IREE_ASSERT_OK(iree_hal_amdgpu_queue_execution_resource_topology_initialize(
      {.major = 11, .minor = 0, .stepping = 0},
      /*execution_unit_count=*/40, /*partition_count=*/1, &topology));

  ASSERT_EQ(iree_hal_amdgpu_queue_execution_resource_count(&topology), 20u);
  iree_hal_queue_execution_resource_spec_t resources[20];
  iree_hal_amdgpu_queue_execution_resource_populate_resources(&topology,
                                                              resources);
  for (uint32_t i = 0; i < IREE_ARRAYSIZE(resources); ++i) {
    EXPECT_EQ(resources[i].group_ordinal, 0u);
    EXPECT_EQ(resources[i].first_execution_unit_ordinal, i * 2u);
    EXPECT_EQ(resources[i].execution_unit_count, 2u);
  }

  const iree_hal_queue_execution_resource_ordinal_t selected_resources[] = {0,
                                                                            19};
  uint32_t selected_mask[2] = {0};
  IREE_ASSERT_OK(iree_hal_amdgpu_queue_execution_resource_write_mask(
      &topology,
      {.count = IREE_ARRAYSIZE(selected_resources),
       .ordinals = selected_resources},
      /*out_mask_bit_count=*/64, selected_mask));
  EXPECT_EQ(selected_mask[0], UINT32_C(0x00000003));
  EXPECT_EQ(selected_mask[1], UINT32_C(0x000000C0));

  uint32_t full_mask[2] = {0};
  IREE_ASSERT_OK(iree_hal_amdgpu_queue_execution_resource_write_mask(
      &topology, (iree_hal_queue_execution_resource_list_t){0},
      /*out_mask_bit_count=*/64, full_mask));
  EXPECT_EQ(full_mask[0], UINT32_MAX);
  EXPECT_EQ(full_mask[1], UINT32_C(0x000000FF));
}

}  // namespace
}  // namespace iree::hal::amdgpu
