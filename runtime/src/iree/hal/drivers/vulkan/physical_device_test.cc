// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/vulkan/physical_device.h"

#include "iree/testing/gtest.h"

namespace {

TEST(PhysicalDeviceTest, UnifiedMemoryRequiresIntegratedOrCpuDevice) {
  EXPECT_TRUE(iree_hal_vulkan_physical_device_type_uses_unified_memory(
      VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU));
  EXPECT_TRUE(iree_hal_vulkan_physical_device_type_uses_unified_memory(
      VK_PHYSICAL_DEVICE_TYPE_CPU));
  EXPECT_FALSE(iree_hal_vulkan_physical_device_type_uses_unified_memory(
      VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU));
  EXPECT_FALSE(iree_hal_vulkan_physical_device_type_uses_unified_memory(
      VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU));
  EXPECT_FALSE(iree_hal_vulkan_physical_device_type_uses_unified_memory(
      VK_PHYSICAL_DEVICE_TYPE_OTHER));
}

TEST(PhysicalDeviceTest, HostMemoryLocalityIsIndependentFromCoherence) {
  const iree_hal_memory_type_t memory_type =
      iree_hal_vulkan_memory_type_from_properties(
          VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU,
          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
  EXPECT_EQ(memory_type, IREE_HAL_MEMORY_TYPE_HOST_LOCAL |
                             IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE);
  EXPECT_FALSE(
      iree_all_bits_set(memory_type, IREE_HAL_MEMORY_TYPE_HOST_COHERENT));
}

TEST(PhysicalDeviceTest, IntegratedMemoryCanBeLocalToHostAndDevice) {
  const iree_hal_memory_type_t memory_type =
      iree_hal_vulkan_memory_type_from_properties(
          VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU,
          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
              VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
  EXPECT_EQ(memory_type, IREE_HAL_MEMORY_TYPE_HOST_LOCAL |
                             IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL);
  EXPECT_FALSE(
      iree_all_bits_set(memory_type, IREE_HAL_MEMORY_TYPE_HOST_COHERENT));
}

TEST(PhysicalDeviceTest, CoherentIntegratedMemoryReportsAllProperties) {
  const iree_hal_memory_type_t memory_type =
      iree_hal_vulkan_memory_type_from_properties(
          VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU,
          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
              VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
              VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
              VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
  EXPECT_EQ(memory_type, IREE_HAL_MEMORY_TYPE_HOST_LOCAL |
                             IREE_HAL_MEMORY_TYPE_HOST_COHERENT |
                             IREE_HAL_MEMORY_TYPE_HOST_CACHED |
                             IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL);
}

TEST(PhysicalDeviceTest, DiscreteBarMemoryIsNotHostLocal) {
  const iree_hal_memory_type_t memory_type =
      iree_hal_vulkan_memory_type_from_properties(
          VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU,
          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
              VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
              VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  EXPECT_EQ(memory_type, IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL |
                             IREE_HAL_MEMORY_TYPE_HOST_VISIBLE |
                             IREE_HAL_MEMORY_TYPE_HOST_COHERENT);
  EXPECT_FALSE(iree_all_bits_set(memory_type, IREE_HAL_MEMORY_TYPE_HOST_LOCAL));
}

TEST(PhysicalDeviceTest, DeviceOnlyUnifiedMemoryIsNotHostLocal) {
  const iree_hal_memory_type_t memory_type =
      iree_hal_vulkan_memory_type_from_properties(
          VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU,
          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  EXPECT_EQ(memory_type, IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL);
  EXPECT_FALSE(iree_all_bits_set(memory_type, IREE_HAL_MEMORY_TYPE_HOST_LOCAL));
}

}  // namespace
