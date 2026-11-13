// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/driver/vulkan/physical_device.h"

#include <cstring>

#include "iree/testing/gtest.h"

namespace {

#if !IREE_HAL_VULKAN_LIBVULKAN_STATIC

struct ExecutionUnitQuery {
  // Property structure expected from the advertised extension.
  VkStructureType structure_type;
  // Active execution-unit count supplied by the driver.
  uint32_t unit_count;
  // Number of native property queries performed.
  uint32_t query_count = 0;
};

static VKAPI_ATTR void VKAPI_CALL GetExecutionUnitProperties(
    VkPhysicalDevice handle, VkPhysicalDeviceProperties2* properties) {
  auto* query = reinterpret_cast<ExecutionUnitQuery*>(handle);
  ++query->query_count;
  EXPECT_EQ(properties->sType, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2);
  auto* property = static_cast<VkBaseOutStructure*>(properties->pNext);
  ASSERT_NE(property, nullptr);
  EXPECT_EQ(property->sType, query->structure_type);
  EXPECT_EQ(property->pNext, nullptr);
  switch (query->structure_type) {
    case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_CORE_PROPERTIES_2_AMD:
      static_cast<VkPhysicalDeviceShaderCoreProperties2AMD*>(properties->pNext)
          ->activeComputeUnitCount = query->unit_count;
      break;
    case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_SM_BUILTINS_PROPERTIES_NV:
      static_cast<VkPhysicalDeviceShaderSMBuiltinsPropertiesNV*>(
          properties->pNext)
          ->shaderSMCount = query->unit_count;
      break;
    case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_CORE_BUILTINS_PROPERTIES_ARM:
      static_cast<VkPhysicalDeviceShaderCoreBuiltinsPropertiesARM*>(
          properties->pNext)
          ->shaderCoreCount = query->unit_count;
      break;
    default:
      ADD_FAILURE() << "Unexpected native execution-unit query";
  }
}

TEST(PhysicalDeviceTest, QueriesReportedExecutionUnits) {
  const struct {
    // Physical-device extension advertised by the driver.
    const char* extension_name;
    // Native output property and recorded query count.
    ExecutionUnitQuery query;
  } cases[] = {
      {VK_AMD_SHADER_CORE_PROPERTIES_2_EXTENSION_NAME,
       {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_CORE_PROPERTIES_2_AMD, 37}},
      {VK_NV_SHADER_SM_BUILTINS_EXTENSION_NAME,
       {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_SM_BUILTINS_PROPERTIES_NV,
        127}},
      {VK_ARM_SHADER_CORE_BUILTINS_EXTENSION_NAME,
       {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_CORE_BUILTINS_PROPERTIES_ARM,
        13}},
      {VK_NV_SHADER_SM_BUILTINS_EXTENSION_NAME,
       {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_SM_BUILTINS_PROPERTIES_NV, 0}},
      {VK_AMD_SHADER_CORE_PROPERTIES_EXTENSION_NAME,
       {VK_STRUCTURE_TYPE_MAX_ENUM, 0}},
      {"VK_unrecognized_extension", {VK_STRUCTURE_TYPE_MAX_ENUM, 0}},
  };
  iree_hal_vulkan_instance_t instance = {};
  instance.syms.vkGetPhysicalDeviceProperties2 = GetExecutionUnitProperties;
  EXPECT_EQ(iree_hal_vulkan_physical_device_query_execution_unit_count(
                &instance, VK_NULL_HANDLE, 0, nullptr),
            0u);
  for (const auto& test_case : cases) {
    SCOPED_TRACE(test_case.extension_name);
    VkExtensionProperties extension = {};
    std::strcpy(extension.extensionName, test_case.extension_name);
    ExecutionUnitQuery query = test_case.query;
    EXPECT_EQ(iree_hal_vulkan_physical_device_query_execution_unit_count(
                  &instance, reinterpret_cast<VkPhysicalDevice>(&query), 1,
                  &extension),
              query.unit_count);
    EXPECT_EQ(query.query_count,
              query.structure_type == VK_STRUCTURE_TYPE_MAX_ENUM ? 0u : 1u);
  }
}

#endif  // !IREE_HAL_VULKAN_LIBVULKAN_STATIC

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
