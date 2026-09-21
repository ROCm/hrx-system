// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <string>
#include <utility>
#include <vector>

#include "build_tools/vulkan/testing/context.h"
#include "iree/testing/gtest.h"
#include "loomc/result.h"
#include "loomc/target.h"
#include "loomc/target/spirv/base.h"
#include "loomc/target/spirv/profile.h"
#include "loomc/target/spirv/vulkan.h"
#include "test/util.h"

namespace {

using loomc::testing::HandlePtr;

TEST(TargetSpirvVulkanDeviceTest, ProfileMatchesLivePhysicalDevice) {
  // The profile API borrows only a physical device and three query functions.
  // No logical device, shader compiler or HAL service participates.
  iree::vulkan::testing::Instance instance;
  const auto loaded = instance.Load();
  if (!loaded && !instance.loader_available()) {
    GTEST_SKIP() << loaded.message();
  }
  ASSERT_TRUE(loaded);
  VkApplicationInfo application = {VK_STRUCTURE_TYPE_APPLICATION_INFO};
  application.apiVersion = VK_API_VERSION_1_2;
  VkInstanceCreateInfo create = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
  create.pApplicationInfo = &application;
  const VkResult created = instance.Create(create);
  if (created == VK_ERROR_INCOMPATIBLE_DRIVER) {
    GTEST_SKIP() << "No Vulkan 1.2 implementation";
  }
  ASSERT_EQ(created, VK_SUCCESS);
  const auto enumerate = instance.Get<PFN_vkEnumeratePhysicalDevices>(
      "vkEnumeratePhysicalDevices");
  uint32_t device_count = 0;
  ASSERT_EQ(enumerate(instance.handle(), &device_count, nullptr), VK_SUCCESS);
  std::vector<VkPhysicalDevice> devices(device_count);
  ASSERT_EQ(enumerate(instance.handle(), &device_count, devices.data()),
            VK_SUCCESS);

  loomc_spirv_vulkan_function_table_t functions = {};
  functions.type = LOOMC_STRUCTURE_TYPE_SPIRV_VULKAN_FUNCTION_TABLE;
  functions.structure_size = sizeof(functions);
  functions.get_physical_device_properties2 =
      instance.Get<PFN_vkGetPhysicalDeviceProperties2>(
          "vkGetPhysicalDeviceProperties2");
  functions.get_physical_device_features2 =
      instance.Get<PFN_vkGetPhysicalDeviceFeatures2>(
          "vkGetPhysicalDeviceFeatures2");
  functions.enumerate_device_extension_properties =
      instance.Get<PFN_vkEnumerateDeviceExtensionProperties>(
          "vkEnumerateDeviceExtensionProperties");
  VkPhysicalDevice physical_device = VK_NULL_HANDLE;
  VkPhysicalDeviceSubgroupProperties subgroup = {
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
  VkPhysicalDeviceProperties2 properties = {
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
  properties.pNext = &subgroup;
  for (VkPhysicalDevice device : devices) {
    functions.get_physical_device_properties2(device, &properties);
    if (properties.properties.apiVersion >= VK_API_VERSION_1_2 &&
        properties.properties.deviceType != VK_PHYSICAL_DEVICE_TYPE_CPU) {
      physical_device = device;
      break;
    }
  }
  if (!physical_device) {
    GTEST_SKIP() << "No Vulkan 1.2 GPU";
  }
  SCOPED_TRACE(properties.properties.deviceName);
  VkPhysicalDeviceVulkan12Features features12 = {
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
  VkPhysicalDeviceVulkan11Features features11 = {
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
  features11.pNext = &features12;
  VkPhysicalDeviceFeatures2 features = {
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
  features.pNext = &features11;
  functions.get_physical_device_features2(physical_device, &features);

  loomc_target_environment_t* environment = nullptr;
  LOOMC_ASSERT_OK(loomc_target_environment_create_spirv(
      loomc_allocator_system(), &environment));
  HandlePtr<loomc_target_environment_t, loomc_target_environment_release>
      environment_owner(environment);
  loomc_spirv_vulkan_profile_options_t options = {};
  options.type = LOOMC_STRUCTURE_TYPE_SPIRV_VULKAN_PROFILE_OPTIONS;
  options.structure_size = sizeof(options);
  options.identifier = loomc_make_cstring_view("live-vulkan-device");
  options.physical_device = physical_device;
  options.functions = &functions;
  loomc_target_profile_t* profile = nullptr;
  loomc_result_t* result = nullptr;
  LOOMC_ASSERT_OK(loomc_target_profile_create_spirv_vulkan(
      environment, &options, loomc_allocator_system(), &profile, &result));
  HandlePtr<loomc_result_t, loomc_result_release> result_owner(result);
  HandlePtr<loomc_target_profile_t, loomc_target_profile_release> profile_owner(
      profile);
  ASSERT_NE(result, nullptr);
  if (!loomc_result_succeeded(result)) {
    for (loomc_host_size_t i = 0; i < loomc_result_diagnostic_count(result);
         ++i) {
      const auto* diagnostic = loomc_result_diagnostic_at(result, i);
      ADD_FAILURE() << std::string(diagnostic->message.data,
                                   diagnostic->message.size);
    }
  }
  ASSERT_TRUE(loomc_result_succeeded(result));
  ASSERT_NE(profile, nullptr);

  const std::pair<loomc_spirv_feature_t, VkBool32> feature_expectations[] = {
      {LOOMC_SPIRV_FEATURE_FLOAT64, features.features.shaderFloat64},
      {LOOMC_SPIRV_FEATURE_INT16, features.features.shaderInt16},
      {LOOMC_SPIRV_FEATURE_INT64, features.features.shaderInt64},
      {LOOMC_SPIRV_FEATURE_FLOAT16, features12.shaderFloat16},
      {LOOMC_SPIRV_FEATURE_INT8, features12.shaderInt8},
      {LOOMC_SPIRV_FEATURE_PHYSICAL_STORAGE_BUFFER,
       features12.bufferDeviceAddress},
      {LOOMC_SPIRV_FEATURE_STORAGE_BUFFER_8BIT_ACCESS,
       features12.storageBuffer8BitAccess},
      {LOOMC_SPIRV_FEATURE_STORAGE_BUFFER_16BIT_ACCESS,
       features11.storageBuffer16BitAccess},
  };
  for (const auto& [feature, supported] : feature_expectations) {
    loomc_target_fact_state_t state = LOOMC_TARGET_FACT_STATE_UNKNOWN;
    LOOMC_ASSERT_OK(
        loomc_spirv_target_profile_query_feature(profile, feature, &state));
    EXPECT_EQ(state, supported ? LOOMC_TARGET_FACT_STATE_TRUE
                               : LOOMC_TARGET_FACT_STATE_FALSE)
        << "feature " << feature;
  }
  const auto& limits = properties.properties.limits;
  const std::pair<loomc_spirv_limit_t, uint64_t> limit_expectations[] = {
      {LOOMC_SPIRV_LIMIT_MAX_WORKGROUP_SIZE_X,
       limits.maxComputeWorkGroupSize[0]},
      {LOOMC_SPIRV_LIMIT_MAX_WORKGROUP_SIZE_Y,
       limits.maxComputeWorkGroupSize[1]},
      {LOOMC_SPIRV_LIMIT_MAX_WORKGROUP_SIZE_Z,
       limits.maxComputeWorkGroupSize[2]},
      {LOOMC_SPIRV_LIMIT_MAX_FLAT_WORKGROUP_SIZE,
       limits.maxComputeWorkGroupInvocations},
      {LOOMC_SPIRV_LIMIT_MAX_WORKGROUP_STORAGE_BYTES,
       limits.maxComputeSharedMemorySize},
      {LOOMC_SPIRV_LIMIT_SUBGROUP_SIZE, subgroup.subgroupSize},
  };
  for (const auto& [limit, expected] : limit_expectations) {
    loomc_spirv_limit_value_t value = {};
    LOOMC_ASSERT_OK(
        loomc_spirv_target_profile_query_limit(profile, limit, &value));
    EXPECT_EQ(value.state, LOOMC_TARGET_FACT_STATE_TRUE) << "limit " << limit;
    EXPECT_EQ(value.value, expected) << "limit " << limit;
  }
  // Local owners release the result/profile/environment before native teardown.
}

}  // namespace
