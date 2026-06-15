// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loomc/target/spirv/vulkan.h"

#include <cstdint>
#include <cstring>
#include <string>

#include "iree/testing/gtest.h"
#include "loomc/result.h"
#include "loomc/target.h"
#include "loomc/target/spirv/base.h"
#include "loomc/target/spirv/profile.h"
#include "test/util.h"

namespace {

using loomc::testing::HandlePtr;

using ResultPtr = HandlePtr<loomc_result_t, loomc_result_release>;
using TargetEnvironmentPtr =
    HandlePtr<loomc_target_environment_t, loomc_target_environment_release>;
using TargetProfilePtr =
    HandlePtr<loomc_target_profile_t, loomc_target_profile_release>;
using TargetSelectionPtr =
    HandlePtr<loomc_target_selection_t, loomc_target_selection_release>;

struct MockVulkanDevice {
  VkPhysicalDevice handle =
      reinterpret_cast<VkPhysicalDevice>(static_cast<uintptr_t>(0x1234));
  uint32_t api_version = VK_API_VERSION_1_3;
  uint32_t max_workgroup_size[3] = {1024, 512, 64};
  uint32_t max_flat_workgroup_size = 1024;
  uint32_t max_workgroup_storage_bytes = 49152;
  uint32_t max_workgroup_count[3] = {65535, 32768, 16384};
  uint32_t subgroup_size = 32;
  VkBool32 shader_float64 = VK_TRUE;
  VkBool32 shader_int16 = VK_TRUE;
  VkBool32 shader_int64 = VK_FALSE;
  VkBool32 shader_float16 = VK_TRUE;
  VkBool32 shader_int8 = VK_TRUE;
  VkBool32 storage_buffer_8bit_access = VK_TRUE;
  VkBool32 storage_buffer_16bit_access = VK_TRUE;
  VkBool32 buffer_device_address = VK_TRUE;
  VkBool32 cooperative_matrix = VK_TRUE;
  bool reports_shader_float16_int8_extension = false;
  bool reports_storage_8bit_extension = false;
  bool reports_storage_16bit_extension = false;
  bool reports_buffer_device_address_extension = false;
  bool reports_cooperative_matrix_extension = true;
};

MockVulkanDevice* g_mock_vulkan_device = nullptr;

void VKAPI_PTR MockGetPhysicalDeviceProperties2(
    VkPhysicalDevice physical_device, VkPhysicalDeviceProperties2* properties) {
  ASSERT_NE(g_mock_vulkan_device, nullptr);
  ASSERT_EQ(physical_device, g_mock_vulkan_device->handle);
  ASSERT_NE(properties, nullptr);
  properties->properties.apiVersion = g_mock_vulkan_device->api_version;
  properties->properties.limits.maxComputeWorkGroupSize[0] =
      g_mock_vulkan_device->max_workgroup_size[0];
  properties->properties.limits.maxComputeWorkGroupSize[1] =
      g_mock_vulkan_device->max_workgroup_size[1];
  properties->properties.limits.maxComputeWorkGroupSize[2] =
      g_mock_vulkan_device->max_workgroup_size[2];
  properties->properties.limits.maxComputeWorkGroupInvocations =
      g_mock_vulkan_device->max_flat_workgroup_size;
  properties->properties.limits.maxComputeSharedMemorySize =
      g_mock_vulkan_device->max_workgroup_storage_bytes;
  properties->properties.limits.maxComputeWorkGroupCount[0] =
      g_mock_vulkan_device->max_workgroup_count[0];
  properties->properties.limits.maxComputeWorkGroupCount[1] =
      g_mock_vulkan_device->max_workgroup_count[1];
  properties->properties.limits.maxComputeWorkGroupCount[2] =
      g_mock_vulkan_device->max_workgroup_count[2];
  for (VkBaseOutStructure* out =
           reinterpret_cast<VkBaseOutStructure*>(properties->pNext);
       out != nullptr; out = out->pNext) {
    switch (out->sType) {
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES:
        reinterpret_cast<VkPhysicalDeviceSubgroupProperties*>(out)
            ->subgroupSize = g_mock_vulkan_device->subgroup_size;
        break;
      default:
        break;
    }
  }
}

void VKAPI_PTR MockGetPhysicalDeviceFeatures2(
    VkPhysicalDevice physical_device, VkPhysicalDeviceFeatures2* features) {
  ASSERT_NE(g_mock_vulkan_device, nullptr);
  ASSERT_EQ(physical_device, g_mock_vulkan_device->handle);
  ASSERT_NE(features, nullptr);
  features->features.shaderFloat64 = g_mock_vulkan_device->shader_float64;
  features->features.shaderInt16 = g_mock_vulkan_device->shader_int16;
  features->features.shaderInt64 = g_mock_vulkan_device->shader_int64;
  for (VkBaseOutStructure* out =
           reinterpret_cast<VkBaseOutStructure*>(features->pNext);
       out != nullptr; out = out->pNext) {
    switch (out->sType) {
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES: {
        auto* vulkan12 =
            reinterpret_cast<VkPhysicalDeviceVulkan12Features*>(out);
        vulkan12->shaderFloat16 = g_mock_vulkan_device->shader_float16;
        vulkan12->shaderInt8 = g_mock_vulkan_device->shader_int8;
        vulkan12->storageBuffer8BitAccess =
            g_mock_vulkan_device->storage_buffer_8bit_access;
        vulkan12->bufferDeviceAddress =
            g_mock_vulkan_device->buffer_device_address;
        break;
      }
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES:
        reinterpret_cast<VkPhysicalDeviceVulkan11Features*>(out)
            ->storageBuffer16BitAccess =
            g_mock_vulkan_device->storage_buffer_16bit_access;
        break;
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES: {
        auto* float16_int8 =
            reinterpret_cast<VkPhysicalDeviceShaderFloat16Int8Features*>(out);
        float16_int8->shaderFloat16 = g_mock_vulkan_device->shader_float16;
        float16_int8->shaderInt8 = g_mock_vulkan_device->shader_int8;
        break;
      }
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES:
        reinterpret_cast<VkPhysicalDevice8BitStorageFeatures*>(out)
            ->storageBuffer8BitAccess =
            g_mock_vulkan_device->storage_buffer_8bit_access;
        break;
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES:
        reinterpret_cast<VkPhysicalDevice16BitStorageFeatures*>(out)
            ->storageBuffer16BitAccess =
            g_mock_vulkan_device->storage_buffer_16bit_access;
        break;
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES:
        reinterpret_cast<VkPhysicalDeviceBufferDeviceAddressFeatures*>(out)
            ->bufferDeviceAddress = g_mock_vulkan_device->buffer_device_address;
        break;
#if defined(VK_KHR_cooperative_matrix)
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR:
        reinterpret_cast<VkPhysicalDeviceCooperativeMatrixFeaturesKHR*>(out)
            ->cooperativeMatrix = g_mock_vulkan_device->cooperative_matrix;
        break;
#endif  // defined(VK_KHR_cooperative_matrix)
      default:
        break;
    }
  }
}

uint32_t MockDeviceExtensionCount(const MockVulkanDevice& device) {
  uint32_t extension_count = 0;
  if (device.reports_shader_float16_int8_extension) ++extension_count;
  if (device.reports_storage_8bit_extension) ++extension_count;
  if (device.reports_storage_16bit_extension) ++extension_count;
  if (device.reports_buffer_device_address_extension) ++extension_count;
#if defined(VK_KHR_cooperative_matrix)
  if (device.reports_cooperative_matrix_extension) ++extension_count;
#endif  // defined(VK_KHR_cooperative_matrix)
  return extension_count;
}

void WriteExtensionProperty(VkExtensionProperties* extension,
                            const char* extension_name,
                            uint32_t specification_version) {
  std::memset(extension, 0, sizeof(*extension));
  std::strncpy(extension->extensionName, extension_name,
               VK_MAX_EXTENSION_NAME_SIZE - 1);
  extension->specVersion = specification_version;
}

void MaybeWriteExtensionProperty(VkExtensionProperties* properties,
                                 uint32_t capacity,
                                 uint32_t* inout_written_count,
                                 const char* extension_name,
                                 uint32_t specification_version) {
  if (*inout_written_count < capacity) {
    WriteExtensionProperty(&properties[*inout_written_count], extension_name,
                           specification_version);
  }
  *inout_written_count += 1;
}

VkResult VKAPI_PTR MockEnumerateDeviceExtensionProperties(
    VkPhysicalDevice physical_device, const char* layer_name,
    uint32_t* property_count, VkExtensionProperties* properties) {
  if (g_mock_vulkan_device == nullptr ||
      physical_device != g_mock_vulkan_device->handle ||
      property_count == nullptr) {
    return VK_ERROR_INITIALIZATION_FAILED;
  }
  EXPECT_EQ(layer_name, nullptr);

  const MockVulkanDevice& device = *g_mock_vulkan_device;
  const uint32_t extension_count = MockDeviceExtensionCount(device);
  if (properties == nullptr) {
    *property_count = extension_count;
    return VK_SUCCESS;
  }

  const uint32_t capacity = *property_count;
  uint32_t written_count = 0;
  if (device.reports_shader_float16_int8_extension) {
    MaybeWriteExtensionProperty(properties, capacity, &written_count,
                                VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME,
                                VK_KHR_SHADER_FLOAT16_INT8_SPEC_VERSION);
  }
  if (device.reports_storage_8bit_extension) {
    MaybeWriteExtensionProperty(properties, capacity, &written_count,
                                VK_KHR_8BIT_STORAGE_EXTENSION_NAME,
                                VK_KHR_8BIT_STORAGE_SPEC_VERSION);
  }
  if (device.reports_storage_16bit_extension) {
    MaybeWriteExtensionProperty(properties, capacity, &written_count,
                                VK_KHR_16BIT_STORAGE_EXTENSION_NAME,
                                VK_KHR_16BIT_STORAGE_SPEC_VERSION);
  }
  if (device.reports_buffer_device_address_extension) {
    MaybeWriteExtensionProperty(properties, capacity, &written_count,
                                VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME,
                                VK_KHR_BUFFER_DEVICE_ADDRESS_SPEC_VERSION);
  }
#if defined(VK_KHR_cooperative_matrix)
  if (device.reports_cooperative_matrix_extension) {
    MaybeWriteExtensionProperty(properties, capacity, &written_count,
                                VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME,
                                VK_KHR_COOPERATIVE_MATRIX_SPEC_VERSION);
  }
#endif  // defined(VK_KHR_cooperative_matrix)

  if (capacity < extension_count) {
    *property_count = capacity;
    return VK_INCOMPLETE;
  }
  *property_count = written_count;
  return VK_SUCCESS;
}

enum class ExtensionEnumeration {
  kEnabled,
  kOmitted,
};

std::string ToString(loomc_string_view_t value) {
  return value.data ? std::string(value.data, value.size) : std::string();
}

void ExpectSucceededResult(const loomc_result_t* result) {
  ASSERT_NE(result, nullptr);
  if (!loomc_result_succeeded(result) &&
      loomc_result_diagnostic_count(result) != 0) {
    const loomc_diagnostic_t* diagnostic =
        loomc_result_diagnostic_at(result, 0);
    ASSERT_NE(diagnostic, nullptr);
    ADD_FAILURE() << ToString(diagnostic->message);
  }
  EXPECT_TRUE(loomc_result_succeeded(result));
}

TargetEnvironmentPtr CreateSpirvTargetEnvironment() {
  loomc_target_environment_t* target_environment = nullptr;
  loomc_status_t status = loomc_target_environment_create_spirv(
      loomc_allocator_system(), &target_environment);
  LOOMC_EXPECT_OK(status);
  return TargetEnvironmentPtr(target_environment);
}

TargetProfilePtr CreateVulkanProfile(
    loomc_target_environment_t* target_environment,
    const MockVulkanDevice& device,
    ExtensionEnumeration extension_enumeration =
        ExtensionEnumeration::kEnabled) {
  g_mock_vulkan_device = const_cast<MockVulkanDevice*>(&device);
  loomc_spirv_vulkan_function_table_t functions = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_SPIRV_VULKAN_FUNCTION_TABLE,
      /*.structure_size=*/sizeof(functions),
      /*.next=*/nullptr,
      /*.get_physical_device_properties2=*/MockGetPhysicalDeviceProperties2,
      /*.get_physical_device_features2=*/MockGetPhysicalDeviceFeatures2,
      /*.enumerate_device_extension_properties=*/
      extension_enumeration == ExtensionEnumeration::kEnabled
          ? MockEnumerateDeviceExtensionProperties
          : nullptr,
  };
  loomc_spirv_vulkan_profile_options_t options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_SPIRV_VULKAN_PROFILE_OPTIONS,
      /*.structure_size=*/sizeof(options),
      /*.next=*/nullptr,
      /*.identifier=*/loomc_make_cstring_view("mock-vulkan-device"),
      /*.physical_device=*/device.handle,
      /*.functions=*/&functions,
  };
  loomc_target_profile_t* profile = nullptr;
  loomc_result_t* result = nullptr;
  loomc_status_t status = loomc_target_profile_create_spirv_vulkan(
      target_environment, &options, loomc_allocator_system(), &profile,
      &result);
  g_mock_vulkan_device = nullptr;
  LOOMC_EXPECT_OK(status);
  ResultPtr result_ptr(result);
  ExpectSucceededResult(result_ptr.get());
  return TargetProfilePtr(profile);
}

void ExpectFeatureState(const loomc_target_profile_t* profile,
                        loomc_spirv_feature_t feature,
                        loomc_target_fact_state_t expected_state) {
  loomc_target_fact_state_t state = LOOMC_TARGET_FACT_STATE_UNKNOWN;
  LOOMC_EXPECT_OK(
      loomc_spirv_target_profile_query_feature(profile, feature, &state));
  EXPECT_EQ(state, expected_state);
}

void ExpectLimitValue(const loomc_target_profile_t* profile,
                      loomc_spirv_limit_t limit, uint64_t expected_value) {
  loomc_spirv_limit_value_t value = {};
  LOOMC_EXPECT_OK(
      loomc_spirv_target_profile_query_limit(profile, limit, &value));
  EXPECT_EQ(value.state, LOOMC_TARGET_FACT_STATE_TRUE);
  EXPECT_EQ(value.value, expected_value);
}

TEST(TargetSpirvVulkanTest, CreatesProfileFromRawVulkanDevice) {
  TargetEnvironmentPtr target_environment = CreateSpirvTargetEnvironment();
  MockVulkanDevice device;
  TargetProfilePtr profile =
      CreateVulkanProfile(target_environment.get(), device);

  ExpectFeatureState(profile.get(), LOOMC_SPIRV_FEATURE_VULKAN_SHADER,
                     LOOMC_TARGET_FACT_STATE_TRUE);
  ExpectFeatureState(profile.get(), LOOMC_SPIRV_FEATURE_PHYSICAL_STORAGE_BUFFER,
                     LOOMC_TARGET_FACT_STATE_TRUE);
  ExpectFeatureState(profile.get(), LOOMC_SPIRV_FEATURE_FLOAT16,
                     LOOMC_TARGET_FACT_STATE_TRUE);
  ExpectFeatureState(profile.get(), LOOMC_SPIRV_FEATURE_INT8,
                     LOOMC_TARGET_FACT_STATE_TRUE);
  ExpectFeatureState(profile.get(), LOOMC_SPIRV_FEATURE_INT64,
                     LOOMC_TARGET_FACT_STATE_FALSE);
  ExpectFeatureState(profile.get(),
                     LOOMC_SPIRV_FEATURE_STORAGE_BUFFER_8BIT_ACCESS,
                     LOOMC_TARGET_FACT_STATE_TRUE);
  ExpectFeatureState(profile.get(),
                     LOOMC_SPIRV_FEATURE_STORAGE_BUFFER_16BIT_ACCESS,
                     LOOMC_TARGET_FACT_STATE_TRUE);
#if defined(VK_KHR_cooperative_matrix)
  ExpectFeatureState(profile.get(), LOOMC_SPIRV_FEATURE_COOPERATIVE_MATRIX_KHR,
                     LOOMC_TARGET_FACT_STATE_TRUE);
#endif  // defined(VK_KHR_cooperative_matrix)

  ExpectLimitValue(profile.get(), LOOMC_SPIRV_LIMIT_MAX_WORKGROUP_SIZE_X,
                   device.max_workgroup_size[0]);
  ExpectLimitValue(profile.get(), LOOMC_SPIRV_LIMIT_MAX_WORKGROUP_SIZE_Y,
                   device.max_workgroup_size[1]);
  ExpectLimitValue(profile.get(), LOOMC_SPIRV_LIMIT_MAX_WORKGROUP_SIZE_Z,
                   device.max_workgroup_size[2]);
  ExpectLimitValue(profile.get(), LOOMC_SPIRV_LIMIT_MAX_FLAT_WORKGROUP_SIZE,
                   device.max_flat_workgroup_size);
  ExpectLimitValue(profile.get(), LOOMC_SPIRV_LIMIT_MAX_WORKGROUP_STORAGE_BYTES,
                   device.max_workgroup_storage_bytes);
  ExpectLimitValue(profile.get(), LOOMC_SPIRV_LIMIT_SUBGROUP_SIZE,
                   device.subgroup_size);

  loomc_spirv_environment_value_t environment = {};
  LOOMC_EXPECT_OK(loomc_spirv_target_profile_query_environment(
      profile.get(), LOOMC_SPIRV_ENVIRONMENT_MAX_SPIRV_VERSION, &environment));
  EXPECT_EQ(environment.state, LOOMC_TARGET_FACT_STATE_TRUE);
  EXPECT_EQ(environment.value, LOOMC_SPIRV_VERSION_1_6);

  loomc_target_selection_t* selection = nullptr;
  loomc_status_t status = loomc_target_selection_create_from_profile(
      profile.get(), loomc_allocator_system(), &selection);
  LOOMC_EXPECT_OK(status);
  TargetSelectionPtr selection_ptr(selection);
  EXPECT_NE(selection_ptr.get(), nullptr);
}

TEST(TargetSpirvVulkanTest, UsesExtensionFeatureStructsBeforeCorePromotion) {
  TargetEnvironmentPtr target_environment = CreateSpirvTargetEnvironment();
  MockVulkanDevice device;
  device.api_version = VK_API_VERSION_1_1;
  device.reports_shader_float16_int8_extension = true;
  device.reports_storage_8bit_extension = true;
  device.reports_buffer_device_address_extension = true;
  device.reports_cooperative_matrix_extension = false;
  TargetProfilePtr profile =
      CreateVulkanProfile(target_environment.get(), device);

  ExpectFeatureState(profile.get(), LOOMC_SPIRV_FEATURE_PHYSICAL_STORAGE_BUFFER,
                     LOOMC_TARGET_FACT_STATE_TRUE);
  ExpectFeatureState(profile.get(), LOOMC_SPIRV_FEATURE_FLOAT16,
                     LOOMC_TARGET_FACT_STATE_TRUE);
  ExpectFeatureState(profile.get(), LOOMC_SPIRV_FEATURE_INT8,
                     LOOMC_TARGET_FACT_STATE_TRUE);
  ExpectFeatureState(profile.get(),
                     LOOMC_SPIRV_FEATURE_STORAGE_BUFFER_8BIT_ACCESS,
                     LOOMC_TARGET_FACT_STATE_TRUE);
  ExpectFeatureState(profile.get(),
                     LOOMC_SPIRV_FEATURE_STORAGE_BUFFER_16BIT_ACCESS,
                     LOOMC_TARGET_FACT_STATE_TRUE);
#if defined(VK_KHR_cooperative_matrix)
  ExpectFeatureState(profile.get(), LOOMC_SPIRV_FEATURE_COOPERATIVE_MATRIX_KHR,
                     LOOMC_TARGET_FACT_STATE_UNKNOWN);
#endif  // defined(VK_KHR_cooperative_matrix)

  loomc_spirv_environment_value_t environment = {};
  LOOMC_EXPECT_OK(loomc_spirv_target_profile_query_environment(
      profile.get(), LOOMC_SPIRV_ENVIRONMENT_MAX_SPIRV_VERSION, &environment));
  EXPECT_EQ(environment.state, LOOMC_TARGET_FACT_STATE_TRUE);
  EXPECT_EQ(environment.value, LOOMC_SPIRV_VERSION_1_3);
}

TEST(TargetSpirvVulkanTest,
     LeavesExtensionOnlyFactsUnknownWhenEnumerationIsOmitted) {
  TargetEnvironmentPtr target_environment = CreateSpirvTargetEnvironment();
  MockVulkanDevice device;
  device.api_version = VK_API_VERSION_1_1;
  device.reports_shader_float16_int8_extension = true;
  device.reports_storage_8bit_extension = true;
  device.reports_buffer_device_address_extension = true;
  TargetProfilePtr profile = CreateVulkanProfile(
      target_environment.get(), device, ExtensionEnumeration::kOmitted);

  ExpectFeatureState(profile.get(), LOOMC_SPIRV_FEATURE_PHYSICAL_STORAGE_BUFFER,
                     LOOMC_TARGET_FACT_STATE_UNKNOWN);
  ExpectFeatureState(profile.get(), LOOMC_SPIRV_FEATURE_FLOAT16,
                     LOOMC_TARGET_FACT_STATE_UNKNOWN);
  ExpectFeatureState(profile.get(), LOOMC_SPIRV_FEATURE_INT8,
                     LOOMC_TARGET_FACT_STATE_UNKNOWN);
  ExpectFeatureState(profile.get(),
                     LOOMC_SPIRV_FEATURE_STORAGE_BUFFER_8BIT_ACCESS,
                     LOOMC_TARGET_FACT_STATE_UNKNOWN);
  ExpectFeatureState(profile.get(),
                     LOOMC_SPIRV_FEATURE_STORAGE_BUFFER_16BIT_ACCESS,
                     LOOMC_TARGET_FACT_STATE_TRUE);
#if defined(VK_KHR_cooperative_matrix)
  ExpectFeatureState(profile.get(), LOOMC_SPIRV_FEATURE_COOPERATIVE_MATRIX_KHR,
                     LOOMC_TARGET_FACT_STATE_UNKNOWN);
#endif  // defined(VK_KHR_cooperative_matrix)
}

TEST(TargetSpirvVulkanTest, RejectsMissingFunctionPointers) {
  TargetEnvironmentPtr target_environment = CreateSpirvTargetEnvironment();
  MockVulkanDevice device;
  loomc_spirv_vulkan_function_table_t functions = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_SPIRV_VULKAN_FUNCTION_TABLE,
      /*.structure_size=*/sizeof(functions),
      /*.next=*/nullptr,
      /*.get_physical_device_properties2=*/nullptr,
      /*.get_physical_device_features2=*/MockGetPhysicalDeviceFeatures2,
      /*.enumerate_device_extension_properties=*/nullptr,
  };
  loomc_spirv_vulkan_profile_options_t options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_SPIRV_VULKAN_PROFILE_OPTIONS,
      /*.structure_size=*/sizeof(options),
      /*.next=*/nullptr,
      /*.identifier=*/loomc_make_cstring_view("bad-vulkan-device"),
      /*.physical_device=*/device.handle,
      /*.functions=*/&functions,
  };
  loomc_target_profile_t* profile = nullptr;
  loomc_result_t* result = nullptr;
  loomc_status_t status = loomc_target_profile_create_spirv_vulkan(
      target_environment.get(), &options, loomc_allocator_system(), &profile,
      &result);
  LOOMC_EXPECT_STATUS_IS(LOOMC_STATUS_INVALID_ARGUMENT, status);
  EXPECT_EQ(profile, nullptr);
  EXPECT_EQ(result, nullptr);
}

}  // namespace
