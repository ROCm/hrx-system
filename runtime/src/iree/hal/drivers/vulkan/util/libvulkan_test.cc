// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/vulkan/util/libvulkan.h"

#include <string_view>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

#if !IREE_HAL_VULKAN_LIBVULKAN_STATIC
namespace iree::hal::vulkan {
namespace {

// Models a device with core entry points and explicitly enabled extensions.
// Resolved functions are lookup sentinels and are never called.
struct DeviceResolver {
  // Optional extension entry point exposed by this device, if any.
  std::string_view extension_symbol;
  // Core entry point withheld to exercise an incomplete loader, if any.
  std::string_view missing_symbol;
};

void VKAPI_CALL ResolvedFunction() {}

PFN_vkVoidFunction VKAPI_CALL ResolveDeviceSymbol(VkDevice device,
                                                  const char* symbol) {
  const auto& resolver = *reinterpret_cast<const DeviceResolver*>(device);
  const std::string_view name(symbol);
  if (name == resolver.missing_symbol) {
    return nullptr;
  }
  const auto suffix = name.substr(name.size() - 3);
  if ((suffix == "EXT" || suffix == "KHR") &&
      name != resolver.extension_symbol) {
    return nullptr;
  }
  return ResolvedFunction;
}

TEST(LibVulkanTest, LoadsCoreDeviceWithoutOptionalExtensions) {
  DeviceResolver resolver;
  iree_hal_vulkan_instance_syms_t instance_syms = {};
  instance_syms.vkGetDeviceProcAddr = ResolveDeviceSymbol;
  iree_hal_vulkan_device_syms_t device_syms = {};
  IREE_ASSERT_OK(iree_hal_vulkan_libvulkan_load_device_syms(
      &instance_syms, reinterpret_cast<VkDevice>(&resolver), &device_syms));
  EXPECT_NE(device_syms.vkGetBufferDeviceAddress, nullptr);
  EXPECT_NE(device_syms.vkQueueSubmit2, nullptr);
  EXPECT_EQ(device_syms.vkGetMemoryHostPointerPropertiesEXT, nullptr);
  EXPECT_EQ(device_syms.vkCmdPushDescriptorSetKHR, nullptr);
}

TEST(LibVulkanTest, LoadsEnabledHostMemoryExtension) {
  DeviceResolver resolver;
  resolver.extension_symbol = "vkGetMemoryHostPointerPropertiesEXT";
  iree_hal_vulkan_instance_syms_t instance_syms = {};
  instance_syms.vkGetDeviceProcAddr = ResolveDeviceSymbol;
  iree_hal_vulkan_device_syms_t device_syms = {};
  IREE_ASSERT_OK(iree_hal_vulkan_libvulkan_load_device_syms(
      &instance_syms, reinterpret_cast<VkDevice>(&resolver), &device_syms));
  EXPECT_EQ(reinterpret_cast<PFN_vkVoidFunction>(
                device_syms.vkGetMemoryHostPointerPropertiesEXT),
            ResolvedFunction);
}

TEST(LibVulkanTest, RejectsMissingRequiredCoreSymbol) {
  DeviceResolver resolver;
  resolver.extension_symbol = "vkGetMemoryHostPointerPropertiesEXT";
  resolver.missing_symbol = "vkGetBufferDeviceAddress";
  iree_hal_vulkan_instance_syms_t instance_syms = {};
  instance_syms.vkGetDeviceProcAddr = ResolveDeviceSymbol;
  iree_hal_vulkan_device_syms_t device_syms = {};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_NOT_FOUND,
      iree_hal_vulkan_libvulkan_load_device_syms(
          &instance_syms, reinterpret_cast<VkDevice>(&resolver), &device_syms));
  EXPECT_EQ(device_syms.vkGetBufferDeviceAddress, nullptr);
  // Partial initialization preserves the device cleanup entry point.
  EXPECT_NE(device_syms.vkDestroyDevice, nullptr);
}

}  // namespace
}  // namespace iree::hal::vulkan
#endif  // !IREE_HAL_VULKAN_LIBVULKAN_STATIC
