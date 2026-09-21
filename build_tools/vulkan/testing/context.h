// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_BUILD_TOOLS_VULKAN_TESTING_CONTEXT_H_
#define IREE_BUILD_TOOLS_VULKAN_TESTING_CONTEXT_H_

#ifndef VK_NO_PROTOTYPES
#define VK_NO_PROTOTYPES
#endif
#include "gtest/gtest.h"
#include "vulkan/vulkan.h"

namespace iree::vulkan::testing {

// Reports a native operation and its unmodified VkResult on failure.
::testing::AssertionResult CheckResult(VkResult result, const char* operation);

// Owns the platform loader and an explicitly created native instance. Children
// borrow this owner and must be destroyed first. No HAL or process cache
// retains the loader after this object is destroyed.
class Instance {
 public:
  Instance() = default;
  ~Instance();
  Instance(const Instance&) = delete;
  Instance& operator=(const Instance&) = delete;

  // Loads the system loader. A failure with loader_available() == false permits
  // a resource-dependent test to skip; a malformed loaded library is an error.
  ::testing::AssertionResult Load();
  bool loader_available() const { return library_ != nullptr; }

  // Preserves VkResult so callers can distinguish an unavailable implementation
  // (VK_ERROR_INCOMPATIBLE_DRIVER) from initialization failures.
  VkResult Create(const VkInstanceCreateInfo& info);
  VkInstance handle() const { return instance_; }
  PFN_vkGetInstanceProcAddr get_instance_proc_addr() const { return get_proc_; }

  // Vulkan guarantees core entry points for the requested API version.
  // Callers query/enable extensions before resolving their optional functions.
  template <typename T>
  T Get(const char* name) const {
    return reinterpret_cast<T>(get_proc_(instance_, name));
  }

 private:
  // Owned platform library handle; outlives the native instance and children.
  void* library_ = nullptr;
  // Root resolver belonging to library_.
  PFN_vkGetInstanceProcAddr get_proc_ = nullptr;
  // Owned instance created from the caller's exact create info.
  VkInstance instance_ = VK_NULL_HANDLE;
};

// Owns a native device with precisely the caller's enabled features, extensions
// and queues. The instance and every native child have explicit outer/inner
// lifetimes; this object performs no implicit device-wide completion wait.
class Device {
 public:
  Device() = default;
  ~Device();
  Device(const Device&) = delete;
  Device& operator=(const Device&) = delete;

  ::testing::AssertionResult Initialize(const Instance& instance,
                                        VkPhysicalDevice physical_device,
                                        const VkDeviceCreateInfo& info);
  VkDevice handle() const { return device_; }
  const VkPhysicalDeviceMemoryProperties& memory_properties() const {
    return memory_properties_;
  }

  template <typename T>
  T Get(const char* name) const {
    return reinterpret_cast<T>(get_proc_(device_, name));
  }

 private:
  // Instance-scoped resolver; the caller keeps the parent instance alive.
  PFN_vkGetDeviceProcAddr get_proc_ = nullptr;
  // Owned device; all submitted work and native children finish first.
  VkDevice device_ = VK_NULL_HANDLE;
  // Native memory types used for explicit allocation admission.
  VkPhysicalDeviceMemoryProperties memory_properties_ = {};
};

}  // namespace iree::vulkan::testing

#endif  // IREE_BUILD_TOOLS_VULKAN_TESTING_CONTEXT_H_
