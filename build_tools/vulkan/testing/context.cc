// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "build_tools/vulkan/testing/context.h"

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace iree::vulkan::testing {

::testing::AssertionResult CheckResult(VkResult result, const char* operation) {
  if (result == VK_SUCCESS) {
    return ::testing::AssertionSuccess();
  }
  return ::testing::AssertionFailure()
         << operation << " returned VkResult " << static_cast<int>(result);
}

Instance::~Instance() {
  if (instance_) {
    Get<PFN_vkDestroyInstance>("vkDestroyInstance")(instance_, nullptr);
  }
  if (library_) {
#if defined(_WIN32)
    EXPECT_NE(FreeLibrary(static_cast<HMODULE>(library_)), 0);
#else
    EXPECT_EQ(dlclose(library_), 0);
#endif
  }
}

::testing::AssertionResult Instance::Load() {
#if defined(_WIN32)
  library_ =
      LoadLibraryExW(L"vulkan-1.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
  if (!library_) {
    return ::testing::AssertionFailure()
           << "LoadLibraryExW(vulkan-1.dll) failed: " << GetLastError();
  }
  get_proc_ = reinterpret_cast<PFN_vkGetInstanceProcAddr>(
      GetProcAddress(static_cast<HMODULE>(library_), "vkGetInstanceProcAddr"));
#else
#if defined(__APPLE__)
  constexpr const char* kLibraryName = "libvulkan.1.dylib";
#else
  constexpr const char* kLibraryName = "libvulkan.so.1";
#endif
  library_ = dlopen(kLibraryName, RTLD_NOW | RTLD_LOCAL);
  if (!library_) {
    return ::testing::AssertionFailure()
           << "dlopen(" << kLibraryName << ") failed: " << dlerror();
  }
  get_proc_ = reinterpret_cast<PFN_vkGetInstanceProcAddr>(
      dlsym(library_, "vkGetInstanceProcAddr"));
#endif
  if (!get_proc_ || !Get<PFN_vkCreateInstance>("vkCreateInstance")) {
    return ::testing::AssertionFailure()
           << "Vulkan loader is missing its required global entry points";
  }
  return ::testing::AssertionSuccess();
}

VkResult Instance::Create(const VkInstanceCreateInfo& info) {
  return Get<PFN_vkCreateInstance>("vkCreateInstance")(&info, nullptr,
                                                       &instance_);
}

Device::~Device() {
  if (device_) {
    Get<PFN_vkDestroyDevice>("vkDestroyDevice")(device_, nullptr);
  }
}

::testing::AssertionResult Device::Initialize(const Instance& instance,
                                              VkPhysicalDevice physical_device,
                                              const VkDeviceCreateInfo& info) {
  get_proc_ = instance.Get<PFN_vkGetDeviceProcAddr>("vkGetDeviceProcAddr");
  instance.Get<PFN_vkGetPhysicalDeviceMemoryProperties>(
      "vkGetPhysicalDeviceMemoryProperties")(physical_device,
                                             &memory_properties_);
  return CheckResult(instance.Get<PFN_vkCreateDevice>("vkCreateDevice")(
                         physical_device, &info, nullptr, &device_),
                     "vkCreateDevice");
}

}  // namespace iree::vulkan::testing
