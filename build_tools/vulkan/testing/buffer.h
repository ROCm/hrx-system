// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_BUILD_TOOLS_VULKAN_TESTING_BUFFER_H_
#define IREE_BUILD_TOOLS_VULKAN_TESTING_BUFFER_H_

#include "build_tools/vulkan/testing/context.h"

namespace iree::vulkan::testing {

// Owns an ordinary buffer/allocation and an optional full-allocation mapping.
// Callers complete all accesses before destruction. External-memory payload
// import/ownership and GPU barriers belong to the component's interop test.
class Buffer {
 public:
  Buffer() = default;
  ~Buffer();
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;

  ::testing::AssertionResult Initialize(
      const Device& device, const VkBufferCreateInfo& info,
      VkMemoryPropertyFlags required_properties,
      VkMemoryAllocateFlags allocation_flags);
  ::testing::AssertionResult Map();
  // Flush/invalidate the complete mapping. These are no-ops on coherent memory;
  // explicit GPU/host completion is still the caller's responsibility.
  ::testing::AssertionResult Flush();
  ::testing::AssertionResult Invalidate();

  VkBuffer handle() const { return buffer_; }
  void* mapped() const { return mapped_; }
  VkDeviceAddress device_address() const;

 private:
  // Borrowed parent, alive through native allocation teardown.
  const Device* device_ = nullptr;
  // Owned buffer, destroyed before its backing allocation.
  VkBuffer buffer_ = VK_NULL_HANDLE;
  // Owned allocation bound at offset zero.
  VkDeviceMemory memory_ = VK_NULL_HANDLE;
  // Actual selected native memory flags, including host coherency.
  VkMemoryPropertyFlags properties_ = 0;
  // Full-allocation mapping, or null until Map succeeds.
  void* mapped_ = nullptr;
};

}  // namespace iree::vulkan::testing

#endif  // IREE_BUILD_TOOLS_VULKAN_TESTING_BUFFER_H_
