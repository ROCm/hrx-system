// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "build_tools/vulkan/testing/buffer.h"

namespace iree::vulkan::testing {

Buffer::~Buffer() {
  if (mapped_) {
    device_->Get<PFN_vkUnmapMemory>("vkUnmapMemory")(device_->handle(),
                                                     memory_);
  }
  if (buffer_) {
    device_->Get<PFN_vkDestroyBuffer>("vkDestroyBuffer")(device_->handle(),
                                                         buffer_, nullptr);
  }
  if (memory_) {
    device_->Get<PFN_vkFreeMemory>("vkFreeMemory")(device_->handle(), memory_,
                                                   nullptr);
  }
}

::testing::AssertionResult Buffer::Initialize(
    const Device& device, const VkBufferCreateInfo& info,
    VkMemoryPropertyFlags required_properties,
    VkMemoryAllocateFlags allocation_flags) {
  device_ = &device;
  auto result = CheckResult(device.Get<PFN_vkCreateBuffer>("vkCreateBuffer")(
                                device.handle(), &info, nullptr, &buffer_),
                            "vkCreateBuffer");
  if (!result) {
    return result;
  }
  VkMemoryRequirements requirements;
  device.Get<PFN_vkGetBufferMemoryRequirements>(
      "vkGetBufferMemoryRequirements")(device.handle(), buffer_, &requirements);
  const auto& properties = device.memory_properties();
  uint32_t memory_type = UINT32_MAX;
  for (uint32_t i = 0; i < properties.memoryTypeCount; ++i) {
    if ((requirements.memoryTypeBits & (1u << i)) &&
        (properties.memoryTypes[i].propertyFlags & required_properties) ==
            required_properties) {
      memory_type = i;
      properties_ = properties.memoryTypes[i].propertyFlags;
      break;
    }
  }
  if (memory_type == UINT32_MAX) {
    return ::testing::AssertionFailure()
           << "Buffer has no memory type satisfying flags "
           << required_properties;
  }
  VkMemoryAllocateFlagsInfo flags = {
      VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
  flags.flags = allocation_flags;
  VkMemoryAllocateInfo allocation = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  allocation.pNext = &flags;
  allocation.allocationSize = requirements.size;
  allocation.memoryTypeIndex = memory_type;
  result = CheckResult(device.Get<PFN_vkAllocateMemory>("vkAllocateMemory")(
                           device.handle(), &allocation, nullptr, &memory_),
                       "vkAllocateMemory");
  if (!result) {
    return result;
  }
  return CheckResult(device.Get<PFN_vkBindBufferMemory>("vkBindBufferMemory")(
                         device.handle(), buffer_, memory_, 0),
                     "vkBindBufferMemory");
}

::testing::AssertionResult Buffer::Map() {
  return CheckResult(
      device_->Get<PFN_vkMapMemory>("vkMapMemory")(
          device_->handle(), memory_, 0, VK_WHOLE_SIZE, 0, &mapped_),
      "vkMapMemory");
}

::testing::AssertionResult Buffer::Flush() {
  if (properties_ & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) {
    return ::testing::AssertionSuccess();
  }
  VkMappedMemoryRange range = {VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
  range.memory = memory_;
  range.size = VK_WHOLE_SIZE;
  return CheckResult(
      device_->Get<PFN_vkFlushMappedMemoryRanges>("vkFlushMappedMemoryRanges")(
          device_->handle(), 1, &range),
      "vkFlushMappedMemoryRanges");
}

::testing::AssertionResult Buffer::Invalidate() {
  if (properties_ & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) {
    return ::testing::AssertionSuccess();
  }
  VkMappedMemoryRange range = {VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
  range.memory = memory_;
  range.size = VK_WHOLE_SIZE;
  return CheckResult(
      device_->Get<PFN_vkInvalidateMappedMemoryRanges>(
          "vkInvalidateMappedMemoryRanges")(device_->handle(), 1, &range),
      "vkInvalidateMappedMemoryRanges");
}

VkDeviceAddress Buffer::device_address() const {
  VkBufferDeviceAddressInfo info = {
      VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
  info.buffer = buffer_;
  return device_->Get<PFN_vkGetBufferDeviceAddress>("vkGetBufferDeviceAddress")(
      device_->handle(), &info);
}

}  // namespace iree::vulkan::testing
