// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "build_tools/vulkan/testing/submission.h"

#include <cstdlib>

namespace iree::vulkan::testing {

Submission::~Submission() {
  if (pending_) {
    ADD_FAILURE() << "Cannot destroy native resources after an uncompleted "
                     "Vulkan submission";
    std::abort();
  }
  if (fence_) {
    device_->Get<PFN_vkDestroyFence>("vkDestroyFence")(device_->handle(),
                                                       fence_, nullptr);
  }
  if (pool_) {
    device_->Get<PFN_vkDestroyCommandPool>("vkDestroyCommandPool")(
        device_->handle(), pool_, nullptr);
  }
}

::testing::AssertionResult Submission::Initialize(const Device& device,
                                                  uint32_t queue_family_index,
                                                  uint32_t queue_index) {
  device_ = &device;
  device.Get<PFN_vkGetDeviceQueue>("vkGetDeviceQueue")(
      device.handle(), queue_family_index, queue_index, &queue_);
  VkCommandPoolCreateInfo pool_info = {
      VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  pool_info.queueFamilyIndex = queue_family_index;
  auto result =
      CheckResult(device.Get<PFN_vkCreateCommandPool>("vkCreateCommandPool")(
                      device.handle(), &pool_info, nullptr, &pool_),
                  "vkCreateCommandPool");
  if (!result) {
    return result;
  }
  VkCommandBufferAllocateInfo command_info = {
      VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  command_info.commandPool = pool_;
  command_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  command_info.commandBufferCount = 1;
  result = CheckResult(
      device.Get<PFN_vkAllocateCommandBuffers>("vkAllocateCommandBuffers")(
          device.handle(), &command_info, &command_buffer_),
      "vkAllocateCommandBuffers");
  if (!result) {
    return result;
  }
  VkFenceCreateInfo fence_info = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
  return CheckResult(device.Get<PFN_vkCreateFence>("vkCreateFence")(
                         device.handle(), &fence_info, nullptr, &fence_),
                     "vkCreateFence");
}

::testing::AssertionResult Submission::Begin() {
  auto result =
      CheckResult(device_->Get<PFN_vkResetCommandPool>("vkResetCommandPool")(
                      device_->handle(), pool_, 0),
                  "vkResetCommandPool");
  if (!result) {
    return result;
  }
  VkCommandBufferBeginInfo info = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  return CheckResult(device_->Get<PFN_vkBeginCommandBuffer>(
                         "vkBeginCommandBuffer")(command_buffer_, &info),
                     "vkBeginCommandBuffer");
}

::testing::AssertionResult Submission::SubmitAndWait() {
  auto result = CheckResult(device_->Get<PFN_vkEndCommandBuffer>(
                                "vkEndCommandBuffer")(command_buffer_),
                            "vkEndCommandBuffer");
  if (!result) {
    return result;
  }
  result = CheckResult(device_->Get<PFN_vkResetFences>("vkResetFences")(
                           device_->handle(), 1, &fence_),
                       "vkResetFences");
  if (!result) {
    return result;
  }
  VkSubmitInfo info = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
  info.commandBufferCount = 1;
  info.pCommandBuffers = &command_buffer_;
  result = CheckResult(device_->Get<PFN_vkQueueSubmit>("vkQueueSubmit")(
                           queue_, 1, &info, fence_),
                       "vkQueueSubmit");
  if (!result) {
    return result;
  }
  pending_ = true;
  VkResult wait_result = device_->Get<PFN_vkWaitForFences>("vkWaitForFences")(
      device_->handle(), 1, &fence_, VK_TRUE, UINT64_MAX);
  if (wait_result == VK_SUCCESS || wait_result == VK_ERROR_DEVICE_LOST) {
    pending_ = false;
  }
  return CheckResult(wait_result, "vkWaitForFences");
}

}  // namespace iree::vulkan::testing
