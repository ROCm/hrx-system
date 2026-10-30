// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/vulkan/queue_test_util.h"

#include "iree/hal/drivers/vulkan/queue.h"

namespace iree::hal::cts {

void RetireVulkanQueue(iree_hal_queue_t* queue) {
  iree_hal_vulkan_queue_retire_frontier(
      reinterpret_cast<iree_hal_vulkan_queue_t*>(queue));
}

ScopedVulkanQueueGate::~ScopedVulkanQueueGate() {
  Open();
  auto* queue = reinterpret_cast<iree_hal_vulkan_queue_t*>(queue_);
  if (completed_) {
    iree_vkDestroyFence(IREE_VULKAN_DEVICE(&queue->syms), queue->logical_device,
                        completed_, nullptr);
  }
  if (semaphore_) {
    iree_vkDestroySemaphore(IREE_VULKAN_DEVICE(&queue->syms),
                            queue->logical_device, semaphore_, nullptr);
  }
}

iree_status_t ScopedVulkanQueueGate::Begin() {
  auto* queue = reinterpret_cast<iree_hal_vulkan_queue_t*>(queue_);
  const VkSemaphoreTypeCreateInfo timeline_info = {
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
      .semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE,
  };
  const VkSemaphoreCreateInfo semaphore_info = {
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
      .pNext = &timeline_info,
  };
  IREE_RETURN_IF_ERROR(iree_vkCreateSemaphore(
      IREE_VULKAN_DEVICE(&queue->syms), queue->logical_device, &semaphore_info,
      nullptr, &semaphore_));
  const VkFenceCreateInfo fence_info = {
      .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
  };
  IREE_RETURN_IF_ERROR(iree_vkCreateFence(IREE_VULKAN_DEVICE(&queue->syms),
                                          queue->logical_device, &fence_info,
                                          nullptr, &completed_));
  const VkSemaphoreSubmitInfo wait_info = {
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
      .semaphore = semaphore_,
      .value = 1,
      .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
  };
  const VkSubmitInfo2 submit = {
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
      .waitSemaphoreInfoCount = 1,
      .pWaitSemaphoreInfos = &wait_info,
  };
  iree_slim_mutex_lock(&queue->queue_handle_mutex);
  iree_status_t status = iree_vkQueueSubmit2(
      IREE_VULKAN_DEVICE(&queue->syms), queue->queue, 1, &submit, completed_);
  iree_slim_mutex_unlock(&queue->queue_handle_mutex);
  submitted_ = iree_status_is_ok(status);
  return status;
}

void ScopedVulkanQueueGate::Open() {
  if (!submitted_) {
    return;
  }
  auto* queue = reinterpret_cast<iree_hal_vulkan_queue_t*>(queue_);
  const VkSemaphoreSignalInfo signal = {
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO,
      .semaphore = semaphore_,
      .value = 1,
  };
  IREE_CHECK_OK(iree_vkSignalSemaphore(IREE_VULKAN_DEVICE(&queue->syms),
                                       queue->logical_device, &signal));
  IREE_CHECK_OK(iree_vkWaitForFences(IREE_VULKAN_DEVICE(&queue->syms),
                                     queue->logical_device, 1, &completed_,
                                     VK_TRUE, UINT64_MAX));
  submitted_ = false;
}

iree_hal_queue_t* FindVulkanSparseBindingQueue(iree_hal_device_t* device) {
  const auto* queues =
      iree_hal_device_spec_queues(iree_hal_device_spec(device));
  for (iree_host_size_t i = 0; i < queues->family_count; ++i) {
    if (!queues->families[i].provisioned_queue_count) {
      continue;
    }
    auto* queue =
        reinterpret_cast<iree_hal_vulkan_queue_t*>(iree_hal_device_queue(
            device, static_cast<iree_hal_queue_family_ordinal_t>(i), 0));
    if (iree_any_bit_set(queue->queue_flags, VK_QUEUE_SPARSE_BINDING_BIT)) {
      return &queue->base;
    }
  }
  return nullptr;
}

}  // namespace iree::hal::cts
