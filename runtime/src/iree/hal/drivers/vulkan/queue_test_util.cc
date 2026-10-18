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
