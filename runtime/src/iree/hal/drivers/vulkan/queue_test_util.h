// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_VULKAN_QUEUE_TEST_UTIL_H_
#define IREE_HAL_DRIVERS_VULKAN_QUEUE_TEST_UTIL_H_

#include "iree/hal/api.h"
#include "iree/hal/drivers/vulkan/util/libvulkan.h"

namespace iree::hal::cts {

// Returns a borrowed provisioned Vulkan queue supporting sparse publication,
// or nullptr if none exists. Transfer support alone does not imply this role.
iree_hal_queue_t* FindVulkanSparseBindingQueue(iree_hal_device_t* device);

// Runs the production queue teardown boundary without destroying the fixture's
// device, allowing callers to observe cancellation before releasing resources.
void RetireVulkanQueue(iree_hal_queue_t* queue);

// Holds native execution behind a timeline wait while HAL submissions continue
// to acquire their normal resources. Opening also retires the native wait, so
// even a failed HAL submission leaves no gate resource in use. Destruction
// opens an armed gate before freeing it.
class ScopedVulkanQueueGate {
 public:
  explicit ScopedVulkanQueueGate(iree_hal_queue_t* queue) : queue_(queue) {}
  ~ScopedVulkanQueueGate();
  ScopedVulkanQueueGate(const ScopedVulkanQueueGate&) = delete;
  ScopedVulkanQueueGate& operator=(const ScopedVulkanQueueGate&) = delete;

  iree_status_t Begin();
  void Open();

 private:
  // Borrowed queue whose native submission order is gated.
  iree_hal_queue_t* queue_;
  // Owned timeline semaphore signaled by Open.
  VkSemaphore semaphore_ = VK_NULL_HANDLE;
  // Owned fence proving the native wait no longer references the semaphore.
  VkFence completed_ = VK_NULL_HANDLE;
  // True while the successfully submitted native wait has not retired.
  bool submitted_ = false;
};

}  // namespace iree::hal::cts

#endif  // IREE_HAL_DRIVERS_VULKAN_QUEUE_TEST_UTIL_H_
