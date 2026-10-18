// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_VULKAN_QUEUE_TEST_UTIL_H_
#define IREE_HAL_DRIVERS_VULKAN_QUEUE_TEST_UTIL_H_

#include "iree/hal/api.h"

namespace iree::hal::cts {

// Returns a borrowed provisioned Vulkan queue supporting sparse publication,
// or nullptr if none exists. Transfer support alone does not imply this role.
iree_hal_queue_t* FindVulkanSparseBindingQueue(iree_hal_device_t* device);

// Runs the production queue teardown boundary without destroying the fixture's
// device, allowing callers to observe cancellation before releasing resources.
void RetireVulkanQueue(iree_hal_queue_t* queue);

}  // namespace iree::hal::cts

#endif  // IREE_HAL_DRIVERS_VULKAN_QUEUE_TEST_UTIL_H_
