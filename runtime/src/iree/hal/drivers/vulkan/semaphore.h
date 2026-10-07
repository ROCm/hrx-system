// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_VULKAN_SEMAPHORE_H_
#define IREE_HAL_DRIVERS_VULKAN_SEMAPHORE_H_

#include <stdint.h>

#include "iree/async/semaphore.h"
#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/vulkan/util/libvulkan.h"
#include "iree/hal/utils/submitted_signal.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_hal_vulkan_logical_device_t
    iree_hal_vulkan_logical_device_t;

// Creates a Vulkan HAL semaphore backed by a native timeline VkSemaphore.
iree_status_t iree_hal_vulkan_semaphore_create(
    iree_hal_vulkan_logical_device_t* device,
    const iree_hal_vulkan_device_syms_t* syms, VkDevice logical_device,
    iree_async_proactor_t* proactor, uint64_t initial_value,
    iree_hal_semaphore_flags_t flags, iree_allocator_t host_allocator,
    iree_hal_semaphore_t** out_semaphore);

// Returns true if |semaphore| is a Vulkan timeline semaphore.
bool iree_hal_vulkan_semaphore_isa(iree_hal_semaphore_t* semaphore);

// Returns true if |semaphore| belongs to |device|.
bool iree_hal_vulkan_semaphore_is_local(
    iree_hal_semaphore_t* semaphore,
    const iree_hal_vulkan_logical_device_t* device);

// Returns the Vulkan semaphore creation flags.
iree_hal_semaphore_flags_t iree_hal_vulkan_semaphore_flags(
    iree_hal_semaphore_t* semaphore);

// Returns the native Vulkan timeline semaphore handle.
iree_status_t iree_hal_vulkan_semaphore_handle(iree_hal_semaphore_t* semaphore,
                                               VkSemaphore* out_handle);

// Returns the latest submitted-signal metadata on a Vulkan semaphore.
iree_hal_submitted_signal_t* iree_hal_vulkan_semaphore_submitted_signal(
    iree_hal_semaphore_t* semaphore);

// Publishes submission-time frontier metadata for a future queue signal.
bool iree_hal_vulkan_semaphore_publish_signal(
    iree_hal_semaphore_t* semaphore, iree_async_axis_t producer_axis,
    const iree_async_frontier_t* producer_frontier, uint64_t producer_epoch,
    uint64_t producer_value);

// Advances the HAL/async timeline after a native queue signal has retired.
iree_status_t iree_hal_vulkan_semaphore_retire_signal(
    iree_hal_semaphore_t* semaphore, uint64_t value,
    const iree_async_frontier_t* frontier);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_VULKAN_SEMAPHORE_H_
