// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVER_VULKAN_SEMAPHORE_H_
#define IREE_HAL_DRIVER_VULKAN_SEMAPHORE_H_

#include <stdint.h>

#include "iree/async/semaphore.h"
#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/driver/vulkan/util/libvulkan.h"

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

// Device-ready point for one accepted queue signal. The logical device owns
// the native timeline through all submissions using this point. Native
// readiness does not publish host completion or retire captured resources.
typedef struct iree_hal_vulkan_semaphore_submission_t {
  // Native producer timeline, or VK_NULL_HANDLE for a host-dependent producer.
  VkSemaphore semaphore;
  // Causal coordinate of the producing queue.
  iree_async_axis_t axis;
  // Native timeline value making the producer's device effects visible.
  uint64_t epoch;
  // Exact public timeline value associated with this producer.
  uint64_t value;
} iree_hal_vulkan_semaphore_submission_t;

// Looks up the exact requested producer. A newer producer cannot substitute:
// it may depend on this consumer. Returns false for unknown or host-dependent
// producers, which require ordinary software timepoints before native
// admission.
bool iree_hal_vulkan_semaphore_query_submission(
    iree_hal_semaphore_t* semaphore, uint64_t value,
    iree_hal_vulkan_semaphore_submission_t* out_submission);

// Returns a borrowed availability timeline, owned by |semaphore|. A point
// becomes available after native submission or after public host completion.
// Lazily allocates its callback storage on first use.
iree_status_t iree_hal_vulkan_semaphore_submission_timeline(
    iree_hal_semaphore_t* semaphore, iree_async_semaphore_t** out_timeline);

// Announces accepted native work outside the queue's submission mutex. May
// synchronously make dependent queue operations ready. Public success is
// separate.
void iree_hal_vulkan_semaphore_notify_submission(
    iree_hal_semaphore_t* semaphore, uint64_t value);

// Publishes submission-time frontier metadata for a future queue signal.
void iree_hal_vulkan_semaphore_publish_signal(
    iree_hal_semaphore_t* semaphore, iree_async_axis_t producer_axis,
    const iree_async_frontier_t* producer_frontier,
    VkSemaphore producer_semaphore, uint64_t producer_epoch,
    uint64_t producer_value);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVER_VULKAN_SEMAPHORE_H_
