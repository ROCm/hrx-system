// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_HOST_QUEUE_POLICY_H_
#define IREE_HAL_DRIVERS_AMDGPU_HOST_QUEUE_POLICY_H_

#include "iree/hal/drivers/amdgpu/host_queue.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Returns the stronger of two HSA fence scopes.
static inline iree_hsa_fence_scope_t iree_hal_amdgpu_host_queue_max_fence_scope(
    iree_hsa_fence_scope_t lhs, iree_hsa_fence_scope_t rhs) {
  return lhs > rhs ? lhs : rhs;
}

// Returns the acquire scope required when |queue| consumes a wait edge on
// |semaphore|. This is derived from the semaphore's visibility contract only;
// operation buffers/bindings are intentionally not inspected.
iree_hsa_fence_scope_t iree_hal_amdgpu_host_queue_wait_acquire_scope(
    const iree_hal_amdgpu_host_queue_t* queue, iree_hal_semaphore_t* semaphore);

// Returns the acquire scope required when |queue| waits on a producer
// frontier axis. This is used for pool/death-frontier waits where there is no
// semaphore edge to classify.
iree_hsa_fence_scope_t iree_hal_amdgpu_host_queue_axis_acquire_scope(
    const iree_hal_amdgpu_host_queue_t* queue, iree_async_axis_t axis);

// Returns the release scope required when |queue| publishes a signal edge on
// |semaphore|. This is derived from the semaphore's visibility contract only;
// operation buffers/bindings are intentionally not inspected.
iree_hsa_fence_scope_t iree_hal_amdgpu_host_queue_signal_release_scope(
    const iree_hal_amdgpu_host_queue_t* queue, iree_hal_semaphore_t* semaphore);

// Returns the release scope required for all signal semaphores in |semaphores|.
// An empty list still requires AGENT scope.
iree_hsa_fence_scope_t iree_hal_amdgpu_host_queue_signal_list_release_scope(
    const iree_hal_amdgpu_host_queue_t* queue,
    iree_hal_semaphore_list_t semaphores);

// Returns the acquire scope an operation's reads from |source_buffer| require
// on their own, independent of any semaphore edge. A source the host can map
// requires SYSTEM scope so host writes are visible to the device operation.
iree_hsa_fence_scope_t iree_hal_amdgpu_host_queue_buffer_acquire_scope(
    const iree_hal_buffer_t* source_buffer);

// Returns the release scope an operation's writes into |target_buffer| require
// on their own, independent of any semaphore edge. A target the host can map
// must be published at SYSTEM scope: an AGENT release need not make writes
// visible outside the agent, and nothing running on the host can write back the
// GPU L2 to compensate.
iree_hsa_fence_scope_t iree_hal_amdgpu_host_queue_buffer_release_scope(
    const iree_hal_buffer_t* target_buffer);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_HOST_QUEUE_POLICY_H_
