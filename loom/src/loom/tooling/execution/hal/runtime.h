// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// HAL runtime setup and completion handling for Loom execution sessions.

#ifndef LOOM_TOOLING_EXECUTION_HAL_RUNTIME_H_
#define LOOM_TOOLING_EXECUTION_HAL_RUNTIME_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif

// Shared HAL runtime state created by the execution layer.
typedef struct loom_run_hal_runtime_t {
  // Selected HAL device used for executable preparation and dispatch.
  iree_hal_device_t* device;
  // Borrowed provisioned queue used for dispatch submissions.
  // The runtime-owned |device| outlives this pointer.
  iree_hal_queue_t* dispatch_queue;
  // Borrowed provisioned queue used for host data transfers, if available.
  // The runtime-owned |device| outlives this pointer.
  iree_hal_queue_t* transfer_queue;
  // Reusable queue-allocation pool spanning dispatch and transfer access.
  iree_hal_pool_t* staging_pool;
  // Topology group assigning frontier state to |device|.
  iree_hal_device_group_t* device_group;
} loom_run_hal_runtime_t;

typedef struct loom_run_hal_runtime_options_t {
  // HAL driver component of the selected `--device=` URI.
  iree_string_view_t hal_driver_name;
  // Device event sink used for feedback emitted by the HAL device.
  iree_hal_device_event_sink_t event_sink;
  // HAL runtime services requested for executables loaded into the device.
  iree_hal_device_runtime_feature_flags_t runtime_features;
} loom_run_hal_runtime_options_t;

// Initializes |out_options| with default HAL runtime creation policy.
void loom_run_hal_runtime_options_initialize(
    iree_string_view_t hal_driver_name,
    loom_run_hal_runtime_options_t* out_options);

// Initializes the HAL runtime state using |options|.
iree_status_t loom_run_hal_runtime_initialize(
    const loom_run_hal_runtime_options_t* options, iree_allocator_t allocator,
    loom_run_hal_runtime_t* out_runtime);

// Returns the physical devices served by the provisioned dispatch queue.
iree_hal_physical_device_affinity_t
loom_run_hal_runtime_dispatch_physical_device_affinity(
    const loom_run_hal_runtime_t* runtime);

// Releases all resources owned by |runtime|.
void loom_run_hal_runtime_deinitialize(loom_run_hal_runtime_t* runtime);

// Waits for |semaphore| to reach |value|. On failure, queries the semaphore for
// its saved diagnostic and joins the wait error as context. If the semaphore
// has not failed, returns the original wait error. The returned status is owned
// independently of the semaphore. Successful waits do not query the semaphore.
iree_status_t loom_run_hal_semaphore_wait(iree_hal_semaphore_t* semaphore,
                                          uint64_t value,
                                          iree_timeout_t timeout,
                                          iree_async_wait_flags_t flags);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLING_EXECUTION_HAL_RUNTIME_H_
