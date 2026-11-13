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

// Synchronous lifecycle phases entered while constructing a HAL runtime.
//
// Each phase is published immediately before the named work begins. The next
// phase therefore proves that the preceding phase completed.
typedef enum loom_run_hal_runtime_progress_phase_e {
  LOOM_RUN_HAL_RUNTIME_PROGRESS_PHASE_PROACTOR_POOL_CREATE = 0,
  LOOM_RUN_HAL_RUNTIME_PROGRESS_PHASE_FRONTIER_TRACKER_CREATE,
  LOOM_RUN_HAL_RUNTIME_PROGRESS_PHASE_DEVICE_CREATE,
  LOOM_RUN_HAL_RUNTIME_PROGRESS_PHASE_PROACTOR_POOL_RELEASE,
  LOOM_RUN_HAL_RUNTIME_PROGRESS_PHASE_DEVICE_GROUP_CREATE,
  LOOM_RUN_HAL_RUNTIME_PROGRESS_PHASE_FRONTIER_TRACKER_RELEASE,
  LOOM_RUN_HAL_RUNTIME_PROGRESS_PHASE_DISPATCH_QUEUE_SELECT,
  LOOM_RUN_HAL_RUNTIME_PROGRESS_PHASE_TRANSFER_QUEUE_SELECT,
  LOOM_RUN_HAL_RUNTIME_PROGRESS_PHASE_DEVICE_SOURCE_POOL_CREATE,
  LOOM_RUN_HAL_RUNTIME_PROGRESS_PHASE_DEVICE_POOL_CREATE,
  LOOM_RUN_HAL_RUNTIME_PROGRESS_PHASE_STAGING_POOL_CREATE,
  LOOM_RUN_HAL_RUNTIME_PROGRESS_PHASE_FAILURE_DEINITIALIZE,
  LOOM_RUN_HAL_RUNTIME_PROGRESS_PHASE_COUNT,
} loom_run_hal_runtime_progress_phase_t;

// Publishes one runtime construction phase. Callback errors propagate to the
// caller after already-owned resources are released.
typedef iree_status_t (*loom_run_hal_runtime_progress_fn_t)(
    void* user_data, loom_run_hal_runtime_progress_phase_t phase);

typedef struct loom_run_hal_runtime_progress_sink_t {
  // Optional callback invoked at each runtime construction boundary.
  loom_run_hal_runtime_progress_fn_t fn;
  // Opaque callback state borrowed until initialization returns.
  void* user_data;
} loom_run_hal_runtime_progress_sink_t;

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
  // Topology group assigning frontier state to |device|.
  iree_hal_device_group_t* device_group;
  // Reusable queue-allocation pool spanning dispatch and transfer access.
  iree_hal_pool_t* device_pool;
  // Host-mappable pool for upload, readback, and tooling values.
  iree_hal_pool_t* staging_pool;
} loom_run_hal_runtime_t;

typedef struct loom_run_hal_runtime_options_t {
  // HAL driver component of the selected `--device=` URI.
  iree_string_view_t hal_driver_name;
  // Device event sink used for feedback emitted by the HAL device.
  iree_hal_device_event_sink_t event_sink;
  // HAL runtime services requested for executables loaded into the device.
  iree_hal_device_runtime_feature_flags_t runtime_features;
  // Optional synchronous runtime construction progress sink.
  loom_run_hal_runtime_progress_sink_t progress_sink;
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
