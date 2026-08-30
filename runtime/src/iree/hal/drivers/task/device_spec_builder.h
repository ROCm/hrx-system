// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_TASK_DEVICE_SPEC_BUILDER_H_
#define IREE_HAL_DRIVERS_TASK_DEVICE_SPEC_BUILDER_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/task/executable/loader.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Parameters for creating a task CPU HAL device spec.
typedef struct iree_hal_task_device_spec_params_t {
  // Stable logical device identifier.
  iree_string_view_t logical_device_id;
  // Human-readable logical device name.
  iree_string_view_t display_name;
  // HAL driver identifier.
  iree_string_view_t driver_id;
  // Backend API identifier.
  iree_string_view_t backend_id;
  // Number of HAL queues exposed by the task device.
  iree_host_size_t queue_count;
  // Worker count used by the default dispatch queue.
  iree_host_size_t default_queue_worker_count;
  // Atomic operations implemented by the task queue family.
  iree_hal_atomic_capabilities_t atomic_capabilities;
  // Atomic operations implemented without occupying dispatch resources.
  iree_hal_atomic_capabilities_t zero_compute_atomic_capabilities;
  // Number of executable loaders in |loaders|.
  iree_host_size_t loader_count;
  // Borrowed executable loaders used to advertise supported target families.
  iree_hal_executable_loader_t** loaders;
} iree_hal_task_device_spec_params_t;

// Creates an immutable spec for a task CPU HAL device.
IREE_API_EXPORT iree_status_t iree_hal_task_device_spec_create(
    const iree_hal_task_device_spec_params_t* params,
    iree_allocator_t host_allocator, iree_hal_device_spec_t** out_spec);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_TASK_DEVICE_SPEC_BUILDER_H_
