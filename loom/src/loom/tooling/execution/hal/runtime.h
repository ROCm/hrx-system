// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Generic HAL runtime setup for Loom execution sessions.

#ifndef LOOM_TOOLING_EXECUTION_HAL_RUNTIME_H_
#define LOOM_TOOLING_EXECUTION_HAL_RUNTIME_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/vm/api.h"

#ifdef __cplusplus
extern "C" {
#endif

// Shared HAL runtime state created by the execution layer.
typedef struct loom_run_hal_runtime_t {
  // VM instance retaining HAL ref-type registrations for function I/O helpers.
  iree_vm_instance_t* instance;
  // Selected HAL device used for executable preparation and dispatch.
  iree_hal_device_t* device;
  // Topology group assigning frontier state to |device|.
  iree_hal_device_group_t* device_group;
  // Executable cache owned by |device| and used for target probing/loading.
  iree_hal_executable_cache_t* executable_cache;
} loom_run_hal_runtime_t;

// Initializes the HAL runtime state for |hal_driver_name|.
iree_status_t loom_run_hal_runtime_initialize(
    iree_string_view_t hal_driver_name, iree_allocator_t allocator,
    loom_run_hal_runtime_t* out_runtime);

// Releases all resources owned by |runtime|.
void loom_run_hal_runtime_deinitialize(loom_run_hal_runtime_t* runtime);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLING_EXECUTION_HAL_RUNTIME_H_
