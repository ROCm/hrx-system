// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_WEBGPU_WEBGPU_MEMORY_H_
#define IREE_HAL_DRIVERS_WEBGPU_WEBGPU_MEMORY_H_

#include "iree/hal/drivers/webgpu/handle_table.h"
#include "iree/hal/drivers/webgpu/webgpu.h"
#include "iree/hal/memory_backend.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_async_notification_t iree_async_notification_t;

// Minimum binding alignment used by explicit WebGPU storage pools.
#define IREE_HAL_WEBGPU_SLAB_ALIGNMENT 256

// Borrowed native resources used to construct explicit WebGPU slab pools.
// The containing device and sealed group outlive every constructed pool.
typedef struct iree_hal_webgpu_memory_backend_t {
  // Generic native schema and WebGPU construction factory.
  iree_hal_memory_backend_t base;
  // HAL device owning the native GPUDevice.
  iree_hal_device_t* device;
  // Bridge handle for the native GPUDevice.
  iree_hal_webgpu_handle_t device_handle;
  // Device-owned capacity notification shared by constructed pools.
  iree_async_notification_t* notification;
  // Completion probe with a sealed-group lifetime context.
  iree_hal_pool_epoch_query_t epoch_query;
} iree_hal_webgpu_memory_backend_t;

// Publishes the factory header after the caller fills the borrowed resources.
void iree_hal_webgpu_memory_backend_initialize(
    iree_hal_webgpu_memory_backend_t* backend);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_WEBGPU_WEBGPU_MEMORY_H_
