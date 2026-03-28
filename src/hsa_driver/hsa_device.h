// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_HSA_DEVICE_H_
#define IREE_HAL_DRIVERS_HSA_DEVICE_H_

#include <stdint.h>

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "hsa_driver/api.h"
#include "hsa_driver/hsa_headers.h"

// Creates a device that manages an HSA GPU agent.
iree_status_t iree_hal_hsa_device_create(
    iree_hal_driver_t* driver, iree_string_view_t identifier,
    const iree_hal_hsa_device_params_t* params, hsa_agent_t gpu_agent,
    hsa_agent_t cpu_agent,
    const iree_hal_device_create_params_t* create_params,
    iree_allocator_t host_allocator, iree_hal_device_t** out_device);

// Hide the cast in a function.
static inline iree_device_size_t iree_hal_hsa_device_ptr_to_device_size(
    void* p) {
  return (uintptr_t)p;
}
static inline void* iree_hal_hsa_device_size_to_hsa_device_ptr(
    iree_device_size_t p) {
  return (void*)p;
}

#endif  // IREE_HAL_DRIVERS_HSA_DEVICE_H_

