// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_DEVICE_BLIT_PM4_H_
#define IREE_HAL_DRIVERS_AMDGPU_DEVICE_BLIT_PM4_H_

#include "iree/hal/drivers/amdgpu/device/blit.h"
#include "iree/hal/drivers/amdgpu/device/kernel_pm4.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Per-physical-device PM4 launch metadata for builtin transfer kernels.
typedef struct iree_hal_amdgpu_device_buffer_transfer_pm4_context_t {
  // Launches indexed by iree_hal_amdgpu_device_buffer_transfer_kernel_t.
  iree_hal_amdgpu_device_kernel_pm4_launch_t
      launches[IREE_HAL_AMDGPU_DEVICE_BUFFER_TRANSFER_KERNEL_COUNT];
} iree_hal_amdgpu_device_buffer_transfer_pm4_context_t;

// Initializes immutable PM4 launch metadata for every builtin transfer kernel.
//
// The context owns no resources and remains valid independently of the HSA
// loader mappings used to derive it.
iree_status_t iree_hal_amdgpu_device_buffer_transfer_pm4_context_initialize(
    const iree_hal_amdgpu_libhsa_t* libhsa,
    iree_hal_amdgpu_gfxip_version_t gfxip_version,
    const iree_hal_amdgpu_device_buffer_transfer_context_t* transfer_context,
    iree_hal_amdgpu_device_buffer_transfer_pm4_context_t* out_context);

// Returns immutable PM4 launch metadata for a validated transfer kernel.
static inline const iree_hal_amdgpu_device_kernel_pm4_launch_t*
iree_hal_amdgpu_device_buffer_transfer_pm4_context_select(
    const iree_hal_amdgpu_device_buffer_transfer_pm4_context_t* context,
    iree_hal_amdgpu_device_buffer_transfer_kernel_t kernel) {
  IREE_ASSERT(kernel >= 0 &&
              kernel < IREE_HAL_AMDGPU_DEVICE_BUFFER_TRANSFER_KERNEL_COUNT);
  return &context->launches[kernel];
}

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_DEVICE_BLIT_PM4_H_
