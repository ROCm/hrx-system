// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_DEVICE_ATOMIC_PM4_H_
#define IREE_HAL_DRIVERS_AMDGPU_DEVICE_ATOMIC_PM4_H_

#include "iree/hal/drivers/amdgpu/device/atomic.h"
#include "iree/hal/drivers/amdgpu/util/libhsa.h"
#include "iree/hal/drivers/amdgpu/util/pm4_dispatch.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Cached PM4 launch metadata for one builtin atomic kernel.
typedef struct iree_hal_amdgpu_device_atomic_pm4_launch_t {
  // Immutable HSA kernel launch arguments.
  iree_hal_amdgpu_device_kernel_args_t kernel_args;
  // Derived PM4 shader launch state.
  iree_hal_amdgpu_pm4_dispatch_launch_state_t launch_state;
  // Complete static PM4 shader setup packet stream.
  uint32_t setup_dwords[IREE_HAL_AMDGPU_PM4_DISPATCH_SETUP_DWORD_COUNT];
} iree_hal_amdgpu_device_atomic_pm4_launch_t;

// Per-physical-device PM4 launch metadata for the builtin atomic kernels.
typedef struct iree_hal_amdgpu_device_atomic_pm4_context_t {
  // 32-bit atomic wait launch.
  iree_hal_amdgpu_device_atomic_pm4_launch_t wait_x32;
  // 64-bit atomic wait launch.
  iree_hal_amdgpu_device_atomic_pm4_launch_t wait_x64;
  // 32-bit atomic store launch.
  iree_hal_amdgpu_device_atomic_pm4_launch_t store_x32;
  // 64-bit atomic store launch.
  iree_hal_amdgpu_device_atomic_pm4_launch_t store_x64;
  // 32-bit atomic read-modify-write launch.
  iree_hal_amdgpu_device_atomic_pm4_launch_t rmw_x32;
  // 64-bit atomic read-modify-write launch.
  iree_hal_amdgpu_device_atomic_pm4_launch_t rmw_x64;
} iree_hal_amdgpu_device_atomic_pm4_context_t;

// Initializes immutable PM4 launch metadata for |kernels|.
//
// The context owns no resources and remains valid independently of the HSA
// loader mappings used to derive it.
iree_status_t iree_hal_amdgpu_device_atomic_pm4_context_initialize(
    const iree_hal_amdgpu_libhsa_t* libhsa,
    iree_hal_amdgpu_gfxip_version_t gfxip_version,
    const iree_hal_amdgpu_device_kernels_t* kernels,
    iree_hal_amdgpu_device_atomic_pm4_context_t* out_context);

// Selects the launch metadata for a validated atomic wait width.
const iree_hal_amdgpu_device_atomic_pm4_launch_t*
iree_hal_amdgpu_device_atomic_pm4_context_select_wait(
    const iree_hal_amdgpu_device_atomic_pm4_context_t* context,
    iree_hal_atomic_width_t width);

// Selects the launch metadata for a validated atomic store width.
const iree_hal_amdgpu_device_atomic_pm4_launch_t*
iree_hal_amdgpu_device_atomic_pm4_context_select_store(
    const iree_hal_amdgpu_device_atomic_pm4_context_t* context,
    iree_hal_atomic_width_t width);

// Selects the launch metadata for a validated atomic RMW width.
const iree_hal_amdgpu_device_atomic_pm4_launch_t*
iree_hal_amdgpu_device_atomic_pm4_context_select_rmw(
    const iree_hal_amdgpu_device_atomic_pm4_context_t* context,
    iree_hal_atomic_width_t width);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_DEVICE_ATOMIC_PM4_H_
