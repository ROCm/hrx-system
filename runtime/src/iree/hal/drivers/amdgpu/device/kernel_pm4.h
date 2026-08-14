// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_DEVICE_KERNEL_PM4_H_
#define IREE_HAL_DRIVERS_AMDGPU_DEVICE_KERNEL_PM4_H_

#include "iree/hal/drivers/amdgpu/device/kernels.h"
#include "iree/hal/drivers/amdgpu/util/libhsa.h"
#include "iree/hal/drivers/amdgpu/util/pm4_dispatch.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Immutable PM4 launch metadata for one builtin device kernel.
typedef struct iree_hal_amdgpu_device_kernel_pm4_launch_t {
  // HSA kernel launch arguments with the selected workgroup size.
  iree_hal_amdgpu_device_kernel_args_t kernel_args;
  // Derived PM4 shader launch state.
  iree_hal_amdgpu_pm4_dispatch_launch_state_t launch_state;
  // Complete static PM4 shader setup packet stream.
  uint32_t setup_dwords[IREE_HAL_AMDGPU_PM4_DISPATCH_SETUP_DWORD_COUNT];
} iree_hal_amdgpu_device_kernel_pm4_launch_t;

// Initializes immutable PM4 launch metadata for one builtin kernel.
//
// |workgroup_size| may specialize the loaded kernel metadata for a physical
// device, as used by wavefront-sized builtin blit launches. The result owns no
// resources and remains valid independently of the HSA loader mappings used to
// derive it.
iree_status_t iree_hal_amdgpu_device_kernel_pm4_launch_initialize(
    const iree_hal_amdgpu_libhsa_t* libhsa,
    iree_hal_amdgpu_gfxip_version_t gfxip_version,
    const iree_hal_amdgpu_device_kernel_args_t* kernel_args,
    const uint16_t workgroup_size[3],
    iree_hal_amdgpu_device_kernel_pm4_launch_t* out_launch);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_DEVICE_KERNEL_PM4_H_
