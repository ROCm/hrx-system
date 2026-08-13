// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/device/atomic_pm4.h"

iree_status_t iree_hal_amdgpu_device_atomic_pm4_context_initialize(
    const iree_hal_amdgpu_libhsa_t* libhsa,
    iree_hal_amdgpu_gfxip_version_t gfxip_version,
    const iree_hal_amdgpu_device_kernels_t* kernels,
    iree_hal_amdgpu_device_atomic_pm4_context_t* out_context) {
  IREE_ASSERT_ARGUMENT(libhsa);
  IREE_ASSERT_ARGUMENT(kernels);
  IREE_ASSERT_ARGUMENT(out_context);
  memset(out_context, 0, sizeof(*out_context));

#define IREE_HAL_AMDGPU_INITIALIZE_ATOMIC_PM4_LAUNCH(kind, width)              \
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_device_kernel_pm4_launch_initialize(    \
      libhsa, gfxip_version,                                                   \
      &kernels->iree_hal_amdgpu_device_atomic_##kind##_x##width,               \
      kernels->iree_hal_amdgpu_device_atomic_##kind##_x##width.workgroup_size, \
      &out_context->kind##_x##width))
  IREE_HAL_AMDGPU_INITIALIZE_ATOMIC_PM4_LAUNCH(wait, 32);
  IREE_HAL_AMDGPU_INITIALIZE_ATOMIC_PM4_LAUNCH(wait, 64);
  IREE_HAL_AMDGPU_INITIALIZE_ATOMIC_PM4_LAUNCH(store, 32);
  IREE_HAL_AMDGPU_INITIALIZE_ATOMIC_PM4_LAUNCH(store, 64);
  IREE_HAL_AMDGPU_INITIALIZE_ATOMIC_PM4_LAUNCH(rmw, 32);
  IREE_HAL_AMDGPU_INITIALIZE_ATOMIC_PM4_LAUNCH(rmw, 64);
#undef IREE_HAL_AMDGPU_INITIALIZE_ATOMIC_PM4_LAUNCH

  return iree_ok_status();
}

const iree_hal_amdgpu_device_kernel_pm4_launch_t*
iree_hal_amdgpu_device_atomic_pm4_context_select_wait(
    const iree_hal_amdgpu_device_atomic_pm4_context_t* context,
    iree_hal_atomic_width_t width) {
  return width == IREE_HAL_ATOMIC_WIDTH_32 ? &context->wait_x32
                                           : &context->wait_x64;
}

const iree_hal_amdgpu_device_kernel_pm4_launch_t*
iree_hal_amdgpu_device_atomic_pm4_context_select_store(
    const iree_hal_amdgpu_device_atomic_pm4_context_t* context,
    iree_hal_atomic_width_t width) {
  return width == IREE_HAL_ATOMIC_WIDTH_32 ? &context->store_x32
                                           : &context->store_x64;
}

const iree_hal_amdgpu_device_kernel_pm4_launch_t*
iree_hal_amdgpu_device_atomic_pm4_context_select_rmw(
    const iree_hal_amdgpu_device_atomic_pm4_context_t* context,
    iree_hal_atomic_width_t width) {
  return width == IREE_HAL_ATOMIC_WIDTH_32 ? &context->rmw_x32
                                           : &context->rmw_x64;
}
