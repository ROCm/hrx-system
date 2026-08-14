// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/device/kernel_pm4.h"

#include <string.h>

#include "iree/hal/drivers/amdgpu/util/loaded_code_object.h"

iree_status_t iree_hal_amdgpu_device_kernel_pm4_launch_initialize(
    const iree_hal_amdgpu_libhsa_t* libhsa,
    iree_hal_amdgpu_gfxip_version_t gfxip_version,
    const iree_hal_amdgpu_device_kernel_args_t* kernel_args,
    const uint16_t workgroup_size[3],
    iree_hal_amdgpu_device_kernel_pm4_launch_t* out_launch) {
  IREE_ASSERT_ARGUMENT(libhsa);
  IREE_ASSERT_ARGUMENT(kernel_args);
  IREE_ASSERT_ARGUMENT(workgroup_size);
  IREE_ASSERT_ARGUMENT(out_launch);
  memset(out_launch, 0, sizeof(*out_launch));
  if (IREE_UNLIKELY(kernel_args->kernel_object == 0)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "builtin kernel object is unavailable");
  }
  if (IREE_UNLIKELY(kernel_args->workgroup_cluster_size[0] != 0 ||
                    kernel_args->workgroup_cluster_size[1] != 0 ||
                    kernel_args->workgroup_cluster_size[2] != 0)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "builtin kernel unexpectedly requires workgroup clusters");
  }

  const iree_hal_amdgpu_kernel_descriptor_t* descriptor = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_loaded_code_object_query_host_address(
      libhsa, kernel_args->kernel_object, (const void**)&descriptor));
  out_launch->kernel_args = *kernel_args;
  memcpy(out_launch->kernel_args.workgroup_size, workgroup_size,
         sizeof(out_launch->kernel_args.workgroup_size));
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_pm4_dispatch_launch_state_initialize(
      gfxip_version, descriptor, kernel_args->kernel_object, workgroup_size,
      IREE_HAL_AMDGPU_PM4_DISPATCH_LAUNCH_FLAG_NONE,
      &out_launch->launch_state));
  uint32_t setup_dword_count = 0;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_pm4_dispatch_emit_setup(
      &out_launch->launch_state, IREE_HAL_AMDGPU_PM4_DISPATCH_SETUP_DWORD_COUNT,
      out_launch->setup_dwords, &setup_dword_count));
  if (IREE_UNLIKELY(setup_dword_count !=
                    IREE_HAL_AMDGPU_PM4_DISPATCH_SETUP_DWORD_COUNT)) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "builtin PM4 dispatch setup emission changed size");
  }
  return iree_ok_status();
}
