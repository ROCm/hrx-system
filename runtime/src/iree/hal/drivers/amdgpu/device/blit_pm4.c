// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/device/blit_pm4.h"

#include <inttypes.h>
#include <string.h>

iree_status_t iree_hal_amdgpu_device_buffer_transfer_pm4_context_initialize(
    const iree_hal_amdgpu_libhsa_t* libhsa,
    iree_hal_amdgpu_gfxip_version_t gfxip_version,
    const iree_hal_amdgpu_device_buffer_transfer_context_t* transfer_context,
    iree_hal_amdgpu_device_buffer_transfer_pm4_context_t* out_context) {
  IREE_ASSERT_ARGUMENT(libhsa);
  IREE_ASSERT_ARGUMENT(transfer_context);
  IREE_ASSERT_ARGUMENT(out_context);
  memset(out_context, 0, sizeof(*out_context));

  const uint16_t workgroup_size[3] = {
      transfer_context->workgroup_size_x,
      1,
      1,
  };
  for (iree_host_size_t i = 0;
       i < IREE_HAL_AMDGPU_DEVICE_BUFFER_TRANSFER_KERNEL_COUNT; ++i) {
    const iree_hal_amdgpu_device_buffer_transfer_kernel_t kernel =
        (iree_hal_amdgpu_device_buffer_transfer_kernel_t)i;
    iree_status_t status = iree_hal_amdgpu_device_kernel_pm4_launch_initialize(
        libhsa, gfxip_version,
        iree_hal_amdgpu_device_buffer_transfer_kernel_args(transfer_context,
                                                           kernel),
        workgroup_size, &out_context->launches[i]);
    if (!iree_status_is_ok(status)) {
      return iree_status_annotate_f(status, "transfer kernel ordinal %" PRIhsz,
                                    i);
    }
  }
  return iree_ok_status();
}
