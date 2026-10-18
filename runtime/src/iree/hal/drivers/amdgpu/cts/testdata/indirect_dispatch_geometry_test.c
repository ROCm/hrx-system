// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/device/support/kernel.h"

IREE_AMDGPU_ATTRIBUTE_KERNEL void write_dispatch_counts(uint32_t* output) {
  output[0] = 32;
  output[1] = 2;
  output[2] = 3;
}

IREE_AMDGPU_ATTRIBUTE_KERNEL void read_dispatch_geometry(uint32_t* output) {
  if (iree_hal_amdgpu_device_local_id_x() != 0) {
    return;
  }

  // Use the known producer shape for addressing so stale geometry is reported
  // as a wrong value instead of changing the output address or losing a record.
  const uint32_t record = (iree_hal_amdgpu_device_group_id_z() * 2 +
                           iree_hal_amdgpu_device_group_id_y()) *
                              32 +
                          iree_hal_amdgpu_device_group_id_x();
  const iree_amdgpu_kernel_implicit_args_t* implicit_args =
      (const iree_amdgpu_kernel_implicit_args_t*)
          __builtin_amdgcn_implicitarg_ptr();
  output[record * 6 + 0] = iree_hal_amdgpu_device_group_count_x();
  output[record * 6 + 1] = iree_hal_amdgpu_device_group_count_y();
  output[record * 6 + 2] = iree_hal_amdgpu_device_group_count_z();
  output[record * 6 + 3] = implicit_args->block_count[0];
  output[record * 6 + 4] = implicit_args->block_count[1];
  output[record * 6 + 5] = implicit_args->block_count[2];
}
