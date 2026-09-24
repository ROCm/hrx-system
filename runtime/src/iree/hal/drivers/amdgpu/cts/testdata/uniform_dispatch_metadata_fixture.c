// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/device/support/kernel.h"

IREE_AMDGPU_ATTRIBUTE_KERNEL void uniform_dispatch_metadata_fixture(
    uint32_t* output) {
  // Exactly one workitem writes the caller's first output word in any
  // nonempty dispatch, independently of the caller's nominal XYZ shape.
  if ((iree_hal_amdgpu_device_group_id_x() |
       iree_hal_amdgpu_device_group_id_y() |
       iree_hal_amdgpu_device_group_id_z() |
       iree_hal_amdgpu_device_local_id_x() |
       iree_hal_amdgpu_device_local_id_y() |
       iree_hal_amdgpu_device_local_id_z()) == 0) {
    output[0] = 0x554e4946u;
  }
}
