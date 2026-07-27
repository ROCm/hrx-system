// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/device/support/kernel.h"

IREE_AMDGPU_ATTRIBUTE_KERNEL void command_buffer_dispatch_multi_workgroup_test(
    uint32_t* output) {
  uint32_t workgroup_id = iree_hal_amdgpu_device_group_id_x();
  output[workgroup_id] = workgroup_id;
}
