// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/device/support/kernel.h"

IREE_AMDGPU_ATTRIBUTE_KERNEL void command_buffer_dispatch_constants_test(
    uint32_t* output, uint32_t c0, uint32_t c1, uint32_t c2, uint32_t c3) {
  output[0] = c0;
  output[1] = c1;
  output[2] = c2;
  output[3] = c3;
}
