// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/device/support/kernel.h"

IREE_AMDGPU_ATTRIBUTE_KERNEL void command_buffer_dispatch_test(float* input,
                                                               float* output) {
  for (uint32_t i = 0; i < 2; ++i) {
    float value = input[i];
    output[i] = value < 0.0f ? -value : value;
  }
}
