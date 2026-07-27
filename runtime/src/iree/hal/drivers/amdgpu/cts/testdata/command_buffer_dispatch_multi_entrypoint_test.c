// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/device/support/kernel.h"

IREE_AMDGPU_ATTRIBUTE_KERNEL void entry0_negate(uint32_t* input,
                                                int32_t* output) {
  for (uint32_t i = 0; i < 4; ++i) {
    output[i] = -(int32_t)input[i];
  }
}

IREE_AMDGPU_ATTRIBUTE_KERNEL void entry1_double_it(uint32_t* input,
                                                   uint32_t* output) {
  for (uint32_t i = 0; i < 4; ++i) {
    output[i] = input[i] * 2;
  }
}
