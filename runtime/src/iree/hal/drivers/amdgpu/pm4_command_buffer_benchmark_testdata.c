// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/device/support/kernel.h"

IREE_AMDGPU_ATTRIBUTE_KERNEL void model_a(uint32_t* input, uint32_t* output) {
  output[0] = input[0];
}

IREE_AMDGPU_ATTRIBUTE_KERNEL void model_b(uint32_t* input, uint32_t* output) {
  output[1] = input[1];
}
