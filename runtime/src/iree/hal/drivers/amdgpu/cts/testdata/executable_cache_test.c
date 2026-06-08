// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/device/support/kernel.h"

IREE_AMDGPU_ATTRIBUTE_KERNEL void executable_cache_test(float* input,
                                                        float* output) {
  float value = input[0];
  output[0] = value < 0.0f ? -value : value;
}
