// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/device/support/kernel.h"

[[gnu::visibility("protected"), gnu::used]]
volatile uint64_t executable_test_global = 0x0123456789ABCDEFull;

IREE_AMDGPU_ATTRIBUTE_KERNEL void export0(uint64_t* lhs, uint64_t* rhs,
                                          uint32_t c0, uint32_t c1) {
  if (c0 == c1) {
    lhs[0] = executable_test_global;
  } else {
    lhs[0] = rhs[0];
  }
}
