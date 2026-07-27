// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/device/support/kernel.h"

IREE_AMDGPU_ATTRIBUTE_KERNEL void
command_buffer_dispatch_constants_bindings_test(uint32_t* input,
                                                uint32_t* output,
                                                uint32_t scale,
                                                uint32_t offset) {
  for (uint32_t i = 0; i < 4; ++i) {
    output[i] = input[i] * scale + offset;
  }
}

// Groups opaque device pointers into one by-value kernel argument. The host
// custom-direct path must preserve these bytes without trying to discover or
// rewrite the pointers nested in the structure.
typedef struct command_buffer_dispatch_pointer_args_t {
  // Input values to transform.
  uint32_t* input;
  // Output values written by the kernel.
  uint32_t* output;
} command_buffer_dispatch_pointer_args_t;

IREE_AMDGPU_ATTRIBUTE_KERNEL void command_buffer_dispatch_nested_pointers_test(
    command_buffer_dispatch_pointer_args_t pointers, uint32_t scale,
    uint32_t offset) {
  for (uint32_t i = 0; i < 4; ++i) {
    pointers.output[i] = pointers.input[i] * scale + offset;
  }
}
