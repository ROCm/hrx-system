// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/device/support/kernel.h"

IREE_AMDGPU_ATTRIBUTE_KERNEL void hostcall_buffer_test(uint64_t* output) {
  const iree_amdgpu_kernel_implicit_args_t* implicit_args =
      (const iree_amdgpu_kernel_implicit_args_t*)
          __builtin_amdgcn_implicitarg_ptr();
  output[0] = (uint64_t)(uintptr_t)implicit_args->hostcall_buffer;
}
