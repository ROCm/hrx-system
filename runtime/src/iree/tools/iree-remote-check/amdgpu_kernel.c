// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#define IREE_REMOTE_CHECK_KERNEL \
  [[clang::amdgpu_kernel, gnu::visibility("protected")]]

IREE_REMOTE_CHECK_KERNEL void iree_remote_check_add7(const int* input,
                                                     int* output, int addend) {
  for (unsigned int i = 0; i < 4; ++i) {
    output[i] = input[i] + addend + 7;
  }
}
