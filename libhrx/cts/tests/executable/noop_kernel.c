// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#define HRX_CTS_TEST_ATTRIBUTE_KERNEL                                           \
  [[clang::amdgpu_kernel, gnu::visibility("protected")]]

HRX_CTS_TEST_ATTRIBUTE_KERNEL void hrx_noop(void) {}
