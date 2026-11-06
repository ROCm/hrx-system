// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#define IREE_PM4_TEST_ATTRIBUTE_KERNEL \
  [[clang::amdgpu_kernel, gnu::visibility("protected")]]

typedef unsigned int uint32_t;
typedef unsigned long uint64_t;

// Static LDS has no defined initial contents; loader_uninitialized emits an
// undef addrspace(3) global that AMDGPU lld accepts and still reserves group
// segment storage for the kernel descriptor.
static __attribute__((address_space(3), loader_uninitialized)) uint32_t
    iree_hal_amdgpu_pm4_dispatch_test_lds_values[64];

IREE_PM4_TEST_ATTRIBUTE_KERNEL void iree_hal_amdgpu_pm4_dispatch_test_store_a(
    uint32_t* target, uint32_t value) {
  target[0] = value;
}

IREE_PM4_TEST_ATTRIBUTE_KERNEL void iree_hal_amdgpu_pm4_dispatch_test_store_b(
    uint32_t* target, uint32_t value) {
  target[0] = value + 0x100u;
}

IREE_PM4_TEST_ATTRIBUTE_KERNEL void iree_hal_amdgpu_pm4_dispatch_test_read_add(
    uint32_t* source, uint32_t* target, uint32_t value) {
  target[0] = source[0] + value;
}

IREE_PM4_TEST_ATTRIBUTE_KERNEL void
iree_hal_amdgpu_pm4_dispatch_test_patch_user_data(uint32_t* target_dwords,
                                                  uint32_t dword_offset,
                                                  uint64_t kernarg_address) {
  target_dwords[dword_offset + 0] = (uint32_t)kernarg_address;
  target_dwords[dword_offset + 1] = (uint32_t)(kernarg_address >> 32);
}

IREE_PM4_TEST_ATTRIBUTE_KERNEL void iree_hal_amdgpu_pm4_dispatch_test_lds_sum(
    uint32_t* target, uint32_t value) {
  const uint32_t local_id = __builtin_amdgcn_workitem_id_x();
  iree_hal_amdgpu_pm4_dispatch_test_lds_values[local_id] = value + local_id;
  __builtin_amdgcn_s_barrier();
  if (local_id == 0) {
    uint32_t sum = 0;
    for (uint32_t i = 0; i < 64; ++i) {
      sum += iree_hal_amdgpu_pm4_dispatch_test_lds_values[i];
    }
    target[0] = sum;
  }
}
