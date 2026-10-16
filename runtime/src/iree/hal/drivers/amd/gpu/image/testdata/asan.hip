// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// An actual HIP code object using the existing marked-global ABI. Assembly
// gives each exported object its exact logical size and trailing redzone. The
// HIP host-link identity is hidden because it is not shader-visible user data.

extern "C" {
extern __attribute__((device)) unsigned long long iree_asan_config[12];

// Observe shadow through the actual shader address mapping. This kernel itself
// is uninstrumented so it can inspect poison without producing a report.
__attribute__((global)) void asan_read_shadow(unsigned char* output,
                                              unsigned long long address,
                                              unsigned count) {
  asm volatile(
      ".hidden __hip_cuid_\n"
      ".pushsection .data\n"
      ".p2align 3\n"
      ".globl iree_asan_global_layout_v0\n"
      ".type iree_asan_global_layout_v0,@object\n"
      "iree_asan_global_layout_v0: .byte 0\n"
      ".size iree_asan_global_layout_v0,1\n"
      ".zero 15\n"
      ".globl iree_asan_config\n"
      ".type iree_asan_config,@object\n"
      "iree_asan_config: .zero 96\n"
      ".size iree_asan_config,96\n"
      ".zero 8\n"
      ".globl lookup_table.kd\n"
      ".type lookup_table.kd,@object\n"
      "lookup_table.kd: .long 7, 11, 13\n"
      ".size lookup_table.kd,12\n"
      ".zero 12\n"
      ".popsection\n");
  if (__builtin_amdgcn_workgroup_id_x() == 0 &&
      __builtin_amdgcn_workitem_id_x() == 0) {
    const unsigned shift = reinterpret_cast<unsigned*>(iree_asan_config)[3];
    const auto* shadow = reinterpret_cast<const unsigned char*>(
        iree_asan_config[2] + (address >> shift));
    for (unsigned i = 0; i < count; ++i) {
      output[i] = shadow[i];
    }
  }
}
}
