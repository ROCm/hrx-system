// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdint.h>

__attribute__((amdgpu_kernel, visibility("protected"),
               amdgpu_flat_work_group_size(256, 256))) void
blit(const uint64_t* src, uint64_t* dst, uint64_t count) {
  uint64_t x = (uint64_t)__builtin_amdgcn_workgroup_id_x() * 256 +
               __builtin_amdgcn_workitem_id_x();
  // Grid has a fixed, explicitly supplied 1024 groups.
  for (; x < count; x += 256 * 1024) dst[x] = src[x];
}
