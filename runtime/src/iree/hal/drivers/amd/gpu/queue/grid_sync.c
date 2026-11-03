// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// The initializer consumes the same 48-byte AMDHSA grid record as its user
// kernel. Only the workgroup count at byte 40 is read; the preceding fields
// and arrival words have no meaning to hardware GWS initialization.
typedef struct {
  // Opaque prefix of the compiler's grid synchronization ABI.
  unsigned long long prefix[5];
  // Nonzero grid extent admitted by the host before publication.
  unsigned int workgroup_count;
  // Initialized ABI padding.
  unsigned int reserved;
} grid_arguments_t;

#if defined(__GFX9__) || defined(__GFX10__)
__attribute__((amdgpu_kernel, visibility("protected"), used, target("gws"),
               amdgpu_flat_work_group_size(
                   1, 1))) void initialize_grid_sync(grid_arguments_t grid) {
  __builtin_amdgcn_ds_gws_init(grid.workgroup_count - 1, 0);
}
#endif  // __GFX9__ || __GFX10__
