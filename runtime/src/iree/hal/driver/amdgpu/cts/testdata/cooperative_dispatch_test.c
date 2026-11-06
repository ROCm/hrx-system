// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

extern "C" __attribute__((device)) void __ockl_grid_sync(void);

// Publishes one value per workgroup, synchronizes the complete grid, and then
// has every workgroup observe the complete published set.
extern "C" __attribute__((global, visibility("protected"), used)) void
cooperative_dispatch_test(unsigned int* scratch, unsigned int* output,
                          unsigned int incarnation,
                          unsigned int workgroup_count) {
  const unsigned int workgroup_id = __builtin_amdgcn_workgroup_id_x();
  const unsigned int workitem_id = __builtin_amdgcn_workitem_id_x();
  if (workitem_id == 0) {
    scratch[workgroup_id] = incarnation + workgroup_id;
  }

  __ockl_grid_sync();

  if (workitem_id == 0) {
    unsigned int sum = 0;
    for (unsigned int i = 0; i < workgroup_count; ++i) {
      sum += scratch[i];
    }
    output[workgroup_id] = sum;
  }
}
