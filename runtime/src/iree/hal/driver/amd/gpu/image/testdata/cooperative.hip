// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

extern "C" __attribute__((device)) void __ockl_grid_sync(void);
extern "C" __attribute__((device)) unsigned int __ockl_grid_bar_arrive(void);
extern "C" __attribute__((device)) void __ockl_grid_bar_wait(unsigned int token);

// Publishes one value per workgroup, synchronizes the complete grid, and has
// every workgroup consume the complete published set. The result accumulates
// into |output| so an earlier stream operation remains an observable ordering
// dependency.
template <bool split_barrier>
__attribute__((device, always_inline)) inline void cooperative_grid_sync(
    unsigned int* scratch, unsigned int* output, unsigned int incarnation,
    unsigned int workgroup_count) {
  const unsigned int workgroup_id = __builtin_amdgcn_workgroup_id_x();
  const unsigned int workitem_id = __builtin_amdgcn_workitem_id_x();
  if (workitem_id == 0) {
    scratch[workgroup_id] = incarnation + workgroup_id;
  }

  if constexpr (split_barrier) {
    const unsigned int token = __ockl_grid_bar_arrive();
    __ockl_grid_bar_wait(token);
  } else {
    __ockl_grid_sync();
  }

  if (workitem_id == 0) {
    unsigned int sum = 0;
    for (unsigned int i = 0; i < workgroup_count; ++i) {
      sum += scratch[i];
    }
    output[workgroup_id] += sum;
  }
}

extern "C" __attribute__((global, visibility("protected"), used)) void
hrx_cooperative_grid_sync(unsigned int* scratch, unsigned int* output,
                          unsigned int incarnation,
                          unsigned int workgroup_count) {
  cooperative_grid_sync<false>(scratch, output, incarnation, workgroup_count);
}

extern "C" __attribute__((global, visibility("protected"), used)) void
hrx_cooperative_grid_sync_split(unsigned int* scratch, unsigned int* output,
                                unsigned int incarnation,
                                unsigned int workgroup_count) {
  cooperative_grid_sync<true>(scratch, output, incarnation, workgroup_count);
}
