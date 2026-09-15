// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

extern "C" __attribute__((device, managed)) unsigned int hrx_managed_value = 19;

// HIP device compilation marks this kernel as requiring uniform workgroups.
// The extended launch API must reject a partial final workgroup before it can
// write |output|.
extern "C" __attribute__((global)) void hrx_uniform_store_output(
    unsigned int* output, unsigned int value) {
  output[0] = value;
}

extern "C" __attribute__((global)) void hrx_increment_managed_value(
    unsigned int increment, unsigned int* observed_value) {
  if (__builtin_amdgcn_workgroup_id_x() == 0 &&
      __builtin_amdgcn_workitem_id_x() == 0) {
    hrx_managed_value += increment;
    observed_value[0] = hrx_managed_value;
  }
}
