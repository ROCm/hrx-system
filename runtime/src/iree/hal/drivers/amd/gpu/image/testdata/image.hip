// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Real HIP globals exercise initialized storage, BSS and dynamic pointer
// relocation. Public visibility prevents internalization from erasing the
// module/global lookup contract exercised by HIP clients.
extern "C" {
__attribute__((device, const)) __SIZE_TYPE__ __ockl_get_num_groups(unsigned);
__attribute__((device, const)) unsigned __ockl_get_work_dim();
__attribute__((device, used, visibility("default"))) unsigned image_value = 19;
__attribute__((device, used, visibility("default"))) unsigned image_zero[17];
__attribute__((device, used, visibility("default"))) unsigned* image_pointer =
    &image_value;
// Ordinary user storage has no runtime service semantics. Admission tests
// rename this symbol to exercise each reserved runtime configuration name.
__attribute__((device, used,
               visibility("default"))) unsigned image_runtime_config[16];

__attribute__((global)) void image_transform(unsigned* output, unsigned value) {
  if (__builtin_amdgcn_workgroup_id_x() == 0 &&
      __builtin_amdgcn_workitem_id_x() == 0) {
    output[0] = value * 3 + *image_pointer;
    image_zero[0] = value;
  }
}

__attribute__((global)) void image_increment(unsigned* output, unsigned value) {
  if (__builtin_amdgcn_workgroup_id_x() == 0 &&
      __builtin_amdgcn_workitem_id_x() == 0) {
    *image_pointer += value;
    output[0] = *image_pointer;
  }
}

// Invocation-private state allows one immutable recording to run concurrently
// on several queues with independent bindings and explicit intra-recording
// edges.
__attribute__((global)) void image_accumulate(unsigned* output,
                                              unsigned value) {
  if (__builtin_amdgcn_workgroup_id_x() == 0 &&
      __builtin_amdgcn_workitem_id_x() == 0) {
    output[0] += value;
  }
}

// Device-produced launch parameters exercise the producer-to-CP boundary.
__attribute__((global)) void image_counts(unsigned* counts,
                                          const unsigned* source) {
  if (__builtin_amdgcn_workitem_id_x() == 0) {
    counts[0] = source[0];
    counts[1] = source[1];
    counts[2] = source[2];
  }
}

// Fixed-stride records expose the actual set of launched groups independently
// of the shader's hidden count inputs. Each group has one writing work-item.
__attribute__((global)) void image_geometry(unsigned* output, unsigned marker) {
  if (__builtin_amdgcn_workitem_id_x() == 0 &&
      __builtin_amdgcn_workitem_id_y() == 0 &&
      __builtin_amdgcn_workitem_id_z() == 0) {
    const unsigned group = __builtin_amdgcn_workgroup_id_x() +
                           4 * __builtin_amdgcn_workgroup_id_y() +
                           16 * __builtin_amdgcn_workgroup_id_z();
    output += group * 8;
    output[0] = (unsigned)__ockl_get_num_groups(0);
    output[1] = (unsigned)__ockl_get_num_groups(1);
    output[2] = (unsigned)__ockl_get_num_groups(2);
    output[3] = __builtin_amdgcn_workgroup_size_x();
    output[4] = __builtin_amdgcn_workgroup_size_y();
    output[5] = __builtin_amdgcn_workgroup_size_z();
    output[6] = marker;
    output[7] = __ockl_get_work_dim();
  }
}

// Observe the shader ABI directly, independently of how the runtime encodes its
// command queue. Only one invocation writes the complete packet and its
// address.
__attribute__((global)) void image_dispatch_packet(unsigned* output,
                                                   unsigned marker) {
  if (__builtin_amdgcn_workgroup_id_x() == 0 &&
      __builtin_amdgcn_workgroup_id_y() == 0 &&
      __builtin_amdgcn_workgroup_id_z() == 0 &&
      __builtin_amdgcn_workitem_id_x() == 0 &&
      __builtin_amdgcn_workitem_id_y() == 0 &&
      __builtin_amdgcn_workitem_id_z() == 0) {
    const __attribute__((address_space(4))) unsigned* packet =
        (const __attribute__((
            address_space(4))) unsigned*)__builtin_amdgcn_dispatch_ptr();
    for (unsigned i = 0; i < 16; ++i) {
      output[i] = packet[i];
    }
    const unsigned long long address = (unsigned long long)packet;
    output[16] = (unsigned)address;
    output[17] = (unsigned)(address >> 32);
    output[18] = marker;
    const unsigned long long object =
        (unsigned long long)packet[8] | ((unsigned long long)packet[9] << 32);
    const __attribute__((address_space(4))) unsigned* descriptor =
        (const __attribute__((address_space(4))) unsigned*)object;
    output[19] = descriptor[0];
    output[20] = descriptor[1];
    output[21] = descriptor[2];
  }
}
}
