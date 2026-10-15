// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Real HIP globals exercise initialized storage, BSS and dynamic pointer
// relocation. Public visibility prevents internalization from erasing the
// module/global lookup contract exercised by HIP clients.
extern "C" {
__attribute__((device, used, visibility("default"))) unsigned image_value = 19;
__attribute__((device, used, visibility("default"))) unsigned image_zero[17];
__attribute__((device, used, visibility("default"))) unsigned* image_pointer =
    &image_value;

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
// on several queues with independent bindings and explicit intra-recording edges.
__attribute__((global)) void image_accumulate(unsigned* output, unsigned value) {
  if (__builtin_amdgcn_workgroup_id_x() == 0 &&
      __builtin_amdgcn_workitem_id_x() == 0) {
    output[0] += value;
  }
}
}
