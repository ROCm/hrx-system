// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "binding/hip/device_properties.h"

#include "common/amdgpu_architecture.h"

bool iree_hip_parse_gcn_arch_name(const char* name, int* out_architecture) {
  if (!out_architecture) return false;
  iree_hal_streaming_amdgpu_architecture_t architecture = {0};
  if (!iree_hal_streaming_parse_amdgpu_architecture(name, &architecture)) {
    return false;
  }
  *out_architecture = (int)(architecture.major * 100 + architecture.minor * 10 +
                            architecture.stepping);
  return true;
}
