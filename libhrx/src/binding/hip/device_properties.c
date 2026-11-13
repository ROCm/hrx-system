// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "binding/hip/device_properties.h"

#include <string.h>

#include "common/amdgpu_architecture.h"

bool iree_hip_parse_gcn_arch_name(const char* name, int* out_architecture) {
  if (!out_architecture) {
    return false;
  }
  iree_hal_streaming_amdgpu_architecture_t architecture = {0};
  if (!iree_hal_streaming_parse_amdgpu_architecture(name, &architecture)) {
    return false;
  }
  *out_architecture = (int)(architecture.major * 100 + architecture.minor * 10 +
                            architecture.stepping);
  return true;
}

void iree_hip_format_device_uuid(iree_string_view_t driver_id,
                                 const iree_hal_uuid_t* source,
                                 uint8_t out_bytes[16]) {
  if (iree_string_view_equal(driver_id, IREE_SV("amd"))) {
    static const char digits[] = "0123456789abcdef";
    for (uint32_t i = 0; i < 8; ++i) {
      out_bytes[i * 2] = digits[source->bytes[i] >> 4];
      out_bytes[i * 2 + 1] = digits[source->bytes[i] & 0xF];
    }
  } else {
    memcpy(out_bytes, source->bytes, sizeof(source->bytes));
  }
}
