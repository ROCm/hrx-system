// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_UTIL_SDMA_CAPABILITIES_H_
#define IREE_HAL_DRIVERS_AMDGPU_UTIL_SDMA_CAPABILITIES_H_

#include "iree/base/api.h"
#include "iree/hal/drivers/amdgpu/target/identity.h"

#ifdef __cplusplus
extern "C" {
#endif

// Packet encoding profile, not a graphics ISA or engine ordinal.
// Add profiles only with independently verified packet and visibility rules.
typedef enum iree_hal_amdgpu_sdma_profile_e {
  IREE_HAL_AMDGPU_SDMA_PROFILE_NONE = 0,
  IREE_HAL_AMDGPU_SDMA_PROFILE_GFX12_0 = 1,
} iree_hal_amdgpu_sdma_profile_t;

typedef struct iree_hal_amdgpu_sdma_capabilities_t {
  iree_hal_amdgpu_sdma_profile_t profile;
  uint32_t max_copy_bytes;
  bool supports_timestamps;
  // Whether indirect execution is supported by the user-queue path. Packet
  // encoding alone is insufficient: KFD must also enable IB processing.
  bool supports_indirect_buffers;
} iree_hal_amdgpu_sdma_capabilities_t;

// Leaves output unchanged for unknown targets. Queue availability is queried
// separately when creating a native queue.
iree_status_t iree_hal_amdgpu_sdma_query_capabilities(
    iree_hal_amdgpu_gfxip_version_t version,
    iree_hal_amdgpu_sdma_capabilities_t* out_capabilities);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_UTIL_SDMA_CAPABILITIES_H_
