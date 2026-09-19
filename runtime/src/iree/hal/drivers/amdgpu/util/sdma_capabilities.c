// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/util/sdma_capabilities.h"

iree_status_t iree_hal_amdgpu_sdma_query_capabilities(
    iree_hal_amdgpu_gfxip_version_t version,
    iree_hal_amdgpu_sdma_capabilities_t* out_capabilities) {
  IREE_ASSERT_ARGUMENT(out_capabilities);
  if (version.major != 12 || version.minor != 0 || version.stepping > 1) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "no SDMA encoding profile for gfx IP %u.%u.%u",
                            version.major, version.minor, version.stepping);
  }
  *out_capabilities = (iree_hal_amdgpu_sdma_capabilities_t){
      .profile = IREE_HAL_AMDGPU_SDMA_PROFILE_GFX12_0,
      .max_copy_bytes = 1u << 22,
      .supports_timestamps = true,
      // TODO: Enable once KFD sets IB_ENABLE for user SDMA queues and the
      // supported kernel/runtime combination provides a qualified IB contract.
      .supports_indirect_buffers = false,
  };
  return iree_ok_status();
}
