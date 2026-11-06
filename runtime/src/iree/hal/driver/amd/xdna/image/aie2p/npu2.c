// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/driver/amd/xdna/image/aie2p/npu2.h"

// Image execution contracts shared by compatible NPU2 devices.
static const struct {
  // Exact passive endpoint key, independent of native platform transport.
  iree_string_view_t target_id;
  // Compiler-owned identity serialized in the image metadata.
  uint64_t device_profile_id;
} iree_hal_amd_xdna_aie2p_npu2_profiles[] = {
    {IREE_SVL("amd.xdna.strix.17f0_10"), UINT64_C(0x5354524958000001)},
    // AMD npu6_regs.c selects NPU4 firmware, hardware operations, feature
    // contracts and device-memory layout. Krackan uses that same image ABI.
    {IREE_SVL("amd.xdna.krackan.17f0_20"), UINT64_C(0x5354524958000001)},
    {IREE_SVL("amd.xdna.strix_halo.17f0_11"), UINT64_C(0x535848414C4F0001)},
};

iree_status_t iree_hal_amd_xdna_aie2p_npu2_target_initialize(
    iree_string_view_t target_id, uint16_t context_column_count,
    iree_hal_amd_xdna_aie2p_target_t* out_target) {
  IREE_ASSERT_ARGUMENT(out_target);
  uint64_t device_profile_id = 0;
  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(iree_hal_amd_xdna_aie2p_npu2_profiles); ++i) {
    if (iree_string_view_equal(
            target_id, iree_hal_amd_xdna_aie2p_npu2_profiles[i].target_id)) {
      device_profile_id =
          iree_hal_amd_xdna_aie2p_npu2_profiles[i].device_profile_id;
      break;
    }
  }
  if (device_profile_id == 0) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "no NPU2 image target for '%.*s'",
                            (int)target_id.size, target_id.data);
  }
  if (context_column_count == 0 || context_column_count > 8) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "NPU2 XDNA context column count must be in [1, 8]");
  }
  *out_target = (iree_hal_amd_xdna_aie2p_target_t){
      .identity =
          {
              .device_profile_revision = 1,
              .device_profile_id = device_profile_id,
              .firmware_abi_id = UINT64_C(0x4E5055320006000C),
          },
      .context = {.column_count = context_column_count, .row_count = 6},
      .instruction_alignment = 32u * 1024u,
  };
  return iree_ok_status();
}
