// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/device/support/kernel.h"

#define GETREG_IMMED(size, offset, reg) \
  ((((size) - 1u) << 11) | ((offset) << 6) | (reg))

IREE_AMDGPU_ATTRIBUTE_KERNEL void execution_queue_mask_test(uint32_t* output) {
  if (iree_hal_amdgpu_device_local_id_x() != 0 ||
      iree_hal_amdgpu_device_local_id_y() != 0 ||
      iree_hal_amdgpu_device_local_id_z() != 0) {
    return;
  }

#if defined(__gfx942__) || defined(__gfx9_4_generic__)
  // Match HIP's gfx942 __smid decoding. HW_REG_HW_ID is register 4 and
  // provides CU_ID[11:8] and SE_ID[14:13]. HW_REG_XCC_ID is register 20 and
  // provides XCC_ID[3:0]. gfx942 has one shader array per shader engine, so
  // (XCC, SE, CU) uniquely identifies the executing compute unit.
  const uint32_t cu_id = __builtin_amdgcn_s_getreg(GETREG_IMMED(4, 8, 4));
  const uint32_t se_id = __builtin_amdgcn_s_getreg(GETREG_IMMED(2, 13, 4));
  const uint32_t xcc_id = __builtin_amdgcn_s_getreg(GETREG_IMMED(4, 0, 20));
  const uint32_t execution_unit_id = (xcc_id << 6) | (se_id << 4) | cu_id;
#else
  // The host test executes only on gfx942. Retaining a definition for other
  // target families keeps the shared CTS testdata aggregate buildable.
  const uint32_t execution_unit_id = UINT32_MAX;
#endif

  output[iree_hal_amdgpu_device_group_id_x()] = execution_unit_id;
}
