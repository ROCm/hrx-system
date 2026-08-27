// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/util/aql_ring.h"

iree_status_t iree_hal_amdgpu_query_aql_queue_execution_mode(
    const iree_hal_amdgpu_libhsa_t* libhsa, hsa_agent_t device_agent,
    iree_hal_amdgpu_aql_queue_execution_mode_t* out_execution_mode) {
  IREE_ASSERT_ARGUMENT(libhsa);
  IREE_ASSERT_ARGUMENT(out_execution_mode);

  bool pm4_emulation = false;
  IREE_RETURN_IF_ERROR(
      iree_hsa_agent_get_info(
          IREE_LIBHSA(libhsa), device_agent,
          (hsa_agent_info_t)HSA_AMD_AGENT_INFO_PM4_EMULATION, &pm4_emulation),
      "querying HSA_AMD_AGENT_INFO_PM4_EMULATION");
  *out_execution_mode =
      pm4_emulation ? IREE_HAL_AMDGPU_AQL_QUEUE_EXECUTION_MODE_PM4_EMULATED
                    : IREE_HAL_AMDGPU_AQL_QUEUE_EXECUTION_MODE_NATIVE;
  return iree_ok_status();
}
