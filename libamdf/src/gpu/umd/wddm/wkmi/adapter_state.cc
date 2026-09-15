// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/gpu/umd/wddm/wkmi/adapter_state.h"

#include <cstdlib>

amdf_wkmi_bridge_gpu_adapter_t::~amdf_wkmi_bridge_gpu_adapter_t() {
  amdf_assert(live_queue_count == 0);
  std::free(device_info.adapter_info);
}

namespace amdf::wkmi_bridge {

amdf_wkmi_bridge_result_t PrepareGpuAdapterClose(
    amdf_wkmi_bridge_gpu_adapter_t* adapter,
    uint32_t* out_native_status) noexcept {
  if (adapter == nullptr || out_native_status == nullptr) {
    return AMDF_WKMI_BRIDGE_RESULT_INVALID_ARGUMENT;
  }
  *out_native_status = 0;
  AcquireSRWLockExclusive(&adapter->state_lock);
  if (adapter->live_queue_count != 0) {
    ReleaseSRWLockExclusive(&adapter->state_lock);
    return AMDF_WKMI_BRIDGE_RESULT_BUSY;
  }
  ReleaseSRWLockExclusive(&adapter->state_lock);
  return AMDF_WKMI_BRIDGE_RESULT_SUCCESS;
}

}  // namespace amdf::wkmi_bridge
