// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/gpu/umd/dxg/rocdxg/adapter_state.h"

#include <cassert>

amdf_dxg_bridge_gpu_adapter_t::~amdf_dxg_bridge_gpu_adapter_t() {
  assert(live_queue_count.load(std::memory_order_relaxed) == 0);
  delete device;
  delete chain;
}

namespace amdf::rocdxg_bridge {

amdf_dxg_bridge_result_t PrepareGpuAdapterClose(
    amdf_dxg_bridge_gpu_adapter_t* adapter,
    uint32_t* out_native_status) noexcept {
  if (adapter == nullptr || out_native_status == nullptr) {
    return AMDF_DXG_BRIDGE_RESULT_INVALID_ARGUMENT;
  }
  *out_native_status = 0;
  return adapter->live_queue_count.load(std::memory_order_acquire) == 0
             ? AMDF_DXG_BRIDGE_RESULT_SUCCESS
             : AMDF_DXG_BRIDGE_RESULT_BUSY;
}

}  // namespace amdf::rocdxg_bridge
