// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_GPU_UMD_DXG_ROCDXG_ADAPTER_STATE_H_
#define AMDF_SRC_GPU_UMD_DXG_ROCDXG_ADAPTER_STATE_H_

#include <atomic>

#include "libamdf/src/gpu/umd/dxg/bridge_api.h"
#include "shared/include/thunk_proxy/thunk_proxy.h"

// Immutable native scheduler facts for one kernel-queue command family.
struct amdf_rocdxg_kernel_queue_profile_t {
  // Native scheduler identifier encoded into WKMI context private data.
  uint32_t scheduler_id = 0;
  // Node ordinal passed to context creation.
  uint32_t node_ordinal = 0;
  // Whether this command family has a usable scheduler.
  bool is_supported = false;
  // Whether the scheduler requires a hardware-queue object.
  bool uses_hardware_scheduling = false;
};

// Private ROCDXG state retained for one live bridge adapter.
struct amdf_dxg_bridge_gpu_adapter_t {
  // Parent-instance allocator copied for all bridge-controlled storage.
  amdf_allocator_t host_allocator = {};
  // Borrowed native transport installed only during synchronous bridge calls.
  amdf_dxg_bridge_native_transport_t native_transport = {};
  // Private topology context owned by this adapter.
  thunk_proxy::ChainContext* chain = nullptr;
  // Private physical-device context owned by this adapter.
  thunk_proxy::DeviceContext* device = nullptr;
  // Whether context payloads enable firmware-managed graphics state.
  bool state_shadowing_by_cpfw = false;
  // Qualified immutable-stream compute scheduler.
  amdf_rocdxg_kernel_queue_profile_t pm4_queue_profile = {};
  // Qualified immutable-stream SDMA scheduler.
  amdf_rocdxg_kernel_queue_profile_t sdma_queue_profile = {};
  // Number of live queues borrowing this adapter.
  std::atomic<uint32_t> live_queue_count = 0;

  ~amdf_dxg_bridge_gpu_adapter_t();
};

namespace amdf::rocdxg_bridge {

// Verifies that no live queue borrows `adapter`.
amdf_dxg_bridge_result_t PrepareGpuAdapterClose(
    amdf_dxg_bridge_gpu_adapter_t* adapter,
    uint32_t* out_native_status) noexcept;

}  // namespace amdf::rocdxg_bridge

#endif  // AMDF_SRC_GPU_UMD_DXG_ROCDXG_ADAPTER_STATE_H_
