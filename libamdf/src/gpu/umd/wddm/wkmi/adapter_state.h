// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_GPU_UMD_WDDM_WKMI_ADAPTER_STATE_H_
#define AMDF_SRC_GPU_UMD_WDDM_WKMI_ADAPTER_STATE_H_

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif  // WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif  // NOMINMAX
#ifndef WIN32_NO_STATUS
#define WIN32_NO_STATUS
#define AMDF_WKMI_BRIDGE_UNDEFINE_WIN32_NO_STATUS
#endif  // WIN32_NO_STATUS
#include <windows.h>
#include <winternl.h>
#ifdef AMDF_WKMI_BRIDGE_UNDEFINE_WIN32_NO_STATUS
#undef WIN32_NO_STATUS
#undef AMDF_WKMI_BRIDGE_UNDEFINE_WIN32_NO_STATUS
#endif  // AMDF_WKMI_BRIDGE_UNDEFINE_WIN32_NO_STATUS

#include <d3dkmthk.h>
#include <ntstatus.h>

#include "libamdf/src/gpu/umd/wddm/wkmi/bridge_api.h"
#include "wkmi.h"

// Parsed adapter state shared by live bridge objects.
struct amdf_wkmi_bridge_gpu_adapter_t {
  // Parent-instance allocator copied for all bridge-controlled storage.
  amdf_allocator_t host_allocator = {};
  // Private normalized adapter properties owned by WKMI.
  Wkmi::DeviceInfo device_info = {};
  // Protects live-child accounting.
  SRWLOCK state_lock = SRWLOCK_INIT;
  // Number of live queues borrowing this adapter.
  uint32_t live_queue_count = 0;

  ~amdf_wkmi_bridge_gpu_adapter_t();
};

namespace amdf::wkmi_bridge {

// Verifies that no live child borrows `adapter`.
amdf_wkmi_bridge_result_t PrepareGpuAdapterClose(
    amdf_wkmi_bridge_gpu_adapter_t* adapter,
    uint32_t* out_native_status) noexcept;

}  // namespace amdf::wkmi_bridge

#endif  // AMDF_SRC_GPU_UMD_WDDM_WKMI_ADAPTER_STATE_H_
