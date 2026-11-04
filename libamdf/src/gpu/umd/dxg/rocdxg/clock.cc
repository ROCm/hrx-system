// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/gpu/umd/dxg/rocdxg/clock.h"

#include <sys/ioctl.h>
#include <time.h>

#include "libamdf/src/gpu/umd/dxg/rocdxg/adapter_state.h"
#include "shared/include/d3dkmt_types.h"

namespace amdf::rocdxg_bridge {
namespace {

constexpr uint32_t kQueryClockCalibrationOperation = 0x3D;

bool ReadHostClock(uint64_t* out_timestamp) {
  timespec now = {};
  if (clock_gettime(CLOCK_MONOTONIC_RAW, &now) != 0) {
    return false;
  }
  *out_timestamp = static_cast<uint64_t>(now.tv_sec) * 1'000'000'000u +
                   static_cast<uint64_t>(now.tv_nsec);
  return true;
}

}  // namespace

amdf_dxg_bridge_result_t AMDF_DXG_BRIDGE_CALL
GpuClockSample(amdf_dxg_bridge_gpu_adapter_t* adapter, uint32_t adapter_handle,
               uint32_t physical_adapter_index,
               amdf_dxg_bridge_gpu_queue_command_type_t command_type,
               amdf_dxg_bridge_gpu_clock_sample_t* out_sample,
               uint32_t* out_native_status) noexcept {
  if (adapter == nullptr || adapter_handle == 0 || out_sample == nullptr ||
      out_native_status == nullptr) {
    return AMDF_DXG_BRIDGE_RESULT_INVALID_ARGUMENT;
  }
  *out_native_status = 0;
  const amdf_rocdxg_kernel_queue_profile_t* profile =
      SelectGpuKernelQueueProfile(adapter, command_type);
  if (profile == nullptr || !profile->is_supported) {
    return AMDF_DXG_BRIDGE_RESULT_UNSUPPORTED;
  }

  uint64_t host_before = 0;
  if (!ReadHostClock(&host_before)) {
    return AMDF_DXG_BRIDGE_RESULT_INTERNAL;
  }
  D3DKMT_QUERYCLOCKCALIBRATION query = {
      .hAdapter = static_cast<D3DKMT_HANDLE>(adapter_handle),
      .NodeOrdinal = profile->node_ordinal,
      .PhysicalAdapterIndex = physical_adapter_index,
  };
  const NTSTATUS status =
      static_cast<NTSTATUS>(adapter->native_transport.invoke(
          adapter->native_transport.context,
          _IOWR('G', kQueryClockCalibrationOperation,
                D3DKMT_QUERYCLOCKCALIBRATION),
          &query));
  if (status == STATUS_NOT_SUPPORTED) {
    return AMDF_DXG_BRIDGE_RESULT_UNSUPPORTED;
  }
  if (status != STATUS_SUCCESS) {
    *out_native_status = static_cast<uint32_t>(status);
    return AMDF_DXG_BRIDGE_RESULT_NATIVE_FAILURE;
  }
  if (query.ClockData.Flags.ContextManagementProcessor ||
      query.ClockData.GpuFrequency == 0) {
    return AMDF_DXG_BRIDGE_RESULT_UNSUPPORTED;
  }
  uint64_t host_after = 0;
  if (!ReadHostClock(&host_after) || host_after < host_before) {
    return AMDF_DXG_BRIDGE_RESULT_INTERNAL;
  }
  // DXG returns a Windows QPC sample, which does not share Linux's boot epoch.
  // The native GPU/QPC pair was sampled during the ioctl, so project that
  // observation into the tighter Linux-host bracket. The public outer bracket
  // retains the complete call latency and bounds this midpoint estimate.
  const uint64_t host_timestamp = host_before + (host_after - host_before) / 2;
  *out_sample = {
      query.ClockData.GpuFrequency,
      query.ClockData.GpuClockCounter,
      host_timestamp,
  };
  return AMDF_DXG_BRIDGE_RESULT_SUCCESS;
}

}  // namespace amdf::rocdxg_bridge
