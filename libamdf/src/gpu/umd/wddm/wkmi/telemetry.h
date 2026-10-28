// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_GPU_UMD_WDDM_WKMI_TELEMETRY_H_
#define AMDF_SRC_GPU_UMD_WDDM_WKMI_TELEMETRY_H_

#include "libamdf/src/gpu/umd/wddm/wkmi/bridge_api.h"

namespace amdf::wkmi_bridge {

// Preserves the native snapshot and physical attribution without retaining a
// sensor session. The caller owns both KMT handles throughout this call.
amdf_status_t AMDF_WKMI_BRIDGE_CALL
GpuTelemetryRead(uint32_t adapter_handle, uint32_t device_handle,
                 uint32_t physical_adapter_index,
                 amdf_wkmi_bridge_gpu_telemetry_t* out_snapshot) noexcept;

}  // namespace amdf::wkmi_bridge

#endif  // AMDF_SRC_GPU_UMD_WDDM_WKMI_TELEMETRY_H_
