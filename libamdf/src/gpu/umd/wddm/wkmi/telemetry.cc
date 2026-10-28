// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/gpu/umd/wddm/wkmi/telemetry.h"

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#define WIN32_NO_STATUS
#include <windows.h>
#include <winternl.h>
#undef WIN32_NO_STATUS

#include <d3dkmthk.h>
#include <ntstatus.h>

#include "wkmi.h"

namespace amdf::wkmi_bridge {

amdf_status_t AMDF_WKMI_BRIDGE_CALL
GpuTelemetryRead(uint32_t adapter_handle, uint32_t device_handle,
                 uint32_t physical_adapter_index,
                 amdf_wkmi_bridge_gpu_telemetry_t* out_snapshot) noexcept {
  if (adapter_handle == 0 || device_handle == 0 || out_snapshot == nullptr) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  if (physical_adapter_index != 0) {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
  D3DKMT_PHYSICAL_ADAPTER_COUNT physical_count = {};
  D3DKMT_QUERYADAPTERINFO query = {};
  query.hAdapter = adapter_handle;
  query.Type = KMTQAITYPE_PHYSICALADAPTERCOUNT;
  query.pPrivateDriverData = &physical_count;
  query.PrivateDriverDataSize = sizeof(physical_count);
  NTSTATUS status = D3DKMTQueryAdapterInfo(&query);
  if (status != STATUS_SUCCESS) {
    return amdf_make_status(AMDF_STATUS_DOMAIN_NTSTATUS,
                            static_cast<uint32_t>(status));
  }
  if (physical_count.Count != 1) {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
  Wkmi::PmlogQueryResult native = {};
  status = Wkmi::QueryPMLogData(adapter_handle, device_handle, &native);
  if (status == STATUS_NOT_SUPPORTED || status == STATUS_NOT_IMPLEMENTED) {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
  if (status != STATUS_SUCCESS) {
    return amdf_make_status(AMDF_STATUS_DOMAIN_NTSTATUS,
                            static_cast<uint32_t>(status));
  }
  static_assert(Wkmi::kPmlogMaxSensors == 256);
  amdf_wkmi_bridge_gpu_telemetry_t snapshot = {};
  for (uint32_t i = 0; i < Wkmi::kPmlogMaxSensors; ++i) {
    snapshot.supported[i] = native.supported[i] ? 1 : 0;
    snapshot.values[i] = native.value[i];
  }
  *out_snapshot = snapshot;
  return AMDF_STATUS_OK;
}

}  // namespace amdf::wkmi_bridge
