// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Exercises one-shot sensor queries against the installed driver. Each AMD
// adapter owns one device for all reads; no queue, GPU allocation, sampling
// session, or power-policy change is involved. Values are printed in native
// units so this test does not depend on a higher-level interpretation.

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#define WIN32_NO_STATUS
#include <windows.h>
#include <winternl.h>
#undef WIN32_NO_STATUS

#include <d3dkmthk.h>
#include <ntstatus.h>

#include <cinttypes>
#include <cstdio>
#include <vector>

#include "wkmi.h"

namespace {

bool CheckStatus(const char* operation, NTSTATUS status) {
  if (status == STATUS_SUCCESS) {
    return true;
  }
  std::fprintf(stderr, "%s failed: NTSTATUS=0x%08" PRIx32 "\n", operation,
               static_cast<uint32_t>(status));
  return false;
}

bool QueryTelemetry(const D3DKMT_ADAPTERINFO& adapter,
                    uint32_t& sampled_count) {
  D3DKMT_ADAPTERTYPE adapter_type = {};
  D3DKMT_QUERYADAPTERINFO query = {};
  query.hAdapter = adapter.hAdapter;
  query.Type = KMTQAITYPE_ADAPTERTYPE;
  query.pPrivateDriverData = &adapter_type;
  query.PrivateDriverDataSize = sizeof(adapter_type);
  if (!CheckStatus("QueryAdapterInfo(ADAPTERTYPE)",
                   D3DKMTQueryAdapterInfo(&query))) {
    return false;
  }
  if (adapter_type.SoftwareDevice) {
    return true;
  }

  D3DKMT_QUERY_DEVICE_IDS device_ids = {};
  query.Type = KMTQAITYPE_PHYSICALADAPTERDEVICEIDS;
  query.pPrivateDriverData = &device_ids;
  query.PrivateDriverDataSize = sizeof(device_ids);
  if (!CheckStatus("QueryAdapterInfo(PHYSICALADAPTERDEVICEIDS)",
                   D3DKMTQueryAdapterInfo(&query))) {
    return false;
  }
  if (device_ids.DeviceIds.VendorID != 0x1002) {
    return true;
  }

  D3DKMT_PHYSICAL_ADAPTER_COUNT physical_count = {};
  query.Type = KMTQAITYPE_PHYSICALADAPTERCOUNT;
  query.pPrivateDriverData = &physical_count;
  query.PrivateDriverDataSize = sizeof(physical_count);
  if (!CheckStatus("QueryAdapterInfo(PHYSICALADAPTERCOUNT)",
                   D3DKMTQueryAdapterInfo(&query))) {
    return false;
  }
  std::printf("Adapter LUID=%08" PRIx32 ":%08" PRIx32 " PCI=%04" PRIx32
              ":%04" PRIx32 " physical_count=%u\n",
              static_cast<uint32_t>(adapter.AdapterLuid.HighPart),
              static_cast<uint32_t>(adapter.AdapterLuid.LowPart),
              static_cast<uint32_t>(device_ids.DeviceIds.VendorID),
              static_cast<uint32_t>(device_ids.DeviceIds.DeviceID),
              physical_count.Count);

  // The native sensor signature has no physical-adapter selector. A single
  // physical adapter gives these observations an unambiguous device identity.
  if (physical_count.Count != 1) {
    std::printf(
        "SKIP: sensor routing within a linked adapter is unspecified\n");
    return true;
  }
  D3DKMT_CREATEDEVICE create = {};
  create.hAdapter = adapter.hAdapter;
  if (!CheckStatus("CreateDevice", D3DKMTCreateDevice(&create))) {
    return false;
  }

  uint16_t supported_sensors[Wkmi::kPmlogMaxSensors] = {};
  NTSTATUS status = Wkmi::QueryPMLogSupport(adapter.hAdapter, create.hDevice,
                                            supported_sensors);
  bool success = true;
  if (status == STATUS_NOT_SUPPORTED || status == STATUS_NOT_IMPLEMENTED) {
    std::printf("SKIP: PMLog support query unavailable: 0x%08" PRIx32 "\n",
                static_cast<uint32_t>(status));
  } else {
    success = CheckStatus("QueryPMLogSupport", status);
    if (success) {
      std::printf("Supported native sensor IDs:");
      for (uint16_t sensor : supported_sensors) {
        if (sensor != 0) {
          std::printf(" %u", sensor);
        }
      }
      std::printf("\n");
    }
    for (unsigned sample = 0; success && sample < 3; ++sample) {
      Wkmi::PmlogQueryResult readings = {};
      LARGE_INTEGER before, after;
      success = QueryPerformanceCounter(&before) != 0;
      if (success) {
        status =
            Wkmi::QueryPMLogData(adapter.hAdapter, create.hDevice, &readings);
        success = CheckStatus("QueryPMLogData", status);
      }
      if (success) {
        success = QueryPerformanceCounter(&after) != 0;
      }
      if (!success && status == STATUS_SUCCESS) {
        std::fprintf(stderr, "QueryPerformanceCounter failed: Win32=%lu\n",
                     GetLastError());
      }
      if (success) {
        std::printf("Sample %u qpc_before=%" PRId64 " qpc_after=%" PRId64 "\n",
                    sample, before.QuadPart, after.QuadPart);
        for (uint32_t sensor = 0; sensor < Wkmi::kPmlogMaxSensors; ++sensor) {
          if (readings.supported[sensor]) {
            std::printf("  id=%" PRIu32 " supported=1 value=%" PRIu32
                        " hex=%08" PRIx32 "\n",
                        sensor, readings.value[sensor], readings.value[sensor]);
          } else if (readings.value[sensor] != Wkmi::kSensorUnavailable) {
            std::fprintf(stderr,
                         "Unsupported sensor %" PRIu32
                         " lacks the documented sentinel\n",
                         sensor);
            success = false;
          }
        }
      }
    }
    if (success) {
      ++sampled_count;
    }
  }
  D3DKMT_DESTROYDEVICE destroy = {};
  destroy.hDevice = create.hDevice;
  if (!CheckStatus("DestroyDevice", D3DKMTDestroyDevice(&destroy))) {
    success = false;
  }
  return success;
}

}  // namespace

int main() {
  LARGE_INTEGER frequency;
  if (!QueryPerformanceFrequency(&frequency)) {
    std::fprintf(stderr, "QueryPerformanceFrequency failed: Win32=%lu\n",
                 GetLastError());
    return 1;
  }
  std::printf("Host QPC frequency=%" PRId64 "\n", frequency.QuadPart);
  HMODULE gdi_module = GetModuleHandleW(L"gdi32.dll");
  auto enumerate_adapters =
      gdi_module ? reinterpret_cast<PFND3DKMT_ENUMADAPTERS3>(
                       GetProcAddress(gdi_module, "D3DKMTEnumAdapters3"))
                 : nullptr;
  if (!enumerate_adapters) {
    std::fprintf(stderr, "D3DKMTEnumAdapters3 unavailable\n");
    return 1;
  }
  D3DKMT_ENUMADAPTERS3 enumeration = {};
  enumeration.Filter.IncludeComputeOnly = 1;
  enumeration.Filter.IncludeDisplayOnly = 1;
  if (!CheckStatus("EnumAdapters3(count)", enumerate_adapters(&enumeration))) {
    return 1;
  }
  std::vector<D3DKMT_ADAPTERINFO> adapters(enumeration.NumAdapters);
  bool success = true;
  if (!adapters.empty()) {
    enumeration.pAdapters = adapters.data();
    success = CheckStatus("EnumAdapters3", enumerate_adapters(&enumeration));
    if (enumeration.NumAdapters > adapters.size()) {
      std::fprintf(stderr, "Adapter enumeration exceeded its capacity\n");
      success = false;
    }
  }
  uint32_t sampled_count = 0;
  for (uint32_t i = 0; success && i < enumeration.NumAdapters; ++i) {
    success = QueryTelemetry(adapters[i], sampled_count);
  }
  for (const auto& adapter : adapters) {
    if (!adapter.hAdapter) {
      continue;
    }
    D3DKMT_CLOSEADAPTER close = {};
    close.hAdapter = adapter.hAdapter;
    if (!CheckStatus("CloseAdapter", D3DKMTCloseAdapter(&close))) {
      success = false;
    }
  }
  std::printf("Successfully sampled adapters: %" PRIu32 "\n", sampled_count);
  return success ? 0 : 1;
}
