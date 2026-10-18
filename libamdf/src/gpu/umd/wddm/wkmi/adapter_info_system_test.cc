// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Exercises the binary SDK's adapter query against the installed driver. Raw
// native queue support is distinct from the services exposed by libamdf. This
// test opens adapters only; it creates no devices, contexts, or queues.

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
#include <cstdlib>
#include <cstring>
#include <memory>
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

void PrintScheduler(const char* role, int scheduler, Wkmi::DeviceInfo& info) {
  std::printf("  %s: scheduler=%d engine_ordinal=%d hws_enabled=%u\n", role,
              scheduler, Wkmi::EngineOrdinal(scheduler, &info),
              static_cast<unsigned>(Wkmi::GetHwsEnabled(scheduler, &info)));
}

bool QueryAdapter(const D3DKMT_ADAPTERINFO& adapter,
                  const char* required_target, uint32_t& matched_count) {
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

  // WKMI's query accepts a logical adapter handle and describes its primary
  // physical adapter, matching physical index zero in the public KMT query.
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

  std::printf("Adapter LUID=%08" PRIx32 ":%08" PRIx32
              " physical_index=0 PCI=%04" PRIx32 ":%04" PRIx32 "\n",
              static_cast<uint32_t>(adapter.AdapterLuid.HighPart),
              static_cast<uint32_t>(adapter.AdapterLuid.LowPart),
              static_cast<uint32_t>(device_ids.DeviceIds.VendorID),
              static_cast<uint32_t>(device_ids.DeviceIds.DeviceID));
  Wkmi::DeviceInfo info = {};
  const NTSTATUS status = Wkmi::ParseAdapterInfo(adapter.hAdapter, &info);
  // The SDK allocates this opaque block even on some rejected queries.
  // Releasing it in the same CRT preserves the binary dependency's ownership.
  std::unique_ptr<void, decltype(&std::free)> adapter_info_owner(
      info.adapter_info, &std::free);
  if (status == STATUS_OBJECT_NAME_NOT_FOUND ||
      status == STATUS_REVISION_MISMATCH || status == STATUS_NOT_SUPPORTED) {
    std::printf("  SDK adapter query unsupported: NTSTATUS=0x%08" PRIx32 "\n",
                static_cast<uint32_t>(status));
    return true;
  }
  if (!CheckStatus("Wkmi::ParseAdapterInfo", status)) {
    return false;
  }

  char target[32];
  std::snprintf(target, sizeof(target), "gfx%d%d%x", info.major, info.minor,
                static_cast<unsigned>(info.stepping));
  std::printf("  name=%.*s target=%s kmd_version=%" PRIu32
              " sdma_fw_version=%" PRIu32 "\n",
              static_cast<int>(sizeof(info.product_name)), info.product_name,
              target, info.kmd_version, info.sdma_fw_version);
  std::printf("  os_hws_enable_flags=0x%08" PRIx32
              " engine_ordinal_mask=0x%016" PRIx64 "\n",
              info.hwsInfo.osHwsEnableFlags, info.hwsInfo.engineOrdinalMask);
  std::printf("  native_aql_queue=%u native_sdma_queue=%u cp_queues=%" PRIu32
              " sdma_schedulers=%zu\n",
              static_cast<unsigned>(info.hwsInfo.hwsMask.aql_queue),
              static_cast<unsigned>(info.hwsInfo.hwsMask.sdma_queue),
              info.num_cp_queues, info.sdma_schedid.size());
  PrintScheduler("pm4", static_cast<int>(info.pm4_compute_schedid), info);
  PrintScheduler("compute", static_cast<int>(info.compute_schedid), info);
  for (int scheduler : info.sdma_schedid) {
    PrintScheduler("sdma", scheduler, info);
  }
  if (!required_target || std::strcmp(required_target, target) == 0) {
    ++matched_count;
  }
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  const char* required_target = nullptr;
  constexpr char kTargetPrefix[] = "--amdf_gpu_target=";
  if (argc == 2 &&
      std::strncmp(argv[1], kTargetPrefix, sizeof(kTargetPrefix) - 1) == 0 &&
      argv[1][sizeof(kTargetPrefix) - 1] != '\0') {
    required_target = argv[1] + sizeof(kTargetPrefix) - 1;
  } else if (argc != 1) {
    std::fprintf(stderr, "Usage: %s [--amdf_gpu_target=gfxNNNN]\n", argv[0]);
    return 1;
  }

  // gdi32 is already imported for QueryAdapterInfo and CloseAdapter. Resolve
  // the newer enumeration entry point just as the provider does; some SDK
  // import libraries omit it even when the operating system exports it.
  HMODULE gdi_module = GetModuleHandleW(L"gdi32.dll");
  auto enumerate_adapters =
      gdi_module ? reinterpret_cast<PFND3DKMT_ENUMADAPTERS3>(
                       GetProcAddress(gdi_module, "D3DKMTEnumAdapters3"))
                 : nullptr;
  if (!enumerate_adapters) {
    std::fprintf(stderr, "D3DKMTEnumAdapters3 unavailable: Win32=%lu\n",
                 GetLastError());
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
  uint32_t matched_count = 0;
  for (uint32_t i = 0; success && i < enumeration.NumAdapters; ++i) {
    success = QueryAdapter(adapters[i], required_target, matched_count);
  }
  // Close every published handle, including partial enumeration on failure.
  for (const auto& adapter : adapters) {
    if (adapter.hAdapter == 0) {
      continue;
    }
    D3DKMT_CLOSEADAPTER close = {};
    close.hAdapter = adapter.hAdapter;
    if (!CheckStatus("CloseAdapter", D3DKMTCloseAdapter(&close))) {
      success = false;
    }
  }
  if (required_target && matched_count == 0) {
    std::fprintf(stderr, "No successfully queried adapter matched %s\n",
                 required_target);
    success = false;
  } else if (success && matched_count == 0) {
    std::printf("SKIP: no AMD GPU supports the native SDK query\n");
  }
  std::printf("Queried matching adapters: %" PRIu32 "\n", matched_count);
  return success ? 0 : 1;
}
