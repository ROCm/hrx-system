// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Exercises the shared private-record decoder against the installed driver.
// The test opens adapters only; it creates no devices, contexts, or queues.

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#define WIN32_NO_STATUS
#include <windows.h>
#include <winternl.h>
#undef WIN32_NO_STATUS

#include <d3dkmthk.h>
#include <ntstatus.h>

#include <algorithm>
#include <array>
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "libamdf/src/gpu/umd/dxg/private_records.h"

namespace {

bool CheckStatus(const char* operation, NTSTATUS status) {
  if (status == STATUS_SUCCESS) {
    return true;
  }
  std::fprintf(stderr, "%s failed: NTSTATUS=0x%08" PRIx32 "\n", operation,
               static_cast<uint32_t>(status));
  return false;
}

uint32_t QueryPrivateAdapter(void*, uint32_t adapter_handle, void* buffer,
                             uint32_t buffer_byte_length) {
  D3DKMT_QUERYADAPTERINFO query = {};
  query.hAdapter = static_cast<D3DKMT_HANDLE>(adapter_handle);
  query.Type = KMTQAITYPE_UMDRIVERPRIVATE;
  query.pPrivateDriverData = buffer;
  query.PrivateDriverDataSize = buffer_byte_length;
  return static_cast<uint32_t>(D3DKMTQueryAdapterInfo(&query));
}

void PrintQueueProfile(const char* role,
                       const amdf_dxg_private_queue_profile_t& profile) {
  if (!profile.is_supported) {
    std::printf("  %s: unsupported\n", role);
    return;
  }
  std::printf("  %s: scheduler=%" PRIu32 " node=%" PRIu32 " hws=%" PRIu32 "\n",
              role, profile.scheduler_id, profile.node_ordinal,
              profile.uses_hardware_scheduling);
}

bool QueryAdapter(const D3DKMT_ADAPTERINFO& adapter,
                  const std::vector<std::string>& required_targets,
                  uint32_t& matched_count) {
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
  if (device_ids.DeviceIds.VendorID != 0x1002 &&
      device_ids.DeviceIds.VendorID != 0x1022) {
    return true;
  }

  std::array<uint8_t, AMDF_DXG_PRIVATE_SINGLE_QUERY_SCRATCH_BYTE_LENGTH>
      scratch;
  const amdf_dxg_private_adapter_query_t private_query = {
      .adapter_handle = adapter.hAdapter,
      .physical_adapter_index = 0,
      .query_context = nullptr,
      .query = QueryPrivateAdapter,
      .scratch = scratch.data(),
  };
  amdf_dxg_private_adapter_info_t info = {};
  uint32_t native_status = 0;
  const amdf_dxg_private_result_t result =
      amdf_dxg_private_query_single_adapter(&private_query, &info,
                                            &native_status);
  if (result == AMDF_DXG_PRIVATE_RESULT_UNSUPPORTED) {
    std::printf("Adapter LUID=%08" PRIx32 ":%08" PRIx32 " PCI=%04" PRIx32
                ":%04" PRIx32 " private query unsupported\n",
                static_cast<uint32_t>(adapter.AdapterLuid.HighPart),
                static_cast<uint32_t>(adapter.AdapterLuid.LowPart),
                static_cast<uint32_t>(device_ids.DeviceIds.VendorID),
                static_cast<uint32_t>(device_ids.DeviceIds.DeviceID));
    return true;
  }
  if (result == AMDF_DXG_PRIVATE_RESULT_NATIVE_FAILURE) {
    std::fprintf(stderr,
                 "Private adapter query failed: NTSTATUS=0x%08" PRIx32 "\n",
                 native_status);
    return false;
  }
  if (result != AMDF_DXG_PRIVATE_RESULT_SUCCESS) {
    std::fprintf(stderr, "Private adapter response was malformed\n");
    return false;
  }

  char target[32];
  std::snprintf(target, sizeof(target), "gfx%d%d%x", info.gfx_ip_major,
                info.gfx_ip_minor, static_cast<unsigned>(info.gfx_ip_stepping));
  std::printf("Adapter LUID=%08" PRIx32 ":%08" PRIx32 " PCI=%04" PRIx32
              ":%04" PRIx32 " target=%s\n",
              static_cast<uint32_t>(adapter.AdapterLuid.HighPart),
              static_cast<uint32_t>(adapter.AdapterLuid.LowPart),
              static_cast<uint32_t>(device_ids.DeviceIds.VendorID),
              static_cast<uint32_t>(device_ids.DeviceIds.DeviceID), target);
  std::printf("  cu=%" PRIu32 " se=%" PRIu32 " xcc=%" PRIu32 " wave=%" PRIu32
              " native_queues=0x%02" PRIx8 "\n",
              info.compute_unit_count, info.shader_engine_count, info.xcc_count,
              info.wavefront_size, info.native_queue_flags);
  amdf_dxg_private_queue_profile_t pm4_profile = {};
  amdf_dxg_private_queue_profile_t sdma_profile = {};
  amdf_dxg_private_initialize_queue_profiles(&info, &pm4_profile,
                                             &sdma_profile);
  PrintQueueProfile("pm4", pm4_profile);
  PrintQueueProfile("sdma", sdma_profile);
  if (required_targets.empty() ||
      std::find(required_targets.begin(), required_targets.end(), target) !=
          required_targets.end()) {
    ++matched_count;
  }
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  std::vector<std::string> required_targets;
  constexpr char kTargetPrefix[] = "--amdf_gpu_target=";
  for (int i = 1; i < argc; ++i) {
    if (std::strncmp(argv[i], kTargetPrefix, sizeof(kTargetPrefix) - 1) != 0 ||
        argv[i][sizeof(kTargetPrefix) - 1] == '\0') {
      std::fprintf(stderr, "Usage: %s [--amdf_gpu_target=gfxNNNN ...]\n",
                   argv[0]);
      return 1;
    }
    required_targets.emplace_back(argv[i] + sizeof(kTargetPrefix) - 1);
  }

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
    success = QueryAdapter(adapters[i], required_targets, matched_count);
  }
  for (const D3DKMT_ADAPTERINFO& adapter : adapters) {
    if (adapter.hAdapter == 0) {
      continue;
    }
    D3DKMT_CLOSEADAPTER close = {};
    close.hAdapter = adapter.hAdapter;
    if (!CheckStatus("CloseAdapter", D3DKMTCloseAdapter(&close))) {
      success = false;
    }
  }
  if (!required_targets.empty() && matched_count == 0) {
    std::fprintf(stderr, "No successfully queried adapter matched:");
    for (const std::string& target : required_targets) {
      std::fprintf(stderr, " %s", target.c_str());
    }
    std::fprintf(stderr, "\n");
    success = false;
  } else if (success && matched_count == 0) {
    std::printf("SKIP: no AMD GPU supports the private adapter query\n");
  }
  std::printf("Queried matching adapters: %" PRIu32 "\n", matched_count);
  return success ? 0 : 1;
}
