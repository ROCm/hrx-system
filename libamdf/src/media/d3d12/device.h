// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_MEDIA_D3D12_DEVICE_H_
#define AMDF_SRC_MEDIA_D3D12_DEVICE_H_

#include <d3d12.h>
#include <d3d12video.h>
#include <wrl/client.h>

#include "libamdf/src/atomics.h"
#include "libamdf/src/media/provider.h"
#include "libamdf/src/platform/windows/kmt_api.h"

namespace amdf::media::d3d12 {

using Microsoft::WRL::ComPtr;

enum : uint32_t {
  kDecodeFamilyOrdinal = 0,
  kEncodeFamilyOrdinal = 1,
  kFamilyCount = 2,
};

struct Device {
  // Provider-neutral device and implementation dispatch.
  amdf_media_device_t base;
  // Dynamic system module retained through all factory-created COM objects.
  HMODULE dxgi_module;
  // Dynamic system module retained through native device destruction.
  HMODULE d3d12_module;
  // Independent native media device; no compute UMD is loaded here.
  ComPtr<ID3D12Device3> native;
  // Native video capability and session construction interface.
  ComPtr<ID3D12VideoDevice> video;
  // Optional native encoder capability and construction interface.
  ComPtr<ID3D12VideoDevice3> encoder;
  // KMT API borrowed from the endpoint's retained platform instance.
  const amdf_kmt_api_t* kmt;
  // Private KMT device for shared progress and cold allocation metadata.
  D3DKMT_HANDLE kmt_device;
  // Physical adapter identity used to qualify native sharing.
  uint64_t adapter_luid;
  // Resource metadata query loaded from the borrowed system KMT module.
  PFND3DKMT_QUERYRESOURCEINFOFROMNTHANDLE query_resource;
  // Cold metadata import loaded from the borrowed system KMT module.
  PFND3DKMT_OPENRESOURCEFROMNTHANDLE open_resource;
  // Immutable independently activated decode and encode families.
  amdf_queue_family_info_t families[kFamilyCount];
  // Native support for exact caller-owned host allocations.
  bool existing_heaps;
  // Native resource-address envelope, leaving room for allocation rounding.
  uint64_t maximum_memory_byte_length;
  // First observed native device-removal reason, shared by all media queues.
  amdf_atomic_uint64_t removed_status;
};

}  // namespace amdf::media::d3d12

#endif  // AMDF_SRC_MEDIA_D3D12_DEVICE_H_
