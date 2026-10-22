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

#include "amdf/media/h264.h"
#include "libamdf/src/atomics.h"
#include "libamdf/src/media/provider.h"
#include "libamdf/src/platform/windows/kmt_api.h"

namespace amdf::media::d3d12 {

using Microsoft::WRL::ComPtr;

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
  // Immutable independently activated decode family.
  amdf_queue_family_info_t family;
  // Native support for exact caller-owned host allocations.
  bool existing_heaps;
  // Native resource-address envelope, leaving room for allocation rounding.
  uint64_t maximum_memory_byte_length;
  // First observed native device-removal reason, shared by all media queues.
  amdf_atomic_uint64_t removed_status;
};

// Copies a codec envelope after validating its public schema and native tuple.
// Unsupported configurations are answered queries; native failures are
// statuses.
amdf_status_t QueryConfiguration(
    Device* device, const amdf_media_configuration_t& configuration,
    const void* codec_configuration, amdf_media_configuration_info_t* out_info,
    D3D12_VIDEO_DECODE_CONFIGURATION* out_native_configuration);

// Native resource description for an already-admitted immutable image shape.
D3D12_RESOURCE_DESC ImageDescription(
    const amdf_media_image_description_t& description);

// Exact native placement requirements; no backing allocation is performed.
amdf_status_t ImageRequirements(
    Device* device, const amdf_media_image_description_t& description,
    amdf_media_image_requirements_t* out_requirements);

}  // namespace amdf::media::d3d12

#endif  // AMDF_SRC_MEDIA_D3D12_DEVICE_H_
