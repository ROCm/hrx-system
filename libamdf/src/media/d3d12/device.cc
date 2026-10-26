// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/media/d3d12/device.h"

#include <dxgi1_4.h>

#include "libamdf/src/endpoint.h"
#include "libamdf/src/media/d3d12/command.h"
#include "libamdf/src/media/d3d12/configuration.h"
#include "libamdf/src/media/d3d12/image.h"
#include "libamdf/src/media/d3d12/memory.h"
#include "libamdf/src/media/d3d12/object.h"
#include "libamdf/src/media/d3d12/queue.h"
#include "libamdf/src/media/d3d12/session.h"
#include "libamdf/src/platform/windows/endpoint.h"
#include "libamdf/src/platform/windows/endpoint_properties.h"
#include "libamdf/src/structure.h"

namespace amdf::media::d3d12 {
namespace {

void QueryInfo(const amdf_device_t* base, amdf_device_info_t* out_info) {
  const auto* device = reinterpret_cast<const Device*>(base);
  out_info->id.words[0] = device->adapter_luid;
  out_info->id.words[1] =
      (uint64_t(device->kmt_device) << 32) | AMDF_EXTENSION_MEDIA;
  out_info->reset_epoch = 1;
}

amdf_status_t QueryReservationProfile(amdf_device_t*, uint32_t,
                                      amdf_memory_reservation_profile_t*) {
  return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
}

amdf_status_t DestroyNative(amdf_device_t* base) {
  auto* device = reinterpret_cast<Device*>(base);
  device->encoder.Reset();
  device->video.Reset();
  device->native.Reset();
  if (device->kmt_device != 0) {
    D3DKMT_DESTROYDEVICE destroy = {
        .hDevice = device->kmt_device,
    };
    const amdf_status_t status =
        amdf_kmt_make_status(device->kmt->destroy_device(&destroy));
    if (!amdf_status_is_ok(status)) {
      return status;
    }
    device->kmt_device = 0;
  }
  if (device->d3d12_module != nullptr) {
    if (!FreeLibrary(device->d3d12_module)) {
      return amdf_make_status(AMDF_STATUS_DOMAIN_WIN32, GetLastError());
    }
    device->d3d12_module = nullptr;
  }
  if (device->dxgi_module != nullptr) {
    if (!FreeLibrary(device->dxgi_module)) {
      return amdf_make_status(AMDF_STATUS_DOMAIN_WIN32, GetLastError());
    }
    device->dxgi_module = nullptr;
  }
  return AMDF_STATUS_OK;
}

amdf_status_t AMDF_CALL EnumerateCodecs(amdf_device_t*, uint32_t ordinal,
                                        uint32_t capacity,
                                        amdf_media_codec_info_t* codecs,
                                        uint32_t* count) {
  static constexpr amdf_media_codec_info_t decode[] = {
      {AMDF_MEDIA_CODEC_H264, AMDF_MEDIA_DIRECTION_DECODE},
      {AMDF_MEDIA_CODEC_HEVC, AMDF_MEDIA_DIRECTION_DECODE},
      {AMDF_MEDIA_CODEC_VP9, AMDF_MEDIA_DIRECTION_DECODE},
      {AMDF_MEDIA_CODEC_AV1, AMDF_MEDIA_DIRECTION_DECODE},
  };
  static constexpr amdf_media_codec_info_t encode[] = {
      {AMDF_MEDIA_CODEC_H264, AMDF_MEDIA_DIRECTION_ENCODE},
      {AMDF_MEDIA_CODEC_HEVC, AMDF_MEDIA_DIRECTION_ENCODE},
      {AMDF_MEDIA_CODEC_AV1, AMDF_MEDIA_DIRECTION_ENCODE},
  };
  const amdf_media_codec_info_t* supported =
      ordinal == kDecodeFamilyOrdinal ? decode : encode;
  *count = ordinal == kDecodeFamilyOrdinal ? sizeof(decode) / sizeof(decode[0])
                                           : sizeof(encode) / sizeof(encode[0]);
  if (capacity != 0 && capacity < *count) {
    return amdf_make_api_status(AMDF_STATUS_CODE_BUFFER_TOO_SMALL);
  }
  for (uint32_t index = 0; index < capacity && index < *count; ++index) {
    codecs[index] = supported[index];
  }
  return AMDF_STATUS_OK;
}

amdf_status_t AMDF_CALL QueryDeviceConfiguration(
    amdf_device_t* base, const amdf_media_configuration_query_t* query,
    amdf_media_configuration_info_t* out_info) {
  auto* encode_info =
      static_cast<amdf_media_encode_configuration_info_t*>(out_info->next);
  if (encode_info != nullptr) {
    amdf_status_t status = amdf_structure_validate_output(
        encode_info, AMDF_STRUCTURE_TYPE_MEDIA_ENCODE_CONFIGURATION_INFO,
        sizeof(*encode_info));
    if (!amdf_status_is_ok(status)) {
      return status;
    }
    if (query->configuration.direction != AMDF_MEDIA_DIRECTION_ENCODE) {
      return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
    }
  }
  amdf_media_configuration_info_t info = {
      .type = AMDF_STRUCTURE_TYPE_MEDIA_CONFIGURATION_INFO,
      .structure_size = out_info->structure_size,
  };
  amdf_status_t status = QueryConfiguration(
      reinterpret_cast<Device*>(base), query->queue_family_ordinal,
      query->configuration, query->next, &info, nullptr, nullptr);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  amdf_media_encode_configuration_info_t encode_value = {};
  if (encode_info != nullptr) {
    encode_value.type = AMDF_STRUCTURE_TYPE_MEDIA_ENCODE_CONFIGURATION_INFO;
    encode_value.structure_size = encode_info->structure_size;
    encode_value.maximum_temporal_layer_count =
        info.reason == AMDF_MEDIA_CONFIGURATION_REASON_CODEC ? 0 : 1;
  }
  info.next = encode_info;
  *out_info = info;
  if (encode_info != nullptr) {
    *encode_info = encode_value;
  }
  return AMDF_STATUS_OK;
}

amdf_status_t AMDF_CALL QueryImageRequirements(
    amdf_device_t* base, const amdf_media_image_description_t* description,
    amdf_media_image_requirements_t* out_requirements) {
  return ImageRequirements(reinterpret_cast<Device*>(base), *description,
                           out_requirements);
}

const amdf_media_api_t kOperations = {
    .structure_size = sizeof(amdf_media_api_t),
    .extension_version = AMDF_MEDIA_EXTENSION_VERSION_1,
    .device_enumerate_codecs = EnumerateCodecs,
    .device_query_configuration = QueryDeviceConfiguration,
    .device_query_image_requirements = QueryImageRequirements,
    .kernel_queue_create = QueueCreate,
    .image_create = ImageCreate,
    .image_query_info = ImageQueryInfo,
    .image_destroy = ImageDestroy,
    .session_create = SessionCreate,
    .session_query_info = SessionQueryInfo,
    .command_create = CommandCreate,
    .command_prepare = CommandPrepare,
    .kernel_queue_submit = QueueSubmit,
    .command_query_result = CommandQueryResult,
    .command_destroy = CommandDestroy,
    .session_destroy = SessionDestroy,
};

const amdf_device_vtable_t kDeviceVtable = {
    QueryInfo,     QueryReservationProfile, QueryMemoryProfile,
    PrepareMemory, PrepareMemoryImport,     DestroyNative};

amdf_status_t InitializeNative(Device* device,
                               amdf_platform_endpoint_t* endpoint) {
  device->kmt = &endpoint->instance->kmt;
  if (endpoint->physical_adapter_index != 0 ||
      device->kmt->create_device == nullptr ||
      device->kmt->destroy_device == nullptr ||
      device->kmt->destroy_allocation == nullptr) {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
  device->query_resource =
      reinterpret_cast<PFND3DKMT_QUERYRESOURCEINFOFROMNTHANDLE>(GetProcAddress(
          device->kmt->module, "D3DKMTQueryResourceInfoFromNtHandle"));
  device->open_resource = reinterpret_cast<PFND3DKMT_OPENRESOURCEFROMNTHANDLE>(
      GetProcAddress(device->kmt->module, "D3DKMTOpenResourceFromNtHandle"));
  if (device->query_resource == nullptr || device->open_resource == nullptr) {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
  device->dxgi_module =
      LoadLibraryExW(L"dxgi.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
  if (device->dxgi_module == nullptr) {
    return amdf_make_status(AMDF_STATUS_DOMAIN_WIN32, GetLastError());
  }
  device->d3d12_module =
      LoadLibraryExW(L"d3d12.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
  if (device->d3d12_module == nullptr) {
    return amdf_make_status(AMDF_STATUS_DOMAIN_WIN32, GetLastError());
  }
  using CreateFactory = HRESULT(WINAPI*)(UINT, REFIID, void**);
  const auto create_factory = reinterpret_cast<CreateFactory>(
      GetProcAddress(device->dxgi_module, "CreateDXGIFactory2"));
  const auto create_device = reinterpret_cast<PFN_D3D12_CREATE_DEVICE>(
      GetProcAddress(device->d3d12_module, "D3D12CreateDevice"));
  if (create_factory == nullptr || create_device == nullptr) {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
  LUID luid;
  uint32_t physical_adapter_index;
  amdf_windows_endpoint_id_decode(&endpoint->id, &luid,
                                  &physical_adapter_index);
  device->adapter_luid =
      (uint64_t(static_cast<uint32_t>(luid.HighPart)) << 32) | luid.LowPart;
  ComPtr<IDXGIFactory4> factory;
  ComPtr<IDXGIAdapter> adapter;
  amdf_status_t status = Hresult(create_factory(0, IID_PPV_ARGS(&factory)));
  if (amdf_status_is_ok(status)) {
    status = Hresult(factory->EnumAdapterByLuid(luid, IID_PPV_ARGS(&adapter)));
  }
  if (amdf_status_is_ok(status)) {
    status = Hresult(create_device(adapter.Get(), D3D_FEATURE_LEVEL_11_0,
                                   IID_PPV_ARGS(&device->native)));
  }
  if (amdf_status_is_ok(status)) {
    status = Hresult(device->native.As(&device->video));
  }
  if (amdf_status_is_ok(status)) {
    const HRESULT encoder_status = device->video.As(&device->encoder);
    if (FAILED(encoder_status) && encoder_status != E_NOINTERFACE) {
      status = Hresult(encoder_status);
    }
  }
  D3D12_FEATURE_DATA_GPU_VIRTUAL_ADDRESS_SUPPORT addresses = {};
  if (amdf_status_is_ok(status)) {
    status = Hresult(device->native->CheckFeatureSupport(
        D3D12_FEATURE_GPU_VIRTUAL_ADDRESS_SUPPORT, &addresses,
        sizeof(addresses)));
  }
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  if (addresses.MaxGPUVirtualAddressBitsPerResource < 18 ||
      addresses.MaxGPUVirtualAddressBitsPerResource > 63) {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
  device->maximum_memory_byte_length =
      (UINT64_C(1) << addresses.MaxGPUVirtualAddressBitsPerResource) - 262144;
  D3D12_FEATURE_DATA_EXISTING_HEAPS existing = {};
  status = Hresult(device->native->CheckFeatureSupport(
      D3D12_FEATURE_EXISTING_HEAPS, &existing, sizeof(existing)));
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  device->existing_heaps = existing.Supported != FALSE;
  D3DKMT_CREATEDEVICE create = {
      .hAdapter = endpoint->adapter,
  };
  status = amdf_kmt_make_status(device->kmt->create_device(&create));
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  device->kmt_device = create.hDevice;
  if (device->kmt_device == 0) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INTERNAL);
  }
  const amdf_queue_family_info_t family = {
      .type = AMDF_STRUCTURE_TYPE_QUEUE_FAMILY_INFO,
      .structure_size = sizeof(family),
      .command_type = AMDF_QUEUE_COMMAND_TYPE_MEDIA,
      .publication_modes = AMDF_QUEUE_PUBLICATION_MODE_KERNEL,
      .format_version = AMDF_MEDIA_EXTENSION_VERSION_1,
      .roles = AMDF_QUEUE_ROLE_MEDIA,
      .cache_operations = AMDF_CACHE_OPERATIONS_RELEASE_TO_SYSTEM |
                          AMDF_CACHE_OPERATIONS_ACQUIRE_FROM_SYSTEM,
      .cache_transition_kinds = AMDF_CACHE_TRANSITION_KINDS_GLOBAL,
  };
  device->families[kDecodeFamilyOrdinal] = family;
  device->families[kEncodeFamilyOrdinal] = family;
  if (amdf_windows_monitored_fence_is_supported(device->kmt)) {
    for (auto& candidate : device->families) {
      candidate.kernel_queue_capabilities =
          AMDF_KERNEL_QUEUE_CAPABILITY_EXTERNAL_SYNCHRONIZATION;
    }
  }
  return AMDF_STATUS_OK;
}

}  // namespace
}  // namespace amdf::media::d3d12

amdf_status_t amdf_media_provider_device_create(
    amdf_endpoint_t* endpoint, const amdf_media_device_create_info_t*,
    amdf_device_t** out_device) {
  using namespace amdf::media::d3d12;
  if (amdf_endpoint_get_cached_info(endpoint)->engine_kind !=
      AMDF_ENGINE_KIND_GPU) {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
  const auto allocator = amdf_endpoint_host_allocator(endpoint);
  Device* device = nullptr;
  amdf_status_t status = Allocate(allocator, &device);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  device->base.operations = &kOperations;
  amdf_atomic_uint64_initialize(&device->removed_status, AMDF_STATUS_OK);
  status = InitializeNative(device, amdf_endpoint_get_platform(endpoint));
  if (amdf_status_is_ok(status)) {
    const uint32_t family_count = device->encoder != nullptr ? kFamilyCount : 1;
    status = amdf_device_initialize(&device->base.base, &kDeviceVtable,
                                    endpoint, AMDF_EXTENSION_MEDIA,
                                    family_count, device->families);
  }
  if (!amdf_status_is_ok(status)) {
    const amdf_status_t cleanup = DestroyNative(&device->base.base);
    Delete(allocator, device);
    return amdf_status_is_ok(cleanup) ? status : cleanup;
  }
  *out_device = &device->base.base;
  return AMDF_STATUS_OK;
}
