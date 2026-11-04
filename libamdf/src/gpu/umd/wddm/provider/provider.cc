// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>

#include "libamdf/src/gpu/umd/dxg/private_records.h"
#include "libamdf/src/gpu/umd/dxg/provider_allocator.h"
#include "libamdf/src/gpu/umd/dxg/provider_api.h"
#include "libamdf/src/gpu/umd/wddm/provider/api.h"

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif  // WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif  // NOMINMAX
#define WIN32_NO_STATUS
#include <windows.h>
#include <winternl.h>
#undef WIN32_NO_STATUS

#include <d3dkmthk.h>
#include <ntstatus.h>

#include "libamdf/src/gpu/umd/wddm/provider/adapter_state.h"
#include "libamdf/src/gpu/umd/wddm/provider/allocation.h"
#include "libamdf/src/gpu/umd/wddm/provider/kernel_queue.h"
#include "libamdf/src/gpu/umd/wddm/provider/shared_buffer.h"

namespace {

uint32_t QueryPrivateAdapter(void* context, uint32_t adapter_handle,
                             void* buffer, uint32_t buffer_byte_length) {
  const auto* kmt = static_cast<const amdf_kmt_api_t*>(context);
  D3DKMT_QUERYADAPTERINFO query = {};
  query.hAdapter = static_cast<D3DKMT_HANDLE>(adapter_handle);
  query.Type = KMTQAITYPE_UMDRIVERPRIVATE;
  query.pPrivateDriverData = buffer;
  query.PrivateDriverDataSize = buffer_byte_length;
  return static_cast<uint32_t>(kmt->query_adapter_info(&query));
}

bool IsUnsupportedAdapterStatus(uint32_t status) {
  return status == static_cast<uint32_t>(STATUS_OBJECT_NAME_NOT_FOUND) ||
         status == static_cast<uint32_t>(STATUS_REVISION_MISMATCH) ||
         status == static_cast<uint32_t>(STATUS_NOT_SUPPORTED);
}

amdf_dxg_provider_result_t MapPrivateResult(amdf_dxg_private_result_t result,
                                            uint32_t* inout_native_status) {
  switch (result) {
    case AMDF_DXG_PRIVATE_RESULT_SUCCESS:
      return AMDF_DXG_PROVIDER_RESULT_SUCCESS;
    case AMDF_DXG_PRIVATE_RESULT_UNSUPPORTED:
      return AMDF_DXG_PROVIDER_RESULT_UNSUPPORTED;
    case AMDF_DXG_PRIVATE_RESULT_NATIVE_FAILURE:
      if (*inout_native_status == static_cast<uint32_t>(STATUS_NO_MEMORY)) {
        *inout_native_status = 0;
        return AMDF_DXG_PROVIDER_RESULT_RESOURCE_EXHAUSTED;
      }
      if (IsUnsupportedAdapterStatus(*inout_native_status)) {
        *inout_native_status = 0;
        return AMDF_DXG_PROVIDER_RESULT_UNSUPPORTED;
      }
      return AMDF_DXG_PROVIDER_RESULT_NATIVE_FAILURE;
    default:
      return AMDF_DXG_PROVIDER_RESULT_INTERNAL;
  }
}

void PopulateGpuProperties(const amdf_dxg_provider_gpu_adapter_t& adapter,
                           amdf_dxg_provider_gpu_properties_t* out_properties) {
  const amdf_dxg_private_adapter_info_t& info = adapter.private_info;
  const amdf_dxg_provider_gpu_properties_t properties = {
      .gfx_ip_major = info.gfx_ip_major,
      .gfx_ip_minor = info.gfx_ip_minor,
      .gfx_ip_stepping = info.gfx_ip_stepping,
      .asic_revision = info.asic_revision,
      .wavefront_size = info.wavefront_size,
      .compute_unit_count = info.compute_unit_count,
      .maximum_wave_count_per_compute_unit =
          info.maximum_wave_count_per_compute_unit,
      .maximum_scratch_wave_count_per_compute_unit =
          info.maximum_scratch_wave_count_per_compute_unit,
      .local_data_share_byte_length = info.local_data_share_byte_length,
      .xcc_count = info.xcc_count,
      .shader_engine_count = info.shader_engine_count,
      .supports_pm4_kernel_queue = adapter.pm4_queue_profile.is_supported,
      .supports_sdma_kernel_queue = adapter.sdma_queue_profile.is_supported,
      .is_discrete = info.is_discrete,
      .supports_sdma_hardware_queue =
          adapter.sdma_queue_profile.is_supported &&
          adapter.sdma_queue_profile.uses_hardware_scheduling,
      .platform_atomic_support = info.supports_platform_atomics,
  };
  *out_properties = properties;
}

amdf_dxg_provider_result_t GpuAdapterOpenImpl(
    const amdf_dxg_provider_gpu_adapter_open_info_t& open_info,
    const amdf_gpu_wddm_provider_native_api_t& native_api,
    amdf_allocator_t host_allocator,
    amdf_dxg_provider_gpu_adapter_t** out_adapter,
    amdf_dxg_provider_gpu_properties_t* out_properties,
    uint32_t* out_native_status) {
  amdf::dxg_provider::HostObject<amdf_dxg_provider_gpu_adapter_t> adapter(
      host_allocator);
  if (!adapter.Allocate()) {
    return AMDF_DXG_PROVIDER_RESULT_RESOURCE_EXHAUSTED;
  }
  adapter->host_allocator = host_allocator;
  adapter->kmt = native_api.kmt;
  adapter->d3d12 = native_api.d3d12;

  std::array<uint8_t, AMDF_DXG_PRIVATE_SINGLE_QUERY_SCRATCH_BYTE_LENGTH>
      query_scratch;
  const amdf_dxg_private_adapter_query_t query = {
      .adapter_handle = open_info.adapter_handle,
      .physical_adapter_index = 0,
      .query_context = const_cast<amdf_kmt_api_t*>(adapter->kmt),
      .query = QueryPrivateAdapter,
      .scratch = query_scratch.data(),
  };
  amdf_dxg_private_adapter_info_t private_info;
  const amdf_dxg_private_result_t private_result =
      amdf_dxg_private_query_single_adapter(&query, &private_info,
                                            out_native_status);
  if (private_result != AMDF_DXG_PRIVATE_RESULT_SUCCESS) {
    return MapPrivateResult(private_result, out_native_status);
  }
  adapter->private_info = private_info;
  amdf_dxg_private_initialize_queue_profiles(&adapter->private_info,
                                             &adapter->pm4_queue_profile,
                                             &adapter->sdma_queue_profile);
  PopulateGpuProperties(*adapter.get(), out_properties);
  *out_adapter = adapter.release();
  return AMDF_DXG_PROVIDER_RESULT_SUCCESS;
}

amdf_dxg_provider_result_t GpuAdapterOpen(
    const amdf_dxg_provider_gpu_adapter_open_info_t* open_info,
    const amdf_allocator_t* host_allocator,
    amdf_dxg_provider_gpu_adapter_t** out_adapter,
    amdf_dxg_provider_gpu_properties_t* out_properties,
    uint32_t* out_native_status) noexcept {
  if (open_info == nullptr || open_info->adapter_handle == 0 ||
      open_info->native_transport.context == nullptr ||
      open_info->native_transport.invoke != nullptr ||
      host_allocator == nullptr || host_allocator->allocate == nullptr ||
      host_allocator->free == nullptr || out_adapter == nullptr ||
      out_properties == nullptr || out_native_status == nullptr) {
    return AMDF_DXG_PROVIDER_RESULT_INVALID_ARGUMENT;
  }
  *out_native_status = 0;
  if (open_info->physical_adapter_index != 0) {
    return AMDF_DXG_PROVIDER_RESULT_UNSUPPORTED;
  }
  const auto* native_api =
      static_cast<const amdf_gpu_wddm_provider_native_api_t*>(
          open_info->native_transport.context);
  if (native_api->kmt == nullptr ||
      native_api->kmt->query_adapter_info == nullptr ||
      native_api->d3d12 == nullptr) {
    return AMDF_DXG_PROVIDER_RESULT_INVALID_ARGUMENT;
  }
  return GpuAdapterOpenImpl(*open_info, *native_api, *host_allocator,
                            out_adapter, out_properties, out_native_status);
}

amdf_dxg_provider_result_t GpuAdapterClose(
    amdf_dxg_provider_gpu_adapter_t* adapter,
    uint32_t* out_native_status) noexcept {
  const amdf_dxg_provider_result_t result =
      amdf::wddm_provider::PrepareGpuAdapterClose(adapter, out_native_status);
  if (result != AMDF_DXG_PROVIDER_RESULT_SUCCESS) {
    return result;
  }
  const amdf_allocator_t host_allocator = adapter->host_allocator;
  amdf::dxg_provider::DestroyHostObject(host_allocator, adapter);
  return AMDF_DXG_PROVIDER_RESULT_SUCCESS;
}

amdf_dxg_provider_result_t GpuClockSample(
    amdf_dxg_provider_gpu_adapter_t* adapter, uint32_t adapter_handle,
    uint32_t physical_adapter_index,
    amdf_dxg_provider_gpu_queue_command_type_t command_type,
    amdf_dxg_provider_gpu_clock_sample_t* out_sample,
    uint32_t* out_native_status) noexcept {
  if (adapter == nullptr || adapter_handle == 0 || out_sample == nullptr ||
      out_native_status == nullptr) {
    return AMDF_DXG_PROVIDER_RESULT_INVALID_ARGUMENT;
  }
  *out_native_status = 0;
  const amdf_dxg_private_queue_profile_t* profile =
      amdf::wddm_provider::SelectGpuKernelQueueProfile(adapter, command_type);
  if (profile == nullptr || !profile->is_supported) {
    return AMDF_DXG_PROVIDER_RESULT_UNSUPPORTED;
  }
  D3DKMT_QUERYCLOCKCALIBRATION query = {};
  query.hAdapter = static_cast<D3DKMT_HANDLE>(adapter_handle);
  query.PhysicalAdapterIndex = physical_adapter_index;
  query.NodeOrdinal = profile->node_ordinal;
  if (adapter->kmt->query_clock_calibration == nullptr) {
    return AMDF_DXG_PROVIDER_RESULT_UNSUPPORTED;
  }
  const NTSTATUS status = adapter->kmt->query_clock_calibration(&query);
  if (status == STATUS_NOT_SUPPORTED) {
    return AMDF_DXG_PROVIDER_RESULT_UNSUPPORTED;
  }
  if (status != STATUS_SUCCESS) {
    *out_native_status = static_cast<uint32_t>(status);
    return AMDF_DXG_PROVIDER_RESULT_NATIVE_FAILURE;
  }
  if (query.ClockData.Flags.ContextManagementProcessor ||
      query.ClockData.GpuFrequency == 0) {
    return AMDF_DXG_PROVIDER_RESULT_UNSUPPORTED;
  }
  *out_sample = {
      query.ClockData.GpuFrequency,
      query.ClockData.GpuClockCounter,
      query.ClockData.CpuClockCounter,
  };
  return AMDF_DXG_PROVIDER_RESULT_SUCCESS;
}

const amdf_dxg_provider_api_t kProviderApi = {
    GpuAdapterOpen,
    GpuAdapterClose,
    amdf::wddm_provider::GpuAllocationQueryLayout,
    amdf::wddm_provider::GpuAllocationCreate,
    amdf::wddm_provider::GpuKernelQueueCreate,
    amdf::wddm_provider::GpuKernelQueueSubmit,
    amdf::wddm_provider::GpuKernelQueueDestroy,
    amdf::wddm_provider::GpuBufferPrepareImport,
    GpuClockSample,
};

}  // namespace

extern "C" const amdf_dxg_provider_api_t* amdf_gpu_wddm_default_api(void) {
  return &kProviderApi;
}
