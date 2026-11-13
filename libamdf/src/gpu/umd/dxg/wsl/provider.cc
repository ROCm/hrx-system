// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <sys/ioctl.h>

#include "libamdf/src/gpu/umd/dxg/private_records.h"
#include "libamdf/src/gpu/umd/dxg/provider_allocator.h"
#include "libamdf/src/gpu/umd/dxg/provider_api.h"
#include "libamdf/src/gpu/umd/dxg/wsl/adapter_state.h"
#include "libamdf/src/gpu/umd/dxg/wsl/allocation.h"
#include "libamdf/src/gpu/umd/dxg/wsl/api.h"
#include "libamdf/src/gpu/umd/dxg/wsl/clock.h"
#include "libamdf/src/gpu/umd/dxg/wsl/kernel_queue.h"
#include "libamdf/src/platform/linux/dxg/kmt_types.h"

namespace {

uint32_t QueryPrivateAdapter(void* context, uint32_t adapter_handle,
                             void* buffer, uint32_t buffer_byte_length) {
  const auto* transport =
      static_cast<const amdf_dxg_provider_native_transport_t*>(context);
  D3DKMT_QUERYADAPTERINFO query = {
      .hAdapter = static_cast<D3DKMT_HANDLE>(adapter_handle),
      .Type = KMTQAITYPE_UMDRIVERPRIVATE,
      .pPrivateDriverData = buffer,
      .PrivateDriverDataSize = buffer_byte_length,
  };
  return transport->invoke(transport->context,
                           _IOWR('G', 0x09, D3DKMT_QUERYADAPTERINFO), &query);
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
      .is_discrete = info.is_discrete != 0,
      .supports_sdma_hardware_queue =
          adapter.sdma_queue_profile.is_supported &&
          adapter.sdma_queue_profile.uses_hardware_scheduling,
      .platform_atomic_support = info.supports_platform_atomics != 0,
  };
  *out_properties = properties;
}

amdf_dxg_provider_result_t GpuAdapterOpenImpl(
    const amdf_dxg_provider_gpu_adapter_open_info_t& open_info,
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
  adapter->native_transport = open_info.native_transport;

  amdf::dxg_provider::HostBuffer query_scratch;
  if (!query_scratch.Allocate(
          host_allocator, AMDF_DXG_PRIVATE_LINKED_QUERY_SCRATCH_BYTE_LENGTH)) {
    return AMDF_DXG_PROVIDER_RESULT_RESOURCE_EXHAUSTED;
  }
  const amdf_dxg_private_adapter_query_t query = {
      .adapter_handle = open_info.adapter_handle,
      .physical_adapter_index = open_info.physical_adapter_index,
      .query_context = &adapter->native_transport,
      .query = QueryPrivateAdapter,
      .scratch = query_scratch.data(),
  };
  amdf_dxg_private_adapter_info_t private_info;
  const amdf_dxg_private_result_t private_result =
      amdf_dxg_private_query_linked_adapter(&query, &private_info,
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
      open_info->native_transport.invoke == nullptr ||
      host_allocator == nullptr || host_allocator->allocate == nullptr ||
      host_allocator->free == nullptr || out_adapter == nullptr ||
      out_properties == nullptr || out_native_status == nullptr) {
    return AMDF_DXG_PROVIDER_RESULT_INVALID_ARGUMENT;
  }
  *out_native_status = 0;
  return GpuAdapterOpenImpl(*open_info, *host_allocator, out_adapter,
                            out_properties, out_native_status);
}

amdf_dxg_provider_result_t GpuAdapterClose(
    amdf_dxg_provider_gpu_adapter_t* adapter,
    uint32_t* out_native_status) noexcept {
  const amdf_dxg_provider_result_t result =
      amdf::wsl_provider::PrepareGpuAdapterClose(adapter, out_native_status);
  if (result != AMDF_DXG_PROVIDER_RESULT_SUCCESS) {
    return result;
  }
  const amdf_allocator_t host_allocator = adapter->host_allocator;
  amdf::dxg_provider::DestroyHostObject(host_allocator, adapter);
  return AMDF_DXG_PROVIDER_RESULT_SUCCESS;
}

amdf_status_t GpuBufferPrepareImport(
    amdf_dxg_provider_gpu_adapter_t*,
    const amdf_dxg_provider_gpu_buffer_import_info_t*, uint32_t*, uint32_t*,
    uint64_t*, uint64_t*) noexcept {
  return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
}

const amdf_dxg_provider_api_t kProviderApi = {
    GpuAdapterOpen,
    GpuAdapterClose,
    amdf::wsl_provider::GpuAllocationQueryLayout,
    amdf::wsl_provider::GpuAllocationCreate,
    amdf::wsl_provider::GpuKernelQueueCreate,
    amdf::wsl_provider::GpuKernelQueueSubmit,
    amdf::wsl_provider::GpuKernelQueueDestroy,
    GpuBufferPrepareImport,
    amdf::wsl_provider::GpuClockSample,
};

}  // namespace

extern "C" const amdf_dxg_provider_api_t* amdf_gpu_dxg_wsl_default_api(void) {
  return &kProviderApi;
}
