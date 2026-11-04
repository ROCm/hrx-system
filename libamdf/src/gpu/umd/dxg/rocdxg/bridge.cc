// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <new>

#include "libamdf/src/gpu/umd/dxg/bridge_allocator.h"
#include "libamdf/src/gpu/umd/dxg/bridge_api.h"
#include "libamdf/src/gpu/umd/dxg/rocdxg/adapter_state.h"
#include "libamdf/src/gpu/umd/dxg/rocdxg/allocation.h"
#include "libamdf/src/gpu/umd/dxg/rocdxg/kernel_queue.h"
#include "libamdf/src/gpu/umd/dxg/rocdxg/native_transport.h"
#include "shared/include/dxcore_loader.h"
#include "shared/include/thunk_proxy/thunk_proxy.h"

namespace {

amdf_dxg_bridge_result_t MapAdapterError(ErrorCode error) {
  switch (error) {
    case ErrorCode::Success:
      return AMDF_DXG_BRIDGE_RESULT_SUCCESS;
    case ErrorCode::UnSupported:
    case ErrorCode::IncompatibleDevice:
    case ErrorCode::NotFound:
      return AMDF_DXG_BRIDGE_RESULT_UNSUPPORTED;
    case ErrorCode::OutOfMemory:
    case ErrorCode::OutOfGpuMemory:
    case ErrorCode::OutOfHandleApeMemory:
      return AMDF_DXG_BRIDGE_RESULT_RESOURCE_EXHAUSTED;
    default:
      return AMDF_DXG_BRIDGE_RESULT_INTERNAL;
  }
}

void PopulateGpuProperties(const amdf_dxg_bridge_gpu_adapter_t& adapter,
                           amdf_dxg_bridge_gpu_properties_t* out_properties) {
  const thunk_proxy::DeviceContext& device = *adapter.device;
  const amdf_dxg_bridge_gpu_properties_t properties = {
      .gfx_ip_major = device.Major(),
      .gfx_ip_minor = device.Minor(),
      .gfx_ip_stepping = device.Stepping(),
      .asic_revision = device.AsicRevision(),
      .wavefront_size = device.WavefrontSize(),
      .compute_unit_count = device.ComputeUnitCount(),
      .maximum_wave_count_per_compute_unit = device.WavePerCu(),
      .maximum_scratch_wave_count_per_compute_unit =
          device.MaxScratchSlotsPerCu(),
      .local_data_share_byte_length = device.LdsSize(),
      // The released ROCDXG matrix contains single-XCC consumer GPUs, and the
      // published proxy surface exposes no distinct XCC topology.
      .xcc_count = 1,
      .shader_engine_count = device.NumShaderEngine(),
      .supports_pm4_kernel_queue = adapter.pm4_queue_profile.is_supported,
      .supports_sdma_kernel_queue = adapter.sdma_queue_profile.is_supported,
      .is_discrete = device.IsDgpu(),
      .supports_sdma_hardware_queue =
          adapter.sdma_queue_profile.is_supported &&
          adapter.sdma_queue_profile.uses_hardware_scheduling,
      .platform_atomic_support = device.SupportPlatformAtomic(),
  };
  *out_properties = properties;
}

amdf_dxg_bridge_result_t GpuAdapterOpenImpl(
    const amdf_dxg_bridge_gpu_adapter_open_info_t& open_info,
    amdf_allocator_t host_allocator,
    amdf_dxg_bridge_gpu_adapter_t** out_adapter,
    amdf_dxg_bridge_gpu_properties_t* out_properties) {
  amdf::rocdxg_bridge::NativeTransportScope transport_scope(
      open_info.native_transport);
  if (!wsl::thunk::dxcore::DxcoreLoader::Instance().Initialize()) {
    return AMDF_DXG_BRIDGE_RESULT_INTERNAL;
  }

  amdf::dxg_bridge::HostObject<amdf_dxg_bridge_gpu_adapter_t> adapter(
      host_allocator);
  if (!adapter.Allocate()) {
    return AMDF_DXG_BRIDGE_RESULT_RESOURCE_EXHAUSTED;
  }
  adapter->host_allocator = host_allocator;
  adapter->native_transport = open_info.native_transport;

  const LUID luid = {
      static_cast<ULONG>(open_info.adapter_luid),
      static_cast<LONG>(open_info.adapter_luid >> 32),
  };
  adapter->chain =
      thunk_proxy::ChainContext::Create(open_info.adapter_handle, luid);
  if (adapter->chain == nullptr) {
    return AMDF_DXG_BRIDGE_RESULT_INTERNAL;
  }
  const ErrorCode query_error = adapter->chain->QueryAdapterInfo();
  if (query_error != ErrorCode::Success) {
    return MapAdapterError(query_error);
  }
  if (open_info.physical_adapter_index >= adapter->chain->NumChainedGpus()) {
    return AMDF_DXG_BRIDGE_RESULT_UNSUPPORTED;
  }
  const uint32_t vendor_id =
      adapter->chain->VendorId(open_info.physical_adapter_index);
  if (vendor_id != 0x1002 && vendor_id != 0x1022) {
    return AMDF_DXG_BRIDGE_RESULT_UNSUPPORTED;
  }

  adapter->device = adapter->chain->CreateDevice(
      open_info.device_handle, open_info.physical_adapter_index);
  if (adapter->device == nullptr || !adapter->device->IsWddm2Supported()) {
    return AMDF_DXG_BRIDGE_RESULT_UNSUPPORTED;
  }

  amdf::rocdxg_bridge::InitializeGpuKernelQueueProfiles(adapter.get());
  PopulateGpuProperties(*adapter.get(), out_properties);
  *out_adapter = adapter.release();
  return AMDF_DXG_BRIDGE_RESULT_SUCCESS;
}

amdf_dxg_bridge_result_t AMDF_DXG_BRIDGE_CALL
GpuAdapterOpen(const amdf_dxg_bridge_gpu_adapter_open_info_t* open_info,
               const amdf_allocator_t* host_allocator,
               amdf_dxg_bridge_gpu_adapter_t** out_adapter,
               amdf_dxg_bridge_gpu_properties_t* out_properties,
               uint32_t* out_native_status) noexcept {
  if (open_info == nullptr ||
      open_info->structure_size <
          AMDF_DXG_BRIDGE_GPU_ADAPTER_OPEN_INFO_VERSION_13_SIZE ||
      open_info->adapter_handle == 0 || open_info->device_handle == 0 ||
      open_info->native_transport.context == nullptr ||
      open_info->native_transport.invoke == nullptr ||
      host_allocator == nullptr || host_allocator->allocate == nullptr ||
      host_allocator->free == nullptr || out_adapter == nullptr ||
      out_properties == nullptr || out_native_status == nullptr) {
    return AMDF_DXG_BRIDGE_RESULT_INVALID_ARGUMENT;
  }
  *out_native_status = 0;
  try {
    return GpuAdapterOpenImpl(*open_info, *host_allocator, out_adapter,
                              out_properties);
  } catch (const std::bad_alloc&) {
    return AMDF_DXG_BRIDGE_RESULT_RESOURCE_EXHAUSTED;
  } catch (...) {
    return AMDF_DXG_BRIDGE_RESULT_INTERNAL;
  }
}

amdf_dxg_bridge_result_t AMDF_DXG_BRIDGE_CALL
GpuAdapterClose(amdf_dxg_bridge_gpu_adapter_t* adapter,
                uint32_t* out_native_status) noexcept {
  const amdf_dxg_bridge_result_t result =
      amdf::rocdxg_bridge::PrepareGpuAdapterClose(adapter, out_native_status);
  if (result != AMDF_DXG_BRIDGE_RESULT_SUCCESS) {
    return result;
  }
  amdf::rocdxg_bridge::NativeTransportScope transport_scope(
      adapter->native_transport);
  const amdf_allocator_t host_allocator = adapter->host_allocator;
  amdf::dxg_bridge::DestroyHostObject(host_allocator, adapter);
  return AMDF_DXG_BRIDGE_RESULT_SUCCESS;
}

amdf_status_t AMDF_DXG_BRIDGE_CALL
GpuBufferPrepareImport(amdf_dxg_bridge_gpu_adapter_t*,
                       const amdf_dxg_bridge_gpu_buffer_import_info_t*,
                       uint32_t*, uint32_t*, uint64_t*, uint64_t*) noexcept {
  return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
}

amdf_dxg_bridge_result_t AMDF_DXG_BRIDGE_CALL GpuClockSample(
    amdf_dxg_bridge_gpu_adapter_t*, uint32_t, uint32_t,
    amdf_dxg_bridge_gpu_queue_command_type_t,
    amdf_dxg_bridge_gpu_clock_sample_t*, uint32_t* out_native_status) noexcept {
  if (out_native_status != nullptr) {
    *out_native_status = 0;
  }
  return AMDF_DXG_BRIDGE_RESULT_UNSUPPORTED;
}

const amdf_dxg_bridge_api_t kBridgeApi = {
    sizeof(amdf_dxg_bridge_api_t),
    AMDF_DXG_BRIDGE_ABI_VERSION_13,
    GpuAdapterOpen,
    GpuAdapterClose,
    amdf::rocdxg_bridge::GpuAllocationQueryLayout,
    amdf::rocdxg_bridge::GpuAllocationCreate,
    amdf::rocdxg_bridge::GpuKernelQueueCreate,
    amdf::rocdxg_bridge::GpuKernelQueueSubmit,
    amdf::rocdxg_bridge::GpuKernelQueueDestroy,
    GpuBufferPrepareImport,
    GpuClockSample,
};

}  // namespace

extern "C" __attribute__((visibility("default"))) amdf_dxg_bridge_result_t
    AMDF_DXG_BRIDGE_CALL
    amdf_dxg_bridge_query_api(uint32_t minimum_version,
                              uint32_t maximum_version,
                              const amdf_dxg_bridge_api_t** out_api) {
  if (out_api == nullptr) {
    return AMDF_DXG_BRIDGE_RESULT_INVALID_ARGUMENT;
  }
  if (minimum_version > AMDF_DXG_BRIDGE_ABI_VERSION_13 ||
      maximum_version < AMDF_DXG_BRIDGE_ABI_VERSION_13) {
    return AMDF_DXG_BRIDGE_RESULT_VERSION_MISMATCH;
  }
  *out_api = &kBridgeApi;
  return AMDF_DXG_BRIDGE_RESULT_SUCCESS;
}
