// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/gpu/umd/wddm/wkmi/bridge_api.h"

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

#include "libamdf/src/gpu/umd/wddm/wkmi/adapter_state.h"
#include "libamdf/src/gpu/umd/wddm/wkmi/allocation.h"
#include "libamdf/src/gpu/umd/wddm/wkmi/bridge_allocator.h"
#include "libamdf/src/gpu/umd/wddm/wkmi/kernel_queue.h"
#include "wkmi.h"

namespace {

bool IsUnsupportedAdapterStatus(NTSTATUS status) {
  return status == STATUS_OBJECT_NAME_NOT_FOUND ||
         status == STATUS_REVISION_MISMATCH || status == STATUS_NOT_SUPPORTED;
}

void PopulateGpuProperties(Wkmi::DeviceInfo& device_info,
                           amdf_wkmi_bridge_gpu_properties_t* out_properties) {
  amdf_wkmi_bridge_gpu_properties_t properties = {};
  properties.gfx_ip_major = device_info.major;
  properties.gfx_ip_minor = device_info.minor;
  properties.gfx_ip_stepping = device_info.stepping;
  properties.asic_revision = device_info.asic_revision;
  properties.wavefront_size = device_info.wavefront_size;
  properties.compute_unit_count = device_info.compute_unit_count;
  properties.maximum_wave_count_per_compute_unit = device_info.wave_per_cu;
  properties.maximum_scratch_wave_count_per_compute_unit =
      device_info.max_scratch_slots_per_cu;
  properties.local_data_share_byte_length = device_info.lds_size;
  properties.xcc_count = device_info.num_xcc;
  properties.shader_engine_count = device_info.num_shader_engine;
  const bool has_kernel_queue_api = Wkmi::GetContextPrivDataSize() > 0 &&
                                    Wkmi::GetHwQueuePrivDataSize() > 0 &&
                                    Wkmi::GetSubmitPrivDataSize() > 0;
  uint32_t scheduler = 0;
  properties.supports_pm4_kernel_queue =
      has_kernel_queue_api &&
      amdf::wkmi_bridge::SelectGpuHardwareQueueScheduler(
          device_info, AMDF_WKMI_BRIDGE_GPU_QUEUE_COMMAND_TYPE_PM4, &scheduler);
  properties.supports_sdma_kernel_queue =
      has_kernel_queue_api &&
      amdf::wkmi_bridge::SelectGpuHardwareQueueScheduler(
          device_info, AMDF_WKMI_BRIDGE_GPU_QUEUE_COMMAND_TYPE_SDMA,
          &scheduler);
  *out_properties = properties;
}

amdf_wkmi_bridge_result_t GpuAdapterOpenImpl(
    uint32_t adapter_handle, amdf_allocator_t host_allocator,
    amdf_wkmi_bridge_gpu_adapter_t** out_adapter,
    amdf_wkmi_bridge_gpu_properties_t* out_properties,
    uint32_t* out_native_status) {
  amdf::wkmi_bridge::HostObject<amdf_wkmi_bridge_gpu_adapter_t> adapter(
      host_allocator);
  if (!adapter.Allocate()) {
    return AMDF_WKMI_BRIDGE_RESULT_RESOURCE_EXHAUSTED;
  }
  adapter->host_allocator = host_allocator;
  const NTSTATUS native_status = Wkmi::ParseAdapterInfo(
      static_cast<D3DKMT_HANDLE>(adapter_handle), &adapter->device_info);
  if (IsUnsupportedAdapterStatus(native_status)) {
    return AMDF_WKMI_BRIDGE_RESULT_UNSUPPORTED;
  }
  if (native_status != STATUS_SUCCESS) {
    *out_native_status = static_cast<uint32_t>(native_status);
    return AMDF_WKMI_BRIDGE_RESULT_NATIVE_FAILURE;
  }
  PopulateGpuProperties(adapter->device_info, out_properties);
  *out_adapter = adapter.release();
  return AMDF_WKMI_BRIDGE_RESULT_SUCCESS;
}

amdf_wkmi_bridge_result_t AMDF_WKMI_BRIDGE_CALL
GpuAdapterOpen(uint32_t adapter_handle, uint32_t physical_adapter_index,
               const amdf_allocator_t* host_allocator,
               amdf_wkmi_bridge_gpu_adapter_t** out_adapter,
               amdf_wkmi_bridge_gpu_properties_t* out_properties,
               uint32_t* out_native_status) noexcept {
  if (adapter_handle == 0 || host_allocator == nullptr ||
      host_allocator->allocate == nullptr || host_allocator->free == nullptr ||
      out_adapter == nullptr || out_properties == nullptr ||
      out_native_status == nullptr) {
    return AMDF_WKMI_BRIDGE_RESULT_INVALID_ARGUMENT;
  }
  *out_native_status = 0;
  if (physical_adapter_index != 0) {
    return AMDF_WKMI_BRIDGE_RESULT_UNSUPPORTED;
  }

  try {
    return GpuAdapterOpenImpl(adapter_handle, *host_allocator, out_adapter,
                              out_properties, out_native_status);
  } catch (const std::bad_alloc&) {
    return AMDF_WKMI_BRIDGE_RESULT_RESOURCE_EXHAUSTED;
  } catch (...) {
    return AMDF_WKMI_BRIDGE_RESULT_INTERNAL;
  }
}

amdf_wkmi_bridge_result_t AMDF_WKMI_BRIDGE_CALL
GpuAdapterClose(amdf_wkmi_bridge_gpu_adapter_t* adapter,
                uint32_t* out_native_status) noexcept {
  const amdf_wkmi_bridge_result_t result =
      amdf::wkmi_bridge::PrepareGpuAdapterClose(adapter, out_native_status);
  if (result != AMDF_WKMI_BRIDGE_RESULT_SUCCESS) {
    return result;
  }
  const amdf_allocator_t host_allocator = adapter->host_allocator;
  amdf::wkmi_bridge::DestroyHostObject(host_allocator, adapter);
  return AMDF_WKMI_BRIDGE_RESULT_SUCCESS;
}

const amdf_wkmi_bridge_api_t kBridgeApiV2 = {
    sizeof(amdf_wkmi_bridge_api_t),
    AMDF_WKMI_BRIDGE_ABI_VERSION_2,
    GpuAdapterOpen,
    GpuAdapterClose,
    amdf::wkmi_bridge::GpuAllocationQueryLayout,
    amdf::wkmi_bridge::GpuAllocationCreate,
    amdf::wkmi_bridge::GpuKernelQueueCreate,
    amdf::wkmi_bridge::GpuKernelQueueSubmit,
    amdf::wkmi_bridge::GpuKernelQueueDestroy,
};

}  // namespace

extern "C" __declspec(dllexport) amdf_wkmi_bridge_result_t AMDF_WKMI_BRIDGE_CALL
amdf_wkmi_bridge_query_api(uint32_t minimum_version, uint32_t maximum_version,
                           const amdf_wkmi_bridge_api_t** out_api) {
  if (out_api == nullptr) {
    return AMDF_WKMI_BRIDGE_RESULT_INVALID_ARGUMENT;
  }
  if (minimum_version > AMDF_WKMI_BRIDGE_ABI_VERSION_2 ||
      maximum_version < AMDF_WKMI_BRIDGE_ABI_VERSION_2) {
    return AMDF_WKMI_BRIDGE_RESULT_VERSION_MISMATCH;
  }
  *out_api = &kBridgeApiV2;
  return AMDF_WKMI_BRIDGE_RESULT_SUCCESS;
}
