// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstdint>

#include "libamdf/src/gpu/umd/wddm/wkmi/bridge_api.h"

struct amdf_wkmi_bridge_gpu_adapter_t {};
struct amdf_wkmi_bridge_gpu_kernel_queue_t {};

namespace {

struct FakeBridgeState {
  // Stable parsed adapter identity returned by successful opens.
  amdf_wkmi_bridge_gpu_adapter_t adapter;
  // Result selected for bridge API negotiation.
  amdf_wkmi_bridge_result_t query_result = AMDF_WKMI_BRIDGE_RESULT_SUCCESS;
  // Number of adapter-close operations rejected before native consumption.
  uint32_t adapter_close_failures_remaining = 1;
  // Number of successfully parsed adapters.
  uint32_t adapter_open_success_count = 0;
  // Number of adapter-close attempts.
  uint32_t adapter_close_attempt_count = 0;
  // Number of successfully released adapters.
  uint32_t adapter_close_success_count = 0;
};

FakeBridgeState state;

amdf_wkmi_bridge_result_t AMDF_WKMI_BRIDGE_CALL
GpuAdapterOpen(uint32_t adapter_handle, uint32_t physical_adapter_index,
               const amdf_allocator_t* host_allocator,
               amdf_wkmi_bridge_gpu_adapter_t** out_adapter,
               amdf_wkmi_bridge_gpu_properties_t* out_properties,
               uint32_t* out_native_status) {
  if (adapter_handle == 0 || physical_adapter_index != 0 ||
      host_allocator == nullptr || host_allocator->allocate == nullptr ||
      host_allocator->free == nullptr || out_adapter == nullptr ||
      out_properties == nullptr || out_native_status == nullptr) {
    return AMDF_WKMI_BRIDGE_RESULT_INVALID_ARGUMENT;
  }
  amdf_wkmi_bridge_gpu_properties_t properties = {};
  properties.gfx_ip_major = 11;
  properties.gfx_ip_minor = 5;
  properties.gfx_ip_stepping = 1;
  properties.asic_revision = 3;
  properties.wavefront_size = 32;
  properties.compute_unit_count = 16;
  properties.maximum_wave_count_per_compute_unit = 32;
  properties.maximum_scratch_wave_count_per_compute_unit = 32;
  properties.local_data_share_byte_length = 64 * 1024;
  properties.xcc_count = 1;
  properties.shader_engine_count = 2;
  properties.supports_pm4_kernel_queue = 1;
  properties.supports_sdma_kernel_queue = 1;
  ++state.adapter_open_success_count;
  *out_adapter = &state.adapter;
  *out_properties = properties;
  *out_native_status = 0;
  return AMDF_WKMI_BRIDGE_RESULT_SUCCESS;
}

amdf_wkmi_bridge_result_t AMDF_WKMI_BRIDGE_CALL GpuAdapterClose(
    amdf_wkmi_bridge_gpu_adapter_t* adapter, uint32_t* out_native_status) {
  if (adapter != &state.adapter || out_native_status == nullptr) {
    return AMDF_WKMI_BRIDGE_RESULT_INVALID_ARGUMENT;
  }
  *out_native_status = 0;
  ++state.adapter_close_attempt_count;
  if (state.adapter_close_failures_remaining != 0) {
    --state.adapter_close_failures_remaining;
    return AMDF_WKMI_BRIDGE_RESULT_BUSY;
  }
  ++state.adapter_close_success_count;
  return AMDF_WKMI_BRIDGE_RESULT_SUCCESS;
}

amdf_wkmi_bridge_result_t AMDF_WKMI_BRIDGE_CALL GpuAllocationQueryLayout(
    amdf_wkmi_bridge_gpu_adapter_t*, uint64_t, uint32_t*, uint64_t*) {
  return AMDF_WKMI_BRIDGE_RESULT_UNSUPPORTED;
}

amdf_wkmi_bridge_result_t AMDF_WKMI_BRIDGE_CALL
GpuAllocationCreate(amdf_wkmi_bridge_gpu_adapter_t*,
                    const amdf_wkmi_bridge_gpu_allocation_create_info_t*,
                    uint32_t, uint32_t*, uint32_t*, uint32_t*, uint32_t*) {
  return AMDF_WKMI_BRIDGE_RESULT_UNSUPPORTED;
}

amdf_wkmi_bridge_result_t AMDF_WKMI_BRIDGE_CALL
GpuKernelQueueCreate(amdf_wkmi_bridge_gpu_adapter_t*,
                     const amdf_wkmi_bridge_gpu_kernel_queue_create_info_t*,
                     amdf_wkmi_bridge_gpu_kernel_queue_t**,
                     amdf_wkmi_bridge_gpu_kernel_queue_info_t*, uint32_t*) {
  return AMDF_WKMI_BRIDGE_RESULT_UNSUPPORTED;
}

amdf_wkmi_bridge_result_t AMDF_WKMI_BRIDGE_CALL
GpuKernelQueueSubmit(amdf_wkmi_bridge_gpu_kernel_queue_t*, uint64_t, uint64_t,
                     uint64_t, uint32_t*) {
  return AMDF_WKMI_BRIDGE_RESULT_UNSUPPORTED;
}

amdf_wkmi_bridge_result_t AMDF_WKMI_BRIDGE_CALL
GpuKernelQueueDestroy(amdf_wkmi_bridge_gpu_kernel_queue_t*, uint32_t*) {
  return AMDF_WKMI_BRIDGE_RESULT_UNSUPPORTED;
}

const amdf_wkmi_bridge_api_t kApi = {
    sizeof(amdf_wkmi_bridge_api_t),
    AMDF_WKMI_BRIDGE_ABI_VERSION_2,
    GpuAdapterOpen,
    GpuAdapterClose,
    GpuAllocationQueryLayout,
    GpuAllocationCreate,
    GpuKernelQueueCreate,
    GpuKernelQueueSubmit,
    GpuKernelQueueDestroy,
};

}  // namespace

extern "C" __declspec(dllexport) amdf_wkmi_bridge_result_t AMDF_WKMI_BRIDGE_CALL
amdf_wkmi_bridge_query_api(uint32_t minimum_version, uint32_t maximum_version,
                           const amdf_wkmi_bridge_api_t** out_api) {
  if (out_api == nullptr) {
    return AMDF_WKMI_BRIDGE_RESULT_INVALID_ARGUMENT;
  }
  if (state.query_result != AMDF_WKMI_BRIDGE_RESULT_SUCCESS) {
    return state.query_result;
  }
  if (minimum_version > AMDF_WKMI_BRIDGE_ABI_VERSION_2 ||
      maximum_version < AMDF_WKMI_BRIDGE_ABI_VERSION_2) {
    return AMDF_WKMI_BRIDGE_RESULT_VERSION_MISMATCH;
  }
  *out_api = &kApi;
  return AMDF_WKMI_BRIDGE_RESULT_SUCCESS;
}

extern "C" __declspec(dllexport) void AMDF_WKMI_BRIDGE_CALL
amdf_test_wkmi_bridge_reset(void) {
  state = {};
  state.query_result = AMDF_WKMI_BRIDGE_RESULT_SUCCESS;
  state.adapter_close_failures_remaining = 1;
}

extern "C" __declspec(dllexport) void AMDF_WKMI_BRIDGE_CALL
amdf_test_wkmi_bridge_set_query_result(amdf_wkmi_bridge_result_t result) {
  state.query_result = result;
}

extern "C" __declspec(dllexport) void AMDF_WKMI_BRIDGE_CALL
amdf_test_wkmi_bridge_set_adapter_close_failures(uint32_t failure_count) {
  state.adapter_close_failures_remaining = failure_count;
}

extern "C" __declspec(dllexport) uint32_t AMDF_WKMI_BRIDGE_CALL
amdf_test_wkmi_bridge_query_adapter_open_success_count(void) {
  return state.adapter_open_success_count;
}

extern "C" __declspec(dllexport) uint32_t AMDF_WKMI_BRIDGE_CALL
amdf_test_wkmi_bridge_query_adapter_close_attempt_count(void) {
  return state.adapter_close_attempt_count;
}

extern "C" __declspec(dllexport) uint32_t AMDF_WKMI_BRIDGE_CALL
amdf_test_wkmi_bridge_query_adapter_close_success_count(void) {
  return state.adapter_close_success_count;
}
