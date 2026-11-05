// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/gpu/umd/wddm/provider/fake_api.h"

#include <cstdint>

#include "libamdf/src/gpu/umd/dxg/provider_api.h"
#include "libamdf/src/gpu/umd/wddm/provider/api.h"

struct amdf_dxg_provider_gpu_adapter_t {};
struct amdf_dxg_provider_gpu_kernel_queue_t {};

namespace {

struct FakeProviderState {
  // Stable parsed adapter identity returned by successful opens.
  amdf_dxg_provider_gpu_adapter_t adapter;
  // Number of adapter-close operations rejected before native consumption.
  uint32_t adapter_close_failures_remaining = 1;
  // Number of successfully parsed adapters.
  uint32_t adapter_open_success_count = 0;
  // Number of adapter-close attempts.
  uint32_t adapter_close_attempt_count = 0;
  // Number of successfully released adapters.
  uint32_t adapter_close_success_count = 0;
};

FakeProviderState state;

amdf_dxg_provider_result_t GpuAdapterOpen(
    const amdf_dxg_provider_gpu_adapter_open_info_t* open_info,
    const amdf_allocator_t* host_allocator,
    amdf_dxg_provider_gpu_adapter_t** out_adapter,
    amdf_dxg_provider_gpu_properties_t* out_properties,
    uint32_t* out_native_status) {
  if (open_info == nullptr || open_info->adapter_handle == 0 ||
      open_info->physical_adapter_index != 0 || host_allocator == nullptr ||
      host_allocator->allocate == nullptr || host_allocator->free == nullptr ||
      out_adapter == nullptr || out_properties == nullptr ||
      out_native_status == nullptr) {
    return AMDF_DXG_PROVIDER_RESULT_INVALID_ARGUMENT;
  }
  amdf_dxg_provider_gpu_properties_t properties = {};
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
  return AMDF_DXG_PROVIDER_RESULT_SUCCESS;
}

amdf_dxg_provider_result_t GpuAdapterClose(
    amdf_dxg_provider_gpu_adapter_t* adapter, uint32_t* out_native_status) {
  if (adapter != &state.adapter || out_native_status == nullptr) {
    return AMDF_DXG_PROVIDER_RESULT_INVALID_ARGUMENT;
  }
  *out_native_status = 0;
  ++state.adapter_close_attempt_count;
  if (state.adapter_close_failures_remaining != 0) {
    --state.adapter_close_failures_remaining;
    return AMDF_DXG_PROVIDER_RESULT_BUSY;
  }
  ++state.adapter_close_success_count;
  return AMDF_DXG_PROVIDER_RESULT_SUCCESS;
}

amdf_dxg_provider_result_t GpuAllocationQueryLayout(
    amdf_dxg_provider_gpu_adapter_t*, uint64_t, uint32_t*, uint64_t*) {
  return AMDF_DXG_PROVIDER_RESULT_UNSUPPORTED;
}

amdf_dxg_provider_result_t GpuAllocationCreate(
    amdf_dxg_provider_gpu_adapter_t*,
    const amdf_dxg_provider_gpu_allocation_create_info_t*, uint32_t, uint32_t*,
    uint32_t*, uint32_t*, uint32_t*) {
  return AMDF_DXG_PROVIDER_RESULT_UNSUPPORTED;
}

amdf_dxg_provider_result_t GpuKernelQueueCreate(
    amdf_dxg_provider_gpu_adapter_t*,
    const amdf_dxg_provider_gpu_kernel_queue_create_info_t*,
    amdf_dxg_provider_gpu_kernel_queue_t**,
    amdf_dxg_provider_gpu_kernel_queue_info_t*, uint32_t*) {
  return AMDF_DXG_PROVIDER_RESULT_UNSUPPORTED;
}

amdf_dxg_provider_result_t GpuKernelQueueSubmit(
    amdf_dxg_provider_gpu_kernel_queue_t*, uint64_t, uint64_t, uint64_t,
    uint32_t*, uint32_t*) {
  return AMDF_DXG_PROVIDER_RESULT_UNSUPPORTED;
}

amdf_dxg_provider_result_t GpuKernelQueueDestroy(
    amdf_dxg_provider_gpu_kernel_queue_t*, uint32_t*) {
  return AMDF_DXG_PROVIDER_RESULT_UNSUPPORTED;
}

amdf_status_t GpuBufferPrepareImport(
    amdf_dxg_provider_gpu_adapter_t*,
    const amdf_dxg_provider_gpu_buffer_import_info_t*, uint32_t*, uint32_t*,
    uint64_t*, uint64_t*) {
  return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
}

amdf_dxg_provider_result_t GpuClockSample(
    amdf_dxg_provider_gpu_adapter_t*, uint32_t, uint32_t,
    amdf_dxg_provider_gpu_queue_command_type_t,
    amdf_dxg_provider_gpu_clock_sample_t*, uint32_t*) {
  return AMDF_DXG_PROVIDER_RESULT_UNSUPPORTED;
}

amdf_dxg_provider_api_t api = {
    GpuAdapterOpen,        GpuAdapterClose,        GpuAllocationQueryLayout,
    GpuAllocationCreate,   GpuKernelQueueCreate,   GpuKernelQueueSubmit,
    GpuKernelQueueDestroy, GpuBufferPrepareImport, GpuClockSample,
};

}  // namespace

extern "C" const amdf_dxg_provider_api_t* amdf_gpu_wddm_default_api(void) {
  return &api;
}

extern "C" void amdf_test_wddm_provider_reset(void) {
  state = {};
  state.adapter_close_failures_remaining = 1;
}

extern "C" void amdf_test_wddm_provider_set_adapter_close_failures(
    uint32_t failure_count) {
  state.adapter_close_failures_remaining = failure_count;
}

extern "C" uint32_t amdf_test_wddm_provider_query_adapter_open_success_count(
    void) {
  return state.adapter_open_success_count;
}

extern "C" uint32_t amdf_test_wddm_provider_query_adapter_close_attempt_count(
    void) {
  return state.adapter_close_attempt_count;
}

extern "C" uint32_t amdf_test_wddm_provider_query_adapter_close_success_count(
    void) {
  return state.adapter_close_success_count;
}
