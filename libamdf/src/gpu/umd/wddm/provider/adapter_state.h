// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_GPU_UMD_WDDM_PROVIDER_ADAPTER_STATE_H_
#define AMDF_SRC_GPU_UMD_WDDM_PROVIDER_ADAPTER_STATE_H_

#include <atomic>

#include "libamdf/src/gpu/umd/dxg/private_records.h"
#include "libamdf/src/gpu/umd/dxg/provider_api.h"

// Parsed adapter state shared by live provider objects.
struct amdf_dxg_provider_gpu_adapter_t {
  // Parent-instance allocator copied for all provider-controlled storage.
  amdf_allocator_t host_allocator = {};
  // Immutable facts decoded from the native private adapter records.
  amdf_dxg_private_adapter_info_t private_info = {};
  // Qualified immutable-stream compute scheduler.
  amdf_dxg_private_queue_profile_t pm4_queue_profile = {};
  // Qualified immutable-stream SDMA scheduler.
  amdf_dxg_private_queue_profile_t sdma_queue_profile = {};
  // Number of live queues borrowing this adapter.
  std::atomic<uint32_t> live_queue_count = 0;

  ~amdf_dxg_provider_gpu_adapter_t();
};

namespace amdf::wddm_provider {

// Selects the immutable scheduler profile for one command family.
const amdf_dxg_private_queue_profile_t* SelectGpuKernelQueueProfile(
    const amdf_dxg_provider_gpu_adapter_t* adapter,
    amdf_dxg_provider_gpu_queue_command_type_t command_type) noexcept;

// Verifies that no live queue borrows `adapter`.
amdf_dxg_provider_result_t PrepareGpuAdapterClose(
    amdf_dxg_provider_gpu_adapter_t* adapter,
    uint32_t* out_native_status) noexcept;

}  // namespace amdf::wddm_provider

#endif  // AMDF_SRC_GPU_UMD_WDDM_PROVIDER_ADAPTER_STATE_H_
