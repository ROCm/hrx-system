// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_GPU_UMD_DXG_ROCDXG_KERNEL_QUEUE_H_
#define AMDF_SRC_GPU_UMD_DXG_ROCDXG_KERNEL_QUEUE_H_

#include "libamdf/src/gpu/umd/dxg/bridge_api.h"

namespace amdf::rocdxg_bridge {

// Records the adapter's qualified immutable-stream scheduler profiles.
void InitializeGpuKernelQueueProfiles(
    amdf_dxg_bridge_gpu_adapter_t* adapter) noexcept;

// Creates one native GPU execution context and its completion resources.
amdf_dxg_bridge_result_t AMDF_DXG_BRIDGE_CALL GpuKernelQueueCreate(
    amdf_dxg_bridge_gpu_adapter_t* adapter,
    const amdf_dxg_bridge_gpu_kernel_queue_create_info_t* create_info,
    amdf_dxg_bridge_gpu_kernel_queue_t** out_queue,
    amdf_dxg_bridge_gpu_kernel_queue_info_t* out_info,
    uint32_t* out_native_status) noexcept;

// Publishes one already-materialized native command stream without allocating.
amdf_dxg_bridge_result_t AMDF_DXG_BRIDGE_CALL GpuKernelQueueSubmit(
    amdf_dxg_bridge_gpu_kernel_queue_t* queue, uint64_t command_buffer_address,
    uint64_t command_buffer_byte_length, uint64_t progress_value,
    uint32_t* out_accepted, uint32_t* out_native_status) noexcept;

// Consumes an idle queue after attempting every native cleanup operation.
amdf_dxg_bridge_result_t AMDF_DXG_BRIDGE_CALL
GpuKernelQueueDestroy(amdf_dxg_bridge_gpu_kernel_queue_t* queue,
                      uint32_t* out_native_status) noexcept;

}  // namespace amdf::rocdxg_bridge

#endif  // AMDF_SRC_GPU_UMD_DXG_ROCDXG_KERNEL_QUEUE_H_
