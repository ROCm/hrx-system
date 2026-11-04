// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_GPU_UMD_WDDM_WKMI_KERNEL_QUEUE_H_
#define AMDF_SRC_GPU_UMD_WDDM_WKMI_KERNEL_QUEUE_H_

#include "libamdf/src/gpu/umd/dxg/bridge_api.h"

namespace amdf::wkmi_bridge {

// Creates one native GPU execution context and its completion resources.
amdf_dxg_bridge_result_t AMDF_DXG_BRIDGE_CALL GpuKernelQueueCreate(
    amdf_dxg_bridge_gpu_adapter_t* adapter,
    const amdf_dxg_bridge_gpu_kernel_queue_create_info_t* create_info,
    amdf_dxg_bridge_gpu_kernel_queue_t** out_queue,
    amdf_dxg_bridge_gpu_kernel_queue_info_t* out_info,
    uint32_t* out_native_status) noexcept;

// Publishes one already-materialized native command stream. Acceptance remains
// set if a subsequent native completion-signal publication fails.
amdf_dxg_bridge_result_t AMDF_DXG_BRIDGE_CALL GpuKernelQueueSubmit(
    amdf_dxg_bridge_gpu_kernel_queue_t* queue, uint64_t command_buffer_address,
    uint64_t command_buffer_byte_length, uint64_t progress_value,
    uint32_t* out_accepted, uint32_t* out_native_status) noexcept;

// Consumes an idle queue on every native result. Unreleased native handles
// leak; no host owner or adapter child remains after a cleanup failure.
amdf_dxg_bridge_result_t AMDF_DXG_BRIDGE_CALL
GpuKernelQueueDestroy(amdf_dxg_bridge_gpu_kernel_queue_t* queue,
                      uint32_t* out_native_status) noexcept;

}  // namespace amdf::wkmi_bridge

#endif  // AMDF_SRC_GPU_UMD_WDDM_WKMI_KERNEL_QUEUE_H_
