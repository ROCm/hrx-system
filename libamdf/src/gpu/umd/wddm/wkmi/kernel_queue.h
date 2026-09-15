// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_GPU_UMD_WDDM_WKMI_KERNEL_QUEUE_H_
#define AMDF_SRC_GPU_UMD_WDDM_WKMI_KERNEL_QUEUE_H_

#include "libamdf/src/gpu/umd/wddm/wkmi/bridge_api.h"

namespace Wkmi {
struct DeviceInfo;
}

namespace amdf::wkmi_bridge {

// Selects a scheduler whose node and hardware-scheduling capability both exist.
// Qualification and construction share this predicate so an advertised family
// cannot select a software-scheduled engine for a hardware queue.
bool SelectGpuHardwareQueueScheduler(
    Wkmi::DeviceInfo& device_info,
    amdf_wkmi_bridge_gpu_queue_command_type_t command_type,
    uint32_t* out_scheduler);

// Creates one native GPU hardware queue and execution context.
amdf_wkmi_bridge_result_t AMDF_WKMI_BRIDGE_CALL GpuKernelQueueCreate(
    amdf_wkmi_bridge_gpu_adapter_t* adapter,
    const amdf_wkmi_bridge_gpu_kernel_queue_create_info_t* create_info,
    amdf_wkmi_bridge_gpu_kernel_queue_t** out_queue,
    amdf_wkmi_bridge_gpu_kernel_queue_info_t* out_info,
    uint32_t* out_native_status) noexcept;

// Publishes one already-materialized native command stream.
amdf_wkmi_bridge_result_t AMDF_WKMI_BRIDGE_CALL GpuKernelQueueSubmit(
    amdf_wkmi_bridge_gpu_kernel_queue_t* queue, uint64_t command_buffer_address,
    uint64_t command_buffer_byte_length, uint64_t progress_value,
    uint32_t* out_native_status) noexcept;

// Releases one native GPU hardware queue and execution context.
amdf_wkmi_bridge_result_t AMDF_WKMI_BRIDGE_CALL
GpuKernelQueueDestroy(amdf_wkmi_bridge_gpu_kernel_queue_t* queue,
                      uint32_t* out_native_status) noexcept;

}  // namespace amdf::wkmi_bridge

#endif  // AMDF_SRC_GPU_UMD_WDDM_WKMI_KERNEL_QUEUE_H_
