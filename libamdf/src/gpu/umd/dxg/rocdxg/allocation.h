// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_GPU_UMD_DXG_ROCDXG_ALLOCATION_H_
#define AMDF_SRC_GPU_UMD_DXG_ROCDXG_ALLOCATION_H_

#include "libamdf/src/gpu/umd/dxg/bridge_api.h"

namespace amdf::rocdxg_bridge {

amdf_dxg_bridge_result_t AMDF_DXG_BRIDGE_CALL
GpuAllocationQueryLayout(amdf_dxg_bridge_gpu_adapter_t* adapter,
                         uint64_t byte_length, uint32_t* out_allocation_count,
                         uint64_t* out_maximum_allocation_byte_length) noexcept;

amdf_dxg_bridge_result_t AMDF_DXG_BRIDGE_CALL GpuAllocationCreate(
    amdf_dxg_bridge_gpu_adapter_t* adapter,
    const amdf_dxg_bridge_gpu_allocation_create_info_t* create_info,
    uint32_t allocation_handle_capacity, uint32_t* out_allocation_handles,
    uint32_t* out_resource_handle, uint32_t* out_allocation_count,
    uint32_t* out_native_status) noexcept;

}  // namespace amdf::rocdxg_bridge

#endif  // AMDF_SRC_GPU_UMD_DXG_ROCDXG_ALLOCATION_H_
