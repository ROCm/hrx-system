// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_GPU_UMD_WDDM_WKMI_ALLOCATION_H_
#define AMDF_SRC_GPU_UMD_WDDM_WKMI_ALLOCATION_H_

#include "libamdf/src/gpu/umd/wddm/wkmi/bridge_api.h"

namespace amdf::wkmi_bridge {

// Queries the native allocation layout for an aggregate byte length.
amdf_wkmi_bridge_result_t AMDF_WKMI_BRIDGE_CALL
GpuAllocationQueryLayout(amdf_wkmi_bridge_gpu_adapter_t* adapter,
                         uint64_t byte_length, uint32_t* out_allocation_count,
                         uint64_t* out_maximum_allocation_byte_length) noexcept;

// Creates one grouped set of native allocations through pinned WKMI.
amdf_wkmi_bridge_result_t AMDF_WKMI_BRIDGE_CALL GpuAllocationCreate(
    amdf_wkmi_bridge_gpu_adapter_t* adapter,
    const amdf_wkmi_bridge_gpu_allocation_create_info_t* create_info,
    uint32_t allocation_handle_capacity, uint32_t* out_allocation_handles,
    uint32_t* out_resource_handle, uint32_t* out_allocation_count,
    uint32_t* out_native_status) noexcept;

}  // namespace amdf::wkmi_bridge

#endif  // AMDF_SRC_GPU_UMD_WDDM_WKMI_ALLOCATION_H_
