// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_GPU_UMD_WDDM_WKMI_SHARED_MEMORY_H_
#define AMDF_SRC_GPU_UMD_WDDM_WKMI_SHARED_MEMORY_H_

#include "libamdf/src/gpu/umd/wddm/wkmi/bridge_api.h"

namespace amdf::wkmi_bridge {

// Acquires typed D3D12 backing into an already-live native memory owner.
// Acquired handle slots remain owned by the caller on every result.
amdf_status_t AMDF_WKMI_BRIDGE_CALL GpuMemoryPrepareImport(
    amdf_wkmi_bridge_gpu_adapter_t* adapter,
    const amdf_wkmi_bridge_gpu_memory_import_info_t* import_info,
    uint32_t* resource_handle, uint32_t* allocation_handle,
    uint64_t* out_native_byte_length,
    uint64_t* out_logical_byte_length) noexcept;

}  // namespace amdf::wkmi_bridge

#endif  // AMDF_SRC_GPU_UMD_WDDM_WKMI_SHARED_MEMORY_H_
