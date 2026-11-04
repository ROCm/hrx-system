// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_GPU_UMD_WDDM_PROVIDER_SHARED_BUFFER_H_
#define AMDF_SRC_GPU_UMD_WDDM_PROVIDER_SHARED_BUFFER_H_

#include "libamdf/src/gpu/umd/dxg/bridge_api.h"

namespace amdf::wddm_provider {

// Acquires a typed D3D12 buffer into an already-live native memory owner.
// Acquired handle slots remain owned by the caller on every result.
amdf_status_t AMDF_DXG_BRIDGE_CALL GpuBufferPrepareImport(
    amdf_dxg_bridge_gpu_adapter_t* adapter,
    const amdf_dxg_bridge_gpu_buffer_import_info_t* import_info,
    uint32_t* resource_handle, uint32_t* allocation_handle,
    uint64_t* out_native_byte_length,
    uint64_t* out_buffer_byte_length) noexcept;

}  // namespace amdf::wddm_provider

#endif  // AMDF_SRC_GPU_UMD_WDDM_PROVIDER_SHARED_BUFFER_H_
