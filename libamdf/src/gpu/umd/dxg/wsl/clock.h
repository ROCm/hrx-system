// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_GPU_UMD_DXG_WSL_CLOCK_H_
#define AMDF_SRC_GPU_UMD_DXG_WSL_CLOCK_H_

#include "libamdf/src/gpu/umd/dxg/provider_api.h"

namespace amdf::wsl_provider {

// Samples the native engine clock selected by one command family.
amdf_dxg_provider_result_t GpuClockSample(
    amdf_dxg_provider_gpu_adapter_t* adapter, uint32_t adapter_handle,
    uint32_t physical_adapter_index,
    amdf_dxg_provider_gpu_queue_command_type_t command_type,
    amdf_dxg_provider_gpu_clock_sample_t* out_sample,
    uint32_t* out_native_status) noexcept;

}  // namespace amdf::wsl_provider

#endif  // AMDF_SRC_GPU_UMD_DXG_WSL_CLOCK_H_
