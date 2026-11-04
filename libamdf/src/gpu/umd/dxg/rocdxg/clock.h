// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_GPU_UMD_DXG_ROCDXG_CLOCK_H_
#define AMDF_SRC_GPU_UMD_DXG_ROCDXG_CLOCK_H_

#include "libamdf/src/gpu/umd/dxg/bridge_api.h"

namespace amdf::rocdxg_bridge {

// Samples the native engine clock selected by one command family.
amdf_dxg_bridge_result_t AMDF_DXG_BRIDGE_CALL
GpuClockSample(amdf_dxg_bridge_gpu_adapter_t* adapter, uint32_t adapter_handle,
               uint32_t physical_adapter_index,
               amdf_dxg_bridge_gpu_queue_command_type_t command_type,
               amdf_dxg_bridge_gpu_clock_sample_t* out_sample,
               uint32_t* out_native_status) noexcept;

}  // namespace amdf::rocdxg_bridge

#endif  // AMDF_SRC_GPU_UMD_DXG_ROCDXG_CLOCK_H_
