// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_GPU_UMD_DXG_CLOCK_H_
#define AMDF_SRC_GPU_UMD_DXG_CLOCK_H_

#include "libamdf/src/gpu/umd/clock.h"
#include "libamdf/src/gpu/umd/dxg/adapter.h"

#ifdef __cplusplus
extern "C" {
#endif

// Reads native rate metadata through one live DXG adapter.
amdf_status_t amdf_gpu_dxg_query_queue_clock_info(
    const amdf_gpu_dxg_adapter_t* adapter, uint32_t adapter_handle,
    uint32_t physical_adapter_index, amdf_queue_command_type_t command_type,
    amdf_observation_clock_info_t* out_info);

// Correlates the selected DXG queue clock with the portable host clock.
amdf_status_t amdf_gpu_dxg_query_clock_sample(
    const amdf_gpu_dxg_adapter_t* adapter, uint32_t adapter_handle,
    uint32_t physical_adapter_index, amdf_queue_command_type_t command_type,
    amdf_observation_clock_sample_t* out_sample);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AMDF_SRC_GPU_UMD_DXG_CLOCK_H_
