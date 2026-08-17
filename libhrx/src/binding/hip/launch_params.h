// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef HRX_BINDING_HIP_LAUNCH_PARAMS_H_
#define HRX_BINDING_HIP_LAUNCH_PARAMS_H_

#include "api.h"
#include "common/internal.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

hipError_t iree_hip_parse_launch_extra(void** extra, void** out_buffer,
                                       size_t* out_buffer_size);

hipError_t iree_hip_validate_launch_block_configuration(
    iree_hal_streaming_device_t* device, iree_hal_streaming_symbol_t* symbol,
    unsigned int block_dim_x, unsigned int block_dim_y,
    unsigned int block_dim_z, size_t shared_memory_bytes);

hipError_t iree_hip_validate_launch_configuration(
    iree_hal_streaming_device_t* device, iree_hal_streaming_symbol_t* symbol,
    unsigned int grid_dim_x, unsigned int grid_dim_y, unsigned int grid_dim_z,
    unsigned int block_dim_x, unsigned int block_dim_y,
    unsigned int block_dim_z, size_t shared_memory_bytes);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // HRX_BINDING_HIP_LAUNCH_PARAMS_H_
