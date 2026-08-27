// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_DEVICE_TSAN_H_
#define IREE_HAL_DRIVERS_AMDGPU_DEVICE_TSAN_H_

#include "iree/hal/drivers/amdgpu/abi/tsan.h"
#include "iree/hal/drivers/amdgpu/device/dispatch.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Populates a builtin dispatch packet and kernargs that initializes queue-local
// TSAN state before user work can consume it.
//
// |max_workgroup_count| bounds the resident clear grid. The kernel covers
// larger shadow allocations with a grid-stride loop.
//
// |dispatch_packet| and |kernarg_ptr| must point to reserved queue storage.
// The caller owns packet header commit and barrier placement.
// |max_workgroup_count| must be non-zero.
void iree_hal_amdgpu_device_tsan_emplace_queue_initialize(
    const iree_hal_amdgpu_device_kernel_args_t* IREE_AMDGPU_RESTRICT
        queue_initialize_kernel_args,
    const iree_hal_amdgpu_tsan_queue_initialize_args_t* IREE_AMDGPU_RESTRICT
        queue_initialize_args,
    uint32_t max_workgroup_count,
    iree_hsa_kernel_dispatch_packet_t* IREE_AMDGPU_RESTRICT dispatch_packet,
    void* IREE_AMDGPU_RESTRICT kernarg_ptr);

#if defined(IREE_AMDGPU_TARGET_DEVICE)

// Device builtin that initializes queue-local TSAN state.
IREE_AMDGPU_ATTRIBUTE_KERNEL void
iree_hal_amdgpu_device_tsan_initialize_queue_state(
    iree_hal_amdgpu_tsan_queue_state_t* IREE_AMDGPU_RESTRICT queue_state,
    uint8_t* IREE_AMDGPU_RESTRICT shadow_base, uint64_t shadow_size,
    uint32_t clear_workgroup_size, uint32_t reserved0,
    uint64_t clear_byte_stride,
    const iree_hal_amdgpu_tsan_queue_state_t* IREE_AMDGPU_RESTRICT
        queue_state_template);

#endif  // IREE_AMDGPU_TARGET_DEVICE

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_DEVICE_TSAN_H_
