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

// Populates a builtin dispatch packet and kernargs that clears queue-local TSAN
// shadow storage before user work can consume it.
//
// |max_workgroup_count| bounds the resident clear grid. The kernel covers
// larger shadow allocations with a grid-stride loop.
//
// |dispatch_packet| and |kernarg_ptr| must point to reserved queue storage.
// The packet's first word is preserved; |out_setup| receives its setup bits in
// private storage for the caller's atomic publication.
// The caller owns packet header commit and barrier placement.
// |max_workgroup_count| must be non-zero.
void iree_hal_amdgpu_device_tsan_emplace_shadow_initialize(
    const iree_hal_amdgpu_device_kernel_args_t* IREE_AMDGPU_RESTRICT
        shadow_initialize_kernel_args,
    const iree_hal_amdgpu_tsan_shadow_initialize_args_t* IREE_AMDGPU_RESTRICT
        shadow_initialize_args,
    uint32_t max_workgroup_count,
    iree_hsa_kernel_dispatch_packet_t* IREE_AMDGPU_RESTRICT dispatch_packet,
    void* IREE_AMDGPU_RESTRICT kernarg_ptr, uint16_t* out_setup);

#if defined(IREE_AMDGPU_TARGET_DEVICE)

// Device builtin that clears queue-local TSAN shadow storage.
IREE_AMDGPU_ATTRIBUTE_KERNEL void iree_hal_amdgpu_device_tsan_initialize_shadow(
    uint8_t* IREE_AMDGPU_RESTRICT shadow_base, uint64_t shadow_size,
    uint32_t clear_workgroup_size, uint32_t reserved0,
    uint64_t clear_byte_stride);

#endif  // IREE_AMDGPU_TARGET_DEVICE

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_DEVICE_TSAN_H_
