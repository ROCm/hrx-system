// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMD_GPU_QUEUE_PM4_BACKING_H_
#define IREE_HAL_DRIVERS_AMD_GPU_QUEUE_PM4_BACKING_H_

#include "iree/hal/drivers/amd/gpu/queue/backing.h"

#ifdef __cplusplus
extern "C" {
#endif

// Attaches acquired eight-byte PM4 native cells to their shared host owners.
iree_status_t iree_hal_amd_gpu_pm4_signal_storage_initialize(
    iree_hal_amd_gpu_context_t* context, iree_hal_amd_gpu_signal_pool_t* pool,
    iree_hal_amd_gpu_signal_storage_t* storage);

// Allocates coherent native cells, inline host reference metadata, and a dense
// queue_count-entry frontier per cell. Both counts are already admitted by
// device/queue construction. No cell is initially acquired.
// Success returns the unique owner; failure leaves out_storage NULL.
iree_status_t iree_hal_amd_gpu_pm4_signal_storage_allocate(
    iree_hal_amd_gpu_context_t* context, iree_hal_amd_gpu_signal_pool_t* pool,
    const iree_hal_amd_fabric_memory_options_t* options, uint32_t capacity,
    uint32_t queue_count, iree_allocator_t host_allocator,
    iree_hal_amd_gpu_signal_storage_t** out_storage);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_HAL_DRIVERS_AMD_GPU_QUEUE_PM4_BACKING_H_
