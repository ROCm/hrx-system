// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVER_AMD_GPU_QUEUE_SDMA_GROWTH_H_
#define IREE_HAL_DRIVER_AMD_GPU_QUEUE_SDMA_GROWTH_H_

#include "iree/hal/driver/amd/gpu/queue/backing.h"
#include "iree/hal/driver/amd/gpu/queue/sdma/operation.h"

#ifdef __cplusplus
extern "C" {
#endif

// Parks one unpublished operation behind coalesced native signal or private
// data acquisition. Success transfers it to the result continuation. Failure
// retains the operation and starts no callback for it. Called exclusively by
// the native publisher; acquisition runs on the existing allocation service.
iree_status_t iree_hal_amd_gpu_sdma_growth_begin(
    iree_hal_amd_gpu_sdma_operation_t* operation,
    iree_hal_amd_gpu_program_t* program);

// Pending signal backing participates in cold membership expansion before it
// is attached to the device-owned cache. Called by the exact publisher.
iree_hal_amd_gpu_signal_storage_t* iree_hal_amd_gpu_sdma_growth_pending_signals(
    iree_hal_amd_gpu_sdma_queue_t* queue);

// Enqueues one coalesced metadata turn after the device cache reports a return.
// The publisher checks capacity; this notification never accesses pool state.
void iree_hal_amd_gpu_sdma_growth_signals_available(
    iree_hal_amd_gpu_sdma_queue_t* queue);

#ifdef __cplusplus
}  // extern "C"
#endif
#endif  // IREE_HAL_DRIVER_AMD_GPU_QUEUE_SDMA_GROWTH_H_
