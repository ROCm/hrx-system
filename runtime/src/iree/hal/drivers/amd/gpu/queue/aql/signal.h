// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMD_GPU_QUEUE_AQL_SIGNAL_H_
#define IREE_HAL_DRIVERS_AMD_GPU_QUEUE_AQL_SIGNAL_H_

#include "amdf/gpu.h"
#include "iree/hal/drivers/amd/gpu/queue/aql/packet.h"
#include "iree/hal/drivers/amd/gpu/queue/signal.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Initializes an exclusively owned native extent and attaches its host
// metadata to the existing signal pool. Native storage is 64-byte aligned,
// coherent, and accessible to the admitted AQL signal domain. Each signal
// borrows the same native event through final queue destruction.
//
// The publisher acquires these signals with initial value one. Completion
// decrements to zero; native dependent readers use the descriptor address.
// Signal metadata and causal vectors follow the shared independent-reader
// contract. No native allocation or additional thread is created here.
void iree_hal_amd_gpu_aql_signal_block_initialize(
    iree_hal_amd_gpu_signal_pool_t* pool, const void* domain,
    const amdf_gpu_event_info_t* event,
    iree_hal_amd_gpu_aql_signal_t* native_signals, uint64_t device_address,
    iree_hal_amd_gpu_signal_t* signals, uint32_t capacity, uint32_t queue_count,
    uint64_t* frontiers, iree_hal_amd_gpu_signal_block_t* out_block);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMD_GPU_QUEUE_AQL_SIGNAL_H_
