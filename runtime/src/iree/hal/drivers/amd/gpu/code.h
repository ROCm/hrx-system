// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMD_GPU_CODE_H_
#define IREE_HAL_DRIVERS_AMD_GPU_CODE_H_

#include "iree/hal/drivers/amd/gpu/context.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct iree_hal_amd_gpu_sdma_queue_t iree_hal_amd_gpu_sdma_queue_t;
typedef struct iree_hal_amd_gpu_code_upload_t iree_hal_amd_gpu_code_upload_t;

// Allocates final device-local code and a writable initialization view directly
// on the load caller, independently of queued slab allocation batches. The
// view aliases final backing when qualified; otherwise upload owns temporary
// host storage. Relocations always use the final memory's device address.
// Failure leaves all outputs empty. The caller owns memory independently of
// upload and cancels an unpublished upload before releasing that memory.
iree_status_t iree_hal_amd_gpu_code_allocate(
    iree_hal_amd_gpu_context_t* context,
    const iree_hal_amd_fabric_memory_options_t* resident,
    const iree_hal_amd_fabric_memory_options_t* staging, uint64_t byte_length,
    uint64_t alignment, iree_allocator_t host_allocator,
    iree_hal_amd_fabric_memory_t* out_memory,
    iree_hal_amd_gpu_code_upload_t** out_upload,
    iree_byte_span_t* out_contents);

// Flushes mapped code, or copies staging through the device's existing SDMA
// owner and waits for native retirement. Queue sources must already be live.
// No queue or worker is created here. Large images use bounded native copies.
// This establishes data visibility only. AQL loading separately completes
// instruction-cache invalidation; PM4 emits it before shader use.
//
// Consumes upload on every outcome. Successful copies submit staging release
// without waiting for the native allocator. An accepted-but-unretired failure
// preserves both allocations and clears memory's native handles: the failed
// executable can be destroyed without releasing a possible GPU operand.
// Mapped code needs no queue and allocates no upload metadata.
iree_status_t iree_hal_amd_gpu_code_publish(
    iree_hal_amd_gpu_context_t* context, iree_hal_amd_gpu_sdma_queue_t* queue,
    uint32_t transfer_family_ordinal, uint32_t compute_family_ordinal,
    iree_hal_amd_fabric_memory_t* memory,
    iree_hal_amd_gpu_code_upload_t** upload);

// Releases temporary backing whose native consumers are absent or retired.
// The final code allocation remains owned by its caller. NULL is ignored.
void iree_hal_amd_gpu_code_upload_free(iree_hal_amd_gpu_context_t* context,
                                       iree_hal_amd_gpu_code_upload_t* upload);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_HAL_DRIVERS_AMD_GPU_CODE_H_
