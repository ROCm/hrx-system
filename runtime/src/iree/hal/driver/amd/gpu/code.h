// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVER_AMD_GPU_CODE_H_
#define IREE_HAL_DRIVER_AMD_GPU_CODE_H_

#include "iree/hal/driver/amd/gpu/code_queue.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct iree_hal_amd_gpu_code_upload_t iree_hal_amd_gpu_code_upload_t;

// Allocates final device-local code and a writable initialization view directly
// on the load caller, independently of queued slab allocation batches. The
// view aliases final backing when qualified; otherwise upload owns temporary
// ordinary host bytes. Relocations always use the final memory's device
// address. Native code extent is padded to a DWORD and upload padding is
// zeroed. Failure leaves all outputs empty. The caller owns memory
// independently of upload and cancels an unpublished upload before releasing
// that memory.
iree_status_t iree_hal_amd_gpu_code_allocate(
    iree_hal_amd_gpu_context_t* context,
    const iree_hal_amd_fabric_memory_options_t* resident, uint64_t byte_length,
    uint64_t alignment, iree_allocator_t host_allocator,
    iree_hal_amd_fabric_memory_t* out_memory,
    iree_hal_amd_gpu_code_upload_t** out_upload,
    iree_byte_span_t* out_contents);

// Flushes mapped code or uploads through the selected compute family's cold
// code publisher. A non-NULL publisher also completes instruction-cache
// invalidation. No queue or worker is created here. Unmapped code requires a
// publisher; mapped PM4 code can omit it and invalidate before shader use.
//
// Consumes upload on every outcome. Native commands capture input bytes, so no
// temporary native allocation remains borrowed after failure. An accepted but
// unretired failure transfers memory's existing handles into the code publisher
// until proven native stop. Executable rollback cannot free that native
// operand. Mapped code allocates no upload metadata.
iree_status_t iree_hal_amd_gpu_code_publish(
    iree_hal_amd_gpu_context_t* context, iree_hal_amd_gpu_code_queue_t* queue,
    iree_hal_amd_fabric_memory_t* memory,
    iree_hal_amd_gpu_code_upload_t** upload);

// Releases ordinary temporary host bytes, never borrowed by native execution.
// The final code allocation remains owned by its caller. NULL is ignored.
void iree_hal_amd_gpu_code_upload_free(iree_hal_amd_gpu_code_upload_t* upload);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_HAL_DRIVER_AMD_GPU_CODE_H_
