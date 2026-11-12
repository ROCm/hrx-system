// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVER_AMD_GPU_QUEUE_PM4_TRANSFER_H_
#define IREE_HAL_DRIVER_AMD_GPU_QUEUE_PM4_TRANSFER_H_

#include "iree/hal/driver/amd/gpu/queue/dma_transfer.h"
#include "iree/hal/driver/amd/gpu/queue/pm4/launch.h"
#include "iree/hal/driver/amd/gpu/queue/transfer.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

enum {
  // Short download retirements bound host-copy interference.
  IREE_HAL_AMD_GPU_PM4_TRANSFER_MAX_DOWNLOAD_LENGTH = 1024 * 1024,
  // Upload preparation overlaps native reads with two bounded regions.
  IREE_HAL_AMD_GPU_PM4_TRANSFER_MAX_UPLOAD_LENGTH = 16 * 1024 * 1024,
};

// PM4 packet selection for the shared capture and staging traversal.
extern const iree_hal_amd_gpu_dma_transfer_encoding_t
    iree_hal_amd_gpu_pm4_transfer_encoding;

// Emits consecutive COPY operations with the resident blit kernel. Capture
// has proved that every native slot also admits PROGRAM transfer access.
// Each launch owns one 32-byte ABI argument record; no payload is copied by the
// host. Other operation types end the chunk. out_argument_length reports ABI
// bytes independently of the chunk's zero staging length. The caller publishes
// arguments before execution and retains them through shader completion.
iree_status_t iree_hal_amd_gpu_pm4_transfer_prepare_program_chunk(
    const iree_hal_amd_gpu_transfer_t* transfer,
    const iree_hal_amd_gpu_pm4_launch_t* launch, uint32_t word_capacity,
    iree_byte_span_t arguments, uint64_t argument_address,
    iree_hal_amd_gpu_dma_transfer_chunk_t* out_chunk,
    uint32_t* out_argument_length);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVER_AMD_GPU_QUEUE_PM4_TRANSFER_H_
