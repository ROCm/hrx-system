// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVER_AMD_GPU_QUEUE_TRANSFER_MEMORY_H_
#define IREE_HAL_DRIVER_AMD_GPU_QUEUE_TRANSFER_MEMORY_H_

#include "iree/hal/driver/amd/fabric/memory.h"
#include "iree/hal/driver/amd/gpu/context.h"

#ifdef __cplusplus
extern "C" {
#endif

// Family-owned placement for explicit upload/download payloads. Both readers
// use the same coherent backing contract; the queue retains each generation
// through native completion and the final host copy. Kernel ABI and native
// command/signal storage have independent policies and lifetimes.
typedef struct iree_hal_amd_gpu_transfer_memory_t {
  // Stable device attachment requirements borrowed by options.
  amdf_memory_device_access_t access;
  // Prepared construction facts, borrowed by initial and deferred acquisition.
  iree_hal_amd_fabric_memory_options_t options;
} iree_hal_amd_gpu_transfer_memory_t;

// Selects host write-back, GPU-uncached SYSTEM backing with host/device
// read-write access for one admitted queue family. Payloads are streamed data;
// kernel arguments retain their independent cache policy. Both visibility
// directions must use that family's existing system acquire/release operations,
// without host cache maintenance. Queries passive capabilities only; no backing
// is acquired. On success out_memory contains a self-reference and remains at
// this address until all borrowing queues and native acquisition callbacks have
// finished.
iree_status_t iree_hal_amd_gpu_transfer_memory_initialize(
    const iree_hal_amd_gpu_context_t* context, uint32_t family_ordinal,
    iree_hal_amd_gpu_transfer_memory_t* out_memory);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_HAL_DRIVER_AMD_GPU_QUEUE_TRANSFER_MEMORY_H_
