// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMD_GPU_QUEUE_AQL_BARRIER_H_
#define IREE_HAL_DRIVERS_AMD_GPU_QUEUE_AQL_BARRIER_H_

#include "iree/hal/command_buffer.h"
#include "iree/hal/drivers/amd/gpu/queue/aql/packet.h"
#include "iree/hal/queue.h"

#ifdef __cplusplus
extern "C" {
#endif

// Resolves an admitted HAL boundary to native ordering/fence bits. NULL is a
// full system boundary; an explicit empty list has no payload effects. Ranged
// effects promote to the native whole-cache scope. The caller places the
// acquire and release at their respective operation boundaries.
uint32_t iree_hal_amd_gpu_aql_barrier_prepare(
    const iree_hal_barrier_list_t* barriers);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_HAL_DRIVERS_AMD_GPU_QUEUE_AQL_BARRIER_H_
