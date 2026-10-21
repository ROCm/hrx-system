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

// Joins prepared effects without inventing the reserved fence scope (3).
// This is a pure arithmetic operation; it consults no topology or device.
uint32_t iree_hal_amd_gpu_aql_barrier_join(uint32_t a, uint32_t b);

// Atomic producer completion/release precedes the scalar carrier; consumer
// acquire follows it. Inapplicable flags have already been removed.
void iree_hal_amd_gpu_aql_barrier_atomic(
    iree_hal_atomic_flags_t flags, iree_hal_execution_stage_t source_stage,
    iree_hal_execution_stage_t target_stage, uint32_t* before, uint32_t* after);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_HAL_DRIVERS_AMD_GPU_QUEUE_AQL_BARRIER_H_
