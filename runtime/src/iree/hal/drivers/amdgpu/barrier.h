// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_BARRIER_H_
#define IREE_HAL_DRIVERS_AMDGPU_BARRIER_H_

#include "iree/hal/command_buffer.h"
#include "iree/hal/drivers/amdgpu/abi/queue.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// AMDGPU acquire and release scopes resolved for one HAL execution barrier.
typedef struct iree_hal_amdgpu_barrier_scopes_t {
  // Minimum acquire scope required after the barrier.
  iree_hsa_fence_scope_t acquire;

  // Minimum release scope required before the barrier.
  iree_hsa_fence_scope_t release;
} iree_hal_amdgpu_barrier_scopes_t;

// Resolves HAL execution-barrier semantics into independent HSA fence scopes.
iree_hal_amdgpu_barrier_scopes_t iree_hal_amdgpu_barrier_resolve_scopes(
    iree_hal_execution_stage_t source_stage_mask,
    iree_hal_execution_stage_t target_stage_mask,
    iree_hal_execution_barrier_flags_t flags,
    iree_host_size_t memory_barrier_count,
    const iree_hal_memory_barrier_t* memory_barriers,
    iree_host_size_t buffer_barrier_count,
    const iree_hal_buffer_barrier_t* buffer_barriers);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_BARRIER_H_
