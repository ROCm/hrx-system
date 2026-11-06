// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVER_AMDGPU_BARRIER_H_
#define IREE_HAL_DRIVER_AMDGPU_BARRIER_H_

#include "iree/hal/atomic.h"
#include "iree/hal/command_buffer.h"
#include "iree/hal/driver/amdgpu/abi/queue.h"

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

// Captured global scopes for one direct queue boundary. HSA fence scope values
// fit in a byte; descriptors and buffer ranges need no retained storage after
// this backend promotes their dependencies to global cache operations.
typedef struct iree_hal_amdgpu_queue_barrier_t {
  // HSA acquire scope at this boundary.
  uint8_t acquire;
  // HSA release scope at this boundary.
  uint8_t release;
} iree_hal_amdgpu_queue_barrier_t;

// Captured visibility surrounding the entire logical queue operation.
typedef struct iree_hal_amdgpu_queue_barriers_t {
  // Dependency after waits and before the operation.
  iree_hal_amdgpu_queue_barrier_t before;
  // Dependency after all operation children and before signals.
  iree_hal_amdgpu_queue_barrier_t after;
} iree_hal_amdgpu_queue_barriers_t;

// Resolves validated direct-operation boundaries. Missing boundaries use
// conservative system scopes; explicit empty lists have no payload scopes.
iree_hal_amdgpu_queue_barriers_t iree_hal_amdgpu_queue_barriers_resolve(
    const iree_hal_queue_barriers_t* barriers);

// Resolves HAL execution-barrier semantics into independent HSA fence scopes.
iree_hal_amdgpu_barrier_scopes_t iree_hal_amdgpu_barrier_resolve_scopes(
    iree_hal_execution_stage_t source_stage_mask,
    iree_hal_execution_stage_t target_stage_mask,
    iree_hal_barrier_flags_t flags, iree_host_size_t memory_barrier_count,
    const iree_hal_memory_barrier_t* memory_barriers,
    iree_host_size_t buffer_barrier_count,
    const iree_hal_buffer_barrier_t* buffer_barriers);

// Resolves one atomic ordering handoff into its minimum HSA fence scope.
// Returns no fence when |atomic_flags| does not contain |ordering_flag|.
iree_hsa_fence_scope_t iree_hal_amdgpu_barrier_resolve_atomic_handoff_scope(
    iree_hal_execution_stage_t stage_mask, iree_hal_atomic_flags_t atomic_flags,
    iree_hal_atomic_flags_t ordering_flag);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVER_AMDGPU_BARRIER_H_
