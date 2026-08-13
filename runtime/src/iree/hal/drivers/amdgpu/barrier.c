// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/barrier.h"

static iree_hsa_fence_scope_t iree_hal_amdgpu_barrier_max_scope(
    iree_hsa_fence_scope_t lhs, iree_hsa_fence_scope_t rhs) {
  return lhs > rhs ? lhs : rhs;
}

static void iree_hal_amdgpu_barrier_accumulate_access_scopes(
    iree_hal_access_scope_t source_scope, iree_hal_access_scope_t target_scope,
    iree_hal_amdgpu_barrier_scopes_t* scopes) {
  if (source_scope != 0) {
    scopes->release = iree_hal_amdgpu_barrier_max_scope(
        scopes->release, IREE_HSA_FENCE_SCOPE_AGENT);
  }
  if (target_scope != 0) {
    scopes->acquire = iree_hal_amdgpu_barrier_max_scope(
        scopes->acquire, IREE_HSA_FENCE_SCOPE_AGENT);
  }
}

iree_hal_amdgpu_barrier_scopes_t iree_hal_amdgpu_barrier_resolve_scopes(
    iree_hal_execution_stage_t source_stage_mask,
    iree_hal_execution_stage_t target_stage_mask,
    iree_hal_execution_barrier_flags_t flags,
    iree_host_size_t memory_barrier_count,
    const iree_hal_memory_barrier_t* memory_barriers,
    iree_host_size_t buffer_barrier_count,
    const iree_hal_buffer_barrier_t* buffer_barriers) {
  iree_hal_amdgpu_barrier_scopes_t scopes = {
      .acquire = IREE_HSA_FENCE_SCOPE_NONE,
      .release = IREE_HSA_FENCE_SCOPE_NONE,
  };

  if (iree_any_bit_set(source_stage_mask, IREE_HAL_EXECUTION_STAGE_HOST) ||
      iree_any_bit_set(flags,
                       IREE_HAL_EXECUTION_BARRIER_FLAG_ACQUIRE_SYSTEM_SCOPE)) {
    scopes.acquire = IREE_HSA_FENCE_SCOPE_SYSTEM;
  }
  if (iree_any_bit_set(target_stage_mask, IREE_HAL_EXECUTION_STAGE_HOST) ||
      iree_any_bit_set(flags,
                       IREE_HAL_EXECUTION_BARRIER_FLAG_RELEASE_SYSTEM_SCOPE)) {
    scopes.release = IREE_HSA_FENCE_SCOPE_SYSTEM;
  }

  for (iree_host_size_t i = 0; i < memory_barrier_count; ++i) {
    iree_hal_amdgpu_barrier_accumulate_access_scopes(
        memory_barriers[i].source_scope, memory_barriers[i].target_scope,
        &scopes);
  }
  for (iree_host_size_t i = 0; i < buffer_barrier_count; ++i) {
    iree_hal_amdgpu_barrier_accumulate_access_scopes(
        buffer_barriers[i].source_scope, buffer_barriers[i].target_scope,
        &scopes);
  }

  return scopes;
}
