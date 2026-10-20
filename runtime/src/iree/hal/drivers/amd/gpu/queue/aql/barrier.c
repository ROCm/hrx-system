// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/gpu/queue/aql/barrier.h"

uint32_t iree_hal_amd_gpu_aql_barrier_prepare(
    const iree_hal_barrier_list_t* barriers) {
  if (!barriers) {
    return IREE_HAL_AMD_GPU_AQL_HEADER_ORDERED |
           IREE_HAL_AMD_GPU_AQL_HEADER_ACQUIRE_SYSTEM |
           IREE_HAL_AMD_GPU_AQL_HEADER_RELEASE_SYSTEM;
  }
  uint32_t acquire = 0;
  uint32_t release = 0;
  uint32_t ordered = 0;
  for (iree_host_size_t i = 0; i < barriers->count; ++i) {
    const iree_hal_barrier_t* barrier = &barriers->values[i];
    iree_hal_access_scope_t source = 0;
    iree_hal_access_scope_t target = 0;
    for (iree_host_size_t j = 0; j < barrier->memory_barrier_count; ++j) {
      source |= barrier->memory_barriers[j].source_scope;
      target |= barrier->memory_barriers[j].target_scope;
    }
    for (iree_host_size_t j = 0; j < barrier->buffer_barrier_count; ++j) {
      source |= barrier->buffer_barriers[j].source_scope;
      target |= barrier->buffer_barriers[j].target_scope;
    }
    const iree_hal_access_scope_t external =
        IREE_HAL_ACCESS_SCOPE_HOST_READ | IREE_HAL_ACCESS_SCOPE_HOST_WRITE |
        IREE_HAL_ACCESS_SCOPE_MEMORY_READ | IREE_HAL_ACCESS_SCOPE_MEMORY_WRITE;
    if (source) {
      release = iree_max(release, 1u);
    }
    if (target) {
      acquire = iree_max(acquire, 1u);
    }
    if ((source & external) ||
        (barrier->flags & IREE_HAL_BARRIER_FLAG_ACQUIRE_SYSTEM_SCOPE) ||
        (barrier->effects.bits &
         (IREE_HAL_MEMORY_EFFECT_GLOBAL_ACQUIRE_FROM_SYSTEM |
          IREE_HAL_MEMORY_EFFECT_RANGE_ACQUIRE_FROM_SYSTEM))) {
      acquire = 2;
    }
    if ((target & external) ||
        (barrier->flags & IREE_HAL_BARRIER_FLAG_RELEASE_SYSTEM_SCOPE) ||
        (barrier->effects.bits &
         (IREE_HAL_MEMORY_EFFECT_GLOBAL_RELEASE_TO_SYSTEM |
          IREE_HAL_MEMORY_EFFECT_RANGE_RELEASE_TO_SYSTEM))) {
      release = 2;
    }
    if ((barrier->source_stage_mask & ~IREE_HAL_EXECUTION_STAGE_HOST) ||
        source || acquire || release) {
      ordered = IREE_HAL_AMD_GPU_AQL_HEADER_ORDERED;
    }
  }
  return ordered | (acquire << 9) | (release << 11);
}

uint32_t iree_hal_amd_gpu_aql_barrier_join(uint32_t a, uint32_t b) {
  return ((a | b) & IREE_HAL_AMD_GPU_AQL_HEADER_ORDERED) |
         iree_max(a & (3u << 9), b & (3u << 9)) |
         iree_max(a & (3u << 11), b & (3u << 11));
}
