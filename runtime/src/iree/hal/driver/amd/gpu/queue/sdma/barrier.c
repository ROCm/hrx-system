// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/driver/amd/gpu/queue/sdma/barrier.h"

iree_hal_amd_gpu_sdma_barrier_profile_t iree_hal_amd_gpu_sdma_barrier_profile(
    amdf_queue_format_features_t features,
    amdf_queue_publication_modes_t publication_mode) {
  const bool stream = publication_mode == AMDF_QUEUE_PUBLICATION_MODE_USER &&
                      (features & AMDF_GPU_SDMA_FORMAT_FEATURE_USER_GCR);
  return (iree_hal_amd_gpu_sdma_barrier_profile_t){
      .acquire = stream ? 0xc3c0u : 0,
      .release = stream ? 0x8040u : 0,
      .submission = publication_mode == AMDF_QUEUE_PUBLICATION_MODE_KERNEL &&
                    (features & AMDF_GPU_SDMA_FORMAT_FEATURE_SUBMISSION_GCR),
  };
}

iree_hal_amd_gpu_sdma_barrier_t iree_hal_amd_gpu_sdma_barrier_effects(
    iree_hal_memory_effects_t effects) {
  iree_hal_amd_gpu_sdma_barrier_t result = 0;
  if (effects.bits & (IREE_HAL_MEMORY_EFFECT_GLOBAL_RELEASE_TO_SYSTEM |
                      IREE_HAL_MEMORY_EFFECT_RANGE_RELEASE_TO_SYSTEM)) {
    result |= IREE_HAL_AMD_GPU_SDMA_BARRIER_JOIN |
              IREE_HAL_AMD_GPU_SDMA_BARRIER_RELEASE;
  }
  if (effects.bits & (IREE_HAL_MEMORY_EFFECT_GLOBAL_ACQUIRE_FROM_SYSTEM |
                      IREE_HAL_MEMORY_EFFECT_RANGE_ACQUIRE_FROM_SYSTEM)) {
    result |= IREE_HAL_AMD_GPU_SDMA_BARRIER_ACQUIRE;
  }
  return result;
}

iree_hal_amd_gpu_sdma_barrier_t iree_hal_amd_gpu_sdma_barrier_prepare(
    const iree_hal_barrier_list_t* barriers) {
  if (!barriers) {
    return IREE_HAL_AMD_GPU_SDMA_BARRIER_JOIN |
           IREE_HAL_AMD_GPU_SDMA_BARRIER_ACQUIRE |
           IREE_HAL_AMD_GPU_SDMA_BARRIER_RELEASE;
  }
  iree_hal_amd_gpu_sdma_barrier_t result = 0;
  for (iree_host_size_t i = 0; i < barriers->count; ++i) {
    const iree_hal_barrier_t* barrier = &barriers->values[i];
    result |= iree_hal_amd_gpu_sdma_barrier_effects(barrier->effects);
    if (barrier->flags & IREE_HAL_BARRIER_FLAG_RELEASE_SYSTEM_SCOPE) {
      result |= IREE_HAL_AMD_GPU_SDMA_BARRIER_RELEASE |
                IREE_HAL_AMD_GPU_SDMA_BARRIER_JOIN;
    }
    if (barrier->flags & IREE_HAL_BARRIER_FLAG_ACQUIRE_SYSTEM_SCOPE) {
      result |= IREE_HAL_AMD_GPU_SDMA_BARRIER_ACQUIRE;
    }
    iree_hal_access_scope_t source = 0, target = 0;
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
    if (source & external) {
      result |= IREE_HAL_AMD_GPU_SDMA_BARRIER_ACQUIRE;
    }
    if (target & external) {
      result |= IREE_HAL_AMD_GPU_SDMA_BARRIER_RELEASE;
    }
    if ((source && target) ||
        (barrier->source_stage_mask & ~IREE_HAL_EXECUTION_STAGE_HOST) ||
        (result & IREE_HAL_AMD_GPU_SDMA_BARRIER_RELEASE)) {
      result |= IREE_HAL_AMD_GPU_SDMA_BARRIER_JOIN;
    }
  }
  return result;
}
