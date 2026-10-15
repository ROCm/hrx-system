// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/gpu/queue/pm4/barrier.h"

#include "iree/hal/drivers/amd/gpu/queue/pm4/commands.h"

// ACQUIRE_MEM GCR fields. Bit 6 writes back scalar K$ on GFX11/GFX12.0 and
// vector V$ on GFX12.5. Whole-cache promotion must preserve either kind of
// dirty line; a graphics compiler's read-only scalar-cache premise does not
// apply to arbitrary HIP/Loom code. Forward order carries lower writeback
// through L2.
enum {
  IREE_HAL_AMD_GPU_PM4_GCR_GLM_INV = 1u << 5,
  IREE_HAL_AMD_GPU_PM4_GCR_LOWER_WB = 1u << 6,
  IREE_HAL_AMD_GPU_PM4_GCR_GLK_INV = 1u << 7,
  IREE_HAL_AMD_GPU_PM4_GCR_GLV_INV = 1u << 8,
  IREE_HAL_AMD_GPU_PM4_GCR_GL1_INV = 1u << 9,
  IREE_HAL_AMD_GPU_PM4_GCR_GL2_INV = 1u << 14,
  IREE_HAL_AMD_GPU_PM4_GCR_GL2_WB = 1u << 15,
  IREE_HAL_AMD_GPU_PM4_GCR_SEQ_FORWARD = 1u << 16,
  IREE_HAL_AMD_GPU_PM4_GCR_GL2_SCOPE_ALL = 2u << 4,
};

iree_status_t iree_hal_amd_gpu_pm4_barrier_profile_initialize(
    iree_hal_amd_gpu_gfxip_version_t gfxip,
    iree_hal_amd_gpu_pm4_barrier_profile_t* out_profile) {
  *out_profile = (iree_hal_amd_gpu_pm4_barrier_profile_t){0};
  const bool gfx11 = gfxip.major == 11 &&
                     (gfxip.minor == 0 || gfxip.minor == 5 || gfxip.minor == 7);
  const bool gfx12 = gfxip.major == 12 && gfxip.minor == 0;
  const bool gfx125 = gfxip.major == 12 && gfxip.minor == 5;
  if (!gfx11 && !gfx12 && !gfx125) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "PM4 cache encoding is unavailable for gfx%u.%u.%u",
                            gfxip.major, gfxip.minor, gfxip.stepping);
  }
  const uint32_t lower_writeback =
      IREE_HAL_AMD_GPU_PM4_GCR_LOWER_WB | IREE_HAL_AMD_GPU_PM4_GCR_SEQ_FORWARD;
  const uint32_t shader = lower_writeback | IREE_HAL_AMD_GPU_PM4_GCR_GLK_INV |
                          IREE_HAL_AMD_GPU_PM4_GCR_GLV_INV |
                          (gfx11 ? IREE_HAL_AMD_GPU_PM4_GCR_GLM_INV |
                                       IREE_HAL_AMD_GPU_PM4_GCR_GL1_INV
                                 : 0);
  const uint32_t system_scope =
      gfx125 ? IREE_HAL_AMD_GPU_PM4_GCR_GL2_SCOPE_ALL : 0;
  *out_profile = (iree_hal_amd_gpu_pm4_barrier_profile_t){
      .shader = shader,
      .release =
          lower_writeback | IREE_HAL_AMD_GPU_PM4_GCR_GL2_WB | system_scope,
      .acquire = shader | IREE_HAL_AMD_GPU_PM4_GCR_GL2_WB |
                 IREE_HAL_AMD_GPU_PM4_GCR_GL2_INV | system_scope,
      .command_processor_bypasses_l2 = !gfx11,
  };
  return iree_ok_status();
}

static uint32_t iree_hal_amd_gpu_pm4_barrier_execution(
    iree_hal_execution_stage_t stages) {
  if (stages & (IREE_HAL_EXECUTION_STAGE_COMMAND_PROCESS |
                IREE_HAL_EXECUTION_STAGE_TRANSFER |
                IREE_HAL_EXECUTION_STAGE_COMMAND_RETIRE)) {
    return IREE_HAL_AMD_GPU_PM4_BARRIER_EXECUTION_COMPUTE |
           IREE_HAL_AMD_GPU_PM4_BARRIER_EXECUTION_DMA;
  }
  // MEC writes, copies, and atomics use confirmed forms. Only asynchronous
  // shader and CP-DMA work needs an additional join at this boundary.
  return (stages & IREE_HAL_EXECUTION_STAGE_DISPATCH)
             ? IREE_HAL_AMD_GPU_PM4_BARRIER_EXECUTION_COMPUTE
             : 0;
}

iree_hal_amd_gpu_pm4_barrier_t iree_hal_amd_gpu_pm4_barrier_prepare(
    const iree_hal_amd_gpu_pm4_barrier_profile_t* profile,
    const iree_hal_barrier_list_t* barriers) {
  if (!barriers) {
    return (iree_hal_amd_gpu_pm4_barrier_t){
        .cache_control = profile->acquire | profile->release,
        .execution = IREE_HAL_AMD_GPU_PM4_BARRIER_EXECUTION_COMPUTE |
                     IREE_HAL_AMD_GPU_PM4_BARRIER_EXECUTION_DMA,
    };
  }
  iree_hal_amd_gpu_pm4_barrier_t result = {0};
  for (iree_host_size_t i = 0; i < barriers->count; ++i) {
    const iree_hal_barrier_t* barrier = &barriers->values[i];
    iree_hal_barrier_flags_t flags = barrier->flags;
    // Validation proves that the recipes cover all ranged effects and use the
    // queue executor. Global promotion needs only their already captured bits.
    if (barrier->effects.bits &
        (IREE_HAL_MEMORY_EFFECT_GLOBAL_RELEASE_TO_SYSTEM |
         IREE_HAL_MEMORY_EFFECT_RANGE_RELEASE_TO_SYSTEM)) {
      flags |= IREE_HAL_BARRIER_FLAG_RELEASE_SYSTEM_SCOPE;
    }
    if (barrier->effects.bits &
        (IREE_HAL_MEMORY_EFFECT_GLOBAL_ACQUIRE_FROM_SYSTEM |
         IREE_HAL_MEMORY_EFFECT_RANGE_ACQUIRE_FROM_SYSTEM)) {
      flags |= IREE_HAL_BARRIER_FLAG_ACQUIRE_SYSTEM_SCOPE;
    }
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
    const iree_hal_access_scope_t transfer =
        IREE_HAL_ACCESS_SCOPE_TRANSFER_READ |
        IREE_HAL_ACCESS_SCOPE_TRANSFER_WRITE;
    const iree_hal_access_scope_t shader =
        IREE_HAL_ACCESS_SCOPE_CONSTANT_READ |
        IREE_HAL_ACCESS_SCOPE_DISPATCH_READ |
        IREE_HAL_ACCESS_SCOPE_DISPATCH_WRITE | transfer;
    const iree_hal_access_scope_t command_processor =
        IREE_HAL_ACCESS_SCOPE_INDIRECT_COMMAND_READ |
        IREE_HAL_ACCESS_SCOPE_ATOMIC_READ | IREE_HAL_ACCESS_SCOPE_ATOMIC_WRITE |
        transfer;
    const iree_hal_access_scope_t external =
        IREE_HAL_ACCESS_SCOPE_HOST_READ | IREE_HAL_ACCESS_SCOPE_HOST_WRITE |
        IREE_HAL_ACCESS_SCOPE_MEMORY_READ | IREE_HAL_ACCESS_SCOPE_MEMORY_WRITE;
    if (source && target && ((source | target) & shader)) {
      result.cache_control |= profile->shader;
    }
    if ((source & external) ||
        (profile->command_processor_bypasses_l2 &&
         (source & command_processor) && (target & shader))) {
      flags |= IREE_HAL_BARRIER_FLAG_ACQUIRE_SYSTEM_SCOPE;
    }
    if ((target & external) ||
        (profile->command_processor_bypasses_l2 && (source & shader) &&
         (target & command_processor))) {
      // GFX12 shader reads can leave earlier dirty L2 lines behind. Crossing to
      // a bypassing CP reader requires writeback even for a read/read edge.
      flags |= IREE_HAL_BARRIER_FLAG_RELEASE_SYSTEM_SCOPE;
    }
    if (flags & IREE_HAL_BARRIER_FLAG_RELEASE_SYSTEM_SCOPE) {
      result.cache_control |= profile->release;
    }
    if (flags & IREE_HAL_BARRIER_FLAG_ACQUIRE_SYSTEM_SCOPE) {
      result.cache_control |= profile->acquire;
    }
    iree_hal_execution_stage_t stages = barrier->source_stage_mask;
    if (!stages && (source || flags)) {
      // A visibility-only record still covers its local producers. Explicit
      // source stages (including HOST) let the caller narrow that domain.
      if ((source & (transfer | IREE_HAL_ACCESS_SCOPE_MEMORY_READ |
                     IREE_HAL_ACCESS_SCOPE_MEMORY_WRITE)) ||
          !source) {
        stages = IREE_HAL_EXECUTION_STAGE_COMMAND_RETIRE;
      } else if (source & shader) {
        stages = IREE_HAL_EXECUTION_STAGE_DISPATCH;
      }
    }
    result.execution |= iree_hal_amd_gpu_pm4_barrier_execution(stages);
  }
  return result;
}

uint32_t* iree_hal_amd_gpu_pm4_barrier_emit(
    uint32_t* words, iree_hal_amd_gpu_pm4_barrier_t barrier) {
  if (barrier.execution & IREE_HAL_AMD_GPU_PM4_BARRIER_EXECUTION_COMPUTE) {
    words = iree_hal_amd_gpu_pm4_emit_compute_flush(words);
  }
  if (barrier.execution & IREE_HAL_AMD_GPU_PM4_BARRIER_EXECUTION_DMA) {
    words = iree_hal_amd_gpu_pm4_emit_dma_flush(words);
  }
  if (barrier.cache_control) {
    words = iree_hal_amd_gpu_pm4_emit_acquire(words, barrier.cache_control);
  }
  return words;
}

uint32_t* iree_hal_amd_gpu_pm4_barrier_emit_tail(
    uint32_t* words, iree_hal_amd_gpu_pm4_barrier_t barrier,
    uint32_t* out_cache_control) {
  // RELEASE_MEM cannot represent instruction invalidation. Only an ordinary
  // emission may combine mandatory instruction publication with payload work.
  IREE_ASSERT(!(barrier.cache_control & 3u));
  if (barrier.execution & IREE_HAL_AMD_GPU_PM4_BARRIER_EXECUTION_DMA) {
    words = iree_hal_amd_gpu_pm4_emit_dma_flush(words);
  }
  const uint32_t acquire = barrier.cache_control;
  // GCR's middle fields retain their relative order in the release layout.
  // Low metadata/scope bits, lower writeback, and scalar invalidation move
  // separately. These mappings also preserve GFX12.5's repurposed fields.
  *out_cache_control = ((acquire >> 4) & 3u) | ((acquire >> 6) & 0xffcu) |
                       ((acquire & (1u << 6)) << 6) |
                       ((acquire & (1u << 7)) << 11);
  return words;
}
