// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVER_AMD_GPU_QUEUE_SDMA_BARRIER_H_
#define IREE_HAL_DRIVER_AMD_GPU_QUEUE_SDMA_BARRIER_H_

#include "amdf/gpu.h"
#include "iree/hal/command_buffer.h"

#ifdef __cplusplus
extern "C" {
#endif

enum iree_hal_amd_gpu_sdma_barrier_bits_e {
  IREE_HAL_AMD_GPU_SDMA_BARRIER_JOIN = 1u << 0,
  IREE_HAL_AMD_GPU_SDMA_BARRIER_ACQUIRE = 1u << 1,
  IREE_HAL_AMD_GPU_SDMA_BARRIER_RELEASE = 1u << 2,
};
typedef uint32_t iree_hal_amd_gpu_sdma_barrier_t;

// Captures public ordering intent, independently of its carrier. NULL selects
// conservative system release/acquire; an empty list omits optional work.
// Public validation has already admitted all effects and queue executors.
iree_hal_amd_gpu_sdma_barrier_t iree_hal_amd_gpu_sdma_barrier_prepare(
    const iree_hal_barrier_list_t* barriers);

// Adds already-qualified file/memory transition effects. The native family
// owns visibility; these bits never substitute for incoming dependencies.
iree_hal_amd_gpu_sdma_barrier_t iree_hal_amd_gpu_sdma_barrier_effects(
    iree_hal_memory_effects_t effects);

// Cold selection of the native cache carrier. Stream control is zero on
// coherent/bypassing families without GCR. KERNEL SUBMISSION_GCR instead needs
// checked retirement and completed incoming dependencies.
typedef struct iree_hal_amd_gpu_sdma_barrier_profile_t {
  // Native whole-cache acquire encoding, or zero without USER_GCR.
  uint32_t acquire;
  // Native whole-cache release encoding, or zero without USER_GCR.
  uint32_t release;
  // Cache visibility belongs to checked KERNEL submission retirement.
  bool submission;
} iree_hal_amd_gpu_sdma_barrier_profile_t;

iree_hal_amd_gpu_sdma_barrier_profile_t iree_hal_amd_gpu_sdma_barrier_profile(
    amdf_queue_format_features_t features,
    amdf_queue_publication_modes_t publication_mode);

// Returns the exact number of DWORDs for the optional join NOP and GCR packet.
// Stores the GCR control bits in |out_cache_control|, or zero to omit it.
static inline uint32_t iree_hal_amd_gpu_sdma_barrier_word_count(
    iree_hal_amd_gpu_sdma_barrier_profile_t profile,
    iree_hal_amd_gpu_sdma_barrier_t barrier, uint32_t* out_cache_control) {
  *out_cache_control =
      ((barrier & IREE_HAL_AMD_GPU_SDMA_BARRIER_ACQUIRE) ? profile.acquire
                                                         : 0) |
      ((barrier & IREE_HAL_AMD_GPU_SDMA_BARRIER_RELEASE) ? profile.release : 0);
  return ((barrier & IREE_HAL_AMD_GPU_SDMA_BARRIER_JOIN) ? 1u : 0u) +
         (*out_cache_control ? 5u : 0u);
}

#ifdef __cplusplus
}  // extern "C"
#endif
#endif  // IREE_HAL_DRIVER_AMD_GPU_QUEUE_SDMA_BARRIER_H_
