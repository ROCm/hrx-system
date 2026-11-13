// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/driver/amd/gpu/queue/sdma/commands.h"

uint32_t* iree_hal_amd_gpu_sdma_emit_copy(uint32_t* words,
                                          amdf_queue_format_features_t features,
                                          uint64_t source_address,
                                          uint64_t target_address,
                                          uint32_t byte_length) {
  *words++ = 1;
  *words++ = byte_length - 1;
  *words++ = features & AMDF_GPU_SDMA_FORMAT_FEATURE_MEMORY_SCOPE
                 ? (3u << 18) | (3u << 26)
                 : 0;
  *words++ = (uint32_t)source_address;
  *words++ = (uint32_t)(source_address >> 32);
  *words++ = (uint32_t)target_address;
  *words++ = (uint32_t)(target_address >> 32);
  return words;
}

uint32_t* iree_hal_amd_gpu_sdma_emit_fence(
    uint32_t* words, amdf_queue_format_features_t features, uint64_t address,
    uint32_t value) {
  uint32_t header = 5;
  if (features & (AMDF_GPU_SDMA_FORMAT_FEATURE_FENCE_MEMORY_TYPE |
                  AMDF_GPU_SDMA_FORMAT_FEATURE_FENCE_SYSTEM)) {
    header |= 3u << 16;
  }
  if (features & AMDF_GPU_SDMA_FORMAT_FEATURE_FENCE_SYSTEM) {
    header |= 1u << 20;
  }
  if (features & AMDF_GPU_SDMA_FORMAT_FEATURE_MEMORY_SCOPE) {
    header |= 3u << 24;
  }
  *words++ = header;
  *words++ = (uint32_t)address;
  *words++ = (uint32_t)(address >> 32);
  *words++ = value;
  return words;
}

uint32_t* iree_hal_amd_gpu_sdma_emit_cache(uint32_t* words, uint32_t control) {
  *words++ = 17u | (1u << 8);
  *words++ = 0;
  *words++ = control << 16;
  *words++ = 0;
  *words++ = 0;
  return words;
}

uint32_t* iree_hal_amd_gpu_sdma_emit_wait(uint32_t* words,
                                          amdf_queue_format_features_t features,
                                          uint64_t address, uint32_t value) {
  *words++ = 8u | (3u << 28) | (1u << 31);
  *words++ = (uint32_t)address;
  *words++ = (uint32_t)(address >> 32);
  *words++ = value;
  *words++ = UINT32_MAX;
  *words++ =
      (0xfffu << 16) | 4u |
      ((features & AMDF_GPU_SDMA_FORMAT_FEATURE_MEMORY_SCOPE) ? 3u << 28 : 0);
  return words;
}

uint32_t* iree_hal_amd_gpu_sdma_emit_fill(uint32_t* words,
                                          amdf_queue_format_features_t features,
                                          uint64_t address,
                                          uint32_t byte_length,
                                          uint32_t pattern) {
  *words++ =
      11u | (2u << 30) |
      ((features & AMDF_GPU_SDMA_FORMAT_FEATURE_MEMORY_SCOPE) ? 3u << 24 : 0);
  *words++ = (uint32_t)address;
  *words++ = (uint32_t)(address >> 32);
  *words++ = pattern;
  *words++ = byte_length - 1;
  return words;
}

uint32_t* iree_hal_amd_gpu_sdma_emit_timestamp(
    uint32_t* words, amdf_queue_format_features_t features, uint64_t address) {
  *words++ =
      13u | (2u << 8) |
      ((features & AMDF_GPU_SDMA_FORMAT_FEATURE_MEMORY_SCOPE) ? 3u << 24 : 0);
  *words++ = (uint32_t)address;
  *words++ = (uint32_t)(address >> 32);
  return words;
}
