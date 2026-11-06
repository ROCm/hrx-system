// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/driver/amd/gpu/queue/sdma/barrier.h"

#include "iree/testing/gtest.h"

namespace {

TEST(BarrierTest, NativeCarrierOwnsVisibility) {
  for (const auto mode :
       {AMDF_QUEUE_PUBLICATION_MODE_USER, AMDF_QUEUE_PUBLICATION_MODE_KERNEL}) {
    for (uint32_t combination = 0; combination < 4; ++combination) {
      SCOPED_TRACE(mode);
      SCOPED_TRACE(combination);
      const auto features =
          ((combination & 1) ? AMDF_GPU_SDMA_FORMAT_FEATURE_USER_GCR : 0) |
          ((combination & 2) ? AMDF_GPU_SDMA_FORMAT_FEATURE_SUBMISSION_GCR : 0);
      const auto profile =
          iree_hal_amd_gpu_sdma_barrier_profile(features, mode);
      const bool stream =
          mode == AMDF_QUEUE_PUBLICATION_MODE_USER && (combination & 1);
      EXPECT_EQ(profile.acquire, stream ? 0xc3c0u : 0u);
      EXPECT_EQ(profile.release, stream ? 0x8040u : 0u);
      EXPECT_EQ(
          profile.submission,
          mode == AMDF_QUEUE_PUBLICATION_MODE_KERNEL && (combination & 2));
    }
  }
}

TEST(BarrierTest, OptionalAndRequiredBoundariesRemainDistinct) {
  EXPECT_EQ(iree_hal_amd_gpu_sdma_barrier_prepare(nullptr), 7u);
  const iree_hal_barrier_list_t empty = {};
  EXPECT_EQ(iree_hal_amd_gpu_sdma_barrier_prepare(&empty), 0u);
  for (const uint32_t effects :
       {IREE_HAL_MEMORY_EFFECT_GLOBAL_RELEASE_TO_SYSTEM,
        IREE_HAL_MEMORY_EFFECT_RANGE_RELEASE_TO_SYSTEM}) {
    EXPECT_EQ(iree_hal_amd_gpu_sdma_barrier_effects({effects}), 5u);
  }
  for (const uint32_t effects :
       {IREE_HAL_MEMORY_EFFECT_GLOBAL_ACQUIRE_FROM_SYSTEM,
        IREE_HAL_MEMORY_EFFECT_RANGE_ACQUIRE_FROM_SYSTEM}) {
    EXPECT_EQ(iree_hal_amd_gpu_sdma_barrier_effects({effects}), 2u);
  }
  const iree_hal_barrier_t join = {
      .source_stage_mask = IREE_HAL_EXECUTION_STAGE_TRANSFER,
      .target_stage_mask = IREE_HAL_EXECUTION_STAGE_TRANSFER,
  };
  const iree_hal_barrier_list_t list = {1, &join};
  EXPECT_EQ(iree_hal_amd_gpu_sdma_barrier_prepare(&list), 1u);
}

TEST(BarrierTest, HostAndGenericMemoryScopesCaptureDirection) {
  const iree_hal_access_scope_t external[] = {
      IREE_HAL_ACCESS_SCOPE_HOST_READ, IREE_HAL_ACCESS_SCOPE_HOST_WRITE,
      IREE_HAL_ACCESS_SCOPE_MEMORY_READ, IREE_HAL_ACCESS_SCOPE_MEMORY_WRITE};
  for (const auto scope : external) {
    const iree_hal_memory_barrier_t memory[] = {
        {scope, IREE_HAL_ACCESS_SCOPE_TRANSFER_READ},
        {IREE_HAL_ACCESS_SCOPE_TRANSFER_WRITE, scope}};
    for (uint32_t i = 0; i < 2; ++i) {
      const iree_hal_barrier_t barrier = {
          .memory_barrier_count = 1,
          .memory_barriers = &memory[i],
      };
      const iree_hal_barrier_list_t list = {1, &barrier};
      EXPECT_EQ(iree_hal_amd_gpu_sdma_barrier_prepare(&list), i ? 5u : 3u);
    }
  }
}

}  // namespace
